// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once

#include <ATen/ATen.h>

#include "inplace_abn.h"
#include "cuda_utils.cuh"

/***********************************************************************************************************************
 * Kernels
 * -------
 *
 * These are copy-pasted (+ some minor modifications) from the pytorch 1.1 native implementation of BN
 **********************************************************************************************************************/

template<typename scalar_t, typename accscalar_t, typename index_t>
__global__ void statistics_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> mean,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> var) {

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  auto plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  auto tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  accscalar_t* shared_avg_var = (accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  accscalar_t avg = 0;
  accscalar_t var_n = 0;
  int n = 0;
  for (auto batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (auto x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      accscalar_t v = input[batch][plane][x];
      accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();

  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.
  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : accscalar_t(0));
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save mean and variance
  if (tid == 0) {
    mean[plane] = avg;
    var[plane] = var_n / N;
  }
}

template<typename scalar_t, typename index_t>
__global__ void reduce_statistics_kernel(
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, index_t> all_mean,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, index_t> all_var,
    const at::PackedTensorAccessor<int64_t, 2, at::RestrictPtrTraits, index_t> all_count,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> mean,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> var) {
  int num = all_mean.size(0), chn = all_mean.size(1);
  auto tid = threadIdx.x, bid = blockIdx.x;

  for (auto c = bid * blockDim.x + tid; c < chn; c += gridDim.x * blockDim.x) {
    scalar_t mean_c = 0;
    scalar_t var_c = 0;
    int64_t count_c = 0;

    for (int n = 0; n < num; ++n) {
      auto count_n = all_count[n][0];
      auto mean_n = all_mean[n][c];
      auto var_n = all_var[n][c] * count_n;

      auto delta = mean_n - mean_c;
      auto new_count = count_c + count_n;

      mean_c = (count_c * mean_c + count_n * mean_n) / new_count;
      var_c += var_n + delta * delta * count_c * count_n / new_count;
      count_c = new_count;
    }

    mean[c] = mean_c;
    var[c] = var_c / count_c;
  }
}

template<typename scalar_t, typename accscalar_t, typename prmscalar_t, typename index_t, Activation activation>
__global__ void forward_kernel(
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> mean_,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> var,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> weight_,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> bias_,
    float eps_, float activation_param) {
  index_t c = blockIdx.x;
  if (c >= x.size(1)) return;

  // Cache channel-wise values
  accscalar_t eps = static_cast<accscalar_t>(eps_);
  accscalar_t mean = mean_[c];
  accscalar_t inv_std = accscalar_t(1) / ::sqrt(var[c] + eps);
  accscalar_t weight = weight_.size(0) > 0 ? ::abs(static_cast<accscalar_t>(weight_[c])) + eps : accscalar_t(1);
  accscalar_t bias = bias_.size(0) > 0 ? static_cast<accscalar_t>(bias_[c]) : accscalar_t(0);

  index_t num = x.size(0);
  index_t sp = x.size(2);

  index_t step = blockDim.y * gridDim.y;
  for (index_t n = threadIdx.y + blockIdx.y * blockDim.y; n < num; n += step) {
    auto x_nc = x[n][c];

    for (index_t s = threadIdx.x; s < sp; s += blockDim.x) {
      x_nc[s] = static_cast<scalar_t>(weight * (static_cast<accscalar_t>(x_nc[s]) - mean) * inv_std + bias);
      ActivationFn<scalar_t, activation>::forward(x_nc[s], activation_param);
    }
  }
}

// Functor used in the backward_reduce kernel
template <typename scalar_t, typename accscalar_t, typename PTA, Activation activation>
struct GradOp {
  __device__ GradOp(const PTA& y_act, const PTA& dy_act, PTA& xhat, PTA& dy,
                    accscalar_t inv_gamma, accscalar_t beta, float activation_param)
    : y_act(y_act), dy_act(dy_act), xhat(xhat), dy(dy), inv_gamma(inv_gamma), beta(beta), activation_param(activation_param) {}

  __device__ __forceinline__ Float2<accscalar_t> operator()(int b, int c, int s) {
    const scalar_t y_act_ = y_act[b][c][s];
    const scalar_t dy_act_ = dy_act[b][c][s];
    scalar_t& xhat_ = xhat[b][c][s];
    scalar_t& dy_ = dy[b][c][s];

    // Invert activation
    ActivationFn<scalar_t, activation>::backward(y_act_, dy_act_, activation_param, xhat_, dy_);

    // Invert affine transform
    xhat_ = (xhat_ - beta) * inv_gamma;

    // Accumulate
    accscalar_t xhat_accscalar = static_cast<accscalar_t>(xhat_);
    accscalar_t dy_accscalar = static_cast<accscalar_t>(dy_);
    return Float2<accscalar_t>(dy_accscalar, xhat_accscalar * dy_accscalar);
  }

  const PTA& y_act;
  const PTA& dy_act;
  PTA& xhat;
  PTA& dy;
  const accscalar_t inv_gamma;
  const accscalar_t beta;
  const float activation_param;
};

template<typename scalar_t, typename accscalar_t, typename prmscalar_t, typename index_t, Activation activation>
__global__ void backward_reduce_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> y_act,
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> dy_act,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> weight,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> bias,
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> xhat,
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> dy,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> sum_dy,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> sum_xhat_dy,
    float eps_, float activation_param) {
  typedef at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> pta_t;
  typedef GradOp<scalar_t, accscalar_t, pta_t, activation> gradop_t;
  index_t c = blockIdx.x;

  accscalar_t eps = static_cast<accscalar_t>(eps_);
  accscalar_t inv_gamma = weight.size(0) > 0
        ? accscalar_t(1) / (::abs(static_cast<accscalar_t>(weight[c])) + eps)
        : accscalar_t(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[c]) : accscalar_t(0);

  gradop_t gop(y_act, dy_act, xhat, dy, inv_gamma, beta, activation_param);
  Float2<accscalar_t> res = reduce<Float2<accscalar_t>, gradop_t, pta_t>(gop, y_act, c);

  if (threadIdx.x == 0) {
    sum_dy[c] = res.v1;
    sum_xhat_dy[c] = res.v2;
  }
}

template<typename scalar_t, typename accscalar_t, typename prmscalar_t, typename index_t>
__global__ void backward_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> xhat,
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> dy,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> var,
    const at::PackedTensorAccessor<int64_t, 1, at::RestrictPtrTraits, index_t> count,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> sum_dy,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> sum_xhat_dy,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> weight_,
    float eps_) {
  index_t c = blockIdx.x;
  if (c >= xhat.size(1)) return;

  // Cache channel-wise values
  accscalar_t eps = static_cast<accscalar_t>(eps_);
  accscalar_t mult = weight_.size(0) > 0
      ? (::abs(static_cast<accscalar_t>(weight_[c])) + eps) / ::sqrt(var[c] + eps)
      : accscalar_t(1) / ::sqrt(var[c] + eps);

  accscalar_t norm = accscalar_t(1) / static_cast<accscalar_t>(count[0]);
  accscalar_t mean_dy_c = sum_dy[c] * norm;
  accscalar_t mean_xhat_dy_c = sum_xhat_dy[c] * norm;

  index_t num = xhat.size(0);
  index_t sp = xhat.size(2);

  index_t step = blockDim.y * gridDim.y;
  for (index_t n = threadIdx.y + blockIdx.y * blockDim.y; n < num; n += step) {
    auto xhat_nc = xhat[n][c];
    auto dy_nc = dy[n][c];

    for (index_t s = threadIdx.x; s < sp; s += blockDim.x) {
      dy_nc[s] = static_cast<scalar_t>(mult * (
          static_cast<accscalar_t>(dy_nc[s]) - mean_dy_c - static_cast<accscalar_t>(xhat_nc[s]) * mean_xhat_dy_c));
    }
  }
}
// Vectorized forward kernel: uses float4 loads/stores when possible (fp32)
template<typename scalar_t, typename accscalar_t, typename prmscalar_t, typename index_t, Activation activation>
__global__ void forward_kernel_vec4(
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> mean_,
    const at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> var,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> weight_,
    const at::PackedTensorAccessor<prmscalar_t, 1, at::RestrictPtrTraits, index_t> bias_,
    float eps_, float activation_param) {

  index_t c = blockIdx.x;
  if (c >= x.size(1)) return;

  // requires scalar_t == float for vectorized path
  static_assert(sizeof(float) == sizeof(scalar_t), "vec4 forward assumes float");

  accscalar_t eps = static_cast<accscalar_t>(eps_);
  accscalar_t mean = mean_[c];
  accscalar_t inv_std = accscalar_t(1) / ::sqrt(var[c] + eps);
  accscalar_t weight = weight_.size(0) > 0 ? ::abs(static_cast<accscalar_t>(weight_[c])) + eps : accscalar_t(1);
  accscalar_t bias = bias_.size(0) > 0 ? static_cast<accscalar_t>(bias_[c]) : accscalar_t(0);

  index_t num = x.size(0);
  index_t sp = x.size(2);

  // Only the launch wrapper should call this kernel when sp % 4 == 0 and pointers are aligned.
  // Use grid-stride on batch dimension first and vectorize across spatial dimension by 4.
  index_t step = blockDim.y * gridDim.y;
  for (index_t n = threadIdx.y + blockIdx.y * blockDim.y; n < num; n += step) {
    // reinterpret row as float4*
    scalar_t* row_ptr = &x[n][c][0];
    float4* row_vec = reinterpret_cast<float4*>(row_ptr);
    index_t vec_len = sp / 4; // ensured divisible by 4

    for (index_t vi = threadIdx.x; vi < vec_len; vi += blockDim.x) {
      float4 v = row_vec[vi];
      // Unpack, convert to accscalar_t, apply BN and activation, then store back
      float vx0 = v.x, vx1 = v.y, vx2 = v.z, vx3 = v.w;

      accscalar_t t0 = weight * (static_cast<accscalar_t>(vx0) - mean) * inv_std + bias;
      accscalar_t t1 = weight * (static_cast<accscalar_t>(vx1) - mean) * inv_std + bias;
      accscalar_t t2 = weight * (static_cast<accscalar_t>(vx2) - mean) * inv_std + bias;
      accscalar_t t3 = weight * (static_cast<accscalar_t>(vx3) - mean) * inv_std + bias;

      // apply activation in-place (ActivationFn specialized for scalar_t)
      scalar_t out0 = static_cast<scalar_t>(t0);
      scalar_t out1 = static_cast<scalar_t>(t1);
      scalar_t out2 = static_cast<scalar_t>(t2);
      scalar_t out3 = static_cast<scalar_t>(t3);

      ActivationFn<scalar_t, activation>::forward(out0, activation_param);
      ActivationFn<scalar_t, activation>::forward(out1, activation_param);
      ActivationFn<scalar_t, activation>::forward(out2, activation_param);
      ActivationFn<scalar_t, activation>::forward(out3, activation_param);

      row_vec[vi] = make_float4(out0, out1, out2, out3);
    }
  }
}
  // first warpSum to get one value per thread to one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = __shfl_xor_sync(FULL_WARP_MASK, avg, 1 << i, WARP_SIZE);
    int o_n = __shfl_xor_sync(FULL_WARP_MASK, n, 1 << i, WARP_SIZE);
    accscalar_t o_var = __shfl_xor_sync(FULL_WARP_MASK, var_n, 1 << i, WARP_SIZE);

    accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n += o_var + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }
// Launch with blockDim.x = 256, gridDim.x = ceil(chn / blockDim.x)
// Each block reduces a contiguous range of 'chn' channels (e.g. a tile)
template<typename scalar_t, typename index_t>
__global__ void reduce_statistics_kernel_tile(
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, index_t> all_mean,
    const at::PackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, index_t> all_var,
    const at::PackedTensorAccessor<int64_t, 2, at::RestrictPtrTraits, index_t> all_count,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> mean,
    at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> var) {

  extern __shared__ double scratch[]; // per-block scratch: layout [local_mean, local_var, local_count] per thread

  int num = all_mean.size(0), chn = all_mean.size(1);
  int tid = threadIdx.x;
  int c = blockIdx.x * blockDim.x + tid;

  // Each thread computes aggregated value for channel c over all chunks (num),
  // but we tile channels across blocks. If chn >> blockDim.x, each block handles a chunk of channels.
  if (c < chn) {
    double mean_c = 0;
    double var_c = 0;
    int64_t count_c = 0;
    for (int n = 0; n < num; ++n) {
      auto count_n = (int64_t)all_count[n][0];
      auto mean_n = (double)all_mean[n][c];
      auto var_n_term = (double)all_var[n][c] * count_n;
      auto delta = mean_n - mean_c;
      auto new_count = count_c + count_n;
      if (new_count == 0) continue;
      var_c += var_n_term + delta * delta * count_c * count_n / (double)new_count;
      mean_c = (count_c * mean_c + count_n * mean_n) / (double)new_count;
      count_c = new_count;
    }
    mean[c] = (scalar_t)mean_c;
    var[c] = (scalar_t)(var_c / (double)count_c);
  }
}
// host-side function (pseudo)
void launch_forward_kernel(
    Tensor x, Tensor mean, Tensor var, Tensor weight, Tensor bias,
    float eps, float act_param, cudaStream_t stream) {

  int c = x.size(1);
  int sp = x.size(2);

  dim3 block(/* x threads */, /* y threads */);
  dim3 grid(c, /* y dim for batch tiling */);

  // prefer vec4 if conditions met (fp32 & sp % 4 == 0 & pointer alignment)
  bool can_vec4 = x.scalar_type() == at::kFloat && (sp % 4 == 0) &&
      reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0;

  if (can_vec4) {
    auto func = forward_kernel_vec4<float, float, float, int, Activation::RELU>; // adapt activation enum
    // grid/block choose...
    func<<<grid, block, 0, stream>>>(
      x.packed_accessor32<float, 3, RestrictPtrTraits>(),
      mean.packed_accessor32<float,1,RestrictPtrTraits>(),
      var.packed_accessor32<float,1,RestrictPtrTraits>(),
      weight.packed_accessor32<float,1,RestrictPtrTraits>(),
      bias.packed_accessor32<float,1,RestrictPtrTraits>(),
      eps, act_param);
  } else {
    // fallback to the scalar kernel (existing forward_kernel)
  }
}
// Helper for timing a single kernel launch
float time_kernel_once(std::function<void()> launch_fn) {
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  launch_fn();
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms;
}
