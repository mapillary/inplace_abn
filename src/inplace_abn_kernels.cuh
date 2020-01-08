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

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

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
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
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
  int tid = threadIdx.x, bid = blockIdx.x;

  for (int c = bid * blockDim.x + tid; c < chn; c += gridDim.x * blockDim.x) {
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
