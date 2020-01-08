#include <tuple>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

#include "inplace_abn.h"
#include "utils.h"
#include "cuda_utils.cuh"
#include "inplace_abn_kernels.cuh"
#include "dispatch.h"

/***********************************************************************************************************************
 * Templated implementations
 **********************************************************************************************************************/

template<typename scalar_t, typename index_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics_template(const at::Tensor& x_) {
  // Normalize shape and get dimensions
  auto x = normalize_shape(x_);
  auto num = x.size(0), chn = x.size(1), sp = x.size(2);

  // Type handling
  using accscalar_t = at::acc_type<scalar_t, true>;
  auto acc_options = x.options();
  if (x.scalar_type() == at::ScalarType::Half) {
    acc_options = acc_options.dtype(at::ScalarType::Float);
  }

  // Initialize output tensors
  auto mean = at::empty({chn}, acc_options);
  auto var = at::empty({chn}, acc_options);
  auto count = at::full({1}, num * sp, x.options().dtype(at::ScalarType::Long));

  // Make accessors
  auto x_accessor = x.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto mean_accessor = mean.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto var_accessor = var.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();

  // Kernel parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(chn);
  int tf = getNumThreads(sp);
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE / tf));

  // Invoke kernel
  statistics_kernel<scalar_t, accscalar_t, index_t><<<blocks, threads, 0, stream>>>(
      x_accessor, mean_accessor, var_accessor);

  return std::make_tuple(mean, var, count);
}

template<typename scalar_t, typename index_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> reduce_statistics_template(
    const at::Tensor& all_mean, const at::Tensor& all_var, const at::Tensor& all_count) {
  auto num = all_mean.size(0), chn = all_mean.size(1);

  // Initialize output tensors
  auto mean = at::empty({chn}, all_mean.options());
  auto var = at::empty({chn}, all_var.options());
  auto count = all_count.sum({0});

  // Make accessors
  auto all_mean_accessor = all_mean.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, index_t>();
  auto all_var_accessor = all_var.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, index_t>();
  auto all_count_accessor = all_count.packed_accessor<int64_t, 2, at::RestrictPtrTraits, index_t>();
  auto mean_accessor = mean.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto var_accessor = var.packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>();

  // Kernel parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = getNumThreads(chn);
  int blocks = std::max<int>(1, chn / threads);

  // Invoke kernel
  reduce_statistics_kernel<scalar_t, index_t><<<blocks, threads, 0, stream>>>(
      all_mean_accessor, all_var_accessor, all_count_accessor, mean_accessor, var_accessor);

  return std::make_tuple(mean, var, count);
}

template<typename scalar_t, typename prmscalar_t, typename index_t>
void forward_template(at::Tensor& x_, const at::Tensor& mean, const at::Tensor& var,
                      const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
                      float eps, Activation activation, float activation_param) {
  // Normalize shape and get dimensions
  auto x = normalize_shape(x_);
  auto num = x.size(0), chn = x.size(1), sp = x.size(2);

  // Type handling
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Make accessors
  auto x_accessor = x.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto mean_accessor = mean.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto var_accessor = var.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto weight_accessor = packed_accessor_or_dummy<prmscalar_t, 1, at::RestrictPtrTraits, index_t>(weight);
  auto bias_accessor = packed_accessor_or_dummy<prmscalar_t, 1, at::RestrictPtrTraits, index_t>(bias);

  // Kernel parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  int tf = std::max<int>(getNumThreads(sp / 4), std::min<int>(getNumThreads(sp), 64));
  int tb = std::max<int>(64 / tf, 1);
  dim3 blocks(chn, std::max<int>(1, std::min<int>((256 * 1024) / chn, (chn + tb - 1) / tb)));
  blocks.y = std::min<int>(blocks.y, 65535);
  dim3 threads(tf, tb);

  // Invoke kernel
  switch (activation) {
  case Activation::LeakyReLU:
    forward_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::LeakyReLU><<<blocks, threads, 0, stream>>>(
        x_accessor, mean_accessor, var_accessor, weight_accessor, bias_accessor, eps, activation_param);
    break;
  case Activation::ELU:
    forward_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::ELU><<<blocks, threads, 0, stream>>>(
        x_accessor, mean_accessor, var_accessor, weight_accessor, bias_accessor, eps, activation_param);
    break;
  case Activation::Identity:
    forward_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::Identity><<<blocks, threads, 0, stream>>>(
        x_accessor, mean_accessor, var_accessor, weight_accessor, bias_accessor, eps, activation_param);
    break;
  }
}

template<typename scalar_t, typename prmscalar_t, typename index_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_template(
    const at::Tensor& y_act_, const at::Tensor& dy_act_, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param) {
  // Normalize shape and get dimensions
  auto y_act = normalize_shape(y_act_);
  auto dy_act = normalize_shape(dy_act_);
  auto num = y_act.size(0), chn = y_act.size(1), sp = y_act.size(2);

  // Type handling
  using accscalar_t = at::acc_type<scalar_t, true>;
  auto acc_options = y_act.options();
  if (y_act.scalar_type() == at::ScalarType::Half) {
    acc_options = acc_options.dtype(at::ScalarType::Float);
  }

  // Initialize output tensors
  auto xhat = at::empty_like(y_act);
  auto dy = at::empty_like(y_act);
  auto sum_dy = at::empty({chn}, acc_options);
  auto sum_xhat_dy = at::empty({chn}, acc_options);

  // Make accessors
  auto y_act_accessor = y_act.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto dy_act_accessor = dy_act.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto xhat_accessor = xhat.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto dy_accessor = dy.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto weight_accessor = packed_accessor_or_dummy<prmscalar_t, 1, at::RestrictPtrTraits, index_t>(weight);
  auto bias_accessor = packed_accessor_or_dummy<prmscalar_t, 1, at::RestrictPtrTraits, index_t>(bias);
  auto sum_dy_accessor = sum_dy.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto sum_xhat_dy_accessor = sum_xhat_dy.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();

  // Kernel parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  int block_y = std::min<int>(lastPow2(num), MAX_BLOCK_SIZE / 32);
  int block_x = std::min<int>(getNumThreads(sp), MAX_BLOCK_SIZE / block_y);
  const dim3 threads(block_x, block_y);
  const dim3 blocks(chn);

  // Invoke kernel
  switch (activation) {
  case Activation::LeakyReLU:
    backward_reduce_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::LeakyReLU><<<blocks, threads, 0, stream>>>(
        y_act_accessor, dy_act_accessor, weight_accessor, bias_accessor, xhat_accessor, dy_accessor, sum_dy_accessor, sum_xhat_dy_accessor,
        eps, activation_param);
    break;
  case Activation::ELU:
    backward_reduce_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::ELU><<<blocks, threads, 0, stream>>>(
        y_act_accessor, dy_act_accessor, weight_accessor, bias_accessor, xhat_accessor, dy_accessor, sum_dy_accessor, sum_xhat_dy_accessor,
        eps, activation_param);
    break;
  case Activation::Identity:
    backward_reduce_kernel<scalar_t, accscalar_t, prmscalar_t, index_t, Activation::Identity><<<blocks, threads, 0, stream>>>(
        y_act_accessor, dy_act_accessor, weight_accessor, bias_accessor, xhat_accessor, dy_accessor, sum_dy_accessor, sum_xhat_dy_accessor,
        eps, activation_param);
    break;
  }

  return std::make_tuple(xhat.view(y_act_.sizes()), dy.view(y_act_.sizes()), sum_dy, sum_xhat_dy);
}

template<typename scalar_t, typename prmscalar_t, typename index_t>
void backward_template(const at::Tensor& xhat_, at::Tensor& dy_, const at::Tensor& var,
                       const at::Tensor& count, const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy,
                       const c10::optional<at::Tensor>& weight, float eps) {
  // Normalize shape and get dimensions
  auto xhat = normalize_shape(xhat_);
  auto dy = normalize_shape(dy_);
  auto num = xhat.size(0), chn = xhat.size(1), sp = xhat.size(2);

  // Type handling
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Make accessors
  auto xhat_accessor = xhat.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto dy_accessor = dy.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
  auto var_accessor = var.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto count_accessor = count.packed_accessor<int64_t, 1, at::RestrictPtrTraits, index_t>();
  auto sum_dy_accessor = sum_dy.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto sum_xhat_dy_accessor = sum_xhat_dy.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, index_t>();
  auto weight_accessor = packed_accessor_or_dummy<prmscalar_t, 1, at::RestrictPtrTraits, index_t>(weight);

  // Kernel parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  int tf = std::max<int>(getNumThreads(sp / 4), std::min<int>(getNumThreads(sp), 64));
  int tb = std::max<int>(64 / tf, 1);
  dim3 blocks(chn, std::max<int>(1, std::min<int>((256 * 1024) / chn, (chn + tb - 1) / tb)));
  blocks.y = std::min<int>(blocks.y, 65535);
  dim3 threads(tf, tb);

  // Invoke kernel
  backward_kernel<scalar_t, accscalar_t, prmscalar_t, index_t><<<blocks, threads, 0, stream>>>(
      xhat_accessor, dy_accessor, var_accessor, count_accessor, sum_dy_accessor, sum_xhat_dy_accessor,
      weight_accessor, eps);
}

/***********************************************************************************************************************
 * Interface methods
 **********************************************************************************************************************/

std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics_cuda(const at::Tensor& x) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "statistics_cuda", [&] {
    if (at::cuda::detail::canUse32BitIndexMath(x)) {
      return statistics_template<scalar_t, int32_t>(x);
    } else {
      return statistics_template<scalar_t, int64_t>(x);
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> reduce_statistics_cuda(
    const at::Tensor& all_mean, const at::Tensor& all_var, const at::Tensor& all_count) {
  return AT_DISPATCH_FLOATING_TYPES(all_mean.scalar_type(), "reduce_statistics_cuda", [&] {
    if (at::cuda::detail::canUse32BitIndexMath(all_mean)) {
      return reduce_statistics_template<scalar_t, int32_t>(all_mean, all_var, all_count);
    } else {
      return reduce_statistics_template<scalar_t, int64_t>(all_mean, all_var, all_count);
    }
  });
}

void forward_cuda(at::Tensor& x, const at::Tensor& mean, const at::Tensor& var,
                  const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
                  float eps, Activation activation, float activation_param) {
  const auto& w_scalar_type = weight.has_value() ? weight.value().scalar_type() : x.scalar_type();

  DOUBLE_DISPATCH(x.scalar_type(), w_scalar_type, "forward_cuda", [&] {
    if (at::cuda::detail::canUse32BitIndexMath(x)) {
      forward_template<scalar_t, prmscalar_t, int32_t>(x, mean, var, weight, bias, eps, activation, activation_param);
    } else {
      forward_template<scalar_t, prmscalar_t, int64_t>(x, mean, var, weight, bias, eps, activation, activation_param);
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_cuda(
    const at::Tensor& y_act, const at::Tensor& dy_act, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param) {
  const auto& w_scalar_type = weight.has_value() ? weight.value().scalar_type() : y_act.scalar_type();

  return DOUBLE_DISPATCH(y_act.scalar_type(), w_scalar_type, "backward_reduce_cuda", [&] {
    if (at::cuda::detail::canUse32BitIndexMath(y_act)) {
      return backward_reduce_template<scalar_t, prmscalar_t, int32_t>(
          y_act, dy_act, weight, bias, eps, activation, activation_param);
    } else {
      return backward_reduce_template<scalar_t, prmscalar_t, int64_t>(
          y_act, dy_act, weight, bias, eps, activation, activation_param);
    }
  });
}

void backward_cuda(const at::Tensor& xhat, at::Tensor& dy, const at::Tensor& var, const at::Tensor& count,
                   const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy,
                   const c10::optional<at::Tensor>& weight, float eps) {
  const auto& w_scalar_type = weight.has_value() ? weight.value().scalar_type() : xhat.scalar_type();

  return DOUBLE_DISPATCH(xhat.scalar_type(), w_scalar_type, "backward_cuda", [&] {
    if (at::cuda::detail::canUse32BitIndexMath(xhat)) {
      backward_template<scalar_t, prmscalar_t, int32_t>(xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, eps);
    } else {
      backward_template<scalar_t, prmscalar_t, int64_t>(xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, eps);
    }
  });
}
