#include <tuple>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include "inplace_abn.h"
#include "utils.h"
#include "checks.h"

/***********************************************************************************************************************
 * Utility functions
 **********************************************************************************************************************/

int32_t count_samples(const at::Tensor& x) {
  return x.size(0) * x.size(2);
}

/***********************************************************************************************************************
 * Templated implementations
 **********************************************************************************************************************/

template<typename scalar_t, Activation activation>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_impl(
    const at::Tensor& y_act_, const at::Tensor& dy_act_, const c10::optional<at::Tensor>& weight_,
    const c10::optional<at::Tensor>& bias_, float eps, float activation_param) {
  // Initialize output tensors
  auto xhat_ = at::empty_like(y_act_);
  auto dy_ = at::empty_like(y_act_);
  auto sum_dy_ = at::zeros({y_act_.size(1)}, y_act_.options());
  auto sum_xhat_dy_ = at::zeros({y_act_.size(1)}, y_act_.options());

  // Normalize shapes
  auto y_act_norm_ = normalize_shape(y_act_);
  auto dy_act_norm_ = normalize_shape(dy_act_);
  auto xhat_norm_ = normalize_shape(xhat_);
  auto dy_norm_ = normalize_shape(dy_);

  // Get dimensions
  int64_t num = y_act_norm_.size(0), chn = y_act_norm_.size(1), sp = y_act_norm_.size(2);

  // Make accessors
  auto y_act = y_act_norm_.accessor<scalar_t, 3>();
  auto dy_act = dy_act_norm_.accessor<scalar_t, 3>();
  auto xhat = xhat_norm_.accessor<scalar_t, 3>();
  auto dy = dy_norm_.accessor<scalar_t, 3>();
  auto weight = accessor_or_dummy<scalar_t, 1>(weight_);
  auto bias = accessor_or_dummy<scalar_t, 1>(bias_);
  auto sum_dy = sum_dy_.accessor<scalar_t, 1>();
  auto sum_xhat_dy = sum_xhat_dy_.accessor<scalar_t, 1>();

  // Main loop
  for (int64_t c = 0; c < chn; ++c) {
    auto inv_gamma_c = weight_.has_value() ? scalar_t(1) / (std::abs(weight[c]) + eps) : scalar_t(1);
    auto beta_c = bias_.has_value() ? bias[c] : scalar_t(0);

    for (int64_t n = 0; n < num; ++n) {
      auto y_act_nc = y_act[n][c];
      auto dy_act_nc = dy_act[n][c];
      auto xhat_nc = xhat[n][c];
      auto dy_nc = dy[n][c];

      for (int64_t s = 0; s < sp; ++s) {
        // Invert activation
        ActivationFn<scalar_t, activation>::backward(y_act_nc[s], dy_act_nc[s], activation_param, xhat_nc[s], dy_nc[s]);

        // Invert affine transformation
        xhat_nc[s] = (xhat_nc[s] - beta_c) * inv_gamma_c;

        // Accumulate
        sum_dy[c] += dy_nc[s];
        sum_xhat_dy[c] += xhat_nc[s] * dy_nc[s];
      }
    }
  }

  return std::make_tuple(xhat_, dy_, sum_dy_, sum_xhat_dy_);
}

/***********************************************************************************************************************
 * Interface methods
 **********************************************************************************************************************/

std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics_cpu(const at::Tensor& x_) {
  CHECK_NOT_HALF(x_);

  auto x = normalize_shape(x_);

  auto mean = x.mean(c10::IntArrayRef({0, 2}));
  auto var = (x - normalize_shape(mean)).pow(2).mean(c10::IntArrayRef({0, 2}));
  auto count = at::full({1}, count_samples(x), x.options().dtype(at::ScalarType::Long));

  return std::make_tuple(mean, var, count);
}

void forward_cpu(at::Tensor& x_, const at::Tensor& mean, const at::Tensor& var,
                 const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
                 float eps, Activation activation, float activation_param) {
  CHECK_NOT_HALF(x_);

  auto x = normalize_shape(x_);

  // Apply normalization
  auto abs_weight = weight.has_value() ? weight.value().abs() + eps : at::ones({mean.size(0)}, mean.options());
  auto inv_std = 1 / at::sqrt(var + eps);

  auto scale = weight.has_value() ? abs_weight * inv_std : inv_std;
  auto shift = weight.has_value() ? bias.value() - mean * abs_weight * inv_std : -mean * inv_std;

  x.mul_(normalize_shape(scale)).add_(normalize_shape(shift));

  switch (activation) {
  case Activation::LeakyReLU:
    at::leaky_relu_(x, activation_param);
    break;
  case Activation::ELU:
    at::elu_(x, activation_param);
    break;
  case Activation::Identity:
    break;
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_cpu(
    const at::Tensor& y_act, const at::Tensor& dy_act, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param) {
  CHECK_NOT_HALF(y_act);

  // Run templated implementation
  return AT_DISPATCH_FLOATING_TYPES(y_act.scalar_type(), "backward_reduce_cpu", [&] {
    switch (activation) {
    case Activation::LeakyReLU:
      return backward_reduce_impl<scalar_t, Activation::LeakyReLU>(y_act, dy_act, weight, bias, eps, activation_param);
    case Activation::ELU:
      return backward_reduce_impl<scalar_t, Activation::ELU>(y_act, dy_act, weight, bias, eps, activation_param);
    case Activation::Identity:
    default:
      return backward_reduce_impl<scalar_t, Activation::Identity>(y_act, dy_act, weight, bias, eps, activation_param);
    }
  });
}

void backward_cpu(const at::Tensor& xhat_, at::Tensor& dy_, const at::Tensor& var, const at::Tensor& count,
                  const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy,
                  const c10::optional<at::Tensor>& weight, float eps) {
  CHECK_NOT_HALF(xhat_);

  auto xhat = normalize_shape(xhat_);
  auto dy = normalize_shape(dy_);
  auto mean_dy = normalize_shape(sum_dy / count.to(sum_dy.options()));
  auto mean_xhat_dy = normalize_shape(sum_xhat_dy / count.to(sum_xhat_dy.options()));

  auto mult = weight.has_value() ? (weight.value().abs() + eps) / (var + eps).sqrt() : 1 / (var + eps).sqrt();

  // dy = (dy - mean_dy - xhat * mean_xhat_dy) * mult
  dy.sub_(mean_dy).sub_(xhat * mean_xhat_dy).mul_(normalize_shape(mult));
}
