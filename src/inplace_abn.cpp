#include <tuple>

#include <torch/extension.h>
#include <c10/util/Optional.h>

#include "inplace_abn.h"
#include "checks.h"
#include "utils.h"

/***********************************************************************************************************************
 * Exposed methods
 **********************************************************************************************************************/

std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics(const at::Tensor& x) {
  AT_CHECK(x.ndimension() >= 2, "x should have at least 2 dimensions");

  if (x.is_cuda()) {
    return statistics_cuda(x);
  } else {
    return statistics_cpu(x);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> reduce_statistics(
    const at::Tensor& all_mean, const at::Tensor& all_var, const at::Tensor& all_count) {
  // Inputs shouldn't be half
  CHECK_NOT_HALF(all_mean);
  CHECK_NOT_HALF(all_var);

  // reduce_statistics is only used on GPU
  CHECK_CUDA(all_mean);
  CHECK_CUDA(all_var);
  CHECK_CUDA(all_count);

  // Check types and dimensions
  AT_CHECK(all_mean.scalar_type() == all_var.scalar_type(), "all_mean and all_var should have the same scalar type");
  AT_CHECK(all_count.scalar_type() == at::ScalarType::Long, "all_count should have type int64");
  AT_CHECK(all_mean.ndimension() == 2, "all_mean should have size N x C");
  AT_CHECK(all_var.ndimension() == 2, "all_var should have size N x C");
  AT_CHECK(all_count.ndimension() == 2 && all_count.size(1) == 1, "all_count should have size N x 1");
  AT_CHECK(all_mean.size(0) == all_var.size(0) && all_mean.size(0) == all_count.size(0),
      "Inputs should have the same size in dimension 0");
  AT_CHECK(all_mean.size(1) == all_var.size(1), "all_mean and all_var should have the same size in dimension 1");

  return reduce_statistics_cuda(all_mean, all_var, all_count);
}

void forward(at::Tensor& x, const at::Tensor& mean, const at::Tensor& var,
             const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
             float eps, Activation activation, float activation_param) {
  // Check dimensions
  AT_CHECK(x.ndimension() >= 2, "x should have at least 2 dimensions");
  AT_CHECK(mean.size(0) == x.size(1), "x and mean don't have compatible dimensions");
  AT_CHECK(var.size(0) == x.size(1), "x and var don't have compatible dimensions");
  AT_CHECK(!weight.has_value() || weight.value().size(0) == x.size(1), "x and weight don't have compatible dimensions");
  AT_CHECK(!bias.has_value() || bias.value().size(0) == x.size(1), "x and bias don't have compatible dimensions");

  // Check types
  AT_CHECK(!weight.has_value() || weight.value().scalar_type() == x.scalar_type(),
      "weight and x must have the same type");
  AT_CHECK(!bias.has_value() || bias.value().scalar_type() == x.scalar_type(),
      "bias and x must have the same type");
  if (x.scalar_type() == at::ScalarType::Half) {
    AT_CHECK(mean.scalar_type() == at::ScalarType::Float, "mean must be float when x is half");
    AT_CHECK(var.scalar_type() == at::ScalarType::Float, "var must be float when x is half");
  } else {
    AT_CHECK(x.scalar_type() == mean.scalar_type(), "x and mean must have the same type");
    AT_CHECK(x.scalar_type() == var.scalar_type(), "x and var must have the same type");
  }

  AT_CHECK((weight.has_value() && bias.has_value()) || (!weight.has_value() && !bias.has_value()),
      "weight and bias must be equally present or not present");

  if (x.is_cuda()) {
    forward_cuda(x, mean, var, weight, bias, eps, activation, activation_param);
  } else {
    forward_cpu(x, mean, var, weight, bias, eps, activation, activation_param);
  }
}

std::tuple<at::Tensor, at::Tensor> backward_reduce(
    at::Tensor& y_act, at::Tensor& dy_act, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param) {
  // Check dimensions
  AT_CHECK(y_act.ndimension() >= 2, "y_act should have at least 2 dimensions");
  AT_CHECK(dy_act.ndimension() >= 2, "dy_act should have at least 2 dimensions");
  for (int dim = 0; dim < y_act.ndimension(); ++dim)
    AT_CHECK(y_act.size(dim) == dy_act.size(dim), "y_act and dy_act should have the same size");
  AT_CHECK(!weight.has_value() || weight.value().size(0) == y_act.size(1),
      "y_act and weight don't have compatible dimensions");
  AT_CHECK(!bias.has_value() || bias.value().size(0) == y_act.size(1),
      "y_act and bias don't have compatible dimensions");

  // Check types
  AT_CHECK(y_act.scalar_type() == dy_act.scalar_type(), "y_act and dy_act should have the same type");
  AT_CHECK(!weight.has_value() || weight.value().scalar_type() == dy_act.scalar_type(),
      "weight and x must have the same type");
  AT_CHECK(!bias.has_value() || bias.value().scalar_type() == dy_act.scalar_type(),
      "bias and x must have the same type");

  AT_CHECK((weight.has_value() && bias.has_value()) || (!weight.has_value() && !bias.has_value()),
      "weight and bias must be equally present or not present");

  if (y_act.is_cuda()) {
    return backward_reduce_cuda(y_act, dy_act, weight, bias, eps, activation, activation_param);
  } else {
    return backward_reduce_cpu(y_act, dy_act, weight, bias, eps, activation, activation_param);
  }
}

at::Tensor backward(const at::Tensor& xhat, const at::Tensor& dy, const at::Tensor& var, const at::Tensor& count,
                    const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy, const c10::optional<at::Tensor>& weight,
                    float eps) {
  // Check dimensions
  AT_CHECK(xhat.ndimension() >= 2, "xhat should have at least 2 dimensions");
  AT_CHECK(dy.ndimension() >= 2, "dy should have at least 2 dimensions");
  for (int dim = 0; dim < xhat.ndimension(); ++dim)
    AT_CHECK(xhat.size(dim) == dy.size(dim), "xhat and dy should have the same size");
  AT_CHECK(var.size(0) == xhat.size(1), "xhat and var don't have compatible dimensions");
  AT_CHECK(count.ndimension() == 1 && count.size(0) == 1, "count should be a vector with a single element");
  AT_CHECK(sum_dy.size(0) == xhat.size(1), "xhat and sum_dy don't have compatible dimensions");
  AT_CHECK(sum_xhat_dy.size(0) == xhat.size(1), "sum_xhat_dy and var don't have compatible dimensions");
  AT_CHECK(!weight.has_value() || weight.value().size(0) == xhat.size(1),
      "xhat and weight don't have compatible dimensions");

  // Check types
  AT_CHECK(xhat.scalar_type() == dy.scalar_type(), "xhat and dy should have the same type");
  AT_CHECK(!weight.has_value() || weight.value().scalar_type() == xhat.scalar_type(),
      "weight and xhat must have the same type");
  AT_CHECK(count.scalar_type() == at::ScalarType::Long, "count should have type int64");

  if (xhat.scalar_type() == at::ScalarType::Half) {
    AT_CHECK(var.scalar_type() == at::ScalarType::Float, "var must be float when xhat is half");
    AT_CHECK(sum_dy.scalar_type() == at::ScalarType::Float, "sum_dy must be float when xhat is half");
    AT_CHECK(sum_xhat_dy.scalar_type() == at::ScalarType::Float, "sum_xhat_dy must be float when xhat is half");
  } else {
    AT_CHECK(var.scalar_type() == xhat.scalar_type(), "xhat and var must have the same type");
    AT_CHECK(sum_dy.scalar_type() == xhat.scalar_type(), "xhat and sum_dy must have the same type");
    AT_CHECK(sum_xhat_dy.scalar_type() == xhat.scalar_type(), "sum_xhat_dy and var must have the same type");
  }

  if (xhat.is_cuda()) {
    return backward_cuda(xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, eps);
  } else {
    return backward_cpu(xhat, dy, var, count, sum_dy, sum_xhat_dy, weight, eps);
  }
}

at::Tensor backward_test(const at::Tensor& dy_, const at::Tensor& var, const c10::optional<at::Tensor>& weight,
                         float eps) {
  // Check dimensions
  AT_CHECK(dy_.ndimension() >= 2, "dy should have at least 2 dimensions");
  AT_CHECK(var.size(0) == dy_.size(1), "dy and var don't have compatible dimensions");
  AT_CHECK(!weight.has_value() || weight.value().size(0) == dy_.size(1),
      "dy and weight don't have compatible dimensions");

  // Check types
  AT_CHECK(!weight.has_value() || weight.value().scalar_type() == dy_.scalar_type(),
      "weight and dy must have the same type");

  if (dy_.scalar_type() == at::ScalarType::Half) {
    AT_CHECK(var.scalar_type() == at::ScalarType::Float, "var must be float when dy is half");
  } else {
    AT_CHECK(var.scalar_type() == dy_.scalar_type(), "dy and var must have the same type");
  }

  // TODO: optimize implementation for GPU
  auto dy = normalize_shape(dy_);
  auto mult = weight.has_value()
      ? (weight.value().to(var.options()).abs() + eps) / (var + eps).sqrt()
      : 1 / (var + eps).sqrt();
  auto dx = normalize_shape(mult) * dy.to(var.options());
  return dx.to(dy_.options()).view(dy_.sizes());
}

/***********************************************************************************************************************
 * Python Bindings
 **********************************************************************************************************************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::enum_<Activation>(m, "Activation")
    .value("LeakyReLU", Activation::LeakyReLU)
    .value("ELU", Activation::ELU)
    .value("Identity", Activation::Identity);

  // Forward methods
  m.def("statistics", &statistics, "Compute iABN statistics, i.e. mean, biased variance and sample count");
  m.def("reduce_statistics", &reduce_statistics, "Reduce statistics from multiple GPUs");
  m.def("forward", &forward, "iABN forward pass. This is an in-place operation w.r.t. x");

  // Backward methods
  m.def("backward_reduce", &backward_reduce,
      "First step of the backward pass. This is an in-place operation w.r.t. y_act and dy_act, which are transformed "
      "into xhat and dy, respectively.");
  m.def("backward", &backward, "Second step of the backward pass");
  m.def("backward_test", &backward_test, "Second step of the backward pass, test mode");
}
