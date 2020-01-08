#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include "utils.h"
#ifdef __CUDACC__
#include "cuda_utils.cuh"
#endif

/***********************************************************************************************************************
 * Enums
 **********************************************************************************************************************/

enum class Activation { LeakyReLU, ELU, Identity };

/***********************************************************************************************************************
 * CPU / Cuda methods
 **********************************************************************************************************************/

std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics_cpu(const at::Tensor& x);
std::tuple<at::Tensor, at::Tensor, at::Tensor> statistics_cuda(const at::Tensor& x);

std::tuple<at::Tensor, at::Tensor, at::Tensor> reduce_statistics_cuda(
    const at::Tensor& all_mean, const at::Tensor& all_var, const at::Tensor& all_count);

void forward_cpu(at::Tensor& x, const at::Tensor& mean, const at::Tensor& var,
                 const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
                 float eps, Activation activation, float activation_param);
void forward_cuda(at::Tensor& x, const at::Tensor& mean, const at::Tensor& var,
                  const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias,
                  float eps, Activation activation, float activation_param);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_cpu(
    const at::Tensor& y_act, const at::Tensor& dy_act, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_reduce_cuda(
    const at::Tensor& y_act, const at::Tensor& dy_act, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, float eps, Activation activation, float activation_param);

void backward_cpu(const at::Tensor& xhat, at::Tensor& dy, const at::Tensor& var, const at::Tensor& count,
                  const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy,
                  const c10::optional<at::Tensor>& weight, float eps);
void backward_cuda(const at::Tensor& xhat, at::Tensor& dy, const at::Tensor& var, const at::Tensor& count,
                   const at::Tensor& sum_dy, const at::Tensor& sum_xhat_dy,
                   const c10::optional<at::Tensor>& weight, float eps);

/***********************************************************************************************************************
 * Handling of activation functions
 **********************************************************************************************************************/

template<typename scalar_t, Activation activation> struct ActivationFn;

template<typename scalar_t>
struct ActivationFn<scalar_t, Activation::LeakyReLU> {
  static INLINE_HOST_DEVICE void forward(scalar_t& x, float activation_param) {
    x = (x >= 0) ? x : static_cast<scalar_t>(x * activation_param);
  }

  static INLINE_HOST_DEVICE void backward(scalar_t y_act, scalar_t dy_act, float activation_param,
                                          scalar_t& y, scalar_t& dy) {
    if (y_act >= 0) {
      y = y_act;
      dy = dy_act;
    } else {
      y = static_cast<scalar_t>(y_act / activation_param);
      dy = static_cast<scalar_t>(dy_act * activation_param);
    }
  }
};

template<typename scalar_t>
struct ActivationFn<scalar_t, Activation::ELU> {
  static INLINE_HOST_DEVICE void forward(scalar_t& x, float activation_param) {
    x = (x >= 0) ? x : static_cast<scalar_t>(activation_param * (::exp(x) - 1));
  }

  static INLINE_HOST_DEVICE void backward(scalar_t y_act, scalar_t dy_act, float activation_param,
                                          scalar_t& y, scalar_t& dy) {
    if (y_act >= 0) {
      y = y_act;
      dy = dy_act;
    } else {
      y = ::log1p(static_cast<scalar_t>(y_act / activation_param));
      dy = static_cast<scalar_t>(dy_act * (y_act + activation_param));
    }
  }
};

template<typename scalar_t>
struct ActivationFn<scalar_t, Activation::Identity> {
  static INLINE_HOST_DEVICE void forward(scalar_t& x, float activation_param) {}

  static INLINE_HOST_DEVICE void backward(scalar_t y_act, scalar_t dy_act, float activation_param,
                                          scalar_t& y, scalar_t& dy) {
    y = y_act;
    dy = dy_act;
  }
};
