#pragma once

#include <ATen/ATen.h>

#ifdef TORCH_CHECK
#define IABN_CHECK TORCH_CHECK
#else
#define IABN_CHECK AT_CHECK
#endif

#define CHECK_CUDA(x) IABN_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) IABN_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")
#define CHECK_NOT_HALF(x) IABN_CHECK((x).scalar_type() != at::ScalarType::Half, #x " can't have type Half")
#define CHECK_SAME_TYPE(x, y) IABN_CHECK((x).scalar_type() == (y).scalar_type(), #x " and " #y " must have the same scalar type")

inline bool have_same_dims(const at::Tensor& x, const at::Tensor& y) {
  bool success = x.ndimension() == y.ndimension();
  for (int64_t dim = 0; dim < x.ndimension(); ++dim)
    success &= x.size(dim) == y.size(dim);
  return success;
}

inline bool is_compatible_weight(const at::Tensor& x, const at::Tensor& w) {
  // Dimensions check
  bool success = w.ndimension() == 1;
  success &= x.size(1) == w.size(0);

  // Typing check
  if (x.scalar_type() == at::ScalarType::Half) {
    success &= (w.scalar_type() == at::ScalarType::Half) || (w.scalar_type() == at::ScalarType::Float);
  } else {
    success &= x.scalar_type() == w.scalar_type();
  }

  return success;
}

inline bool is_compatible_stat(const at::Tensor& x, const at::Tensor& s) {
  // Dimensions check
  bool success = s.ndimension() == 1;
  success &= x.size(1) == s.size(0);

  // Typing check
  if (x.scalar_type() == at::ScalarType::Half) {
    success &= s.scalar_type() == at::ScalarType::Float;
  } else {
    success &= x.scalar_type() == s.scalar_type();
  }

  return success;
}
