#pragma once

#include <cmath>
#include <tuple>

#include <ATen/ATen.h>

/***********************************************************************************************************************
 * General defines
 **********************************************************************************************************************/

#ifdef __CUDACC__

#define HOST_DEVICE __host__ __device__
#define INLINE_HOST_DEVICE __host__ __device__ __forceinline__

#else
// CPU versions

#define HOST_DEVICE
#define INLINE_HOST_DEVICE inline

#endif // #ifdef __CUDACC__

/***********************************************************************************************************************
 * Utility functions
 **********************************************************************************************************************/

at::Tensor normalize_shape(const at::Tensor& x);

template<typename scalar_t, int64_t dim>
static at::TensorAccessor<scalar_t, dim> accessor_or_dummy(
    const c10::optional<at::Tensor>& t) {
  if (!t.has_value()) {
    return at::TensorAccessor<scalar_t, dim>(nullptr, nullptr, nullptr);
  }
  return t.value().accessor<scalar_t, dim>();
}

