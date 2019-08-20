#pragma once

#define NORMAL_CASE_TYPE(enum_type, type, ...)     \
  case enum_type: {                                \
    using scalar_t = type;                         \
    using prmscalar_t = type;                      \
    return __VA_ARGS__();                          \
}

#define HALF_CASE_TYPE(enum_type, x_type, w_scalar_type, ...)         \
  case enum_type: {                                                   \
    using scalar_t = x_type;                                          \
    if (w_scalar_type == at::ScalarType::Half) {                      \
      using prmscalar_t = at::Half;                                   \
      return __VA_ARGS__();                                           \
    } else if (w_scalar_type == at::ScalarType::Float) {              \
      using prmscalar_t = float;                                      \
      return __VA_ARGS__();                                           \
    } else {                                                          \
      AT_ERROR("Unsupported type combination '" #enum_type "', '" #w_scalar_type "'"); \
    }                                                                 \
  }

#define DOUBLE_DISPATCH(XTYPE, WTYPE, NAME, ...)                             \
  [&] {                                                                      \
    const auto& x_type = XTYPE;                                              \
    const auto& w_type = WTYPE;                                              \
    switch (x_type) {                                                        \
      NORMAL_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)          \
      NORMAL_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)            \
      HALF_CASE_TYPE(at::ScalarType::Half, at::Half, w_type, __VA_ARGS__)    \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(x_type), "'");    \
    }                                                                        \
}()

#ifdef WITH_CUDA
#define CUDA_DISPATCH(REF_TENSOR, METHOD, ...) \
  if ((REF_TENSOR).is_cuda()) {                \
    return METHOD ## _cuda(__VA_ARGS__);       \
  } else {                                     \
    return METHOD ## _cpu(__VA_ARGS__);        \
  }
#else
#define CUDA_DISPATCH(REF_TENSOR, METHOD, ...)                \
  if ((REF_TENSOR).is_cuda()) {                               \
    AT_ERROR("CUDA support was not enabled at compile time"); \
  } else {                                                    \
    return METHOD ## _cpu(__VA_ARGS__);                       \
  }
#endif
