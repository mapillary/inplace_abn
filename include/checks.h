#pragma once

#include <ATen/ATen.h>

#define CHECK_CUDA(x) AT_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")
#define CHECK_NOT_HALF(x) AT_CHECK((x).scalar_type() != at::ScalarType::Half, #x " can't have half dtype")