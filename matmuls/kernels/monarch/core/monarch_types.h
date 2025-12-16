#ifndef MONARCH_TYPES_H
#define MONARCH_TYPES_H

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Define Scalar Types
enum class MonarchScalarType { Float16, Bfloat16, Float32 };

// Helper struct for CUDA kernel dispatch
template <typename T> struct CudaTypeTraits;

template <> struct CudaTypeTraits<__half> {
  static constexpr MonarchScalarType type = MonarchScalarType::Float16;
};

template <> struct CudaTypeTraits<__nv_bfloat16> {
  static constexpr MonarchScalarType type = MonarchScalarType::Bfloat16;
};

template <> struct CudaTypeTraits<float> {
  static constexpr MonarchScalarType type = MonarchScalarType::Float32;
};

#endif // MONARCH_TYPES_H
