#include "../core/monarch_types.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <cuda_runtime.h>

// Types
// Using JAX Typed FFI (API v1)

// Declare Core Implementation
extern "C" {
void monarch_forward_cuda_f16(cudaStream_t stream, const void *x,
                              const void *w1, const void *w2, void *out,
                              void *workspace, int batch, int n1, int n2,
                              int m1, int m2);

void monarch_forward_cuda_bf16(cudaStream_t stream, const void *x,
                               const void *w1, const void *w2, void *out,
                               void *workspace, int batch, int n1, int n2,
                               int m1, int m2);
}

namespace ffi = xla::ffi;

// Handler for F16
ffi::Error MonarchF16(cudaStream_t stream, ffi::Buffer<ffi::DataType::F16> x,
                      ffi::Buffer<ffi::DataType::F16> w1,
                      ffi::Buffer<ffi::DataType::F16> w2,
                      ffi::Result<ffi::Buffer<ffi::DataType::F16>> out,
                      ffi::Result<ffi::Buffer<ffi::DataType::F16>> workspace,
                      int32_t batch, int32_t n1, int32_t n2, int32_t m1,
                      int32_t m2) {
  monarch_forward_cuda_f16(stream, x.untyped_data(), w1.untyped_data(),
                           w2.untyped_data(), out->untyped_data(),
                           workspace->untyped_data(), batch, n1, n2, m1, m2);
  return ffi::Error::Success();
}

// Handler for BF16
ffi::Error MonarchBF16(cudaStream_t stream, ffi::Buffer<ffi::DataType::BF16> x,
                       ffi::Buffer<ffi::DataType::BF16> w1,
                       ffi::Buffer<ffi::DataType::BF16> w2,
                       ffi::Result<ffi::Buffer<ffi::DataType::BF16>> out,
                       ffi::Result<ffi::Buffer<ffi::DataType::BF16>> workspace,
                       int32_t batch, int32_t n1, int32_t n2, int32_t m1,
                       int32_t m2) {
  monarch_forward_cuda_bf16(stream, x.untyped_data(), w1.untyped_data(),
                            w2.untyped_data(), out->untyped_data(),
                            workspace->untyped_data(), batch, n1, n2, m1, m2);
  return ffi::Error::Success();
}

// Define the handler symbols
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MonarchF16Symbol, MonarchF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // x
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // w1
        .Arg<ffi::Buffer<ffi::DataType::F16>>() // w2
        .Ret<ffi::Buffer<ffi::DataType::F16>>() // out
        .Ret<ffi::Buffer<ffi::DataType::F16>>() // workspace
        .Attr<int32_t>("batch")
        .Attr<int32_t>("n1")
        .Attr<int32_t>("n2")
        .Attr<int32_t>("m1")
        .Attr<int32_t>("m2"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MonarchBF16Symbol, MonarchBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>() // x
        .Arg<ffi::Buffer<ffi::DataType::BF16>>() // w1
        .Arg<ffi::Buffer<ffi::DataType::BF16>>() // w2
        .Ret<ffi::Buffer<ffi::DataType::BF16>>() // out
        .Ret<ffi::Buffer<ffi::DataType::BF16>>() // workspace
        .Attr<int32_t>("batch")
        .Attr<int32_t>("n1")
        .Attr<int32_t>("n2")
        .Attr<int32_t>("m1")
        .Attr<int32_t>("m2"));
