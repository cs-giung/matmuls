#include "../core/hadamard_types.h"
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <cuda_runtime.h>

// Forward declaration of the kernel launcher
using matmuls::hadamard::HadamardScalarType;

template <HadamardScalarType dtype>
void run_fht(void *a, void *out, uint32_t numel, uint32_t had_size,
             cudaStream_t stream);

namespace ffi = xla::ffi;

ffi::Error HadamardF16(cudaStream_t stream, ffi::Buffer<ffi::DataType::F16> x,
                       ffi::ResultBuffer<ffi::DataType::F16> out,
                       int32_t had_size) {
  uint32_t numel = x.element_count();
  run_fht<HadamardScalarType::Half>(x.untyped_data(), out->untyped_data(),
                                    numel, (uint32_t)had_size, stream);
  return ffi::Error::Success();
}

ffi::Error HadamardBF16(cudaStream_t stream, ffi::Buffer<ffi::DataType::BF16> x,
                        ffi::ResultBuffer<ffi::DataType::BF16> out,
                        int32_t had_size) {
  uint32_t numel = x.element_count();
  run_fht<HadamardScalarType::BFloat16>(x.untyped_data(), out->untyped_data(),
                                        numel, (uint32_t)had_size, stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(HadamardF16Symbol, HadamardF16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<int32_t>("had_size"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(HadamardBF16Symbol, HadamardBF16,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::BF16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::BF16>>()
                                  .Attr<int32_t>("had_size"));
