/*
 * Copyright 2024 Meta
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace torch::indexing;

template <torch::ScalarType dtype>
void run_fht(void *a, void *out, uint32_t numel, uint32_t had_size, cudaStream_t stream);

constexpr bool is_power_of_two(uint32_t x)
{
    return x && !(x & (x - 1));
}

torch::Tensor hadamard_transform(at::Tensor &in, bool inplace)
{
    auto dtype = in.scalar_type();
    TORCH_CHECK(dtype == torch::ScalarType::Half || dtype == torch::ScalarType::BFloat16,
                "Only fp16 and bf16 supported currently");
    TORCH_CHECK(in.is_cuda());

    const int had_size = in.size(-1);
    TORCH_CHECK(is_power_of_two(had_size) && (had_size <= (1U << 15)),
                "Only power of two Hadamard sizes up to 2^15 are supported, got ", had_size);

    const auto res_shape = in.sizes();
    torch::Tensor x = in.reshape({-1, had_size});

    auto numel = in.numel();
    if (numel % 256 != 0)
    {
        x = torch::nn::functional::pad(
            x, torch::nn::functional::PadFuncOptions({0, 0, 0, (256 - numel % 256) / had_size}));
    }

    if (x.stride(-1) != 1)
    {
        x = x.contiguous();
    }
    torch::Tensor out = inplace ? x : torch::empty_like(x);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype == torch::ScalarType::Half)
    {
        run_fht<torch::ScalarType::Half>(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);
    }
    else
    {
        run_fht<torch::ScalarType::BFloat16>(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);
    }

    if (numel % 256 != 0)
    {
        out = out.index({Slice(0, numel / had_size)});
    }

    if (inplace && out.data_ptr() != in.data_ptr())
    {
        in.copy_(out.view(res_shape));
        return in;
    }
    return out.reshape(res_shape);
}

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hadamard_transform", &hadamard_transform, "A function to perform a fast Hadamard transform", py::arg("x"),
          py::arg("inplace") = false);
}
