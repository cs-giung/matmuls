#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#include "../core/monarch_types.h"

// Declaration of C wrappers
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

torch::Tensor monarch_transform(torch::Tensor x, torch::Tensor w1,
                                torch::Tensor w2) {
  // Input checks
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w1.is_cuda(), "w1 must be a CUDA tensor");
  TORCH_CHECK(w2.is_cuda(), "w2 must be a CUDA tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w1.is_contiguous(), "w1 must be contiguous");
  TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");

  // x shape: (..., N) -> View as ( Batch, N )
  // But algorithm expects (Batch, n2, n1)
  // We assume the user provides x such that x.shape[-1] = n1 * n2
  // w1 shape: (n2, m1, n1)
  // w2 shape: (m1, m2, n2)

  int n2 = w1.size(0);
  int m1 = w1.size(1);
  int n1 = w1.size(2);

  TORCH_CHECK(w2.size(0) == m1, "w2 dim 0 must match w1 dim 1 (m1)");
  int m2 = w2.size(1);
  TORCH_CHECK(w2.size(2) == n2, "w2 dim 2 must match w1 dim 0 (n2)");

  // Check x last dimension
  int N = n1 * n2;
  TORCH_CHECK(x.size(-1) == N, "x last dim must match n1*n2");

  // Flatten Batch
  int batch_size = x.numel() / N;

  // Output shape
  int M = m1 * m2;
  std::vector<int64_t> out_shape = x.sizes().vec();
  out_shape.back() = M;

  torch::Tensor out = torch::empty(out_shape, x.options());

  // Allocate Workspace
  // y1: (K, B, Q)
  // y1_perm: (Q, B, K)
  // y2: (Q, B, S)
  // Total Size = batch * (K*Q + Q*K + Q*S) = batch * m1 * (2*n2 + m2)
  int64_t workspace_elements = int64_t(batch_size) * m1 * (2 * n2 + m2);
  torch::Tensor workspace = torch::empty({workspace_elements}, x.options());

  // Stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Dispatch
  if (x.dtype() == torch::kFloat16) {
    monarch_forward_cuda_f16(stream, x.data_ptr(), w1.data_ptr(), w2.data_ptr(),
                             out.data_ptr(), workspace.data_ptr(), batch_size,
                             n1, n2, m1, m2);
  } else if (x.dtype() == torch::kBFloat16) {
    monarch_forward_cuda_bf16(stream, x.data_ptr(), w1.data_ptr(),
                              w2.data_ptr(), out.data_ptr(),
                              workspace.data_ptr(), batch_size, n1, n2, m1, m2);
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float16 or bfloat16.");
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("monarch_transform", &monarch_transform,
        "Monarch Matrix Multiplication");
}
