#include "monarch_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(status) << std::endl;                    \
      return;                                                                  \
    }                                                                          \
  }

#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error at line " << __LINE__ << ": " << status       \
                << std::endl;                                                  \
      return;                                                                  \
    }                                                                          \
  }

// Thread-local cuBLAS handle to avoid recreation overhead
cublasHandle_t get_cublas_handle() {
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

// Permute Kernel Generic: (D1, D2, D3) -> (O1, O2, O3)
// We only need specific permutations.
// P1: (K, B, Q) -> (Q, B, K). (Transpose outer dims).
// P2: (L, B, S) -> (B, L, S). (Rotate).

template <typename T>
__global__ void permute_KBQ_to_QBK(const T *__restrict__ in,
                                   T *__restrict__ out, int K, int B, int Q) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * B * Q;
  if (idx < total) {
    // Input (K, B, Q) -> k, b, q
    int q = idx % Q;
    int tmp = idx / Q;
    int b = tmp % B;
    int k = tmp / B;

    // Output (Q, B, K) -> q, b, k
    int out_idx = q * (B * K) + b * K + k;
    out[out_idx] = in[idx];
  }
}

template <typename T>
__global__ void permute_LBS_to_BLS(const T *__restrict__ in,
                                   T *__restrict__ out, int L, int B, int S) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = L * B * S;
  if (idx < total) {
    // Input (L, B, S) -> l, b, s
    int s = idx % S;
    int tmp = idx / S;
    int b = tmp % B;
    int l = tmp / B;

    // Output (B, L, S) -> b, l, s
    int out_idx = b * (L * S) + l * S + s;
    out[out_idx] = in[idx];
  }
}

template <typename T>
void monarch_forward_impl(cudaStream_t stream, const T *d_x, const T *d_w1,
                          const T *d_w2, T *d_out, T *d_workspace, int batch,
                          int n1, int n2, int m1, int m2) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  float alpha = 1.0f;
  float beta = 0.0f;

  int K = n2;
  int P = n1;
  int Q = m1;

  cudaDataType d_type = CUDA_R_16F;
  cudaDataType c_type = CUDA_R_32F;

  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    d_type = CUDA_R_16BF;
  } else if constexpr (std::is_same_v<T, float>) {
    d_type = CUDA_R_32F;
  }

  // Buffer allocation (offsets in workspace)
  // workspace size assumed sufficient.
  // y1: (K, B, Q). Size K*B*Q.
  // y1_perm: (Q, B, K). Size Q*B*K. (Same).
  // y2: (L, B, S) -> (Q, B, S). Size Q*B*S.
  // We need y1 and y1_perm to be distinct?
  // Yes.
  // y1_perm and y2 distinct?
  // Yes.
  // We can reuse y1 for y2 if sizes allow?
  // Let's assume linear allocation.
  T *y1 = d_workspace;
  T *y1_perm = y1 + (size_t(K) * batch * Q);
  T *y2 = y1_perm + (size_t(Q) * batch * K);

  // Phase 1: x (Batch, K, P) @ w1 (K, Q, P)^T -> y1 (K, Batch, Q)
  // Override: We write to y1 which is Contiguous (K, Batch, Q).
  // StrideC = Batch * Q.
  // LDC = Q. (Stride between col b (0) and col b+1 (1)).
  // Wait. y1[k] is (Batch, Q) Row Major -> (Q, Batch) Col Major.
  // Row stride (b): LDC.
  // Col stride (q): 1.
  // y1[k, b, q].
  // Stride between b and b+1: Q. (So LDC=Q).
  // Stride between k and k+1: Batch*Q. (StrideC).
  // Correct.

  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q, batch, P, &alpha, d_w1, d_type, P,
      int64_t(Q) * P, d_x, d_type, K * P, P, &beta, y1, d_type, Q,
      int64_t(batch) * Q, K, c_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Permute 1: y1 (K, B, Q) -> y1_perm (Q, B, K)
  int total_elements_1 = K * batch * Q;
  int threads = 256;
  int blocks_1 = (total_elements_1 + threads - 1) / threads;
  permute_KBQ_to_QBK<<<blocks_1, threads, 0, stream>>>(y1, y1_perm, K, batch,
                                                       Q);

  // Phase 2: y1_perm (Q, B, K) @ w2 (Q, S, K)^T -> y2 (Q, B, S)
  // L=Q. R=K. S=m2.
  // Input B (y1_perm): L matrices of (B, R).
  // y1_perm[l] is (B, R) Row Major -> (R, B) Col Major.
  // Stride between l and l+1: Batch*R. (StrideB).
  // LDB: R. (Stride between b and b+1).
  // Input A (w2): (L, S, R). w2[l] is (S, R) (P1 logic Q, P).
  // Op=T -> (R, S) Col Major.
  // StrideA = S*R. LDA = R.
  // Output C (y2): (L, Batch, S).
  // StrideC = Batch*S. LDC = S.

  int L = Q;
  int R = K;
  int S = m2;

  CHECK_CUBLAS(cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, S, batch, R, &alpha, d_w2, d_type, R,
      int64_t(S) * R, y1_perm, d_type, R, int64_t(batch) * R, &beta, y2, d_type,
      S, int64_t(batch) * S, L, c_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Permute 2: y2 (Q, B, S) -> out (B, Q, S)
  // (L=Q).
  int total_elements_2 = L * batch * S;
  int blocks_2 = (total_elements_2 + threads - 1) / threads;
  permute_LBS_to_BLS<<<blocks_2, threads, 0, stream>>>(y2, d_out, L, batch, S);
}

// C-API Extensions
extern "C" {
void monarch_forward_cuda_f16(cudaStream_t stream, const void *x,
                              const void *w1, const void *w2, void *out,
                              void *workspace, int batch, int n1, int n2,
                              int m1, int m2) {
  monarch_forward_impl<__half>(stream, (const __half *)x, (const __half *)w1,
                               (const __half *)w2, (__half *)out,
                               (__half *)workspace, batch, n1, n2, m1, m2);
}

void monarch_forward_cuda_bf16(cudaStream_t stream, const void *x,
                               const void *w1, const void *w2, void *out,
                               void *workspace, int batch, int n1, int n2,
                               int m1, int m2) {
  monarch_forward_impl<__nv_bfloat16>(
      stream, (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)w1,
      (const __nv_bfloat16 *)w2, (__nv_bfloat16 *)out,
      (__nv_bfloat16 *)workspace, batch, n1, n2, m1, m2);
}
}
