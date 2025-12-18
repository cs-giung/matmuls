import torch
import math
import triton
import triton.language as tl
from scipy.linalg import hadamard

# Cache for Hadamard matrices
_H_CACHE = {}


def _get_hadamard_matrix(n, device, dtype):
    key = (n, device, dtype)
    if key not in _H_CACHE:
        if not (n & (n - 1) == 0):
            raise ValueError(f"N must be power of 2, got {n}")
        h = torch.tensor(hadamard(n), dtype=dtype, device=device)
        # Normalize to match FHT library (1/sqrt(N)) scaling per dims
        h = h / math.sqrt(n)
        _H_CACHE[key] = h
    return _H_CACHE[key]


@triton.jit
def hadamard_fused_kernel(
    x_ptr,
    h1_ptr,
    h2_ptr,
    B,
    stride_xb,
    stride_xh,
    stride_xw,  # Input strides (Batch, N1, N2)
    stride_h1_r,
    stride_h1_c,  # H1 strides
    stride_h2_r,
    stride_h2_c,  # H2 strides
    stride_out_b,
    stride_out_h,
    stride_out_w,  # Output strides
    out_ptr,  # Output pointer (moved to end for jax_triton compatibility)
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
):
    # Map PID to Batch Index
    pid = tl.program_id(0)
    # We can handle multiple batch items per PID if we want, but let's stick to 1-to-1 or tiled batching.
    # If B is large, 1-to-1 is fine.

    # Offsets for H1 (N1 x N1)
    offs_n1 = tl.arange(0, BLOCK_SIZE_N1)
    offs_k1 = tl.arange(0, BLOCK_SIZE_N1)  # N1 is K for H1
    h1_ptrs = h1_ptr + (offs_n1[:, None] * stride_h1_r + offs_k1[None, :] * stride_h1_c)
    h1 = tl.load(h1_ptrs)  # Load H1 into registers/SRAM

    # Offsets for H2 (N2 x N2)
    offs_k2 = tl.arange(0, BLOCK_SIZE_N2)
    offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
    h2_ptrs = h2_ptr + (offs_k2[:, None] * stride_h2_r + offs_n2[None, :] * stride_h2_c)
    h2 = tl.load(h2_ptrs)  # Load H2

    # Loop over assigned batch items
    # For now, assume grid covers B.
    # pid is batch index?
    # If grid is (B,), then pid is batch index.

    batch_idx = pid
    if batch_idx >= B:
        return

    # Load X [Batch_idx, :, :] -> (N1, N2)
    # Pointers
    # X ptr: Base + b * stride_xb + n1 * stride_xh + n2 * stride_xw
    offs_xn1 = tl.arange(0, BLOCK_SIZE_N1)
    offs_xn2 = tl.arange(0, BLOCK_SIZE_N2)

    x_ptrs_base = x_ptr + batch_idx * stride_xb
    x_ptrs = x_ptrs_base + (
        offs_xn1[:, None] * stride_xh + offs_xn2[None, :] * stride_xw
    )

    # Load X
    x = tl.load(x_ptrs)

    # Compute T = X @ H2 -> (N1, N2) @ (N2, N2) -> (N1, N2)
    # tl.dot(a, b). a=(M, K), b=(K, N).
    # here X=(N1, N2), H2=(N2, N2).
    # Correct.
    # Use fp32 accumulation ??
    t = tl.dot(x, h2, out_dtype=tl.float32)

    # Compute Y = H1 @ T -> (N1, N1) @ (N1, N2) -> (N1, N2)
    # H1 is (N1, N1). T is (N1, N2).
    t = t.to(x.dtype)  # Cast back? Or keep float32? Better keep float32.
    # If keeping float32, H1 should be cast or dot handles it?
    # dot(float16, float32) might not work.
    # Better load H1 as float16.
    # If accumulator is float32, we can chain dots?
    # tl.dot arguments must be float16/bfloat16/float32.
    # If X, H are float16.
    # T is float32.
    # dot(H1, T). H1 is float16. T is float32.
    # Is mixed precision dot supported? inputs must match usually.
    # Cast T back to float16 for second dot?
    # This loses precision but standard Tensor Cores require f16 inputs.
    t_f16 = t.to(x.dtype)

    y = tl.dot(h1, t_f16, out_dtype=tl.float32)

    # Store Y
    y_f16 = y.to(x.dtype)
    out_ptrs_base = out_ptr + batch_idx * stride_out_b
    out_ptrs = out_ptrs_base + (
        offs_xn1[:, None] * stride_out_h + offs_xn2[None, :] * stride_out_w
    )

    tl.store(out_ptrs, y_f16)


def hadamard_triton(x):
    """
    Fused Triton Kernel Implementation.
    """
    B, N = x.shape
    device = x.device
    dtype = x.dtype

    log_n = int(math.log2(N))
    n1_bits = log_n // 2
    n2_bits = log_n - n1_bits

    N1 = 1 << n1_bits
    N2 = 1 << n2_bits

    x_reshaped = x.view(B, N1, N2)

    h1 = _get_hadamard_matrix(N1, device, dtype)
    h2 = _get_hadamard_matrix(N2, device, dtype)

    out = torch.empty_like(x_reshaped)

    # Grid: One block per batch item
    grid = (B,)

    hadamard_fused_kernel[grid](
        x_reshaped,
        h1,
        h2,
        B,
        x_reshaped.stride(0),
        x_reshaped.stride(1),
        x_reshaped.stride(2),
        h1.stride(0),
        h1.stride(1),
        h2.stride(0),
        h2.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_N1=N1,
        BLOCK_SIZE_N2=N2,
        num_warps=4,  # 64x64 typically needs 4-8 warps
    )

    return out.view(B, N)
