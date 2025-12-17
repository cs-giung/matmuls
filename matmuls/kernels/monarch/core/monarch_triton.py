import triton
import triton.language as tl


@triton.jit
def monarch_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    Batch,
    N1,
    K,
    M1,
    S,  # K=N2, S=M2
    stride_xb,
    stride_xk,
    stride_xp,
    stride_w1k,
    stride_w1q,
    stride_w1p,
    stride_w2q,
    stride_w2s,
    stride_w2k,
    stride_out_b,
    stride_out_q,
    stride_out_s,
    BLOCK_B: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_S: tl.constexpr,
    # Legacy args ignored
    BLOCK_K_TILE: tl.constexpr,
    BLOCK_P_TILE: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Fused Monarch Kernel (K-Loop Dual GEMM) with Split-K Support.
    Computes Out[b, q, s] = sum_k ( (X[b, k, :] @ W1[k, q, :].T) * W2[q, s, k] )

    Structure:
    - Loop over K partition.
    - Compute Y[b, q] = X[b, k, :] @ W1[k, q, :].T
    - Accumulate Out[b, q, s] += Y[b, q] * W2[q, s, k]
    """
    # Grid X mapping: (Batch_tiles * SPLIT_K)
    # We recover batch and split_k indices
    pid_flat_b = tl.program_id(0)
    pid_b = pid_flat_b // SPLIT_K
    pid_split_k = pid_flat_b % SPLIT_K

    pid_q = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_q = pid_q * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    offs_p = tl.arange(0, BLOCK_N1)  # Assume P <= BLOCK_N1

    mask_b = offs_b < Batch
    mask_q = offs_q < M1
    mask_s = offs_s < S
    mask_p = offs_p < N1

    # Accumulator (B, Q, S)
    acc = tl.zeros((BLOCK_B, BLOCK_M1, BLOCK_S), dtype=tl.float32)

    # Split-K Logic
    # Partition K into SPLIT_K chunks
    # Chunk size = K // SPLIT_K (rounded up?)
    # Or just simple range:
    k_chunk_size = tl.cdiv(K, SPLIT_K)
    k_start = pid_split_k * k_chunk_size
    k_end = tl.minimum(k_start + k_chunk_size, K)

    for k in range(k_start, k_end):
        # --- Phase 1: Y = X @ W1.T ---
        # Load X (B, P)
        x_ptrs = x_ptr + (
            offs_b[:, None] * stride_xb + k * stride_xk + offs_p[None, :] * stride_xp
        )
        x_k = tl.load(x_ptrs, mask=mask_b[:, None] & mask_p[None, :], other=0.0)

        # Load W1 (Q, P)
        w1_ptrs = w1_ptr + (
            k * stride_w1k + offs_q[:, None] * stride_w1q + offs_p[None, :] * stride_w1p
        )
        w1_k = tl.load(w1_ptrs, mask=mask_q[:, None] & mask_p[None, :], other=0.0)

        # Dot: (B, P) @ (Q, P).T -> (B, Q)
        y_k = tl.dot(x_k, tl.trans(w1_k), allow_tf32=False)
        y_k = y_k.to(x_ptr.dtype.element_ty)

        # --- Phase 2: Acc += Y * W2 ---
        # Load W2 (Q, S)
        w2_ptrs = w2_ptr + (
            offs_q[:, None] * stride_w2q + offs_s[None, :] * stride_w2s + k * stride_w2k
        )
        w2_k = tl.load(w2_ptrs, mask=mask_q[:, None] & mask_s[None, :], other=0.0)

        # Broadcast Multiply
        # acc: (B, Q, S) += y_k[:, :, None] * w2_k[None, :, :]
        acc += y_k[:, :, None].to(tl.float32) * w2_k[None, :, :].to(tl.float32)

    # Store Out
    out_ptrs = out_ptr + (
        offs_b[:, None, None] * stride_out_b
        + offs_q[None, :, None] * stride_out_q
        + offs_s[None, None, :] * stride_out_s
    )

    # Check boundaries + Split-K logic
    mask_out = mask_b[:, None, None] & mask_q[None, :, None] & mask_s[None, None, :]

    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_out)
    else:
        tl.atomic_add(out_ptrs, acc.to(x_ptr.dtype.element_ty), mask=mask_out)
