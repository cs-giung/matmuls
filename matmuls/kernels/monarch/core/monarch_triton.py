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
):
    """
    Fused Monarch Kernel (P-Tiled FP32 Accumulation Version).
    """
    pid_b = tl.program_id(0)
    pid_m1 = tl.program_id(1)

    # Batch range for this block
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < Batch

    # M1 (Q) range for this block
    offs_q_base = pid_m1 * BLOCK_M1

    # K range for this block
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Output S offsets
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    # Tile size for P (Use internal logic for best perf)
    BLOCK_P_TILE_VAL: tl.constexpr = 32 if BLOCK_N1 >= 32 else BLOCK_N1

    # Compute over BLOCK_M1. Process one Q at a time to manage registers.
    for q_idx in range(BLOCK_M1):
        q = offs_q_base + q_idx

        # Check validity of q
        if q < M1:
            # --- Step 1: Compute Y1 slice for this q ---
            # Y1_q[b, k] = sum_p X[b, k, p] * W1[k, q, p]

            # Use FP32 accumulator for precision
            y1_q_f32 = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)

            # Loop over P (N1) in tiles to save register/memory pressure
            for p_start in range(0, N1, BLOCK_P_TILE_VAL):
                # Offsets for P
                offs_p = p_start + tl.arange(0, BLOCK_P_TILE_VAL)
                mask_p = offs_p < N1

                # Load X slice (B, K, P_tile)
                x_ptrs = x_ptr + (
                    offs_b[:, None, None] * stride_xb
                    + offs_k[None, :, None] * stride_xk
                    + offs_p[None, None, :] * stride_xp
                )

                x_slice = tl.load(
                    x_ptrs,
                    mask=mask_b[:, None, None]
                    & mask_k[None, :, None]
                    & mask_p[None, None, :],
                    other=0.0,
                )

                # Load W1 slice (K, P_tile) for current q
                # W1 shape (K, Q, P). Stride K, Q, P.
                w1_ptrs = w1_ptr + (
                    offs_k[:, None] * stride_w1k
                    + q * stride_w1q
                    + offs_p[None, :] * stride_w1p
                )

                w1_slice = tl.load(
                    w1_ptrs, mask=mask_k[:, None] & mask_p[None, :], other=0.0
                )

                # Accumulate in FP32
                # x (B, K, P) * w1 (K, P). w1 broadcasts to (1, K, P).
                # sum axis 2 (P). Result (B, K).
                y1_q_f32 += tl.sum(
                    x_slice.to(tl.float32) * w1_slice.to(tl.float32)[None, :, :], axis=2
                )

            # Cast back to input dtype for Phase 2 dot
            y1_q = y1_q_f32.to(x_ptr.dtype.element_ty)

            # --- Step 2: Compute Out slice for this q ---
            # Out[b, q, s] = sum_k Y1[b, k, q] * W2[q, s, k]
            # Here y1_q is (B, K). (Since q is fixed).
            # We want Out[b, s] = Y1[b, k] * W2[s, k].T
            # W2 shape (Q, S, K). Fixed q -> (S, K).
            # Dot: (B, K) @ (S, K).T -> (B, S).

            # Load W2 slice for this q
            w2_ptrs = w2_ptr + (
                q * stride_w2q
                + offs_s[:, None] * stride_w2s
                + offs_k[None, :] * stride_w2k
            )

            w2_slice = tl.load(
                w2_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0
            )

            # y1_q (B, K). w2_slice (S, K).
            # dot -> (B, S).
            out_qs = tl.dot(y1_q, tl.trans(w2_slice))

            # Write to Output
            out_ptrs = out_ptr + (
                offs_b[:, None] * stride_out_b
                + q * stride_out_q
                + offs_s[None, :] * stride_out_s
            )

            tl.store(
                out_ptrs,
                out_qs.to(x_ptr.dtype.element_ty),
                mask=mask_b[:, None] & mask_s[None, :],
            )
