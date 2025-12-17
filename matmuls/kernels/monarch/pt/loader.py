import torch
import triton
from ..core.monarch_triton import monarch_kernel


def monarch_transform(x, w1, w2):
    """
    x: (Batch, N) or (Batch, K, P) ... flattened N=K*P
    w1: (K, Q, P)
    w2: (Q, S, K)

    Returns: (Batch, M) where M=Q*S
    """
    # Shapes
    batch = x.shape[0]
    n2, m1, n1 = w1.shape  # K, Q, P
    m1_check, m2, n2_check = w2.shape  # Q, S, K

    assert n2 == n2_check
    assert m1 == m1_check

    K, Q, P = n2, m1, n1
    S = m2

    x_view = x.view(batch, K, P)
    out = torch.empty((batch, Q, S), device=x.device, dtype=x.dtype)

    BLOCK_B = 16
    BLOCK_M1 = 16
    BLOCK_S = 64  # Tile S to manage register pressure

    BLOCK_N1 = triton.next_power_of_2(P)
    BLOCK_K = triton.next_power_of_2(K)

    # Split-K Heuristic for Small Batch Occupancy
    # A6000 has ~84 SMs. Target at least 84 blocks.
    num_blocks_batch = triton.cdiv(batch, BLOCK_B)
    num_blocks_q = triton.cdiv(Q, BLOCK_M1)
    num_blocks_s = triton.cdiv(S, BLOCK_S)
    total_blocks = num_blocks_batch * num_blocks_q * num_blocks_s

    SPLIT_K = 1
    # Only split if occupancy is low and K is large enough
    if total_blocks < 84 and K >= 16:
        # Try to fill GPU. Max Split factor: 8 (or K/16)
        split_factor = min(84 // total_blocks, 8)
        split_factor = max(1, split_factor)  # Minimum 1
        # Also ensure chunk size is reasonable (target at least 16)
        if K // split_factor >= 16:
            SPLIT_K = split_factor

    # Initialize Output to Zero if Split-K is used (atomic accumulation)
    if SPLIT_K > 1:
        out.zero_()

    # Grid: (Batch * SPLIT_K, Q, S)
    grid = (
        num_blocks_batch * SPLIT_K,
        num_blocks_q,
        num_blocks_s,
    )

    monarch_kernel[grid](
        x_view,
        w1,
        w2,
        out,
        batch,
        P,
        K,
        Q,
        S,
        x_view.stride(0),
        x_view.stride(1),
        x_view.stride(2),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_B=BLOCK_B,
        BLOCK_N1=BLOCK_N1,
        BLOCK_K=BLOCK_K,
        BLOCK_M1=BLOCK_M1,
        BLOCK_S=BLOCK_S,
        BLOCK_K_TILE=16,
        BLOCK_P_TILE=32,
        SPLIT_K=SPLIT_K,
    )

    return out.view(batch, -1)
