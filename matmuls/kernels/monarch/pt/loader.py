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

    BLOCK_N1 = triton.next_power_of_2(P)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_S = triton.next_power_of_2(S)

    # Tile M1 dimension to improve occupancy and reduce spill
    # Set tile size for M1 (Q)
    BLOCK_M1 = 16  # Process 16 Qs per kernel instance

    grid = (triton.cdiv(batch, BLOCK_B), triton.cdiv(Q, BLOCK_M1))

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
    )

    return out.view(batch, -1)
