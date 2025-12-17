import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from ..core.monarch_triton import monarch_kernel


def monarch_transform(x, w1, w2):
    """
    Monarch Matrix Multiplication (JAX + Triton).

    x: (..., N) or (..., K, P) - flattened N=K*P
    w1: (K, Q, P)
    w2: (Q, S, K)

    Returns: (..., M) where M=Q*S
    """
    # Shapes
    # x might be (Batch..., N)
    batch_shape = x.shape[:-1]

    n2_w1, m1, n1 = w1.shape  # K, Q, P
    m1_check, m2, n2_w2 = w2.shape  # Q, S, K

    assert n2_w1 == n2_w2, f"K mismatch: {n2_w1} vs {n2_w2}"
    assert m1 == m1_check, f"Q mismatch: {m1} vs {m1_check}"

    K, Q, P = n2_w1, m1, n1
    S = m2
    N = K * P
    M = Q * S

    # Flatten Batch
    # Reshape x to (TotalBatch, K, P) assuming contiguous layout (handled by reshape/Jit)
    # Note: JAX arrays are immutable. reshape creates a view/copy.
    x_view = x.reshape(-1, K, P)
    Batch = x_view.shape[0]

    # Check dimensions
    # x input could be (..., N). last dim must be N=K*P.
    assert x.shape[-1] == N or (x.shape[-2] == K and x.shape[-1] == P)

    # Prepare Strides assuming standard layout (C-contiguous)
    # If inputs are not contiguous, JAX/Triton might copy?
    # We pass strides based on the SHAPE we assume.
    # If the underlying buffer is transposed, this might fail unless we check.
    # But JAX generally provides standard layout arrays inside jit unless specialized.
    # We will assume row-major.

    stride_xp = 1
    stride_xk = P
    stride_xb = K * P

    stride_w1p = 1
    stride_w1q = P
    stride_w1k = Q * P

    stride_w2k = 1
    stride_w2s = K
    stride_w2q = S * K

    stride_out_s = 1
    stride_out_q = S
    stride_out_b = Q * S

    # Constants
    BLOCK_B = 16
    BLOCK_N1 = triton.next_power_of_2(P)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_S = triton.next_power_of_2(S)
    BLOCK_M1 = 16  # Same as PyTorch impl

    grid = (triton.cdiv(Batch, BLOCK_B), triton.cdiv(Q, BLOCK_M1))

    out_shape = jax.ShapeDtypeStruct((Batch, Q, S), x.dtype)

    out = jt.triton_call(
        x_view,
        w1,
        w2,
        kernel=monarch_kernel,
        out_shape=out_shape,
        grid=grid,
        # Kernel Arguments (Positional pointers handled by jt)
        # Named arguments:
        Batch=Batch,
        N1=P,
        K=K,
        M1=Q,
        S=S,
        # Strides
        stride_xb=stride_xb,
        stride_xk=stride_xk,
        stride_xp=stride_xp,
        stride_w1k=stride_w1k,
        stride_w1q=stride_w1q,
        stride_w1p=stride_w1p,
        stride_w2q=stride_w2q,
        stride_w2s=stride_w2s,
        stride_w2k=stride_w2k,
        stride_out_b=stride_out_b,
        stride_out_q=stride_out_q,
        stride_out_s=stride_out_s,
        # Meta-parameters
        BLOCK_B=BLOCK_B,
        BLOCK_N1=BLOCK_N1,
        BLOCK_K=BLOCK_K,
        BLOCK_M1=BLOCK_M1,
        BLOCK_S=BLOCK_S,
        BLOCK_K_TILE=16,
        BLOCK_P_TILE=32,
    )

    # Reshape Out to (Batch..., M)
    return out.reshape(*batch_shape, M)
