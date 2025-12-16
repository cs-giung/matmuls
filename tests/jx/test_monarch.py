import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from matmuls.kernels.monarch import monarch_transform


def monarch_matmul_rhs(x, w_bfly1, w_bfly2, transpose_w=False):
    """
    Performs a Monarch matrix multiplication for a rectangular matrix W(M, N).

    N = n1 * n2 (input dim)
    M = m1 * m2 (output dim)

    w_bfly1: (n2, m1, n1)
    w_bfly2: (m1, m2, n2)

    If transpose_w=False:
      - Computes x @ W.T
      - W.T = W(1).T @ P.T @ W(2).T
      - Input x shape: (..., N)
      - Output shape: (..., M)

    If transpose_w=True:
      - Computes x @ W
      - W = W(2) @ P @ W(1)
      - Input x shape: (..., M)
      - Output shape: (..., N)
    """
    n2, m1, n1 = w_bfly1.shape
    m1, m2, n2 = w_bfly2.shape

    if not transpose_w:
        return rearrange(
            jnp.einsum(
                "lsr,...lr->...ls",  # l=m1, s=m2, r=n2
                w_bfly2,
                rearrange(
                    jnp.einsum(
                        "kqp,...kp->...kq",  # k=n2, q=m1, p=n1
                        w_bfly1,
                        rearrange(x, "... (n2 n1) -> ... n2 n1", n1=n1, n2=n2),
                    ),
                    "... n2 m1 -> ... m1 n2",
                    m1=m1,
                    n2=n2,
                ),
            ),
            "... m1 m2 -> ... (m1 m2)",
        )

    else:
        return rearrange(
            jnp.einsum(
                "kqp, ...kq -> ...kp",  # k=n2, q=m1, p=n1
                w_bfly1,
                rearrange(
                    jnp.einsum(
                        "lsr, ...ls -> ...lr",  # l=m1, s=m2, r=n2
                        w_bfly2,
                        rearrange(x, "... (m1 m2) -> ... m1 m2", m1=m1, m2=m2),
                    ),
                    "... m1 n2 -> ... n2 m1",
                    m1=m1,
                    n2=n2,
                ),
            ),
            "... n2 n1 -> ... (n2 n1)",
        )


@pytest.mark.parametrize("batch", [16])  # Keep small for JAX test speed
@pytest.mark.parametrize("n1", [16, 64])
@pytest.mark.parametrize("n2", [16, 64])
@pytest.mark.parametrize("m1", [16, 64])
@pytest.mark.parametrize("m2", [16, 64])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_monarch_transform_jax(batch, n1, n2, m1, m2, dtype):
    # Setup JAX
    key = jax.random.PRNGKey(0)

    N = n1 * n2
    x_jax = jax.random.normal(key, (batch, N), dtype=dtype)
    w1_jax = jax.random.normal(key, (n2, m1, n1), dtype=dtype)
    w2_jax = jax.random.normal(key, (m1, m2, n2), dtype=dtype)

    x_dev = jax.device_put(x_jax)
    w1_dev = jax.device_put(w1_jax)
    w2_dev = jax.device_put(w2_jax)

    # Kernel
    out_jax = monarch_transform(x_dev, w1_dev, w2_dev)

    # Reference
    ref_out_jax = monarch_matmul_rhs(x_jax, w1_jax, w2_jax, transpose_w=False)

    # Verify
    # JAX out might be BF16. Cast to Float32 for comparison
    out_np = np.array(out_jax, dtype=np.float32)
    ref_out_np = np.array(ref_out_jax, dtype=np.float32)

    rtol = 1e-2 if dtype == jnp.float16 else 5e-2
    atol = 1e-2 if dtype == jnp.float16 else 1e-1

    np.testing.assert_allclose(out_np, ref_out_np, rtol=rtol, atol=atol)
