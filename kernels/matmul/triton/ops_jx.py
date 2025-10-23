import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from kernels.matmul.triton.kernel import matmul_fwd_kernel


def matmul_fwd(
    x: jax.Array,  # [m, k]
    w: jax.Array,  # [n, k]
):
    m, k = x.shape
    n, _ = w.shape

    grid = lambda META: (  # noqa: E731
        triton.cdiv(m, META["bm"]) * triton.cdiv(n, META["bn"]),
    )

    o = jt.triton_call(
        x,
        w,
        kernel=matmul_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        grid=grid,
        m=m,
        n=n,
        k=k,
    )

    return o
