import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _next_multiple(x: int, n: int) -> int:
    r = x % n
    return x if r == 0 else x + (n - r)


def matmul_fwd_kernel(
    x_ref, # [bm, pk]
    w_ref, # [bn, pk]
    o_ref, # [bm, bn]
    *,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: jax.typing.DTypeLike,
):
    m_idx, n_idx = pl.program_id(0), pl.program_id(1)

    m_mask = m_idx * bm + jnp.arange(bm) < m
    n_mask = n_idx * bn + jnp.arange(bn) < n
    m_span = pl.dslice(m_idx * bm, bm)
    n_span = pl.dslice(n_idx * bn, bn)

    def body(k_idx, carry):
        o_prev = carry
        k_mask = k_idx * bk + jnp.arange(bk) < k
        k_span = pl.dslice(k_idx * bk, bk)
        x = plgpu.load(
            x_ref.at[m_span, k_span], mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        w = plgpu.load(
            w_ref.at[n_span, k_span], mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)
        return jax.lax.dot(
            x, w,
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=precision,
            preferred_element_type=preferred_element_type,
        ) + o_prev

    o = jnp.zeros((bm, bn), dtype=preferred_element_type)
    o = jax.lax.fori_loop(0, pl.cdiv(k, bk), body, o)
    plgpu.store(
        o_ref.at[m_span, n_span], o.astype(o_ref.dtype), mask=(m_mask[:, None] & n_mask[None, :]))


def matmul_fwd(
    x: jax.Array, # [m, k]
    w: jax.Array, # [n, k]
    *,
    bm: int = 32,
    bn: int = 32,
    bk: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike = jnp.float32,
):
    m, k = x.shape
    n, _ = w.shape

    pm = _next_multiple(m, bm)
    pn = _next_multiple(n, bn)
    pk = _next_multiple(k, bk)

    x = jnp.pad(x, ((0, pm - m), (0, pk - k)))
    w = jnp.pad(w, ((0, pn - n), (0, pk - k)))

    grid = (pm // bm, pn // bn)
    out_shape = [
        jax.ShapeDtypeStruct((pm, pn), x.dtype),
    ]
    in_specs = [
        pl.BlockSpec((bm, pk), lambda i, j: (i, 0)),
        pl.BlockSpec((bn, pk), lambda i, j: (j, 0)),
    ]
    out_specs = [
        pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    ]
    compiler_params = plgpu.CompilerParams(num_warps, num_stages)

    kernel = functools.partial(
        matmul_fwd_kernel,
        m=m, n=n, k=k, bm=bm, bn=bn, bk=bk,
        precision=precision, preferred_element_type=preferred_element_type)
    kernel_name = f"matmul_fwd_{bm}_{bk}_{bn}"
    kernel_call = pl.pallas_call(
        kernel, out_shape,
        grid=grid, in_specs=in_specs, out_specs=out_specs, compiler_params=compiler_params)

    with jax.named_scope(kernel_name):
        o, = kernel_call(x, w)

    return o[:m, :n]
