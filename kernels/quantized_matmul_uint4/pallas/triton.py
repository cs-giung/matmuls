import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _next_multiple(x: int, n: int) -> int:
    r = x % n
    return x if r == 0 else x + (n - r)


def quantized_matmul_uint4_fwd_kernel(
    x_q_ref: jax.Array, # [bm, pk]
    x_s_ref: jax.Array, # [bm,]
    w_q_ref: jax.Array, # [bn, pk]
    w_s_ref: jax.Array, # [bn,]
    o_ref: jax.Array, # [bm, bn]
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

    def body(k_idx, carry):
        o_prev = carry
        k_mask = k_idx * bk + jnp.arange(bk) < k
        k_span = pl.dslice(k_idx * bk, bk)
        x_q = plgpu.load(
            x_q_ref.at[:, k_span], mask=(m_mask[:, None] & k_mask[None, :]), other=0)
        w_q = plgpu.load(
            w_q_ref.at[:, k_span], mask=(n_mask[:, None] & k_mask[None, :]), other=0)
        return jax.lax.dot(
            # NOTE: upcast narrow-width integers
            x_q.astype(jnp.uint8), w_q.astype(jnp.uint8),
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=precision,
            preferred_element_type=preferred_element_type,
        ) + o_prev

    o = jnp.zeros((bm, bn), dtype=preferred_element_type)
    o = jax.lax.fori_loop(0, pl.cdiv(k, bk), body, o)

    x_s = plgpu.load(x_s_ref.at[:], mask=m_mask, other=0.0)
    w_s = plgpu.load(w_s_ref.at[:], mask=n_mask, other=0.0)
    o = x_s[:, None] * w_s[None, :] * o.astype(x_s.dtype)

    plgpu.store(o_ref.at[:, :], o.astype(o_ref.dtype), mask=(m_mask[:, None] & n_mask[None, :]))
    

def quantized_matmul_uint4_fwd(
    x_q: jax.Array, # [m, k]
    x_s: jax.Array, # [m,]
    w_q: jax.Array, # [n, k]
    w_s: jax.Array, # [n,]
    *,
    bm: int = 32,
    bk: int = 32,
    bn: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike = jnp.int32,
):
    m, k = x_q.shape
    n, _ = w_q.shape

    pm = _next_multiple(m, bm)
    pn = _next_multiple(n, bn)
    pk = _next_multiple(k, bk)

    x_q = jnp.pad(x_q, ((0, pm - m), (0, pk - k)))
    x_s = jnp.pad(x_s, (0, pm - m))
    w_q = jnp.pad(w_q, ((0, pn - n), (0, pk - k)))
    w_s = jnp.pad(w_s, (0, pn - n))

    grid = (pm // bm, pn // bn)
    out_shape = [
        jax.ShapeDtypeStruct((pm, pn), x_s.dtype),
    ]
    in_specs = [
        pl.BlockSpec((bm, pk), lambda i, j: (i, 0)),
        pl.BlockSpec((bm,), lambda i, j: (i,)),
        pl.BlockSpec((bn, pk), lambda i, j: (j, 0)),
        pl.BlockSpec((bn,), lambda i, j: (j,)),
    ]
    out_specs = [
        pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    ]
    compiler_params = plgpu.CompilerParams(num_warps, num_stages)

    kernel = functools.partial(
        quantized_matmul_uint4_fwd_kernel,
        m=m, n=n, k=k, bm=bm, bn=bn, bk=bk,
        precision=precision, preferred_element_type=preferred_element_type)
    kernel_name = f"quantized_matmul_uint4_fwd_{bm}_{bk}_{bn}"
    kernel_call = pl.pallas_call(
        kernel, out_shape,
        grid=grid, in_specs=in_specs, out_specs=out_specs, compiler_params=compiler_params)

    with jax.named_scope(kernel_name):
        out, = kernel_call(x_q, x_s, w_q, w_s)

    return out[:m, :n]
