import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"bm": 128, "bn": 256, "bk": 64, "gm": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"bm": 64, "bn": 256, "bk": 32, "gm": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"bm": 128, "bn": 128, "bk": 32, "gm": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"bm": 128, "bn": 64, "bk": 32, "gm": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"bm": 64, "bn": 128, "bk": 32, "gm": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"bm": 128, "bn": 32, "bk": 32, "gm": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"bm": 64, "bn": 32, "bk": 32, "gm": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"bm": 32, "bn": 64, "bk": 32, "gm": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def quantized_matmul_int248_fwd_kernel(
    x_ptr,  # [m, k]
    q_ptr,  # [n, k // features_per_int]
    s_ptr,  # [n, k // q_groupsize]
    z_ptr,  # [n, k // q_groupsize]
    o_ptr,  # [m, n]
    q_bits: int,
    q_groupsize: int,
    m: int,
    n: int,
    k: int,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    gm: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, bm)
    num_pid_n = tl.cdiv(n, bn)

    num_pid_in_group = gm * num_pid_n
    gid = pid // num_pid_in_group
    gm_min = min(num_pid_m - gid * gm, gm)
    pid_m = gid * gm + ((pid % num_pid_in_group) % gm_min)
    pid_n = (pid % num_pid_in_group) // gm_min

    m_span = pid_m * bm + tl.arange(0, bm)
    n_span = pid_n * bn + tl.arange(0, bn)
    m_mask = m_span < m
    n_mask = n_span < n

    features_per_int = 32 // q_bits
    q_shift = (tl.arange(0, bk) % features_per_int) * q_bits

    o = tl.zeros((bm, bn), dtype=tl.float32)
    for k_idx in range(0, tl.cdiv(k, bk)):
        k_span = k_idx * bk + tl.arange(0, bk)
        k_mask = k_span < k

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_ptrs = x_ptr + k * (m_span % m)[:, None] + 1 * k_span[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        q_mask = n_mask[:, None] & k_mask[None, :]
        q_ptrs = (
            q_ptr
            + (k // features_per_int) * (n_span % n)[:, None]
            + 1 * (k_span // features_per_int)[None, :]
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        q = (q >> q_shift[None, :]) & ((1 << q_bits) - 1)

        s_ptrs = (
            s_ptr
            + (k // q_groupsize) * (n_span % n)[:, None]
            + 1 * (k_span // q_groupsize)[None, :]
        )
        z_ptrs = (
            z_ptr
            + (k // q_groupsize) * (n_span % n)[:, None]
            + 1 * (k_span // q_groupsize)[None, :]
        )
        s = tl.load(s_ptrs, mask=q_mask, other=0.0)
        z = tl.load(z_ptrs, mask=q_mask, other=0.0)
        w = tl.where(k_mask[None, :], q * s + z, 0.0)

        o = tl.dot(x, w.T, o)

    o_ptrs = o_ptr + n * m_span[:, None] + 1 * n_span[None, :]
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(o_ptrs, o.to(tl.float16), mask=o_mask)
