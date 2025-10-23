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
def matmul_fwd_kernel(
    x_ptr,  # [m, k]
    w_ptr,  # [n, k]
    o_ptr,  # [m, n]
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

    o = tl.zeros((bm, bn), dtype=tl.float32)
    for k_idx in range(0, tl.cdiv(k, bk)):
        k_span = k_idx * bk + tl.arange(0, bk)
        k_mask = k_span < k

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_ptrs = x_ptr + k * (m_span % m)[:, None] + 1 * k_span[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_mask = n_mask[:, None] & k_mask[None, :]
        w_ptrs = w_ptr + k * (n_span % n)[:, None] + 1 * k_span[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        o = tl.dot(x, w.T, o)

    o_ptrs = o_ptr + n * m_span[:, None] + 1 * n_span[None, :]
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(o_ptrs, o.to(tl.float16), mask=o_mask)
