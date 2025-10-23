import torch
import triton
from kernels.matmul.triton.kernel import matmul_fwd_kernel


def matmul_fwd(
    x: torch.Tensor,  # [m, k]
    w: torch.Tensor,  # [n, k]
):
    m, k = x.shape
    n, _ = w.shape

    grid = lambda META: (  # noqa: E731
        triton.cdiv(m, META["bm"]) * triton.cdiv(n, META["bn"]),
    )

    o = torch.empty((m, n), device=x.device, dtype=torch.float16)
    matmul_fwd_kernel[grid](
        x,
        w,
        o,
        m=m,
        n=n,
        k=k,
    )

    return o
