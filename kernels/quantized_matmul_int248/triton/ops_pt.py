import torch
import triton
from kernels.quantized_matmul_int248.triton.kernel import (
    quantized_matmul_int248_fwd_kernel,
)


def quantized_matmul_int248_fwd(
    x: torch.Tensor,  # [m, k]
    q: torch.Tensor,  # [n, k // features_per_int]
    s: torch.Tensor,  # [n, k // q_gruopsize]
    z: torch.Tensor,  # [n, k // q_groupsize]
    q_bits: int,
    q_groupsize: int = None,
):
    m, k = x.shape
    n, _ = q.shape

    if q_groupsize is None:
        q_groupsize = k // s.shape[1]

    grid = lambda META: (triton.cdiv(m, META["bm"]) * triton.cdiv(n, META["bn"]),)

    o = torch.empty((m, n), device=x.device, dtype=torch.float16)
    quantized_matmul_int248_fwd_kernel[grid](
        x,
        q,
        s,
        z,
        o,
        q_bits=q_bits,
        q_groupsize=q_groupsize,
        m=m,
        n=n,
        k=k,
    )

    return o
