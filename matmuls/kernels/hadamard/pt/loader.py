import math
import os

import torch
from scipy.linalg import hadamard
from torch.utils.cpp_extension import load

from matmuls.kernels.hadamard.core.hadamard_triton import hadamard_fused_kernel

_H_CACHE = {}


def _get_hadamard_matrix(n, dtype, device):
    key = (n, device, dtype)
    if key not in _H_CACHE:
        if not (n & (n - 1) == 0):
            raise ValueError(f"N must be power of 2, got {n}")
        h = torch.tensor(hadamard(n), dtype=dtype, device=device)
        h = h / math.sqrt(n)
        _H_CACHE[key] = h
    return _H_CACHE[key]


_cur_dir = os.path.dirname(os.path.abspath(__file__))
_hadamard = None


def _get_cuda_arch_flags() -> list[str]:
    """
    Determine the CUDA architecture flags for the current device(s).
    """
    if not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    caps = set()
    for i in range(count):
        caps.add(torch.cuda.get_device_capability(i))

    flags = []
    for major, minor in sorted(caps):
        flags.extend(
            ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]
        )

    return flags


def _load_hadamard_extension():
    global _hadamard
    if _hadamard is not None:
        return _hadamard

    cuda_flags = [
        "-O3",
        "-lineinfo",
        "--ptxas-options=--warn-on-local-memory-usage",
        "--ptxas-options=--warn-on-spills",
    ]
    cuda_flags.extend(_get_cuda_arch_flags())

    # Paths relative to this file (in torch/)
    core_dir = os.path.join(os.path.dirname(_cur_dir), "core")

    _hadamard = load(
        name="faster_hadamard_transform",
        sources=[
            os.path.join(_cur_dir, "hadamard_transform.cpp"),
            os.path.join(core_dir, "hadamard_transform_cuda.cu"),
        ],
        extra_cflags=["-O3", f"-I{core_dir}"],
        extra_cuda_cflags=cuda_flags + [f"-I{core_dir}"],
        verbose=True,
    )
    return _hadamard


def hadamard_transform_cuda(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Apply Fast Hadamard Transform to the input tensor (CUDA).

    Args:
        x (torch.Tensor): Input tensor. Must be on CUDA and have dtype float16 or bfloat16.
                          Last dimension must be a power of 2 and <= 2^15.
        inplace (bool): If True, modifies the input tensor in-place.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    module = _load_hadamard_extension()
    return module.hadamard_transform(x, inplace)


def hadamard_transform_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Fast Hadamard Transform to the input tensor (Triton).

    Args:
        x (torch.Tensor): Input tensor. Must be on CUDA.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    B, N = x.shape
    device = x.device
    dtype = x.dtype

    log_n = int(math.log2(N))
    n1_bits = log_n // 2
    n2_bits = log_n - n1_bits

    N1 = 1 << n1_bits
    N2 = 1 << n2_bits

    x_reshaped = x.view(B, N1, N2)

    h1 = _get_hadamard_matrix(N1, dtype, device)
    h2 = _get_hadamard_matrix(N2, dtype, device)

    out = torch.empty_like(x_reshaped)

    # Grid: One block per batch item
    grid = (B,)

    hadamard_fused_kernel[grid](
        x_reshaped,
        h1,
        h2,
        B,
        x_reshaped.stride(0),
        x_reshaped.stride(1),
        x_reshaped.stride(2),
        h1.stride(0),
        h1.stride(1),
        h2.stride(0),
        h2.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_N1=N1,
        BLOCK_SIZE_N2=N2,
        num_warps=4,  # 64x64 typically needs 4-8 warps
    )

    return out.view(B, N)
