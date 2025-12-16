import os

import torch
from torch.utils.cpp_extension import load

_cur_dir = os.path.dirname(os.path.abspath(__file__))

_hadamard = None


def _get_cuda_arch_flags() -> list[str]:
    """
    Determine the CUDA architecture flags for the current device.
    """
    if not torch.cuda.is_available():
        return []

    major, minor = torch.cuda.get_device_capability()
    msg = f"Detecting GPU capability: {major}.{minor}"
    print(msg)  # Print to stdout so user sees it during compilation

    return ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]


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


def hadamard_transform(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Apply Fast Hadamard Transform to the input tensor.

    Args:
        x (torch.Tensor): Input tensor. Must be on CUDA and have dtype float16 or bfloat16.
                          Last dimension must be a power of 2 and <= 2^15.
        inplace (bool): If True, modifies the input tensor in-place.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    module = _load_hadamard_extension()
    return module.hadamard_transform(x, inplace)
