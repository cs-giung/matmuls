import os

import torch
from torch.utils.cpp_extension import load

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.join(os.path.dirname(_cur_dir), "core")


def _get_cuda_arch_flags():
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


_arch_flags = _get_cuda_arch_flags()

# Compile linkage flags
# We need to link cublas
cuda_flags = _arch_flags + ["-lcublas"]

_monarch = load(
    name="monarch_kernel_torch",
    sources=[
        os.path.join(_cur_dir, "monarch.cpp"),
        os.path.join(_core_dir, "monarch_cuda.cu"),
    ],
    extra_cflags=["-O3", f"-I{_core_dir}"],
    extra_cuda_cflags=["-O3", f"-I{_core_dir}"] + cuda_flags,
    extra_ldflags=["-lcublas"],
    verbose=True,
)


def monarch_transform(x, w1, w2):
    return _monarch.monarch_transform(x, w1, w2)
