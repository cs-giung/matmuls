import ctypes
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import torch

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_cur_dir, "libmonarch_jax.so")


def _get_cuda_arch_flags() -> list[str]:
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]


def build_jax_extension():
    output_lib = _lib_path
    core_dir = os.path.join(
        os.path.dirname(os.path.dirname(_cur_dir)), "kernels", "monarch", "core"
    )
    # Logic: _cur_dir = .../matmuls/kernels/monarch/jax. Up 2 levels to kernels/monarch?
    # No, os.path.dirname(_cur_dir) is .../matmuls/kernels/monarch.
    core_dir = os.path.join(os.path.dirname(_cur_dir), "core")

    src_cuda = os.path.join(core_dir, "monarch_cuda.cu")
    src_jax = os.path.join(_cur_dir, "monarch_jax.cu")

    jaxlib_dir = os.path.dirname(jaxlib.__file__)
    include_dirs = [
        os.path.join(jaxlib_dir, "include"),
        "/usr/local/cuda/include",
        core_dir,
        _cur_dir,
    ]

    nvcc_cmd = [
        "nvcc",
        "-O3",
        "--shared",
        "-std=c++17",
        "--compiler-options",
        "-fPIC",
        "-o",
        output_lib,
        src_cuda,
        src_jax,
        "-lcublas",
    ]

    nvcc_cmd.extend(_get_cuda_arch_flags())

    for inc in include_dirs:
        nvcc_cmd.extend(["-I", inc])

    print(f"Build command: {' '.join(nvcc_cmd)}")

    try:
        subprocess.check_call(nvcc_cmd)
        print(f"Successfully built {output_lib}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


def _load_library():
    if not os.path.exists(_lib_path):
        print(f"Building JAX extension at {_lib_path}...")
        build_jax_extension()

    try:
        lib = ctypes.cdll.LoadLibrary(_lib_path)
    except OSError:
        # Rebuild if load fails (e.g. wrong architecture)
        print("Load failed. Rebuilding...")
        build_jax_extension()
        lib = ctypes.cdll.LoadLibrary(_lib_path)

    try:
        jax.ffi.register_ffi_target(
            "monarch_f16", jax.ffi.pycapsule(lib.MonarchF16Symbol), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "monarch_bf16", jax.ffi.pycapsule(lib.MonarchBF16Symbol), platform="CUDA"
        )
    except AttributeError:
        # Rebuild if symbols missing
        print("Symbols missing. Rebuilding...")
        build_jax_extension()
        lib = ctypes.cdll.LoadLibrary(_lib_path)
        jax.ffi.register_ffi_target(
            "monarch_f16", jax.ffi.pycapsule(lib.MonarchF16Symbol), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "monarch_bf16", jax.ffi.pycapsule(lib.MonarchBF16Symbol), platform="CUDA"
        )

    return lib


_lib = _load_library()


def monarch_transform(x, w1, w2):
    """
    Monarch Matrix Multiplication (JAX).

    Args:
        x: (..., n1*n2) input. View as (Batch, n2, n1) internally?
           Reference implementation expects x to be (Batch, N).
           But our kernel assumes Flattened Batch or handles broadcast.
           Kernel params: batch, n1, n2.
           Implies x is treated as (Batch, n2, n1).
           But actually x is likely passed as (Batch, N).
           Reference: n2 blocks of (m1, n1).
           x should be viewed as (..., n2, n1).
    """
    # Shapes
    n2, m1, n1 = w1.shape
    assert w2.shape[0] == m1, f"w2 dim 0 ({w2.shape[0]}) != w1 dim 1 ({m1})"
    m2 = w2.shape[1]
    assert w2.shape[2] == n2, f"w2 dim 2 ({w2.shape[2]}) != w1 dim 0 ({n2})"

    # Flatten x to (Batch, N) or ensure last dim is N
    N = n1 * n2
    assert x.shape[-1] == N

    # We treat all leading dims as batch
    batch_shape = x.shape[:-1]
    batch_dim = 1
    for d in batch_shape:
        batch_dim *= d

    # Out shape
    M = m1 * m2
    out_shape = batch_shape + (M,)

    # Output dtype
    dtype = x.dtype

    # Prepare Output
    out_type = jax.ShapeDtypeStruct(out_shape, x.dtype)

    # Workspace size
    # y1 + y1_perm + y2
    # Size = batch * m1 * (2*n2 + m2)
    workspace_shape = (batch_dim * m1 * (2 * n2 + m2),)
    workspace_type = jax.ShapeDtypeStruct(workspace_shape, x.dtype)

    # Call FFI
    # Note: FFI expects buffers. x_flat might invoke copy if not contiguous.
    # vmap_method="broadcast_all" allows automatic vmap handling if we passed unbatched inputs,
    # but here we manually handle batch.

    if dtype == jnp.float16:
        name = "monarch_f16"
    elif dtype == jnp.bfloat16:
        name = "monarch_bf16"
    else:
        raise TypeError("Unsupported dtype")

    out, _ = jax.ffi.ffi_call(
        name, (out_type, workspace_type), vmap_method="broadcast_all"
    )(
        x,
        w1,
        w2,
        batch=np.int32(batch_dim),
        n1=np.int32(n1),
        n2=np.int32(n2),
        m1=np.int32(m1),
        m2=np.int32(m2),
    )

    return out
