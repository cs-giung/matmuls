import ctypes
import math
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax_triton as jt
import jaxlib
import numpy as np
from jax._src.lib import cuda_versions
from scipy.linalg import hadamard

from matmuls.kernels.hadamard.core.hadamard_triton import hadamard_fused_kernel

_H_CACHE = {}


def _get_hadamard_matrix(n, dtype):
    key = (n, dtype)
    if key not in _H_CACHE:
        if not (n & (n - 1) == 0):
            raise ValueError(f"N must be power of 2, got {n}")
        h = jnp.array(hadamard(n), dtype=dtype)
        h = h / math.sqrt(n)
        _H_CACHE[key] = h
    return _H_CACHE[key]


_cur_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_cur_dir, "libhadamard_jax.so")


def _get_cuda_arch_flags() -> list[str]:
    """
    Determine the CUDA architecture flags for the current device(s).
    """
    try:
        count = jax.local_device_count(backend="gpu")
        caps = set()
        for i in range(count):
            cap = cuda_versions.cuda_compute_capability(i)
            caps.add(cap)

        flags = []
        for cap in sorted(caps):
            flags.extend(["-gencode", f"arch=compute_{cap},code=sm_{cap}"])

        return flags
    except Exception as e:
        print(f"Warning: Failed to detect CUDA architectures via JAX: {e}")
        return []


def build_jax_extension():
    """Compiles the JAX extension shared library using NVCC."""
    output_lib = _lib_path
    core_dir = os.path.join(os.path.dirname(_cur_dir), "core")

    # Source files
    src_cuda = os.path.join(core_dir, "hadamard_transform_cuda.cu")
    src_jax = os.path.join(_cur_dir, "hadamard_transform_jax.cu")

    # Include paths
    jaxlib_dir = os.path.dirname(jaxlib.__file__)
    include_dirs = [
        os.path.join(jaxlib_dir, "include"),
        "/usr/local/cuda/include",
        core_dir,
        _cur_dir,
    ]

    # Compiler flags
    # Use C++17 for JAX FFI
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
    ]

    # Add architecture flags
    nvcc_cmd.extend(_get_cuda_arch_flags())

    for inc in include_dirs:
        nvcc_cmd.extend(["-I", inc])

    try:
        subprocess.check_call(nvcc_cmd)

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


def _load_library():
    """Lazily compile and load the shared library."""
    if not os.path.exists(_lib_path):
        build_jax_extension()

    if os.path.exists(_lib_path):
        lib = ctypes.cdll.LoadLibrary(_lib_path)

        # Register FFI targets
        try:
            jax.ffi.register_ffi_target(
                "hadamard_f16",
                jax.ffi.pycapsule(lib.HadamardF16Symbol),
                platform="CUDA",
            )
            jax.ffi.register_ffi_target(
                "hadamard_bf16",
                jax.ffi.pycapsule(lib.HadamardBF16Symbol),
                platform="CUDA",
            )
        except AttributeError:
            # If symbols aren't found, it might be an old build or mismatch.
            print(f"WARNING: Could not find symbols in {_lib_path}. Rebuilding...")
            build_jax_extension()
            # Retry loading
            lib = ctypes.cdll.LoadLibrary(_lib_path)
            jax.ffi.register_ffi_target(
                "hadamard_f16",
                jax.ffi.pycapsule(lib.HadamardF16Symbol),
                platform="CUDA",
            )
            jax.ffi.register_ffi_target(
                "hadamard_bf16",
                jax.ffi.pycapsule(lib.HadamardBF16Symbol),
                platform="CUDA",
            )

        return lib
    else:
        raise RuntimeError(f"Failed to build or find shared library at {_lib_path}")


# Load library on import (or first use logic could be here, but top-level load is simpler for registration)
_lib = _load_library()


def hadamard_transform_cuda(x):
    """
    Apply Fast Hadamard Transform to the input tensor (JAX) using CUDA implementation.

    Args:
        x (jax.Array): Input array. Must be on GPU and have dtype float16 or bfloat16.
                       Last dimension must be a power of 2 and <= 2^15.

    Returns:
        jax.Array: Transformed array.
    """
    had_size = x.shape[-1]
    original_shape = x.shape

    # Check constraints
    assert x.dtype in [jnp.float16, jnp.bfloat16], (
        "Only float16 and bfloat16 are supported"
    )

    # Reshape to (-1, had_size) to treat as a batch of vectors
    x_flat = x.reshape(-1, had_size)
    numel = x_flat.size

    # Calculate padding for the batch dimension
    # We need total numel to be multiple of 256.
    remainder = numel % 256
    pad_rows = 0
    if remainder != 0:
        pad_elements = 256 - remainder
        pad_rows = pad_elements // had_size

        # Pad along the first dimension (batch dim) of the 2D view
        x_flat = jnp.pad(x_flat, ((0, pad_rows), (0, 0)))

    # Prepare FFI call
    out_type = jax.ShapeDtypeStruct(x_flat.shape, x.dtype)

    if x.dtype == jnp.float16:
        out = jax.ffi.ffi_call("hadamard_f16", out_type, vmap_method="broadcast_all")(
            x_flat, had_size=np.int32(had_size)
        )
    else:
        out = jax.ffi.ffi_call("hadamard_bf16", out_type, vmap_method="broadcast_all")(
            x_flat, had_size=np.int32(had_size)
        )

    # Unpad
    if pad_rows > 0:
        out = out[:-pad_rows]

    # Reshape back
    return out.reshape(original_shape)


def hadamard_transform_triton(x):
    """
    Apply Fast Hadamard Transform to the input tensor (JAX) using Triton implementation.
    """
    had_size = x.shape[-1]
    original_shape = x.shape
    dtype = x.dtype

    B_total = int(np.prod(x.shape[:-1]))
    N = had_size

    log_n = int(math.log2(N))
    n1_bits = log_n // 2
    n2_bits = log_n - n1_bits

    N1 = 1 << n1_bits
    N2 = 1 << n2_bits

    # Prepare inputs
    x_reshaped = x.reshape(B_total, N1, N2)

    # Get Hadamard matrices
    h1 = _get_hadamard_matrix(N1, dtype)
    h2 = _get_hadamard_matrix(N2, dtype)

    # Compute strides
    # Assume contiguous layout for reshaped
    stride_xb = N1 * N2
    stride_xh = N2
    stride_xw = 1

    stride_h1_r = h1.shape[1]  # row major
    stride_h1_c = 1
    stride_h2_r = h2.shape[1]
    stride_h2_c = 1

    stride_out_b = N1 * N2
    stride_out_h = N2
    stride_out_w = 1

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_N1 = N1
    BLOCK_SIZE_N2 = N2

    out_shape = jax.ShapeDtypeStruct((B_total, N1, N2), dtype)

    grid = (B_total,)

    out = jt.triton_call(
        x_reshaped,
        h1,
        h2,
        B_total,
        stride_xb,
        stride_xh,
        stride_xw,
        stride_h1_r,
        stride_h1_c,
        stride_h2_r,
        stride_h2_c,
        stride_out_b,
        stride_out_h,
        stride_out_w,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_N1=BLOCK_SIZE_N1,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        kernel=hadamard_fused_kernel,
        out_shape=out_shape,
        grid=grid,
        num_warps=4,
    )

    return out.reshape(original_shape)
