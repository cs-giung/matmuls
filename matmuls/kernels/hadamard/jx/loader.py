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
_lib_path = os.path.join(_cur_dir, "libhadamard_jax.so")


def _get_cuda_arch_flags() -> list[str]:
    """
    Determine the CUDA architecture flags for the current device.
    """
    if not torch.cuda.is_available():
        return []

    major, minor = torch.cuda.get_device_capability()
    print(f"Detecting GPU capability: {major}.{minor}")

    return ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]


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

    print(f"Build command: {' '.join(nvcc_cmd)}")

    try:
        subprocess.check_call(nvcc_cmd)
        print(f"Successfully built {output_lib}")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)


def _load_library():
    """Lazily compile and load the shared library."""
    if not os.path.exists(_lib_path):
        print(f"Building JAX extension at {_lib_path}...")
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


def hadamard_transform(x):
    """
    Apply Fast Hadamard Transform to the input tensor (JAX).

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
