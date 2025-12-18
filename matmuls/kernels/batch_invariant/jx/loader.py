import functools
import subprocess

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from matmuls.kernels.batch_invariant.core.matmul_triton import matmul_kernel_persistent


@functools.lru_cache()
def _get_num_sms():
    # Attempt to deduce SM count from GPU name via nvidia-smi
    # Mapping for common GPUs in ML
    # This avoids torch dependency
    SM_COUNTS = {
        "H100": 132,
        "A100": 108,
        "A6000": 84,
        "RTX 4090": 128,
        "RTX 3090": 82,
    }

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        gpu_name = output.strip().split("\n")[0]
        # Match against known keys
        for key, count in SM_COUNTS.items():
            if key in gpu_name:
                return count
        # Default fallback if name not recognized but command worked
        return 84
    except Exception:
        # Fallback if command fails (e.g. no nvidia-smi)
        return 84


def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    JAX implementation of Batch Invariant Matmul using Persistent Triton Kernel.
    """
    M, K = a.shape
    N = b.shape[1]

    # Configuration (matching PyTorch loader defaults)
    # Using float32 config for now or generic
    # Note: JAX might want different tuning, but we stick to reference.
    # Assuming float16 for common case.

    dtype = a.dtype

    if dtype == jnp.float16:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 8
    elif dtype == jnp.bfloat16:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        GROUP_SIZE_M = 8
    else:  # float32
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8

    NUM_SMS = _get_num_sms()

    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    # 1D Persistent Grid
    grid_size = min(NUM_SMS, num_pid_m * num_pid_n)
    grid = (grid_size,)

    out_shape = jax.ShapeDtypeStruct((M, N), dtype)

    # Strides
    # JAX Arrays in JIT are usually contiguous, but we should pass strides if possible?
    # jt.triton_call passes arrays.
    # Kernel args: stride_am, stride_ak...
    # We need to calculate them.
    # For Row Major: stride_am = K, stride_ak = 1.
    # But JAX might layout differently?
    # Usually we pass `stride_x` from JAX via `x.strides`?
    # JAX Arrays don't always expose `.strides` inside JIT.
    # We can assume contiguous ROW MAJOR for now.

    stride_am = K
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1

    # JAX Triton call
    out = jt.triton_call(
        a,
        b,
        kernel=matmul_kernel_persistent,
        out_shape=out_shape,
        grid=grid,
        # Named Args for Kernel
        M=M,
        N=N,
        K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        A_LARGE=False,  # Simplifying assumptions
        B_LARGE=False,
        C_LARGE=False,
        num_stages=3,
        num_warps=8,
    )

    return out
