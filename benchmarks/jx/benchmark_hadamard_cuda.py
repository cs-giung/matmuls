import math
import time

import jax
import jax.numpy as jnp
import scipy.linalg

from matmuls.kernels.hadamard import hadamard_transform_cuda

# Constants
TARGET_ELEMENTS = 2**24  # ~16M elements (32MB for fp16)
SIZES = [256, 1024, 4096, 16384]
WARMUP = 10
ITERS = 100


def get_batch_size(n):
    return max(1, TARGET_ELEMENTS // n)


def benchmark_jax(n, batch_size):
    # Setup
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, n), dtype=jnp.float16) * 0.1

    # Reference
    try:
        h_cpu = scipy.linalg.hadamard(n)
        scale = 1.0 / math.sqrt(n)
        H = jnp.array(h_cpu, dtype=jnp.float16) * scale
    except Exception:
        H = None

    @jax.jit
    def ref_op(x, H):
        return jnp.matmul(x, H)

    @jax.jit
    def kernel_op(x):
        return hadamard_transform_cuda(x)

    # Benchmark Reference
    ref_time = float("inf")
    if H is not None:
        # Warmup
        for _ in range(WARMUP):
            ref_op(x, H).block_until_ready()

        # Measure
        start = time.perf_counter()
        for _ in range(ITERS):
            out = ref_op(x, H)
        out.block_until_ready()
        end = time.perf_counter()
        ref_time = (end - start) * 1000 / ITERS

    # Benchmark Kernel
    # Warmup
    for _ in range(WARMUP):
        kernel_op(x).block_until_ready()

    # Measure
    start = time.perf_counter()
    for _ in range(ITERS):
        out = kernel_op(x)
    out.block_until_ready()
    end = time.perf_counter()
    kernel_time = (end - start) * 1000 / ITERS

    return ref_time, kernel_time


def main():
    print(
        f"{'Size':<8} {'Batch':<8} {'Backend':<8} {'Ref (ms)':<12} {'Kernel (ms)':<12} {'Speedup':<8}"
    )
    print("-" * 65)

    for n in SIZES:
        large_batch = get_batch_size(n)
        batch_sizes = [1, 16]
        if large_batch > 16:
            batch_sizes.append(large_batch)

        for batch in batch_sizes:
            # Run JAX
            try:
                j_ref, j_kern = benchmark_jax(n, batch)
                j_speedup = j_ref / j_kern if j_kern > 0 else 0
                print(
                    f"{n:<8} {batch:<8} {'JAX':<8} {j_ref:<12.4f} {j_kern:<12.4f} {j_speedup:<8.2f}x"
                )
            except Exception as e:
                print(f"JAX failed for {n} B={batch}: {e}")
                # import traceback
                # traceback.print_exc()

        print("-" * 65)


if __name__ == "__main__":
    main()
