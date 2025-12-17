import jax
import jax.numpy as jnp
import time
from matmuls.kernels.batch_invariant import matmul

SIZES = [128, 512, 1024, 2048, 4096]
WARMUP = 10
ITERS = 50


def benchmark(m, k, n):
    print(f"Benchmarking M={m}, K={k}, N={n}")

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (m, k), dtype=jnp.float16)
    b = jax.random.normal(k2, (k, n), dtype=jnp.float16)

    # Pre-compile
    jnp_matmul = jax.jit(jnp.matmul)
    custom_matmul = jax.jit(matmul)

    # 1. Reference (jnp.matmul)
    # Warmup
    _ = jnp_matmul(a, b).block_until_ready()
    for _ in range(WARMUP):
        _ = jnp_matmul(a, b).block_until_ready()

    start = time.perf_counter()
    for _ in range(ITERS):
        _ = jnp_matmul(a, b).block_until_ready()
    end = time.perf_counter()
    ref_time = (end - start) * 1000 / ITERS

    # 2. Invariant (custom)
    # Warmup
    _ = custom_matmul(a, b).block_until_ready()
    for _ in range(WARMUP):
        _ = custom_matmul(a, b).block_until_ready()

    start = time.perf_counter()
    for _ in range(ITERS):
        _ = custom_matmul(a, b).block_until_ready()
    end = time.perf_counter()
    kernel_time = (end - start) * 1000 / ITERS

    return ref_time, kernel_time


def main():
    print(
        f"{'Size (MxKxN)':<20} {'Ref (ms)':<10} {'Invariant (ms)':<15} {'Speedup':<10}"
    )
    print("-" * 60)

    for s in SIZES:
        m = k = n = s
        try:
            t_ref, t_kern = benchmark(m, k, n)
            t_speedup = t_ref / t_kern if t_kern > 0 else 0
            print(
                f"{f'{s}x{s}x{s}':<20} {t_ref:<10.3f} {t_kern:<15.3f} {t_speedup:<10.2f}x"
            )
        except Exception as e:
            print(f"{s}x{s}x{s}: Failed - {e}")


if __name__ == "__main__":
    main()
