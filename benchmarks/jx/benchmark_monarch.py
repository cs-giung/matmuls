import time
import jax
import jax.numpy as jnp
from einops import rearrange
from matmuls.kernels.monarch import monarch_transform


def monarch_matmul_ref(x, w_bfly1, w_bfly2):
    n2, m1, n1 = w_bfly1.shape
    m1, m2, n2 = w_bfly2.shape

    return rearrange(
        jnp.einsum(
            "lsr,...lr->...ls",  # l=m1, s=m2, r=n2
            w_bfly2,
            rearrange(
                jnp.einsum(
                    "kqp,...kp->...kq",  # k=n2, q=m1, p=n1
                    w_bfly1,
                    rearrange(x, "... (n2 n1) -> ... n2 n1", n1=n1, n2=n2),
                ),
                "... n2 m1 -> ... m1 n2",
                m1=m1,
                n2=n2,
            ),
        ),
        "... m1 m2 -> ... (m1 m2)",
    )


CONFIGS = [
    # n1, n2, m1, m2
    (16, 16, 16, 16),
    (32, 32, 32, 32),
    (64, 64, 64, 64),
    (128, 86, 128, 86),
    (128, 86, 128, 128),
]

TARGET_ELEMENTS = 2**24
WARMUP = 10
ITERS = 50


def get_batch_size(n_elements):
    return max(1, TARGET_ELEMENTS // n_elements)


def benchmark_jax(config, batch_size):
    n1, n2, m1, m2 = config
    N = n1 * n2

    # print(f"  JAX     | N={N}, Batch={batch_size}, Config={config}")

    key = jax.random.PRNGKey(0)
    dtype = jnp.float16

    x = jax.random.normal(key, (batch_size, N), dtype=dtype)
    w1 = jax.random.normal(key, (n2, m1, n1), dtype=dtype)
    w2 = jax.random.normal(key, (m1, m2, n2), dtype=dtype)

    x = jax.device_put(x)
    w1 = jax.device_put(w1)
    w2 = jax.device_put(w2)

    @jax.jit
    def ref_op(x, w1, w2):
        return monarch_matmul_ref(x, w1, w2)

    @jax.jit
    def kernel_op(x, w1, w2):
        return monarch_transform(x, w1, w2)

    # Benchmark Reference
    # Warmup
    for _ in range(WARMUP):
        ref_op(x, w1, w2).block_until_ready()

    start = time.time()
    for _ in range(ITERS):
        out = ref_op(x, w1, w2)
    out.block_until_ready()
    end = time.time()
    ref_time = (end - start) * 1000 / ITERS

    # Benchmark Kernel
    # Warmup
    for _ in range(WARMUP):
        kernel_op(x, w1, w2).block_until_ready()

    start = time.time()
    for _ in range(ITERS):
        out = kernel_op(x, w1, w2)
    out.block_until_ready()
    end = time.time()
    kernel_time = (end - start) * 1000 / ITERS

    return ref_time, kernel_time


def main():
    print(
        f"{'Config':<20} {'Batch':<8} {'Backend':<8} {'Ref (ms)':<12} {'Kernel (ms)':<12} {'Speedup':<8}"
    )
    print("-" * 75)

    for config in CONFIGS:
        n1, n2, m1, m2 = config
        N = n1 * n2

        calculated_batch = get_batch_size(N)
        batches_to_test = sorted(list(set([1, 16, calculated_batch])))

        for batch in batches_to_test:
            try:
                t_ref, t_kern = benchmark_jax(config, batch)
                t_speedup = t_ref / t_kern if t_kern > 0 else 0
                cfg_str = f"{n1}x{n2}x{m1}x{m2}"
                print(
                    f"{cfg_str:<20} {batch:<8} {'JAX':<8} {t_ref:<12.4f} {t_kern:<12.4f} {t_speedup:<8.2f}x"
                )
            except Exception as e:
                print(f"JAX failed for {config}, Batch={batch}: {e}")
                # import traceback
                # traceback.print_exc()

        print("-" * 75)


if __name__ == "__main__":
    main()
