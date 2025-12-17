import os
import sys
import time
import torch
from einops import rearrange
from matmuls.kernels.monarch import monarch_transform

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def monarch_matmul_ref(x, w_bfly1, w_bfly2):
    n2, m1, n1 = w_bfly1.shape
    m1, m2, n2 = w_bfly2.shape

    # x: (..., N) where N = n1 * n2
    # reshape x to (..., n2, n1)

    return rearrange(
        torch.einsum(
            "lsr,...lr->...ls",  # l=m1, s=m2, r=n2
            w_bfly2,
            rearrange(
                torch.einsum(
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

TARGET_ELEMENTS = 2**24  # Approx load control
WARMUP = 10
ITERS = 50


def get_batch_size(n_elements):
    # n_elements = n1 * n2
    return max(1, TARGET_ELEMENTS // n_elements)


def benchmark_torch(config, batch_size):
    n1, n2, m1, m2 = config
    N = n1 * n2

    if not torch.cuda.is_available():
        return float("nan"), float("nan")

    device = "cuda"
    dtype = torch.float16

    # Setup
    torch.manual_seed(0)
    x = torch.randn(batch_size, N, device=device, dtype=dtype)
    w1 = torch.randn(n2, m1, n1, device=device, dtype=dtype)
    w2 = torch.randn(m1, m2, n2, device=device, dtype=dtype)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark Reference
    # Warmup
    for _ in range(WARMUP):
        _ = monarch_matmul_ref(x, w1, w2)
    torch.cuda.synchronize()

    # Measure
    start_event.record()
    for _ in range(ITERS):
        _ = monarch_matmul_ref(x, w1, w2)
    end_event.record()
    torch.cuda.synchronize()
    ref_time = start_event.elapsed_time(end_event) / ITERS

    # Benchmark Kernel
    # Warmup
    for _ in range(WARMUP):
        _ = monarch_transform(x, w1, w2)
    torch.cuda.synchronize()

    # Measure
    start_event.record()
    for _ in range(ITERS):
        _ = monarch_transform(x, w1, w2)
    end_event.record()
    torch.cuda.synchronize()
    kernel_time = start_event.elapsed_time(end_event) / ITERS

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
                t_ref, t_kern = benchmark_torch(config, batch)
                t_speedup = t_ref / t_kern if t_kern > 0 else 0
                cfg_str = f"{n1}x{n2}x{m1}x{m2}"
                print(
                    f"{cfg_str:<20} {batch:<8} {'Torch':<8} {t_ref:<12.4f} {t_kern:<12.4f} {t_speedup:<8.2f}x"
                )
            except Exception as e:
                print(f"Torch failed for {config}, Batch={batch}: {e}")
                # import traceback
                # traceback.print_exc()

        print("-" * 75)


if __name__ == "__main__":
    main()
