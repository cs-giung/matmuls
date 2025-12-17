import torch
import time
from matmuls.kernels.batch_invariant import matmul

SIZES = [128, 512, 1024, 2048, 4096]
WARMUP = 10
ITERS = 50


def benchmark(m, k, n):
    print(f"Benchmarking M={m}, K={k}, N={n}")

    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    # 1. Reference (torch.mm)
    # Warmup
    for _ in range(WARMUP):
        torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    ref_time = (end - start) * 1000 / ITERS

    # 2. Invariant (custom)
    # Warmup
    for _ in range(WARMUP):
        matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        matmul(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    kernel_time = (end - start) * 1000 / ITERS

    return ref_time, kernel_time


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

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
