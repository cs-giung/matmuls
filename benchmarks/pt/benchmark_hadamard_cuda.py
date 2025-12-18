import math

import scipy.linalg
import torch

from matmuls.kernels.hadamard import hadamard_transform_cuda

# Constants
TARGET_ELEMENTS = 2**24  # ~16M elements (32MB for fp16)
SIZES = [256, 1024, 4096, 16384]
WARMUP = 10
ITERS = 100


def get_batch_size(n):
    return max(1, TARGET_ELEMENTS // n)


def benchmark_torch(n, batch_size):
    # Setup
    torch.manual_seed(0)
    x = torch.randn((batch_size, n), device="cuda", dtype=torch.float16) * 0.1

    # Reference
    with torch.no_grad():
        try:
            h_cpu = scipy.linalg.hadamard(n)
            scale = 1.0 / math.sqrt(n)
            H = torch.tensor(h_cpu, device="cuda", dtype=torch.float16) * scale
        except Exception:
            H = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Benchmark Reference
        ref_time = float("inf")
        if H is not None:
            # Warmup
            for _ in range(WARMUP):
                _ = torch.matmul(x, H)
            torch.cuda.synchronize()

            # Measure
            start_event.record()
            for _ in range(ITERS):
                _ = torch.matmul(x, H)
            end_event.record()
            torch.cuda.synchronize()
            ref_time = start_event.elapsed_time(end_event) / ITERS

        # Benchmark Kernel
        # Warmup
        for _ in range(WARMUP):
            hadamard_transform_cuda(x)
        torch.cuda.synchronize()

        # Measure
        start_event.record()
        for _ in range(ITERS):
            hadamard_transform_cuda(x)
        end_event.record()
        torch.cuda.synchronize()
        kernel_time = start_event.elapsed_time(end_event) / ITERS

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
            # Run Torch
            try:
                t_ref, t_kern = benchmark_torch(n, batch)
                t_speedup = t_ref / t_kern if t_kern > 0 else 0
                print(
                    f"{n:<8} {batch:<8} {'Torch':<8} {t_ref:<12.4f} {t_kern:<12.4f} {t_speedup:<8.2f}x"
                )
            except Exception as e:
                print(f"Torch failed for {n} B={batch}: {e}")
                # import traceback
                # traceback.print_exc()

        print("-" * 65)


if __name__ == "__main__":
    main()
