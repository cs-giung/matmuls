import torch
import pytest
from matmuls.kernels.batch_invariant import matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_correctness(dtype):
    """
    Verify that batch-invariant matmul matches torch.mm results within tolerance.
    """
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    ref = torch.mm(a, b)
    out = matmul(a, b)

    print(f"Dtype: {dtype}")
    print(f"Ref Mean: {ref.mean().item():.4f}, Max: {ref.max().item():.4f}")
    print(f"Out Mean: {out.mean().item():.4f}, Max: {out.max().item():.4f}")

    # Check non-zero
    assert out.abs().max() > 0, "Output should not be all zeros"

    # Check close
    # loose tolerance for float32 TF32 issues
    atol = 1e-2 if dtype == torch.float16 else 1e-1
    rtol = 1e-2 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    print(f"Correctness input {dtype} passed.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invariance():
    """
    Verify that batch-invariant matmul is strictly bitwise identical
    regardless of batch size (batch invariance).
    """
    # Use deterministic inputs similar to user verification
    B, D = 256, 1024  # Smaller than user 2048/4096 for speed in CI

    # Use Linspace/Ranges to simulate structured data where accumulation order matters
    # Large values + small values mixed can trigger order sensitivity
    a = torch.linspace(-100, 100, B * D).reshape(B, D).cuda()
    b = torch.linspace(-100, 100, D * D).reshape(D, D).cuda()

    # 1. Baseline (torch.mm) check
    # Note: torch.mm is often batch-variant on CUDA due to tiling heuristics
    out1_ref = torch.mm(a[:1], b)
    out2_ref = torch.mm(a, b)[:1]
    diff_ref = (out1_ref - out2_ref).abs().max()
    print(f"Baseline (torch.mm) Difference: {diff_ref.item()}")

    # 2. Invariant Kernel check
    out1 = matmul(a[:1], b)
    out2 = matmul(a, b)[:1]
    diff = (out1 - out2).abs().max()
    print(f"Invariant Kernel Difference: {diff.item()}")

    # Strict 0.0 check implies strict bitwise identity
    assert diff == 0.0, f"Kernel failed invariance check. Diff: {diff.item()}"
