import pytest
import torch
from einops import rearrange

from matmuls.kernels.monarch import monarch_transform


def monarch_matmul_rhs(x, w_bfly1, w_bfly2, transpose_w=False):
    """
    Performs a Monarch matrix multiplication for a rectangular matrix W(M, N).

    N = n1 * n2 (input dim)
    M = m1 * m2 (output dim)

    w_bfly1: (n2, m1, n1)
    w_bfly2: (m1, m2, n2)

    If transpose_w=False:
      - Computes x @ W.T
      - W.T = W(1).T @ P.T @ W(2).T
      - Input x shape: (..., N)
      - Output shape: (..., M)

    If transpose_w=True:
      - Computes x @ W
      - W = W(2) @ P @ W(1)
      - Input x shape: (..., M)
      - Output shape: (..., N)
    """
    n2, m1, n1 = w_bfly1.shape
    m1, m2, n2 = w_bfly2.shape

    if not transpose_w:
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

    else:
        return rearrange(
            torch.einsum(
                "kqp, ...kq -> ...kp",  # k=n2, q=m1, p=n1
                w_bfly1,
                rearrange(
                    torch.einsum(
                        "lsr, ...ls -> ...lr",  # l=m1, s=m2, r=n2
                        w_bfly2,
                        rearrange(x, "... (m1 m2) -> ... m1 m2", m1=m1, m2=m2),
                    ),
                    "... m1 n2 -> ... n2 m1",
                    m1=m1,
                    n2=n2,
                ),
            ),
            "... n2 n1 -> ... (n2 n1)",
        )


@pytest.mark.parametrize("batch", [16, 32])
@pytest.mark.parametrize("n1", [16, 64])
@pytest.mark.parametrize("n2", [16, 64])
@pytest.mark.parametrize("m1", [16, 64])
@pytest.mark.parametrize("m2", [16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_monarch_transform(batch, n1, n2, m1, m2, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    device = "cuda"

    # Shapes
    # x: (Batch, n2, n1) flattened to (Batch, N)
    N = n1 * n2
    x = torch.randn(batch, N, device=device, dtype=dtype)

    # Weights
    w1 = torch.randn(n2, m1, n1, device=device, dtype=dtype)
    w2 = torch.randn(m1, m2, n2, device=device, dtype=dtype)

    # Reference
    # Note: Reference handles arbitrary batch dims.
    ref_out = monarch_matmul_rhs(x, w1, w2, transpose_w=False)

    # Kernel
    # Kernel expects x as (Batch, N) or (..., N)
    out = monarch_transform(x, w1, w2)

    # Verify
    # High tolerance for half precision accumulation in kernel vs float32/acc in torch?
    # Actually my kernel uses float32 accumulation for BF16, but F16 for F16?
    # Using `c_type = CUDA_R_16F` for F16 which is half accumulation?
    # Or `CUDA_R_32F`? I set to `CUDA_R_16F` for F16 in `monarch_cuda.cu`.
    # This might cause precision loss.
    # Let's see.

    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    atol = 2e-2 if dtype == torch.float16 else 1e-1

    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
