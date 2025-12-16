import math

import numpy as np
import pytest
import scipy.linalg
import torch

from matmuls.kernels.hadamard import hadamard_transform


def get_scale(size):
    return math.sqrt(1 / size)


@pytest.mark.parametrize("m", [2, 4, 16, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hadamard_transform(m, dtype):
    elem_c = m * 128
    torch.manual_seed(0)
    a = torch.randn((elem_c // m, m), device="cuda", dtype=dtype)

    # Reference implementation
    truth_hadamard = torch.tensor(
        np.array(scipy.linalg.hadamard(m)), device="cuda", dtype=dtype
    ) * get_scale(m)
    expected = a @ truth_hadamard

    # Module implementation (copy to avoid inplace modification of original if we tested inplace)
    # Testing inplace=False first if supported, but kernel seems to be inplace=True based on original test
    # Original test called: faster_hadamard_transform.hadamard_transform(a, inplace=True)

    input_tensor = a.clone()
    hadamard_transform(input_tensor, inplace=True)

    # Higher tolerance for BF16
    atol = 1e-2 if dtype == torch.float16 else 5e-2
    assert torch.allclose(expected, input_tensor, atol=atol), (
        f"Mismatch for size {m}, dtype {dtype}"
    )
