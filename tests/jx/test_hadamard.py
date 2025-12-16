import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from matmuls.kernels.hadamard import hadamard_transform


def get_scale(size):
    return math.sqrt(1 / size)


@pytest.mark.parametrize("m", [2, 4, 16, 128, 1024])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_hadamard_transform(m, dtype):
    elem_c = m * 128
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (elem_c // m, m), dtype=dtype)

    truth_hadamard = jnp.array(
        np.array(scipy.linalg.hadamard(m)), dtype=dtype
    ) * get_scale(m)
    expected = a @ truth_hadamard

    # Module implementation
    out = hadamard_transform(a)

    # Verification
    # Transfer to numpy for easy comparison
    # Higher tolerance for BF16
    atol = 1e-2 if dtype == jnp.float16 else 1e-1
    rtol = 1e-2 if dtype == jnp.float16 else 5e-2

    # Cast BF16 to float32 for comparison to avoid numpy truncation issues if numpy doesn't support bf16 well
    out_np = np.array(out, dtype=np.float32)
    expected_np = np.array(expected, dtype=np.float32)

    np.testing.assert_allclose(out_np, expected_np, rtol=rtol, atol=atol)
