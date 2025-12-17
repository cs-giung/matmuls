from typing import Any


def matmul(a: Any, b: Any):
    # Check for PyTorch
    try:
        import torch

        if isinstance(a, torch.Tensor):
            from .pt import matmul as _torch_impl

            return _torch_impl(a, b)
    except ImportError:
        pass

    # Check for JAX
    try:
        import jax
        import jax.numpy as jnp

        if isinstance(a, (jax.Array, type(jnp.array([])))):
            from .jx import matmul as _jax_impl

            return _jax_impl(a, b)
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported input type: {type(a).__name__}. "
        "Expected torch.Tensor or jax.Array."
    )
