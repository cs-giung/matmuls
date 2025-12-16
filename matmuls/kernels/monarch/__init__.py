from typing import Any


def monarch_transform(x: Any, w1: Any, w2: Any):
    # Check for PyTorch
    try:
        import torch

        if isinstance(x, torch.Tensor):
            from .torch import monarch_transform as _torch_impl

            return _torch_impl(x, w1, w2)
    except ImportError:
        pass

    # Check for JAX
    try:
        import jax

        if isinstance(x, (jax.Array, type(jax.numpy.array([])))):
            from .jax import monarch_transform as _jax_impl

            return _jax_impl(x, w1, w2)
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported input type: {type(x).__name__}. "
        "Expected torch.Tensor or jax.Array."
    )
