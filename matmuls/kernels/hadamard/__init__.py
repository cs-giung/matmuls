from typing import Any


def hadamard_transform(x: Any, *args, **kwargs):
    """
    Apply Fast Hadamard Transform to the input tensor (PyTorch or JAX).

    Args:
        x: Input tensor. Can be a torch.Tensor or a jax.Array.
           Must be on GPU and have dtype float16 or bfloat16.
        *args, **kwargs: Additional arguments passed to the backend implementation.
                         (e.g. 'inplace' for PyTorch)

    Returns:
        The transformed tensor/array matching the input framework.
    """
    # Check for PyTorch Tensor
    try:
        import torch

        if isinstance(x, torch.Tensor):
            from .torch import hadamard_transform as _hadamard_torch

            return _hadamard_torch(x, *args, **kwargs)
    except ImportError:
        pass

    # Check for JAX Array
    # We check for jax.Array or duck-typing slightly if jax isn't imported yet?
    # Better to try importing jax.
    try:
        import jax

        if isinstance(x, (jax.Array, type(jax.numpy.array([])))):
            from .jax import hadamard_transform as _hadamard_jax

            return _hadamard_jax(x, *args, **kwargs)
    except ImportError:
        pass

    # Fallback / Error
    type_name = type(x).__name__
    raise TypeError(
        f"Unsupported input type: {type_name}. Expected torch.Tensor or jax.Array."
    )
