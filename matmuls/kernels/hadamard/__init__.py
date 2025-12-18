from typing import Any


def hadamard_transform_cuda(x: Any, *args, **kwargs):
    """
    Apply Fast Hadamard Transform to the input tensor (PyTorch or JAX) using CUDA backend.
    """
    # Check for PyTorch Tensor
    is_torch = False
    try:
        import torch

        if isinstance(x, torch.Tensor):
            is_torch = True
    except ImportError:
        pass

    if is_torch:
        from .pt.loader import hadamard_transform_cuda as _func

        return _func(x, *args, **kwargs)

    # Check for JAX Array
    is_jax = False
    try:
        import jax

        if isinstance(x, (jax.Array, type(jax.numpy.array([])))):
            is_jax = True
    except ImportError:
        pass

    if is_jax:
        from .jx.loader import hadamard_transform_cuda as _func

        return _func(x, *args, **kwargs)

    type_name = type(x).__name__
    raise TypeError(
        f"Unsupported input type: {type_name}. Expected torch.Tensor or jax.Array."
    )


def hadamard_transform_triton(x: Any, *args, **kwargs):
    """
    Apply Fast Hadamard Transform to the input tensor (PyTorch or JAX) using Triton backend.
    """
    # Check for PyTorch Tensor
    is_torch = False
    try:
        import torch

        if isinstance(x, torch.Tensor):
            is_torch = True
    except ImportError:
        pass

    if is_torch:
        from .pt.loader import hadamard_transform_triton as _func

        return _func(x, *args, **kwargs)

    # Check for JAX Array
    is_jax = False
    try:
        import jax

        if isinstance(x, (jax.Array, type(jax.numpy.array([])))):
            is_jax = True
    except ImportError:
        pass

    if is_jax:
        from .jx.loader import hadamard_transform_triton as _func

        return _func(x, *args, **kwargs)

    type_name = type(x).__name__
    raise TypeError(
        f"Unsupported input type: {type_name}. Expected torch.Tensor or jax.Array."
    )


# Alias for backward compatibility (default to CUDA)
hadamard_transform = hadamard_transform_cuda
