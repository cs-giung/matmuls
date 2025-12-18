import jax
import torch
from jaxtyping import Array, Float


def hadamard_transform_cuda(x: Float[Array, "..."], *args, **kwargs):
    if isinstance(x, torch.Tensor):
        from .pt.loader import hadamard_transform_cuda as _func

        return _func(x, *args, **kwargs)

    elif isinstance(x, jax.Array):
        from .jx.loader import hadamard_transform_cuda as _func

        return _func(x, *args, **kwargs)

    else:
        raise TypeError(
            f"Unsupported input type: {type(x).__name__}. "
            f"Expected torch.Tensor or jax.Array."
        )


def hadamard_transform_triton(x: Float[Array, "..."], *args, **kwargs):
    if isinstance(x, torch.Tensor):
        from .pt.loader import hadamard_transform_triton as _func

        return _func(x, *args, **kwargs)

    elif isinstance(x, jax.Array):
        from .jx.loader import hadamard_transform_triton as _func

        return _func(x, *args, **kwargs)

    else:
        raise TypeError(
            f"Unsupported input type: {type(x).__name__}. "
            f"Expected torch.Tensor or jax.Array."
        )
