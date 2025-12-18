import jax
import torch
from jaxtyping import Array, Float


def matmul(a: Float[Array, "M K"], b: Float[Array, "K N"]):
    if isinstance(a, torch.Tensor):
        from .pt import matmul as _func

        return _func(a, b)

    if isinstance(a, jax.Array):
        from .jx import matmul as _func

        return _func(a, b)

    raise TypeError(
        f"Unsupported input type: {type(a).__name__}. "
        f"Expected torch.Tensor or jax.Array."
    )
