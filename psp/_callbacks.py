"""The contract (agreement) between :func:`psp.PSP` and the user's ``model`` and ``discretize``.

Both callbacks are *batched* by default: ``model`` receives every parameter set
of an iteration at once, as a ``(n, dimensions)`` tensor, and ``discretize``
receives all of ``model``'s outputs as a ``(n, responses)`` tensor. That is what
lets a run use the GPU. Functions written for one parameter set at a time, as in
the R package, are adapted with :func:`rowwise`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch

__all__ = ["rowwise", "Model", "Discretize"]

#: A batched model: ``(n, dimensions)`` parameters in, ``(n, responses)`` continuous outputs out.
Model = Callable[[torch.Tensor], Any]
#: A batched discretizer: ``(n, responses)`` in, ``(n, dimensionality, dimensionality)`` out.
Discretize = Callable[[torch.Tensor], Any]


def rowwise(
    fn: Callable[..., Any], *, numpy: bool = False
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A function that adapts parameter space partitioning evaluations from a
    batched operation to a sequential one. This function is here to help users
    who are porting models from the R package, which is written for one parameter
    set at a time. Use it with ``PSP(..., batched=False)``.
    
    ``rowwise(fn)`` calls ``fn`` once per row of its input and stacks the results,
    so a ``model`` or ``discretize`` written as in the R package, for a single
    vector, can be handed to :func:`psp.PSP`. Each row is passed as a 1-D
    tensor, or as a NumPy array when ``numpy=True``; ``fn`` may return a tensor,
    an array, a list or a scalar.

    Parameters
    ----------
    fn
        Function of a single parameter set (or a single vector of model output).
    numpy
        Pass each row to ``fn`` as a :class:`numpy.ndarray` instead of a tensor.
        Useful for models written with NumPy or SciPy.

    Examples
    --------
    >>> import torch
    >>> model = rowwise(lambda par: par.cumsum(0))
    >>> model(torch.ones(3, 2))
    tensor([[1., 2.],
            [1., 2.],
            [1., 2.]])
    """

    @wraps(fn)
    def batched(batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 0:
            batch = batch.reshape(1)
        rows = batch.detach().cpu().numpy() if numpy else batch
        outputs = []
        for row in rows:
            out = fn(row)
            try:
                outputs.append(out if torch.is_tensor(out) else torch.as_tensor(np.asarray(out)))
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"{getattr(fn, '__name__', 'the function')} must return numeric values, "
                    f"not {type(out).__name__}"
                ) from error
        if not outputs:
            return batch.new_empty((0,))
        return torch.stack([o.to(outputs[0].device) for o in outputs]).to(batch.device)

    return batched


def as_numeric_tensor(value: Any, what: str, device: torch.device) -> torch.Tensor:
    """Utility function. Convert a callback's return value to a numeric tensor on
    ``device``. Used to check that the user's ``model`` and ``discretize`` return
    numeric values as the right type of data structure.

    Raises
    ------
    ValueError
        If ``value`` is not numeric (strings, objects, complex numbers, ...).
    """
    try:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"{what} must return a numeric tensor, not {type(value).__name__}"
        ) from error
    if (
        tensor.dtype in (torch.bool,)
        or tensor.is_complex()
        or not (tensor.dtype.is_floating_point or _is_integer(tensor.dtype))
    ):
        raise ValueError(f"{what} must return a numeric tensor, not {tensor.dtype}")
    return tensor.to(device)


def _is_integer(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
