"""The inequality matrix constructor: the ordinal pattern of a result."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch

from ._utils import Ordering, _check_thresholds, _classify, _is_weak, _tolerance

__all__ = ["imac"]


def imac(
    values: torch.Tensor | Sequence[float],
    thresholds: Sequence[float] | torch.Tensor,
    ordering: Ordering = "semiorder",
    *,
    inclusive: bool = False,
) -> torch.Tensor:
    """Inequality matrix constructor: the ordinal pattern of a result.

    Compares every pair of values. Above the diagonal, cell ``(i, j)`` is ``1`` if
    ``values[i] - values[j] > upper``, ``-1`` if it is ``< lower`` and ``0``
    (approximately equal) otherwise; on and below the diagonal it is ``2``. With
    ``ordering="weak"`` only ties count as approximately equal, whatever the
    thresholds. A difference within :data:`TOLERANCE` of a threshold counts as equal
    to it; for float32 and lower precision, within 8 eps of the dtype.

    Batched over leading dimensions, so ``imac`` fits the ``discretize`` contract of
    :func:`psp.PSP`: ``(n, k)`` in, ``(n, k, k)`` out.

    Parameters
    ----------
    values
        Results of shape ``(..., k)``, one value per stimulus (e.g. response probabilities).
    thresholds
        ``(upper, lower)`` with ``upper >= lower``, e.g. ``(0.1, -0.1)``.
    ordering
        ``"semiorder"`` (the default) or ``"weak"``.
    inclusive
        Count a difference equal to a threshold as larger (``>= upper``) or smaller
        (``<= lower``), without a tolerance, instead of approximately equal. This is
        the rule of clobe before 0.8.0, used by :mod:`psp.models.oscar`. Semiorders only.

    Returns
    -------
    torch.Tensor
        Shape ``(..., k, k)``, the floating dtype of ``values`` (integers become the
        default dtype). A difference that is NaN stays NaN; the R package returns 0.

    Examples
    --------
    >>> imac(torch.tensor([[0.4, 0.3, 0.2], [0.2, 0.2, 0.9]]), (0.1, -0.1))
    tensor([[[ 2.,  0.,  1.],
             [ 2.,  2.,  0.],
             [ 2.,  2.,  2.]],
    <BLANKLINE>
            [[ 2.,  0., -1.],
             [ 2.,  2., -1.],
             [ 2.,  2.,  2.]]])

    As the ``discretize`` of a run:

    >>> from functools import partial
    >>> discretize = partial(imac, thresholds=(0.1, -0.1))

    A difference of exactly a threshold, by default and with ``inclusive``:

    >>> values = torch.tensor([0.5, 0.25], dtype=torch.float64)
    >>> imac(values, (0.25, -0.25))[0, 1].item()
    0.0
    >>> imac(values, (0.25, -0.25), inclusive=True)[0, 1].item()
    1.0
    """
    upper, lower = _check_thresholds(thresholds)
    weak = _is_weak(ordering, inclusive)
    x = torch.as_tensor(values)
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())
    if x.ndim == 0:
        raise ValueError("'values' must have at least one dimension, one value per stimulus")
    difference = x.unsqueeze(-1) - x.unsqueeze(-2)  # (..., k, k): values[i] - values[j]
    cells = _classify(difference, upper, lower, weak, _tolerance(x.dtype), inclusive)
    cells = torch.where(torch.isnan(difference), difference, cells)
    k = x.shape[-1]
    return torch.where(_above(k, x.device), cells, 2.0)


@cache
def _above(k: int, device: torch.device) -> torch.Tensor:
    """The cells above the diagonal of a ``k x k`` matrix."""
    return torch.ones(k, k, dtype=torch.bool, device=device).triu(diagonal=1)
