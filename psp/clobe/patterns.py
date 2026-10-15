"""The unique patterns in a set of results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._utils import Ordering, _identify, _upper_cells
from .imac import imac

__all__ = ["patterns", "PatternsResult"]


@dataclass(frozen=True)
class PatternsResult:
    """The unique patterns in a set of results, returned by :func:`patterns`.

    Attributes
    ----------
    inequality_matrices
        The unique patterns in order of first appearance, ``(n, k, k)``.
    counts
        How many rows of the data have each pattern, ``(n,)`` int64.
    frequencies
        ``counts`` as proportions of the rows, ``(n,)`` float64.
    pattern
        For each row of the data, the index of its pattern, ``(rows,)`` int64.
    """

    inequality_matrices: torch.Tensor
    counts: torch.Tensor
    frequencies: torch.Tensor
    pattern: torch.Tensor

    @property
    def n(self) -> int:
        """Number of unique patterns."""
        return int(self.inequality_matrices.shape[0])


def patterns(
    data: torch.Tensor | Any,
    thresholds: Sequence[float] | torch.Tensor,
    ordering: Ordering = "semiorder",
    *,
    inclusive: bool = False,
) -> PatternsResult:
    """The unique patterns in results, one row per participant or model run.

    Each row is turned into an inequality matrix with :func:`imac`. The output is
    ready for :func:`gdistance`: each pattern appears once, with its frequency.

    Parameters
    ----------
    data
        ``(rows, k)`` numbers, one column per stimulus: a tensor, an array or anything
        :func:`torch.as_tensor` accepts, such as ``DataFrame.to_numpy()``.
    thresholds, ordering, inclusive
        As in :func:`imac`.

    Examples
    --------
    >>> data = torch.tensor([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9], [0.8, 0.5, 0.2]])
    >>> out = patterns(data, (0.1, -0.1))
    >>> out.n, out.counts, out.pattern
    (2, tensor([2, 1]), tensor([0, 1, 0]))
    """
    x = torch.as_tensor(data)
    if x.ndim != 2:
        raise ValueError(
            "'data' must be a matrix with one row per participant and one column per stimulus"
        )
    if x.dtype == torch.bool or x.is_complex():
        raise ValueError(f"'data' must be numeric, not {x.dtype}")
    if x.dtype.is_floating_point and torch.isnan(x).any():
        raise ValueError("'data' must not have missing values")
    matrices = imac(x, thresholds, ordering, inclusive=inclusive)
    rows = x.shape[0]
    (ids,) = _identify(_upper_cells(matrices))
    # renumber the patterns in order of first appearance, as R's unique() does
    groups = int(ids.max()) + 1 if rows else 0
    first = torch.full((groups,), rows, dtype=torch.int64, device=ids.device).scatter_reduce(
        0, ids, torch.arange(rows, device=ids.device), reduce="amin"
    )
    order = torch.argsort(first)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(groups, device=ids.device)
    pattern = rank[ids]
    counts = torch.bincount(pattern, minlength=groups)
    return PatternsResult(
        inequality_matrices=matrices[first[order]],
        counts=counts,
        frequencies=counts.to(torch.float64) / rows,
        pattern=pattern,
    )
