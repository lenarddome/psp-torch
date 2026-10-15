"""Distances between ordinal patterns."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ._utils import _as_patterns, _upper_cells

__all__ = ["pattern_distance"]


def pattern_distance(
    x: torch.Tensor | Any,
    y: torch.Tensor | Any | None = None,
    method: Literal["kemeny", "hamming"] = "kemeny",
) -> torch.Tensor:
    """Distances between every pattern in ``x`` and every pattern in ``y``.

    Compares the strict upper triangles. ``"kemeny"`` sums the absolute differences
    of the cells, so reversing a relation (1 to -1) counts 2 and breaking a tie
    counts 1; ``"hamming"`` counts the cells that differ.

    Parameters
    ----------
    x, y
        ``(n, k, k)`` and ``(m, k, k)`` patterns, or single ``(k, k)`` matrices;
        ``y`` defaults to ``x``.
    method
        ``"kemeny"`` (the default) or ``"hamming"``.

    Returns
    -------
    torch.Tensor
        ``(n, m)`` float64.

    Examples
    --------
    >>> from psp.clobe import imac
    >>> a = imac(torch.tensor([0.9, 0.5, 0.1]), (0.1, -0.1))  # 1 > 2 > 3
    >>> c = imac(torch.tensor([0.1, 0.5, 0.9]), (0.1, -0.1))  # 3 > 2 > 1
    >>> pattern_distance(a, c), pattern_distance(a, c, method="hamming")
    (tensor([[6.]], dtype=torch.float64), tensor([[3.]], dtype=torch.float64))
    """
    if method not in ("kemeny", "hamming"):
        raise ValueError(f"'method' must be either 'kemeny' or 'hamming', got {method!r}")
    a = _as_patterns(x, "x")
    b = _as_patterns(x if y is None else y, "y")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("'x' and 'y' must contain matrices of the same dimensions")
    a_cells = _upper_cells(a, "x")
    b_cells = _upper_cells(b, "y")
    if a_cells.shape[1] == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float64, device=a.device)
    # on a GPU, matrix products beat cdist; on the CPU, float32 cdist is faster
    gpu = a_cells.device.type != "cpu"
    if gpu and a_cells.shape[1] < 2**24 and _usual(a_cells) and _usual(b_cells):
        return _by_indicators(a_cells, b_cells, method == "kemeny")
    # integer codes give exact distances in float32 while every sum stays below 2^24, and
    # float32 is far faster, above all on GPUs with slow float64
    largest = float(torch.cat([a_cells.abs().max(0).values, b_cells.abs().max(0).values]).max())
    exact = (
        a_cells.eq(a_cells.round()).all()
        and b_cells.eq(b_cells.round()).all()
        and 2 * largest * a_cells.shape[1] < 2**24
    )
    if exact:
        a_cells, b_cells = a_cells.float(), b_cells.float()
    distance = torch.cdist(a_cells, b_cells, p=1.0 if method == "kemeny" else 0.0)
    return distance.to(torch.float64)


def _usual(cells: torch.Tensor) -> bool:
    """Whether every cell is -1, 0 or 1."""
    return bool(((cells == -1) | (cells == 0) | (cells == 1)).all())


def _by_indicators(a: torch.Tensor, b: torch.Tensor, kemeny: bool) -> torch.Tensor:
    """Distances between patterns of -1, 0 and 1, as products of 0/1 indicator matrices.

    The number of cells that differ is the number of cells less those where both patterns
    are 1, both 0 or both -1; Kemeny adds 1 more for every cell where one is 1 and the other
    -1. Matrix products are fast on any device, and exact in float32 for counts below 2^24.
    """
    up, tie, down = ((a == v).float() for v in (1, 0, -1))
    distance = (
        a.shape[1]
        - torch.cat([up, tie, down], 1) @ torch.cat([(b == v).float() for v in (1, 0, -1)], 1).T
    )
    if kemeny:
        distance += torch.cat([up, down], 1) @ torch.cat([(b == -1).float(), (b == 1).float()], 1).T
    return distance.to(torch.float64)
