"""Every pattern of a number of stimuli: every semiorder, or every weak ordering."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from ._utils import (
    _CHUNK,
    _MAX_CELLS,
    Ordering,
    _is_weak,
    _pack,
    _to_matrices,
    _unique_keys,
    _unpack,
)
from .combinatorics import fubini, semiorders

__all__ = ["orderings", "OrderingsResult"]


@dataclass(frozen=True)
class OrderingsResult:
    """The result of :func:`orderings`.

    Attributes
    ----------
    inequality_matrices
        Every pattern, ``(n, k, k)`` int8, in the order of the R package.
    """

    inequality_matrices: torch.Tensor

    @property
    def n(self) -> int:
        """Number of patterns."""
        return int(self.inequality_matrices.shape[0])


def _weak_ranks(k: int) -> torch.Tensor:
    """One rank vector per weak ordering of ``k`` items, ``(fubini(k), k)`` int8.

    Each ordering of items ``0..i-1`` extends to item ``i`` either by tying it with an
    existing level or by inserting a new level for it, so each ordering appears once.
    """
    ranks = torch.zeros((1, 1), dtype=torch.int8)
    for i in range(1, k):
        levels = ranks.max(dim=1).values.to(torch.int64) + 1
        parts = []
        for t in range(i + 1):
            tied = ranks[t < levels]
            parts.append(torch.cat([tied, torch.full((tied.shape[0], 1), t, dtype=torch.int8)], 1))
            below = ranks[t <= levels]
            below = below + (below >= t).to(torch.int8)
            parts.append(
                torch.cat([below, torch.full((below.shape[0], 1), t, dtype=torch.int8)], 1)
            )
        ranks = torch.cat(parts)
    return ranks


def _reaches(k: int) -> list[list[int]]:
    """Every non-decreasing sequence with ``p <= reach[p] <= k - 1``; there are Catalan(k)."""
    out: list[list[int]] = []

    def extend(prefix: list[int]) -> None:
        p = len(prefix)
        if p == k:
            out.append(prefix)
            return
        for r in range(max(p, prefix[-1] if prefix else 0), k):
            extend(prefix + [r])

    extend([])
    return out


def orderings(length: int, ordering: Ordering = "semiorder") -> OrderingsResult:
    """Every pattern of ``length`` stimuli: every semiorder, or every weak ordering.

    The patterns any :func:`imac` result can take, so ``orderings(k).n`` equals
    ``semiorders(k)`` and ``orderings(k, "weak").n`` equals ``fubini(k)``.

    Examples
    --------
    >>> orderings(2).inequality_matrices
    tensor([[[ 2, -1],
             [ 2,  2]],
    <BLANKLINE>
            [[ 2,  0],
             [ 2,  2]],
    <BLANKLINE>
            [[ 2,  1],
             [ 2,  2]]], dtype=torch.int8)
    """
    if (
        isinstance(length, bool)
        or not isinstance(length, int | float)
        or length < 1
        or length != int(length)
    ):
        raise ValueError(f"'length' must be a whole number of at least 1, got {length!r}")
    k = int(length)
    if k == 1:  # a single stimulus has one pattern, with no cells
        return OrderingsResult(inequality_matrices=torch.full((1, 1, 1), 2, dtype=torch.int8))
    weak = _is_weak(ordering)
    n = fubini(k) if weak else semiorders(k)
    if n * k * k > _MAX_CELLS:
        raise ValueError(
            f"There are too many patterns to enumerate ({n}); "
            f"use {'fubini' if weak else 'semiorders'}() to count them"
        )
    i, j = torch.triu_indices(k, k, offset=1)
    found = []
    if weak:
        ranks = _weak_ranks(k)
        for start in range(0, ranks.shape[0], _CHUNK):
            r = ranks[start : start + _CHUNK]
            found.append(_pack(torch.sign(r[:, i] - r[:, j])))
    else:
        # Sorting the items by value, the items approximately equal to the item at
        # position p and above it are those at positions p + 1, ..., reach[p]. Each
        # reach sequence is an unlabelled semiorder; it is labelled in every way, and
        # patterns that are labelled alike are kept once.
        position = torch.tensor(list(itertools.permutations(range(k))), dtype=torch.int64)
        pa, pb = position[:, i], position[:, j]
        reaches = torch.tensor(_reaches(k), dtype=torch.int64)
        # a few reach sequences at once: large batches would not stay in the cache
        step = max(1, 4096 // position.shape[0])
        batch: list[torch.Tensor] = []
        for start in range(0, reaches.shape[0], step):
            r = reaches[start : start + step, None, :]  # (reaches, 1, k)
            ra = r.expand(-1, pa.shape[0], -1).gather(2, pa.expand(r.shape[0], -1, -1))
            rb = r.expand(-1, pb.shape[0], -1).gather(2, pb.expand(r.shape[0], -1, -1))
            cells = torch.where(
                pa < pb,
                torch.where(pb <= ra, 0, -1),
                torch.where(pa <= rb, 0, 1),
            )
            batch.append(_unique_keys(_pack(cells.reshape(-1, len(i)))))
            if sum(b.shape[0] for b in batch) >= _CHUNK:
                found.append(_unique_keys(torch.cat(batch)))
                batch = []
        found.extend(batch)
    keys = _unique_keys(torch.cat(found))
    return OrderingsResult(inequality_matrices=_to_matrices(_unpack(keys, len(i)), k))
