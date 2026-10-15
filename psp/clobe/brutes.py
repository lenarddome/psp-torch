"""The patterns a measure with given values can produce, by brute force."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ._utils import (
    _CHUNK,
    _DIGITS_PER_KEY,
    _MAX_CELLS,
    TOLERANCE,
    Ordering,
    _check_thresholds,
    _classify,
    _is_weak,
    _pack,
    _to_matrices,
    _tolerance,
    _unique_keys,
    _unpack,
)

__all__ = ["brutes", "BrutesResult"]


@dataclass(frozen=True)
class BrutesResult:
    """The result of :func:`brutes`.

    Attributes
    ----------
    n
        Number of unique patterns.
    inequality_matrices
        The unique patterns, ``(n, k, k)`` int8, or ``None`` if not requested.
    permutations
        Every assignment of the dependent values to the stimuli, ``(m^k, k)`` float64,
        the last stimulus changing fastest, or ``None`` if not requested.
    """

    n: int
    inequality_matrices: torch.Tensor | None
    permutations: torch.Tensor | None


def brutes(
    dependent: torch.Tensor | Sequence[float],
    length: int,
    thresholds: Sequence[float] | torch.Tensor,
    ordering: Ordering = "semiorder",
    *,
    return_matrices: bool = True,
    return_permutations: bool = False,
    inclusive: bool = False,
    device: torch.device | str | None = None,
    chunk_size: int = _CHUNK,
) -> BrutesResult:
    """The patterns a measure with the given values can produce, by brute force.

    Assigns the ``m`` values of ``dependent`` to ``length`` stimuli in every one of the
    ``m^length`` ways, with repetition, and collects the unique patterns :func:`imac`
    gives them. This is the universal set when the measurement scale is too coarse to
    realise every semiorder (or weak ordering).

    Parameters
    ----------
    dependent
        The values the measure can take, e.g. ``torch.linspace(0, 1, 11)``. Compared
        in float64, with the tolerance of their own dtype (see :func:`imac`).
    length
        Number of stimuli.
    thresholds, ordering, inclusive
        As in :func:`imac`.
    return_matrices
        Return the patterns; otherwise only their number.
    return_permutations
        Also return every assignment of values to stimuli.
    device
        Where to do the work, e.g. ``"cuda"``; the results are returned on the CPU.
    chunk_size
        Assignments evaluated at once; lower it if the device runs out of memory.

    Examples
    --------
    >>> [brutes(torch.linspace(0, 1, 11), k, (0.1, -0.1)).n for k in range(1, 5)]
    [1, 3, 19, 183]
    >>> brutes(torch.tensor([0.0, 1.0, 2.0]), 4, (0.5, -0.5), "weak").n  # at most 3 levels
    51
    """
    values = torch.as_tensor(dependent).reshape(-1)
    tolerance = _tolerance(values.dtype) if values.dtype.is_floating_point else TOLERANCE
    values = values.to(torch.float64)
    if values.numel() == 0:
        raise ValueError("'dependent' must not be empty")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"'length' must be a whole number of at least 1, got {length!r}")
    upper, lower = _check_thresholds(thresholds)
    weak = _is_weak(ordering, inclusive)
    if chunk_size < 1:
        raise ValueError("'chunk_size' must be at least 1")
    m, k = values.numel(), length
    total = m**k
    if total >= 2**63:
        raise ValueError("There are too many permutations (len(dependent) ** length) to enumerate")
    if return_permutations and total * k > _MAX_CELLS:
        raise ValueError("There are too many permutations to return; use return_permutations=False")

    # classify every pair of dependent values once, on the CPU in float64, so that the
    # patterns do not depend on the device
    difference = values.unsqueeze(1) - values.unsqueeze(0)
    classes = _classify(difference, upper, lower, weak, tolerance, inclusive)
    classes = classes.to(torch.int64).to(device)
    cells = k * (k - 1) // 2
    if cells <= _DIGITS_PER_KEY:
        found = _keys_by_stimulus(classes, k, chunk_size)
    else:
        found = _keys_by_pair(classes, k, chunk_size)
    permutations = None
    if return_permutations:
        permutations = torch.cat(
            [values[_digits(begin, min(begin + chunk_size, total), m, k, values.device)]
             for begin in range(0, total, chunk_size)]
        )  # fmt: skip
    return BrutesResult(
        n=int(found.shape[0]),
        inequality_matrices=_to_matrices(_unpack(found, cells), k).cpu()
        if return_matrices
        else None,
        permutations=permutations,
    )


def _digits(begin: int, end: int, m: int, k: int, device: torch.device) -> torch.Tensor:
    """Assignments ``begin`` to ``end - 1`` as ``(end - begin, k)`` indices of values: the
    assignment's number written in base ``m``, the last stimulus the least significant digit."""
    rest = torch.arange(begin, end, device=device)
    digits = torch.empty((rest.numel(), k), dtype=torch.int64, device=device)
    for column in reversed(range(k)):
        digits[:, column] = rest % m
        rest = rest // m
    return digits


def _keys_by_pair(classes: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    """Sorted unique keys of every assignment, classifying all pairs of each; any ``k``."""
    m = classes.shape[0]
    i, j = torch.triu_indices(k, k, offset=1, device=classes.device)
    found = None
    for begin in range(0, m**k, chunk_size):
        digits = _digits(begin, min(begin + chunk_size, m**k), m, k, classes.device)
        keys = _unique_keys(_pack(classes[digits[:, i], digits[:, j]]))
        found = keys if found is None else _unique_keys(torch.cat([found, keys]))
    assert found is not None
    return found


def _not_in(keys: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """The ``keys`` that are not in the sorted ``known``, by binary search."""
    if known.numel() == 0:
        return keys
    at = torch.searchsorted(known, keys).clamp(max=known.numel() - 1)
    return keys[known[at] != keys]


def _keys_by_stimulus(classes: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    """Sorted unique keys of every assignment, for patterns that fit one key (``k <= 9``).

    Builds the keys one stimulus at a time, with the cells grouped by the later stimulus of
    each pair: adding stimulus ``j`` with value ``v`` appends the ``j`` cells
    ``classes[d_0, v], ..., classes[d_{j-1}, v]``. For each prefix of an assignment, ``ahead``
    holds those digits for every possible ``v``, so extending a prefix costs one
    multiply-add instead of classifying ``j`` pairs. The keys are put in the R package's row
    by row order at the end.
    """
    m, device = classes.shape[0], classes.device
    digit = classes + 1  # digit[a, b]: the cell of a pair whose values are a, b
    # the last ``tail`` stimuli are expanded as tensors, the first ``k - tail`` are looped over
    tail = 1
    while tail < k and m ** (tail + 1) <= chunk_size:
        tail += 1
    head = k - tail
    batch = max(1, chunk_size // m**tail)
    known = torch.empty(0, dtype=torch.int64, device=device)  # sorted
    pending: list[torch.Tensor] = []
    waiting = 0
    for begin in range(0, m**head, batch):
        prefix = _digits(begin, min(begin + batch, m**head), m, head, device)
        key = torch.zeros(prefix.shape[0], dtype=torch.int64, device=device)
        ahead = torch.zeros((prefix.shape[0], m), dtype=torch.int64, device=device)
        for j in range(head):  # stimuli whose values are given by the prefix
            v = prefix[:, j]
            key = key * 3**j + ahead.gather(1, v[:, None])[:, 0]
            ahead = ahead * 3 + digit[v]
        for j in range(head, k):  # every value of each remaining stimulus
            key = (key[:, None] * 3**j + ahead).reshape(-1)
            if j < k - 1:
                ahead = (ahead[:, None, :] * 3 + digit[None]).reshape(-1, m)
        new = torch.unique(_not_in(key, known))
        pending.append(new)
        waiting += new.numel()
        # most keys are already known once the search has gone a while, so merges are rare
        if waiting >= max(chunk_size, known.numel()):
            known = torch.unique(torch.cat([known, *pending]))
            pending, waiting = [], 0
    grouped = torch.unique(torch.cat([known, *pending]))
    # from cells grouped by the later stimulus to the R package's row by row order
    by_later = [(i, j) for j in range(k) for i in range(j)]
    by_row = [(i, j) for i in range(k) for j in range(i + 1, k)]
    order = torch.tensor(
        [by_later.index(pair) for pair in by_row], dtype=torch.int64, device=device
    )
    return _unique_keys(_pack(_unpack(grouped[:, None], len(by_later))[:, order]))
