"""Shared by the functions of :mod:`psp.clobe`: validation, classification of differences,
and the packing of patterns into sortable integer keys."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

Ordering = Literal["semiorder", "weak"]

#: Absolute tolerance when comparing a difference against a threshold, so that
#: e.g. 0.3 - 0.2 (= 0.0999...) and 0.2 - 0.1 are treated alike; sqrt of float64 eps,
#: as in the R package. Lower precision dtypes use a few of their own eps instead.
TOLERANCE = 1.490116119384765625e-08


def _tolerance(dtype: torch.dtype) -> float:
    """:data:`TOLERANCE`, or 8 eps of ``dtype`` if that is larger (float32, float16, bfloat16)."""
    return max(TOLERANCE, 8 * torch.finfo(dtype).eps)


# A pattern of only -1, 0 and 1 is packed into int64 keys, 39 base-3 digits per
# key (3^39 < 2^63), first cell most significant. Rows of keys sort in the same
# order as the cells they encode, which is the order of the R package.
_DIGITS_PER_KEY = 39

# Largest output of orderings() and permutations of brutes(), in cells, as in R.
_MAX_CELLS = 2**31

# Rows of brutes() and orderings() processed per step.
_CHUNK = 2**22


# --- inequality matrices --------------------------------------------------------------------


def _check_thresholds(thresholds: Sequence[float] | torch.Tensor) -> tuple[float, float]:
    if torch.is_tensor(thresholds):
        values = thresholds.reshape(-1).tolist()
    else:
        values = [float(v) for v in thresholds]
    if len(values) != 2:
        raise ValueError("'thresholds' must have exactly 2 elements: (upper, lower)")
    upper, lower = values
    if upper < lower:
        raise ValueError("'thresholds' must be (upper, lower) with upper >= lower")
    return upper, lower


def _is_weak(ordering: str, inclusive: bool = False) -> bool:
    if ordering == "semiorder":
        return False
    if ordering == "weak":
        if inclusive:
            raise ValueError(
                "'inclusive' applies to semiorders only; a weak ordering compares with 0"
            )
        return True
    raise ValueError(f"'ordering' must be either 'semiorder' or 'weak', got {ordering!r}")


def _classify(
    difference: torch.Tensor,
    upper: float,
    lower: float,
    weak: bool,
    tolerance: float,
    inclusive: bool = False,
) -> torch.Tensor:
    """1 (larger), -1 (smaller) or 0 (approximately equal) for each difference, in its dtype.

    semiorder: approximately equal if ``lower <= difference <= upper``; with
    ``inclusive``, if ``lower < difference < upper``, without a tolerance.
    weak: approximately equal only if the difference is 0 (a tie).
    """
    if inclusive:
        one = torch.ones((), dtype=difference.dtype, device=difference.device)
        return torch.where(
            difference >= upper, one, torch.where(difference <= lower, -one, 0 * one)
        )
    if weak:
        upper = lower = 0.0
    larger = (difference > upper + tolerance).to(difference.dtype)
    smaller = (difference < lower - tolerance).to(difference.dtype)
    return larger - smaller


def _as_patterns(patterns: Any, name: str) -> torch.Tensor:
    """Inequality matrices as a float64 ``(n, k, k)`` tensor; a ``(k, k)`` matrix is ``n = 1``."""
    x = torch.as_tensor(patterns)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[1] != x.shape[2]:
        raise ValueError(
            f"'{name}' must be a square matrix or a (n, k, k) tensor of square matrices"
        )
    return x.to(torch.float64)


def _upper_cells(matrices: torch.Tensor, name: str | None = None) -> torch.Tensor:
    """The strict upper triangles of ``(n, k, k)`` matrices, read row by row: ``(n, cells)``.

    With ``name``, rejects cells that are not finite: they cannot be compared, as NaN
    is not equal to itself. The diagonal and lower triangle are ignored.
    """
    k = matrices.shape[-1]
    i, j = torch.triu_indices(k, k, offset=1, device=matrices.device)
    cells = matrices[:, i, j]
    if name is not None and not torch.isfinite(cells).all():
        raise ValueError(f"'{name}' must not have missing or infinite values above the diagonal")
    return cells + 0.0  # -0.0 becomes 0.0


def _identify(*groups: torch.Tensor) -> list[torch.Tensor]:
    """One id per row of each ``(n_i, cells)`` group; rows with equal cells share an id."""
    rows = torch.cat(groups)
    if rows.shape[1] == 0:  # 1 x 1 matrices: every pattern is the same
        ids = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
    elif rows.shape[1] <= _DIGITS_PER_KEY and ((rows == -1) | (rows == 0) | (rows == 1)).all():
        # the usual coding packs into one integer per pattern, much faster to compare
        _, ids = torch.unique(_pack(rows)[:, 0], return_inverse=True)
    else:
        _, ids = torch.unique(rows, dim=0, return_inverse=True)
    return list(ids.split([g.shape[0] for g in groups]))


def _pack(cells: torch.Tensor) -> torch.Tensor:
    """Pack ``(n, cells)`` of -1, 0, 1 into ``(n, keys)`` int64 (at least one key)."""
    digits = cells.to(torch.int64) + 1
    keys = []
    for start in range(0, max(digits.shape[1], 1), _DIGITS_PER_KEY):
        chunk = digits[:, start : start + _DIGITS_PER_KEY]
        width = chunk.shape[1]
        powers = 3 ** torch.arange(width - 1, -1, -1, dtype=torch.int64, device=digits.device)
        keys.append((chunk * powers).sum(dim=1))
    return torch.stack(keys, dim=1)


def _unpack(keys: torch.Tensor, cells: int) -> torch.Tensor:
    """The inverse of :func:`_pack`: ``(n, cells)`` int8."""
    out = torch.empty((keys.shape[0], cells), dtype=torch.int8, device=keys.device)
    for w, start in enumerate(range(0, cells, _DIGITS_PER_KEY)):
        key = keys[:, w].clone()
        for c in reversed(range(start, min(start + _DIGITS_PER_KEY, cells))):
            out[:, c] = (key % 3 - 1).to(torch.int8)
            key = key // 3
    return out


def _unique_keys(keys: torch.Tensor) -> torch.Tensor:
    """Sorted unique rows of ``(n, w)`` keys."""
    if keys.shape[1] == 1:
        return torch.unique(keys[:, 0]).unsqueeze(1)
    return torch.unique(keys, dim=0)


def _to_matrices(cells: torch.Tensor, k: int) -> torch.Tensor:
    """``(n, cells)`` upper triangles back to ``(n, k, k)`` int8, 2 on and below the diagonal."""
    out = torch.full((cells.shape[0], k, k), 2, dtype=torch.int8, device=cells.device)
    i, j = torch.triu_indices(k, k, offset=1, device=cells.device)
    out[:, i, j] = cells.to(torch.int8)
    return out
