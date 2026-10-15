"""Counting ordinal patterns: Stirling numbers, weak orderings and semiorders.

Counts are Python integers, so they are exact however large; the R package warns past 2^53."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

__all__ = ["stirling2", "fubini", "semiorders"]


def _check_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative whole number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"'{name}' must be a non-negative whole number, got {value!r}")
    return int(value)


@cache
def _stirling2_row(n: int) -> tuple[int, ...]:
    """``S(n, j)`` for ``j = 0, ..., n``, by ``S(i, j) = j S(i-1, j) + S(i-1, j-1)``."""
    if n == 0:
        return (1,)
    previous = _stirling2_row(n - 1) + (0,)
    return (0,) + tuple(j * previous[j] + previous[j - 1] for j in range(1, n + 1))


def stirling2(n: int, k: int) -> int:
    """Stirling number of the second kind: ways to split ``n`` items into ``k`` non-empty groups.

    >>> stirling2(5, 2), stirling2(10, 3), stirling2(3, 5)
    (15, 9330, 0)
    """
    n = _check_count(n, "n")
    k = _check_count(k, "k")
    return 0 if k > n else _stirling2_row(n)[k]


def fubini(k: int, levels: int | None = None) -> int:
    """The number of weak orderings of ``k`` items (the Fubini or ordered Bell number).

    Parameters
    ----------
    k
        Number of items (stimuli).
    levels
        At most this many tied groups, e.g. the number of distinct values a measure
        can take; ``None`` for no limit.

    Examples
    --------
    >>> [fubini(k) for k in range(6)], fubini(4, levels=3)
    ([1, 1, 3, 13, 75, 541], 51)
    """
    k = _check_count(k, "k")
    most = k if levels is None else min(k, _check_count(levels, "levels"))
    row = _stirling2_row(k)
    # weak orderings with j tied groups: the ways to split k items into j groups,
    # times the ways to order the groups
    return sum(math.factorial(j) * row[j] for j in range(most + 1))


def semiorders(k: int) -> int:
    """The number of labelled semiorders of ``k`` items: the universal set of :func:`imac`.

    The exponential generating function is ``C(1 - exp(-x))``, where ``C`` is that
    of the Catalan numbers, which gives an alternating sum over Stirling numbers.

    >>> [semiorders(k) for k in range(7)]
    [1, 1, 3, 19, 183, 2371, 38703]
    """
    k = _check_count(k, "k")
    row = _stirling2_row(k)
    return sum(
        (-1) ** (k - j) * (math.comb(2 * j, j) // (j + 1)) * math.factorial(j) * row[j]
        for j in range(k + 1)
    )
