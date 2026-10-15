"""g-distance: accommodation and excess flexibility of a model against human data."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._utils import TOLERANCE, _as_patterns, _identify, _upper_cells

__all__ = ["gdistance", "GDistanceResult"]


@dataclass(frozen=True)
class GDistanceResult:
    """The result of :func:`gdistance`.

    Attributes
    ----------
    gdistance
        ``sqrt(weight * (1 - alpha)^2 + (1 - weight) * beta^2)``; without ``beta``
        when it is NaN.
    alpha
        Accommodation: the summed frequency of the human patterns the model produces.
    beta
        Excess flexibility: the unique model patterns that no human shows, as a
        proportion of the patterns in the universal set that no human shows. NaN when
        humans show every pattern of the universal set.
    accommodation
        Indices (0-based) of the human patterns the model produces, ``(m,)`` int64.
    """

    gdistance: float
    alpha: float
    beta: float
    accommodation: torch.Tensor

    def __float__(self) -> float:
        return self.gdistance


def gdistance(
    human: torch.Tensor | Any,
    model: torch.Tensor | Any,
    universal: float,
    weight: float,
    frequencies: torch.Tensor | Sequence[float],
) -> GDistanceResult:
    """g-distance between the heterogeneity of humans and of a model (Dome and Wills, 2025).

    Two matrices are the same pattern if their strict upper triangles are equal;
    any numeric coding of the relations works, not only -1, 0 and 1.

    Parameters
    ----------
    human
        The patterns people show, ``(n, k, k)`` or one ``(k, k)`` matrix, each pattern
        once (see :func:`patterns`).
    model
        The patterns the model produces, ``(m, k, k)`` or ``(k, k)``, e.g.
        :attr:`psp.PSPResult.ordinal_patterns`.
    universal
        The number of possible patterns, e.g. ``semiorders(k)`` or ``brutes(...).n``.
    weight
        Weight of accommodation, between 0 and 1; ``1 - weight`` is the weight of
        excess flexibility.
    frequencies
        The proportion of people with each human pattern, ``(n,)``.

    Warns
    -----
    UserWarning
        If ``frequencies`` do not sum to 1 or ``human`` repeats a pattern, in which
        case ``alpha`` is not a proportion; or if ``universal`` equals the number of
        human patterns, in which case ``beta`` is NaN and left out of the distance.

    Examples
    --------
    >>> from psp.clobe import patterns, semiorders
    >>> human = patterns(torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.5, 0.2], [0.1, 0.5, 0.9]]),
    ...                  (0.1, -0.1))
    >>> model = patterns(torch.tensor([[0.9, 0.5, 0.1], [0.5, 0.5, 0.5]]), (0.1, -0.1))
    >>> out = gdistance(human.inequality_matrices, model.inequality_matrices,
    ...                 universal=semiorders(3), weight=0.5, frequencies=human.frequencies)
    >>> round(out.gdistance, 4), round(out.alpha, 4), round(out.beta, 4), out.accommodation
    (0.2393, 0.6667, 0.0588, tensor([0]))
    """
    h = _as_patterns(human, "human")
    m = _as_patterns(model, "model")
    if h.shape[1:] != m.shape[1:]:
        raise ValueError("'human' and 'model' must contain square matrices of the same dimensions")
    h_cells = _upper_cells(h, "human")
    m_cells = _upper_cells(m, "model")
    given = (
        frequencies
        if torch.is_tensor(frequencies)
        else torch.as_tensor(frequencies, dtype=torch.float64)
    )
    eps = torch.finfo(given.dtype).eps if given.dtype.is_floating_point else 0.0
    freq = given.to(torch.float64).reshape(-1).to(h.device)
    if freq.numel() != h.shape[0]:
        raise ValueError(
            f"'frequencies' has {freq.numel()} values, but 'human' has {h.shape[0]} patterns; "
            "give one frequency per pattern"
        )
    if not torch.isfinite(freq).all() or (freq < 0).any():
        raise ValueError("'frequencies' must be non-negative numbers without missing values")
    h_ids, m_ids = _identify(h_cells, m_cells)
    h_unique = torch.unique(h_ids)
    observed = h_unique.numel()
    if universal < observed:
        raise ValueError(
            "'universal' cannot be smaller than the number of unique patterns in 'human'"
        )
    if not 0 <= weight <= 1:
        raise ValueError("'weight' must be between 0 and 1")

    # alpha sums the frequency of every accommodated pattern, so it is a proportion
    # only if the frequencies sum to 1 and each pattern appears once
    total = float(freq.sum())
    # float32 frequencies of n patterns can be off by n eps of float32
    if abs(total - 1) > max(TOLERANCE, freq.numel() * eps):
        warnings.warn(
            f"'frequencies' sum to {total:.10g}, not 1, so alpha is not a proportion; "
            "give the proportion of people with each pattern",
            stacklevel=2,
        )
    duplicated = h.shape[0] - observed
    if duplicated > 0:
        warnings.warn(
            f"'human' has {duplicated} duplicated pattern{'' if duplicated == 1 else 's'}. "
            "Beta counts each pattern once, but alpha adds the frequency of every copy; "
            "use patterns() to keep each pattern once with its total frequency",
            stacklevel=2,
        )

    accommodated = torch.isin(h_ids, m_ids)
    alpha = float((freq * accommodated).sum())
    if universal == observed:
        # humans show every possible pattern, so there is nothing left to predict
        beta = math.nan
        distance = math.sqrt(weight * (1 - alpha) ** 2)
        warnings.warn(
            "'beta' cannot be calculated because human = universal, so it was excluded "
            "from distance",
            stacklevel=2,
        )
    else:
        novel = torch.unique(m_ids[~torch.isin(m_ids, h_unique)]).numel()
        beta = novel / (universal - observed)
        distance = math.sqrt(weight * (1 - alpha) ** 2 + (1 - weight) * beta**2)
    return GDistanceResult(
        gdistance=distance,
        alpha=alpha,
        beta=beta,
        accommodation=torch.nonzero(accommodated).reshape(-1),
    )
