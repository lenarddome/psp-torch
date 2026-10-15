"""Global benchmarks for computational models: a PyTorch port of the R package clobe 0.8.0.

The tools used to compare model and human heterogeneity (Dome and Wills, 2025):
inequality matrices, g-distance, and the counting and enumeration of the ordinal
patterns (semiorders and weak orderings) that make up the universal set.

The port follows the conventions of the rest of :mod:`psp` rather than those of R:

- A set of inequality matrices has shape ``(n, k, k)``, matrix first, as
  :attr:`psp.PSPResult.ordinal_patterns` does; the R package uses ``(k, k, n)``.
  A single ``(k, k)`` matrix is treated as a set of one.
- Indices are 0-based.
- :func:`imac` is batched, so it can be handed to :func:`psp.PSP` as ``discretize``.
- Counts are Python integers and therefore exact, however large.
- :func:`brutes` runs on any torch device instead of a pool of threads.

One file per function, as in the R package:

- :mod:`psp.clobe.imac`: :func:`imac`, the inequality matrix of a result.
- :mod:`psp.clobe.patterns`: :func:`patterns`, the unique patterns in a set of results.
- :mod:`psp.clobe.gdistance`: :func:`gdistance`, model against human heterogeneity.
- :mod:`psp.clobe.pattern_distance`: :func:`pattern_distance`, Kemeny and Hamming distances.
- :mod:`psp.clobe.combinatorics`: :func:`stirling2`, :func:`fubini` and :func:`semiorders`,
  the size of the universal set.
- :mod:`psp.clobe.orderings`: :func:`orderings`, every semiorder or weak ordering.
- :mod:`psp.clobe.brutes`: :func:`brutes`, the patterns a coarse measurement scale can produce.

Inequality matrices hold ``1`` (larger), ``-1`` (smaller) or ``0`` (approximately
equal) above the diagonal, comparing stimulus ``i`` with stimulus ``j > i``, and
``2`` on and below the diagonal, which carry no information.

Dome, L., & Wills, A. J. (2025). g-Distance: On the comparison of model and human
heterogeneity. *Psychological Review, 132*\\(3), 632-655. https://doi.org/10.1037/rev0000550

>>> import torch
>>> from psp import clobe
>>> clobe.imac(torch.tensor([0.9, 0.5, 0.45]), (0.1, -0.1))
tensor([[2., 1., 1.],
        [2., 2., 0.],
        [2., 2., 2.]])
"""

from ._utils import TOLERANCE
from .brutes import BrutesResult, brutes
from .combinatorics import fubini, semiorders, stirling2
from .gdistance import GDistanceResult, gdistance
from .imac import imac
from .orderings import OrderingsResult, orderings
from .pattern_distance import pattern_distance
from .patterns import PatternsResult, patterns

__all__ = [
    "imac",
    "patterns",
    "gdistance",
    "pattern_distance",
    "orderings",
    "brutes",
    "stirling2",
    "fubini",
    "semiorders",
    "PatternsResult",
    "GDistanceResult",
    "OrderingsResult",
    "BrutesResult",
    "TOLERANCE",
]
