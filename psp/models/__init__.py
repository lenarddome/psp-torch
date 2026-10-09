"""Models used by the tutorials, recipes, tests and benchmarks.

- :mod:`psp.models.prospect`: prospect theory on four bets of Kahneman and Tversky (1979).
- :mod:`psp.models.alcove`: ALCOVE (Kruschke, 1992) on the six category structures of
  Shepard, Hovland and Jenkins (1961), batched over parameter sets.
- :mod:`psp.models.oscar`: OSCAR on the inverse base-rate effect data (dome2025ibre), one
  trial order per participant.
- :mod:`psp.models.voronoi`: a cheap toy model whose regions are Voronoi cells.
"""

from . import alcove, oscar, prospect, voronoi

__all__ = ["alcove", "oscar", "prospect", "voronoi"]
