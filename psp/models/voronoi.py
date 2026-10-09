"""A cheap toy model whose ordinal regions are Voronoi cells of the unit hypercube.

The model returns the squared distance from the parameter set to each of
``regions`` random centres, and ``discretize`` returns the index of the nearest
centre in a 2 x 2 matrix. It is the model of the package tests and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["VoronoiModel", "voronoi_model", "voronoi_control"]


@dataclass(frozen=True)
class VoronoiModel:
    """A Voronoi model: ``centres`` of shape ``(regions, dimensions)``."""

    centres: torch.Tensor

    @property
    def regions(self) -> int:
        return int(self.centres.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self.centres.shape[1])

    def model(self, parameters: torch.Tensor) -> torch.Tensor:
        """Squared distance to every centre, shape ``(n, regions)``."""
        centres = self.centres.to(parameters.device, parameters.dtype)
        return ((parameters.unsqueeze(1) - centres.unsqueeze(0)) ** 2).sum(dim=2)

    @staticmethod
    def discretize(distances: torch.Tensor) -> torch.Tensor:
        """Index of the nearest centre, as a 2 x 2 matrix per row."""
        nearest = distances.argmin(dim=1).to(distances.dtype)
        return nearest.reshape(-1, 1, 1).expand(-1, 2, 2)


def voronoi_model(regions: int, dimensions: int, seed: int = 7624) -> VoronoiModel:
    """A Voronoi model with ``regions`` centres drawn uniformly in the unit hypercube."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.rand((regions, dimensions), generator=generator, dtype=torch.float64)
    return VoronoiModel(centres)


def voronoi_control(
    regions: int,
    dimensions: int,
    iterations: int,
    population: int,
    radius: float = 0.3,
    init: torch.Tensor | None = None,
) -> dict[str, Any]:
    """The control settings used by the tests and benchmarks for a Voronoi model."""
    if init is None:
        init = torch.full((1, dimensions), 0.5, dtype=torch.float64)
    return {
        "iterations": iterations,
        "population": population,
        "radius": radius,
        "lower": [0.0] * dimensions,
        "upper": [1.0] * dimensions,
        "init": init,
        "parameter_names": [f"p{i + 1}" for i in range(dimensions)],
        "stimuli_names": [f"s{i + 1}" for i in range(regions)],
        "dimensionality": 2,
        "responses": regions,
    }
