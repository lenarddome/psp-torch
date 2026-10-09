"""Shared fixtures: the toy models of the R package's test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from psp import read_saved
from psp.models.voronoi import voronoi_control, voronoi_model

DATA = Path(__file__).parent / "data"

__all__ = ["DATA", "voronoi_model", "voronoi_control", "read_saved"]


@pytest.fixture
def tmp_prefix(tmp_path: Path) -> str:
    """A path prefix for the files written with ``save=True``."""
    return str(tmp_path / "psp")


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Every available device; CUDA is skipped when it is not available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(request.param)
