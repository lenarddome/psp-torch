"""The model of tests/testthat/test_psp_cpp.R: 100 regions in a 5-parameter space."""

from __future__ import annotations

import pytest
import torch

from psp import PSPResult, pspGlobal


def _positions() -> torch.Tensor:
    # the R test samples 100 of 500 grid positions per parameter with set.seed(7624);
    # any fixed sample gives a model with 100 regions
    generator = torch.Generator().manual_seed(7624)
    return torch.stack([torch.randperm(500, generator=generator)[:100] for _ in range(5)], dim=1)


POSITIONS = _positions()


def _areas() -> torch.Tensor:
    columns = []
    for i in range(5):
        grid = torch.linspace(1, 0, 500, dtype=torch.float64)
        if (i + 1) % 2 == 0:
            grid = torch.linspace(0, 1, 500, dtype=torch.float64)
        columns.append(grid[POSITIONS[:, i]])
    return torch.stack(columns, dim=1)  # (100, 5)


AREAS = _areas()


def model(par: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from each parameter set to the 100 region centres."""
    return ((AREAS.to(par.device, par.dtype).unsqueeze(0) - par.unsqueeze(1)) ** 2).sum(2).sqrt()


def discretize(distances: torch.Tensor) -> torch.Tensor:
    nearest = distances.argmin(dim=1).to(distances.dtype)
    return nearest.reshape(-1, 1, 1).expand(-1, 2, 2)


def control(iterations: int, population: int) -> dict:
    return {
        "iterations": iterations,
        "population": population,
        "radius": 1,
        "lower": [0] * 5,
        "upper": [1] * 5,
        "init": torch.linspace(0, 1, 10, dtype=torch.float64).reshape(10, 1).expand(10, 5).clone(),
        "parameter_names": ["a", "d", "z", "y", "x"],
        "stimuli_names": [str(i + 1) for i in range(100)],
        "dimensionality": 2,
        "responses": 100,
    }


@pytest.fixture(scope="module")
def out2() -> PSPResult:
    torch.manual_seed(1)
    return pspGlobal(model, discretize, control(1000, 10), quiet=True, save=False)


def test_finds_all_100_regions(out2: PSPResult) -> None:
    assert out2.ordinal_patterns.shape[0] == 100
    assert out2["ordinal_patterns"].shape == (100, 2, 2)


def test_population_terminates_before_iteration_cap(out2: PSPResult) -> None:
    assert out2.iterations < 1000


def test_unique_inequality_matrices_recruited_once(out2: PSPResult) -> None:
    flat = out2.ordinal_patterns.reshape(100, -1)
    assert torch.unique(flat, dim=0).shape[0] == 100
    assert len(out2.ordinal_counts) == 100


def test_iteration_threshold_terminates(out2: PSPResult) -> None:
    torch.manual_seed(2)
    out3 = pspGlobal(model, discretize, control(100, 30000), quiet=True)
    assert out3.iterations == 100


def test_result_behaves_like_the_r_list(out2: PSPResult) -> None:
    assert list(out2) == ["ordinal_patterns", "ordinal_counts", "iterations"]
    assert dict(out2)["iterations"] == out2.iterations
    assert out2.evaluations == int(out2.ordinal_counts.sum())
    assert out2.patterns == 100
    with pytest.raises(KeyError):
        out2["nothing"]
    assert "patterns=100" in repr(out2)


def test_rowwise_contract_matches_the_r_package() -> None:
    """model and discretize written for one parameter set, as in R."""
    torch.manual_seed(3)
    out = pspGlobal(
        lambda par: model(par.unsqueeze(0))[0],
        lambda d: torch.full((2, 2), float(d.argmin())),
        control(1000, 10),
        quiet=True,
        batched=False,
    )
    assert out.patterns == 100
    assert bool((out.ordinal_counts >= 10).all())
