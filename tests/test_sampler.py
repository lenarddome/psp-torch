"""Behavioural guarantees of the sampler: a port of tests/testthat/test_psp_cpp_sampler.R."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from psp import PSP, Control, hyper_points, read_saved, saved_paths

from .conftest import voronoi_control, voronoi_model


def test_every_evaluation_is_counted_exactly_once() -> None:
    torch.manual_seed(1)
    init = torch.rand(3, 3, dtype=torch.float64)
    out = PSP(
        lambda par: par,
        lambda x: torch.ones(x.shape[0], 2, 2),
        {
            "iterations": 5,
            "population": 1000,
            "radius": 0.1,
            "lower": [0] * 3,
            "upper": [1] * 3,
            "init": init,
            "parameter_names": ["a", "b", "c"],
            "stimuli_names": ["a", "b", "c"],
            "dimensionality": 2,
            "responses": 3,
        },
        quiet=True,
    )
    # 3 starting points + 1 proposal for the single pattern in each of 5 iterations
    assert out.ordinal_counts.tolist() == [8]


def test_first_iteration_proposes_from_every_pattern_found_at_the_start(tmp_prefix: str) -> None:
    vor = voronoi_model(regions=50, dimensions=3)
    torch.manual_seed(2)
    init = torch.rand(10, 3, dtype=torch.float64)
    PSP(
        vor.model,
        vor.discretize,
        voronoi_control(50, 3, iterations=1, population=1000, init=init),
        save=True,
        path=tmp_prefix,
        quiet=True,
    )
    saved = read_saved(tmp_prefix).parameters
    start_patterns = saved.loc[saved.iteration == 0, "pattern"].nunique()
    assert (saved.iteration == 1).sum() == start_patterns


def test_each_proposal_is_within_radius_of_the_last_point_in_its_region(tmp_prefix: str) -> None:
    radius = 0.1
    vor = voronoi_model(regions=30, dimensions=3)
    torch.manual_seed(3)
    PSP(
        vor.model,
        vor.discretize,
        voronoi_control(30, 3, iterations=30, population=15, radius=radius),
        save=True,
        path=tmp_prefix,
        quiet=True,
    )
    saved = read_saved(tmp_prefix).parameters
    params = saved[["p1", "p2", "p3"]].to_numpy()
    patterns = saved["pattern"].to_numpy()
    iteration = saved["iteration"].to_numpy()
    # replay the chain: proposals in iteration t come from the underpopulated patterns
    # after iteration t - 1, in pattern order, each centred on the last point of that pattern
    for t in range(1, int(iteration.max()) + 1):
        history = iteration < t
        counts = np.bincount(patterns[history], minlength=patterns.max() + 1)
        sources = np.flatnonzero((counts > 0) & (counts < 15))
        proposals = np.flatnonzero(iteration == t)
        assert len(proposals) == len(sources)
        centres = np.stack([params[np.flatnonzero(history & (patterns == k))[-1]] for k in sources])
        assert np.abs(params[proposals] - centres).max() <= radius + 1e-9


def test_sampler_fills_every_discovered_region_before_the_iteration_cap() -> None:
    vor = voronoi_model(regions=50, dimensions=3)
    torch.manual_seed(4)
    out = PSP(
        vor.model, vor.discretize, voronoi_control(50, 3, iterations=500, population=10), quiet=True
    )
    assert out.iterations < 500
    assert bool((out.ordinal_counts >= 10).all())


def test_saved_files_record_full_precision_values_and_correct_pattern_ids(tmp_prefix: str) -> None:
    vor = voronoi_model(regions=20, dimensions=3)
    torch.manual_seed(5)
    out = PSP(
        vor.model,
        vor.discretize,
        voronoi_control(20, 3, iterations=10, population=1000),
        save=True,
        path=tmp_prefix,
        quiet=True,
    )
    saved = read_saved(tmp_prefix)
    assert list(saved.parameters.columns) == ["iteration", "p1", "p2", "p3", "pattern"]
    assert list(saved.continuous.columns) == [
        "iteration",
        *[f"s{i + 1}" for i in range(20)],
        "pattern",
    ]
    params = torch.tensor(saved.parameters[["p1", "p2", "p3"]].to_numpy())
    continuous = torch.tensor(saved.continuous[[f"s{i + 1}" for i in range(20)]].to_numpy())
    assert params.shape[0] == out.evaluations
    assert saved.parameters["pattern"].tolist() == saved.continuous["pattern"].tolist()
    assert torch.equal(continuous, vor.model(params))  # exact: shortest round-trip text
    patterns = vor.discretize(vor.model(params))
    for i, pattern in enumerate(saved.parameters["pattern"]):
        assert torch.equal(patterns[i], out.ordinal_patterns[pattern])


def test_saved_parameters_and_model_outputs_keep_full_double_precision(tmp_prefix: str) -> None:
    eps = np.finfo(np.float64).eps
    values = torch.tensor([1 + 20 * eps, 1 / 3, math.pi * 1e5], dtype=torch.float64)
    PSP(
        lambda par: values.unsqueeze(0).expand(par.shape[0], 3),
        lambda x: torch.ones(x.shape[0], 2, 2),
        {
            "iterations": 1,
            "population": 1000,
            "radius": 0.1,
            "lower": [0] * 3,
            "upper": [2] * 3,
            "init": (values / 1e5).reshape(1, 3),
            "parameter_names": ["a", "b", "c"],
            "stimuli_names": ["a", "b", "c"],
            "dimensionality": 2,
            "responses": 3,
        },
        save=True,
        path=tmp_prefix,
        quiet=True,
    )
    saved = read_saved(tmp_prefix)
    assert np.array_equal(
        saved.parameters.loc[0, ["a", "b", "c"]].to_numpy(dtype=np.float64), (values / 1e5).numpy()
    )
    for row in saved.continuous[["a", "b", "c"]].to_numpy(dtype=np.float64):
        assert np.array_equal(row, values.numpy())


def test_saved_paths_follow_the_r_naming(tmp_path) -> None:
    parameters, continuous = saved_paths(tmp_path / "run", ".tsv")
    assert parameters.name == "run_parameters.tsv"
    assert continuous.name == "run_continuous.tsv"


def test_stored_ordinal_patterns_are_unique_and_counted() -> None:
    vor = voronoi_model(regions=40, dimensions=3)
    torch.manual_seed(6)
    out = PSP(
        vor.model,
        vor.discretize,
        voronoi_control(40, 3, iterations=20, population=1000),
        quiet=True,
    )
    flat = out.ordinal_patterns.reshape(out.patterns, -1)
    assert torch.unique(flat, dim=0).shape[0] == out.patterns
    assert len(out.ordinal_counts) == out.patterns


def test_negative_zero_and_zero_are_the_same_ordinal_pattern() -> None:
    out = PSP(
        lambda par: par,
        lambda x: torch.where(x[:, :1] > 0.5, -0.0, 0.0).reshape(-1, 1, 1).expand(-1, 2, 2),
        {
            "iterations": 10,
            "population": 1000,
            "radius": 0.5,
            "lower": [0],
            "upper": [1],
            "init": [[0.2], [0.8]],
            "parameter_names": ["a"],
            "stimuli_names": ["a"],
            "dimensionality": 2,
            "responses": 1,
        },
        quiet=True,
    )
    assert out.patterns == 1


def test_identical_patterns_containing_nan_are_the_same_ordinal_pattern() -> None:
    nan = float("nan")
    out = PSP(
        lambda par: par,
        lambda x: torch.tensor([[1.0, nan], [nan, 1.0]]).expand(x.shape[0], 2, 2),
        {
            "iterations": 5,
            "population": 1000,
            "radius": 0.1,
            "lower": [0],
            "upper": [1],
            "init": [[0.5]],
            "parameter_names": ["a"],
            "stimuli_names": ["a"],
            "dimensionality": 2,
            "responses": 1,
        },
        quiet=True,
    )
    assert out.patterns == 1
    assert out.ordinal_counts.tolist() == [6]


def test_hyper_points_stay_within_radius_in_every_dimension() -> None:
    generator = torch.Generator().manual_seed(0)
    radius = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
    points = hyper_points(10000, radius, generator=generator)
    assert points.shape == (10000, 3)
    assert bool((points.abs() <= radius).all())
    # the scale is shared within a call: the largest point in each dimension reaches it
    assert bool(((points.abs().max(dim=0).values / radius) > 0.05).all())


def test_generator_makes_runs_reproducible() -> None:
    vor = voronoi_model(regions=30, dimensions=3)
    runs = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(11)
        runs.append(
            PSP(
                vor.model,
                vor.discretize,
                voronoi_control(30, 3, 20, 5),
                quiet=True,
                generator=generator,
            )
        )
    assert torch.equal(runs[0].ordinal_counts, runs[1].ordinal_counts)
    assert torch.equal(runs[0].ordinal_patterns, runs[1].ordinal_patterns)


def test_control_round_trips_through_the_dataclass() -> None:
    ctrl = Control.from_mapping(voronoi_control(5, 2, iterations=0, population=5))
    assert ctrl.iterations_unbounded and not ctrl.population_unbounded
    assert ctrl.dimensions == 2
    assert ctrl.radius.tolist() == [0.3, 0.3]
    again = Control.from_mapping(ctrl.to_mapping())
    assert again.population == 5 and again.iterations_unbounded
    assert Control.from_mapping(ctrl) is ctrl


def test_runs_on_every_device(device: torch.device) -> None:
    vor = voronoi_model(regions=20, dimensions=3)
    ctrl = voronoi_control(20, 3, iterations=200, population=5)
    generator = torch.Generator(device=device).manual_seed(0)
    out = PSP(
        vor.model,
        vor.discretize,
        ctrl,
        quiet=True,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    assert out.ordinal_patterns.device.type == "cpu"
    assert bool((out.ordinal_counts >= 5).all())


def test_progress_is_printed_unless_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    vor = voronoi_model(regions=5, dimensions=2)
    PSP(vor.model, vor.discretize, voronoi_control(5, 2, iterations=2, population=1000))
    assert capsys.readouterr().out == "Iteration: [1]\nIteration: [2]\n"
