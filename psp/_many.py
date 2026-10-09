"""Many independent parameter space partitioning runs, advanced in lockstep.

:func:`pspMany` runs one :func:`psp.pspGlobal` search per ``control`` (for example
one per participant, each on its own trial order) and keeps them independent:
each has its own pattern store, its own counts, its own files and its own
result. What they share is the model call: every iteration, the proposals of
all runs are concatenated and ``model`` is called once with the rows and the
index of the run each row belongs to. That is what lets a GPU work on hundreds
of small searches at once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ._callbacks import as_numeric_tensor
from ._control import Control
from ._global import PSPResult, clamp_parameters, hyper_points
from ._io import RunWriter
from ._store import PatternStore

__all__ = ["pspMany", "ManyModel"]

#: A model for :func:`pspMany`: ``(parameters (n, d), run (n,)) -> (n, responses)``.
ManyModel = Callable[[torch.Tensor, torch.Tensor], Any]


def pspMany(
    model: ManyModel,
    discretize: Callable[[torch.Tensor], Any],
    controls: Sequence[Mapping[str, Any] | Control],
    *,
    save: bool = False,
    paths: Sequence[str | Path] | None = None,
    extension: str = ".csv",
    quiet: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> list[PSPResult]:
    """Run one independent partitioning per control, sharing one model call per iteration.

    Parameters
    ----------
    model
        Called once per iteration with the proposals of every run that is still
        going, ``(n, dimensions)``, and ``run``, an ``(n,)`` int64 tensor giving the
        index in ``controls`` of the run each row belongs to; returns ``(n, responses)``.
    discretize
        As in :func:`psp.pspGlobal`: ``(n, responses)`` to ``(n, D, D)``.
    controls
        One control mapping per run, with the keys of :func:`psp.pspGlobal`. Runs may
        differ in ``init``, bounds and ``radius``; they must share ``iterations`` and
        ``population`` and the number of parameters.
    save, paths, extension
        Write every evaluation of run ``r`` to ``paths[r]`` as :func:`psp.pspGlobal` does.
    quiet
        Do not print the iteration number, the number of runs still going and the
        number of rows evaluated.
    device, dtype, generator
        As in :func:`psp.pspGlobal`.

    Returns
    -------
    list[PSPResult]
        One result per control, in order. A run's ``iterations`` is the iteration at
        which it stopped proposing, or the cap.
    """
    if not controls:
        raise ValueError("controls must hold at least one run")
    settings = [Control.from_mapping(c, device=device, dtype=dtype) for c in controls]
    first = settings[0]
    for k, s in enumerate(settings[1:], 1):
        if s.dimensions != first.dimensions:
            raise ValueError(f"run {k} has {s.dimensions} parameters, run 0 has {first.dimensions}")
        if s.iterations != first.iterations or s.population != first.population:
            raise ValueError("every run must share the same iterations and population")
    if save and (paths is None or len(paths) != len(settings)):
        raise ValueError("save=True needs one path per run")
    run_device, run_dtype = first.init.device, first.init.dtype
    responses = first.responses
    dimensionality = first.dimensionality
    stores = [PatternStore(s.dimensions, run_device, run_dtype) for s in settings]
    lower = torch.stack([s.lower for s in settings])  # (runs, d)
    upper = torch.stack([s.upper for s in settings])

    def evaluate(jumping: torch.Tensor, run: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal responses, dimensionality
        rows = int(jumping.shape[0])
        continuous = as_numeric_tensor(model(jumping, run), "model", run_device)
        if continuous.ndim != 2 or continuous.shape[0] != rows:
            raise ValueError(
                f"model must return one row per parameter set, ({rows}, responses), "
                f"but returned shape {tuple(continuous.shape)}"
            )
        if responses is None:
            responses = int(continuous.shape[1])
        elif continuous.shape[1] != responses:
            raise ValueError(
                f"model returned {continuous.shape[1]} values, but control['responses'] "
                f"is {responses}"
            )
        continuous = continuous.to(run_dtype)
        ordinal = as_numeric_tensor(discretize(continuous), "discretize", run_device)
        if ordinal.ndim != 3 or ordinal.shape[0] != rows:
            raise ValueError(
                f"discretize must return a matrix per parameter set, ({rows}, D, D), "
                f"but returned shape {tuple(ordinal.shape)}"
            )
        if dimensionality is None:
            if ordinal.shape[1] != ordinal.shape[2]:
                raise ValueError("discretize must return square matrices")
            dimensionality = int(ordinal.shape[1])
        elif ordinal.shape[1] != dimensionality or ordinal.shape[2] != dimensionality:
            raise ValueError(
                f"discretize returned {ordinal.shape[1]} x {ordinal.shape[2]} matrices, but "
                f"control['dimensionality'] is {dimensionality}"
            )
        return continuous, ordinal

    writers: list[RunWriter | None] = [None] * len(settings)
    finished = [0] * len(settings)  # iteration at which each run stopped proposing
    active = list(range(len(settings)))
    iteration = 0
    try:
        # iteration 0: every run's starting points
        jumping = torch.cat([s.init for s in settings])
        run = torch.cat(
            [
                torch.full((int(s.init.shape[0]),), k, dtype=torch.int64, device=run_device)
                for k, s in enumerate(settings)
            ]
        )
        continuous, ordinal = evaluate(jumping, run)
        if save:
            assert paths is not None
            for k, s in enumerate(settings):
                names = s.stimuli_names or tuple(
                    f"stimulus_{i + 1}" for i in range(int(continuous.shape[1]))
                )
                writers[k] = RunWriter(paths[k], extension, s.parameter_names, names)
        _register(stores, writers, 0, jumping, continuous, ordinal, run, len(settings))
        active = [k for k in active if stores[k].underpopulated(first.population).numel() > 0]

        while active and iteration < first.iterations:
            iteration += 1
            pieces, runs = [], []
            for k in active:
                ids = stores[k].underpopulated(first.population)
                steps = hyper_points(int(ids.numel()), settings[k].radius, generator=generator)
                proposals = clamp_parameters(stores[k].centres_of(ids) + steps, lower[k], upper[k])
                pieces.append(proposals)
                runs.append(
                    torch.full((int(ids.numel()),), k, dtype=torch.int64, device=run_device)
                )
            jumping = torch.cat(pieces)
            run = torch.cat(runs)
            if not quiet:
                print(
                    f"Iteration: [{iteration}]  runs {len(active)}  rows {int(jumping.shape[0])}",
                    flush=True,
                )
            continuous, ordinal = evaluate(jumping, run)
            _register(stores, writers, iteration, jumping, continuous, ordinal, run, len(settings))
            still = []
            for k in active:
                if stores[k].underpopulated(first.population).numel() > 0:
                    still.append(k)
                else:
                    finished[k] = iteration
            active = still
        for k in active:
            finished[k] = iteration
    finally:
        for w in writers:
            if w is not None:
                w.close()
    return [
        PSPResult(
            ordinal_patterns=stores[k].patterns(),
            ordinal_counts=stores[k].counts.clone(),
            iterations=finished[k],
        )
        for k in range(len(settings))
    ]


def _register(
    stores: list[PatternStore],
    writers: list[RunWriter | None],
    iteration: int,
    jumping: torch.Tensor,
    continuous: torch.Tensor,
    ordinal: torch.Tensor,
    run: torch.Tensor,
    n_runs: int,
) -> None:
    """Hand each run its rows of one batched evaluation."""
    run_cpu = run.cpu()
    for k in range(n_runs):
        rows = torch.nonzero(run_cpu == k).reshape(-1)
        if rows.numel() == 0:
            continue
        rows_dev = rows.to(jumping.device)
        ids = stores[k].update(ordinal[rows_dev], jumping[rows_dev])
        writer = writers[k]
        if writer is not None:
            writer.write(iteration, jumping[rows_dev], continuous[rows_dev], ids)
