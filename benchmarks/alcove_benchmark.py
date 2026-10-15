"""Partitions ALCOVE's full parameter space: time taken and regions found, per build.

Usage::

    uv run python benchmarks/alcove_benchmark.py [--device cpu] [--dtype float64] [--seeds 1,2,3] \
        [--machine NAME]
    uv run python benchmarks/alcove_benchmark.py --import-r <directory from r/alcove_benchmark.R>

The workload is the ALCOVE recipe on the full ranges of the learning rates
(``c`` 0-20, ``phi`` 0-6, ``lw`` and ``la`` 0-1, lower bounds just above 0 because
ALCOVE rejects a learning rate of exactly 0): 20 random starting points, radius
0.15, population 50, a cap of 1,000 iterations, 5 simulated learners per type
with catlearn's trial order for seed 7624, and the weak order of the six types
with a tie gap of 0.025 as the pattern.

Three files in ``benchmarks/results`` receive one build's rows (replacing that
build's earlier rows):

- ``alcove.csv``: ``build, seed, seconds, iterations, patterns, evaluations, filled``,
  one row per run.
- ``alcove_discovery.csv``: ``build, seed, iteration, evaluations, patterns``, the
  number of distinct patterns found after each iteration.
- ``alcove_patterns.csv``: ``build, seed, order, count``, every ordering a run found
  and how many parameter sets produced it.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from compare import RESULTS, machine_suffix, read_rows, replace_build

import psp
from psp import PSP, read_saved
from psp.models.alcove import (
    ALCOVE_PARAMETERS_FULL,
    alcove_curves,
    alcove_training,
    describe_order,
    weak_order,
)

ITERATIONS = 1000
POPULATION = 50
RADIUS = 0.15
STARTS = 20
GAP = 0.025
FILES = ("alcove", "alcove_discovery", "alcove_patterns")


def discovery_curve(iterations: np.ndarray, patterns: np.ndarray) -> list[dict]:
    """Distinct patterns found after each iteration, from the saved parameters table."""
    rows = []
    seen: set[int] = set()
    evaluations = 0
    for iteration in np.unique(iterations):
        mask = iterations == iteration
        seen.update(patterns[mask].tolist())
        evaluations += int(mask.sum())
        rows.append(
            {"iteration": int(iteration), "evaluations": evaluations, "patterns": len(seen)}
        )
    return rows


def run(
    device: torch.device, dtype: torch.dtype, seeds: list[int], label: str, iterations: int
) -> None:
    training = alcove_training(5, 7624).to(device, dtype)

    def model(unit: torch.Tensor) -> torch.Tensor:
        return alcove_curves(unit, training, parameters=ALCOVE_PARAMETERS_FULL)

    def discretize(curves: torch.Tensor) -> torch.Tensor:
        return weak_order(curves, gap=GAP)

    summary, discovery, found = [], [], []
    path = Path(tempfile.mkdtemp()) / "alcove_benchmark"
    for seed in seeds:
        init = torch.rand(
            (STARTS, 4), generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
        control = {
            "lower": [0.0] * 4,
            "upper": [1.0] * 4,
            "init": init,
            "radius": RADIUS,
            "population": POPULATION,
            "iterations": iterations,
            "parameter_names": ["c", "phi", "lw", "la"],
            "dimensionality": 6,
            "responses": 96,
        }
        generator = torch.Generator(device=device).manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = PSP(
            model,
            discretize,
            control,
            save=True,
            path=path,
            quiet=True,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = {
            "seed": seed,
            "seconds": round(seconds, 3),
            "iterations": out.iterations,
            "patterns": out.patterns,
            "evaluations": out.evaluations,
            "filled": bool((out.ordinal_counts >= POPULATION).all()),
        }
        summary.append(row)
        print(label, row, flush=True)
        saved = read_saved(path).parameters
        for r in discovery_curve(saved["iteration"].to_numpy(), saved["pattern"].to_numpy()):
            discovery.append({"seed": seed, **r})
        for pattern, count in zip(out.ordinal_patterns, out.ordinal_counts.tolist(), strict=True):
            found.append({"seed": seed, "order": describe_order(pattern), "count": count})
    for name, rows in zip(FILES, (summary, discovery, found), strict=True):
        replace_build(RESULTS / f"{name}.csv", label, [{"build": label, **r} for r in rows])


def import_r(directory: Path) -> None:
    for name in FILES:
        rows = read_rows(directory / f"{name}.csv")
        if not rows:
            raise SystemExit(f"{directory / name}.csv is missing or empty")
        for build in {r["build"] for r in rows}:
            replace_build(RESULTS / f"{name}.csv", build, [r for r in rows if r["build"] == build])
        print("imported", name, len(rows), "rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--machine", default=None, help="name of this machine, appended to the default label"
    )
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="for smoke tests only")
    parser.add_argument(
        "--import-r", default=None, help="merge the files r/alcove_benchmark.R wrote"
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.import_r:
        import_r(Path(args.import_r))
        return
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    label = args.label or (
        f"psp-torch {psp.__version__} {device.type} {args.dtype}" + machine_suffix(args.machine)
    )
    run(device, dtype, [int(s) for s in args.seeds.split(",")], label, args.iterations)


if __name__ == "__main__":
    main()
