"""Benchmarks pspGlobal on the fixed set of workloads of the R package's ``benchmarks/benchmark.R``.

Usage::

    uv run python benchmarks/benchmark.py [--label LABEL] [--device cpu|cuda] [--rowwise]

Prints one csv row per run with the same columns as the R script, so the two can
be compared row for row: ``build, workload, seed, seconds, iterations, patterns,
evaluations, filled``. ``--rowwise`` calls the model once per parameter set, as
the R package does, instead of once per batch.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch

import psp
from psp import pspGlobal
from psp.models.voronoi import voronoi_control, voronoi_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_psp import control as test_control  # noqa: E402
from tests.test_psp import discretize as test_discretize  # noqa: E402
from tests.test_psp import model as test_model  # noqa: E402


def run(
    label: str,
    workload: str,
    model: Callable,
    discretize: Callable,
    control: dict,
    seeds: range,
    *,
    device: torch.device,
    rowwise: bool,
    save: bool = False,
) -> None:
    path = Path(tempfile.mkdtemp()) / "psp_benchmark"
    if rowwise:
        model, discretize = _per_row(model), _per_row(discretize)
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = pspGlobal(
            model,
            discretize,
            control,
            save=save,
            path=path,
            quiet=True,
            device=device,
            generator=generator,
            batched=not rowwise,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        filled = int(bool((out.ordinal_counts >= control["population"]).all()))
        print(
            f'"{label}","{workload}",{seed},{seconds:.4f},{out.iterations},{out.patterns},'
            f"{out.evaluations},{filled}",
            flush=True,
        )


def _per_row(fn: Callable) -> Callable:
    """A batched function called one row at a time, like the R package calls its callbacks."""
    return lambda row: fn(row.unsqueeze(0))[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default=None, help='name of this build in the "build" column')
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rowwise", action="store_true", help="call model and discretize once per parameter set"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    label = (
        args.label
        or f"psp-torch {psp.__version__} {device.type}{' rowwise' if args.rowwise else ''}"
    )
    options = {"device": device, "rowwise": args.rowwise}

    print('"build","workload","seed","seconds","iterations","patterns","evaluations","filled"')
    run(
        label,
        "test model, pop 10",
        test_model,
        test_discretize,
        test_control(1000, 10),
        range(1, 11),
        **options,
    )
    v = voronoi_model(100, 5)
    run(
        label,
        "100 regions, pop 20, cap 1500",
        v.model,
        v.discretize,
        voronoi_control(100, 5, 1500, 20),
        range(1, 6),
        **options,
    )
    v = voronoi_model(300, 5)
    run(
        label,
        "300 regions, 100 iterations",
        v.model,
        v.discretize,
        voronoi_control(300, 5, 100, 10**6),
        range(1, 4),
        **options,
    )
    run(
        label,
        "300 regions, 100 iterations, save",
        v.model,
        v.discretize,
        voronoi_control(300, 5, 100, 10**6),
        range(1, 4),
        save=True,
        **options,
    )
    v = voronoi_model(1000, 5)
    run(
        label,
        "1000 regions, 100 iterations",
        v.model,
        v.discretize,
        voronoi_control(1000, 5, 100, 10**6),
        range(1, 4),
        **options,
    )


if __name__ == "__main__":
    main()
