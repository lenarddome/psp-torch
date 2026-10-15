"""Seconds per call of :mod:`psp.clobe` on the workloads of ``r/bench.R``, with the same inputs.

Usage::

    uv run python benchmarks/clobe/bench.py <inputs directory> <output csv> [--devices cpu,cuda]

Every function runs on the CPU in float64; the ones that take tensors also run on the GPU,
with their inputs already on the device. Results stay on the device; the clock stops after
``torch.cuda.synchronize()``.
"""

# record() calls each lambda before the loop moves on, so late binding is not a problem here
# ruff: noqa: B023

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from psp import clobe

THRESHOLDS = (0.1, -0.1)
GPU_FUNCTIONS = {"imac (every row)", "patterns", "gdistance", "pattern_distance", "brutes"}


def upper_sum(matrices: torch.Tensor) -> float:
    k = matrices.shape[-1]
    i, j = torch.triu_indices(k, k, offset=1)
    return float(matrices[..., i, j].sum())


def measure(fn: Callable[[], object], sync: Callable[[], None]) -> tuple[float, int, float]:
    """Median seconds per call: one warm-up call, then samples of at least 0.05 s each, up to
    5 samples or 30 s, as ``r/bench.R`` measures."""
    start = time.perf_counter()
    check = fn()
    sync()
    once = max(time.perf_counter() - start, 1e-6)
    inner = max(1, int(np.ceil(0.05 / once)))
    samples: list[float] = []
    total = 0.0
    while len(samples) < 5 and (not samples or total < 30):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        elapsed = time.perf_counter() - start
        samples.append(elapsed / inner)
        total += elapsed
    return statistics.median(samples), len(samples), float(check)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--devices", default="cpu,cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    def column(name: str) -> torch.Tensor:
        return torch.tensor(np.loadtxt(args.inputs / name, delimiter=",", skiprows=1))

    data = column("data.csv")
    human_idx = column("human_idx.csv").long()
    model_idx = column("model_idx.csv").long()
    x_idx = column("x_idx.csv").long()
    six = clobe.orderings(6).inequality_matrices.double()
    universal = clobe.semiorders(6)

    rows = []
    for device in args.devices.split(","):
        build = f"psp-torch {device} float64"
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

        def record(function: str, size: int, fn: Callable[[], object]) -> None:
            if device != "cpu" and function not in GPU_FUNCTIONS:
                return
            print(f"{build}: {function} {size} ... ", end="", flush=True)
            seconds, samples, check = measure(fn, sync)
            print(f"{seconds:.6g} s", flush=True)
            rows.append(
                {
                    "function": function,
                    "size": size,
                    "build": build,
                    "seconds": f"{seconds:.6g}",
                    "samples": samples,
                    "check": f"{check:.15g}",
                }
            )

        values = data.to(device)
        record("imac (one vector)", 1, lambda: upper_sum(clobe.imac(values[0], THRESHOLDS)))
        for n in (1_000, 10_000, 100_000):
            record("imac (every row)", n, lambda: upper_sum(clobe.imac(values[:n], THRESHOLDS)))
            record("patterns", n, lambda: clobe.patterns(values[:n], THRESHOLDS).n)
        patterns = six.to(device)
        human = patterns[human_idx.to(device)]
        frequencies = torch.full(
            (len(human_idx),), 1 / len(human_idx), dtype=torch.float64, device=device
        )
        for m in (1_000, 10_000, len(model_idx)):
            model = patterns[model_idx[:m].to(device)]
            record(
                "gdistance",
                m,
                lambda: clobe.gdistance(human, model, universal, 0.5, frequencies).gdistance,
            )
        for n in (100, 1_000, 5_000):
            x = patterns[x_idx[:n].to(device)]
            record("pattern_distance", n, lambda: float(clobe.pattern_distance(x).sum()))
        for k in (5, 6, 7):  # R needs 18 GB for the 17.6 million semiorders of 8
            record("orderings (semiorder)", k, lambda: clobe.orderings(k).n)
        for k in (7, 8, 9):
            record("orderings (weak)", k, lambda: clobe.orderings(k, "weak").n)
        grid = torch.arange(11, dtype=torch.float64) / 10
        for k in (5, 6, 7, 8):
            record(
                "brutes",
                k,
                lambda: clobe.brutes(grid, k, THRESHOLDS, return_matrices=k < 8, device=device).n,
            )

    with open(args.out, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
