"""The experiments of the R package's ``benchmarks/experiments.R``, for the torch port.

Usage::

    uv run python benchmarks/experiments.py --label LABEL \
        [--experiments overhead,save,discovery] [--device cpu]

Rows are appended to ``overhead.csv``, ``save.csv`` and ``discovery.csv`` in the
output directory (default ``benchmarks/results``), with the columns of the R script.

- overhead: a near-free model, so any time not spent calling it is PSP's.
- save: the extra cost of ``save=True`` with 300 model outputs per evaluation.
- discovery: regions found by 30 runs on a 50-region model.
- settings: regions found on the same model with other populations, radii and starting
  points, 30 seeds each (the settings table of the R package's benchmarks article).
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path

import torch

import psp
from psp import PSP
from psp.models.voronoi import voronoi_control, voronoi_model

RESULTS = Path(__file__).resolve().parent / "results"


def emit(out_dir: Path, name: str, build: str, row: dict) -> None:
    path = out_dir / f"{name}.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["build", *row], lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({"build": build, **row})


def seconds(fn, device: torch.device) -> tuple[float, object]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--experiments", default="overhead,save,discovery,settings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(RESULTS))
    args = parser.parse_args()
    device = torch.device(args.device)
    build = args.label or f"psp-torch {psp.__version__} {device.type}"
    experiments = args.experiments.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "overhead" in experiments:
        d = 5
        w = (4.0 ** torch.arange(d, dtype=torch.float64)).to(device)

        def model(par: torch.Tensor) -> torch.Tensor:
            return par

        def discretize(x: torch.Tensor) -> torch.Tensor:
            cell = (torch.floor(x * 3.999) * w).sum(dim=1)
            return cell.reshape(-1, 1, 1).expand(-1, 2, 2)

        ctrl = {
            "iterations": 150,
            "population": 10**9,
            "radius": 0.3,
            "lower": [0.0] * d,
            "upper": [1.0] * d,
            "init": torch.full((1, d), 0.5, dtype=torch.float64),
            "parameter_names": [f"p{i + 1}" for i in range(d)],
            "stimuli_names": [f"s{i + 1}" for i in range(d)],
            "dimensionality": 2,
            "responses": d,
        }
        for rep in range(1, 4):
            generator = torch.Generator(device=device).manual_seed(rep)
            total, out = seconds(
                lambda generator=generator: PSP(
                    model, discretize, ctrl, quiet=True, device=device, generator=generator
                ),
                device,
            )
            n = out.evaluations
            # the same number of calls to model and discretize from a plain loop, one row at a time
            points = torch.rand((n, d), dtype=torch.float64, device=device)
            calls, _ = seconds(
                lambda points=points, n=n: [discretize(model(points[i : i + 1])) for i in range(n)],
                device,
            )
            # and in one batch, which is how PSP calls them
            batched, _ = seconds(lambda points=points: discretize(model(points)), device)
            emit(
                out_dir,
                "overhead",
                build,
                {
                    "rep": rep,
                    "evaluations": n,
                    "patterns": out.patterns,
                    "total_us": 1e6 * total / n,
                    "callbacks_us": 1e6 * calls / n,
                    "framework_us": 1e6 * (total - calls) / n,
                    "batched_us": 1e6 * batched / n,
                },
            )

    if "save" in experiments:
        v = voronoi_model(300, 5)
        ctrl = voronoi_control(300, 5, 100, 10**6)
        path = Path(tempfile.mkdtemp()) / "psp_experiments_save"
        for seed in range(1, 4):
            generator = torch.Generator(device=device).manual_seed(seed)
            with_save, out = seconds(
                lambda generator=generator: PSP(
                    v.model,
                    v.discretize,
                    ctrl,
                    save=True,
                    path=path,
                    quiet=True,
                    device=device,
                    generator=generator,
                ),
                device,
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            without, _ = seconds(
                lambda generator=generator: PSP(
                    v.model, v.discretize, ctrl, quiet=True, device=device, generator=generator
                ),
                device,
            )
            n = out.evaluations
            files = psp.saved_paths(path)
            emit(
                out_dir,
                "save",
                build,
                {
                    "seed": seed,
                    "evaluations": n,
                    "save_us": 1e6 * (with_save - without) / n,
                    "total_us": 1e6 * with_save / n,
                    "mb": sum(f.stat().st_size for f in files) / 1e6,
                },
            )

    if "discovery" in experiments:
        v = voronoi_model(50, 3)
        ctrl = voronoi_control(50, 3, 500, 10)
        for seed in range(1, 31):
            generator = torch.Generator(device=device).manual_seed(seed)
            elapsed, out = seconds(
                lambda generator=generator: PSP(
                    v.model, v.discretize, ctrl, quiet=True, device=device, generator=generator
                ),
                device,
            )
            emit(
                out_dir,
                "discovery",
                build,
                {
                    "seed": seed,
                    "seconds": elapsed,
                    "iterations": out.iterations,
                    "regions": out.patterns,
                    "evaluations": out.evaluations,
                    "filled": bool((out.ordinal_counts >= 10).all()),
                },
            )

    if "settings" in experiments:
        v = voronoi_model(50, 3)
        settings = {
            "1 start, population 10, radius 0.3": dict(starts=1, population=10, radius=0.3, runs=1),
            "10 random starts, population 10": dict(starts=10, population=10, radius=0.3, runs=1),
            "1 start, population 30": dict(starts=1, population=30, radius=0.3, runs=1),
            "1 start, population 10, radius 0.5": dict(starts=1, population=10, radius=0.5, runs=1),
            "10 random starts, population 30": dict(starts=10, population=30, radius=0.3, runs=1),
            "3 pooled runs of 10 starts, population 10": dict(
                starts=10, population=10, radius=0.3, runs=3
            ),
        }
        for name, setting in settings.items():
            for seed in range(1, 31):
                found: set[bytes] = set()
                evaluations = 0
                filled = True
                for run in range(setting["runs"]):
                    generator = torch.Generator(device=device).manual_seed(seed + 100 * run)
                    init = (
                        torch.full((1, 3), 0.5, dtype=torch.float64)
                        if setting["starts"] == 1
                        else torch.rand(
                            (setting["starts"], 3),
                            generator=torch.Generator().manual_seed(seed + 100 * run),
                            dtype=torch.float64,
                        )
                    )
                    ctrl = voronoi_control(
                        50, 3, 500, setting["population"], setting["radius"], init
                    )
                    out = PSP(
                        v.model, v.discretize, ctrl, quiet=True, device=device, generator=generator
                    )
                    found.update(p.numpy().tobytes() for p in out.ordinal_patterns)
                    evaluations += out.evaluations
                    filled &= bool((out.ordinal_counts >= setting["population"]).all())
                emit(
                    out_dir,
                    "settings",
                    build,
                    {
                        "setting": name,
                        "seed": seed,
                        "regions": len(found),
                        "evaluations": evaluations,
                        "filled": filled,
                    },
                )


if __name__ == "__main__":
    main()
