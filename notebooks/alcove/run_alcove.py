"""Runs the PSP analysis of ALCOVE that the ALCOVE recipe notebook reads.

A port of ``vignettes/articles/alcove/run-alcove.R`` from the R package. It takes
a while, so the notebook does not run it. Run it from the repository root::

    uv run python notebooks/alcove/run_alcove.py [--stages psp,uniform,...] [--device cuda]

By default it runs every stage. The stages write their results next to this file:

========  ===================  ===============================================================
psp       psp-run.csv.gz       every parameter set the main PSP run evaluated
uniform   uniform.csv.gz       parameter sets sampled evenly, for region sizes
learners  learners.csv         patterns with 5 and with 20 simulated learners (needs uniform)
orders    trial-orders.csv     patterns found by PSP with other trial orders
volume    volumes.csv          region volumes estimated as in Pitt et al. (2006) (needs psp)
========  ===================  ===============================================================

Each stage also records its settings and timing in ``run-info.csv``.
"""

from __future__ import annotations

import argparse
import gzip
import math
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from psp import pspGlobal, read_saved
from psp.models.alcove import (
    ALCOVE_PARAMETERS,
    TYPES,
    alcove_curves,
    alcove_training,
    describe_order,
    to_alcove,
    weak_order,
)

OUT_DIR = Path(__file__).resolve().parent
ALL_STAGES = ("psp", "uniform", "learners", "orders", "volume")
GAP = 0.025


def discretize(curves: torch.Tensor) -> torch.Tensor:
    return weak_order(curves, gap=GAP)


def control(init: torch.Tensor) -> dict:
    return {
        "lower": [0.0] * 4,
        "upper": [1.0] * 4,
        "init": init,
        "radius": 0.15,
        "population": 50,
        "iterations": 2000,
        "parameter_names": list(ALCOVE_PARAMETERS.name),
        "stimuli_names": [f"{t}_{b}" for t in TYPES for b in range(1, 17)],
        "dimensionality": 6,
        "responses": 96,
    }


def summarise_curves(unit: torch.Tensor, curves: torch.Tensor) -> pd.DataFrame:
    """Each evaluation reduced to its parameters, the mean error of each type and its weak order."""
    means = curves.reshape(curves.shape[0], 6, 16).mean(dim=2)
    values = to_alcove(unit)
    frame = pd.DataFrame({name: values[name].cpu().numpy() for name in ALCOVE_PARAMETERS.name})
    for k, t in enumerate(TYPES):
        frame[f"error_{t}"] = means[:, k].cpu().numpy()
    frame["order"] = [describe_order(p) for p in weak_order(curves, gap=GAP)]
    return frame


def write_gz(frame: pd.DataFrame, name: str) -> None:
    with gzip.open(OUT_DIR / name, "wt", encoding="utf-8", newline="") as text:
        frame.to_csv(text, index=False)


def record(stage: str, **values: object) -> None:
    path = OUT_DIR / "run-info.csv"
    row = pd.DataFrame([{"stage": stage, **values}])
    if path.exists():
        info = pd.read_csv(path)
        info = info[info["stage"] != stage]
        row = pd.concat([info, row], ignore_index=True)
    row.sort_values("stage").to_csv(path, index=False)
    print(stage + ":", ", ".join(f"{k} {v}" for k, v in values.items()), flush=True)


def in_ball(n: int, d: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """``n`` points drawn uniformly inside the unit ball in ``d`` dimensions."""
    direction = torch.randn((n, d), generator=generator, device=device, dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    return direction * torch.rand(
        (n, 1), generator=generator, device=device, dtype=torch.float64
    ) ** (1 / d)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages", default=",".join(ALL_STAGES), help="comma-separated subset of the stages"
    )
    parser.add_argument("--device", default="cpu", help="torch device to run ALCOVE on")
    args = parser.parse_args()
    stages = [s for s in args.stages.split(",") if s]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    device = torch.device(args.device)

    training = alcove_training(learners=5, seed=7624).to(device)

    def model(unit: torch.Tensor) -> torch.Tensor:
        return alcove_curves(unit, training)

    # the even sample: the same 4,000 points every time
    unit = torch.rand(
        (4000, 4), generator=torch.Generator().manual_seed(1), dtype=torch.float64
    ).to(device)

    if "psp" in stages:
        init = torch.rand(
            (20, 4), generator=torch.Generator().manual_seed(2026), dtype=torch.float64
        ).to(device)
        path = Path(tempfile.mkdtemp()) / "alcove"
        generator = torch.Generator(device=device).manual_seed(2026)
        start = time.perf_counter()
        result = pspGlobal(
            model, discretize, control(init), save=True, path=path, quiet=True, generator=generator
        )
        seconds = time.perf_counter() - start
        saved = read_saved(path)
        evaluated = torch.tensor(
            saved.parameters[list(ALCOVE_PARAMETERS.name)].to_numpy(), device=device
        )
        curves = torch.tensor(
            saved.continuous[control(init)["stimuli_names"]].to_numpy(), device=device
        )
        frame = summarise_curves(evaluated, curves)
        frame.insert(0, "iteration", saved.parameters["iteration"].to_numpy())
        write_gz(frame, "psp-run.csv.gz")
        record(
            "psp",
            seconds=round(seconds, 3),
            iterations=result.iterations,
            evaluations=result.evaluations,
            patterns=result.patterns,
        )

    if "uniform" in stages:
        start = time.perf_counter()
        curves = model(unit)
        seconds = time.perf_counter() - start
        write_gz(summarise_curves(unit, curves), "uniform.csv.gz")
        record("uniform", seconds=round(seconds, 3), evaluations=unit.shape[0])

    if "learners" in stages:
        # the first 500 points of the even sample again, with 20 simulated learners
        training20 = alcove_training(learners=20, seed=7624).to(device)
        start = time.perf_counter()
        curves20 = alcove_curves(unit[:500], training20)
        seconds = time.perf_counter() - start
        uniform = pd.read_csv(OUT_DIR / "uniform.csv.gz")
        pd.DataFrame(
            {
                "learners_5": uniform["order"][:500].to_numpy(),
                "learners_20": [describe_order(p) for p in weak_order(curves20, gap=GAP)],
            }
        ).to_csv(OUT_DIR / "learners.csv", index=False)
        record("learners", seconds=round(seconds, 3), evaluations=500)

    if "orders" in stages:
        # the main PSP run again, with four other trial orders
        start = time.perf_counter()
        rows = []
        for seed in (11, 22, 33, 44):
            other = alcove_training(learners=5, seed=seed).to(device)
            init = torch.rand(
                (20, 4), generator=torch.Generator().manual_seed(seed), dtype=torch.float64
            ).to(device)
            generator = torch.Generator(device=device).manual_seed(seed)
            out = pspGlobal(
                lambda u, tr=other: alcove_curves(u, tr),
                discretize,
                control(init),
                quiet=True,
                generator=generator,
            )
            for pattern, count in zip(
                out.ordinal_patterns, out.ordinal_counts.tolist(), strict=True
            ):
                rows.append(
                    {
                        "trial_order": seed,
                        "order": describe_order(pattern),
                        "parameter_sets": count,
                        "iterations": out.iterations,
                    }
                )
        pd.DataFrame(rows).to_csv(OUT_DIR / "trial-orders.csv", index=False)
        record("orders", seconds=round(time.perf_counter() - start, 3))

    if "volume" in stages:
        # Pitt et al. (2006): sample each region with its own Metropolis chain whose target
        # is uniform on the region (a step is accepted only if it produces the region's
        # pattern), fit an ellipsoid to the chain from its mean and covariance, then correct
        # the ellipsoid's volume by the share of points drawn uniformly inside it that
        # produce the pattern (hit-or-miss Monte Carlo; Appendix B). Volumes are on the unit
        # hypercube, so they are shares of the parameter space. The chains of all regions
        # advance in lockstep, so each step is one batched model evaluation.
        start = time.perf_counter()
        run = pd.read_csv(OUT_DIR / "psp-run.csv.gz")
        lower = torch.tensor(ALCOVE_PARAMETERS.lower, dtype=torch.float64)
        upper = torch.tensor(ALCOVE_PARAMETERS.upper, dtype=torch.float64)
        unit_run = (torch.tensor(run[list(ALCOVE_PARAMETERS.name)].to_numpy()) - lower) / (
            upper - lower
        )
        d = unit_run.shape[1]
        unit_ball = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
        orders = list(dict.fromkeys(run["order"]))
        n = len(orders)
        generator = torch.Generator(device=device).manual_seed(3)

        def patterns_of(points: torch.Tensor) -> list[str]:
            return [describe_order(p) for p in discretize(model(points))]

        # start each chain from the region's point nearest its median
        current = torch.empty((n, d), dtype=torch.float64)
        for k, o in enumerate(orders):
            points = unit_run[torch.tensor((run["order"] == o).to_numpy())]
            current[k] = points[((points - points.median(dim=0).values) ** 2).sum(dim=1).argmin()]
        current = current.to(device)
        burn_in, kept, draws = 200, 400, 200
        radius = torch.full((n, 1), 0.05, dtype=torch.float64, device=device)
        chain = torch.empty((burn_in + kept, n, d), dtype=torch.float64, device=device)
        accepted = torch.zeros((burn_in + kept, n), dtype=torch.bool)
        for step in range(burn_in + kept):
            proposal = current + radius * in_ball(n, d, generator, device)
            in_bounds = ((proposal >= 0) & (proposal <= 1)).all(dim=1).cpu()
            same = torch.tensor(
                [p == o for p, o in zip(patterns_of(proposal), orders, strict=True)]
            )
            accept = in_bounds & same
            current = torch.where(accept.to(device).unsqueeze(1), proposal, current)
            chain[step] = current
            accepted[step] = accept
            # during burn-in, adapt each radius towards 20-25% acceptance
            if step < burn_in and (step + 1) % 20 == 0:
                rate = accepted[step - 19 : step + 1].double().mean(dim=0).to(device)
                factor = torch.where(rate < 0.2, 0.7, torch.where(rate > 0.25, 1.4, 1.0))
                radius = radius * factor.unsqueeze(1)
        sample = chain[burn_in:]  # (kept, n, d)
        acceptance = accepted[burn_in:].double().mean(dim=0)
        rows = []
        for k, o in enumerate(orders):
            points = sample[:, k]
            covariance = torch.cov(points.T)
            row = {"order": o, "acceptance": round(acceptance[k].item(), 4)}
            if torch.linalg.matrix_rank(covariance) < d:
                rows.append({**row, "ellipsoid": None, "hits": None, "draws": None, "volume": None})
                continue
            ellipsoid = unit_ball * (d + 2) ** (d / 2) * torch.linalg.det(covariance).sqrt().item()
            scale = torch.linalg.cholesky(
                (d + 2) * covariance
            )  # lower triangular L with L L^T = (d + 2) S
            inside = in_ball(draws, d, generator, device) @ scale.T + points.mean(dim=0)
            in_bounds = (
                ((inside >= 0) & (inside <= 1)).all(dim=1).cpu()
            )  # outside the bounds is always a miss
            hit = torch.zeros(draws, dtype=torch.bool)
            if bool(in_bounds.any()):
                hit[in_bounds] = torch.tensor(
                    [p == o for p in patterns_of(inside[in_bounds.to(device)])]
                )
            rows.append(
                {
                    **row,
                    "ellipsoid": ellipsoid,
                    "hits": int(hit.sum()),
                    "draws": draws,
                    "volume": ellipsoid * hit.double().mean().item(),
                }
            )
        pd.DataFrame(rows).to_csv(OUT_DIR / "volumes.csv", index=False)
        record(
            "volume",
            seconds=round(time.perf_counter() - start, 3),
            evaluations=n * (burn_in + kept + draws),
        )


if __name__ == "__main__":
    main()
