"""Partitions OSCAR's parameter space for every participant of dome2025ibre, one run each.

A port of ``psp.R`` / ``run.r``: each participant is an independent PSP run on that
participant's own trial order, with the same settings as the R routine (1,000
random starting points, radius 1, population 1,000,000 so that every pattern
keeps proposing, 10,000 iterations, the ±0.125 inequality matrix of P(rare) at
test for A, B, BC and C behind a 0.5 anchor). The runs advance in lockstep so
that one batched model call per iteration serves all of them, which is what a
large GPU needs.

Usage, from the repository root::

    uv run python notebooks/oscar/run_oscar.py --device cuda --dtype float32
    uv run python notebooks/oscar/run_oscar.py --smoke   # 2 participants, 3 iterations, CPU

Run it once with ``--smoke`` to check the environment, then without. The run
resumes: participants listed in ``tracking.csv`` of the output directory are
skipped, so a stopped run can be restarted with the same command.

Outputs in ``--out``, named as the R routine names them (``OSCAR_4.<ppt>``):

- ``OSCAR_4.<ppt>_parameters.csv`` and ``OSCAR_4.<ppt>_continuous.csv``: every
  evaluation of that participant's run (``--no-save`` turns them off; with the
  default settings they are large, about 1 GB per participant);
- ``patterns/OSCAR_4.<ppt>.npz``: ``ordinal_patterns`` and ``ordinal_counts``;
- ``frequencies.csv``: ``ppt, dataset, id, frequency``, one row per pattern;
- ``tracking.csv``: ``ppt, started, completed, iterations, patterns, evaluations``.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from psp import PSPLockstep
from psp.models.oscar import (
    EVALUATED_STIMULI,
    OSCAR_BOUNDS,
    OSCAR_PARAMETERS,
    discretize,
    dome2025ibre,
    evaluate,
    training_batch,
)

DATASET = "dome2025ibre"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--participants", default="all", help="comma-separated ids, or 'all'")
    parser.add_argument("--first", type=int, default=None, help="only the first N participants")
    parser.add_argument(
        "--group", type=int, default=0, help="participants per lockstep group (0 = all at once)"
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--population", type=int, default=1_000_000)
    parser.add_argument("--starts", type=int, default=1000)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--thresholds", default="0.125,-0.125")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-save", action="store_true", help="do not write the per-evaluation csv files"
    )
    parser.add_argument(
        "--keep-self-column",
        action="store_true",
        help="zero each node's own column in the attention shift instead of column A "
        "(the C++ zeroes column A)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="2 participants, 10 starts, 3 iterations, CPU"
    )
    args = parser.parse_args()
    if args.smoke:
        args.first, args.starts, args.iterations, args.device = 2, 10, 3, "cpu"
        args.out = str(Path(args.out).parent / "results-smoke")
    return args


def main() -> None:
    args = parse()
    out = Path(args.out)
    (out / "patterns").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    data = dome2025ibre()
    ids = data.participants if args.participants == "all" else args.participants.split(",")
    if args.first:
        ids = ids[: args.first]
    done = set()
    tracking = out / "tracking.csv"
    if tracking.exists():
        with open(tracking, newline="", encoding="utf-8") as file:
            done = {row["ppt"] for row in csv.DictReader(file)}
    todo = [p for p in ids if p not in done]
    print(f"{len(todo)} participants to run ({len(done)} already in {tracking})", flush=True)
    group = args.group or len(todo)
    lower, upper = (torch.tensor(b, dtype=torch.float64) for b in OSCAR_BOUNDS)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    init_generator = torch.Generator().manual_seed(args.seed)
    # the R routine draws one matrix of starting points and uses it for every participant
    init = lower + torch.rand((args.starts, 5), generator=init_generator, dtype=torch.float64) * (
        upper - lower
    )

    for start in range(0, len(todo), group):
        members = todo[start : start + group]
        training, members = training_batch(data, members, dtype=dtype)
        training = training.to(device)

        def model(unit: torch.Tensor, run: torch.Tensor, training=training) -> torch.Tensor:
            return evaluate(
                unit,
                training,
                sequence=run,
                generator=generator,
                zero_first_column=not args.keep_self_column,
            )

        def disc(probabilities: torch.Tensor) -> torch.Tensor:
            return discretize(probabilities, thresholds)

        controls = [
            {
                "lower": lower,
                "upper": upper,
                "init": init,
                "radius": args.radius,
                "population": args.population,
                "iterations": args.iterations,
                "parameter_names": list(OSCAR_PARAMETERS),
                "stimuli_names": ["anchor", *EVALUATED_STIMULI],
                "dimensionality": len(EVALUATED_STIMULI) + 1,
                "responses": len(EVALUATED_STIMULI) + 1,
            }
            for _ in members
        ]
        paths = [out / f"OSCAR_4.{p}" for p in members]
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        clock = time.perf_counter()
        print(
            f"group of {len(members)} participants, {training.shape[1]} trials padded, "
            f"started {started}",
            flush=True,
        )
        results = PSPLockstep(
            model,
            disc,
            controls,
            save=not args.no_save,
            paths=paths,
            quiet=False,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"group done in {(time.perf_counter() - clock) / 60:.1f} minutes", flush=True)

        for ppt, result in zip(members, results, strict=True):
            np.savez_compressed(
                out / "patterns" / f"OSCAR_4.{ppt}.npz",
                ordinal_patterns=result.ordinal_patterns.numpy(),
                ordinal_counts=result.ordinal_counts.numpy(),
            )
            _append(
                out / "frequencies.csv",
                ["ppt", "dataset", "id", "frequency"],
                [
                    [ppt, DATASET, k + 1, int(c)]
                    for k, c in enumerate(result.ordinal_counts.tolist())
                ],
            )
            _append(
                tracking,
                ["ppt", "started", "completed", "iterations", "patterns", "evaluations"],
                [[ppt, started, completed, result.iterations, result.patterns, result.evaluations]],
            )
    print("all participants done", flush=True)


def _append(path: Path, header: list[str], rows: list[list]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        if not exists:
            writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
