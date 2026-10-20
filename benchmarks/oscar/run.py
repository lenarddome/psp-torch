"""OSCAR: R + RcppArmadillo against the torch port on the CPU and on the GPU.

OSCAR is the model of Dome and Wills (2026), https://doi.org/10.48550/arXiv.2608.06149.

Usage::

    uv run python benchmarks/oscar/run.py --r-library benchmarks/oscar/r-lib \
        [--steps model,psp] [--participants 14] [--starts 100] [--iterations 200] [--workers 14]

Two measurements, both written to ``benchmarks/oscar/results/``:

- ``model``: seconds per evaluation of OSCAR, the C++ called once per parameter set
  from R (``r/model.R``: ``OSCAR()`` alone and the routine's ``evaluate()``), against one batched
  call of the port's ``evaluate()`` for 1 to 16,384 parameter sets (``model.py``).
- ``psp``: the participant-level routine for the first N participants with reduced settings. R runs
  ``pspGlobal()`` per participant, the participants spread over ``--workers`` processes as psp.R
  does with foreach; the port runs ``notebooks/oscar/run_oscar.py``, every participant in lockstep
  in one process, on the CPU in float64 and on the GPU in float32.

Then ``report.py`` writes ``results/report.md`` and the figures.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = HERE / "results"
TORCH_BUILDS = (("cpu", "float64"), ("cuda", "float32"))


def run(cmd: list[str]) -> float:
    print("$", " ".join(cmd), flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--r-library", required=True)
    parser.add_argument(
        "--models",
        default=str(HERE / "tmp-models"),
        help="directory with OSCAR.cpp and the R helpers",
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--steps", default="model,psp")
    parser.add_argument("--participants", type=int, default=14)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--r-evaluations", type=int, default=1000)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    steps = args.steps.split(",")

    if "model" in steps:
        run(
            [
                args.rscript,
                str(HERE / "r" / "model.R"),
                args.r_library,
                args.models,
                str(RESULTS / "model_r.csv"),
                str(args.r_evaluations),
                "3",
            ]
        )
        run([sys.executable, str(HERE / "model.py"), "--out", str(RESULTS / "model_torch.csv")])

    if "psp" in steps:
        settings = [str(args.participants), str(args.starts), str(args.iterations)]
        r_out = RESULTS / "psp_r"
        shutil.rmtree(r_out, ignore_errors=True)
        run(
            [
                args.rscript,
                str(HERE / "r" / "psp_run.R"),
                args.r_library,
                args.models,
                str(r_out),
                *settings,
                str(args.workers),
            ]
        )
        walls = []
        for device, dtype in TORCH_BUILDS:
            out = RESULTS / f"psp_torch_{device}_{dtype}"
            shutil.rmtree(
                out, ignore_errors=True
            )  # run_oscar.py resumes from an existing tracking.csv
            seconds = run(
                [
                    sys.executable,
                    str(ROOT / "notebooks" / "oscar" / "run_oscar.py"),
                    "--first",
                    str(args.participants),
                    "--starts",
                    str(args.starts),
                    "--iterations",
                    str(args.iterations),
                    "--seed",
                    "2026",
                    "--no-save",
                    "--device",
                    device,
                    "--dtype",
                    dtype,
                    "--out",
                    str(out),
                ]
            )
            walls.append(
                {
                    "device": device,
                    "dtype": dtype,
                    "participants": args.participants,
                    "wall_seconds": seconds,
                }
            )
        with open(RESULTS / "psp_torch_wall.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(walls[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(walls)

    run([sys.executable, str(HERE / "report.py")])


if __name__ == "__main__":
    main()
