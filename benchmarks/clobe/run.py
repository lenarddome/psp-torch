"""psp.clobe against the R package clobe 0.8.0, on the same machine and the same inputs.

Usage::

    uv run python benchmarks/clobe/run.py --r-library <library with clobe 0.8.0> \
        [--devices cpu,cuda] [--skip-r]

Writes the inputs both sides read to a temporary directory: 100,000 results of 5 stimuli
(proportions of 20 trials), and random selections of the 38,703 semiorders of 6 stimuli for
``gdistance`` and ``pattern_distance``. Then runs ``r/bench.R`` and ``bench.py`` and writes
``results/r.csv`` and ``results/torch.csv``, one row per function and size: the median seconds per
call and a checksum of the result, which must agree between the two.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from psp import clobe

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def write_inputs(directory: Path) -> None:
    g = torch.Generator().manual_seed(2026)
    data = torch.randint(0, 21, (100_000, 5), generator=g).double() / 20
    np.savetxt(directory / "data.csv", data.numpy(), delimiter=",", fmt="%.17g",
               header="s1,s2,s3,s4,s5", comments="")  # fmt: skip
    six = clobe.semiorders(6)
    for name, count in (("human_idx", 100), ("model_idx", six), ("x_idx", six)):
        index = torch.randperm(six, generator=g)[:count]
        np.savetxt(directory / f"{name}.csv", index.numpy(), fmt="%d", header="index", comments="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--r-library", help="R library with clobe 0.8.0")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--devices", default="cpu,cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-r", action="store_true")
    args = parser.parse_args()
    if not args.skip_r and not args.r_library:
        parser.error("--r-library is required unless --skip-r is given")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp)
        write_inputs(inputs)
        if not args.skip_r:
            cmd = [args.rscript, str(HERE / "r" / "bench.R"), args.r_library, str(inputs),
                   str(RESULTS / "r.csv")]  # fmt: skip
            subprocess.run(cmd, check=True)
        cmd = [sys.executable, str(HERE / "bench.py"), str(inputs), str(RESULTS / "torch.csv"),
               "--devices", args.devices]  # fmt: skip
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
