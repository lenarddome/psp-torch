"""Benchmarks the R package and the torch port on the same workloads, on this machine.

Usage::

    uv run python benchmarks/compare.py --r-library <path to a library with psp installed> \
        [--rscript Rscript] [--r-label "psp 1.0.6"] [--devices cpu,cuda] [--skip-r] \
        [--machine "5800X, RTX 3060"]

It runs, in turn, ``benchmarks/r/benchmark.R`` and ``benchmarks/r/experiments.R``
(the scripts of the R package) with the given library, then ``benchmark.py`` and
``experiments.py`` for every device listed (and, with ``--rowwise``, the CPU
workloads once more calling the model one parameter set at a time), and writes everything to
``benchmarks/results/``: ``workloads.csv``, ``overhead.csv``, ``save.csv`` and
``discovery.csv``. Rows of a build that is run again replace its earlier rows.
Finally it prints the median seconds per workload for every build. ``--machine``
appends the machine's name to every label, so that runs from several machines sit
side by side in the same files.

Each build runs in its own process while the machine is otherwise idle; run it
with nothing else going on.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FILES = ("workloads", "overhead", "save", "discovery", "settings")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def replace_build(path: Path, build: str, new_rows: list[dict]) -> None:
    rows = [r for r in read_rows(path) if r["build"] != build]
    write_rows(path, rows + new_rows)


def run_r(rscript: str, library: str, label: str, scratch: Path) -> None:
    print(f"R: {label} from {library}", flush=True)
    out = subprocess.run(
        [rscript, str(HERE / "r" / "benchmark.R"), library, label],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(out.stdout.splitlines()))
    replace_build(RESULTS / "workloads.csv", label, rows)
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("overhead", "save", "discovery"):
        (scratch / f"{name}.csv").unlink(missing_ok=True)
    subprocess.run(
        [
            rscript,
            str(HERE / "r" / "experiments.R"),
            library,
            label,
            "overhead,save,discovery",
            str(scratch),
        ],
        check=True,
    )
    for name in ("overhead", "save", "discovery"):
        replace_build(RESULTS / f"{name}.csv", label, read_rows(scratch / f"{name}.csv"))


def machine_suffix(machine: str | None) -> str:
    return f" ({machine})" if machine else ""


def run_torch(
    device: str, scratch: Path, rowwise: bool = False, machine: str | None = None
) -> None:
    import psp

    label = f"psp-torch {psp.__version__} {device}{' rowwise' if rowwise else ''}"
    label += machine_suffix(machine)
    print(f"torch: {label}", flush=True)
    out = subprocess.run(
        [sys.executable, str(HERE / "benchmark.py"), "--label", label, "--device", device]
        + (["--rowwise"] if rowwise else []),
        check=True,
        capture_output=True,
        text=True,
    )
    replace_build(RESULTS / "workloads.csv", label, list(csv.DictReader(out.stdout.splitlines())))
    if rowwise:
        return
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("overhead", "save", "discovery", "settings"):
        (scratch / f"{name}.csv").unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "experiments.py"),
            "--label",
            label,
            "--device",
            device,
            "--out-dir",
            str(scratch),
        ],
        check=True,
    )
    for name in ("overhead", "save", "discovery", "settings"):
        replace_build(RESULTS / f"{name}.csv", label, read_rows(scratch / f"{name}.csv"))


def summary() -> str:
    rows = read_rows(RESULTS / "workloads.csv")
    builds = list(dict.fromkeys(r["build"] for r in rows))
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        times[(r["build"], r["workload"])].append(float(r["seconds"]))
    width = max(len(w) for w in workloads)
    lines = ["median seconds per run", " " * width + "  " + "  ".join(f"{b:>24}" for b in builds)]
    for w in workloads:
        cells = []
        for b in builds:
            t = times.get((b, w))
            cells.append(f"{median(t):>24.3f}" if t else f"{'-':>24}")
        lines.append(f"{w:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--r-library", help="R library containing the build of psp to benchmark")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-label", default="psp 1.0.6")
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--skip-r", action="store_true")
    parser.add_argument(
        "--rowwise",
        action="store_true",
        help="also time the CPU workloads calling the model once per parameter set, as R does",
    )
    parser.add_argument(
        "--machine", default=None, help="name of this machine, appended to every label"
    )
    parser.add_argument("--scratch", default=str(HERE / "results" / "_scratch"))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    if not args.skip_r:
        if not args.r_library:
            parser.error("--r-library is required unless --skip-r is given")
        run_r(args.rscript, args.r_library, args.r_label + machine_suffix(args.machine), scratch)
    for device in args.devices.split(","):
        run_torch(device, scratch, machine=args.machine)
    if args.rowwise:
        run_torch("cpu", scratch, rowwise=True, machine=args.machine)
    print()
    print(summary())


if __name__ == "__main__":
    main()
