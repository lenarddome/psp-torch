"""Summarises the OSCAR benchmark in ``results/``: ``report.md`` and two figures.

Usage::

    uv run python benchmarks/oscar/report.py
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# torch builds: orange, blue, green (validated as a categorical set); R + C++ is the grey reference
COLOURS = {"cpu float64": "#eb6834", "cuda float32": "#2a78d6", "cuda float64": "#1baf7a"}
GREY, INK, MUTED = "#6c757d", "#212529", "#6c757d"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "grid.color": "#e9ecef",
        "axes.axisbelow": True,
        "axes.labelcolor": "#495057",
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


def cpu_name() -> str:
    try:
        return next(
            line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo")
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        return platform.processor() or "unknown"


def r_version() -> str:
    try:
        out = subprocess.run(
            ["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True, timeout=60
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def markdown(table: pd.DataFrame) -> str:
    """A pipe table of the frame and its index, without tabulate."""
    header = [str(table.index.name or ""), *map(str, table.columns)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += [
        "| " + " | ".join(map(str, [i, *row])) + " |"
        for i, row in zip(table.index, table.to_numpy(), strict=True)
    ]
    return "\n".join(lines)


def times(v: float) -> str:
    return f"{v:,.1f}×" if v >= 10 else f"{v:.2g}×"


def fmt_us(v: float) -> str:
    return f"{v / 1000:,.2f} ms" if v >= 1000 else f"{v:,.1f} µs"


def model_section() -> list[str]:
    r = pd.read_csv(RESULTS / "model_r.csv")
    r_us = (1e6 * r.seconds / r.evaluations).groupby(r.what).median()
    t = pd.read_csv(RESULTS / "model_torch.csv")
    t["build"] = t.device + " " + t.dtype
    per_call = t.groupby(["build", "batch"]).seconds.median().unstack("build")
    per_eval = 1e6 * per_call.div(per_call.index, axis=0)
    builds = [b for b in COLOURS if b in per_eval]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for b in builds:
        ax.plot(
            per_eval.index,
            per_eval[b],
            "-o",
            color=COLOURS[b],
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            label=f"torch {b}, one batched call",
        )
    for what, style in [("evaluate()", "--"), ("OSCAR() alone", ":")]:
        ax.axhline(r_us[what], color=GREY, linestyle=style, linewidth=1.6)
        ax.annotate(
            f"R + C++ {what}, one call per set: {fmt_us(r_us[what])}",
            xy=(per_eval.index.max(), r_us[what]),
            xytext=(0, 4),
            textcoords="offset points",
            color=MUTED,
            fontsize=9.5,
            ha="right",
            va="bottom",
        )
    ax.set_xscale("log", base=4)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(
        lambda v, _: f"{v / 1000:g} ms" if v >= 1000 else f"{v:g} µs"
    )
    ax.set_xlabel("parameter sets per call")
    ax.set_title("Time per evaluation of OSCAR, one participant, log scales", pad=28)
    ax.annotate(
        "lower is faster; R evaluates one parameter set per call whatever the batch",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        color=MUTED,
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "model.png")
    plt.close(fig)

    table = pd.DataFrame({f"torch {b}": per_eval[b].map(fmt_us) for b in builds})
    for b in builds:
        table[f"vs R evaluate(), {b}"] = (r_us["evaluate()"] / per_eval[b]).map(times)
    table.index.name = "batch"
    best = {b: (per_eval[b].min(), int(per_eval[b].idxmin())) for b in builds}
    return [
        "## One call of the model",
        "",
        f"One participant ({int(r.evaluations.iloc[0]):,} parameter sets for R, 3 repeats; 5 repeats per batch for torch). "
        "R calls the compiled `OSCAR()` once per parameter set, as `pspGlobal()` calls the model; `evaluate()` adds what "
        "the routine does around it (a fresh random initial state, and the data.table reduction to P(rare) at test). "
        "The port's `evaluate()` does all of that for the whole batch in one call; its time includes `discretize()`.",
        "",
        "| R + RcppArmadillo | per evaluation |",
        "| --- | --- |",
        *[f"| `{w}` | {fmt_us(v)} |" for w, v in r_us.items()],
        "",
        markdown(table),
        "",
        "Fastest per evaluation: "
        + "; ".join(
            f"torch {b} {fmt_us(v)} at {n:,} sets per call "
            f"({times(r_us['evaluate()'] / v)} R's `evaluate()`)"
            for b, (v, n) in best.items()
        )
        + ".",
        "",
        "![time per evaluation](model.png)",
        "",
        "Where the GPU lines show one colour, float32 and float64 coincide: below a few thousand parameter sets per "
        "call the GPU's time is kernel launches (about 112 trials × 10 attention-shift iterations of small operations "
        "per call), not arithmetic, so the precision does not matter. The same fixed cost makes one torch call "
        "with a single parameter set some 35 times slower than one C++ call.",
        "",
    ]


def keys_torch(directory: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((directory / "patterns").glob("OSCAR_4.*.npz")):
        ppt = f.stem.split(".", 1)[1]
        z = np.load(f)
        for m, c in zip(z["ordinal_patterns"], z["ordinal_counts"], strict=True):
            # the upper triangle in column-major order, as compare.R writes it
            key = ",".join(str(int(m[i, j])) for j in range(m.shape[0]) for i in range(j))
            rows.append({"ppt": ppt, "count": int(c), "key": key})
    return pd.DataFrame(rows)


def contention_note(r_sum: pd.DataFrame, workers: int) -> list[str]:
    """How long one R participant took in the parallel run, against the same work in a lone process."""
    if not (RESULTS / "model_r.csv").exists():
        return []
    r = pd.read_csv(RESULTS / "model_r.csv")
    per_eval = (r.seconds / r.evaluations)[r.what == "evaluate()"].median()
    alone = r_sum.evaluations.median() * per_eval
    return [
        f"Each R participant took a median of {r_sum.seconds.median():.0f} s in the parallel run; the same "
        f"{r_sum.evaluations.median():,.0f} evaluations at the {1000 * per_eval:.2f} ms of a lone `evaluate()` "
        f"come to about {alone:.0f} s, so with {workers} workers on {os.cpu_count()} hardware threads each R process "
        f"ran about {r_sum.seconds.median() / alone:.1f} times slower than alone. That is the R routine as `psp.R` "
        "runs it on this machine; the port's CPU run used torch's own threads in one process.",
        "",
    ]


def psp_section() -> list[str]:
    r_dir = RESULTS / "psp_r"
    r_sum = pd.read_csv(r_dir / "summary.csv")
    r_wall = pd.read_csv(r_dir / "wall.csv").iloc[0]
    r_keys = pd.read_csv(r_dir / "patterns.csv")
    t_wall = pd.read_csv(RESULTS / "psp_torch_wall.csv")
    builds = [
        (
            "R + C++",
            f"{int(r_wall.workers)} worker processes",
            float(r_wall.wall_seconds),
            r_sum,
            r_keys,
        )
    ]
    for _, w in t_wall.iterrows():
        d = RESULTS / f"psp_torch_{w['device']}_{w['dtype']}"
        track = pd.read_csv(d / "tracking.csv")
        builds.append(
            (
                f"torch {w['device']} {w['dtype']}",
                "1 process, lockstep",
                float(w["wall_seconds"]),
                track,
                keys_torch(d),
            )
        )
    n = int(r_wall.participants)

    rows, overlap = [], []
    for name, how, seconds, summary, keys in builds:
        rows.append(
            {
                "build": name,
                "how": how,
                "minutes, all participants": f"{seconds / 60:.1f}",
                "evaluations per second": f"{summary.evaluations.sum() / seconds:,.0f}",
                "speed-up vs R": times(builds[0][2] / seconds),
                "patterns per participant": f"{summary.patterns.min()}–{summary.patterns.max()}, median {summary.patterns.median():.0f}",
                "evaluations per participant": f"{summary.evaluations.median():,.0f}",
            }
        )
        if name != "R + C++":
            for ppt, mine in keys.groupby("ppt"):
                theirs = r_keys[r_keys.ppt == ppt]
                shared = set(mine.key) & set(theirs.key)
                overlap.append(
                    {
                        "build": name,
                        "ppt": ppt,
                        "shared": len(shared),
                        "R only": len(set(theirs.key) - shared),
                        "torch only": len(set(mine.key) - shared),
                        "R evaluations in shared": theirs[theirs.key.isin(shared)]["count"].sum()
                        / theirs["count"].sum(),
                        "torch evaluations in shared": mine[mine.key.isin(shared)]["count"].sum()
                        / mine["count"].sum(),
                    }
                )
    table = pd.DataFrame(rows).set_index("build")
    ov = pd.DataFrame(overlap)

    fig, ax = plt.subplots(figsize=(8, 2.6))
    names = [b[0] for b in builds][::-1]
    minutes = [b[2] / 60 for b in builds][::-1]
    colours = [GREY if n_ == "R + C++" else COLOURS[n_.removeprefix("torch ")] for n_ in names]
    ax.barh(names, minutes, color=colours, height=0.6)
    for y, m in enumerate(minutes):
        ax.text(m + max(minutes) * 0.01, y, f"{m:.1f} min", va="center", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g} min")
    ax.set_xlim(0, max(minutes) * 1.15)
    ax.set_title(f"Wall-clock time of the routine for {n} participants", pad=24)
    ax.annotate(
        "lower is faster; R runs the participants in parallel processes, the port all at once in one",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "psp.png")
    plt.close(fig)

    summary = ov.groupby("build").agg(
        shared=("shared", "median"),
        r_only=("R only", "median"),
        torch_only=("torch only", "median"),
        r_in_shared=("R evaluations in shared", "min"),
        t_in_shared=("torch evaluations in shared", "min"),
    )
    return [
        "## The participant-level routine",
        "",
        f"The first {n} participants, 100 starting points (one matrix shared by all participants, as `psp.R`), radius 1, "
        "population 1,000,000 (so every pattern keeps proposing until the cap), 200 iterations, seed 2026. "
        f"R runs `pspGlobal()` with the compiled OSCAR once per participant, spread over {int(r_wall.workers)} forked "
        "processes (`psp.R` runs foreach over `detectCores() - 2`); its time is the wall clock of the whole parallel run. "
        "The port runs `notebooks/oscar/run_oscar.py --no-save`, every participant's search advancing in lockstep with "
        "one batched OSCAR call per iteration; its time is the wall clock of that whole process, "
        "Python's start-up, data loading and CUDA initialisation included.",
        "",
        markdown(table),
        "",
        *contention_note(r_sum, int(r_wall.workers)),
        "![wall-clock time](psp.png)",
        "",
        "Do they find the same inequality matrices? Per participant, against R:",
        "",
        "| build | shared (median) | R only | torch only | share of evaluations in shared matrices (worst participant) |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {b} | {r.shared:.0f} | {r.r_only:.0f} | {r.torch_only:.0f} | R {r.r_in_shared:.1%}, torch {r.t_in_shared:.1%} |"
            for b, r in summary.iterrows()
        ],
        "",
        "The model draws its initial state at random on every evaluation, and R and torch use different random numbers, "
        "so no two runs take the same path; matrices only one run found hold one or a few evaluations each.",
        "",
    ]


def main() -> None:
    lines = ["# OSCAR: R + RcppArmadillo against the torch port, CPU and GPU", ""]
    if (RESULTS / "model_r.csv").exists() and (RESULTS / "model_torch.csv").exists():
        lines += model_section()
    if (RESULTS / "psp_r" / "wall.csv").exists() and (RESULTS / "psp_torch_wall.csv").exists():
        lines += psp_section()
    lines += [
        "## Machine",
        "",
        f"- CPU: {cpu_name()}, torch threads {torch.get_num_threads()}",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"- {platform.system()} {platform.release()}, Python {platform.python_version()}, torch {torch.__version__}, {r_version()}",
        "- R side: psp 1.0.6 from CRAN in `benchmarks/oscar/r-lib`, OSCAR compiled from `tmp-models/` "
        "(`architecture.cpp` + `OSCAR.cpp`) with `Rcpp::sourceCpp()` and R's default flags (`-O2`).",
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RESULTS / "report.md")


if __name__ == "__main__":
    main()
