"""When is the GPU worth it? Times the models of this repository on every device.

Usage::

    uv run python benchmarks/devices.py [--devices cpu,cuda] [--machine NAME] \
        [--out benchmarks/results/devices.csv]

Two things decide whether a run gains from the GPU: how much work one call of
``model`` does, and how many parameter sets each call receives, which in a PSP
run is the number of patterns still filling. This script measures both:

- ``model`` alone: seconds per call for batches of 1 to 4,096 parameter sets,
  for a near-free model (returns its parameters), the Voronoi model with 1,000
  centres, and ALCOVE with 30 simulated learners, in float32 and float64.
- ``pspGlobal`` end to end on the same models, with the batch size set by the
  number of starting points and a population that keeps every pattern filling.

Rows: ``machine, device, dtype, model, kind, batch, seconds, evaluations``. ``kind`` is
``model`` for a single call or ``psp`` for a whole run. The rows of ``--machine``
(by default the host name) replace that machine's earlier rows and leave the others.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Callable
from pathlib import Path

import torch
from compare import read_rows, write_rows

from psp import pspGlobal
from psp.models.alcove import alcove_curves, alcove_training, weak_order
from psp.models.voronoi import voronoi_model

RESULTS = Path(__file__).resolve().parent / "results" / "devices.csv"
BATCHES = (1, 4, 16, 64, 256, 1024, 4096)


def timed(fn: Callable[[], object], device: torch.device, repeats: int) -> float:
    """Median seconds of ``repeats`` calls after one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def models(device: torch.device, dtype: torch.dtype) -> dict[str, tuple[Callable, Callable, int]]:
    """name -> (model, discretize, dimensions)."""
    w = 4.0 ** torch.arange(5, dtype=dtype, device=device)

    def free_model(par: torch.Tensor) -> torch.Tensor:
        return par

    def free_discretize(x: torch.Tensor) -> torch.Tensor:
        return (torch.floor(x * 3.999) * w).sum(dim=1).reshape(-1, 1, 1).expand(-1, 2, 2)

    vor = voronoi_model(1000, 5)
    training = alcove_training(5, 7624).to(device, dtype)

    def alcove_model(unit: torch.Tensor) -> torch.Tensor:
        return alcove_curves(unit, training, chunk=4096)

    return {
        "near-free": (free_model, free_discretize, 5),
        "voronoi 1000": (vor.model, vor.discretize, 5),
        "alcove": (alcove_model, weak_order, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--devices", default="cpu,cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--machine", default=platform.node(), help="name of this machine")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name in args.devices.split(","):
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("skipping cuda: not available")
            continue
        for dtype in (torch.float32, torch.float64):
            for model_name, (model, discretize, d) in models(device, dtype).items():
                for batch in BATCHES:
                    if batch > args.max_batch:
                        continue
                    if (
                        model_name == "alcove"
                        and batch > 1024
                        and dtype is torch.float64
                        and device.type == "cuda"
                    ):
                        continue  # 4 GB cards run out of memory here
                    par = torch.rand((batch, d), device=device, dtype=dtype)
                    repeats = 5 if model_name == "alcove" else 20
                    seconds = timed(
                        lambda par=par, model=model, discretize=discretize: discretize(model(par)),
                        device,
                        repeats,
                    )
                    rows.append(
                        {
                            "machine": args.machine,
                            "device": device.type,
                            "dtype": str(dtype).split(".")[-1],
                            "model": model_name,
                            "kind": "model",
                            "batch": batch,
                            "seconds": seconds,
                            "evaluations": batch,
                        }
                    )
                    print(rows[-1], flush=True)
                # a whole run whose batch size is set by the starting points: every start is
                # its own pattern for the near-free and Voronoi models, so the batch stays
                # near `starts`
                for starts in (1, 16, 256):
                    if model_name == "alcove" and starts == 256:
                        continue
                    iterations = 20 if model_name == "alcove" else 50
                    init = torch.rand((starts, d), device=device, dtype=dtype)
                    ctrl = {
                        "init": init,
                        "lower": [0.0] * d,
                        "upper": [1.0] * d,
                        "radius": 0.1,
                        "iterations": iterations,
                        "population": 10**9,
                    }
                    generator = torch.Generator(device=device).manual_seed(0)

                    def run(
                        ctrl=ctrl,
                        model=model,
                        discretize=discretize,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    ):
                        return pspGlobal(
                            model,
                            discretize,
                            ctrl,
                            quiet=True,
                            device=device,
                            dtype=dtype,
                            generator=generator,
                        )

                    seconds = timed(run, device, 3 if model_name == "alcove" else 5)
                    result = run()
                    rows.append(
                        {
                            "machine": args.machine,
                            "device": device.type,
                            "dtype": str(dtype).split(".")[-1],
                            "model": model_name,
                            "kind": "psp",
                            "batch": starts,
                            "seconds": seconds,
                            "evaluations": result.evaluations,
                        }
                    )
                    print(rows[-1], flush=True)

    others = [r for r in read_rows(out) if r.get("machine") != args.machine]
    write_rows(out, others + rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
