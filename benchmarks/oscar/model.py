"""Seconds per call of the batched OSCAR ``evaluate()`` for the first participant, by batch size.

OSCAR is the model of Dome and Wills (2026), https://doi.org/10.48550/arXiv.2608.06149.

Usage::

    uv run python benchmarks/oscar/model.py --out results/model_torch.csv [--devices cpu,cuda]

The counterpart of ``r/model.R``: the same participant, parameter sets drawn uniformly within
the bounds of the routine, a fresh random initial state per row. Rows:
``device, dtype, batch, rep, seconds`` (one call of ``evaluate`` and ``discretize``).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from psp.models.oscar import OSCAR_BOUNDS, discretize, dome2025ibre, evaluate, training_batch

BATCHES = (1, 4, 16, 64, 256, 1024, 4096, 16384)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--devices", default="cpu,cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-batch", type=int, default=max(BATCHES))
    args = parser.parse_args()
    data = dome2025ibre()
    rows = []
    for name in args.devices.split(","):
        device = torch.device(name)
        dtypes = (torch.float64,) if device.type == "cpu" else (torch.float32, torch.float64)
        for dtype in dtypes:
            training, _ = training_batch(data, data.participants[:1], dtype=dtype)
            training = training.to(device)
            lower, upper = (torch.tensor(b, device=device, dtype=dtype) for b in OSCAR_BOUNDS)
            generator = torch.Generator(device=device).manual_seed(1)
            for batch in BATCHES:
                if batch > args.max_batch:
                    continue
                par = lower + torch.rand(
                    (batch, 5), generator=generator, device=device, dtype=dtype
                ) * (upper - lower)

                def call(par=par, training=training, generator=generator):
                    return discretize(evaluate(par, training, generator=generator))

                call()  # warm-up
                for rep in range(1, args.repeats + 1):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    call()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    rows.append(
                        {
                            "device": device.type,
                            "dtype": str(dtype).split(".")[-1],
                            "batch": batch,
                            "rep": rep,
                            "seconds": seconds,
                        }
                    )
                print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
