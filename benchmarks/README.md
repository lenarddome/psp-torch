# Benchmarks

The workloads of the R package's [benchmarks article](https://lenarddome.com/psp/articles/benchmarks.html), run for both the R package and the torch port on one machine, plus a study of when the GPU pays off and a run of ALCOVE on its full parameter space. The results are read by [notebooks/benchmarks.ipynb](../notebooks/benchmarks.ipynb).

## Scripts

| Script | What it measures | Writes |
| --- | --- | --- |
| `benchmark.py` | Time per run of `pspGlobal()` on five fixed workloads (the R package's `benchmark.R`), one csv row per run on stdout. `--device`, `--rowwise` (call the model one parameter set at a time, as R does). | stdout |
| `experiments.py` | Overhead with a near-free model, the cost of `save=True`, regions found on a 50-region model, and the effect of population, radius and starting points. `--experiments overhead,save,discovery,settings`. | `results/overhead.csv`, `save.csv`, `discovery.csv`, `settings.csv` |
| `compare.py` | Runs `r/benchmark.R` and `r/experiments.R` (the R package's own scripts, unchanged) with a given R library, then `benchmark.py` and `experiments.py` for every device, and merges everything into `results/`. `--rowwise` adds the row-wise CPU run; `--skip-r` runs only the torch side. | `results/workloads.csv` and the files above |
| `devices.py` | Seconds per call of `model` and `discretize` for batches of 1 to 4,096 parameter sets, on every device and dtype, for a near-free model, a 1,000-region Voronoi model and ALCOVE; and whole runs with 1, 16 and 256 starting points. | `results/devices.csv` |
| `alcove_benchmark.py` | ALCOVE with the learning rates free up to 1: 20 starting points, radius 0.15, population 50, cap 1,000 iterations, seeds 1 to 3. Time, regions found, evaluations, and the number of regions found after each iteration. `--device`, `--dtype`, `--seeds`. `--import-r` merges the files the R script writes. | `results/alcove.csv`, `alcove_discovery.csv`, `alcove_patterns.csv` |
| `r/alcove_benchmark.R` | The same ALCOVE workload for the R package, with catlearn's `slpALCOVE()`. | three csv files in a directory of your choice |

Rows of a build that is run again replace its earlier rows, so a single build can be re-measured without touching the others. The build label defaults to `psp-torch <version> <device> [<dtype>|rowwise]`, followed by ` (<machine>)` when `--machine` is given.

## A full run

The R package to compare against is built from `psp-reference/` into its own library, so that the installed CRAN version is not disturbed:

```sh
R CMD INSTALL --library=/path/to/psp-lib psp-reference
```

Then, with nothing else running on the machine (each build runs in its own process; a loaded machine distorts the timings):

```sh
uv run python benchmarks/compare.py --r-library /path/to/psp-lib --devices cpu,cuda --rowwise
uv run python benchmarks/devices.py
uv run python benchmarks/alcove_benchmark.py --device cpu --dtype float64
uv run python benchmarks/alcove_benchmark.py --device cuda --dtype float32
uv run python benchmarks/alcove_benchmark.py --device cuda --dtype float64
Rscript benchmarks/r/alcove_benchmark.R /path/to/psp-lib "psp 1.0.6 (R)" 1,2,3 /tmp/alcove-r
uv run python benchmarks/alcove_benchmark.py --import-r /tmp/alcove-r
uv run python notebooks/execute.py benchmarks.ipynb
```

On Windows, give the paths with forward slashes and `Rscript` as `"C:/Program Files/R/R-4.x.x/bin/Rscript.exe"` (also `--rscript` for `compare.py`). The R package needs `catlearn` installed for the ALCOVE workload.

Budget: the toy workloads take a few minutes per build; `devices.py` about ten minutes; each ALCOVE run about three minutes on the CPU with the port and about an hour with the R package (50,000 evaluations at 66 ms each), so the R part of the ALCOVE benchmark is close to three hours for three seeds.

## Another machine

The GPU results depend on the card and the torch results on the CPU, so a speed-up is only meaningful between builds measured on one machine. The shipped results come from two machines: the full run above, R included, on a Windows PC with an Intel Xeon w5-2455X (12 cores), 32 GB of memory and an NVIDIA T400 4 GB (its builds carry no machine name; `Xeon w5-2455X, T400` in `devices.csv`), and the torch side on a Linux PC with an AMD Ryzen 7 5800X (8 cores), 32 GB and an NVIDIA RTX 3060 12 GB (its builds end in `(5800X, RTX 3060)`). The notebook names every build after the hardware it ran on: the CPU for R and the `cpu` builds, the card for the `cuda` builds. To add another machine alongside them, give every script a `--machine` name; it is appended to every build label, and `devices.py` writes it to its `machine` column, so the rows of the other machines are kept:

```sh
uv sync                                    # CUDA 12.8 build of torch by default
M="5800X, RTX 3060"
uv run python benchmarks/compare.py --skip-r --devices cpu,cuda --rowwise --machine "$M"
uv run python benchmarks/devices.py --machine "$M"
uv run python benchmarks/alcove_benchmark.py --device cpu --dtype float64 --machine "$M"
uv run python benchmarks/alcove_benchmark.py --device cuda --dtype float32 --machine "$M"
uv run python benchmarks/alcove_benchmark.py --device cuda --dtype float64 --machine "$M"
uv run python notebooks/execute.py benchmarks.ipynb
```

Add the new name to `other_machines`, and its CPU and card to `hardware`, in the first code cell of the notebook. For a complete comparison there, also build the R package into a library on that machine and run the two R steps of the full run above with `--machine` (`compare.py`) and a label ending in the same `(<machine>)` (`r/alcove_benchmark.R`). The tables and figures of the notebook are computed from the csv files; a few sentences of its commentary describe the cards and need rewording for a different one.

## Result files

The figures of the notebook are also exported as PNG files in `results/figures/`, one per chart, named after the section they belong to.

| File | Columns |
| --- | --- |
| `workloads.csv` | `build, workload, seed, seconds, iterations, patterns, evaluations, filled` |
| `overhead.csv` | `build, rep, evaluations, patterns, total_us, callbacks_us, framework_us` (torch also `batched_us`) |
| `save.csv` | `build, seed, evaluations, save_us, total_us, mb` |
| `discovery.csv` | `build, seed, seconds, iterations, regions, evaluations, filled` |
| `settings.csv` | `build, setting, seed, regions, evaluations, filled` |
| `devices.csv` | `machine, device, dtype, model, kind, batch, seconds, evaluations` (`kind` is `model` for one call, `psp` for a run) |
| `alcove.csv` | `build, seed, seconds, iterations, patterns, evaluations, filled` |
| `alcove_discovery.csv` | `build, seed, iteration, evaluations, patterns` (distinct patterns found so far) |
| `alcove_patterns.csv` | `build, seed, order, count` (every ordering a run found, with its population) |

Seconds are wall-clock time of one complete `pspGlobal()` call with `quiet=True`: every model evaluation plus everything the package does around them, including csv writing where `save=True`. R uses `system.time()[["elapsed"]]`, Python `time.perf_counter()`.
