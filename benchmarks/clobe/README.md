# clobe: the R package against psp.clobe

Seconds per call of every function of the R package clobe 0.8.0 and of its port `psp.clobe`, on the same machine and the same inputs, with a checksum of every result to check that both give the same answer. The notebook [notebooks/benchmarks-clobe.ipynb](../../notebooks/benchmarks-clobe.ipynb) reads the results and explains them.

| File | What it does |
| --- | --- |
| `run.py` | Writes the inputs to a temporary directory, runs `r/bench.R` and `bench.py`, and writes `results/r.csv` and `results/torch.csv`. |
| `r/bench.R` | The workloads with the R package: one call per measurement, as a user would make it. |
| `bench.py` | The same workloads with `psp.clobe`, on the CPU in float64 and, for the functions that take tensors, on the GPU in float64 with the inputs already on the device. |

The workloads: `imac()` on one result and on 1,000 to 100,000 results of 5 stimuli (R calls it once per result, as `patterns()` does), `patterns()` on the same results, `gdistance()` with 100 human patterns against 1,000 to 38,703 model patterns of 6 stimuli, `pattern_distance()` between every pair of 100 to 5,000 patterns of 6 stimuli, `orderings()` for semiorders of 5 to 7 stimuli and weak orderings of 7 to 9, and `brutes()` with 11 values for 5 to 8 stimuli (8 without the matrices). The semiorders of 8 stimuli are left out because the R package needs about 18 GB of memory for them.

Each measurement is the median of up to five samples after one warm-up call; a sample repeats the call until it lasts at least 0.05 s. Results files: one row per function, size and build, with `seconds` (per call), `samples` and `check` (a sum of the cells, a count or the g-distance, depending on the function).

## Running it

clobe 0.8.0 is installed into its own library, so that the installed packages are not touched; it needs Rcpp, RcppArmadillo and RcppParallel.

```sh
R CMD INSTALL --library=/path/to/clobe-lib /path/to/clobe
uv run python benchmarks/clobe/run.py --r-library /path/to/clobe-lib
uv run python notebooks/execute.py benchmarks-clobe.ipynb
```

`--skip-r` runs only the torch side and keeps `results/r.csv`; `--devices cpu` leaves out the GPU. Run it with nothing else going on: the R package's `brutes()` uses every core.
