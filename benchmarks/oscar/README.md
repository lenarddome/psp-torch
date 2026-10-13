# OSCAR: R + RcppArmadillo against the torch port

A local benchmark of the participant-level OSCAR routine: the compiled C++ model called from R with the R package psp, against `psp.models.oscar` with `pspMany` on the CPU and on the GPU. The results and the report are in `results/` (`report.md`, `model.png`, `psp.png`).

| File | What it does |
| --- | --- |
| `run.py` | Runs every step below in turn, then `report.py`. |
| `r/common.R` | Loads psp from the given library, compiles `architecture.cpp` + `OSCAR.cpp` from `tmp-models/` with `Rcpp::sourceCpp()`, sources `convert.R` and `imac.R`, builds a participant's trial matrix as `psp.R` does. |
| `r/model.R` | Seconds per evaluation for the first participant: `OSCAR()` alone and the routine's `evaluate()`, one parameter set per call. |
| `model.py` | Seconds per batched call of the port's `evaluate()` + `discretize()` for 1 to 16,384 parameter sets, CPU float64, GPU float32 and float64. |
| `r/psp_run.R` | `pspGlobal()` per participant for the first N participants, spread over forked worker processes (`psp.R` uses foreach over `detectCores() - 2`). |
| `report.py` | `results/report.md` and the two figures. |

The torch side of the routine is `notebooks/oscar/run_oscar.py --no-save`, unchanged.

## Running it

The R package is installed into its own library, so the installed packages are not touched; Rcpp, RcppArmadillo, data.table and clobe come from the main library.

```sh
Rscript -e 'install.packages("psp", lib = "benchmarks/oscar/r-lib", repos = "https://cloud.r-project.org")'
uv run python benchmarks/oscar/run.py --r-library benchmarks/oscar/r-lib
```

Defaults: 14 participants, 100 starting points, 200 iterations, 14 R workers, 1,000 parameter sets for the R model timing. `--steps model` or `--steps psp` runs one part; `--participants`, `--starts`, `--iterations` and `--workers` change the routine's settings. Run it with nothing else going on.
