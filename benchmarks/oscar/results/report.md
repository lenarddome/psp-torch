# OSCAR: R + RcppArmadillo against the torch port, CPU and GPU

## One call of the model

One participant (1,000 parameter sets for R, 3 repeats; 5 repeats per batch for torch). R calls the compiled `OSCAR()` once per parameter set, as `pspGlobal()` calls the model; `evaluate()` adds what the routine does around it (a fresh random initial state, and the data.table reduction to P(rare) at test). The port's `evaluate()` does all of that for the whole batch in one call; its time includes `discretize()`.

| R + RcppArmadillo | per evaluation |
| --- | --- |
| `OSCAR() alone` | 2.32 ms |
| `evaluate()` | 5.56 ms |

| batch | torch cpu float64 | torch cuda float32 | torch cuda float64 | vs R evaluate(), cpu float64 | vs R evaluate(), cuda float32 | vs R evaluate(), cuda float64 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 81.35 ms | 229.38 ms | 228.23 ms | 0.068× | 0.024× | 0.024× |
| 4 | 21.75 ms | 58.17 ms | 58.46 ms | 0.26× | 0.096× | 0.095× |
| 16 | 6.54 ms | 14.66 ms | 14.74 ms | 0.85× | 0.38× | 0.38× |
| 64 | 2.44 ms | 3.68 ms | 3.70 ms | 2.3× | 1.5× | 1.5× |
| 256 | 1.32 ms | 926.4 µs | 931.8 µs | 4.2× | 6× | 6× |
| 1024 | 1.03 ms | 232.7 µs | 231.5 µs | 5.4× | 23.9× | 24.0× |
| 4096 | 378.7 µs | 58.6 µs | 125.5 µs | 14.7× | 94.9× | 44.3× |
| 16384 | 367.8 µs | 20.3 µs | 118.7 µs | 15.1× | 273.5× | 46.8× |

Fastest per evaluation: torch cpu float64 367.8 µs at 16,384 sets per call (15.1× R's `evaluate()`); torch cuda float32 20.3 µs at 16,384 sets per call (273.5× R's `evaluate()`); torch cuda float64 118.7 µs at 16,384 sets per call (46.8× R's `evaluate()`).

![time per evaluation](model.png)

Where the GPU lines show one colour, float32 and float64 coincide: below a few thousand parameter sets per call the GPU's time is kernel launches (about 112 trials × 10 attention-shift iterations of small operations per call), not arithmetic, so the precision does not matter. The same fixed cost makes one torch call with a single parameter set some 35 times slower than one C++ call.

## The participant-level routine

The first 14 participants, 100 starting points (one matrix shared by all participants, as `psp.R`), radius 1, population 1,000,000 (so every pattern keeps proposing until the cap), 200 iterations, seed 2026. R runs `pspGlobal()` with the compiled OSCAR once per participant, spread over 14 forked processes (`psp.R` runs foreach over `detectCores() - 2`); its time is the wall clock of the whole parallel run. The port runs `notebooks/oscar/run_oscar.py --no-save`, every participant's search advancing in lockstep with one batched OSCAR call per iteration; its time is the wall clock of that whole process, Python's start-up, data loading and CUDA initialisation included.

| build | how | minutes, all participants | evaluations per second | speed-up vs R | patterns per participant | evaluations per participant |
| --- | --- | --- | --- | --- | --- | --- |
| R + C++ | 14 worker processes | 7.5 | 1,099 | 1× | 198–216, median 204 | 35,417 |
| torch cpu float64 | 1 process, lockstep | 4.4 | 1,861 | 1.7× | 194–215, median 205 | 34,916 |
| torch cuda float32 | 1 process, lockstep | 0.9 | 9,597 | 8.6× | 196–220, median 208 | 35,461 |

Each R participant took a median of 399 s in the parallel run; the same 35,417 evaluations at the 5.56 ms of a lone `evaluate()` come to about 197 s, so with 14 workers on 16 hardware threads each R process ran about 2.0 times slower than alone. That is the R routine as `psp.R` runs it on this machine; the port's CPU run used torch's own threads in one process.

![wall-clock time](psp.png)

Do they find the same inequality matrices? Per participant, against R:

| build | shared (median) | R only | torch only | share of evaluations in shared matrices (worst participant) |
| --- | --- | --- | --- | --- |
| torch cpu float64 | 196 | 8 | 6 | R 99.7%, torch 99.9% |
| torch cuda float32 | 198 | 6 | 9 | R 99.9%, torch 99.7% |

The model draws its initial state at random on every evaluation, and R and torch use different random numbers, so no two runs take the same path; matrices only one run found hold one or a few evaluations each.

## Machine

- CPU: AMD Ryzen 7 5800X 8-Core Processor, torch threads 8
- GPU: NVIDIA GeForce RTX 3060
- Linux 7.0.0-38-generic, Python 3.14.4, torch 2.11.0+cu128, R version 4.5.2 (2025-10-31)
- R side: psp 1.0.6 from CRAN in `benchmarks/oscar/r-lib`, OSCAR compiled from `benchmarks/oscar/tmp-models/` (`architecture.cpp` + `OSCAR.cpp`) with `Rcpp::sourceCpp()` and R's default flags (`-O2`).
