# Notebooks

The articles of the R package, replicated with psp-torch. Each notebook is committed with its outputs; `execute.py` re-runs them.

| Notebook | What it shows |
| --- | --- |
| [tutorial.ipynb](tutorial.ipynb) | **What can prospect theory predict?** A complete first analysis: a two-parameter model of risky choice, how to write `model` and `discretize` for a batch of parameter sets, running `PSP()`, checking the partition against brute force, and checking that the search found everything. |
| [recipe-alcove.ipynb](recipe-alcove.ipynb) | **ALCOVE and the six category structures.** The analysis of Pitt et al. (2006) on a slow learning model with four parameters: patterns that allow ties, planning and saving a long run, reading a partition you cannot plot, and robustness checks. Reads the results of [alcove/run_alcove.py](alcove/run_alcove.py). |
| [oscar/](oscar/README.md) | **OSCAR on dome2025ibre.** One PSP run per participant, all advanced in lockstep on the GPU; prepared for a large GPU, not run here. |
| [benchmarks.ipynb](benchmarks.ipynb) | **Benchmarks.** The port against the R package on the same machine and workloads, and when the GPU is worth using. Reads [../benchmarks/results](../benchmarks/results). |
| [benchmarks-oscar.ipynb](benchmarks-oscar.ipynb) | **Benchmarks: OSCAR.** The compiled C++ of OSCAR called from R against the batched port on the CPU and the GPU: one call of the model by batch size, and the participant-level routine for 14 participants. Reads [../benchmarks/oscar/results](../benchmarks/oscar/results). |

To refresh everything from scratch:

```sh
uv run python notebooks/alcove/run_alcove.py          # about 20 minutes on the CPU
# the benchmarks: see benchmarks/README.md for the full list of commands
uv run python benchmarks/compare.py --r-library <library with psp 1.0.6> --devices cpu,cuda --rowwise
uv run python benchmarks/devices.py
uv run python notebooks/execute.py
```
