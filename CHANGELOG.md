# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-08

The first release: a PyTorch port of `pspGlobal()` from the R package psp 1.0.6.

### Added

- `psp.pspGlobal(model, discretize, control, save, path, extension, quiet)` with the
  interface and the sampling algorithm of the R function, plus keyword-only `batched`,
  `device`, `dtype` and `generator` arguments. `model` and `discretize` receive a whole
  batch of parameter sets per call, so a run can use the GPU; `psp.rowwise()` adapts
  functions written for one parameter set at a time.
- `psp.PSPResult`, which behaves both like the R result list and like a dataclass, and
  `psp.Control`, the validated settings.
- `psp.read_saved()` for the files written with `save=True`; values are written in their
  shortest round-trip form and read back exactly. With pyarrow installed (the `fast-save`
  extra) the files are formatted in C++, several times faster.
- `psp.models.prospect`: prospect theory on four bets of Kahneman and Tversky (1979).
- `psp.models.alcove`: a batched port of catlearn's `slpALCOVE()` and `nosof94train()`
  that matches catlearn trial for trial, with the training orders catlearn generates and
  the `nosof94` data shipped as package data.
- `psp.models.voronoi`: the toy model of the tests and benchmarks.
- `psp.models.oscar`: a batched port of the OSCAR model (C++) on the inverse base-rate effect
  data `dome2025ibre` (354 participants, shipped as package data), with the trial-matrix
  converter, the `evaluate` reduction and clobe's `imac`, validated against the compiled C++.
- `psp.pspMany()`: many independent partitioning runs advanced in lockstep with one model
  call per iteration, and `notebooks/oscar/run_oscar.py`, the participant-level OSCAR run.
- Notebooks replicating the articles of the R package: the prospect theory tutorial, the
  ALCOVE recipe (with `notebooks/alcove/run_alcove.py`, which runs the long analysis), and
  the benchmarks.
- `benchmarks/`: the workloads and experiments of the R package for the torch port,
  `compare.py` to benchmark the R package and the port on the same machine,
  `devices.py` to measure when the GPU pays off, and `alcove_benchmark.py` with
  `r/alcove_benchmark.R` to compare time and regions found on ALCOVE's full parameter space.

### Changed, compared with the R package

- `ordinal_patterns` has shape `(patterns, dimensionality, dimensionality)`, pattern `k`
  being `ordinal_patterns[k]`.
- Pattern ids in the saved files are 0-based, the index into `ordinal_patterns`, and the
  header has no trailing comma.
- `radius` may be one number per parameter; `responses`, `dimensionality`,
  `parameter_names` and `stimuli_names` are optional.
- Random numbers come from torch (`torch.manual_seed()` or a `torch.Generator`), so runs
  are not bit-identical to R runs with the same seed.
- The deprecated `psp_global()` and `psp_control()` are not ported.
