# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-10-20

### Added

- `.github/workflows/publish.yml` builds the package and uploads it to PyPI, with trusted
  publishing, when a GitHub release is published; the release tag must match the version.

### Changed

- The links and the logo in the README are absolute URLs, so they work on PyPI.

## [0.2.0] - 2026-10-20

### Added

- `psp.clobe`: a port of the R package clobe 0.8.0 (Dome and Wills, 2025): `imac`, `patterns`,
  `gdistance`, `pattern_distance`, `orderings`, `brutes`, `stirling2`, `fubini` and
  `semiorders`, checked against the R package's test suite and outputs exported from it.
  `imac` is batched and fits the `discretize` contract of `PSP()`; counts are exact integers;
  `brutes` runs on any torch device. For float32 and lower precision, `imac` compares
  differences with thresholds within 8 eps of the dtype instead of R's `sqrt(.Machine$double.eps)`.
  `imac`, `patterns` and `brutes` take `inclusive=True` for the rule of clobe before 0.8.0, which
  counts a difference equal to a threshold as larger or smaller rather than approximately equal.
- `notebooks/clobe/`: the four vignettes of the R package clobe, ported to `psp.clobe` and aligned
  with the published paper (Dome and Wills, 2025), with figures throughout, and a tutorial on the
  zone of model adequacy (a split-half analysis of the human data).
- `notebooks/figure_style.py`: one publication-ready figure style for every notebook; each figure
  is also saved as a vector PDF and a 300 dpi PNG under `notebooks/figures/` (not tracked).
- `benchmarks/clobe/` and `notebooks/benchmarks-clobe.ipynb`: `psp.clobe` against clobe 0.8.0 on
  the same inputs, function by function, on the CPU and the GPU.
- `docs/`: a documentation site built with Sphinx: getting started, a user guide, the tutorials,
  the clobe vignettes and the benchmarks (the notebooks, shown with their saved outputs), and an
  API reference for `psp`, `psp.clobe` and the built-in models, generated from the docstrings.
  `uv run --group docs sphinx-build -b html docs docs/_build/html` builds it locally.
- Continuous integration: `.github/workflows/ci.yml` lints, type-checks and runs the test suite on
  Python 3.10 to 3.12 (CPU), runs the tests against the built wheel, and executes the tutorial and
  the ALCOVE recipe; `.github/workflows/docs.yml` builds the documentation on every pull request,
  failing on any warning or broken cross-reference, and deploys it to GitHub Pages from `main`.

### Changed

- `psp.models.oscar.imac` now calls `psp.clobe.imac(..., inclusive=True)`; its output is unchanged.
- `psp.pspGlobal()` is now `psp.PSP()`; its arguments and result are unchanged.
- `psp.pspMany()` is now `psp.PSPLockstep()`, and its model type `psp.ManyModel` is now
  `psp.LockstepModel`; arguments and results are unchanged.
- The tutorial no longer has the interactive "Explore it yourself" figure: its cell raised an
  `IndexError`, and its widget made `notebooks/execute.py` hang. The tutorial is re-executed.
- References: the README and the documentation list every work they cite, each checked against
  its publisher's record; Kahneman and Tversky (1979) is now complete, and Shepard, Hovland and
  Jenkins (1961), Nosofsky et al. (1994) and catlearn (Wills et al., 2025) are added. OSCAR is
  cited as Dome and Wills (2026, arXiv:2608.06149) in `psp.models.oscar`, the OSCAR notebooks,
  READMEs and benchmark scripts.

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
