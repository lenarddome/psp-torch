# User guide

## The interface

```python
PSP(model, discretize, control, save=False, path=".", extension=".csv", quiet=False,
    *, batched=True, device=None, dtype=None, generator=None) -> PSPResult
```

| Argument | Meaning |
| --- | --- |
| `model` | Maps parameter sets to the model's continuous outputs: a `(n, dimensions)` tensor in, `(n, responses)` out. |
| `discretize` | Maps model outputs to ordinal patterns: `(n, responses)` in, `(n, dimensionality, dimensionality)` out. `-0.0` equals `0.0` and `NaN` equals `NaN`. |
| `control` | A mapping with the keys of the R package, below, or a {class}`psp.Control`. |
| `save`, `path`, `extension` | Write every evaluated parameter set to `<path>_parameters<extension>` and every model output to `<path>_continuous<extension>`, with the iteration and the 0-based pattern id; read them back with {func}`psp.read_saved`. Values are written exactly (shortest round-trip form). With [pyarrow](https://arrow.apache.org/docs/python/) installed (the `fast-save` extra) the files are written several times faster. |
| `quiet` | Do not print the iteration number. |
| `batched` | `False` calls `model` and `discretize` once per parameter set, as the R package does; the same as wrapping both with {func}`psp.rowwise`. |
| `device`, `dtype` | Where the search runs. Defaults to the device and dtype of `control["init"]`, or CPU and `float64`. |
| `generator` | A {class}`torch.Generator` on `device` for reproducible proposals; by default the global one, seeded with {func}`torch.manual_seed`. |

| `control` key | Meaning |
| --- | --- |
| `init` | Starting parameter sets, one per row. |
| `lower`, `upper` | Bounds of each parameter; proposals that cross them are moved onto them. |
| `radius` | Largest step in any one parameter, a number or one per parameter. |
| `population` | Parameter sets each pattern needs before the search stops proposing from it; `0` or `None` for no limit. |
| `iterations` | Cap on the number of iterations; `0` or `None` for no cap. One of `population` and `iterations` must be set. |
| `parameter_names`, `stimuli_names` | Column names of the saved files (optional). |
| `dimensionality`, `responses` | Size of the pattern matrix and length of the model output, checked on every evaluation (optional). |

The result, a {class}`psp.PSPResult`, has `ordinal_patterns` of shape
`(patterns, dimensionality, dimensionality)`, `ordinal_counts` of shape `(patterns,)` and
`iterations`, as attributes or as keys: `result["ordinal_counts"]` works like the R list.

## Many independent runs at once

{func}`psp.PSPLockstep` runs one independent search per control (for example one per
participant, each on its own trial order) and advances them in lockstep: every iteration, the
proposals of all runs go to `model` in one call, with the index of the run each row belongs to.
Each run keeps its own pattern store, files and result. This is how hundreds of small searches
fill a GPU.

```python
results = PSPLockstep(lambda par, run: evaluate(par, training, sequence=run), discretize, controls)
```

## Models written for one parameter set

Any function that takes one parameter vector and returns one output vector, as `model` does in
R, works with `batched=False` or through {func}`psp.rowwise`, which can also hand the rows over
as NumPy arrays:

```python
from psp import PSP, rowwise

def model(par):          # par: one parameter set, a 1-D tensor
    return par.cumsum(0)

result = PSP(model, discretize, control, batched=False)
result = PSP(rowwise(model, numpy=True), discretize, control)   # par is a numpy array
```

## Using the GPU

Pass `device="cuda"` (and, when the model allows it, `dtype=torch.float32`). The proposals,
bounds and centres then live on the GPU and `model` receives CUDA tensors.

The GPU pays off only when one call of `model` does real tensor work for many parameter sets at
once, and a PSP run only produces large batches while many patterns are still filling. For a
model that runs in microseconds, or a search whose iterations propose a handful of parameter
sets, the CPU is faster, because each iteration costs a few kernel launches whatever the batch
size; and on consumer cards float64 is slow. The [benchmarks](notebooks/benchmarks.ipynb)
measure where the line falls for three models, and
[`benchmarks/devices.py`](https://github.com/lenarddome/psp-torch/blob/main/benchmarks/devices.py)
is a template for measuring it for yours. The [architecture](architecture.md) page shows what
runs where.

## Models shipped with the package

- {mod}`psp.models.prospect`: prospect theory on four bets of Kahneman and Tversky (1979), the
  model of the [tutorial](notebooks/tutorial.ipynb).
- {mod}`psp.models.alcove`: ALCOVE (Kruschke, 1992) on the six category structures of Shepard,
  Hovland and Jenkins (1961): a batched port of the `slpALCOVE()` of catlearn (Wills et al.,
  2025) that reproduces catlearn's trial-by-trial output to machine precision, with catlearn's
  training orders and the `nosof94` data of Nosofsky et al. (1994) shipped as package data. The
  model of the
  [ALCOVE recipe](notebooks/recipe-alcove.ipynb).
- {mod}`psp.models.oscar`: OSCAR (Dome & Wills, 2026;
  [arXiv:2608.06149](https://doi.org/10.48550/arXiv.2608.06149)) on the inverse base-rate effect
  data of 354 participants (`dome2025ibre`, shipped as package data), batched over parameter sets
  and participants, validated against the C++ implementation.
- {mod}`psp.models.voronoi`: a cheap toy model whose regions are Voronoi cells, used by the tests
  and benchmarks.

## Comparing models with human data

{mod}`psp.clobe` is a set of tools supporting the analyses of Dome and Wills (2025), for comparing
the heterogeneity of a model with that of people: inequality matrices ({func}`~psp.clobe.imac`,
{func}`~psp.clobe.patterns`), g-distance ({func}`~psp.clobe.gdistance`), distances between
patterns ({func}`~psp.clobe.pattern_distance`), and the universal set of possible patterns,
counted ({func}`~psp.clobe.semiorders`, {func}`~psp.clobe.fubini`, {func}`~psp.clobe.stirling2`)
or enumerated ({func}`~psp.clobe.orderings`, {func}`~psp.clobe.brutes`).
{func}`~psp.clobe.imac` is batched, so it can be the `discretize` of a run, and
{func}`~psp.clobe.gdistance` takes the patterns a run found:

```python
from functools import partial
from psp import PSP, clobe

thresholds = (0.1, -0.1)
result = PSP(model, partial(clobe.imac, thresholds=thresholds), control)

human = clobe.patterns(data, thresholds)   # (participants, stimuli) -> unique patterns
clobe.gdistance(human.inequality_matrices, result.ordinal_patterns,
                universal=clobe.semiorders(data.shape[1]), weight=0.5,
                frequencies=human.frequencies)
```

The [clobe vignettes](clobe.md) work through each of these.

## Differences from the R package

- `model` and `discretize` are batched by default (see above).
- `ordinal_patterns` is indexed `[k]`, not `[, , k]`, and the saved pattern ids are 0-based.
- The saved csv header has no trailing comma; values are written in their shortest round-trip
  form.
- Random numbers come from torch, so a run is not bit-identical to an R run with the same seed.
  The sampler is otherwise the same: one hypersphere proposal per underpopulated pattern per
  iteration, centred on the last parameter set that produced the pattern, with the same clamping
  and the same pattern identity rules.
- `radius` may be set per parameter, and `responses`, `dimensionality` and the names are
  optional.
- The deprecated `psp_global()` and `psp_control()` are not ported.
- In {mod}`psp.clobe`, matrices are stacked as `(n, k, k)` rather than `(k, k, n)`, indices are
  0-based, counts are exact Python integers, {func}`~psp.clobe.brutes` runs on any device
  instead of a thread pool, and {func}`~psp.clobe.gdistance` always returns `alpha`, `beta` and
  `accommodation` (R's `xtdo = TRUE`).

## References

Dome, L., & Wills, A. J. (2025). g-Distance: On the comparison of model and human heterogeneity.
*Psychological Review, 132*(3), 632–655. <https://doi.org/10.1037/rev0000550>

Dome, L., & Wills, A. J. (2026). Errorless Irrationality: A unified computational account of the
inverse base-rate effect across predictive, observational, and unsupervised procedures
(arXiv:2608.06149). arXiv. <https://doi.org/10.48550/arXiv.2608.06149>

Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk.
*Econometrica, 47*(2), 263–292. <https://doi.org/10.2307/1914185>

Kruschke, J. K. (1992). ALCOVE: An exemplar-based connectionist model of category learning.
*Psychological Review, 99*(1), 22–44. <https://doi.org/10.1037/0033-295X.99.1.22>

Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P. (1994). Comparing
models of rule-based classification learning: A replication and extension of Shepard, Hovland,
and Jenkins (1961). *Memory & Cognition, 22*(3), 352–369. <https://doi.org/10.3758/BF03200862>

Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of
classifications. *Psychological Monographs: General and Applied, 75*(13), 1–42.
<https://doi.org/10.1037/h0093825>

Wills, A. J., Dome, L., Edmunds, C., Honke, G., Inkster, A., Schlegelmilch, R., & Spicer, S.
(2025). catlearn: Formal Psychological Models of Categorization and Learning. R package version
1.1. <https://CRAN.R-project.org/package=catlearn>
