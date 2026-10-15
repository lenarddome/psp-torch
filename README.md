# psp-torch

Parameter space partitioning MCMC for global model evaluation (Pitt, Kim, Navarro & Myung, 2006), in PyTorch.

`psp-torch` is a port of the R package [psp](https://lenarddome.com/psp/) (Dome & Wills) that keeps its interface: `PSP()`, the R package's `pspGlobal()`, takes a `model`, a `discretize` function and a `control` mapping, and returns every ordinal pattern the model produced together with the population of its region. Unlike the R package, it evaluates the model on a whole batch of parameter sets at once, so a run can use the GPU, and models written as tensor code need no loop.

- [Tutorial: what can prospect theory predict?](notebooks/tutorial.ipynb)
- [Recipe: ALCOVE and the six category structures](notebooks/recipe-alcove.ipynb)
- [Benchmarks: the port against the R package, and when the GPU pays off](notebooks/benchmarks.ipynb) (how to run them: [benchmarks/README.md](benchmarks/README.md))
- [Architecture: how ALCOVE and the sampler run on the GPU](docs/architecture.md)
- [Changelog](CHANGELOG.md)

## Install

```sh
pip install git+https://github.com/lenarddome/psp-torch
```

The library needs only `torch` and `numpy`. The notebooks and `psp.read_saved()` need the `notebooks` extra (`pandas`, `matplotlib`, `jupyter`, `ipywidgets`, `scikit-learn`); the `fast-save` extra (`pyarrow`) speeds up `save=True`.

For development, the project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                                   # CUDA 12.8 build of torch, dev tools and notebook extras
uv sync --no-group cu128 --group cpu      # CPU-only build of torch instead
uv run pytest
uv run ruff check psp tests && uv run mypy psp
```

## Quick start

A model of risky choice with two parameters, and the pattern of choices it makes on four bets (the [tutorial](notebooks/tutorial.ipynb) explains the model):

```python
import torch
from psp import PSP
from psp.models import prospect

torch.manual_seed(2026)
control = {
    "lower": [0.2, 0.3],
    "upper": [1.0, 1.2],
    "init": torch.stack([torch.rand(5) * 0.8 + 0.2, torch.rand(5) * 0.9 + 0.3], dim=1),
    "radius": 0.5,
    "population": 200,
    "iterations": 30000,
    "parameter_names": ["alpha", "gamma"],
    "dimensionality": 4,
    "responses": 4,
}
result = PSP(prospect.model, prospect.discretize, control, quiet=True)
for pattern, count in zip(result.ordinal_patterns, result.ordinal_counts):
    print(f"{int(count):>5}  {prospect.describe(pattern)}")
```

`prospect.model` maps a `(n, 2)` tensor of parameter sets to a `(n, 4)` tensor of predictions, and `prospect.discretize` maps those to `(n, 4, 4)` ordinal patterns. Two parameter sets belong to the same region when their patterns are identical.

## The interface

```python
PSP(model, discretize, control, save=False, path=".", extension=".csv", quiet=False,
          *, batched=True, device=None, dtype=None, generator=None) -> PSPResult
```

| Argument | Meaning |
| --- | --- |
| `model` | Maps parameter sets to the model's continuous outputs: a `(n, dimensions)` tensor in, `(n, responses)` out. |
| `discretize` | Maps model outputs to ordinal patterns: `(n, responses)` in, `(n, dimensionality, dimensionality)` out. `-0.0` equals `0.0` and `NaN` equals `NaN`. |
| `control` | A mapping with the keys of the R package, below. |
| `save`, `path`, `extension` | Write every evaluated parameter set to `<path>_parameters<extension>` and every model output to `<path>_continuous<extension>`, with the iteration and the 0-based pattern id; read them back with `psp.read_saved(path)`. Values are written exactly (shortest round-trip form). With [pyarrow](https://arrow.apache.org/docs/python/) installed (`pip install psp-torch[fast-save]`) the files are written several times faster. |
| `quiet` | Do not print the iteration number. |
| `batched` | `False` calls `model` and `discretize` once per parameter set, as the R package does; the same as wrapping both with `psp.rowwise()`. |
| `device`, `dtype` | Where the search runs. Defaults to the device and dtype of `control["init"]`, or CPU and `float64`. |
| `generator` | A `torch.Generator` on `device` for reproducible proposals; by default the global one, seeded with `torch.manual_seed()`. |

| `control` key | Meaning |
| --- | --- |
| `init` | Starting parameter sets, one per row. |
| `lower`, `upper` | Bounds of each parameter; proposals that cross them are moved onto them. |
| `radius` | Largest step in any one parameter, a number or one per parameter. |
| `population` | Parameter sets each pattern needs before the search stops proposing from it; `0` or `None` for no limit. |
| `iterations` | Cap on the number of iterations; `0` or `None` for no cap. One of `population` and `iterations` must be set. |
| `parameter_names`, `stimuli_names` | Column names of the saved files (optional). |
| `dimensionality`, `responses` | Size of the pattern matrix and length of the model output, checked on every evaluation (optional). |

The result has `ordinal_patterns` of shape `(patterns, dimensionality, dimensionality)`, `ordinal_counts` of shape `(patterns,)` and `iterations`, as attributes or as keys: `result["ordinal_counts"]` works like the R list.

### Many independent runs at once

`psp.PSPLockstep(model, discretize, controls, ...)` runs one independent search per control (for example one per participant, each on its own trial order) and advances them in lockstep: every iteration, the proposals of all runs go to `model` in one call, with the index of the run each row belongs to. Each run keeps its own pattern store, files and result. This is how hundreds of small searches fill a GPU.

```python
results = PSPLockstep(lambda par, run: evaluate(par, training, sequence=run), discretize, controls)
```

### Models written for one parameter set

Any function that takes one parameter vector and returns one output vector, as `model` does in R, works with `batched=False` or through `psp.rowwise()`, which can also hand the rows over as NumPy arrays:

```python
from psp import PSP, rowwise

def model(par):          # par: one parameter set, a 1-D tensor
    return par.cumsum(0)

result = PSP(model, discretize, control, batched=False)
result = PSP(rowwise(model, numpy=True), discretize, control)   # par is a numpy array
```

### Using the GPU

Pass `device="cuda"` (and, when the model allows it, `dtype=torch.float32`). The proposals, bounds and centres then live on the GPU and `model` receives CUDA tensors. The GPU pays off only when one call of `model` does real tensor work for many parameter sets at once, and a PSP run only produces large batches while many patterns are still filling. For a model that runs in microseconds, or a search whose iterations propose a handful of parameter sets, the CPU is faster, because each iteration costs a few kernel launches whatever the batch size; and on consumer cards float64 is slow. The [benchmarks notebook](notebooks/benchmarks.ipynb) measures where the line falls for three models, and `benchmarks/devices.py` is a template for measuring it for yours.

### Models shipped with the package

- `psp.models.prospect`: prospect theory on four bets of Kahneman and Tversky (1979), the model of the tutorial.
- `psp.models.alcove`: ALCOVE (Kruschke, 1992) on the six category structures of Shepard, Hovland and Jenkins (1961): a batched port of catlearn's `slpALCOVE()` that reproduces catlearn's trial-by-trial output to machine precision, with catlearn's training orders and the `nosof94` data shipped as package data. The model of the ALCOVE recipe, with the parameter ranges of Pitt et al. (`ALCOVE_PARAMETERS`) or the learning rates free up to 1 (`ALCOVE_PARAMETERS_FULL`).
- `psp.models.oscar`: OSCAR on the inverse base-rate effect data of 354 participants (`dome2025ibre`, shipped as package data), batched over parameter sets and participants, validated against the C++ implementation; with the trial-matrix converter, the `evaluate` reduction and clobe's `imac` inequality matrix. The model of the participant-level run in [notebooks/oscar](notebooks/oscar/README.md).
- `psp.models.voronoi`: a cheap toy model whose regions are Voronoi cells, used by the tests and benchmarks.

## Differences from the R package

- `model` and `discretize` are batched by default (see above).
- `ordinal_patterns` is indexed `[k]`, not `[, , k]`, and the saved pattern ids are 0-based.
- The saved csv header has no trailing comma; values are written in their shortest round-trip form.
- Random numbers come from torch, so a run is not bit-identical to an R run with the same seed. The sampler is otherwise the same: one hypersphere proposal per underpopulated pattern per iteration, centred on the last parameter set that produced the pattern, with the same clamping and the same pattern identity rules.
- `radius` may be set per parameter, and `responses`, `dimensionality` and the names are optional.
- The deprecated `psp_global()` and `psp_control()` are not ported.

## Repository

| Path | Contents |
| --- | --- |
| `psp/` | The library. |
| `tests/` | The test suite, a port of the R package's tests plus checks of the models against R output. |
| `notebooks/` | The tutorial, the ALCOVE recipe and the benchmarks, with `alcove/run_alcove.py` for the long run, `oscar/run_oscar.py` for the participant-level OSCAR run, and `execute.py` to refresh the notebooks. |
| `benchmarks/` | The benchmark scripts and their results; see [benchmarks/README.md](benchmarks/README.md) for what each measures and the commands to run them, here or on another machine. |
| `docs/` | The architecture diagrams (`alcove-gpu.svg`, `psp-gpu.svg`) and their notes. |
| `psp-reference/` | The source of the R package this port follows, for reference. |

## Citation

Dome, L., & Wills, A. J. (2026). psp: Parameter Space Partitioning MCMC for Global Model Evaluation. R package version 1.0.6. <https://CRAN.R-project.org/package=psp>

Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter space partitioning. *Psychological Review, 113*(1), 57–83. <https://doi.org/10.1037/0033-295X.113.1.57>

## License

AGPL-3.0, see [LICENSE](LICENSE).
