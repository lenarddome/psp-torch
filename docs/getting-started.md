# Getting started

## Install

```sh
pip install git+https://github.com/lenarddome/psp-torch
```

The library needs only `torch` and `numpy`. Two extras add more:

- `notebooks`: what the notebooks and {func}`psp.read_saved` need (`pandas`, `matplotlib`,
  `jupyter`, `ipywidgets`, `scikit-learn`).
- `fast-save`: `pyarrow`, which makes `save=True` several times faster.

```sh
pip install "psp-torch[notebooks,fast-save] @ git+https://github.com/lenarddome/psp-torch"
```

For development, the project is managed with [uv](https://docs.astral.sh/uv/):

```sh
uv sync                                   # CUDA 12.8 build of torch, dev tools and notebook extras
uv sync --no-group cu128 --group cpu      # CPU-only build of torch instead
uv run pytest
uv run ruff check psp tests && uv run mypy psp
uv run --group docs sphinx-build -b html docs docs/_build/html   # this site
```

## A first partition

A model of risky choice with two parameters, and the pattern of choices it makes on four bets.
The [tutorial](notebooks/tutorial.ipynb) explains the model.

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

{func}`psp.models.prospect.model` maps a `(n, 2)` tensor of parameter sets to a `(n, 4)` tensor
of predictions, and {func}`psp.models.prospect.discretize` maps those to `(n, 4, 4)` ordinal
patterns. Two parameter sets belong to the same region when their patterns are identical.

## Where next

- The [user guide](guide.md) covers every argument of {func}`psp.PSP`, running many searches at
  once, and the GPU.
- The [tutorial](notebooks/tutorial.ipynb) is a complete first analysis, and the
  [ALCOVE recipe](notebooks/recipe-alcove.ipynb) a longer one on a slow model.
