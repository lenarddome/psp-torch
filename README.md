# psp-torch

[![psp: parameter space partitioning](https://raw.githubusercontent.com/lenarddome/psp-torch/main/docs/assets/psp-lockup-light.svg)](https://lenarddome.github.io/psp-torch/)

Parameter space partitioning MCMC for global model evaluation (Pitt, Kim, Navarro & Myung, 2006), in PyTorch. The methods implemented here were designed to support the type of analyses done by Dome & Wills (2025).

- [Documentation](https://lenarddome.github.io/psp-torch/)
- [Tutorial: what can prospect theory predict?](https://github.com/lenarddome/psp-torch/blob/main/notebooks/tutorial.ipynb)
- [Recipe: ALCOVE and the six category structures](https://github.com/lenarddome/psp-torch/blob/main/notebooks/recipe-alcove.ipynb)
- [Benchmarks: the port against the R package, and when the GPU pays off](https://github.com/lenarddome/psp-torch/blob/main/notebooks/benchmarks.ipynb) (how to run them: [benchmarks/README.md](https://github.com/lenarddome/psp-torch/blob/main/benchmarks/README.md))
- [Architecture: how ALCOVE and the sampler run on the GPU](https://github.com/lenarddome/psp-torch/blob/main/docs/architecture.md)
- [Changelog](https://github.com/lenarddome/psp-torch/blob/main/CHANGELOG.md)

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

## Citing psp-torch

If you are using psp-torch in your work, please cite the following papers:

* Dome, L., & Wills, A. J. (2025). g-Distance: On the comparison of model and human heterogeneity. Psychological Review, 132(3), 632–655. https://doi.org/10.1037/rev0000550
* Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter space partitioning. Psychological Review, 113(1), 57–83. https://doi.org/10.1037/0033-295X.113.1.57

```bibtex
@article{dome2025gdistance,
  title = {g-{{Distance}}: {{On}} the Comparison of Model and Human Heterogeneity.},
  author = {Dome, Lenard and Wills, Andy J.},
  year = 2025,
  journal = {Psychological Review},
  volume = {132},
  number = {3},
  pages = {632--655},
  publisher = {American Psychological Association},
  doi = {10.1037/rev0000550},
  langid = {english},
} 

@article{pitt2006global,
  title = {Global Model Analysis by Parameter Space Partitioning.},
  author = {Pitt, Mark A. and Kim, Woojae and Navarro, Daniel J. and Myung, Jay I.},
  year = 2006,
  journal = {Psychological Review},
  volume = {113},
  number = {1},
  pages = {57--83},
  publisher = {American Psychological Association},
  doi = {10.1037/0033-295X.113.1.57},
}
```

## Quick start

A model of risky choice with two parameters, and the pattern of choices it makes on four bets (the [tutorial](https://github.com/lenarddome/psp-torch/blob/main/notebooks/tutorial.ipynb) explains the model):

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

## Documentation

To learn how to use psp-torch, and more about the type of analyses it can support, please visit the package documentation [here](https://lenarddome.github.io/psp-torch/). 

## License

AGPL-3.0, see [LICENSE](https://github.com/lenarddome/psp-torch/blob/main/LICENSE).
