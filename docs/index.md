```{image} assets/psp-lockup-light.svg
:alt: psp, parameter space partitioning
:class: only-light psp-hero
```

```{image} assets/psp-lockup-dark.svg
:alt: psp, parameter space partitioning
:class: only-dark psp-hero
```

# psp-torch

Parameter space partitioning MCMC for global model evaluation (Pitt, Kim, Navarro & Myung,
2006), in PyTorch, built to evaluate a model on many parameter sets at once, on the CPU or the
GPU, and to support the kind of analyses of Dome & Wills (2025).

Parameter space partitioning asks what a model *can* do: which qualitatively different patterns of
behaviour it produces anywhere in its parameter space, and how much of the space each pattern
occupies. `psp-torch` finds those regions with a Markov chain Monte Carlo search, and
{mod}`psp.clobe` compares them with the patterns people produce.

```{figure} images/hypercube.jpg
:alt: Three cubes of three parameters each, filled with points coloured by the region they fall in.
:width: 100%

Three parameter spaces, partitioned. Each point is a parameter set the search evaluated,
coloured by the ordinal pattern it produced.
```

```sh
pip install git+https://github.com/lenarddome/psp-torch
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc

Install the package and run a first partition of a two-parameter model.
:::

:::{grid-item-card} User guide
:link: guide
:link-type: doc

The interface of {func}`psp.PSP`, many runs at once, models written for one parameter set,
and when the GPU pays off.
:::

:::{grid-item-card} Tutorials
:link: tutorials
:link-type: doc

What can prospect theory predict? And the analysis of Pitt et al. (2006) on ALCOVE.
:::

:::{grid-item-card} Comparing models with people
:link: clobe
:link-type: doc

Inequality matrices, the universal set and g-distance with {mod}`psp.clobe`.
:::

:::{grid-item-card} Benchmarks
:link: benchmarks
:link-type: doc

The port against the R package, and when the GPU is worth using.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function and class, from its docstring.
:::
::::

## Citation

If you use `psp-torch`, please cite the paper the methods were written for:

Dome, L., & Wills, A. J. (2025). g-Distance: On the comparison of model and human
heterogeneity. *Psychological Review, 132*(3), 632–655. <https://doi.org/10.1037/rev0000550>

Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter
space partitioning. *Psychological Review, 113*(1), 57–83.
<https://doi.org/10.1037/0033-295X.113.1.57>

```{toctree}
:hidden:

getting-started
guide
tutorials
clobe
api/index
benchmarks
architecture
changelog
```
