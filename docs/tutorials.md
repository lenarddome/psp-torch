# Tutorials

Each is a notebook shown with the outputs it was committed with; the source is in
[`notebooks/`](https://github.com/lenarddome/psp-torch/tree/main/notebooks).

- **What can prospect theory predict?** A complete first analysis: a two-parameter model of risky
  choice, how to write `model` and `discretize` for a batch of parameter sets, running
  {func}`psp.PSP`, checking the partition against brute force, and checking that the search found
  everything.
- **ALCOVE and the six category structures.** The analysis of Pitt et al. (2006) on a slow
  learning model with four parameters: patterns that allow ties, planning and saving a long run,
  reading a partition you cannot plot, and robustness checks.

```{toctree}
:maxdepth: 1

notebooks/tutorial
notebooks/recipe-alcove
```

## References

Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter
space partitioning. *Psychological Review, 113*(1), 57–83.
<https://doi.org/10.1037/0033-295X.113.1.57>
