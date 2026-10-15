# clobe

The vignettes of the R package [clobe](https://github.com/lenarddome/clobe) 0.8.0, ported to `psp.clobe`. The text follows the R vignettes and the published paper (Dome & Wills, 2025); the code is rewritten for the port, and the notes on where the port differs (matrices stacked as `(n, k, k)`, indices from 0, exact counts, `inclusive`, the float32 tolerance, `brutes()` on the GPU) are added where they apply. The simulated data come from torch's random number generator, so the numbers differ from the R vignettes.

| Notebook | R vignette | What it shows |
| --- | --- | --- |
| [getting-started.ipynb](getting-started.ipynb) | `vignette("clobe")` | The whole workflow on simulated data, from results to g-distance, and the same with a `psp.PSP` run in place of a grid. |
| [inequality-matrices.ipynb](inequality-matrices.ipynb) | `vignette("inequality-matrices")` | How a result becomes an ordinal pattern; semiorders and weak orderings; rounding, float32 and inclusive thresholds. |
| [universal-set.ipynb](universal-set.ipynb) | `vignette("universal-set")` | Counting and listing every possible pattern, and when the measurement scale matters. |
| [gdistance.ipynb](gdistance.ipynb) | `vignette("gdistance")` | The sets of g-distance, accommodation (ACC) and unobserved model patterns (UMP), Equations 1 to 4 of the paper, a worked example, models on the α-β plane against the PAS, degenerate and anti-PAS models, the effect of the weight, and near misses. |
| [zone-of-adequacy.ipynb](zone-of-adequacy.ipynb) | (new) | The zone of model adequacy: a split-half analysis of the human data, as in Dome and Wills (2025), with 10,000 splits computed at once, and where each model falls against it. |

[../benchmarks-clobe.ipynb](../benchmarks-clobe.ipynb) compares the speed of `psp.clobe` with the R package.

To run them again:

```sh
uv run python notebooks/execute.py clobe/getting-started.ipynb clobe/inequality-matrices.ipynb \
    clobe/universal-set.ipynb clobe/gdistance.ipynb clobe/zone-of-adequacy.ipynb
```
