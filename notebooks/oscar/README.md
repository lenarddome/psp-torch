# OSCAR on dome2025ibre: the participant-level PSP run

One parameter space partitioning run per participant of the inverse base-rate effect data (354 participants, each on their own trial order), for the OSCAR model. This is the port of `psp.R` / `run.r`; the model is `psp.models.oscar`, validated against the compiled C++ to 1e-10 (`tests/test_oscar.py`).

## What one run does

For a participant, `PSP` searches the five parameters P, α, μ, ρ, β within the bounds (1, 5), (0.001, 1), (0.001, 1), (0.001, 5), (0.001, 5). The model runs OSCAR through that participant's training and test trials (supervised design: training trials learn with feedback, test trials only respond) and returns, behind a 0.5 anchor, the mean P(rare) at test for A, B, BC and C. The ordinal pattern is clobe's inequality matrix of those five values with thresholds ±0.125; any NaN makes the whole matrix 99. As in `evaluate()` of `imac.R`, the initial saliences and weights are drawn afresh, uniform in (0, 0.001), on every evaluation.

Settings, as in `run.r`: 1,000 random starting points (one matrix of starts shared by all participants, as the R routine does), radius 1, population 1,000,000 (so every pattern keeps proposing until the cap), 10,000 iterations.

## How the runs are batched

All participants advance in lockstep (`psp.PSPLockstep`): each iteration, the proposals of every run are concatenated and OSCAR is called once with every row routed to its participant's trial sequence (padded to 112 trials). Each participant keeps its own pattern store, counts, files and result, exactly as separate `PSP` calls would. `--group N` limits how many participants share a group, if memory is tight.

## Running it

```sh
uv run python notebooks/oscar/run_oscar.py --smoke                       # 2 participants, 3 iterations, CPU
uv run python notebooks/oscar/run_oscar.py --device cuda --dtype float32  # the full run
```

The run resumes: participants already listed in `results/tracking.csv` are skipped, so a stopped run restarts with the same command. Options: `--iterations`, `--population`, `--starts`, `--radius`, `--thresholds`, `--seed`, `--first N` (only the first N participants), `--participants id1,id2`, `--group`, `--no-save`, `--keep-self-column` (see below).

## What to expect on a large GPU

One batched OSCAR call costs about 112 trials × (10 recurrent attention iterations × ~10 kernels + ~25) ≈ 14,000 kernel launches, whatever the batch size, until the batch is large enough to make the kernels themselves the cost. Measured here (NVIDIA T400, float32): 0.5 s per call for up to 1,000 rows, 1.0 s for 10,000 rows. The CPU (12 threads, float64) takes 0.15 s for 1 row but 4.3 s for 10,000.

For the full run: iteration 0 evaluates 354 × 1,000 starting points in one call; every later iteration evaluates one proposal per pattern per participant, so with 50 to 150 patterns per participant that is 20,000 to 50,000 rows per call. On a card like an RTX 3090 expect a few tenths of a second per iteration, so 10,000 iterations in the order of one to two hours of GPU time. Float32 is enough for this model (the ordinal pattern depends on differences of 0.125) and is several times faster than float64 on consumer cards.

Saving is the other cost. With the default settings every participant's run evaluates about a million parameter sets, and `--no-save` skips the two per-evaluation csv files (about 200 MB per participant, tens of GB in total, and a comparable amount of CPU time to write). The pattern files, `frequencies.csv` and `tracking.csv` are always written and are what the R routine's `.RData` and `frequencies.csv` hold.

## Two details of the C++ that the port reproduces

- The initial state is random on every evaluation, so the same parameter set can produce different patterns in different evaluations. The port draws from the torch generator (`--seed`).
- `AttentionShift` calls `.diag().zeros()` on a one-row matrix, which zeroes the attended activation of node A (column 0) for every node rather than each node's own column. The port does the same by default; `--keep-self-column` zeroes each node's own column instead.

## Outputs

| File | Contents |
| --- | --- |
| `OSCAR_4.<ppt>_parameters.csv`, `OSCAR_4.<ppt>_continuous.csv` | every evaluation: iteration, parameters / the five probabilities, 0-based pattern id |
| `patterns/OSCAR_4.<ppt>.npz` | `ordinal_patterns` (patterns, 5, 5) and `ordinal_counts` |
| `frequencies.csv` | `ppt, dataset, id, frequency`: the population of each pattern (ids 1-based, as in R) |
| `tracking.csv` | `ppt, started, completed, iterations, patterns, evaluations` |
