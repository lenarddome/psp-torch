# %% [markdown]
# # Trying out `psp.models.oscar`, one step at a time
#
# Run cell by cell (`# %%`, Shift+Enter in VS Code) or line by line in a REPL started at the
# repository root (`uv run python`). Every step prints what it made and checks it with an
# `assert`, so a cell that runs without an error has passed. The order follows the module:
# data -> trial matrices -> initial state -> gating -> OSCAR -> evaluate -> imac/discretize ->
# the C++ fixtures -> a small PSP run. The last cell (optional) runs the compiled C++ through R.

# %% Setup
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from psp import pspGlobal, pspMany
from psp.models import oscar as osc

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
FIXTURES = ROOT / "tests" / "data" / "oscar"
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("repository:", ROOT)

# %% 1. The constants of the module
print("parameters:", osc.OSCAR_PARAMETERS)
print("bounds lower:", osc.OSCAR_BOUNDS[0])
print("bounds upper:", osc.OSCAR_BOUNDS[1])
print("nodes:", osc.NODES)
print("stimuli (code = index):", dict(enumerate(osc.STIMULI)))
print("evaluated at test:", osc.EVALUATED_STIMULI)
print("trial columns:", osc.TRIAL_COLUMNS)
assert len(osc.TRIAL_COLUMNS) == 19 and osc.TRIAL_COLUMNS[osc.COLSKIP] == "x1"

# %% 2. The dataset: dome2025ibre() and IbreData
data = osc.dome2025ibre()
ids = data.participants
print(f"{len(ids)} participants, {len(data.ppt):,} trials")
print("phases:", set(data.phase), " responses:", set(data.abresp))
assert len(ids) == 354

first = data.participant(ids[0])
print(f"participant {ids[0]}: {len(first.ppt)} trials, {(first.phase == 'test').sum()} at test")
print(pd.DataFrame({"stim": first.abstim, "trial": first.trial, "phase": first.phase, "resp": first.abresp}).head(8))
assert (first.phase == "test").sum() == 72

try:
    data.participant("nobody")
except KeyError as error:
    print("unknown participant ->", error)

# %% 3. convert_to_state_list() on a tiny made-up schedule
schedule = ["AB", "AC", "AB", "A", "BC"]
block = [4, 4, 4, 2, 2]  # 4 = supervised training, 2 = test
toy = osc.convert_to_state_list(schedule, block, trials=[1, 2, 3, 4, 5], ppt=7)
print(pd.DataFrame(toy, columns=osc.TRIAL_COLUMNS).astype(int))
assert toy[0, 0] == 3  # the first trial resets the network
assert list(toy[1:, 0]) == [4, 4, 2, 2]
assert list(toy[0, 14:]) == [0, 0, 0, 1, 0]  # AB: feedback on the common outcome (X)
assert list(toy[3, 14:]) == [0, 0, 0, 0, 0]  # no feedback at test

unsup = osc.convert_to_state_list(schedule, [0, 0, 0, 2, 2], trials=[1, 2, 3, 4, 5], design="unsupervised")
print("unsupervised control codes:", unsup[:, 0], " eligible outputs in training:", unsup[1, 9:14])
assert unsup[0, 0] == 1 and not unsup[1, 9:14].any()

# %% 4. participant_training() and training_batch(): the padded batch of trial matrices
one = osc.participant_training(data, ids[0])
print("one participant:", one.shape, " control codes:", sorted(set(one[:, 0])))
assert one[0, 0] == 3 and set(one[1:, 0]) == {2, 4}

training, members = osc.training_batch(data, ids[:3])
lengths = [len(data.participant(p).ppt) for p in members]
print("batch:", tuple(training.shape), " trials per participant:", lengths)
padded = (training[:, :, 0] == osc.PAD).sum(dim=1)
print("padding rows per participant:", padded.tolist())
assert training.shape[1] == max(lengths)
assert padded.tolist() == [training.shape[1] - n for n in lengths]
assert torch.equal(training[0, : lengths[0]], torch.tensor(one))

# %% 5. random_initial_state(): uniform in (0, 0.001), zero diagonal on eta
g = torch.Generator().manual_seed(1)
eta0, weights0 = osc.random_initial_state(4, generator=g)
print("eta0[0]:\n", eta0[0])
print("weights0 range:", float(weights0.min()), float(weights0.max()))
assert eta0.shape == weights0.shape == (4, 5, 5)
assert torch.all(torch.diagonal(eta0, dim1=1, dim2=2) == 0)
assert 0 <= float(weights0.min()) and float(weights0.max()) < 1e-3

again, _ = osc.random_initial_state(4, generator=torch.Generator().manual_seed(1))
assert torch.equal(eta0, again)  # same seed, same draws
print("same seed gives the same state: ok")

# %% 6. _gating(): the p-norm attention gating (equation 1)
x = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)  # stimulus AB
eta = torch.full((1, 5, 5), 0.5, dtype=torch.float64)
P = torch.tensor([2.0], dtype=torch.float64).reshape(-1, 1, 1)
gated, pnorm = osc._gating(x, eta, P)
print("gated input of node 0:", gated[0, 0], " p-norm:", pnorm[0])
# after gating, every node's gated input has p-norm 1
norms = (gated**2).sum(dim=2) ** 0.5
print("p-norms after gating:", norms)
assert torch.allclose(norms, torch.ones_like(norms))

# %% 7. oscar(): one parameter set on one participant, with every trial returned
params = {"P": 3.0, "alpha": 0.59, "mu": 0.35, "rho": 2.0, "beta": 3.0}  # the default of imac.R
prm = {k: torch.tensor([v], dtype=torch.float64) for k, v in params.items()}
seq = torch.tensor(one)
eta0, weights0 = osc.random_initial_state(1, generator=torch.Generator().manual_seed(2))
out = osc.oscar(prm, seq, eta0=eta0, weights0=weights0, return_trials=True)
print({k: tuple(v.shape) for k, v in out.items()})
probs = pd.DataFrame(out["prob"][0].numpy(), columns=osc.NODES)
probs["stim"] = [osc.STIMULI[int(s)] for s in one[:, 3]]
probs["ctrl"] = one[:, 0].astype(int)
print("first trials:\n", probs.head(), "\nlast trials:\n", probs.tail())
assert torch.allclose(out["prob"][0].sum(dim=1), torch.ones(len(one), dtype=torch.float64))
assert torch.all(torch.diagonal(out["weights"], dim1=1, dim2=2) == 0)
off_diagonal = out["eta"][0][~torch.eye(5, dtype=torch.bool)]
print("smallest salience after training:", float(off_diagonal.min()), " floor:", osc.ETA_FLOOR)
assert float(off_diagonal.min()) >= osc.ETA_FLOOR

# P(rare) at test, per stimulus, from the sums
means = out["rare_sum"] / out["rare_count"]
print("test trials per stimulus:", dict(zip(osc.STIMULI, out["rare_count"][0].int().tolist(), strict=True)))
print("mean P(rare) at test:", dict(zip(osc.STIMULI, means[0].numpy().round(3), strict=True)))
test = probs[probs.ctrl == 2].groupby("stim").rare.mean()
assert np.allclose(test.to_numpy(), means[0, [osc.STIMULI.index(s) for s in test.index]].numpy())

# %% 8. oscar(): a batch, each row on its own participant's sequence
B = 6
g = torch.Generator().manual_seed(3)
lower, upper = (torch.tensor(b, dtype=torch.float64) for b in osc.OSCAR_BOUNDS)
par = lower + torch.rand((B, 5), generator=g, dtype=torch.float64) * (upper - lower)
batch_params = {k: par[:, i] for i, k in enumerate(osc.OSCAR_PARAMETERS)}
sequence = torch.tensor([0, 1, 2, 0, 1, 2])
eta0, weights0 = osc.random_initial_state(B, generator=g)
batched = osc.oscar(batch_params, training, eta0=eta0, weights0=weights0, sequence=sequence)
print("rare_sum:", tuple(batched["rare_sum"].shape))

# the same rows one at a time, each on its unpadded sequence, give the same result
for r in range(B):
    single = osc.oscar(
        {k: v[r : r + 1] for k, v in batch_params.items()},
        torch.tensor(osc.participant_training(data, members[int(sequence[r])])),
        eta0=eta0[r : r + 1],
        weights0=weights0[r : r + 1],
    )
    assert torch.allclose(single["rare_sum"], batched["rare_sum"][r : r + 1], atol=1e-12)
    assert torch.allclose(single["weights"], batched["weights"][r : r + 1], atol=1e-12)
print("batched with padding == one at a time: ok")

# %% 9. zero_first_column: the C++ quirk in AttentionShift, and the alternative
quirk = osc.oscar(prm, seq, eta0=eta0[:1], weights0=weights0[:1])
fixed = osc.oscar(prm, seq, eta0=eta0[:1], weights0=weights0[:1], zero_first_column=False)
diff = (quirk["rare_sum"] / quirk["rare_count"] - fixed["rare_sum"] / fixed["rare_count"]).abs().nan_to_num()
print("largest change in mean P(rare) from zeroing each node's own column:", float(diff.max()))

# %% 10. evaluate(): the model of the PSP routine
g = torch.Generator().manual_seed(4)
values = osc.evaluate(par, training, sequence=sequence, generator=g)
print(pd.DataFrame(values.numpy(), columns=["anchor", *osc.EVALUATED_STIMULI]).round(3))
assert values.shape == (B, 5) and torch.all(values[:, 0] == 0.5)
assert torch.all((values >= 0) & (values <= 1))

# a fresh random initial state on every call: the same parameters give slightly different values
again = osc.evaluate(par, training, sequence=sequence, generator=g)
print("change between two calls:", float((again - values).abs().max()))

# with the initial state given, evaluate() is deterministic and agrees with oscar()
fixed_values = osc.evaluate(par, training, sequence=sequence, eta0=eta0, weights0=weights0)
cols = [osc.STIMULI.index(s) for s in osc.EVALUATED_STIMULI]
assert torch.allclose(fixed_values[:, 1:], (batched["rare_sum"] / batched["rare_count"])[:, cols])
print("evaluate() == oscar() with the same initial state: ok")

# %% 11. imac() and discretize(): the inequality matrix
p = torch.tensor([[0.5, 0.1, 0.45, 0.9, 0.62]], dtype=torch.float64)
matrix = osc.imac(p)
print("probabilities:", p[0].numpy())
print(pd.DataFrame(matrix[0].int().numpy(), index=["anchor", *osc.EVALUATED_STIMULI],
                   columns=["anchor", *osc.EVALUATED_STIMULI]))
assert matrix[0, 0, 1] == 1  # 0.5 - 0.1 >= 0.125
assert matrix[0, 0, 2] == 0  # 0.5 - 0.45 within the thresholds
assert matrix[0, 0, 3] == -1  # 0.5 - 0.9 <= -0.125
assert torch.all(torch.tril(matrix[0]) == torch.tril(torch.full((5, 5), 2.0, dtype=torch.float64)))

with_nan = torch.tensor([[0.5, float("nan"), 0.2, 0.3, 0.4]], dtype=torch.float64)
print("imac with a NaN keeps NaN cells:\n", osc.imac(with_nan)[0])
print("discretize with a NaN sets every cell to 99:\n", osc.discretize(with_nan)[0])
assert torch.all(osc.discretize(with_nan) == 99)
assert torch.equal(osc.discretize(p), matrix)
print("thresholds ±0.3:\n", osc.imac(p, thresholds=(0.3, -0.3))[0].int())

# %% 12. Against the compiled C++: the fixtures of tests/test_oscar.py
fixture_params = pd.read_csv(FIXTURES / "params.csv").to_numpy()
fixture_ids = [ids[0], ids[99], ids[-1]]  # participants 1, 2, 3 of the fixtures
worst = 0.0
for idx, ppt in enumerate(fixture_ids, 1):
    seq_i = torch.tensor(osc.participant_training(data, ppt))
    for k in range(1, 5):
        e0 = torch.tensor(pd.read_csv(FIXTURES / f"eta0_{idx}_{k}.csv").to_numpy()).unsqueeze(0)
        w0 = torch.tensor(pd.read_csv(FIXTURES / f"weights0_{idx}_{k}.csv").to_numpy()).unsqueeze(0)
        prm_k = {name: torch.tensor([fixture_params[k - 1][i]], dtype=torch.float64)
                 for i, name in enumerate(osc.OSCAR_PARAMETERS)}
        mine = osc.oscar(prm_k, seq_i, eta0=e0, weights0=w0, return_trials=True)
        for name in ("prob", "activations", "error"):
            ref = pd.read_csv(FIXTURES / f"{name}_{idx}_{k}.csv").to_numpy()
            worst = max(worst, float(np.abs(mine[name][0].numpy() - ref).max()))
        for name in ("weights", "eta"):
            ref = pd.read_csv(FIXTURES / f"{name}_{idx}_{k}.csv").to_numpy()
            worst = max(worst, float(np.abs(mine[name][0].numpy() - ref).max()))
print(f"3 participants x 4 parameter sets: largest difference from the C++ = {worst:.2e}")
assert worst < 1e-9

# the evaluate() vector and its inequality matrix (stored column-major in X1..X25)
ev = pd.read_csv(FIXTURES / "evaluate_and_imac.csv")
for _, row in ev.iterrows():
    idx, k = int(row.participant), int(row.param)
    seq_i = torch.tensor(osc.participant_training(data, fixture_ids[idx - 1]))
    e0 = torch.tensor(pd.read_csv(FIXTURES / f"eta0_{idx}_{k}.csv").to_numpy()).unsqueeze(0)
    w0 = torch.tensor(pd.read_csv(FIXTURES / f"weights0_{idx}_{k}.csv").to_numpy()).unsqueeze(0)
    vec = osc.evaluate(torch.tensor(fixture_params[k - 1 : k]), seq_i, eta0=e0, weights0=w0)
    assert np.allclose(vec[0].numpy(), row[["V1", "A", "B", "BC", "C"]].to_numpy(float), atol=1e-12)
    ref = row[[f"X{i}" for i in range(1, 26)]].to_numpy(float).reshape(5, 5, order="F")
    assert np.array_equal(osc.discretize(vec)[0].numpy(), ref)
print(f"evaluate() and discretize() match R for all {len(ev)} fixture rows")

# %% 13. On the GPU, if there is one: float64 matches the CPU, float32 is close
if torch.cuda.is_available():
    gpu = osc.evaluate(par.cuda(), training.cuda(), sequence=sequence.cuda(), eta0=eta0.cuda(), weights0=weights0.cuda())
    gpu32 = osc.evaluate(par.cuda().float(), training.cuda().float(), sequence=sequence.cuda(),
                         eta0=eta0.cuda().float(), weights0=weights0.cuda().float())
    print("GPU float64 vs CPU:", float((gpu.cpu() - fixed_values).abs().max()))
    print("GPU float32 vs CPU:", float((gpu32.double().cpu() - fixed_values).abs().max()))
    assert torch.allclose(gpu.cpu(), fixed_values, atol=1e-10)
else:
    print("no GPU here; skipped")

# %% 14. A small PSP run for one participant with pspGlobal()
seq0 = torch.tensor(one)
gen = torch.Generator().manual_seed(2026)
init = lower + torch.rand((20, 5), generator=gen, dtype=torch.float64) * (upper - lower)
control = {"lower": lower, "upper": upper, "init": init, "radius": 1.0, "population": 1_000_000,
           "iterations": 20, "parameter_names": list(osc.OSCAR_PARAMETERS),
           "dimensionality": 5, "responses": 5}
result = pspGlobal(lambda u: osc.evaluate(u, seq0, generator=gen), osc.discretize, control,
                   quiet=True, generator=gen)
print(f"{result.patterns} patterns from {result.evaluations:,} evaluations in {result.iterations} iterations")
top = result.ordinal_counts.argsort(descending=True)[:3]
for k in top.tolist():
    print(int(result.ordinal_counts[k]), "evaluations:\n", result.ordinal_patterns[k].int())

# %% 15. The same for three participants at once with pspMany(), as run_oscar.py does
controls = [dict(control) for _ in members]
results = pspMany(lambda u, run: osc.evaluate(u, training, sequence=run, generator=gen), osc.discretize,
                  controls, quiet=True, generator=gen)
for ppt, res in zip(members, results, strict=True):
    print(f"{ppt}: {res.patterns} patterns, {res.evaluations:,} evaluations")

# %% 16. Optional: the compiled C++ through R, on the same parameters and initial state
# Needs R with Rcpp, RcppArmadillo, data.table and clobe, the R package psp in the library below
# (benchmarks/oscar/README.md installs it), and the C++ sources in tmp-models/.
import subprocess
import tempfile

LIB, MODELS = ROOT / "benchmarks" / "oscar" / "r-lib", ROOT / "tmp-models"
if LIB.exists() and (MODELS / "OSCAR.cpp").exists():
    work = Path(tempfile.mkdtemp())
    pd.DataFrame(eta0[0].numpy()).to_csv(work / "eta0.csv", index=False)
    pd.DataFrame(weights0[0].numpy()).to_csv(work / "weights0.csv", index=False)
    pd.DataFrame([par[0].numpy()]).to_csv(work / "params.csv", index=False)
    script = f"""
    lib <- "{LIB}"; models <- "{MODELS}"
    source("{ROOT / 'benchmarks' / 'oscar' / 'r' / 'common.R'}")
    p <- unlist(read.csv("{work / 'params.csv'}"))
    st <- list(P = p[1], alpha = p[2], mu = p[3], rho = p[4], beta = p[5], colskip = 4,
               eta = as.matrix(read.csv("{work / 'eta0.csv'}")), weights = as.matrix(read.csv("{work / 'weights0.csv'}")))
    out <- OSCAR(st = st, tr = data.matrix(participant_training("{members[0]}")))
    write.csv(out$output, "{work / 'prob.csv'}", row.names = FALSE)
    """
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True, text=True)
    cpp = pd.read_csv(work / "prob.csv").to_numpy()
    mine = osc.oscar({k: v[:1] for k, v in batch_params.items()}, torch.tensor(osc.participant_training(data, members[0])),
                     eta0=eta0[:1], weights0=weights0[:1], return_trials=True)["prob"][0].numpy()
    print(f"trial-by-trial probabilities, torch vs C++: largest difference {np.abs(mine - cpp).max():.2e}")
    assert np.abs(mine - cpp).max() < 1e-9
else:
    print("R library or tmp-models/ missing; skipped")

# %%
print("all checks passed")
