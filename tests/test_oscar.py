"""OSCAR and the participant-level PSP routine, checked against the compiled C++ and R."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from psp import PSPLockstep
from psp.models import oscar as osc

from .conftest import DATA

FIXTURES = DATA / "oscar"


@pytest.fixture(scope="module")
def data() -> osc.IbreData:
    return osc.dome2025ibre()


@pytest.fixture(scope="module")
def chosen(data: osc.IbreData) -> list[str]:
    ids = data.participants
    return [ids[0], ids[99], ids[-1]]


def test_dataset(data: osc.IbreData) -> None:
    assert len(data.participants) == 354
    assert len(data.ppt) == 37088
    assert set(data.phase) == {"training", "test"}
    assert set(data.abresp) == {"common", "rare", "none"}
    one = data.participant(data.participants[0])
    assert (one.phase == "test").sum() == 72


def test_trial_matrices_match_convert_r(data: osc.IbreData, chosen: list[str]) -> None:
    for idx, ppt in enumerate(chosen, 1):
        ref = pd.read_csv(FIXTURES / f"train_{idx}.csv")
        mine = osc.participant_training(data, ppt)
        assert np.array_equal(
            ref.drop(columns=["ppt", "stim"]).to_numpy(dtype=float), np.delete(mine, [1, 3], axis=1)
        )
        assert [osc.STIMULI[int(c)] for c in mine[:, 3]] == ref.stim.tolist()
        assert mine[0, 0] == 3 and set(mine[1:, 0]) == {2, 4}


@pytest.mark.parametrize("idx", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_oscar_matches_the_cpp(data: osc.IbreData, chosen: list[str], idx: int, p: int) -> None:
    params = pd.read_csv(FIXTURES / "params.csv").to_numpy()
    training = torch.tensor(osc.participant_training(data, chosen[idx - 1]))
    eta0 = torch.tensor(pd.read_csv(FIXTURES / f"eta0_{idx}_{p}.csv").to_numpy()).unsqueeze(0)
    w0 = torch.tensor(pd.read_csv(FIXTURES / f"weights0_{idx}_{p}.csv").to_numpy()).unsqueeze(0)
    prm = {
        k: torch.tensor([params[p - 1][i]], dtype=torch.float64)
        for i, k in enumerate(osc.OSCAR_PARAMETERS)
    }
    out = osc.oscar(prm, training, eta0=eta0, weights0=w0, return_trials=True)
    for name, key in [("prob", "prob"), ("activations", "activations"), ("error", "error")]:
        ref = pd.read_csv(FIXTURES / f"{name}_{idx}_{p}.csv").to_numpy()
        np.testing.assert_allclose(out[key][0].numpy(), ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        out["weights"][0].numpy(),
        pd.read_csv(FIXTURES / f"weights_{idx}_{p}.csv").to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        out["eta"][0].numpy(), pd.read_csv(FIXTURES / f"eta_{idx}_{p}.csv").to_numpy(), atol=1e-9
    )
    # the evaluate() vector and its inequality matrix
    ev = pd.read_csv(FIXTURES / "evaluate_and_imac.csv")
    row = ev[(ev.participant == idx) & (ev.param == p)].iloc[0]
    vec = osc.evaluate(
        torch.tensor(params[p - 1 : p], dtype=torch.float64), training, eta0=eta0, weights0=w0
    )
    np.testing.assert_allclose(
        vec[0].numpy(), row[["V1", "A", "B", "BC", "C"]].to_numpy(dtype=float), atol=1e-12
    )
    ref_matrix = row[[f"X{k}" for k in range(1, 26)]].to_numpy(dtype=float).reshape(5, 5, order="F")
    assert np.array_equal(osc.discretize(vec)[0].numpy(), ref_matrix)


def test_imac_matches_clobe() -> None:
    probe = pd.read_csv(FIXTURES / "imac_probe.csv").to_numpy(dtype=float)
    mine = osc.discretize(torch.tensor(probe[:, :5])).numpy().reshape(3, 25, order="F")
    assert np.array_equal(mine, probe[:, 5:])
    m = osc.imac(torch.tensor([[0.5, 0.2, 0.9]]))[0]
    assert m.tolist() == [[2, 1, -1], [2, 2, -1], [2, 2, 2]]


def test_padded_batch_equals_single_sequences(data: osc.IbreData, chosen: list[str]) -> None:
    batch, ids = osc.training_batch(data, chosen)
    assert ids == chosen and batch.shape[2] == 19
    g = torch.Generator().manual_seed(0)
    lower, upper = (torch.tensor(b, dtype=torch.float64) for b in osc.OSCAR_BOUNDS)
    par = lower + torch.rand((6, 5), generator=g, dtype=torch.float64) * (upper - lower)
    eta0, w0 = osc.random_initial_state(6, generator=g)
    seq = torch.tensor([0, 1, 2, 2, 1, 0])
    multi = osc.evaluate(par, batch, sequence=seq, eta0=eta0, weights0=w0)
    for i in range(6):
        single = osc.evaluate(
            par[i : i + 1],
            torch.tensor(osc.participant_training(data, chosen[seq[i]])),
            eta0=eta0[i : i + 1],
            weights0=w0[i : i + 1],
        )
        assert torch.allclose(multi[i], single[0], atol=1e-13)


def test_random_initial_state_is_small_with_zero_salience_diagonal() -> None:
    eta0, w0 = osc.random_initial_state(4, generator=torch.Generator().manual_seed(1))
    assert eta0.shape == (4, 5, 5) and bool((eta0.diagonal(dim1=1, dim2=2) == 0).all())
    assert bool((eta0 >= 0).all()) and bool((eta0 < 1e-3).all()) and bool((w0 < 1e-3).all())


def test_pspmany_runs_independent_searches(data: osc.IbreData, chosen: list[str], tmp_path) -> None:
    batch, ids = osc.training_batch(data, chosen[:2])
    generator = torch.Generator().manual_seed(3)
    lower, upper = (torch.tensor(b, dtype=torch.float64) for b in osc.OSCAR_BOUNDS)
    init = lower + torch.rand((4, 5), generator=generator, dtype=torch.float64) * (upper - lower)
    controls = [
        {
            "lower": lower,
            "upper": upper,
            "init": init,
            "radius": 1.0,
            "population": 1_000_000,
            "iterations": 3,
            "parameter_names": list(osc.OSCAR_PARAMETERS),
            "stimuli_names": ["anchor", *osc.EVALUATED_STIMULI],
        }
        for _ in ids
    ]
    seen_runs = []

    def model(unit: torch.Tensor, run: torch.Tensor) -> torch.Tensor:
        seen_runs.append(run.clone())
        return osc.evaluate(unit, batch, sequence=run, generator=generator)

    results = PSPLockstep(
        model,
        osc.discretize,
        controls,
        save=True,
        paths=[tmp_path / p for p in ids],
        quiet=True,
        generator=generator,
    )
    assert len(results) == 2 and all(r.iterations == 3 for r in results)
    assert (
        sorted(seen_runs[0].tolist()) == [0] * 4 + [1] * 4
    )  # iteration 0: both runs' starts in one call
    assert len(seen_runs) == 4  # one model call per iteration
    for r, p in zip(results, ids, strict=True):
        assert r.ordinal_patterns.shape[1:] == (5, 5)
        assert (tmp_path / f"{p}_parameters.csv").exists()
        assert r.evaluations == sum(1 for s in seen_runs for v in s.tolist() if v == ids.index(p))


def test_pspmany_rejects_mismatched_runs() -> None:
    base = {
        "lower": [0, 0],
        "upper": [1, 1],
        "init": [[0.5, 0.5]],
        "radius": 0.1,
        "iterations": 2,
        "population": 5,
    }
    with pytest.raises(ValueError, match="iterations and population"):
        PSPLockstep(
            lambda u, r: u,
            lambda x: torch.ones(x.shape[0], 2, 2),
            [base, {**base, "iterations": 3}],
        )
    with pytest.raises(ValueError, match="one path per run"):
        PSPLockstep(
            lambda u, r: u,
            lambda x: torch.ones(x.shape[0], 2, 2),
            [base, base],
            save=True,
            paths=["a"],
        )
