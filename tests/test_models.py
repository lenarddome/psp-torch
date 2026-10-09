"""The models shipped with the package, checked against R where fixtures exist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from psp import pspGlobal
from psp.models import alcove as alc
from psp.models import prospect

from .conftest import DATA


class TestAlcove:
    def test_matches_catlearn_unit_test_fixture(self) -> None:
        """catlearn's own test: shin92, Euclidean metric, colskip 5, learning frozen in transfer."""
        tr = torch.tensor(pd.read_csv(DATA / "catlearn_alcove_tr.csv").to_numpy())
        h = torch.tensor(np.loadtxt(DATA / "catlearn_alcove_h.csv", delimiter=","))
        expected = np.loadtxt(DATA / "catlearn_alcove_prob1.csv", delimiter=",")
        params = {
            k: torch.tensor([v], dtype=torch.float64)
            for k, v in {"c": 2.0, "phi": 1.0, "la": 0.25, "lw": 0.75}.items()
        }
        prob = alc.alcove(
            params,
            tr.unsqueeze(0),
            h=h,
            alpha=torch.full((4,), 0.25, dtype=torch.float64),
            w=torch.zeros(2, 16, dtype=torch.float64),
            colskip=5,
            r=2.0,
            q=1.0,
        )
        np.testing.assert_allclose(prob[0, 0, :, 0].numpy(), expected, rtol=0, atol=1e-12)

    def test_curves_match_the_r_recipe(self) -> None:
        unit = torch.tensor(pd.read_csv(DATA / "alcove_fixture_unit.csv").to_numpy())
        expected = pd.read_csv(DATA / "alcove_fixture_curves.csv").to_numpy()
        orders = pd.read_csv(DATA / "alcove_fixture_orders.csv")["order"].tolist()
        curves = alc.alcove_curves(unit, alc.alcove_training(5, 7624), chunk=3)
        np.testing.assert_allclose(curves.numpy(), expected, rtol=0, atol=1e-12)
        assert [alc.describe_order(p) for p in alc.weak_order(curves)] == orders

    def test_trial_probabilities_match_catlearn(self) -> None:
        unit = torch.tensor(pd.read_csv(DATA / "alcove_fixture_unit.csv").to_numpy())[:1]
        expected = pd.read_csv(DATA / "alcove_fixture_prob_type1.csv").to_numpy()
        seq = alc.sequences(alc.alcove_training(5, 7624))[:5]
        prob = alc.alcove(
            alc.to_alcove(unit),
            seq,
            h=alc.ALCOVE_STIMULI,
            alpha=torch.full((3,), 1 / 3, dtype=torch.float64),
            w=torch.zeros(2, 8, dtype=torch.float64),
            colskip=4,
        )
        np.testing.assert_allclose(prob[0].reshape(-1, 2).numpy(), expected, rtol=0, atol=1e-12)

    def test_shipped_training_orders_have_the_expected_shape(self) -> None:
        for seed, learners in [(7624, 5), (7624, 20), (11, 5), (44, 5)]:
            training = alc.alcove_training(learners, seed)
            assert training.shape == (6 * learners * 256, 12)  # 16 blocks of 16 trials
            seq = alc.sequences(training)
            assert seq.shape == (6 * learners, 256, 12)
            assert seq[:, 0, 1].tolist() == [t for t in range(1, 7) for _ in range(learners)]

    def test_numpy_training_generator_has_the_structure_of_catlearn(self) -> None:
        tr = alc.nosof94train(cond=2, blocks=3, subjs=2, seed=1)
        assert tr.shape == (2 * 3 * 16, 12)
        assert tr[:, 0].tolist() == [1] + [0] * 47 + [1] + [0] * 47
        # every block shows each stimulus twice, and category labels follow Type II
        for block in range(3):
            rows = tr[block * 16 : (block + 1) * 16]
            assert sorted(rows[:, 3].tolist()) == sorted([s for s in range(1, 9) for _ in range(2)])
            assert all(row[7] == (1 if row[3] in (1, 2, 7, 8) else -1) for row in rows)
        padded = alc.nosof94train(cond=1, blocks=1, absval=0)
        assert set(padded[:, 7].tolist()) == {0, 1}

    def test_weak_order_and_description(self) -> None:
        human = alc.describe_order(alc.weak_order(alc.human_curves()))
        assert human == "I < II < (III, IV, V) < VI"
        errors = torch.tensor(
            [[0.1, 0.2, 0.3, 0.3, 0.3, 0.4], [0.5] * 6, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
        )
        patterns = alc.weak_order(errors)
        assert patterns.shape == (3, 6, 6)
        assert [alc.describe_order(p) for p in patterns] == [
            "I < II < (III, IV, V) < VI",
            "(I, II, III, IV, V, VI)",
            "VI < V < IV < III < II < I",
        ]
        assert torch.equal(patterns[0], -patterns[0].T)
        assert alc.type_ranks(errors[0]).tolist() == [1, 2, 4, 4, 4, 6]
        assert alc.type_ranks(errors[1]).tolist() == [3.5] * 6

    def test_to_alcove_maps_the_unit_cube_onto_the_ranges(self) -> None:
        corners = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.float64)
        values = alc.to_alcove(corners)
        assert tuple(values) == alc.ALCOVE_PARAMETERS.name
        assert [values[n][0].item() for n in values] == list(alc.ALCOVE_PARAMETERS.lower)
        assert [values[n][1].item() for n in values] == list(alc.ALCOVE_PARAMETERS.upper)

    def test_full_ranges_reach_learning_rates_of_one(self) -> None:
        values = alc.to_alcove(torch.ones(1, 4, dtype=torch.float64), alc.ALCOVE_PARAMETERS_FULL)
        assert (
            values["lw"].item() == 1.0 and values["la"].item() == 1.0 and values["c"].item() == 20.0
        )
        curves = alc.alcove_curves(
            torch.ones(1, 4, dtype=torch.float64),
            alc.alcove_training(5, 7624),
            parameters=alc.ALCOVE_PARAMETERS_FULL,
        )
        assert curves.shape == (1, 96) and bool(torch.isfinite(curves).all())

    def test_nosof94_data(self) -> None:
        data = alc.nosof94()
        assert data.shape == (96, 3)
        assert data[0].tolist() == [1, 1, 0.211]


class TestProspect:
    def test_published_estimates_predict_three_of_four_choices(self) -> None:
        delta = prospect.model(torch.tensor([[0.88, 0.61]], dtype=torch.float64))
        assert torch.round(delta).tolist() == [[-250.0, 52.0, 18.0, 6.0]]
        pattern = prospect.discretize(delta)[0]
        assert prospect.describe(pattern) == "Riskier in Problems 4, 7 and 8"
        assert prospect.describe(prospect.human_pattern()) == "Riskier in Problems 4 and 8"

    def test_descriptions(self) -> None:
        assert prospect.describe(torch.diag(torch.tensor([-1.0, -1, -1, -1]))) == "Always safer"
        assert prospect.describe(torch.diag(torch.tensor([1.0, 1, 1, 1]))) == "Always riskier"
        assert (
            prospect.describe(torch.diag(torch.tensor([1.0, -1, -1, -1])))
            == "Riskier in Problem 3 only"
        )

    def test_partition_finds_eight_patterns(self) -> None:
        torch.manual_seed(2026)
        control = {
            "lower": [0.2, 0.3],
            "upper": [1, 1.2],
            "init": torch.stack([torch.rand(5) * 0.8 + 0.2, torch.rand(5) * 0.9 + 0.3], dim=1),
            "radius": 0.5,
            "population": 200,
            "iterations": 30000,
            "parameter_names": list(prospect.PARAMETER_NAMES),
            "dimensionality": 4,
            "responses": 4,
        }
        out = pspGlobal(prospect.model, prospect.discretize, control, quiet=True)
        found = {prospect.describe(p) for p in out.ordinal_patterns}
        assert "Riskier in Problems 4 and 8" in found
        assert len(found) == 8


def test_voronoi_model_regions_are_distinct() -> None:
    from psp.models.voronoi import voronoi_model

    vor = voronoi_model(10, 2)
    nearest = vor.discretize(vor.model(vor.centres))
    assert nearest[:, 0, 0].tolist() == list(range(10))
