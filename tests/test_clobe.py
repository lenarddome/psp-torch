"""psp.clobe: the test suite of the R package clobe 0.8.0, and its outputs exported by
tests/data/clobe/export.R."""

from __future__ import annotations

import itertools
import math
import warnings

import pandas as pd
import pytest
import torch

from psp import clobe
from psp.clobe import (
    brutes,
    fubini,
    gdistance,
    imac,
    orderings,
    pattern_distance,
    patterns,
    semiorders,
    stirling2,
)

from .conftest import DATA

FIXTURES = DATA / "clobe"
T = (0.1, -0.1)


def csv(name: str) -> torch.Tensor:
    return torch.tensor(pd.read_csv(FIXTURES / name).to_numpy(dtype=float))


def cells_csv(name: str, cells: int) -> torch.Tensor:
    """Upper triangles exported from R; R writes a single cell per pattern as one row."""
    return csv(name).reshape(-1, cells)


def upper(x: torch.Tensor) -> torch.Tensor:
    """Strict upper triangles of (n, k, k), read row by row, as the fixtures store them."""
    k = x.shape[-1]
    i, j = torch.triu_indices(k, k, offset=1)
    return x[:, i, j].double()


def m(*values: float) -> torch.Tensor:
    return imac(torch.tensor(values, dtype=torch.float64), T)


# --- imac -----------------------------------------------------------------------------------


def test_imac_matches_the_r_fixture() -> None:
    values = torch.tensor(pd.read_csv(FIXTURES / "imac_values.csv")["value"].to_numpy(float))
    assert torch.equal(imac(values, T), csv("imac_stored.csv"))


def test_imac_differences_equal_to_a_threshold_are_approximately_equal() -> None:
    # every adjacent step is 0.1, but 0.3 - 0.2 is 0.0999... in floating point
    out = imac(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64), T)
    expected = torch.tensor(
        [[2, 0, 1, 1], [2, 2, 0, 1], [2, 2, 2, 0], [2, 2, 2, 2]], dtype=torch.float64
    )
    assert torch.equal(out, expected)
    out = imac(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64), T)
    assert torch.equal(out, -expected + 4 * (expected == 2))


def test_imac_weak_ordering_treats_only_ties_as_equal() -> None:
    out = imac(torch.tensor([0.4, 0.3, 0.2, 0.2], dtype=torch.float64), T, ordering="weak")
    assert upper(out[None]).tolist() == [[1, 1, 1, 1, 1, 0]]


def test_imac_classifies_any_range_and_integers() -> None:
    expected = torch.tensor([[2, 1, 1], [2, 2, -1], [2, 2, 2]], dtype=torch.get_default_dtype())
    assert torch.equal(imac(torch.tensor([5, 1, 3]), T), expected)
    assert imac(torch.tensor([0.50, 0.55]), T)[0, 1] == 0


def test_imac_is_batched_and_keeps_dtype_and_nan() -> None:
    values = torch.rand(7, 5, 4, dtype=torch.float32)
    out = imac(values, T)
    assert out.shape == (7, 5, 4, 4) and out.dtype == torch.float32
    assert torch.equal(out[3, 2], imac(values[3, 2], T))
    out = imac(torch.tensor([[0.1, float("nan"), 0.9]]), T)[0]
    assert torch.isnan(out[0, 1]) and torch.isnan(out[1, 2]) and out[0, 2] == -1


def test_imac_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="ordering"):
        imac([0.1, 0.2], T, ordering="total")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="thresholds"):
        imac([0.1, 0.2], (0.1,))
    with pytest.raises(ValueError, match="thresholds"):
        imac([0.1, 0.2], (-0.1, 0.1))


# --- patterns -------------------------------------------------------------------------------

DATA_ROWS = torch.tensor(
    [
        [0.9, 0.5, 0.1],
        [0.1, 0.5, 0.9],
        [0.8, 0.5, 0.2],  # same pattern as the first row
        [0.5, 0.5, 0.5],
        [0.9, 0.4, 0.0],  # same pattern as the first row
    ],
    dtype=torch.float64,
)


def test_patterns_in_order_of_appearance() -> None:
    out = patterns(DATA_ROWS, T)
    assert out.n == 3
    assert out.counts.tolist() == [3, 1, 1]
    assert out.frequencies.tolist() == [0.6, 0.2, 0.2]
    assert out.pattern.tolist() == [0, 1, 0, 2, 0]
    for slot, row in enumerate([0, 1, 3]):
        assert torch.equal(out.inequality_matrices[slot], imac(DATA_ROWS[row], T))


def test_patterns_accepts_arrays_and_the_ordering() -> None:
    assert torch.equal(patterns(DATA_ROWS.numpy(), T).pattern, patterns(DATA_ROWS, T).pattern)
    rows = [[0.50, 0.45], [0.50, 0.50]]
    assert patterns(rows, T).n == 1
    assert patterns(rows, T, ordering="weak").n == 2


@pytest.mark.parametrize("ordering", ["semiorder", "weak"])
def test_patterns_match_r(ordering: str) -> None:
    out = patterns(csv("random_human.csv"), T, ordering)  # type: ignore[arg-type]
    assert torch.equal(upper(out.inequality_matrices), csv(f"random_human_{ordering}.csv"))
    index = pd.read_csv(FIXTURES / f"random_human_{ordering}_index.csv")["pattern"]
    assert out.pattern.tolist() == (index - 1).tolist()


def test_patterns_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="matrix"):
        patterns(torch.tensor([0.1, 0.2]), T)
    with pytest.raises(ValueError, match="missing"):
        patterns(torch.tensor([[0.1, float("nan")]]), T)
    with pytest.raises(ValueError, match="numeric"):
        patterns(torch.tensor([[True, False]]), T)


# --- gdistance ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """The R fixture: 6 human slices holding 3 unique patterns, 20 model slices holding 8."""

    def cube(name: str) -> torch.Tensor:
        cells = csv(name)
        k = round((1 + math.sqrt(1 + 8 * cells.shape[1])) / 2)
        out = torch.full((cells.shape[0], k, k), 2.0, dtype=torch.float64)
        i, j = torch.triu_indices(k, k, offset=1)
        out[:, i, j] = cells
        return out

    return cube("gdistance_predicted.csv"), cube("gdistance_discovered.csv")


def quiet(**kwargs):  # the fixture repeats patterns and its frequencies do not sum to 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "'frequencies' sum to|'human' has")
        return gdistance(**kwargs)


def test_gdistance_fixture(fixture) -> None:
    predicted, discovered = fixture
    human_unique, novel = 3, 5
    assert torch.unique(upper(predicted), dim=0).shape[0] == human_unique

    frequencies = torch.ones(6) / 21
    out = quiet(
        human=predicted, model=discovered, universal=30, weight=0.5, frequencies=frequencies
    )
    alpha, beta = float(frequencies.sum()), novel / (30 - human_unique)
    assert out.alpha == pytest.approx(alpha)
    assert out.beta == pytest.approx(beta)
    assert out.gdistance == pytest.approx(math.sqrt(0.5 * (1 - alpha) ** 2 + 0.5 * beta**2))
    assert out.accommodation.tolist() == list(range(6))

    weights = torch.tensor([2.0, 31, 5, 32, 31, 9])
    out = quiet(
        human=predicted,
        model=discovered,
        universal=20,
        weight=0.5,
        frequencies=weights / weights.sum(),
    )
    beta = novel / (20 - human_unique)
    assert out.alpha == pytest.approx(1)
    assert out.gdistance == pytest.approx(math.sqrt(0.5 * beta**2))

    for w in [i / 10 for i in range(11)]:
        out = quiet(
            human=predicted, model=discovered, universal=20, weight=w, frequencies=frequencies
        )
        expected = math.sqrt(w * (1 - alpha) ** 2 + (1 - w) * beta**2)
        assert out.gdistance == pytest.approx(expected)


@pytest.mark.parametrize("ordering", ["semiorder", "weak"])
def test_gdistance_matches_r(ordering: str) -> None:
    human = patterns(csv("random_human.csv"), T, ordering)  # type: ignore[arg-type]
    model = patterns(csv("random_model.csv"), T, ordering)  # type: ignore[arg-type]
    universal = fubini(4) if ordering == "weak" else semiorders(4)
    reference = pd.read_csv(FIXTURES / "random_gdistance.csv")
    for row in reference[reference.ordering == ordering].itertuples():
        out = gdistance(
            human.inequality_matrices,
            model.inequality_matrices,
            universal,
            row.weight,
            human.frequencies,
        )
        assert out.gdistance == pytest.approx(row.gdistance, rel=1e-14)
        assert out.alpha == pytest.approx(row.alpha, rel=1e-14)
        assert out.beta == pytest.approx(row.beta, rel=1e-14)
        assert out.accommodation.tolist() == [int(a) for a in row.accommodation.split()]


def test_gdistance_accommodation_is_0_based() -> None:
    human = torch.stack([m(0.9, 0.1, 0.5), m(0.1, 0.9, 0.5)])
    out = gdistance(human, human[1], universal=13, weight=0.5, frequencies=[0.5, 0.5])
    assert out.accommodation.tolist() == [1]
    assert (out.alpha, out.beta) == (0.5, 0.0)


def test_gdistance_ignores_the_diagonal_and_lower_triangle() -> None:
    human = m(0.9, 0.1, 0.5)
    model = human.triu(diagonal=1)
    model[1, 0] = float("nan")
    out = gdistance(human, model, universal=13, weight=0.5, frequencies=[1])
    assert (out.alpha, out.beta, out.gdistance) == (1.0, 0.0, 0.0)


def test_gdistance_beta_counts_each_pattern_once() -> None:
    a, b, c = m(0.9, 0.1, 0.5), m(0.1, 0.9, 0.5), m(0.5, 0.5, 0.5)
    with pytest.warns(UserWarning, match="1 duplicated pattern\\."):
        out = gdistance(
            torch.stack([a, a]),
            torch.stack([a, b, b, c]),
            universal=13,
            weight=0.5,
            frequencies=[0.5, 0.5],
        )
    assert out.beta == pytest.approx(2 / 12)


def test_gdistance_any_numeric_coding() -> None:
    def recode(x: torch.Tensor) -> torch.Tensor:
        above = torch.ones_like(x, dtype=torch.bool).triu(diagonal=1)
        return torch.where(above, torch.tensor([5.0, 7.0, 9.0])[(x + 1).clamp(0, 2).long()], x)

    human = torch.stack([m(0.9, 0.1, 0.5), m(0.1, 0.9, 0.5)])
    model = torch.stack([m(0.9, 0.1, 0.5), m(0.5, 0.5, 0.5)])
    plain = gdistance(human, model, universal=13, weight=0.5, frequencies=[0.5, 0.5])
    coded = gdistance(
        recode(human), recode(model), universal=13, weight=0.5, frequencies=[0.5, 0.5]
    )
    assert (plain.gdistance, plain.alpha, plain.beta) == (coded.gdistance, coded.alpha, coded.beta)
    assert torch.equal(plain.accommodation, coded.accommodation)


DISTINCT = torch.stack([m(0.9, 0.1, 0.5), m(0.1, 0.9, 0.5), m(0.5, 0.5, 0.5)])


def test_gdistance_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gdistance(DISTINCT, DISTINCT[0], universal=19, weight=0.5, frequencies=[1 / 3] * 3)
    with pytest.warns(UserWarning, match="'frequencies' sum to 0.9"):
        gdistance(DISTINCT, DISTINCT, universal=19, weight=0.5, frequencies=[0.5, 0.3, 0.1])
    with pytest.warns(UserWarning, match="'frequencies' sum to 20"):
        gdistance(DISTINCT, DISTINCT, universal=19, weight=0.5, frequencies=[12, 5, 3])
    with pytest.warns(UserWarning, match="2 duplicated patterns.*patterns\\(\\)"):
        gdistance(
            DISTINCT[[0, 1, 0, 0]], DISTINCT, universal=19, weight=0.5, frequencies=[0.25] * 4
        )
    with pytest.warns(UserWarning, match="beta"):
        out = gdistance(DISTINCT, DISTINCT, universal=3, weight=0.5, frequencies=[1 / 3] * 3)
    assert math.isnan(out.beta) and out.gdistance == 0


def test_gdistance_rejects_invalid_input(fixture) -> None:
    predicted, discovered = fixture
    f = torch.ones(6) / 6
    with pytest.raises(ValueError, match="'frequencies' has 5 values, but 'human' has 6"):
        gdistance(predicted, discovered, universal=30, weight=0.5, frequencies=torch.ones(5))
    with pytest.raises(ValueError, match="universal"):
        quiet(human=predicted, model=discovered, universal=2, weight=0.5, frequencies=f)
    with pytest.raises(ValueError, match="weight"):
        quiet(human=predicted, model=discovered, universal=30, weight=2, frequencies=f)
    with pytest.raises(ValueError, match="dimensions"):
        gdistance(predicted, discovered[:, :2, :2], universal=30, weight=0.5, frequencies=f)
    with pytest.raises(ValueError, match="'human' must be"):
        gdistance(torch.tensor([1.0, 2, 3]), DISTINCT[0], universal=13, weight=0.5, frequencies=[1])
    with pytest.raises(ValueError, match="'model' must be"):
        gdistance(DISTINCT[0], torch.zeros(1, 1, 3, 3), universal=13, weight=0.5, frequencies=[1])
    for bad in ([0.5, float("nan"), 0.5], [0.5, -0.5, 1]):
        with pytest.raises(ValueError, match="frequencies"):
            gdistance(DISTINCT, DISTINCT, universal=19, weight=0.5, frequencies=bad)
    for value in (float("nan"), float("inf")):
        broken = DISTINCT[0].clone()
        broken[0, 1] = value
        with pytest.raises(ValueError, match="human"):
            gdistance(broken, DISTINCT[0], universal=13, weight=0.5, frequencies=[1])
        with pytest.raises(ValueError, match="model"):
            gdistance(DISTINCT[0], broken, universal=13, weight=0.5, frequencies=[1])


def test_gdistance_takes_the_patterns_of_a_psp_run() -> None:
    from psp import PSP

    def model(par: torch.Tensor) -> torch.Tensor:
        return torch.stack([par[:, 0], par[:, 1], 1 - par[:, 0]], dim=1)

    result = PSP(
        model,
        lambda out: imac(out, T),
        {
            "lower": [0, 0],
            "upper": [1, 1],
            "init": [[0.5, 0.5]],
            "radius": 0.2,
            "population": 20,
            "iterations": 30,
        },
        generator=torch.Generator().manual_seed(1),
        quiet=True,
    )
    human = patterns(DATA_ROWS, T)
    out = gdistance(
        human.inequality_matrices,
        result.ordinal_patterns,
        universal=semiorders(3),
        weight=0.5,
        frequencies=human.frequencies,
    )
    assert 0 <= out.gdistance <= 1


# --- pattern_distance -----------------------------------------------------------------------

A, B, C = m(0.9, 0.5, 0.1), m(0.9, 0.9, 0.1), m(0.1, 0.5, 0.9)  # 1>2>3, 1=2>3, 3>2>1


def test_pattern_distance_of_single_matrices() -> None:
    assert pattern_distance(A, A).tolist() == [[0]]
    assert pattern_distance(A, B).tolist() == [[1]]
    assert pattern_distance(A, B, method="hamming").tolist() == [[1]]
    assert pattern_distance(A, C).tolist() == [[6]]
    assert pattern_distance(A, C, method="hamming").tolist() == [[3]]


def test_pattern_distance_every_pair() -> None:
    x = torch.stack([A, B, C])
    assert pattern_distance(x).tolist() == [[0, 1, 6], [1, 0, 5], [6, 5, 0]]
    assert pattern_distance(x, x[1:], method="hamming").tolist() == [[1, 3], [0, 3], [3, 0]]
    assert pattern_distance(A, x).tolist() == [[0, 1, 6]]


def test_pattern_distance_graded_codes_and_triangles() -> None:
    x = torch.tensor([[2.0, 2, -2], [2, 2, 1], [2, 2, 2]])
    y = torch.tensor([[2.0, 0, 2], [2, 2, -1], [2, 2, 2]])
    assert pattern_distance(x, y).tolist() == [[8]]
    assert pattern_distance(x, y, method="hamming").tolist() == [[3]]
    lower = A.clone()
    lower[2, 0] = float("nan")
    assert pattern_distance(lower, A.triu(diagonal=1)).tolist() == [[0]]


def test_pattern_distance_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        pattern_distance(A, imac(torch.arange(4.0), T))
    with pytest.raises(ValueError, match="method"):
        pattern_distance(A, B, method="euclid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        pattern_distance(torch.tensor([1.0, 2, 3]), A)
    broken = A.clone()
    broken[0, 2] = float("nan")
    with pytest.raises(ValueError, match="'x'"):
        pattern_distance(broken, B)
    with pytest.raises(ValueError, match="'y'"):
        pattern_distance(A, broken)


# --- counting -------------------------------------------------------------------------------


def test_counts() -> None:
    assert (stirling2(0, 0), stirling2(5, 0), stirling2(3, 5)) == (1, 0, 0)
    assert (stirling2(5, 2), stirling2(10, 3)) == (15, 9330)
    assert [stirling2(4, k) for k in range(1, 5)] == [1, 7, 6, 1]
    assert [fubini(k) for k in range(9)] == [1, 1, 3, 13, 75, 541, 4683, 47293, 545835]
    assert (fubini(4, levels=3), fubini(4, levels=1), fubini(3, levels=10)) == (51, 1, 13)
    assert [semiorders(k) for k in range(9)] == [1, 1, 3, 19, 183, 2371, 38703, 763099, 17648823]


def test_counts_are_exact_where_r_warns() -> None:
    # R's doubles are not exact past 2^53; Python integers are. Fubini numbers by
    # their own recurrence, a(n) = sum_k C(n, k) a(n - k):
    a = [1]
    for n in range(1, 31):
        a.append(sum(math.comb(n, k) * a[n - k] for k in range(1, n + 1)))
    assert [fubini(n) for n in range(31)] == a
    assert fubini(30) > 2**53 and semiorders(20) > 2**53
    # the explicit formula: S(n, k) = sum_j (-1)^j C(k, j) (k - j)^n / k!
    explicit = sum((-1) ** j * math.comb(20, j) * (20 - j) ** 40 for j in range(21))
    assert stirling2(40, 20) == explicit // math.factorial(20) > 2**53


def test_counts_reject_invalid_input() -> None:
    for call in (
        lambda: stirling2(-1, 0),
        lambda: fubini(2.5),  # type: ignore[arg-type]
        lambda: semiorders(None),  # type: ignore[arg-type]
        lambda: fubini(3, levels=-1),
        lambda: semiorders(True),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            call()


# --- orderings and brutes -------------------------------------------------------------------


def test_orderings_count() -> None:
    for k in range(1, 7):
        assert orderings(k, "weak").n == fubini(k)
        assert orderings(k).n == semiorders(k)


@pytest.mark.parametrize("ordering", ["semiorder", "weak"])
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_orderings_match_r(ordering: str, k: int) -> None:
    out = orderings(k, ordering)  # type: ignore[arg-type]
    assert out.inequality_matrices.dtype == torch.int8
    expected = cells_csv(f"orderings_{ordering}_{k}.csv", k * (k - 1) // 2)
    assert torch.equal(upper(out.inequality_matrices), expected)


def test_orderings_agree_with_brutes() -> None:
    grid = torch.arange(11.0)
    for k, ordering in itertools.product(range(1, 5), ["semiorder", "weak"]):
        enumerated = orderings(k, ordering).inequality_matrices  # type: ignore[arg-type]
        found = brutes(grid, k, (1, -1), ordering).inequality_matrices  # type: ignore[arg-type]
        assert torch.equal(enumerated, found)
    # 5 stimuli need a finer grid for every semiorder
    found = brutes(torch.arange(51.0), 5, (5, -5)).inequality_matrices
    assert torch.equal(orderings(5).inequality_matrices, found)


def test_orderings_edge_cases() -> None:
    out = orderings(1)
    assert out.n == 1 and out.inequality_matrices.tolist() == [[[2]]]
    with pytest.raises(ValueError, match="length"):
        orderings(0)
    with pytest.raises(ValueError, match="ordering"):
        orderings(3, "total")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too many"):
        orderings(12)


def test_brutes_counts() -> None:
    grid = torch.linspace(0, 1, 11, dtype=torch.float64)
    assert [brutes(grid, k, T).n for k in range(1, 5)] == [1, 3, 19, 183]
    assert [brutes(grid, k, T, "weak").n for k in range(1, 6)] == [1, 3, 13, 75, 541]
    assert brutes(torch.arange(3.0), 4, (0.5, -0.5), "weak").n == 51
    for k in range(1, 5):
        assert brutes(torch.arange(11.0), k, (1, -1), "weak").n == fubini(k)
    assert brutes(torch.arange(3.0), 4, (1, -1), "weak").n == fubini(4, levels=3)


def test_brutes_match_r_on_a_coarse_scale() -> None:
    out = brutes(torch.arange(0, 1.01, 0.25, dtype=torch.float64), 4, (0.3, -0.3))
    assert torch.equal(upper(out.inequality_matrices), csv("brutes_coarse_4.csv"))


def test_brutes_agree_with_imac_on_integers() -> None:
    for ordering, k in itertools.product(["semiorder", "weak"], range(2, 5)):
        grid = torch.tensor(list(itertools.product(range(5), repeat=k)), dtype=torch.float64)
        exact = patterns(grid, (2, -2), ordering).n  # type: ignore[arg-type]
        assert brutes(torch.arange(5.0), k, (2, -2), ordering).n == exact  # type: ignore[arg-type]
        scaled = torch.arange(5, dtype=torch.float64) / 10
        assert brutes(scaled, k, (0.2, -0.2), ordering).n == exact  # type: ignore[arg-type]


def test_brutes_chunks_and_more_than_39_cells() -> None:
    grid = torch.linspace(0, 1, 11, dtype=torch.float64)
    whole = brutes(grid, 4, T)
    chunked = brutes(grid, 4, T, chunk_size=777)
    assert whole.n == chunked.n == 183
    assert torch.equal(whole.inequality_matrices, chunked.inequality_matrices)
    # 10 stimuli have 45 cells; with 2 values, all tied or two tied groups: 1 + 2^10 - 2
    out = brutes(torch.tensor([0.0, 1.0]), 10, (0.5, -0.5))
    assert out.n == 1023
    assert torch.unique(upper(out.inequality_matrices), dim=0).shape[0] == 1023


def test_brutes_permutations_and_options() -> None:
    dependent = torch.tensor([0.3, 0.1, 0.2], dtype=torch.float64)
    assert brutes(dependent, 2, T).permutations is None
    out = brutes(dependent, 2, T, return_permutations=True)
    expected = torch.tensor(
        list(itertools.product(dependent.tolist(), repeat=2)), dtype=torch.float64
    )
    assert torch.equal(out.permutations, expected)
    out = brutes(torch.linspace(0, 1, 11), 4, T, return_matrices=False)
    assert out.inequality_matrices is None and out.n == 183
    single = brutes(torch.linspace(0, 1, 11), 1, T)
    assert single.n == 1 and single.inequality_matrices.shape == (1, 1, 1)


def test_brutes_rejects_invalid_input() -> None:
    grid = torch.linspace(0, 1, 11)
    with pytest.raises(ValueError, match="length"):
        brutes(grid, 0, T)
    with pytest.raises(ValueError, match="dependent"):
        brutes(torch.tensor([]), 2, T)
    with pytest.raises(ValueError, match="thresholds"):
        brutes(grid, 2, (0.1,))
    with pytest.raises(ValueError, match="ordering"):
        brutes(grid, 2, T, "total")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="too many"):
        brutes(torch.linspace(0, 1, 1000), 10, T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_brutes_on_cuda() -> None:
    cpu = brutes(torch.linspace(0, 1, 11), 5, T)
    gpu = brutes(torch.linspace(0, 1, 11), 5, T, device="cuda")
    assert gpu.n == cpu.n == 2131  # 11 values are too coarse for every semiorder of 5
    assert torch.equal(gpu.inequality_matrices, cpu.inequality_matrices)


def test_module_is_exposed() -> None:
    import psp

    assert psp.clobe is clobe


# --- inclusive thresholds (clobe before 0.8.0, psp.models.oscar) -----------------------------


def old_imac(p: torch.Tensor, upper: float, lower: float) -> torch.Tensor:
    """The implementation psp.models.oscar had before it called psp.clobe.imac."""
    diff = p.unsqueeze(2) - p.unsqueeze(1)
    value = torch.where(diff >= upper, 1.0, torch.where(diff <= lower, -1.0, 0.0)).to(p.dtype)
    value = torch.where(torch.isnan(diff), diff, value)
    strict_upper = torch.triu(torch.ones_like(diff[0], dtype=torch.bool), diagonal=1)
    return torch.where(strict_upper, value, torch.full_like(value, 2.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inclusive_imac_reproduces_the_oscar_rule(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    p = torch.randint(0, 9, (500, 5), generator=g).to(dtype) / 8  # many differences of 0.125
    p[::17, 2] = float("nan")
    for upper, lower in [(0.125, -0.125), (0.25, 0.0), (0.0, 0.0)]:
        assert torch.equal(
            imac(p, (upper, lower), inclusive=True).nan_to_num(7),
            old_imac(p, upper, lower).nan_to_num(7),
        )


def test_inclusive_differs_only_at_the_thresholds() -> None:
    values = torch.tensor([0.5, 0.25, 0.0], dtype=torch.float64)
    assert upper(imac(values, (0.25, -0.25))[None]).tolist() == [[0, 1, 0]]
    assert upper(imac(values, (0.25, -0.25), inclusive=True)[None]).tolist() == [[1, 1, 1]]
    out = patterns(values[None], (0.25, -0.25), inclusive=True)
    assert torch.equal(out.inequality_matrices[0], imac(values, (0.25, -0.25), inclusive=True))
    grid = torch.arange(0, 1.01, 0.25, dtype=torch.float64)
    found = brutes(grid, 3, (0.25, -0.25), inclusive=True)
    rows = torch.tensor(list(itertools.product(grid.tolist(), repeat=3)), dtype=torch.float64)
    assert found.n == patterns(rows, (0.25, -0.25), inclusive=True).n
    assert found.n != brutes(grid, 3, (0.25, -0.25)).n


def test_inclusive_rejects_weak_orderings() -> None:
    with pytest.raises(ValueError, match="inclusive"):
        imac([0.1, 0.2], T, "weak", inclusive=True)
    with pytest.raises(ValueError, match="inclusive"):
        brutes([0.1, 0.2], 2, T, "weak", inclusive=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_functions_on_cuda_agree_with_the_cpu() -> None:
    data = csv("random_human.csv")
    model = csv("random_model.csv")
    for device in ("cpu", "cuda"):
        human = patterns(data.to(device), T)
        found = patterns(model.to(device), T)
        out = gdistance(
            human.inequality_matrices,
            found.inequality_matrices,
            semiorders(4),
            0.5,
            human.frequencies,
        )
        distances = pattern_distance(human.inequality_matrices, found.inequality_matrices)
        if device == "cpu":
            reference = (human.pattern, out.gdistance, out.accommodation, distances)
        else:
            assert torch.equal(human.pattern.cpu(), reference[0])
            assert out.gdistance == pytest.approx(reference[1], rel=1e-14)  # order of summation
            assert torch.equal(out.accommodation.cpu(), reference[2])
            assert torch.equal(distances.cpu(), reference[3])


def test_gdistance_accepts_float32_frequencies() -> None:
    human = DISTINCT
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gdistance(
            human,
            human,
            universal=19,
            weight=0.5,
            frequencies=torch.full((3,), 1 / 3, dtype=torch.float32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pattern_distance_on_cuda_matches_cdist() -> None:
    six = orderings(5).inequality_matrices.double()
    for method, p in (("kemeny", 1.0), ("hamming", 0.0)):
        expected = torch.cdist(upper(six), upper(six), p=p)
        assert torch.equal(pattern_distance(six.cuda(), method=method).cpu(), expected)
