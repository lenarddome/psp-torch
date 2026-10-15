"""ALCOVE (Kruschke, 1992) on the six category structures of Shepard, Hovland and Jenkins (1961).

This is a batched port of ``slpALCOVE()`` and ``nosof94train()`` from the R
package catlearn (Wills et al., 2025), wrapped for :func:`psp.PSP` as in
the ALCOVE recipe of the R package psp. Many parameter sets are run through the
same training sequences at once, which is what makes a long PSP run affordable
on a GPU.

The training orders are the ones catlearn generates for its seeds, shipped with
the package, so that results can be compared with the R analysis trial for trial.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from importlib import resources
from typing import Literal

import numpy as np
import torch

__all__ = [
    "ALCOVE_PARAMETERS",
    "ALCOVE_PARAMETERS_FULL",
    "ALCOVE_COLUMNS",
    "ALCOVE_STIMULI",
    "TYPES",
    "AlcoveParameters",
    "alcove",
    "alcove_curves",
    "alcove_training",
    "nosof94",
    "nosof94train",
    "human_curves",
    "sequences",
    "to_alcove",
    "weak_order",
    "describe_order",
    "type_ranks",
]

#: The six category structures, Types I to VI.
TYPES = ("I", "II", "III", "IV", "V", "VI")

#: Columns of a training matrix, as catlearn's ``nosof94train()`` produces them.
ALCOVE_COLUMNS = ("ctrl", "cond", "blk", "stim", "x1", "x2", "x3", "t1", "t2", "m1", "m2", "m3")

#: The eight stimuli as hidden nodes: three binary features each, one column per stimulus.
ALCOVE_STIMULI = torch.tensor(
    [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]],
    dtype=torch.float64,
)


@dataclass(frozen=True)
class AlcoveParameters:
    """The four free parameters with the ranges of Pitt, Kim, Navarro and Myung (2006).

    The lower bounds are kept above 0 because ALCOVE rejects a learning rate of 0.
    """

    name: tuple[str, ...] = ("c", "phi", "lw", "la")
    lower: tuple[float, ...] = (0.01, 0.01, 0.001, 0.001)
    upper: tuple[float, ...] = (20.0, 6.0, 0.2, 0.2)


ALCOVE_PARAMETERS = AlcoveParameters()

#: The full ranges of the learning rates, 0 to 1, with the same specificity and consistency ranges.
ALCOVE_PARAMETERS_FULL = AlcoveParameters(upper=(20.0, 6.0, 1.0, 1.0))

# category A (t1 = 1) of each type, by stimulus number 1-8 (catlearn's numbering)
_CATEGORY_A = {
    1: (1, 2, 3, 4),
    2: (1, 2, 7, 8),
    3: (1, 2, 3, 6),
    4: (1, 2, 3, 5),
    5: (1, 2, 3, 8),
    6: (1, 4, 6, 7),
}


def _shj61(cond: int) -> np.ndarray:
    """The eight stimuli of one type: columns stim, x1, x2, x3, t1, t2, m1, m2, m3."""
    rows = []
    for stim in range(1, 9):
        x = [(stim - 1) >> 2 & 1, (stim - 1) >> 1 & 1, (stim - 1) & 1]
        in_a = stim in _CATEGORY_A[cond]
        rows.append([stim, *x, 1 if in_a else -1, -1 if in_a else 1, 0, 0, 0])
    return np.array(rows, dtype=np.float64)


def nosof94train(
    cond: int = 1,
    blocks: int = 16,
    absval: float = -1,
    subjs: int = 1,
    seed: int | None = 7624,
    blkstyle: Literal["accurate", "eights"] = "accurate",
) -> np.ndarray:
    """Randomised training blocks for one category structure, like catlearn's ``nosof94train()``.

    Each block presents the eight stimuli twice. In block 1 (or in every block with
    ``blkstyle="eights"``) each half of the block is shuffled separately; later
    blocks shuffle all sixteen trials together. Each simulated subject starts
    with a reset trial (``ctrl = 1``).

    The shuffling uses NumPy's generator, so the orders differ from R's for the
    same seed. For the orders catlearn produces, use :func:`alcove_training`.

    Returns
    -------
    numpy.ndarray
        Shape ``(subjs * blocks * 16, 12)`` with the columns of :data:`ALCOVE_COLUMNS`.
    """
    rng = np.random.default_rng(seed)
    items = _shj61(cond)
    subjects = []
    for _ in range(subjs):
        trials = []
        for blk in range(1, blocks + 1):
            if blk == 1 or blkstyle == "eights":
                block = np.vstack([items[rng.permutation(8)], items[rng.permutation(8)]])
            else:
                block = np.vstack([items, items])[rng.permutation(16)]
            trials.append(np.column_stack([np.full(16, cond), np.full(16, blk), block]))
        subject = np.vstack(trials)
        ctrl = np.zeros((subject.shape[0], 1))
        ctrl[0, 0] = 1
        subjects.append(np.hstack([ctrl, subject]))
    out = np.vstack(subjects)
    if absval != -1:
        for column in (7, 8):
            out[out[:, column] == -1, column] = absval
    return out


def alcove_training(learners: int = 5, seed: int = 7624) -> torch.Tensor:
    """One training sequence per type and simulated learner, in catlearn's trial order.

    The orders catlearn's ``nosof94train()`` generates with R's random number
    generator are shipped for ``seed`` 7624 with 5 and 20 learners, and for seeds
    11, 22, 33 and 44 with 5 learners (the settings of the ALCOVE recipe). Other
    settings fall back to :func:`nosof94train` and NumPy's generator.

    Returns
    -------
    torch.Tensor
        Shape ``(6 * learners * 16 * 8 * 2, 12)``, ``float64``: types I to VI in
        turn, each with ``learners`` sequences of 16 blocks.
    """
    name = f"nosof94train_seed{seed}_learners{learners}.csv.gz"
    resource = resources.files("psp.data").joinpath(name)
    if resource.is_file():
        with resource.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as text:
            reader = csv.reader(text)
            header = next(reader)
            assert tuple(header) == ALCOVE_COLUMNS, header
            rows = [[float(v) for v in row] for row in reader]
        return torch.tensor(rows, dtype=torch.float64)
    return torch.from_numpy(
        np.vstack(
            [
                nosof94train(cond, blocks=16, absval=-1, subjs=learners, seed=seed)
                for cond in range(1, 7)
            ]
        )
    )


def sequences(training: torch.Tensor) -> torch.Tensor:
    """Split a training matrix into equal-length sequences at every reset trial.

    Returns a tensor of shape ``(sequences, trials, columns)``.
    """
    starts = torch.nonzero(training[:, 0] == 1).reshape(-1)
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError("a training matrix must start with a reset trial (ctrl = 1)")
    length = int(training.shape[0]) // int(starts.numel())
    if length * int(starts.numel()) != int(training.shape[0]) or not bool(
        (starts == torch.arange(0, training.shape[0], length, device=starts.device)).all()
    ):
        raise ValueError("every sequence must have the same number of trials")
    return training.reshape(int(starts.numel()), length, -1)


def to_alcove(
    unit: torch.Tensor, parameters: AlcoveParameters = ALCOVE_PARAMETERS
) -> dict[str, torch.Tensor]:
    """Stretch points of the unit hypercube onto ALCOVE's parameter ranges.

    ``unit`` has shape ``(n, 4)``; the result maps ``c``, ``phi``, ``lw`` and
    ``la`` to tensors of shape ``(n,)``. ``parameters`` gives the ranges: those of
    Pitt et al. by default, or :data:`ALCOVE_PARAMETERS_FULL`.
    """
    unit = unit.reshape(-1, 4)
    lower = torch.tensor(parameters.lower, dtype=unit.dtype, device=unit.device)
    upper = torch.tensor(parameters.upper, dtype=unit.dtype, device=unit.device)
    values = lower + unit * (upper - lower)
    return {name: values[:, i] for i, name in enumerate(parameters.name)}


def alcove(
    params: dict[str, torch.Tensor],
    training: torch.Tensor,
    *,
    h: torch.Tensor,
    alpha: torch.Tensor,
    w: torch.Tensor,
    colskip: int,
    r: float = 1.0,
    q: float = 1.0,
    dec: Literal["ER", "BN"] = "ER",
    humble: bool = True,
    attcon: bool = False,
    absval: float = -1.0,
) -> torch.Tensor:
    """Run ALCOVE for a batch of parameter sets over a batch of training sequences.

    A port of catlearn's ``slpALCOVE()``. Every parameter set runs through every
    sequence, all in parallel, and the state is carried from trial to trial within
    a sequence.

    Parameters
    ----------
    params
        ``c``, ``phi``, ``lw`` and ``la``, each a tensor of shape ``(batch,)``.
    training
        Shape ``(sequences, trials, columns)``, see :func:`sequences`. Column 0 holds
        the control code (0 normal, 1 reset the network, 2 freeze learning); the
        inputs, teaching signals and missing-dimension flags follow ``colskip``
        columns, as in catlearn.
    h
        Hidden node positions, shape ``(inputs, hidden)``.
    alpha
        Initial attention weights, shape ``(inputs,)``.
    w
        Initial association weights, shape ``(outputs, hidden)``.
    colskip
        Number of columns before the inputs, counting the control column.
    r, q
        Distance metric (1 city-block, 2 Euclidean) and similarity gradient
        (1 exponential, 2 Gaussian).
    dec
        ``"ER"`` for the exponential ratio rule, ``"BN"`` for the background noise rule.
    humble
        Use a humble teacher (Kruschke, 1992, Eq. 4b).
    attcon
        Constrain the attention weights to sum to 1.
    absval
        Teaching value for an absent category.

    Returns
    -------
    torch.Tensor
        Response probabilities, shape ``(batch, sequences, trials, outputs)``.

    Notes
    -----
    The exponential ratio rule is computed as a numerically stable softmax. Where
    catlearn's ``exp(phi * o) / sum(exp(phi * o))`` overflows (the association
    weights diverge when the specificity is tiny and the association learning rate
    large, so every exemplar responds alike and the updates overshoot), catlearn
    returns ``NaN`` and this function returns the limiting probabilities. Everywhere
    else the two agree to machine precision.
    """
    c = params["c"].reshape(-1, 1, 1)
    phi = params["phi"].reshape(-1, 1, 1)
    lw = params["lw"].reshape(-1, 1, 1, 1)
    la = params["la"].reshape(-1, 1, 1)
    device, dtype = c.device, c.dtype
    training = training.to(device, dtype)
    h = h.to(device, dtype)
    nin, nhid = h.shape
    nout = int(w.shape[0])
    batch = int(c.shape[0])
    n_seq, n_trials, _ = training.shape

    alpha0 = alpha.to(device, dtype).reshape(1, 1, nin).expand(batch, n_seq, nin)
    w0 = w.to(device, dtype).reshape(1, 1, nout, nhid).expand(batch, n_seq, nout, nhid)
    attention = alpha0.clone()
    weights = w0.clone()
    probabilities = torch.empty((batch, n_seq, n_trials, nout), device=device, dtype=dtype)

    ctrl = training[:, :, 0]
    inputs = training[:, :, colskip : colskip + nin]
    teachers = training[:, :, colskip + nin : colskip + nin + nout]
    present = training[:, :, colskip + nin + nout : colskip + 2 * nin + nout] == 0

    for trial in range(n_trials):
        reset = (ctrl[:, trial] == 1).reshape(1, n_seq, 1)
        attention = torch.where(reset, alpha0, attention)
        weights = torch.where(reset.unsqueeze(-1), w0, weights)
        learn = (ctrl[:, trial] != 2).to(dtype).reshape(1, n_seq, 1)

        x = inputs[:, trial]  # (seq, nin)
        t = teachers[:, trial]  # (seq, nout)
        hmx = (h.unsqueeze(0) - x.unsqueeze(2)).abs()  # (seq, nin, nhid)
        hmx = hmx * present[:, trial].unsqueeze(2)
        # hidden node activations: exp(-c * (sum_i alpha_i |h_ji - x_i|^r)^(q/r))
        distance = torch.einsum("bsi,sij->bsj", attention, hmx**r)
        if r != 1.0:
            distance = distance ** (1.0 / r)
        if q != 1.0:
            distance = distance**q
        ah = torch.exp(-c * distance)  # (batch, seq, nhid)
        ao = torch.einsum("bskj,bsj->bsk", weights, ah)  # (batch, seq, nout)

        if dec == "ER":
            probabilities[:, :, trial] = torch.softmax(phi * ao, dim=-1)
        elif dec == "BN":
            positive = ao.clamp_min(0)
            probabilities[:, :, trial] = (positive + phi) / (
                positive.sum(-1, keepdim=True) + nout * phi
            )
        else:
            raise ValueError(f"dec must be 'ER' or 'BN', not {dec!r}")

        target = t.unsqueeze(0).expand_as(ao)
        if humble:
            target = torch.where(target == 1.0, torch.maximum(target, ao), target)
            target = torch.where(target == absval, torch.minimum(target, ao), target)
        error = target - ao  # (batch, seq, nout)
        delta_w = lw * learn.unsqueeze(-1) * error.unsqueeze(-1) * ah.unsqueeze(-2)
        back = torch.einsum("bsk,bskj->bsj", error, weights) * ah  # (batch, seq, nhid)
        delta_alpha = -la * learn * c * torch.einsum("bsj,sij->bsi", back, hmx)
        weights = weights + delta_w
        if attcon:
            delta_alpha = delta_alpha - delta_alpha.mean(-1, keepdim=True)
            attention = (attention + delta_alpha).clamp_min(0)
            attention = attention / attention.sum(-1, keepdim=True)
        else:
            attention = (attention + delta_alpha).clamp_min(0)
    return probabilities


def alcove_curves(
    unit: torch.Tensor,
    training: torch.Tensor,
    *,
    chunk: int = 256,
    parameters: AlcoveParameters = ALCOVE_PARAMETERS,
) -> torch.Tensor:
    """Error rate of each type in each block for parameter sets on the unit hypercube.

    This is the ``model`` of the ALCOVE recipe. ``unit`` has shape ``(n, 4)`` and
    is mapped onto ALCOVE's ranges, ``parameters``, with :func:`to_alcove`.
    ``training`` is a matrix from :func:`alcove_training`. The result has shape
    ``(n, 96)``: the 16 blocks of Type I, then of Type II, and so on, each averaged
    over the simulated learners.

    ``chunk`` bounds how many parameter sets run at once, to limit memory.
    """
    unit = unit.reshape(-1, 4)
    seq = sequences(training.to(unit.device, unit.dtype))
    cond = seq[:, 0, 1].long() - 1  # type of each sequence, 0-5
    block = seq[:, :, 2].long() - 1  # block of each trial, 0-15
    correct_is_first = seq[:, :, 7] == 1  # t1 == 1: category A is correct
    n_types, n_blocks = len(TYPES), int(block.max()) + 1
    outputs = []
    for start in range(0, unit.shape[0], chunk):
        part = unit[start : start + chunk]
        prob = alcove(
            to_alcove(part, parameters),
            seq,
            h=ALCOVE_STIMULI,
            alpha=torch.full((3,), 1 / 3, dtype=torch.float64),
            w=torch.zeros((2, 8), dtype=torch.float64),
            colskip=4,
        )
        correct = torch.where(
            correct_is_first.unsqueeze(0), prob[..., 0], prob[..., 1]
        )  # (b, seq, trials)
        error = 1 - correct
        # mean error over the trials of each (type, block) cell, across learners
        cell = (cond.unsqueeze(1) * n_blocks + block).reshape(-1)  # (seq * trials,)
        sums = torch.zeros(
            (part.shape[0], n_types * n_blocks), device=unit.device, dtype=unit.dtype
        )
        sums.index_add_(1, cell, error.reshape(part.shape[0], -1))
        counts = torch.bincount(cell, minlength=n_types * n_blocks).to(unit.dtype)
        outputs.append(sums / counts)
    return torch.cat(outputs)


def weak_order(curves: torch.Tensor, gap: float = 0.025) -> torch.Tensor:
    """The weak order of the six types by mean error, as a 6 x 6 comparison matrix.

    This is the ``discretize`` of the ALCOVE recipe. The six mean errors are sorted
    and a new group starts wherever the gap to the next type exceeds ``gap``. The
    entry ``(i, j)`` is 1 when type ``i`` is harder than type ``j``, -1 when easier
    and 0 when they are tied. ``curves`` may be ``(96,)``, ``(n, 96)`` or ``(n, 6)``
    (mean errors); the result has shape ``(n, 6, 6)``, or ``(6, 6)`` for a vector.
    """
    single = curves.ndim == 1
    curves = curves.reshape(1, -1) if single else curves
    n_types = len(TYPES)
    if curves.shape[1] == n_types:
        error = curves
    else:
        error = curves.reshape(curves.shape[0], n_types, -1).mean(dim=2)
    sorted_error, order = error.sort(dim=1)
    new_group = torch.ones_like(sorted_error, dtype=torch.int64)
    new_group[:, 1:] = (sorted_error.diff(dim=1) > gap).long()
    group_sorted = new_group.cumsum(dim=1)
    group = torch.empty_like(group_sorted).scatter_(1, order, group_sorted)
    pattern = torch.sign(group.unsqueeze(2) - group.unsqueeze(1)).to(curves.dtype)
    return pattern[0] if single else pattern


def describe_order(pattern: torch.Tensor) -> str:
    """Write a weak order the usual way, easiest first: ``"I < II < (III, IV, V) < VI"``."""
    rank = (torch.as_tensor(pattern) > 0).sum(dim=1).tolist()  # how many types each is harder than
    groups: dict[int, list[str]] = {}
    for name, r in zip(TYPES, rank, strict=True):
        groups.setdefault(int(r), []).append(name)
    parts = []
    for r in sorted(groups):
        names = groups[r]
        parts.append(f"({', '.join(names)})" if len(names) > 1 else names[0])
    return " < ".join(parts)


def type_ranks(errors: torch.Tensor, gap: float = 0.025) -> torch.Tensor:
    """Average rank of each type in its weak order, 1 for the easiest, ties sharing ranks.

    ``errors`` holds the six mean errors, shape ``(6,)`` or ``(n, 6)``.
    """
    single = errors.ndim == 1
    pattern = weak_order(errors.reshape(1, -1) if single else errors, gap=gap)
    group = (pattern > 0).sum(dim=2)  # (n, 6): 0 for the easiest group
    n = group.shape[0]
    ranks = torch.empty(group.shape, dtype=torch.float64)
    for i in range(n):
        g = group[i]
        for value in g.unique():
            members = g == value
            below = int((g < value).sum())
            size = int(members.sum())
            ranks[i, members] = below + (size + 1) / 2
    return ranks[0] if single else ranks


def nosof94() -> np.ndarray:
    """Nosofsky et al.'s (1994) replication: columns ``type`` (1-6), ``block`` (1-16), ``error``."""
    resource = resources.files("psp.data").joinpath("nosof94.csv")
    with resource.open("r", encoding="utf-8") as text:
        reader = csv.reader(text)
        header = next(reader)
        assert header == ["type", "block", "error"], header
        rows = [[float(v) for v in row] for row in reader]
    return np.array(rows, dtype=np.float64)


def human_curves() -> torch.Tensor:
    """The human error rates in the order :func:`alcove_curves` uses, shape ``(96,)``."""
    data = nosof94()
    order = np.lexsort((data[:, 1], data[:, 0]))
    return torch.from_numpy(data[order, 2].copy())
