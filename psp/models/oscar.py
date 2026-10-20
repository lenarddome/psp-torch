"""OSCAR on the inverse base-rate effect data (dome2025ibre), batched over sets and participants.

A port of the OSCAR model (``architecture.cpp`` and ``model-row-by-row.cpp``) and of
the participant-level PSP routine around it (``convert.R``, ``imac.R``, ``psp.R``):

- :func:`oscar` runs the network for a batch of parameter sets, each row on one
  of a batch of trial sequences, and returns the response probabilities;
- :func:`dome2025ibre` loads the dataset (354 participants) and
  :func:`training_batch` builds the padded trial matrices of
  ``ConvertToStateList()`` for any set of participants;
- :func:`evaluate` is the ``model`` of the PSP routine (P(rare) at test for
  A, B, BC and C behind a 0.5 anchor) and :func:`discretize` its ``discretize``
  (clobe's ``imac()`` with the ±0.125 thresholds).

Every function takes whole batches: no loop runs over parameter sets or
participants, only over trials and over the ten recurrent attention-shift
iterations of a learning trial.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from importlib import resources
from typing import Literal

import numpy as np
import torch

from .. import clobe

__all__ = [
    "OSCAR_PARAMETERS",
    "OSCAR_BOUNDS",
    "STIMULI",
    "NODES",
    "TRIAL_COLUMNS",
    "IbreData",
    "dome2025ibre",
    "convert_to_state_list",
    "training_batch",
    "oscar",
    "random_initial_state",
    "evaluate",
    "imac",
    "discretize",
    "EVALUATED_STIMULI",
]

#: The five free parameters, in the order PSP() passes them.
OSCAR_PARAMETERS = ("P", "alpha", "mu", "rho", "beta")
#: Their bounds in the PSP routine (run.r): lower, upper.
OSCAR_BOUNDS = ((1.0, 1e-3, 1e-3, 1e-3, 1e-3), (5.0, 1.0, 1.0, 5.0, 5.0))
#: The five nodes of the network: three cues and the two outcomes.
NODES = ("A", "B", "C", "common", "rare")
#: Stimulus names, in the order their integer codes take in a trial matrix.
STIMULI = ("A", "B", "C", "AB", "ABX", "AC", "ACY", "BC")
#: The stimuli whose P(rare) at test enter the ordinal pattern, after the 0.5 anchor.
EVALUATED_STIMULI = ("A", "B", "BC", "C")
#: Columns of a trial matrix, as ``ConvertToStateList()`` lays them out (``colskip = 4``).
TRIAL_COLUMNS = (
    "ctrl",
    "ppt",
    "trials",
    "stim",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "y1",
    "y2",
    "y3",
    "y4",
    "y5",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
)
COLSKIP = 4
#: Control code of padding rows in a batch of sequences of unequal length.
PAD = -1.0
#: Saliences are kept at or above this value after every update, as in the C++.
ETA_FLOOR = 1e-3
#: Recurrent attention-shift iterations per node per learning trial.
SHIFT_ITERATIONS = 10

_STIMULUS_FEATURES = {
    #        A  B  C  X  Y
    "A": (1, 0, 0, 0, 0),
    "B": (0, 1, 0, 0, 0),
    "C": (0, 0, 1, 0, 0),
    "AB": (1, 1, 0, 0, 0),
    "ABX": (1, 1, 0, 1, 0),
    "AC": (1, 0, 1, 0, 0),
    "ACY": (1, 0, 1, 0, 1),
    "BC": (0, 1, 1, 0, 0),
}
_FEEDBACK = {"AB": (0, 0, 0, 1, 0), "AC": (0, 0, 0, 0, 1), "NULL": (0, 0, 0, 0, 0)}


@dataclass(frozen=True)
class IbreData:
    """The trials of the dataset, one entry per row: participant, stimulus, trial, phase, response.

    Arrays of equal length.
    """

    ppt: np.ndarray
    abstim: np.ndarray
    trial: np.ndarray
    phase: np.ndarray
    abresp: np.ndarray

    @property
    def participants(self) -> list[str]:
        """Participant ids in order of first appearance."""
        return list(dict.fromkeys(self.ppt.tolist()))

    def participant(self, ppt: str) -> IbreData:
        mask = self.ppt == ppt
        if not mask.any():
            raise KeyError(f"no participant {ppt!r}")
        return IbreData(
            self.ppt[mask], self.abstim[mask], self.trial[mask], self.phase[mask], self.abresp[mask]
        )


def dome2025ibre() -> IbreData:
    """The inverse base-rate effect data (ply207): 354 participants, training then 72 test trials.

    ``abresp`` is ``common``, ``rare`` or ``none``; ``phase`` is ``training`` or ``test``.

    References:
        Dome, L., & Wills, A. J. (2025). g-Distance: On the Comparison of Model and Human
        Heterogeneity. Psychological Review, 132(3), 632–655. https://doi.org/10.1037/rev0000550
    """

    resource = resources.files("psp.data").joinpath("dome2025ibre.csv.gz")
    with resource.open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as text:
        reader = csv.reader(text)
        header = next(reader)
        assert header == ["ppt", "abstim", "trial", "phase", "abresp"], header
        rows = list(reader)
    columns = list(zip(*rows, strict=True))
    return IbreData(
        ppt=np.array(columns[0]),
        abstim=np.array(columns[1]),
        trial=np.array(columns[2], dtype=np.int64),
        phase=np.array(columns[3]),
        abresp=np.array(columns[4]),
    )


def convert_to_state_list(
    schedule: list[str] | np.ndarray,
    block: list[int] | np.ndarray,
    trials: list[int] | np.ndarray,
    ppt: int = 0,
    design: Literal["supervised", "unsupervised"] = "supervised",
) -> np.ndarray:
    """One participant's trial matrix, as ``ConvertToStateList()`` in convert.R builds it.

    Parameters
    ----------
    schedule
        Stimulus of each trial, names from :data:`STIMULI`.
    block
        Phase code of each trial: ``0`` or ``4`` for training (``4`` is what psp.R
        uses for a supervised design), ``2`` for test.
    trials
        Trial numbers, kept as a column.
    ppt
        Participant code, kept as a column (an integer here; R keeps the string).
    design
        ``"supervised"``: outcomes are eligible outputs in both phases and training
        trials carry feedback; the first trial resets the network with code 3.
        ``"unsupervised"``: no feedback and no eligible outputs in training; code 1.

    Returns
    -------
    numpy.ndarray
        Shape ``(trials, 19)`` with :data:`TRIAL_COLUMNS`; ``stim`` holds the index
        of the stimulus in :data:`STIMULI`.
    """
    block = np.asarray(block, dtype=np.float64)
    rows = []
    for idx, (name, phase, trial) in enumerate(zip(schedule, block, trials, strict=True)):
        x = _STIMULUS_FEATURES[name]
        test = phase != 0 and phase != 4
        if design == "supervised":
            y = (0, 0, 0, 1, 1)
            f = _FEEDBACK["NULL"] if test else _FEEDBACK.get(name, _FEEDBACK["NULL"])
        elif design == "unsupervised":
            y = (0, 0, 0, 1, 1) if test else (0, 0, 0, 0, 0)
            f = _FEEDBACK["NULL"]
        else:
            raise ValueError("design must be 'supervised' or 'unsupervised'")
        ctrl = phase
        if idx == 0:
            ctrl = 3 if design == "supervised" else 1
        rows.append([ctrl, ppt, trial, STIMULI.index(name), *x, *y, *f])
    return np.array(rows, dtype=np.float64)


def participant_training(data: IbreData, ppt: str, code: int = 0) -> np.ndarray:
    """The trial matrix of one participant, pre-processed as psp.R does for ply207.

    Training trials get control code 4 (supervised learning with a response),
    test trials code 2 (respond, no learning), and the first trial code 3 (reset).
    """
    d = data.participant(ppt)
    block = np.where(d.phase == "test", 2, 4)
    return convert_to_state_list(d.abstim, block, d.trial, ppt=code, design="supervised")


def training_batch(
    data: IbreData, participants: list[str] | None = None, *, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, list[str]]:
    """Trial matrices of several participants, padded to the longest with control code ``PAD``.

    Returns the tensor of shape ``(participants, trials, 19)`` and the participant ids
    in row order.
    """
    ids = data.participants if participants is None else list(participants)
    matrices = [participant_training(data, p, code=k) for k, p in enumerate(ids)]
    length = max(m.shape[0] for m in matrices)
    out = np.full((len(ids), length, len(TRIAL_COLUMNS)), 0.0, dtype=np.float64)
    out[:, :, 0] = PAD
    for k, m in enumerate(matrices):
        out[k, : m.shape[0]] = m
    return torch.tensor(out, dtype=dtype), ids


def random_initial_state(
    batch: int,
    *,
    scale: float = 1e-3,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize non-zero saliences and weights for a batch of parameter sets.
    Returns ``eta0`` with a zero diagonal and ``weights0``, both of shape ``(batch, 5, 5)``.
    """
    n = len(NODES)
    eta0 = torch.rand((batch, n, n), generator=generator, device=device, dtype=dtype) * scale
    weights0 = torch.rand((batch, n, n), generator=generator, device=device, dtype=dtype) * scale
    eta0 = eta0 * (1 - torch.eye(n, device=device, dtype=dtype))
    return eta0, weights0


def _gating(
    x: torch.Tensor, eta: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 1, 2, 3: per-row p-norm gating of the input by the saliences.

    ``x`` is ``(B, n)``, ``eta`` ``(B, n, n)`` (row k: the saliences feeding node k),
    ``P`` ``(B, 1, 1)``. Returns the gated input ``(B, n, n)`` and the p-norm ``(B, n)``.
    """
    gain = x.unsqueeze(1) * eta
    pnorm = (gain**P).sum(dim=2) ** (1.0 / P.squeeze(-1))
    return gain / pnorm.unsqueeze(2), pnorm


def oscar(
    params: dict[str, torch.Tensor],
    training: torch.Tensor,
    *,
    eta0: torch.Tensor,
    weights0: torch.Tensor,
    sequence: torch.Tensor | None = None,
    zero_first_column: bool = True,
    return_trials: bool = False,
) -> dict[str, torch.Tensor]:
    """Run OSCAR for a batch of parameter sets, each on one trial sequence.

    Parameters
    ----------
    params
        ``P``, ``alpha``, ``mu``, ``rho`` and ``beta``, each of shape ``(B,)``.
    training
        Trial matrices, shape ``(S, T, 19)`` with :data:`TRIAL_COLUMNS`, or ``(T, 19)``
        for a single sequence shared by the batch. Control codes: 1 and 3 reset the
        network before the trial; 2 responds without learning; 0, 4 and 5 learn
        (3 and 4 with feedback as part of the teacher); :data:`PAD` rows are skipped.
    eta0, weights0
        Initial saliences and connection weights, shape ``(B, 5, 5)``; their diagonals
        are zeroed as in the C++.
    sequence
        For each row of the batch, the index of its sequence in ``training``,
        shape ``(B,)``. Defaults to sequence 0 for every row.
    zero_first_column
        Reproduce the C++ ``AttentionShift``, whose ``.diag().zeros()`` on a one-row
        matrix zeroes the attended activation of node A (column 0) for every row,
        instead of each row's own node. ``False`` zeroes each row's own node.
    return_trials
        Also return the trial-by-trial probabilities, activations and errors.

    Returns
    -------
    dict
        ``rare_sum`` ``(B, 8)`` and ``rare_count`` ``(B, 8)``: the sum and count of
        P(rare) over the test trials (code 2) of each stimulus of :data:`STIMULI`,
        ``weights`` and ``eta`` ``(B, 5, 5)`` after the last trial, and with
        ``return_trials`` also ``prob``, ``activations`` and ``error`` of shape
        ``(B, T, 5)``.
    """
    P = params["P"].reshape(-1, 1, 1)  ## attention normalization parameter
    alpha = params["alpha"].reshape(-1, 1, 1)  ## weight learning parameter
    mu = params["mu"].reshape(-1, 1, 1)  ## attention learning parameter
    rho = params["rho"].reshape(-1, 1, 1)  ## attention shift parameter (step-size)
    beta = params["beta"].reshape(-1, 1)  ## inverse temperature

    ## set up sizes, devices, and dtypes (float32 or float64) for the model run
    ## this sets up the parameters needed for pytorch to distribute the work
    B = int(P.shape[0])  ## determine batch size
    device, dtype = P.device, P.dtype
    n = len(NODES)  ## the number of nodes (layers are fully connected)
    if training.ndim == 2:
        training = training.unsqueeze(0)
    ## move participant data to appropriate device (either CPU or GPU)
    training = training.to(device, dtype)
    ## S = number of participants (sequences)
    ## T = number of trials per sequence (participants)
    ## _ = number of columns (not needed)
    S, T, _ = training.shape

    ## if no sequence is given, e.g. only one participant is run,
    ## parameters are mapped to sequence 0
    if sequence is None:
        sequence = torch.zeros(B, dtype=torch.int64, device=device)
    sequence = sequence.to(device)  ## move

    ## create a mask to reset diagonals to 0
    eye = torch.eye(n, device=device, dtype=dtype)  ## identity matrix
    off = 1 - eye  ## complement of an identity matrix

    ## copy initial attention and connection weights
    eta0 = eta0.to(device, dtype) * off
    weights0 = weights0.to(device, dtype) * off
    eta = eta0.clone()
    weights = weights0.clone()

    ## allocate empty tensors to store output for the current run(s)
    rare_sum = torch.zeros((B, len(STIMULI)), device=device, dtype=dtype)
    rare_count = torch.zeros((B, len(STIMULI)), device=device, dtype=dtype)
    if return_trials:
        prob_all = torch.zeros((B, T, n), device=device, dtype=dtype)
        act_all = torch.zeros((B, T, n), device=device, dtype=dtype)
        err_all = torch.zeros((B, T, n), device=device, dtype=dtype)
    rare_index = NODES.index("rare")
    first_column = torch.zeros(n, device=device, dtype=dtype)
    first_column[0] = 1.0

    ## run model on all trials
    for t in range(T):
        ## extract trials
        row = training[sequence, t]  # (B, 19)
        ctrl = row[:, 0]
        valid = ctrl != PAD
        ## reset weights for new participant or session
        reset = ((ctrl == 1) | (ctrl == 3)).reshape(B, 1, 1)
        eta = torch.where(reset, eta0, eta)
        weights = torch.where(reset, weights0, weights)

        x = row[:, COLSKIP : COLSKIP + n]  ## extract input vector
        ## extract excitability vector (binary gates for which output node can respond)
        y = row[:, COLSKIP + n : COLSKIP + 2 * n]
        f = row[:, -n:]  ## extract feedback vector

        ## if all excitability is 0, avoid numerical instability in softmax by setting all to 1
        y = torch.where((y.sum(dim=1, keepdim=True) == 0), torch.ones_like(y), y)
        ## if no feedback is provided, the input vector is its own teaching signal
        f = torch.where((f.sum(dim=1, keepdim=True) == 0), x, f)

        # Equation 1, 2, 3: gated connection weights, self-connections removed
        gated, _ = _gating(x, eta, P)
        # Equation 4: calculate model predictions
        connection = weights * gated * y.unsqueeze(2) * off
        activation = connection.sum(dim=2)  # (B, n)
        # Equation 5: calculate choice probabilities over excitable output nodes (gated softmax)
        power = torch.exp(beta * activation) * y
        prob = power / power.sum(dim=1, keepdim=True)

        ## check whether current trial is a valid test trials
        test = valid & (ctrl == 2)
        stim = row[:, 3].long()  ## extract current trial for all rows in the batch

        ## save current outputs before learning:
        ## sums up all probabilities and then divide them by their counts
        ## to get the average probability of rare for each stimulus
        rare = torch.where(test, prob[:, rare_index], torch.zeros_like(prob[:, rare_index]))
        rare_sum.scatter_add_(1, stim.unsqueeze(1), rare.unsqueeze(1))
        rare_count.scatter_add_(1, stim.unsqueeze(1), test.to(dtype).unsqueeze(1))
        if return_trials:
            prob_all[:, t] = torch.where(valid.unsqueeze(1), prob, torch.zeros_like(prob))
            act_all[:, t] = torch.where(
                valid.unsqueeze(1), activation, torch.zeros_like(activation)
            )

        # learning on every trial that is neither a test trial nor padding
        learn = (valid & (ctrl != 2)).reshape(B, 1, 1)
        supervised = ((ctrl == 3) | (ctrl == 4)).reshape(B, 1)
        teacher = torch.where(supervised, x + f, x)  # (B, n): the complete input + teacher
        present = (teacher != 0).to(dtype)  # (B, n); convert all non-zero teacher to float

        ## Equation 9-10: ten recurrent attention shifts, all five nodes at once
        eta_p = eta
        for _ in range(SHIFT_ITERATIONS):
            gated_p, pnorm_p = _gating(teacher, eta_p, P)
            prediction = (weights * gated_p).sum(dim=2)  # (B, n), output all ones
            error_p = teacher - prediction  # (B, n); Equation 7: calculate prediction error
            salience = gated_p ** (P - 1) * teacher.unsqueeze(1)  # (B, n, n)
            activations = weights * teacher.unsqueeze(1)
            attended = prediction.unsqueeze(2) * salience
            attended = attended * (1 - first_column) if zero_first_column else attended * off
            out = (activations - attended) * error_p.unsqueeze(2)
            delta_eta = torch.tanh(rho / pnorm_p.unsqueeze(2) * out * teacher.unsqueeze(1))
            eta_p = (eta_p + delta_eta).clamp_min(ETA_FLOOR)
        ## Equation 11: state displacement (reconstruction error)
        update = mu * (eta_p - eta) * present.unsqueeze(1)
        eta_new = (eta + update).clamp_min(ETA_FLOOR)

        # Equation 12: the delta rule with the final saliences
        gated_final, _ = _gating(teacher, eta_new, P)
        prediction_final = (weights * gated_final * off).sum(dim=2)
        error = teacher - prediction_final
        delta_w = alpha * error.unsqueeze(2) * teacher.unsqueeze(1) * gated_final
        weights_new = (weights + delta_w) * off

        ## update weights for all current learning trials
        eta = torch.where(learn, eta_new, eta)
        weights = torch.where(learn, weights_new, weights)
        if return_trials:
            err_all[:, t] = torch.where(learn.reshape(B, 1), error, torch.zeros_like(error))

    result: dict[str, torch.Tensor] = {
        "rare_sum": rare_sum,
        "rare_count": rare_count,
        "weights": weights,
        "eta": eta,
    }
    if return_trials:
        result.update(prob=prob_all, activations=act_all, error=err_all)
    return result


def evaluate(
    par: torch.Tensor,
    training: torch.Tensor,
    *,
    sequence: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    eta0: torch.Tensor | None = None,
    weights0: torch.Tensor | None = None,
    zero_first_column: bool = True,
) -> torch.Tensor:
    """The ``model`` for the PSP routine:

        1. Evaluates OSCAR.
        2. Adds a 0.5 anchor.
        3. Computes P(rare) at test for A, B, BC and C.

    ``par`` holds P, alpha, mu, rho and beta in its columns, shape ``(B, 5)``. The
    initial saliences and weights are drawn afresh for every row, as ``evaluate()``
    in imac.R does, unless ``eta0`` and ``weights0`` are given.

    Returns ``(B, 5)``.
    """
    par = par.reshape(-1, len(OSCAR_PARAMETERS))
    params = {name: par[:, k] for k, name in enumerate(OSCAR_PARAMETERS)}
    if eta0 is None or weights0 is None:
        eta0, weights0 = random_initial_state(
            par.shape[0], generator=generator, device=par.device, dtype=par.dtype
        )
    out = oscar(
        params,
        training,
        eta0=eta0,
        weights0=weights0,
        sequence=sequence,
        zero_first_column=zero_first_column,
    )
    means = out["rare_sum"] / out["rare_count"]
    columns = [STIMULI.index(s) for s in EVALUATED_STIMULI]
    anchor = torch.full((par.shape[0], 1), 0.5, device=par.device, dtype=par.dtype)
    return torch.cat([anchor, means[:, columns]], dim=1)


def imac(
    probabilities: torch.Tensor, thresholds: tuple[float, float] = (0.125, -0.125)
) -> torch.Tensor:
    """clobe's inequality matrix constructor as the PSP routine used it:
    :func:`psp.clobe.imac` with ``inclusive=True``.

    For ``probabilities`` of shape ``(B, m)`` returns ``(B, m, m)``: above the
    diagonal, 1 where ``p_i - p_j >= thresholds[0]``, -1 where ``<= thresholds[1]``,
    0 between; 2 on and below the diagonal. A NaN difference stays NaN.
    """
    return clobe.imac(probabilities, thresholds, inclusive=True)


def discretize(
    probabilities: torch.Tensor, thresholds: tuple[float, float] = (0.125, -0.125)
) -> torch.Tensor:
    """The ``discretize`` of the PSP routine: :func:`imac`, every cell 99 for a row with any NaN."""
    matrix = imac(probabilities, thresholds)
    bad = torch.isnan(probabilities).any(dim=1).reshape(-1, 1, 1)
    return torch.where(bad, torch.full_like(matrix, 99.0), matrix)
