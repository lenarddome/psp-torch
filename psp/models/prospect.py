"""Prospect theory on four bets of Kahneman and Tversky (1979), the model of the tutorial.

Each option is a prospect: an amount ``x`` won with probability ``p``. Its value is
``w(p) * v(x)`` with ``v(x) = x ** alpha`` and the probability weighting function
of Tversky and Kahneman (1992), ``w(p) = p**g / (p**g + (1 - p)**g) ** (1 / g)``.
The model returns, for each problem, the value of the riskier option minus the
value of the safer one, and the ordinal pattern is the sign of each difference on
the diagonal of a 4 x 4 matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "PROBLEMS",
    "Problem",
    "weight",
    "prospect_value",
    "model",
    "discretize",
    "describe",
    "human_pattern",
    "PARAMETER_NAMES",
]

PARAMETER_NAMES = ("alpha", "gamma")


@dataclass(frozen=True)
class Problem:
    """One choice between a riskier and a safer prospect, with what people chose."""

    problem: int
    risky_amount: float
    risky_chance: float
    safe_amount: float
    safe_chance: float
    chose_risky: float  # percentage of people choosing the riskier option
    people: int


#: Problems 3, 4, 7 and 8 of Kahneman and Tversky (1979).
PROBLEMS: tuple[Problem, ...] = (
    Problem(3, 4000, 0.80, 3000, 1.00, 20, 95),
    Problem(4, 4000, 0.20, 3000, 0.25, 65, 95),
    Problem(7, 6000, 0.45, 3000, 0.90, 14, 66),
    Problem(8, 6000, 0.001, 3000, 0.002, 73, 66),
)

_RISKY_AMOUNT = torch.tensor([p.risky_amount for p in PROBLEMS], dtype=torch.float64)
_RISKY_CHANCE = torch.tensor([p.risky_chance for p in PROBLEMS], dtype=torch.float64)
_SAFE_AMOUNT = torch.tensor([p.safe_amount for p in PROBLEMS], dtype=torch.float64)
_SAFE_CHANCE = torch.tensor([p.safe_chance for p in PROBLEMS], dtype=torch.float64)


def weight(p: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """The decision weight of probability ``p`` with weighting parameter ``gamma``."""
    pg = p**gamma
    return pg / (pg + (1 - p) ** gamma) ** (1 / gamma)


def prospect_value(
    amount: torch.Tensor, chance: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """Subjective value of winning ``amount`` with probability ``chance``."""
    return weight(chance, gamma) * amount**alpha


def model(par: torch.Tensor) -> torch.Tensor:
    """Value of the riskier minus the safer option in each problem, shape ``(n, 4)``.

    ``par`` holds ``alpha`` and ``gamma`` in its two columns.
    """
    par = par.reshape(-1, 2)
    alpha = par[:, :1]
    gamma = par[:, 1:]
    risky_amount = _RISKY_AMOUNT.to(par.device, par.dtype)
    risky_chance = _RISKY_CHANCE.to(par.device, par.dtype)
    safe_amount = _SAFE_AMOUNT.to(par.device, par.dtype)
    safe_chance = _SAFE_CHANCE.to(par.device, par.dtype)
    return prospect_value(risky_amount, risky_chance, alpha, gamma) - prospect_value(
        safe_amount, safe_chance, alpha, gamma
    )


def discretize(delta: torch.Tensor) -> torch.Tensor:
    """The sign of each difference on the diagonal of a 4 x 4 matrix, shape ``(n, 4, 4)``."""
    return torch.diag_embed(torch.sign(delta.reshape(-1, len(PROBLEMS))))


def describe(pattern: torch.Tensor) -> str:
    """Name a pattern by the problems in which the model takes the riskier option."""
    riskier = [
        p.problem
        for p, choice in zip(PROBLEMS, torch.diagonal(pattern).tolist(), strict=True)
        if choice > 0
    ]
    if not riskier:
        return "Always safer"
    if len(riskier) == len(PROBLEMS):
        return "Always riskier"
    if len(riskier) == 1:
        return f"Riskier in Problem {riskier[0]} only"
    return "Riskier in Problems " + ", ".join(map(str, riskier[:-1])) + f" and {riskier[-1]}"


def human_pattern() -> torch.Tensor:
    """What most people chose, as an ordinal pattern."""
    chose = torch.tensor(
        [1.0 if p.chose_risky > 50 else -1.0 for p in PROBLEMS], dtype=torch.float64
    )
    return torch.diag(chose)
