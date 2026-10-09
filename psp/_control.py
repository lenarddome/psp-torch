"""Validation of the ``control`` settings of :func:`psp.pspGlobal`.

The R package takes ``control`` as a plain list. This module accepts the same
keys in a :class:`dict` (or a :class:`Control` instance) and turns them into a
validated, device-resident :class:`Control`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["Control", "UNBOUNDED"]

#: Value of ``iterations`` or ``population`` that lifts that stopping rule.
#: ``0``, ``None`` and ``inf`` are all accepted as "unbounded", as in R where ``0`` is used.
UNBOUNDED = 0

_INT_MAX = 2**63 - 1


def _is_unbounded(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isinf(value):
        return True
    return int(value) == 0


def _as_int(name: str, value: Any) -> int:
    if _is_unbounded(value):
        return _INT_MAX
    if isinstance(value, bool) or not float(value).is_integer():
        raise ValueError(f"control['{name}'] must be a positive integer, 0 or None, got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"control['{name}'] must be a positive integer, 0 or None, got {value!r}")
    return value


def _as_vector(name: str, value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"control['{name}'] must not be empty")
    return tensor


def _as_names(name: str, value: Any, prefix: str, count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}_{i + 1}" for i in range(count))
    if isinstance(value, str):
        value = [value]
    names = tuple(str(v) for v in value)
    return names


@dataclass(frozen=True)
class Control:
    """Validated settings of a parameter space partitioning run.

    Build it from the same keys the R package uses, for example
    ``Control.from_mapping({"lower": [0, 0], "upper": [1, 1], ...})``, or pass the
    mapping to :func:`psp.pspGlobal` directly.

    Attributes
    ----------
    init
        Starting parameter sets, one per row, shape ``(starts, dimensions)``.
    lower, upper
        Bounds of each parameter, shape ``(dimensions,)``. Use ``-inf``/``inf``
        for an unbounded parameter.
    radius
        Largest step the search takes in any one parameter, shape ``(dimensions,)``.
        The R package takes a single number; a vector gives each parameter its own.
    population
        Number of parameter sets each pattern needs before the search stops
        proposing from it. ``2**63 - 1`` when unbounded.
    iterations
        Cap on the number of iterations. ``2**63 - 1`` when unbounded.
    parameter_names, stimuli_names
        Column names of the files written with ``save=True``.
    dimensionality
        Size of the square matrix ``discretize`` returns, or ``None`` to take it
        from the first evaluation.
    responses
        Length of the vector ``model`` returns, or ``None`` to take it from the
        first evaluation.
    """

    init: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    radius: torch.Tensor
    population: int
    iterations: int
    parameter_names: tuple[str, ...]
    stimuli_names: tuple[str, ...] | None
    dimensionality: int | None
    responses: int | None

    @property
    def dimensions(self) -> int:
        """Number of free parameters."""
        return int(self.init.shape[1])

    @property
    def population_unbounded(self) -> bool:
        return self.population == _INT_MAX

    @property
    def iterations_unbounded(self) -> bool:
        return self.iterations == _INT_MAX

    @classmethod
    def from_mapping(
        cls,
        control: Mapping[str, Any] | Control,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Control:
        """Validate a ``control`` mapping with the keys of the R package.

        Required keys: ``init``, ``lower``, ``upper``, ``radius``, and at least one
        of ``population`` and ``iterations``. Optional keys: ``parameter_names``,
        ``stimuli_names``, ``dimensionality``, ``responses``.

        Raises
        ------
        ValueError
            With the same messages as the R package for the same mistakes.
        """
        if isinstance(control, Control):
            if device is None and dtype is None:
                return control
            control = control.to_mapping()
        if not isinstance(control, Mapping):
            raise TypeError("control must be a mapping with the keys of the R package")
        unknown = set(control) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"unknown control keys: {sorted(unknown)}")
        for key in ("init", "lower", "upper", "radius"):
            if key not in control:
                raise ValueError(f"control['{key}'] is required")

        iterations = _as_int("iterations", control.get("iterations", UNBOUNDED))
        population = _as_int("population", control.get("population", UNBOUNDED))
        if iterations == _INT_MAX and population == _INT_MAX:
            raise ValueError(
                "A reasonable threshold must be set by either adjusting iterations or population."
            )

        init_value = control["init"]
        if dtype is None:
            dtype = init_value.dtype if torch.is_tensor(init_value) else torch.float64
            if not dtype.is_floating_point:
                dtype = torch.float64
        if device is None:
            device = init_value.device if torch.is_tensor(init_value) else torch.device("cpu")
        device = torch.device(device)

        init = torch.as_tensor(init_value, dtype=dtype, device=device)
        if init.ndim == 1:
            init = init.unsqueeze(0)
        if init.ndim != 2 or init.numel() == 0:
            raise ValueError("control['init'] must be a matrix with one parameter set per row")
        dimensions = int(init.shape[1])

        lower = _as_vector("lower", control["lower"], device, dtype)
        upper = _as_vector("upper", control["upper"], device, dtype)
        if lower.numel() != dimensions or upper.numel() != dimensions:
            raise ValueError("init, lower and upper must have the same length.")
        if bool((lower > upper).any()):
            raise ValueError("control['lower'] must not exceed control['upper']")

        radius = _as_vector("radius", control["radius"], device, dtype)
        if radius.numel() == 1:
            radius = radius.expand(dimensions).clone()
        elif radius.numel() != dimensions:
            raise ValueError(
                "control['radius'] must be a single number or one number per parameter"
            )
        if bool((radius <= 0).any()) or not bool(torch.isfinite(radius).all()):
            raise ValueError("control['radius'] must be positive and finite")

        parameter_names = _as_names(
            "parameter_names", control.get("parameter_names"), "parameter", dimensions
        )
        if len(parameter_names) != dimensions:
            raise ValueError("Length of param_names must equal to the number of dimensions")

        responses = control.get("responses")
        if responses is not None:
            responses = _as_int("responses", responses)
            if responses in (0, _INT_MAX):
                raise ValueError("control['responses'] must be a positive integer")
        dimensionality = control.get("dimensionality")
        if dimensionality is not None:
            dimensionality = _as_int("dimensionality", dimensionality)
            if dimensionality in (0, _INT_MAX):
                raise ValueError("control['dimensionality'] must be a positive integer")

        stimuli = control.get("stimuli_names")
        stimuli_names: tuple[str, ...] | None
        if stimuli is None:
            stimuli_names = (
                None
                if responses is None
                else _as_names("stimuli_names", None, "stimulus", responses)
            )
        else:
            stimuli_names = _as_names("stimuli_names", stimuli, "stimulus", 0)
            if responses is not None and len(stimuli_names) != responses:
                raise ValueError("Length of stimuli_names must equal to control['responses']")

        return cls(
            init=init,
            lower=lower,
            upper=upper,
            radius=radius,
            population=population,
            iterations=iterations,
            parameter_names=parameter_names,
            stimuli_names=stimuli_names,
            dimensionality=dimensionality,
            responses=responses,
        )

    def to_mapping(self) -> dict[str, Any]:
        """The settings as a plain dict with the keys of the R package."""
        return {
            "init": self.init,
            "lower": self.lower,
            "upper": self.upper,
            "radius": self.radius,
            "population": UNBOUNDED if self.population_unbounded else self.population,
            "iterations": UNBOUNDED if self.iterations_unbounded else self.iterations,
            "parameter_names": list(self.parameter_names),
            "stimuli_names": None if self.stimuli_names is None else list(self.stimuli_names),
            "dimensionality": self.dimensionality,
            "responses": self.responses,
        }


_KNOWN_KEYS = {
    "init",
    "lower",
    "upper",
    "radius",
    "population",
    "iterations",
    "parameter_names",
    "stimuli_names",
    "dimensionality",
    "responses",
}
