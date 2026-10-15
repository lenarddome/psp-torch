"""Parameter space partitioning: the :func:`PSP` sampler and its result."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ._callbacks import Discretize, Model, as_numeric_tensor, rowwise
from ._control import Control
from ._io import RunWriter
from ._store import PatternStore

__all__ = ["PSP", "PSPResult", "hyper_points", "clamp_parameters"]


@dataclass(frozen=True)
class PSPResult(Mapping[str, Any]):
    """Container. It holds and validates the results of a :func:`PSP` run.
    It behaves like a mapping and a dataclass.

    Attributes
    ----------
    ordinal_patterns
        Every pattern found, shape ``(patterns, dimensionality, dimensionality)``,
        ``float64``, on the CPU. Pattern ``k`` is ``ordinal_patterns[k]``; the R
        package stores the same matrices as ``ordinal_patterns[, , k + 1]``.
    ordinal_counts
        How many evaluated parameter sets produced each pattern, shape ``(patterns,)``.
    iterations
        Number of iterations completed before a stopping rule was reached.
    """

    ordinal_patterns: torch.Tensor
    ordinal_counts: torch.Tensor
    iterations: int

    _KEYS = ("ordinal_patterns", "ordinal_counts", "iterations")

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    @property
    def patterns(self) -> int:
        """Number of patterns found."""
        return int(self.ordinal_patterns.shape[0])

    @property
    def evaluations(self) -> int:
        """Number of times the model was evaluated."""
        return int(self.ordinal_counts.sum())

    def __repr__(self) -> str:
        return (
            f"PSPResult(patterns={self.patterns}, evaluations={self.evaluations}, "
            f"iterations={self.iterations})"
        )


def hyper_points(
    counts: int,
    radius: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random steps of at most ``radius`` in each parameter, shape ``(counts, dimensions)``.

    This is the jumping distribution: a direction picked uniformly on the unit
    hypersphere (Weisstein, "Hypersphere Point Picking"),
    scaled in each dimension by a factor drawn uniformly between 0 and that
    dimension's radius. The scale factors are drawn once per call and shared by
    all ``counts`` points, as in the C++ implementation.

    Parameters
    ----------
    counts
        Number of points.
    radius
        Radius of each dimension, shape ``(dimensions,)``; its device and dtype are used.
    generator
        Random number generator on the same device as ``radius``.
        
    References
    ----------
    Weisstein, Eric W. "Hypersphere Point Picking." From MathWorld--A Wolfram Resource. https://mathworld.wolfram.com/HyperspherePointPicking.html
    """
    dimensions = int(radius.numel())
    direction = torch.randn(
        (counts, dimensions), generator=generator, device=radius.device, dtype=radius.dtype
    )
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(
        torch.finfo(radius.dtype).tiny
    )
    scale = torch.rand(dimensions, generator=generator, device=radius.device, dtype=radius.dtype)
    return direction * (scale * radius)


def clamp_parameters(
    parameters: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Move every parameter set onto the bounds where it left them."""
    return torch.maximum(torch.minimum(parameters, upper), lower)

def PSP(
    model: Model,
    discretize: Discretize,
    control: Mapping[str, Any] | Control,
    save: bool = False,
    path: str | Path = ".",
    extension: str = ".csv",
    quiet: bool = False,
    *,
    batched: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> PSPResult:
    """Partition a model's parameter space by the ordinal patterns it produces.

    A PyTorch implementation of the parameter space partitioning MCMC algorithm
    of Pitt, Kim, Navarro and Myung (2006). The algorithm:

    0. Evaluate ``model`` on every row of ``control["init"]``. The patterns
       ``discretize`` makes of the outputs are the first regions found.
    1. For every pattern with fewer than ``population`` parameter sets, propose a
       new parameter set at random within ``radius`` of the last parameter set
       that produced it, and move it onto the bounds if it crossed them.
    2. Evaluate the model on all proposals in one batch.
    3. Record the patterns found and the parameter sets that produced them,
       then return to step 1 until every pattern is full or ``iterations`` is reached.

    Parameters
    ----------
    model
        Maps parameter sets to the model's continuous outputs. Batched by default:
        it receives a ``(n, dimensions)`` tensor and returns ``(n, responses)``.
        With ``batched=False`` it receives one ``(dimensions,)`` tensor at a time
        and returns ``(responses,)``, as in R.
    discretize
        Maps model outputs to an ordinal pattern, a square numeric matrix. Batched
        by default: ``(n, responses)`` in, ``(n, dimensionality, dimensionality)`` out.
        With ``batched=False``, ``(responses,)`` in and a matrix out. Two parameter
        sets produce the same pattern when the matrices are identical, with
        ``-0.0`` equal to ``0.0`` and ``NaN`` equal to ``NaN``.
    control
        The settings as a set of control arguments, either a :class:`Control` or
        a mapping with the same keys. The required keys are:
ß
        ``init``
            Starting parameter sets, one per row.
        ``lower``, ``upper``
            Bounds of each parameter.
        ``radius``
            Largest step in any one parameter; a single number, or one per parameter.
        ``population``
            Parameter sets each pattern needs; ``0``/``None`` for no limit.
        ``iterations``
            Cap on the number of iterations; ``0``/``None`` for no cap. At least
            one of ``population`` and ``iterations`` must be set.
        ``parameter_names``, ``stimuli_names``
            Column names of the saved files (optional).
        ``dimensionality``, ``responses``
            Size of the pattern matrix and length of the model output, checked
            on every evaluation. Optional: taken from the first evaluation if absent.
    save
        Write every evaluated parameter set and model output to disk.
    path
        Prefix of the two files, ``<path>_parameters<extension>`` and
        ``<path>_continuous<extension>``. Each row holds the iteration, the values
        and the 0-based pattern id.
    extension
        Extension of the saved files.
    quiet
        Do not print the iteration number.
    batched
        Whether ``model`` and ``discretize`` take the whole batch (default) or one
        parameter set at a time. Equivalent to wrapping both with :func:`psp.rowwise`.
    device, dtype
        Where the search runs and in what precision. Defaults to the device and
        dtype of ``control["init"]`` if it is a tensor, otherwise CPU and ``float64``.
    generator
        A :class:`torch.Generator` on ``device`` for reproducible proposals. By
        default the global generator is used, seeded with :func:`torch.manual_seed`.

    Returns
    -------
    PSPResult
        ``ordinal_patterns``, ``ordinal_counts`` and ``iterations``.

    Raises
    ------
    ValueError
        For invalid control settings, and when ``model`` or ``discretize`` returns
        something of the wrong type or size. Errors raised by the callbacks propagate.

    References
    ----------
    Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model
    analysis by parameter space partitioning. *Psychological Review, 113*(1), 57-83.
    https://doi.org/10.1037/0033-295X.113.1.57

    Examples
    --------
    >>> import torch
    >>> from psp import PSP
    >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
    <torch._C.Generator object at ...>
    >>> model = lambda par: par                       # the outputs are the parameters
    >>> discretize = lambda out: (out > 0.5).double().diag_embed()  # which half of each axis
    >>> result = PSP(model, discretize, {
    ...     "init": [[0.5, 0.5]], "lower": [0, 0], "upper": [1, 1],
    ...     "radius": 0.5, "population": 20, "iterations": 100}, quiet=True)
    >>> result.patterns
    4
    """
    settings = Control.from_mapping(control, device=device, dtype=dtype)
    run_device = settings.init.device
    run_dtype = settings.init.dtype
    if not batched:
        model = rowwise(model)
        discretize = rowwise(discretize)

    responses = settings.responses
    dimensionality = settings.dimensionality

    def evaluate(jumping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal responses, dimensionality
        rows = int(jumping.shape[0])
        continuous = as_numeric_tensor(model(jumping), "model", run_device)
        if continuous.ndim == 1 and rows == 1:
            continuous = continuous.unsqueeze(0)
        if continuous.ndim != 2 or continuous.shape[0] != rows:
            raise ValueError(
                f"model must return one row of values per parameter set, a ({rows}, responses) "
                f"tensor, but returned shape {tuple(continuous.shape)}"
            )
        if responses is None:
            responses = int(continuous.shape[1])
        elif continuous.shape[1] != responses:
            raise ValueError(
                f"model returned {continuous.shape[1]} values, but control['responses'] "
                f"is {responses}"
            )
        continuous = continuous.to(run_dtype)

        ordinal = as_numeric_tensor(discretize(continuous), "discretize", run_device)
        if ordinal.ndim == 2 and rows == 1:
            ordinal = ordinal.unsqueeze(0)
        if ordinal.ndim != 3 or ordinal.shape[0] != rows:
            raise ValueError(
                f"discretize must return a numeric matrix per parameter set, a "
                f"({rows}, dimensionality, dimensionality) tensor, but returned shape "
                f"{tuple(ordinal.shape)}"
            )
        if dimensionality is None:
            if ordinal.shape[1] != ordinal.shape[2]:
                raise ValueError(
                    f"discretize must return a square matrix, but returned "
                    f"{ordinal.shape[1]} x {ordinal.shape[2]}"
                )
            dimensionality = int(ordinal.shape[1])
        elif ordinal.shape[1] != dimensionality or ordinal.shape[2] != dimensionality:
            raise ValueError(
                f"discretize returned a {ordinal.shape[1]} x {ordinal.shape[2]} matrix, but "
                f"control['dimensionality'] is {dimensionality}"
            )
        return continuous, ordinal

    store = PatternStore(settings.dimensions, run_device, run_dtype)
    iteration = 0
    writer: RunWriter | None = None
    try:
        jumping = settings.init.clone()
        continuous, ordinal = evaluate(jumping)
        ids = store.update(ordinal, jumping)
        if save:
            stimuli_names = settings.stimuli_names
            if stimuli_names is None:
                stimuli_names = tuple(f"stimulus_{k + 1}" for k in range(int(continuous.shape[1])))
            writer = RunWriter(path, extension, settings.parameter_names, stimuli_names)
            writer.write(0, jumping, continuous, ids)
        underpopulated = store.underpopulated(settings.population)

        while underpopulated.numel() > 0 and iteration < settings.iterations:
            iteration += 1
            if not quiet:
                print(f"Iteration: [{iteration}]", flush=True)
            steps = hyper_points(int(underpopulated.numel()), settings.radius, generator=generator)
            jumping = clamp_parameters(
                store.centres_of(underpopulated) + steps, settings.lower, settings.upper
            )
            continuous, ordinal = evaluate(jumping)
            ids = store.update(ordinal, jumping)
            if writer is not None:
                writer.write(iteration, jumping, continuous, ids)
            underpopulated = store.underpopulated(settings.population)
    finally:
        if writer is not None:
            writer.close()

    return PSPResult(
        ordinal_patterns=store.patterns(),
        ordinal_counts=store.counts.clone(),
        iterations=iteration,
    )
