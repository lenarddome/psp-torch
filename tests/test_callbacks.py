"""The contract between pspGlobal and the user-supplied model and discretize functions.

A port of tests/testthat/test_psp_cpp_callbacks.R.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psp import pspGlobal, rowwise


def callback_control() -> dict:
    return {
        "iterations": 2,
        "population": 100,
        "radius": 0.1,
        "lower": [0, 0],
        "upper": [1, 1],
        "init": [[0.5, 0.5]],
        "parameter_names": ["a", "b"],
        "stimuli_names": ["a", "b"],
        "dimensionality": 2,
        "responses": 2,
    }


def test_model_receives_a_batch_of_parameter_sets() -> None:
    seen = []

    def model(par: torch.Tensor) -> torch.Tensor:
        seen.append(par)
        return torch.tensor([[1.0, 2.0]]).expand(par.shape[0], 2)

    pspGlobal(model, lambda x: torch.ones(x.shape[0], 2, 2), callback_control(), quiet=True)
    assert len(seen) == 3
    for par in seen:
        assert torch.is_tensor(par)
        assert par.dtype == torch.float64
        assert par.shape == (1, 2)


def test_rowwise_model_receives_a_plain_vector() -> None:
    seen = []

    def model(par: torch.Tensor) -> list[float]:
        seen.append(par)
        return [1.0, 2.0]

    pspGlobal(model, lambda x: np.ones((2, 2)), callback_control(), quiet=True, batched=False)
    assert len(seen) == 3
    for par in seen:
        assert par.shape == (2,)
        assert par.dtype == torch.float64


def test_rowwise_numpy_passes_arrays() -> None:
    seen = []
    model = rowwise(lambda par: (seen.append(par), np.array([1, 2]))[1], numpy=True)
    out = model(torch.zeros(3, 2))
    assert all(isinstance(s, np.ndarray) for s in seen)
    assert out.shape == (3, 2)


def test_discretize_receives_model_output_as_doubles() -> None:
    seen = []

    def discretize(x: torch.Tensor) -> torch.Tensor:
        seen.append(x)
        return torch.ones(x.shape[0], 2, 2, dtype=torch.int64)

    out = pspGlobal(
        lambda par: torch.tensor([[1, 2]], dtype=torch.int64).expand(par.shape[0], 2),
        discretize,
        callback_control(),
        quiet=True,
    )
    assert seen[0].dtype == torch.float64
    assert torch.equal(seen[0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert torch.equal(out.ordinal_patterns[0], torch.ones(2, 2, dtype=torch.float64))


def test_errors_raised_by_callbacks_reach_the_caller() -> None:
    def broken_model(par: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("model broke")

    def broken_discretize(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("discretize broke")

    with pytest.raises(RuntimeError, match="model broke"):
        pspGlobal(broken_model, lambda x: torch.ones(1, 2, 2), callback_control(), quiet=True)
    with pytest.raises(RuntimeError, match="discretize broke"):
        pspGlobal(
            lambda par: torch.ones(par.shape[0], 2),
            broken_discretize,
            callback_control(),
            quiet=True,
        )


def test_error_in_a_later_iteration_reaches_the_caller() -> None:
    calls = 0

    def model(par: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError(f"failed on call {calls}")
        return torch.ones(par.shape[0], 2)

    with pytest.raises(RuntimeError, match="failed on call 2"):
        pspGlobal(model, lambda x: torch.ones(x.shape[0], 2, 2), callback_control(), quiet=True)


def test_model_output_of_the_wrong_length_or_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="responses"):
        pspGlobal(
            lambda par: torch.ones(par.shape[0], 3),
            lambda x: torch.ones(1, 2, 2),
            callback_control(),
            quiet=True,
        )
    with pytest.raises(ValueError, match="numeric"):
        pspGlobal(
            lambda par: ["a", "b"],
            lambda x: torch.ones(1, 2, 2),
            callback_control(),
            quiet=True,
            batched=False,
        )
    with pytest.raises(ValueError, match="one row of values per parameter set"):
        pspGlobal(
            lambda par: torch.ones(5, 2),
            lambda x: torch.ones(1, 2, 2),
            callback_control(),
            quiet=True,
        )


def test_discretize_output_that_is_not_a_dimensionality_sized_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="matrix"):
        pspGlobal(
            lambda par: torch.ones(par.shape[0], 2),
            lambda x: torch.ones(4),
            callback_control(),
            quiet=True,
        )
    with pytest.raises(ValueError, match="dimensionality"):
        pspGlobal(
            lambda par: torch.ones(par.shape[0], 2),
            lambda x: torch.ones(1, 3, 3),
            callback_control(),
            quiet=True,
        )


def test_sizes_are_inferred_when_control_omits_them() -> None:
    ctrl = callback_control()
    del ctrl["responses"], ctrl["dimensionality"], ctrl["stimuli_names"]
    out = pspGlobal(
        lambda par: torch.ones(par.shape[0], 7),
        lambda x: torch.ones(x.shape[0], 3, 3),
        ctrl,
        quiet=True,
    )
    assert out.ordinal_patterns.shape == (1, 3, 3)
    with pytest.raises(ValueError, match="square"):
        pspGlobal(
            lambda par: torch.ones(par.shape[0], 7),
            lambda x: torch.ones(x.shape[0], 2, 3),
            ctrl,
            quiet=True,
        )


def test_invalid_control_settings_are_rejected() -> None:
    ctrl = callback_control()
    ctrl["iterations"] = 0
    ctrl["population"] = 0
    with pytest.raises(ValueError, match="threshold"):
        pspGlobal(lambda par: torch.ones(1, 2), lambda x: torch.ones(1, 2, 2), ctrl, quiet=True)
    ctrl = callback_control()
    ctrl["upper"] = [1]
    with pytest.raises(ValueError, match="same length"):
        pspGlobal(lambda par: torch.ones(1, 2), lambda x: torch.ones(1, 2, 2), ctrl, quiet=True)
    ctrl = callback_control()
    ctrl["parameter_names"] = ["p1"]
    with pytest.raises(ValueError, match="param_names"):
        pspGlobal(lambda par: torch.ones(1, 2), lambda x: torch.ones(1, 2, 2), ctrl, quiet=True)
    ctrl = callback_control()
    ctrl["typo"] = 1
    with pytest.raises(ValueError, match="unknown control keys"):
        pspGlobal(lambda par: torch.ones(1, 2), lambda x: torch.ones(1, 2, 2), ctrl, quiet=True)
    ctrl = callback_control()
    ctrl["radius"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="radius"):
        pspGlobal(lambda par: torch.ones(1, 2), lambda x: torch.ones(1, 2, 2), ctrl, quiet=True)


def test_unbounded_settings_accept_none_and_inf() -> None:
    ctrl = callback_control()
    ctrl["population"] = None
    out = pspGlobal(
        lambda par: torch.ones(par.shape[0], 2),
        lambda x: torch.ones(x.shape[0], 2, 2),
        ctrl,
        quiet=True,
    )
    assert out.iterations == 2
    ctrl["population"] = float("inf")
    out = pspGlobal(
        lambda par: torch.ones(par.shape[0], 2),
        lambda x: torch.ones(x.shape[0], 2, 2),
        ctrl,
        quiet=True,
    )
    assert out.iterations == 2
