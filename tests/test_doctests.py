"""The examples in the docstrings run and give the documented output."""

from __future__ import annotations

import doctest
import importlib

import pytest

import psp
import psp._callbacks
import psp._global
import psp.models.alcove
import psp.models.prospect

# psp.clobe.<name> is the function, so its file is looked up by module name
CLOBE = [
    "imac",
    "patterns",
    "gdistance",
    "pattern_distance",
    "combinatorics",
    "orderings",
    "brutes",
]

MODULES = [
    psp,
    psp._callbacks,
    psp.clobe,
    *(importlib.import_module(f"psp.clobe.{name}") for name in CLOBE),
    psp._global,
    psp.models.alcove,
    psp.models.prospect,
]


@pytest.mark.parametrize("module", MODULES, ids=[m.__name__ for m in MODULES])
def test_docstring_examples(module) -> None:
    failures, _ = doctest.testmod(
        module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert failures == 0
