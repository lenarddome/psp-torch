"""The csv writer: both formatting paths write the same, exact, text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from psp import _io
from psp._io import RunWriter, has_pyarrow, saved_paths


def _write(path: Path, use_pyarrow: bool, rows: int) -> tuple[str, str]:
    generator = torch.Generator().manual_seed(0)
    parameters = torch.rand((rows, 3), generator=generator, dtype=torch.float64) * 1e3 - 500
    continuous = torch.rand((rows, 4), generator=generator, dtype=torch.float64) * 1e-3
    continuous[0, 0] = 1 / 3
    continuous[0, 1] = 1 + 20 * np.finfo(np.float64).eps
    ids = torch.arange(rows) % 7
    with RunWriter(
        path, ".csv", ["a", "b", "c"], ["w", "x", "y", "z"], use_pyarrow=use_pyarrow
    ) as w:
        w.write(0, parameters[:1], continuous[:1], ids[:1])
        w.write(1, parameters[1:], continuous[1:], ids[1:])
    p, c = saved_paths(path)
    return p.read_text(encoding="utf-8"), c.read_text(encoding="utf-8")


def test_python_formatting_is_exact(tmp_path: Path) -> None:
    text, continuous = _write(tmp_path / "run", use_pyarrow=False, rows=10)
    lines = text.splitlines()
    assert lines[0] == "iteration,a,b,c,pattern"
    assert lines[1].startswith("0,") and lines[1].endswith(",0")
    assert lines[2].startswith("1,")
    for line in continuous.splitlines()[1:]:
        cells = line.split(",")
        assert [float(x) for x in cells[1:-1]] == [float(repr(float(x))) for x in cells[1:-1]]
    assert "0.3333333333333333" in continuous.splitlines()[1]
    assert "1.0000000000000044" in continuous.splitlines()[1]


@pytest.mark.skipif(not has_pyarrow(), reason="pyarrow is not installed")
def test_pyarrow_and_python_formatting_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_io, "FLUSH_ROWS", 16)  # several flushes within one run
    fast = _write(tmp_path / "fast", use_pyarrow=True, rows=50)
    slow = _write(tmp_path / "slow", use_pyarrow=False, rows=50)
    for a, b in zip(fast, slow, strict=True):
        rows_a = [[float(x) for x in line.split(",")] for line in a.splitlines()[1:]]
        rows_b = [[float(x) for x in line.split(",")] for line in b.splitlines()[1:]]
        assert rows_a == rows_b
        assert len(rows_a) == 50


def test_rows_are_flushed_on_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_io, "FLUSH_ROWS", 10**9)
    text, _ = _write(tmp_path / "late", use_pyarrow=False, rows=5)
    assert len(text.splitlines()) == 6
