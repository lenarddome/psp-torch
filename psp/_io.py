"""Writing and reading the files produced by ``pspGlobal(save=True)``."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch

__all__ = ["SavedRun", "read_saved", "saved_paths", "has_pyarrow"]

#: Rows are formatted and written in chunks of this many, not one iteration at a time.
FLUSH_ROWS = 4096


def saved_paths(path: str | Path, extension: str = ".csv") -> tuple[Path, Path]:
    """The two files a run with ``save=True`` writes: parameters and continuous outputs."""
    path = str(path)
    return Path(f"{path}_parameters{extension}"), Path(f"{path}_continuous{extension}")


def has_pyarrow() -> bool:
    """Whether pyarrow, the optional fast csv writer, is installed."""
    return find_spec("pyarrow") is not None


class _Table:
    """One csv file: a header, then rows of ``iteration, <values...>, pattern``.

    Floats are written in their shortest round-trip form, so reading them back
    gives exactly the same doubles. With pyarrow installed, the formatting runs
    in C++ and is several times faster; without it, Python's ``repr`` is used.
    Rows are buffered and written in chunks of :data:`FLUSH_ROWS`.
    """

    def __init__(self, path: Path, names: Sequence[str], use_pyarrow: bool) -> None:
        self._file: IO[bytes] = open(path, "wb")
        self._file.write((",".join(["iteration", *names, "pattern"]) + "\n").encode("utf-8"))
        self._use_pyarrow = use_pyarrow
        self._iterations: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._ids: list[np.ndarray] = []
        self._pending = 0

    def append(self, iteration: int, values: np.ndarray, ids: np.ndarray) -> None:
        self._iterations.append(np.full(values.shape[0], iteration, dtype=np.int64))
        self._values.append(values)
        self._ids.append(ids)
        self._pending += int(values.shape[0])
        if self._pending >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        iterations = np.concatenate(self._iterations)
        values = np.concatenate(self._values)
        ids = np.concatenate(self._ids)
        self._iterations.clear()
        self._values.clear()
        self._ids.clear()
        self._pending = 0
        if self._use_pyarrow:
            self._file.write(_format_pyarrow(iterations, values, ids))
        else:
            self._file.write(_format_python(iterations, values, ids))

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._file.close()


def _format_python(iterations: np.ndarray, values: np.ndarray, ids: np.ndarray) -> bytes:
    # repr() gives the shortest text that reads back as exactly the same double
    lines = [
        f"{iteration},{','.join(map(repr, row))},{pattern}"
        for iteration, row, pattern in zip(
            iterations.tolist(), values.tolist(), ids.tolist(), strict=True
        )
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _format_pyarrow(iterations: np.ndarray, values: np.ndarray, ids: np.ndarray) -> bytes:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    columns: dict[str, Any] = {"iteration": pa.array(iterations)}
    for k in range(values.shape[1]):
        columns[f"c{k}"] = pa.array(np.ascontiguousarray(values[:, k]))
    columns["pattern"] = pa.array(ids)
    sink = pa.BufferOutputStream()
    pacsv.write_csv(pa.table(columns), sink, write_options=pacsv.WriteOptions(include_header=False))
    return sink.getvalue().to_pybytes()


class RunWriter:
    """Writes every evaluated parameter set and its model output to two csv files.

    Each row is ``iteration, <values...>, pattern``, with 0-based pattern ids that
    index ``ordinal_patterns``. See :class:`_Table` for the number formatting.

    Parameters
    ----------
    use_pyarrow
        Format with pyarrow when it is installed (the default), or always with Python.
    """

    def __init__(
        self,
        path: str | Path,
        extension: str,
        parameter_names: Sequence[str],
        stimuli_names: Sequence[str],
        *,
        use_pyarrow: bool | None = None,
    ) -> None:
        fast = has_pyarrow() if use_pyarrow is None else use_pyarrow
        parameters_path, continuous_path = saved_paths(path, extension)
        self._parameters = _Table(parameters_path, parameter_names, fast)
        self._continuous = _Table(continuous_path, stimuli_names, fast)

    def write(
        self, iteration: int, parameters: torch.Tensor, continuous: torch.Tensor, ids: torch.Tensor
    ) -> None:
        pattern_ids = ids.detach().cpu().numpy().astype(np.int64)
        self._parameters.append(iteration, _as_doubles(parameters), pattern_ids)
        self._continuous.append(iteration, _as_doubles(continuous), pattern_ids)

    def close(self) -> None:
        try:
            self._parameters.close()
        finally:
            self._continuous.close()

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _as_doubles(values: torch.Tensor) -> np.ndarray:
    return values.detach().to("cpu", torch.float64).numpy().reshape(values.shape[0], -1)


class SavedRun:
    """The two tables written by a run with ``save=True``, as pandas data frames.

    Attributes
    ----------
    parameters
        One row per evaluation: ``iteration``, one column per parameter, ``pattern``.
    continuous
        One row per evaluation: ``iteration``, one column per model output, ``pattern``.
    """

    def __init__(self, parameters: Any, continuous: Any) -> None:
        self.parameters = parameters
        self.continuous = continuous

    def __iter__(self):
        yield self.parameters
        yield self.continuous


def read_saved(path: str | Path, extension: str = ".csv") -> SavedRun:
    """Read the files written by ``pspGlobal(save=True, path=path, extension=extension)``.

    Requires pandas, which is an optional dependency (``pip install psp-torch[notebooks]``).
    """
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("read_saved needs pandas: pip install pandas") from error
    parameters_path, continuous_path = saved_paths(path, extension)
    # the round-trip parser reads back exactly the doubles that were written
    return SavedRun(
        pd.read_csv(parameters_path, float_precision="round_trip"),
        pd.read_csv(continuous_path, float_precision="round_trip"),
    )
