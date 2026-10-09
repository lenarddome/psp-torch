"""Re-executes the notebooks in this directory in place, refreshing their outputs.

Usage::

    uv run python notebooks/execute.py [tutorial.ipynb recipe-alcove.ipynb benchmarks.ipynb]

Run ``alcove/run_alcove.py`` first if the ALCOVE results it writes are missing,
and ``benchmarks/compare.py`` for the benchmark results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
DEFAULT = ("tutorial.ipynb", "recipe-alcove.ipynb", "benchmarks.ipynb")


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    for name in names:
        path = HERE / name
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(HERE)}},
        ).execute()
        nbformat.write(notebook, path)
        print("executed", path.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
