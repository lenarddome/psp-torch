"""Sphinx configuration of the psp-torch documentation.

Build it with::

    uv run --group docs sphinx-build -b html docs docs/_build/html

The notebooks are not executed: they are copied from ``notebooks/`` into ``docs/notebooks/``
(not tracked) at the start of every build and shown with the outputs they were committed with.
"""

from __future__ import annotations

import json
import re
import shutil
from importlib.metadata import version as package_version
from pathlib import Path

from sphinx.application import Sphinx

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
GITHUB = "https://github.com/lenarddome/psp-torch"

project = "psp-torch"
author = "Lenard Dome"
copyright = "2026, Lenard Dome"
release = package_version("psp-torch")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    # a local copy of the manuscript, not to be distributed
    "dome2025g-dista.pdf",
]

# -- API reference ---------------------------------------------------------------------------

autodoc_typehints = "none"  # the numpydoc sections describe the types
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "shape", "optional", "default"}

# build with -n to check every cross-reference; these have no page to link to
nitpick_ignore = [
    # documented in the Attributes section of PSPResult, which numpydoc renders without targets
    ("py:attr", "psp.PSPResult.ordinal_patterns"),
    # psp.clobe documents its functions, not the files they live in
    ("py:mod", "psp.clobe.combinatorics"),
    # the example models document what their __all__ exports, and PAD is not in it
    ("py:data", "PAD"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# -- Notebooks and Markdown ------------------------------------------------------------------

nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML ------------------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"psp-torch {release}"
html_favicon = "assets/psp-favicon.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/psp-wordmark-light.svg",
        "image_dark": "_static/psp-wordmark-dark.svg",
        "alt_text": "psp",
    },
    "github_url": GITHUB,
    "navbar_align": "left",
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {
    "github_user": "lenarddome",
    "github_repo": "psp-torch",
    "github_version": "main",
    "doc_path": "docs",
}
# the logos in assets/ are served from _static/, next to the stylesheet
html_static_path = ["_static", "assets"]
html_css_files = ["custom.css"]

# -- Copying the notebooks in ----------------------------------------------------------------

# notebooks/<path> -> docs/notebooks/<path>; the order is that of the site
NOTEBOOKS = [
    "tutorial.ipynb",
    "recipe-alcove.ipynb",
    "clobe/getting-started.ipynb",
    "clobe/inequality-matrices.ipynb",
    "clobe/universal-set.ipynb",
    "clobe/gdistance.ipynb",
    "clobe/zone-of-adequacy.ipynb",
    "benchmarks.ipynb",
    "benchmarks-oscar.ipynb",
    "benchmarks-clobe.ipynb",
]

LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _rewrite_link(target: str, source: Path, output: bool) -> str:
    """Keep links to web pages, anchors and the notebooks on the site; send the rest to GitHub.

    Sphinx resolves links to notebooks in Markdown cells but not in Markdown outputs, so there
    they point at the built page instead.
    """
    if re.match(r"^[a-z]+:", target) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    resolved = (source.parent / path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        return target
    if relative.parts[0] == "notebooks" and str(Path(*relative.parts[1:])) in NOTEBOOKS:
        return re.sub(r"\.ipynb$", ".html", path) + target[len(path) :] if output else target
    kind = "tree" if resolved.is_dir() else "blob"
    return f"{GITHUB}/{kind}/main/{relative.as_posix()}" + (f"#{anchor}" if anchor else "")


def _rewrite_links(text: str | list[str], source: Path, output: bool = False) -> str:
    return LINK.sub(lambda m: m[1] + _rewrite_link(m[2], source, output) + m[3], "".join(text))


def copy_notebooks(app: Sphinx) -> None:
    destination = DOCS / "notebooks"
    shutil.rmtree(destination, ignore_errors=True)
    for name in NOTEBOOKS:
        source = ROOT / "notebooks" / name
        notebook = json.loads(source.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                cell["source"] = _rewrite_links(cell["source"], source)
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "text/markdown" in data:
                    data["text/markdown"] = _rewrite_links(data["text/markdown"], source, True)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


# -- Docstrings ------------------------------------------------------------------------------

# reST ends emphasis only before whitespace or punctuation, so the journal volume in a reference,
# "*Psychological Review, 113*(1)", needs an escaped parenthesis to render
VOLUME = re.compile(r"(\*[^*\n]+\*)\(")


def escape_volumes(
    app: Sphinx, what: str, name: str, obj: object, options: object, lines: list[str]
) -> None:
    lines[:] = [VOLUME.sub(r"\1\\(", line) for line in lines]


def setup(app: Sphinx) -> None:
    app.connect("builder-inited", copy_notebooks)
    app.connect("autodoc-process-docstring", escape_volumes)
