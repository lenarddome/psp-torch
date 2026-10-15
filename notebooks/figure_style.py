"""The figure style shared by the notebooks: publication-ready figures, shown at high resolution and
saved as vector PDF (fonts embedded) and 300 dpi PNG.

Each notebook does::

    from figure_style import figures, titled
    show = figures("tutorial")      # files go to notebooks/figures/tutorial/
    ...
    show(fig, "choices")            # saves choices.pdf and choices.png, then displays the figure

The notebooks in subfolders add the parent folder to ``sys.path`` first.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:  # twice the pixels on screen, for sharp figures on high-resolution displays
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
except ImportError:  # outside Jupyter
    pass

FIGURES = Path(__file__).resolve().parent / "figures"

# The categorical palette, in this fixed order; validated for colour-vision deficiency (adjacent
# pairs ΔE ≥ 9 for deuteranopia). Aqua has low contrast on white, so every chart that uses it
# also labels its series directly or encodes them with markers or line styles.
blue, orange, aqua, yellow, magenta, violet = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#6250d6",
)
ink, text, muted, light, reference = "#212529", "#495057", "#6c757d", "#e9ecef", "#adb5bd"

plt.rcParams.update(
    {
        # resolution: 110 dpi on screen (doubled by retina), 300 dpi in saved bitmaps
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        # fonts embedded as TrueType in PDF and PostScript, as journals ask; SVG keeps its text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        # quiet axes: no frame, a light grid behind the data
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "grid.color": light,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.labelcolor": text,
        "xtick.color": text,
        "ytick.color": text,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.figsize": (7, 4.6),
        "lines.solid_capstyle": "round",
    }
)


def titled(ax: Axes, title: str, subtitle: str | None = None, pad: float = 0) -> Axes:
    """A bold title and a grey subtitle, both left-aligned."""
    ax.set_title(title, loc="left", pad=(28 if subtitle else 10) + pad)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 7 + pad),
            textcoords="offset points",
            color=muted,
            fontsize=11,
            va="bottom",
        )
    return ax


def figures(notebook: str) -> Callable[[Figure, str], None]:
    """``show(fig, name)`` for one notebook: save the figure as PDF and PNG, then display it."""
    folder = FIGURES / notebook

    def show(fig: Figure, name: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.pdf")
        fig.savefig(folder / f"{name}.png")
        plt.show()

    return show
