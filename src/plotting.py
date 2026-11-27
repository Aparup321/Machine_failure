"""Shared matplotlib styling so every figure in the report looks consistent."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering, works on any machine
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

PALETTE = {0: "#2E86AB", 1: "#E4572E"}  # healthy / failure
ACCENT = "#264653"
MODEL_COLORS = ["#2A9D8F", "#E9C46A", "#E76F51", "#8E7DBE"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
        }
    )


def save_figure(fig, filename: str, subdir: str | None = None) -> str:
    """Save a figure into reports/figures (optionally a subfolder)."""
    folder = config.FIGURES_DIR
    if subdir:
        folder = folder / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


setup_style()
