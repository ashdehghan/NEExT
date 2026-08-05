"""Shared publication figure style for the NEExT-EgoNet manuscript.

Conventions (print-first):
  - Vector PDF output sized to the manuscript's text width (6.3 in for
    geometry margin=1.1in on letter); half-width figures at 3.05 in.
  - Serif/Computer-Modern-matched text at 7-8 pt so type prints at the
    size it was set (no in-figure titles: captions carry titles).
  - Categorical colors: the first three slots of the validated palette
    (all-pairs CVD-safe on white; validated 2026-08-05). Baselines are
    neutral ink with distinct dash patterns + markers, so identity never
    rides on color alone.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_W = 6.3
HALF_W = 3.05

# Validated categorical slots (fair methods) + neutral reference inks.
COLOR = {
    "wasserstein": "#2a78d6",
    "pooled_all": "#eb6834",
    "pooled_max": "#1baf7a",
    "size_only": "#52514e",
    "node_oracle": "#898781",
}
LABEL = {
    "wasserstein": "Wasserstein",
    "pooled_all": "Pooled (mean+max+p90)",
    "pooled_max": "Pooled (max)",
    "size_only": "Size only (baseline)",
    "node_oracle": "Node oracle (full graph)",
}
MARKER = {
    "wasserstein": "o",
    "pooled_all": "s",
    "pooled_max": "^",
    "size_only": "x",
    "node_oracle": "",
}
LINESTYLE = {
    "wasserstein": "-",
    "pooled_all": "-",
    "pooled_max": "-",
    "size_only": (0, (4, 2)),
    "node_oracle": (0, (1, 1.6)),
}

GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"

RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def use_style():
    plt.rcParams.update(RC)


def panel_tag(ax, text):
    """Top-left in-axes panel tag, e.g. '(a) $k=1$' (captions do titles)."""
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=6,
    )


def save(fig, path_stem):
    """Write vector PDF (for LaTeX) + PNG (for quick inspection)."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=300)
    plt.close(fig)
