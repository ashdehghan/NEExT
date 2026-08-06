"""Experiment 1 figure: accuracy box plot, approaches on the x-axis.

Reads the session results.csv aggregate only. Single panel: accuracy across
the 10 shared splits for the permutation floor and egonet k=1/2/3. Majority
and the other metrics (macro-F1, AUC) stay in the artifacts, off the figure.
Floor grey, egonets blue (plotstyle family palette); legend above the axes,
never inside. Output: figures/exp1_signal_box.{pdf,png}.

Usage: uv run --with matplotlib python code/plot_results.py
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps  # noqa: E402  (sets Agg before pyplot)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

FIGURES = SESSION_ROOT / "figures"
CHANCE = 0.25  # 4 near-balanced quartile classes


def family_of(method: str) -> str:
    return "floor" if method == "permuted" else "egonet_hop"


def main():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df["status"] == "ok"]

    ps.use_style()
    methods = list(C.PLOT_METHODS)
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.6))
    data = [df.loc[df["method"] == m, "accuracy"].to_numpy() for m in methods]
    boxes = ax.boxplot(data, positions=range(len(methods)), widths=0.55, showfliers=False, patch_artist=True)
    for patch, med, m in zip(boxes["boxes"], boxes["medians"], methods):
        color = ps.FAMILY_COLOR[family_of(m)]
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        med.set_color(ps.INK)
    ax.axhline(CHANCE, color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(len(methods)), [C.PLOT_METHODS[m] for m in methods])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["egonet_hop"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["egonet_hop"], label="Egonet + approx. Wasserstein"),
        plt.Line2D([], [], color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                   label="Uniform chance (0.25)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "exp1_signal_box")
    print(f"wrote {FIGURES / 'exp1_signal_box'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
