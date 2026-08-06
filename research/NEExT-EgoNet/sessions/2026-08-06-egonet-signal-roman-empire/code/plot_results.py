"""Experiment 2 first-pass figure: Roman Empire, local scope, k in {1,2}.

Accuracy across the 10 shared splits: permutation floor + local k=1 and k=2.
Chance for 17 surviving classes is majority-share-dependent; the dashed grey
line marks the permutation-floor mean. Legend above the axes, never inside.

Output: figures/exp2_roman_local.{pdf,png}

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

FIGURES = SESSION_ROOT / "figures"
LOCAL_TAG = "fall-vl1"
METHODS = {
    "permuted": "Random\n(permuted labels)",
    "egonet_k1_wass": "Egonet $k{=}1$\n(local)",
    "egonet_k2_wass": "Egonet $k{=}2$\n(local)",
    "egonet_k3_wass": "Egonet $k{=}3$\n(local)",
    "egonet_k4_wass": "Egonet $k{=}4$\n(local)",
}


def main():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[(df["status"] == "ok") & (df["feature_tag"] == LOCAL_TAG)]

    def acc(method):
        vals = df.loc[df["method"] == method, "accuracy"].to_numpy()
        if len(vals) == 0:
            raise ValueError(f"No rows for method={method}")
        return vals

    ps.use_style()
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.6))
    for i, (method, label) in enumerate(METHODS.items()):
        color = ps.FAMILY_COLOR["floor"] if method == "permuted" else ps.FAMILY_COLOR["egonet_hop"]
        boxes = ax.boxplot([acc(method)], positions=[i], widths=0.5, showfliers=False, patch_artist=True)
        patch, med = boxes["boxes"][0], boxes["medians"][0]
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        med.set_color(ps.INK)
    ax.axhline(float(acc("permuted").mean()), color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(len(METHODS)), list(METHODS.values()))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["egonet_hop"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["egonet_hop"], label="Egonet, local scope"),
        plt.Line2D([], [], color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                   label="Floor mean"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "exp2_roman_local")
    print(f"wrote {FIGURES / 'exp2_roman_local'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
