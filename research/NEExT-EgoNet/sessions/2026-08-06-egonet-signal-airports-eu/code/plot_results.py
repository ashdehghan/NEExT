"""Experiment 1 figure: accuracy box plot, local vs global feature scope.

Reads the session results.csv aggregate only. Single panel: accuracy across
the 10 shared splits — the random floor (permuted labels) on the left, then
one pair of boxes per k in {1,2,3}: local (in-bag features, tag fall-vl1)
next to global (full-graph features projected onto members, tag
fall-vl1-glob). Majority and macro-F1/AUC stay in the artifacts, off the
figure. Legend above the axes, never inside.

Output: figures/exp1_signal_box.{pdf,png}

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
CHANCE = 0.25  # 4 near-balanced quartile classes

LOCAL_TAG = "fall-vl1"
GLOBAL_TAG = "fall-vl1-glob"
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
K_HOPS = [1, 2, 3]
PAIR_OFFSET = 0.21
BOX_W = 0.36


def draw_box(ax, values, position, color, width=BOX_W):
    boxes = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True)
    patch, med = boxes["boxes"][0], boxes["medians"][0]
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor(color)
    med.set_color(ps.INK)


def main():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df["status"] == "ok"]

    def acc(tag, method):
        vals = df.loc[(df["feature_tag"] == tag) & (df["method"] == method), "accuracy"].to_numpy()
        if len(vals) == 0:
            raise ValueError(f"No rows for tag={tag} method={method} — run the experiment for that scope first")
        return vals

    ps.use_style()
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.6))

    draw_box(ax, acc(LOCAL_TAG, "permuted"), 0.0, ps.FAMILY_COLOR["floor"], width=BOX_W)
    for i, k in enumerate(K_HOPS):
        center = 1.2 + i
        draw_box(ax, acc(LOCAL_TAG, f"egonet_k{k}_wass"), center - PAIR_OFFSET, SCOPE_COLOR["local"])
        draw_box(ax, acc(GLOBAL_TAG, f"egonet_k{k}_wass"), center + PAIR_OFFSET, SCOPE_COLOR["global"])

    ax.axhline(CHANCE, color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xticks([0.0] + [1.2 + i for i in range(len(K_HOPS))])
    ax.set_xticklabels(["Random\n(permuted labels)"] + [f"Egonet $k{{=}}{k}$" for k in K_HOPS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["local"], label="Local (in-bag features)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["global"], label="Global (full-graph features)"),
        plt.Line2D([], [], color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                   label="Uniform chance (0.25)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "exp1_signal_box")
    print(f"wrote {FIGURES / 'exp1_signal_box'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
