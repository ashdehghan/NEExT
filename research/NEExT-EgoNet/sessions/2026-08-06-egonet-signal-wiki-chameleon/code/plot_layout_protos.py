"""Prototype the two candidate one-figure-per-dataset layouts (Chameleon data).

Layout A (merged axis): floor, node features, k-hop pairs, family gap, walk
pairs — all on one ~2.8 in axis.
Layout B (stacked panels): (a) k-hop and (b) walk bags as vertically stacked
subplots sharing the legend, ~4.4 in.

Outputs: figures/proto_layout_A.{pdf,png}, figures/proto_layout_B.{pdf,png}
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps  # noqa: E402

import math  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIGURES = SESSION_ROOT / "figures"
LOCAL_TAG = "fall-vl1"
GLOBAL_TAG = "fall-vl1-glob"
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
NODE_COLOR = ps.FAMILY_COLOR["egonet_walk"]
PAIR_OFFSET = 0.21
BOX_W = 0.36

KHOP = {f"egonet_k{k}_wass": f"$k{{=}}{k}$" for k in [1, 2, 3]}
WALK = {"walk_tight_wass": "tight", "walk_default_wass": "default", "walk_wide_wass": "wide"}

LEGEND = lambda: [  # noqa: E731
    plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                  edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
    plt.Rectangle((0, 0), 1, 1, facecolor=NODE_COLOR, alpha=0.35,
                  edgecolor=NODE_COLOR, label="Node features"),
    plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.35,
                  edgecolor=SCOPE_COLOR["local"], label="Local (in-bag)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.35,
                  edgecolor=SCOPE_COLOR["global"], label="Global (full-graph)"),
    plt.Line2D([], [], color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), label="Floor mean"),
]


def load():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df["status"] == "ok"]
    plotted = df[df["method"] != "majority"]["accuracy"]
    ylim = (math.floor((plotted.min() - 0.02) * 20) / 20, math.ceil((plotted.max() + 0.02) * 20) / 20)
    return df, ylim


def acc(df, tag, method):
    vals = df.loc[(df["feature_tag"] == tag) & (df["method"] == method), "accuracy"].to_numpy()
    if len(vals) == 0:
        raise ValueError(f"No rows for {tag}/{method}")
    return vals


def box(ax, values, position, color, width=BOX_W):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True)
    patch, med = b["boxes"][0], b["medians"][0]
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor(color)
    med.set_color(ps.INK)


def refs_and_pairs(ax, df, methods, start_x):
    """Draw local/global pairs for `methods` beginning at start_x; return tick positions."""
    ticks = []
    for i, method in enumerate(methods):
        center = start_x + i
        box(ax, acc(df, LOCAL_TAG, method), center - PAIR_OFFSET, SCOPE_COLOR["local"])
        box(ax, acc(df, GLOBAL_TAG, method), center + PAIR_OFFSET, SCOPE_COLOR["global"])
        ticks.append(center)
    return ticks


def layout_a(df, ylim):
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.8))
    floor_vals = acc(df, LOCAL_TAG, "permuted")
    node_vals = acc(df, LOCAL_TAG, "node_struct")
    box(ax, floor_vals, 0.0, ps.FAMILY_COLOR["floor"])
    box(ax, node_vals, 1.05, NODE_COLOR)
    ax.axhline(float(node_vals.mean()), color=NODE_COLOR, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.axhline(float(floor_vals.mean()), color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)

    khop_ticks = refs_and_pairs(ax, df, KHOP, 2.35)
    gap_x = khop_ticks[-1] + 0.75
    ax.axvline(gap_x, color=ps.GRID, linewidth=0.8, zorder=0)
    walk_ticks = refs_and_pairs(ax, df, WALK, gap_x + 0.75)

    ax.set_xticks([0.0, 1.05] + khop_ticks + walk_ticks)
    ax.set_xticklabels(["Random\n(permuted)", "Node feat.\n(full graph)"] + list(KHOP.values()) + list(WALK.values()))
    ax.set_ylim(*ylim)
    ax.set_ylabel("accuracy")
    # family captions under the groups
    ax.text((khop_ticks[0] + khop_ticks[-1]) / 2, -0.24, "$k$-hop egonets",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=ps.MUTED)
    ax.text((walk_ticks[0] + walk_ticks[-1]) / 2, -0.24, "random-walk bags",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=ps.MUTED)

    fig.legend(handles=LEGEND(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "proto_layout_A")
    print("wrote proto_layout_A")


def layout_b(df, ylim):
    fig, axes = plt.subplots(2, 1, figsize=(ps.FULL_W, 4.4), sharex=False)
    for ax, methods, tag_text in zip(axes, (KHOP, WALK), ("(a) $k$-hop egonets", "(b) random-walk bags")):
        floor_vals = acc(df, LOCAL_TAG, "permuted")
        node_vals = acc(df, LOCAL_TAG, "node_struct")
        box(ax, floor_vals, 0.0, ps.FAMILY_COLOR["floor"])
        box(ax, node_vals, 0.85, NODE_COLOR)
        ax.axhline(float(node_vals.mean()), color=NODE_COLOR, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        ax.axhline(float(floor_vals.mean()), color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        ticks = refs_and_pairs(ax, df, methods, 2.05)
        ax.set_xticks([0.0, 0.85] + ticks)
        ax.set_xticklabels(["Random\n(permuted)", "Node feat.\n(full graph)"] + list(methods.values()))
        ax.set_ylim(*ylim)
        ax.set_ylabel("accuracy")
        ps.panel_tag(ax, tag_text)

    fig.legend(handles=LEGEND(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "proto_layout_B")
    print("wrote proto_layout_B")


if __name__ == "__main__":
    ps.use_style()
    df, ylim = load()
    layout_a(df, ylim)
    layout_b(df, ylim)
