"""The dataset overview figure: one merged axis per dataset (Layout A).

Headline metric: macro-F1 (Ash, 2026-08-07) — equal class weighting, immune
to majority-class inflation; accuracy stays in the artifacts and tables.

Chosen by Ash (2026-08-06) as the one-figure-per-dataset manuscript layout:
Random floor and node-features reference boxes on the left (each with a
dashed mean line), then one local/global box pair per k-hop construction, a
light family separator, and one pair per walk construction — all on a single
axis with shared, data-driven y-limits (margin + 0.05-tick snap; box plots
encode position, not area, so the truncated axis is standard and the two
reference lines stay in-window as absolute anchors).

Import plotstyle before pyplot (it selects the Agg backend); sessions call
`dataset_overview_figure` from a thin `code/plot_results.py` wrapper so every
dataset renders identically.
"""

import math

from ..containment import plotstyle as ps

import matplotlib.pyplot as plt
import pandas as pd

LOCAL_TAG = "fall-vl1"
GLOBAL_TAG = "fall-vl1-glob"
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
NODE_COLOR = ps.FAMILY_COLOR["egonet_walk"]  # green: the node-features reference
PAIR_OFFSET = 0.21
BOX_W = 0.36


def _box(ax, values, position, color, width=BOX_W):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True)
    patch, med = b["boxes"][0], b["medians"][0]
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor(color)
    med.set_color(ps.INK)


def _pairs(ax, acc, methods, start_x):
    ticks = []
    for i, method in enumerate(methods):
        center = start_x + i
        for tag, side in ((LOCAL_TAG, -PAIR_OFFSET), (GLOBAL_TAG, +PAIR_OFFSET)):
            vals = acc(tag, method)
            if vals is not None:
                _box(ax, vals, center + side, SCOPE_COLOR["local" if side < 0 else "global"])
        ticks.append(center)
    return ticks


def shared_ylim(df: pd.DataFrame, metric: str = "f1_macro") -> tuple:
    """Data-driven limits over every plotted method (majority stays off figures)."""
    plotted = df.loc[df["method"] != "majority", metric]
    return (math.floor((plotted.min() - 0.02) * 20) / 20, math.ceil((plotted.max() + 0.02) * 20) / 20)


def dataset_overview_figure(df: pd.DataFrame, khop_methods: dict, walk_methods: dict, stem, height: float = 2.8, strict: bool = True, metric: str = "f1_macro", metric_label: str = "macro-F1"):
    """Render the merged-axis overview for one dataset.

    Args:
        df: the session's results.csv frame (status=="ok" rows).
        khop_methods / walk_methods: {method_id: tick label}, plot order.
        stem: output path stem (Path, no extension); writes .pdf + .png.
        height: figure height in inches.
    """

    def acc(tag, method):
        vals = df.loc[(df["feature_tag"] == tag) & (df["method"] == method), metric].to_numpy()
        if len(vals) == 0:
            if strict:
                raise ValueError(f"No rows for tag={tag} method={method} — run that scope first")
            return None
        return vals

    ylim = shared_ylim(df, metric)
    ps.use_style()
    fig, ax = plt.subplots(figsize=(ps.FULL_W, height))

    floor_vals = acc(LOCAL_TAG, "permuted")
    node_vals = acc(LOCAL_TAG, "node_struct")
    if floor_vals is not None:
        _box(ax, floor_vals, 0.0, ps.FAMILY_COLOR["floor"])
        ax.axhline(float(floor_vals.mean()), color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    if node_vals is not None:
        _box(ax, node_vals, 1.05, NODE_COLOR)
        ax.axhline(float(node_vals.mean()), color=NODE_COLOR, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)

    khop_ticks = _pairs(ax, acc, khop_methods, 2.35)
    gap_x = khop_ticks[-1] + 0.75
    ax.axvline(gap_x, color=ps.GRID, linewidth=0.8, zorder=0)
    walk_ticks = _pairs(ax, acc, walk_methods, gap_x + 0.75)

    ax.set_xticks([0.0, 1.05] + khop_ticks + walk_ticks)
    ax.set_xticklabels(
        ["Random\n(permuted)", "Node feat.\n(full graph)"] + list(khop_methods.values()) + list(walk_methods.values())
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(metric_label)
    ax.text((khop_ticks[0] + khop_ticks[-1]) / 2, -0.24, "$k$-hop egonets",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=ps.MUTED)
    ax.text((walk_ticks[0] + walk_ticks[-1]) / 2, -0.24, "random-walk bags",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=ps.MUTED)

    handles = [
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
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, stem)
    return stem
