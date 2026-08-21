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
import numpy as np
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
    if walk_methods:
        gap_x = khop_ticks[-1] + 0.75
        ax.axvline(gap_x, color=ps.GRID, linewidth=0.8, zorder=0)
        walk_ticks = _pairs(ax, acc, walk_methods, gap_x + 0.75)
    else:
        walk_ticks = []

    ax.set_xticks([0.0, 1.05] + khop_ticks + walk_ticks)
    ax.set_xticklabels(
        ["Random\n(permuted)", "Node feat.\n(full graph)"] + list(khop_methods.values()) + list(walk_methods.values())
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(metric_label)
    ax.text((khop_ticks[0] + khop_ticks[-1]) / 2, -0.24, "$k$-hop egonets",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=ps.MUTED)
    if walk_ticks:
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


# ---------------------------------------------------------------------------
# External node-embedding baselines (session 2026-08-21-embedding-baselines)
# ---------------------------------------------------------------------------

EMB_FAMILY_COLOR = {"proximity": ps.FAMILY_COLOR["emb_proximity"], "structural": ps.FAMILY_COLOR["emb_structural"]}
BEST_BAG_COLOR = ps.FAMILY_COLOR["egonet_hop"]


def baselines_overview_figure(comb: pd.DataFrame, panels: list, methods: dict, stem, metric: str = "f1_macro", metric_label: str = "macro-F1", ncols: int = 2, height: float = 4.8):
    """One panel per dataset: a box per embedding method (values over the 10
    shared splits), colored by family, against three dashed references read
    from the same combined frame — the paper's permutation floor (grey), the
    node-features baseline (green) and the best egonet bag (blue).

    Args:
        comb: combined_results.csv frame (status == "ok").
        panels: list of dicts {key, title, best_bag: (method, tag), emb_tag}.
        methods: {method_id: (family, tick label)} in plot order.
        stem: output path stem (no extension).
    """
    ps.use_style()
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ps.FULL_W, height), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        d = comb[comb.dataset_key == panel["key"]]
        floor = d.loc[(d.method == "permuted") & (d.feature_tag == "fall-vl1"), metric]
        node = d.loc[(d.method == "node_struct") & (d.feature_tag == "fall-vl1"), metric]
        bm, bt = panel["best_bag"]
        bag = d.loc[(d.method == bm) & (d.feature_tag == bt), metric]
        for x, (method, (family, _label)) in enumerate(methods.items()):
            vals = d.loc[(d.method == method) & (d.feature_tag == panel["emb_tag"]), metric].to_numpy()
            if len(vals):
                _box(ax, vals, x, EMB_FAMILY_COLOR[family], width=0.55)
        for vals, color in ((floor, ps.MUTED), (node, NODE_COLOR), (bag, BEST_BAG_COLOR)):
            if len(vals):
                ax.axhline(float(vals.mean()), color=color, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([lab for _, lab in methods.values()], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ps.panel_tag(ax, panel["title"])
        if len(methods) > 3:
            ax.axvline(2.5, color=ps.GRID, linewidth=0.8, zorder=0)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_label)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EMB_FAMILY_COLOR["proximity"], alpha=0.35, edgecolor=EMB_FAMILY_COLOR["proximity"], label="Proximity embedding"),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMB_FAMILY_COLOR["structural"], alpha=0.35, edgecolor=EMB_FAMILY_COLOR["structural"], label="Structural embedding"),
        plt.Line2D([], [], color=ps.MUTED, linewidth=0.8, linestyle=(0, (4, 3)), label="Permutation floor"),
        plt.Line2D([], [], color=NODE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), label="Node features"),
        plt.Line2D([], [], color=BEST_BAG_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), label="Best egonet bag"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, stem)
    return stem


def _lift_per_split(sep: pd.DataFrame, method: str, tag: str, B: int):
    ok = sep[(sep.status == "ok") & (sep.buffer_hops == B)]
    m = ok[(ok.method == method) & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    f = ok[(ok.method == f"permuted_{method}") & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    idx = m.index.intersection(f.index)
    return (m.loc[idx] - f.loc[idx]).to_numpy()


def separation_families_figure(panels: list, stem, height: float = 3.0, min_splits: int = 5):
    """Region-holdout lift vs buffer for the embedding families next to the
    best bag and node features. `panels` = list of dicts:
      {title, buffers, series: [(sep_df, method, tag, color, label, shared_ref)]}
    where sep_df is the separation CSV frame the series lives in and
    shared_ref its shared-split lift (computed by the caller, never hard-coded)."""
    ps.use_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(ps.FULL_W, height), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        buffers = panel["buffers"]
        xpos = {B: i for i, B in enumerate(buffers)}
        offsets = np.linspace(-0.28, 0.28, len(panel["series"]))
        for (sep, method, tag, color, _label, shared_ref), off in zip(panel["series"], offsets):
            xs, meds = [], []
            for B in buffers:
                lifts = _lift_per_split(sep, method, tag, B)
                if len(lifts) >= min_splits:
                    _box(ax, lifts, xpos[B] + off, color, width=0.16)
                    xs.append(xpos[B] + off)
                    meds.append(float(np.median(lifts)))
            if xs:
                ax.plot(xs, meds, color=color, lw=1.0, zorder=2)
            if shared_ref is not None and np.isfinite(shared_ref):
                ax.axhline(shared_ref, color=color, lw=0.8, linestyle=(0, (2, 2)), alpha=0.7)
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(B) for B in buffers])
        ax.set_xlabel("separation buffer $B$ (hops)")
        ps.panel_tag(ax, panel["title"])
    axes[0, 0].set_ylabel("macro-F1 lift over paired floor")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.35, edgecolor=c, label=lab) for c, lab in panels[0].get("legend", [])]
    handles.append(plt.Line2D([], [], color=ps.INK, lw=0.8, linestyle=(0, (2, 2)), alpha=0.7, label="shared-split lift (reference)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    fig.tight_layout()
    ps.save(fig, stem)
    return stem
