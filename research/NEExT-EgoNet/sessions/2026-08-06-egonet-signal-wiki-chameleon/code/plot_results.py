"""Experiment 3 figures: Wikipedia Chameleon, local vs global scope.

Two figures, same layout as the airports session: the permutation floor and
the node-features box (green, with its dashed reference line) on the left,
then one local/global pair per construction — k in {1..4} for the k-hop
figure, the tuned walk trio for the walk figure. The dashed grey line marks
the permutation-floor mean (18 imbalanced classes — uniform chance is not
meaningful). Legend above the axes, never inside.

Outputs: figures/exp3_wiki_scopes.{pdf,png}, figures/exp3_wiki_walk.{pdf,png}

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
GLOBAL_TAG = "fall-vl1-glob"
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
NODE_COLOR = ps.FAMILY_COLOR["egonet_walk"]  # green: the node-features reference
PAIR_OFFSET = 0.21
BOX_W = 0.36


def draw_box(ax, values, position, color, width=BOX_W):
    boxes = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True)
    patch, med = boxes["boxes"][0], boxes["medians"][0]
    patch.set_facecolor(color)
    patch.set_alpha(0.35)
    patch.set_edgecolor(color)
    med.set_color(ps.INK)


def paired_figure(df, methods_labels, stem, ylim):
    def acc(tag, method):
        vals = df.loc[(df["feature_tag"] == tag) & (df["method"] == method), "accuracy"].to_numpy()
        if len(vals) == 0:
            raise ValueError(f"No rows for tag={tag} method={method} — run that scope first")
        return vals

    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.6))
    floor_vals = acc(LOCAL_TAG, "permuted")
    draw_box(ax, floor_vals, 0.0, ps.FAMILY_COLOR["floor"], width=BOX_W)
    node_vals = acc(LOCAL_TAG, "node_struct")
    draw_box(ax, node_vals, 0.85, NODE_COLOR, width=BOX_W)
    ax.axhline(float(node_vals.mean()), color=NODE_COLOR, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    for i, method in enumerate(methods_labels):
        center = 2.05 + i
        draw_box(ax, acc(LOCAL_TAG, method), center - PAIR_OFFSET, SCOPE_COLOR["local"])
        draw_box(ax, acc(GLOBAL_TAG, method), center + PAIR_OFFSET, SCOPE_COLOR["global"])

    ax.axhline(float(floor_vals.mean()), color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
    ax.set_xticks([0.0, 0.85] + [2.05 + i for i in range(len(methods_labels))])
    ax.set_xticklabels(["Random\n(permuted)", "Node features\n(full graph)"] + list(methods_labels.values()))
    ax.set_ylim(*ylim)
    ax.set_ylabel("accuracy")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=NODE_COLOR, alpha=0.35,
                      edgecolor=NODE_COLOR, label="Node features"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["local"], label="Local (in-bag)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["global"], label="Global (full-graph)"),
        plt.Line2D([], [], color=ps.MUTED, linewidth=0.6, linestyle=(0, (4, 3)),
                   label="Floor mean"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / stem)
    print(f"wrote {FIGURES / stem}.{{pdf,png}}")


def main():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df["status"] == "ok"]
    ps.use_style()

    # Shared, data-driven y-limits across BOTH figures of this dataset:
    # a margin below the lowest / above the highest plotted value, snapped to
    # 0.05 ticks. Box plots encode position, not area, so a truncated axis is
    # standard — the floor and node-features reference lines stay in-window
    # as absolute anchors.
    import math

    plotted = df[df["method"] != "majority"]["accuracy"]
    ylim = (math.floor((plotted.min() - 0.02) * 20) / 20, math.ceil((plotted.max() + 0.02) * 20) / 20)

    khop_methods = {f"egonet_k{k}_wass": f"Egonet $k{{=}}{k}$" for k in [1, 2, 3]}
    paired_figure(df, khop_methods, "exp3_wiki_scopes", ylim)

    walk_methods = {
        "walk_tight_wass": "Walk tight\n($\\ell{=}5$, $r{=}.30$)",
        "walk_default_wass": "Walk default\n($\\ell{=}10$, $r{=}.15$)",
        "walk_wide_wass": "Walk wide\n($\\ell{=}20$, $r{=}.05$)",
    }
    paired_figure(df, walk_methods, "exp3_wiki_walk", ylim)


if __name__ == "__main__":
    main()
