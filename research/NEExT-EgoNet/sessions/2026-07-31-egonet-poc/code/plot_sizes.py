"""Regenerate the AIRPORTS_USA egonet-size figure at publication quality.

The original run saved only summary size stats, so this script rebuilds the
egonets (cheap post-0.3.10: construction only, no features), persists the
raw per-egonet sizes to outputs/egonet_sizes.csv (closing the
figures-regenerate-from-CSVs gap for this session), and draws ONE
two-panel figure (k=1 | k=2) replacing the two separate default-styled
histograms. Rerun draws from the CSV if it already exists.
"""

import sys
from pathlib import Path

import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps  # noqa: E402

ps.use_style()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"
SIZES_CSV = OUTPUTS / "egonet_sizes.csv"


def compute_sizes() -> pd.DataFrame:
    from datasets import load_single_graph_dataset

    from NEExT import NEExT

    nxt = NEExT(log_level="ERROR")
    edges_df, nodes_df = load_single_graph_dataset("AIRPORTS_USA", label_column="activity_quartile", structural_only=True)
    frames = []
    for k in (1, 2):
        gc = nxt.load_single_graph_from_dfs(
            edges_df=edges_df, nodes_df=nodes_df, graph_id="AIRPORTS_USA", filter_largest_component=True
        )
        egonets = nxt.compute_k_hop_egonets(gc, k_hop=k, egonet_feature_target="activity_quartile", random_seed=13)
        frames.append(pd.DataFrame({"k": k, "n_nodes": [len(g.nodes) for g in egonets.graphs]}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(SIZES_CSV, index=False)
    return df


def main():
    df = pd.read_csv(SIZES_CSV) if SIZES_CSV.exists() else compute_sizes()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.2))
    for ax, k, tag in zip(axes, (1, 2), "ab"):
        sizes = df.loc[df["k"] == k, "n_nodes"]
        bins = np.logspace(np.log10(max(sizes.min(), 2)), np.log10(sizes.max()), 30)
        ax.hist(sizes, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.4)
        median = sizes.median()
        ax.axvline(median, color=ps.INK, lw=0.8, linestyle=(0, (4, 2)))
        ax.text(
            median * 1.25, 0.95, f"median {median:.0f}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=7, color=ps.INK,
        )
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("egonet size (nodes)")
        ps.panel_tag(ax, f"({tag}) $k={k}$")
    axes[0].set_ylabel("egonets")
    fig.subplots_adjust(wspace=0.18)
    ps.save(fig, FIGURES / "egonet_sizes")
    print(f"Figure written to {FIGURES / 'egonet_sizes'}.{{pdf,png}}; sizes cached in {SIZES_CSV}")


if __name__ == "__main__":
    main()
