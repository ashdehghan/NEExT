"""Manuscript print variants (F3-F5 + appendix triptych), from persisted CSVs.

Same content as the working figures, resized to the manuscript text width
(6.3 in) with fonts that print at set size. Reuses the plotting logic of
plot_results.py / plot_field_maps.py with print geometry.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps  # noqa: E402

ps.use_style()
import matplotlib.pyplot as plt  # noqa: E402

from plot_field_maps import LANDSCAPE_OUTPUTS, PANELS, novelty_for  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"

NETWORKS = ["calibration_tails", "fraud_rings", "infiltrators"]
NETWORK_COLOR = {"calibration_tails": "#2a78d6", "fraud_rings": "#eb6834", "infiltrators": "#1baf7a"}
NETWORK_MARKER = {"calibration_tails": "o", "fraud_rings": "s", "infiltrators": "^"}
NETWORK_LABEL = {"calibration_tails": "calibration (tails)", "fraud_rings": "fraud rings", "infiltrators": "infiltrators"}
VARIANTS = ["hop_k1", "hop_k2", "walk", "walk_mv3"]
VARIANT_LABEL = {"hop_k1": "hop-1", "hop_k2": "hop-2", "walk": "walk", "walk_mv3": "w+floor"}


def field_triptych(network: str, out_name: str):
    layout = pd.read_csv(LANDSCAPE_OUTPUTS / f"{network}_k1" / "node_meta.csv")
    meta = pd.read_csv(OUTPUTS / f"{network}__hop_k1" / "node_meta.csv")
    check = layout.merge(meta, on="node_id", suffixes=("_a", "_b"))
    assert (check["degree_a"] == check["degree_b"]).all()
    edges = pd.read_csv(OUTPUTS / f"{network}__hop_k1" / "edges.csv")
    xy = layout.set_index("node_id")

    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.25))
    for ax, (variant, title) in zip(axes, PANELS):
        nov = novelty_for(OUTPUTS / f"{network}__{variant}").set_index("center_node")
        for u, v in edges.itertuples(index=False, name=None):
            if u in xy.index and v in xy.index:
                ax.plot([xy.at[u, "x"], xy.at[v, "x"]], [xy.at[u, "y"], xy.at[v, "y"]],
                        color="#c3c2b7", lw=0.06, alpha=0.18, zorder=1)
        common = nov.index.intersection(xy.index)
        ax.scatter(xy.loc[common, "x"], xy.loc[common, "y"], c=nov.loc[common, "novelty_pct"],
                   cmap="Blues", s=1.8, linewidths=0, zorder=2)
        anoms = layout[layout["is_anomaly"] == 1]
        ax.scatter(anoms["x"], anoms["y"], marker="*", s=26, facecolors="none",
                   edgecolors="#d03b3b", linewidths=0.7, zorder=3)
        ax.set_title(title, fontsize=8, loc="left", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(wspace=0.03)
    ps.save(fig, FIGURES / out_name)
    print("written", out_name)


def c1_metrics():
    df = pd.read_csv(OUTPUTS / "headtohead.csv")
    df = df[df["variant"].isin(VARIANTS)]
    panels = [
        ("smoothness_ratio", "(a)  smoothness", (0.58, 1.18), 1.0),
        ("spike_pctile", "(b)  anomaly spike", (0.5, 1.05), None),
        ("hub_pull_rho", "(c)  hub pull", (-0.65, 0.12), 0.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.15))
    x = np.arange(len(VARIANTS))
    for ax, (metric, title, ylim, refline) in zip(axes, panels):
        if refline is not None:
            ax.axhline(refline, color=ps.AXIS, lw=0.6, zorder=1)
        for network in NETWORKS:
            sub = df[df["network"] == network].set_index("variant").reindex(VARIANTS)
            ax.plot(x, sub[metric], marker=NETWORK_MARKER[network], color=NETWORK_COLOR[network],
                    markersize=3.6, lw=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=6.6)
        ax.set_xlim(-0.35, len(VARIANTS) - 0.65)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=7.6, loc="left", pad=5)
    handles = [
        plt.Line2D([], [], color=NETWORK_COLOR[n], marker=NETWORK_MARKER[n], markersize=3.6, lw=1.0, label=NETWORK_LABEL[n])
        for n in NETWORKS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, columnspacing=1.4, handlelength=1.6, fontsize=7)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.14)
    ps.save(fig, FIGURES / "c1_metrics_print")
    print("written c1_metrics_print")


def c2_parity():
    df = pd.read_csv(OUTPUTS / "containment_comparison.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    lim = (-0.05, 0.42)
    ax.set_aspect("equal")
    ax.plot(lim, lim, color=ps.AXIS, lw=0.8, linestyle=(0, (4, 2)), zorder=1)
    ax.fill_between(lim, lim, lim[1], color="#1baf7a", alpha=0.045, zorder=0)
    ax.text(0.0, 0.4, "walk wins", fontsize=7, color="#1baf7a", ha="left", va="top")
    ax.text(0.41, -0.04, "hop wins", fontsize=7, color="#52514e", ha="right", va="bottom")
    style = {"tail": ("#2a78d6", "o"), "clique": ("#eb6834", "s")}
    for anomaly, sub in df.groupby("anomaly"):
        color, marker = style[anomaly]
        ax.scatter(sub["best_hop_margin"], sub["walk_margin"], s=22, color=color, marker=marker,
                   label=f"{anomaly}", zorder=3, linewidths=0)
    callouts = {
        ("ba", "tail", 0.02): (-0.1, 0.045, "ba/tail π=.02"),
        ("er", "tail", 0.01): (0.04, -0.04, "er/tail π=.01"),
        ("ba", "clique", 0.05): (0.1, -0.026, "ba/clique π=.05"),
    }
    for _, row in df.iterrows():
        key = (row["family"], row["anomaly"], row["prevalence"])
        if key in callouts:
            dx, dy, text = callouts[key]
            ax.annotate(text, xy=(row["best_hop_margin"], row["walk_margin"]),
                        xytext=(row["best_hop_margin"] + dx, row["walk_margin"] + dy),
                        fontsize=6.4, color=ps.INK, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.MUTED, shrinkA=1, shrinkB=2))
    ax.legend(loc="upper right", handlelength=1.0, fontsize=6.8, borderaxespad=0.2)
    ax.set_xlabel("hop-bag margin (best of $k{=}1,2$)", fontsize=7.5)
    ax.set_ylabel("walk-bag margin", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ps.save(fig, FIGURES / "c2_parity_print")
    print("written c2_parity_print")


def main():
    FIGURES.mkdir(exist_ok=True)
    field_triptych("infiltrators", "field_maps_print")
    field_triptych("fraud_rings", "field_maps_rings_print")
    c1_metrics()
    c2_parity()


if __name__ == "__main__":
    main()
