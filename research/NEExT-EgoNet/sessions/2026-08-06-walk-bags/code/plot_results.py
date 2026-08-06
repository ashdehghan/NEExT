"""Walk-bags figures from the saved CSVs (headtohead.csv, containment_comparison.csv).

Layout rules learned the hard way: legends live OUTSIDE the axes (shared
figure-level row), tick labels are short single-liners, reference lines are
unlabeled hairlines (the caption/card explains them), and every annotation is
placed away from data and edges. The parity plot is drawn square so the
diagonal is a true 45 degrees.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps  # noqa: E402

ps.use_style()
import matplotlib.pyplot as plt  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"

NETWORKS = ["calibration_tails", "fraud_rings", "infiltrators"]
NETWORK_COLOR = {"calibration_tails": "#2a78d6", "fraud_rings": "#eb6834", "infiltrators": "#1baf7a"}
NETWORK_MARKER = {"calibration_tails": "o", "fraud_rings": "s", "infiltrators": "^"}
NETWORK_LABEL = {"calibration_tails": "calibration (tails)", "fraud_rings": "fraud rings", "infiltrators": "infiltrators"}
VARIANTS = ["hop_k1", "hop_k2", "walk", "walk_mv3"]
VARIANT_LABEL = {"hop_k1": "hop-1", "hop_k2": "hop-2", "walk": "walk", "walk_mv3": "walk+floor"}


def headtohead():
    df = pd.read_csv(OUTPUTS / "headtohead.csv")
    df = df[df["variant"].isin(VARIANTS)]
    panels = [
        ("smoothness_ratio", "(a)  smoothness  —  lower is smoother", (0.58, 1.18), 1.0),
        ("spike_pctile", "(b)  anomaly spike  —  higher is sharper", (0.5, 1.05), None),
        ("hub_pull_rho", "(c)  hub pull  —  zero is degree-neutral", (-0.65, 0.12), 0.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    x = np.arange(len(VARIANTS))
    for ax, (metric, title, ylim, refline) in zip(axes, panels):
        if refline is not None:
            ax.axhline(refline, color=ps.AXIS, lw=0.7, zorder=1)
        for network in NETWORKS:
            sub = df[df["network"] == network].set_index("variant").reindex(VARIANTS)
            ax.plot(
                x, sub[metric], marker=NETWORK_MARKER[network], color=NETWORK_COLOR[network],
                markersize=5, lw=1.2, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=8)
        ax.set_xlim(-0.35, len(VARIANTS) - 0.65)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8.5, loc="left", pad=8)
    handles = [
        plt.Line2D([], [], color=NETWORK_COLOR[n], marker=NETWORK_MARKER[n], markersize=5, lw=1.2, label=NETWORK_LABEL[n])
        for n in NETWORKS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, columnspacing=1.6, handlelength=1.8)
    fig.subplots_adjust(wspace=0.3, top=0.86, bottom=0.12)
    ps.save(fig, FIGURES / "headtohead")


def containment():
    df = pd.read_csv(OUTPUTS / "containment_comparison.csv")
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    lim = (-0.05, 0.42)
    ax.set_aspect("equal")
    ax.plot(lim, lim, color=ps.AXIS, lw=0.9, linestyle=(0, (4, 2)), zorder=1)
    ax.fill_between(lim, lim, lim[1], color="#1baf7a", alpha=0.045, zorder=0)

    # Zone labels pinned to empty corners, clear of data and edges.
    ax.text(0.02, 0.395, "walk wins", fontsize=8, color="#1baf7a", ha="left", va="top")
    ax.text(0.405, -0.035, "hop wins", fontsize=8, color="#52514e", ha="right", va="bottom")

    style = {"tail": ("#2a78d6", "o"), "clique": ("#eb6834", "s")}
    for anomaly, sub in df.groupby("anomaly"):
        color, marker = style[anomaly]
        ax.scatter(sub["best_hop_margin"], sub["walk_margin"], s=46, color=color, marker=marker, zorder=3, linewidths=0)

    # Only the three big divergers get labels; hand-placed with leader lines,
    # each pushed toward open space (far-right point labels leftward).
    callouts = {
        ("ba", "tail", 0.02): (-0.11, 0.045, "ba/tail π=.02"),
        ("er", "tail", 0.01): (0.035, -0.042, "er/tail π=.01"),
        ("ba", "clique", 0.05): (0.09, -0.028, "ba/clique π=.05"),
    }
    for _, row in df.iterrows():
        key = (row["family"], row["anomaly"], row["prevalence"])
        if key in callouts:
            dx, dy, text = callouts[key]
            ax.annotate(
                text,
                xy=(row["best_hop_margin"], row["walk_margin"]),
                xytext=(row["best_hop_margin"] + dx, row["walk_margin"] + dy),
                fontsize=7.5, color=ps.INK, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.MUTED, shrinkA=1, shrinkB=3),
            )

    handles = [
        plt.Line2D([], [], color=style[a][0], marker=style[a][1], markersize=6, lw=0, label=f"{a} anomalies")
        for a in ("tail", "clique")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.04), ncol=2, columnspacing=1.6, handlelength=1.2)
    ax.set_xlabel("hop-bag margin over size-only (best of k=1,2 — frozen phase 1)")
    ax.set_ylabel("walk-bag margin over size-only")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    fig.subplots_adjust(top=0.9)
    ps.save(fig, FIGURES / "containment_comparison")


def main():
    FIGURES.mkdir(exist_ok=True)
    headtohead()
    containment()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
