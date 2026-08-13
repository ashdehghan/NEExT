"""Real-data separation figure (house style): lift over paired floor vs
buffer, boxes over the 10 region repeats, dotted reference line at each
series' shared-splits lift. Chameleon shows only the buffers with viable
train pools (the dense graph cannot be separated beyond B=2 — reported,
not hidden).

Run from the repo root.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = {
    "amazon": {
        "csv": "separation_amazon.csv",
        "title": "Amazon Ratings",
        "buffers": [0, 1, 2, 4, 6, 8, 10],
        "series": [
            ("egonet_k4_wass", "fall-vl1-glob", ps.FAMILY_COLOR["structural"], "global $k$=4", 0.103),
            ("egonet_k4_wass", "fall-vl1", ps.FAMILY_COLOR["egonet_hop"], "local $k$=4", 0.093),
            ("node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], "node features", 0.088),
        ],
    },
    "chameleon": {
        "csv": "separation_chameleon.csv",
        "title": "Wikipedia -- Chameleon",
        "buffers": [0, 1, 2],
        "series": [
            ("egonet_k2_wass", "fall-vl1-glob", ps.FAMILY_COLOR["structural"], "global $k$=2", 0.445),
            ("node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], "node features", 0.442),
        ],
    },
}


def _box(ax, values, position, color, width=0.2):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True, zorder=3)
    b["boxes"][0].set_facecolor(color)
    b["boxes"][0].set_alpha(0.4)
    b["boxes"][0].set_edgecolor(color)
    b["medians"][0].set_color(ps.INK)
    for part in ("whiskers", "caps"):
        for line in b[part]:
            line.set_color(color)
            line.set_alpha(0.4)


def main():
    ps.use_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, (key, cfg) in zip(axes, SERIES.items()):
        df = pd.read_csv(SESSION_ROOT / "outputs" / cfg["csv"])
        ok = df[df.status == "ok"]
        xpos = {B: i for i, B in enumerate(cfg["buffers"])}
        n_series = len(cfg["series"])
        offsets = np.linspace(-0.25, 0.25, n_series)
        for (method, tag, color, label, shared_ref), off in zip(cfg["series"], offsets):
            meds = []
            xs = []
            for B in cfg["buffers"]:
                lifts = []
                for r in range(10):
                    split = f"B{B}_r{r}"
                    m = ok[(ok.method == method) & (ok.feature_tag == tag) & (ok.buffer_hops == B) & (ok.split == split)]["f1_macro"]
                    f = ok[(ok.method == f"permuted_{method}") & (ok.feature_tag == tag) & (ok.buffer_hops == B) & (ok.split == split)]["f1_macro"]
                    if len(m) and len(f):
                        lifts.append(float(m.iloc[0]) - float(f.iloc[0]))
                if len(lifts) >= 5:
                    _box(ax, lifts, xpos[B] + off, color)
                    meds.append(float(np.median(lifts)))
                    xs.append(xpos[B] + off)
            ax.plot(xs, meds, color=color, lw=1.3, zorder=2)
            ax.axhline(shared_ref, color=color, lw=1.0, linestyle=(0, (2, 2)), alpha=0.65)
        ax.axhline(0, color=ps.MUTED, lw=0.9)
        ax.set_title(cfg["title"], fontsize=11)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(B) for B in cfg["buffers"]])
        ax.set_xlabel("separation buffer $B$ (hops)")
    axes[0].set_ylabel("macro-F1 lift over paired floor")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["structural"], alpha=0.4,
                      edgecolor=ps.FAMILY_COLOR["structural"], label="global bag"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["egonet_hop"], alpha=0.4,
                      edgecolor=ps.FAMILY_COLOR["egonet_hop"], label="local bag"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["egonet_walk"], alpha=0.4,
                      edgecolor=ps.FAMILY_COLOR["egonet_walk"], label="node features"),
        plt.Line2D([], [], color=ps.INK, lw=1.0, linestyle=(0, (2, 2)), alpha=0.65,
                   label="shared-splits lift (reference)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    w, h = fig.get_size_inches()
    ps.save(fig, SESSION_ROOT / "figures" / "realdata_separation")
    print(f"realdata_separation aspect {w / h:.3f}")


if __name__ == "__main__":
    main()
