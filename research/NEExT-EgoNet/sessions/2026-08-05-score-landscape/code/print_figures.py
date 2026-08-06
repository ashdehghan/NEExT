"""Manuscript print figure F1: the bed of nails, from persisted CSVs.

Panel (a): the infiltrators novelty field (hop k=1) on the force layout —
the salt-and-pepper landscape, anomalies starred. Panel (b): novelty vs
true hop distance to the nearest anomaly, median with IQR band, for k=1 and
k=2 fields — the spike at d=0 and the flat terrain beyond, at both radii.
Sized to the manuscript text width; colors from the validated palette.
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
K_REFS = 25
REF_SEED = 123


def novelty(run_id: str) -> pd.DataFrame:
    emb = pd.read_csv(OUTPUTS / run_id / "embeddings.csv")
    bags = pd.read_csv(OUTPUTS / run_id / "bag_table.csv")
    meta = pd.read_csv(OUTPUTS / run_id / "node_meta.csv")
    df = bags.merge(emb, on="graph_id").merge(meta, left_on="center_node", right_on="node_id")
    emb_cols = [c for c in emb.columns if c != "graph_id"]
    X = df[emb_cols].to_numpy(float)
    refs = df.sample(frac=1.0, random_state=REF_SEED).head(K_REFS)
    D = np.linalg.norm(X[:, None, :] - refs[emb_cols].to_numpy(float)[None, :, :], axis=2)
    df["novelty"] = D.mean(axis=1)
    return df


def main():
    FIGURES.mkdir(exist_ok=True)
    k1 = novelty("infiltrators_k1")
    k2 = novelty("infiltrators_k2")
    edges = pd.read_csv(OUTPUTS / "infiltrators_k1" / "edges.csv")

    fig = plt.figure(figsize=(ps.FULL_W, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.18)

    # (a) the field, linear robust-clipped color
    ax = fig.add_subplot(gs[0])
    lo, hi = k1["novelty"].quantile([0.02, 0.98])
    color = ((k1["novelty"] - lo) / (hi - lo)).clip(0, 1)
    xy = k1.set_index("center_node")
    for u, v in edges.itertuples(index=False, name=None):
        if u in xy.index and v in xy.index:
            ax.plot([xy.at[u, "x"], xy.at[v, "x"]], [xy.at[u, "y"], xy.at[v, "y"]],
                    color="#c3c2b7", lw=0.08, alpha=0.2, zorder=1)
    ax.scatter(k1["x"], k1["y"], c=color, cmap="Blues", s=2.4, linewidths=0, zorder=2)
    anoms = k1[k1["is_anomaly"] == 1]
    ax.scatter(anoms["x"], anoms["y"], marker="*", s=34, facecolors="none",
               edgecolors="#d03b3b", linewidths=0.8, zorder=3)
    ps.panel_tag(ax, "(a)")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    # (b) novelty vs hop distance: median + IQR band per radius
    ax = fig.add_subplot(gs[1])
    for df, label, color_hex, marker in ((k1, "$k=1$ field", "#2a78d6", "o"), (k2, "$k=2$ field", "#eb6834", "s")):
        d = df["dist_to_anomaly"].clip(upper=5).astype(int)
        grouped = df.groupby(d)["novelty"]
        med, q1, q3 = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
        ax.plot(med.index, med.values, marker=marker, color=color_hex, markersize=3.8, lw=1.2, label=label)
        ax.fill_between(med.index, q1.values, q3.values, color=color_hex, alpha=0.13, linewidth=0)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5+"])
    ax.set_xlabel("hop distance to nearest infiltrator")
    ax.set_ylabel("novelty")
    ax.legend(loc="upper right", handlelength=1.5)
    ps.panel_tag(ax, "(b)")
    ps.save(fig, FIGURES / "bed_of_nails")
    print("Figure written:", FIGURES / "bed_of_nails")


if __name__ == "__main__":
    main()
