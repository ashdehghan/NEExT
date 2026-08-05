"""Mechanism test: does bag overlap explain field smoothness?

Hypothesis (brainstorm 2026-08-05): the salt-and-pepper field at k=1 is
small-overlap + small-sample variance — adjacent nodes' bags are mostly
disjoint sets in sparse graphs, so the field jumps across edges. Prediction:
per-edge |Δnovelty| decreases with the Jaccard overlap of the two endpoint
bags, and k=2 edges live at much higher overlap than k=1 (explaining the
measured smoothness flip).

Bag memberships are reconstructed from the persisted edges.csv (k-hop BFS on
the identical graph; sizes verified against bag_table.csv). Edges incident to
a planted anomaly are analyzed separately — their jumps are signal (the
spike), not roughness.

Outputs: figures/smoothness_mechanism.{pdf,png}, outputs/smoothness_stats.csv
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps  # noqa: E402

ps.use_style()
import matplotlib.pyplot as plt  # noqa: E402

from plot_landscape import K_DEFAULT, fields, load_run  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"
NETWORKS = ["calibration_tails", "fraud_rings", "infiltrators"]


def bag_members(edges: pd.DataFrame, nodes: list, k: int) -> dict:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.itertuples(index=False, name=None))
    return {v: set(nx.single_source_shortest_path_length(G, v, cutoff=k)) for v in nodes}


def analyze_run(run_id: str, k: int) -> dict:
    df, emb_cols, edges, config = load_run(OUTPUTS / run_id)
    novelty, _, _ = fields(df, emb_cols, K_DEFAULT)
    df = df.assign(novelty=novelty).set_index("center_node")

    members = bag_members(edges, list(df.index), k)
    sizes = {v: len(m) for v, m in members.items()}
    mismatch = sum(1 for v in df.index if sizes[v] != df.at[v, "n_nodes"])
    assert mismatch == 0, f"{run_id}: {mismatch} reconstructed bag sizes disagree with bag_table"

    anomalies = set(df.index[df["is_anomaly"] == 1])
    rows = []
    for u, v in edges.itertuples(index=False, name=None):
        if u not in members or v not in members:
            continue
        inter = len(members[u] & members[v])
        union = len(members[u] | members[v])
        rows.append(
            {
                "jaccard": inter / union,
                "dnov": abs(df.at[u, "novelty"] - df.at[v, "novelty"]),
                "touches_anomaly": u in anomalies or v in anomalies,
            }
        )
    e = pd.DataFrame(rows)
    bg = e[~e.touches_anomaly]
    rho_bg = spearmanr(bg.jaccard, bg.dnov)
    rho_all = spearmanr(e.jaccard, e.dnov)
    return {
        "run_id": run_id,
        "k": k,
        "n_edges": len(e),
        "jaccard_median": round(float(e.jaccard.median()), 3),
        "rho_background": round(float(rho_bg.statistic), 3),
        "rho_background_p": float(rho_bg.pvalue),
        "rho_all": round(float(rho_all.statistic), 3),
        "_edges": e,
    }


def main():
    stats, per_run_edges = [], {}
    for network in NETWORKS:
        for k in (1, 2):
            run_id = f"{network}_k{k}"
            res = analyze_run(run_id, k)
            per_run_edges[run_id] = res.pop("_edges")
            stats.append(res)
            print(
                f"[done] {run_id}: median J={res['jaccard_median']}, "
                f"rho_bg={res['rho_background']} (p={res['rho_background_p']:.1e})",
                flush=True,
            )
    pd.DataFrame(stats).to_csv(OUTPUTS / "smoothness_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.7))
    # (a) |Δnovelty| vs Jaccard in equal-count decile bins per run (fixed-width
    # bins are noise: high-overlap bins are nearly empty at k=1). Curves are
    # per-run, normalized by that run's median jump.
    for k, color in ((1, "#2a78d6"), (2, "#eb6834")):
        for i, network in enumerate(NETWORKS):
            e = per_run_edges[f"{network}_k{k}"].query("~touches_anomaly").copy()
            e["nov_norm"] = e["dnov"] / e["dnov"].median()
            e["bin"] = pd.qcut(e["jaccard"], 10, labels=False, duplicates="drop")
            g = e.groupby("bin").agg(x=("jaccard", "median"), y=("nov_norm", "median"))
            axes[0].plot(
                g.x, g.y, marker="o", color=color, markersize=2.5, lw=0.9, alpha=0.85,
                label=f"$k={k}$" if i == 0 else None,
            )
    axes[0].axhline(1.0, color=ps.AXIS, lw=0.6, zorder=0)
    axes[0].set_xlabel("bag overlap across the edge (Jaccard, decile bins)")
    axes[0].set_ylabel("median |Δnovelty| (per-run norm.)")
    axes[0].legend()
    ps.panel_tag(axes[0], "(a)")

    # (b) where edges live: Jaccard distributions at k=1 vs k=2
    for k, color in ((1, "#2a78d6"), (2, "#eb6834")):
        pooled = pd.concat([per_run_edges[f"{n}_k{k}"] for n in NETWORKS], ignore_index=True)
        axes[1].hist(pooled.jaccard, bins=30, density=True, alpha=0.55, color=color, label=f"$k={k}$")
    axes[1].set_xlabel("bag overlap across the edge (Jaccard)")
    axes[1].set_ylabel("edge density")
    axes[1].legend()
    ps.panel_tag(axes[1], "(b)")
    ps.save(fig, FIGURES / "smoothness_mechanism")
    print(f"Figure written to {FIGURES / 'smoothness_mechanism'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
