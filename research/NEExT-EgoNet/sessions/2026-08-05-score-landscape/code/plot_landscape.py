"""Landscape maps from the saved CSVs — exploratory figures we look at.

Per run:
  map_<run>.png     novelty + affinity fields (K=25) on the force layout,
                    edges faint underneath, planted anomalies starred
  ksweep_<run>.png  affinity field at K in {5,10,25,50} (nested samples)
  slope_<run>.png   field value vs hop distance to nearest anomaly
  smooth_<run>.png  |field difference| across edges vs random node pairs
  map_pooled_<run>.png  affinity field from pooled features (variant)

Fields:
  novelty(v)  = mean distance from v's bag embedding to the K reference bags
  affinity(v) = mean dist to reference NEGATIVES - mean dist to POSITIVES
                (higher = more anomaly-like; undefined if sample has no positive)
Colors show within-map percentiles (rank-normalized) — glow is relative.
"""

import json
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
K_SWEEP = [5, 10, 25, 50]
K_DEFAULT = 25
REF_SEED = 123


def load_run(run_dir: Path):
    emb = pd.read_csv(run_dir / "embeddings.csv")
    bags = pd.read_csv(run_dir / "bag_table.csv")
    meta = pd.read_csv(run_dir / "node_meta.csv")
    edges = pd.read_csv(run_dir / "edges.csv")
    config = json.loads((run_dir / "config.json").read_text())
    df = bags.merge(emb, on="graph_id").merge(meta, left_on="center_node", right_on="node_id")
    emb_cols = [c for c in emb.columns if c != "graph_id"]
    return df, emb_cols, edges, config


def reference_prefix(df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Nested reference samples: prefixes of one shuffled draw (stability read)."""
    shuffled = df.sample(frac=1.0, random_state=REF_SEED).reset_index(drop=True)
    return shuffled.head(size)


def fields(df: pd.DataFrame, emb_cols, K: int):
    X = df[emb_cols].to_numpy(float)
    refs = reference_prefix(df, K)
    R = refs[emb_cols].to_numpy(float)
    D = np.linalg.norm(X[:, None, :] - R[None, :, :], axis=2)  # nodes x K
    novelty = D.mean(axis=1)
    pos = refs["y_contains"].to_numpy() == 1
    n_pos = int(pos.sum())
    affinity = D[:, ~pos].mean(axis=1) - D[:, pos].mean(axis=1) if 0 < n_pos < K else None
    return novelty, affinity, n_pos


def rank01(v: np.ndarray) -> np.ndarray:
    order = v.argsort().argsort()
    return order / (len(v) - 1)


def draw_map(ax, df, edges, values, title, config):
    xy = df.set_index("center_node")
    for u, v in edges.itertuples(index=False, name=None):
        if u in xy.index and v in xy.index:
            ax.plot(
                [xy.at[u, "x"], xy.at[v, "x"]], [xy.at[u, "y"], xy.at[v, "y"]],
                color="#c3c2b7", lw=0.15, alpha=0.25, zorder=1,
            )
    if values is None:
        ax.text(0.5, 0.5, "no positives in\nreference sample", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.scatter(df["x"], df["y"], c=rank01(values), cmap="Blues", s=7, linewidths=0, zorder=2)
    anoms = df[df["is_anomaly"] == 1]
    if "ring_density" in anoms.columns and anoms["ring_density"].max() > 0:
        for rho, sub in anoms.groupby("ring_density"):
            ax.scatter(
                sub["x"], sub["y"], marker="*", s=110, facecolors="none",
                edgecolors="#d03b3b", linewidths=1.2, zorder=3,
            )
            cx, cy = sub["x"].mean(), sub["y"].mean()
            ax.text(cx, cy, f"ρ={rho:g}", fontsize=7, color="#d03b3b", ha="center", va="bottom", zorder=4)
    else:
        ax.scatter(
            anoms["x"], anoms["y"], marker="*", s=110, facecolors="none",
            edgecolors="#d03b3b", linewidths=1.2, zorder=3,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def plot_run(run_dir: Path):
    run_id = run_dir.name
    df, emb_cols, edges, config = load_run(run_dir)

    novelty, affinity, n_pos = fields(df, emb_cols, K_DEFAULT)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    draw_map(axes[0], df, edges, novelty, f"novelty (K={K_DEFAULT})", config)
    draw_map(axes[1], df, edges, affinity, f"affinity (K={K_DEFAULT}, {n_pos} positive refs)", config)
    fig.suptitle(f"{run_id} — pos_rate={config['bag_positive_rate']:.2f}, median bag={config['bag_nodes_median']:.0f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"map_{run_id}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, K in zip(axes, K_SWEEP):
        _, aff, npos = fields(df, emb_cols, K)
        draw_map(ax, df, edges, aff, f"K={K} ({npos} pos)", config)
    fig.suptitle(f"{run_id} — affinity field vs reference size", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"ksweep_{run_id}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    dist = df["dist_to_anomaly"].clip(upper=6).astype(int)
    for ax, (name, vals) in zip(axes, [("novelty", novelty), ("affinity", affinity)]):
        if vals is None:
            continue
        data = [vals[dist == d] for d in range(0, 7)]
        ax.boxplot(data, positions=range(0, 7), widths=0.6, showfliers=False)
        ax.set_xlabel("hop distance to nearest anomaly (6 = 6+)")
        ax.set_ylabel(name)
    fig.suptitle(f"{run_id} — does the surface slope toward anomalies?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"slope_{run_id}.png", dpi=180)
    plt.close(fig)

    if affinity is not None:
        rng = np.random.default_rng(0)
        idx = df.set_index("center_node").index
        pos_of = {c: i for i, c in enumerate(idx)}
        pairs = [(pos_of[u], pos_of[v]) for u, v in edges.itertuples(index=False, name=None) if u in pos_of and v in pos_of]
        edge_diff = np.array([abs(affinity[i] - affinity[j]) for i, j in pairs])
        ri = rng.integers(0, len(affinity), size=(len(pairs), 2))
        rand_diff = np.abs(affinity[ri[:, 0]] - affinity[ri[:, 1]])
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bins = np.linspace(0, max(edge_diff.max(), rand_diff.max()), 50)
        ax.hist(rand_diff, bins=bins, alpha=0.55, label="random pairs", color="#c3c2b7")
        ax.hist(edge_diff, bins=bins, alpha=0.7, label="graph edges", color="#2a78d6")
        ax.set_xlabel("|affinity difference|")
        ax.set_ylabel("count")
        ax.set_title(f"{run_id} — smoothness: edges vs random pairs", fontsize=9)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"smooth_{run_id}.png", dpi=180)
        plt.close(fig)

    pooled = pd.read_csv(run_dir / "pooled.csv")
    dfp = df[["graph_id", "center_node", "x", "y", "is_anomaly", "y_contains", "dist_to_anomaly"]].merge(pooled, on="graph_id")
    if "ring_density" in df.columns:
        dfp["ring_density"] = df["ring_density"].values
    pooled_cols = [c for c in pooled.columns if c != "graph_id"]
    vals = dfp[pooled_cols].to_numpy(float)
    vals = (vals - vals.mean(axis=0)) / (vals.std(axis=0) + 1e-9)
    dfp_std = dfp.copy()
    _, aff_p, npos_p = fields(dfp_std.assign(**{c: vals[:, i] for i, c in enumerate(pooled_cols)}), pooled_cols, K_DEFAULT)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    draw_map(ax, dfp, edges, aff_p, f"pooled affinity (K={K_DEFAULT}, {npos_p} pos)", config)
    fig.tight_layout()
    fig.savefig(FIGURES / f"map_pooled_{run_id}.png", dpi=200)
    plt.close(fig)

    print(f"[plotted] {run_id}")


def main():
    FIGURES.mkdir(exist_ok=True)
    for run_dir in sorted(OUTPUTS.iterdir()):
        if (run_dir / "embeddings.csv").exists() and (run_dir / "edges.csv").exists():
            plot_run(run_dir)


if __name__ == "__main__":
    main()
