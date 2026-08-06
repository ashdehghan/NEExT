"""Novelty-field maps per bag construction, on the landscape session's layout.

The score-landscape session (2026-08-05) computed force layouts for the same
seeded networks; the head-to-head runs here persisted embeddings per variant.
Joining the two draws the field the way the original salt-and-pepper maps
did — same layout, same K=25 reference protocol — so the constructions are
visually comparable panel to panel. Node identity across sessions is
verified via degree sequences before drawing.
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
LANDSCAPE_OUTPUTS = SESSION_ROOT.parents[0] / "2026-08-05-score-landscape" / "outputs"
FIGURES = SESSION_ROOT / "figures"
K_REFS = 25
REF_SEED = 123
PANELS = [("hop_k1", "(a)  hop $k{=}1$"), ("hop_k2", "(b)  hop $k{=}2$"), ("walk_mv3", "(c)  walk + floor")]


def novelty_for(run_dir: Path) -> pd.DataFrame:
    emb = pd.read_csv(run_dir / "embeddings.csv")
    bags = pd.read_csv(run_dir / "bag_table.csv")
    df = bags.merge(emb, on="graph_id")
    emb_cols = [c for c in emb.columns if c != "graph_id"]
    X = df[emb_cols].to_numpy(float)
    refs = df.sample(frac=1.0, random_state=REF_SEED).head(K_REFS)
    D = np.linalg.norm(X[:, None, :] - refs[emb_cols].to_numpy(float)[None, :, :], axis=2)
    df["novelty"] = D.mean(axis=1)
    # Linear color scale with a robust clip: rank/percentile coloring forces
    # every panel to the same color histogram, hiding exactly the smooth-vs-
    # salt-and-pepper contrast these maps exist to show.
    lo, hi = df["novelty"].quantile([0.02, 0.98])
    df["novelty_pct"] = ((df["novelty"] - lo) / (hi - lo)).clip(0, 1)
    return df[["center_node", "novelty_pct"]]


def draw(network: str):
    layout = pd.read_csv(LANDSCAPE_OUTPUTS / f"{network}_k1" / "node_meta.csv")
    meta = pd.read_csv(OUTPUTS / f"{network}__hop_k1" / "node_meta.csv")
    check = layout.merge(meta, on="node_id", suffixes=("_a", "_b"))
    assert (check["degree_a"] == check["degree_b"]).all(), f"{network}: graphs differ between sessions"
    edges = pd.read_csv(OUTPUTS / f"{network}__hop_k1" / "edges.csv")
    xy = layout.set_index("node_id")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    for ax, (variant, title) in zip(axes, PANELS):
        nov = novelty_for(OUTPUTS / f"{network}__{variant}").set_index("center_node")
        for u, v in edges.itertuples(index=False, name=None):
            if u in xy.index and v in xy.index:
                ax.plot([xy.at[u, "x"], xy.at[v, "x"]], [xy.at[u, "y"], xy.at[v, "y"]],
                        color="#c3c2b7", lw=0.12, alpha=0.22, zorder=1)
        common = nov.index.intersection(xy.index)
        ax.scatter(xy.loc[common, "x"], xy.loc[common, "y"], c=nov.loc[common, "novelty_pct"],
                   cmap="Blues", s=8, linewidths=0, zorder=2)
        anoms = layout[layout["is_anomaly"] == 1]
        ax.scatter(anoms["x"], anoms["y"], marker="*", s=90, facecolors="none",
                   edgecolors="#d03b3b", linewidths=1.1, zorder=3)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(wspace=0.04)
    ps.save(fig, FIGURES / f"field_maps_{network}")
    print(f"[done] {network}")


def main():
    FIGURES.mkdir(exist_ok=True)
    for network in ("infiltrators", "fraud_rings"):
        draw(network)


if __name__ == "__main__":
    main()
