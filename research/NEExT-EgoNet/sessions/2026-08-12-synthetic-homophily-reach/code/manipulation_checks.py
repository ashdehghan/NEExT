"""Manipulation checks for the planted homophily-reach grid (P1).

Per cell: (a) marginal degree eta^2 (faintness of the beta coupling),
(b) adjusted edge homophily, (c) the distance-homophily excess curve
P(same label | hop distance d) - sum(p_c^2), with reach measured as the
FIRST ZERO CROSSING (the single-frequency field oscillates past that
point — registered in notes/predictions.md addendum).

Artifacts: outputs/manipulation_checks.csv (summary) and
outputs/distance_homophily_curves.csv (per-cell curves) — the reach
numbers feed the money figure.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SESSION_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "research" / "NEExT-EgoNet"))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import igraph as ig
import numpy as np
import pandas as pd

from datasets import COUPLINGS, WAVELENGTHS, load_single_graph_dataset

CELLS = [f"SYNTH_HR_{L}_{B}" for L in WAVELENGTHS for B in COUPLINGS]
MAX_D = 10
N_SOURCES = 800
SEED = 13


def eta2(vals: pd.Series, labels: pd.Series) -> float:
    overall = vals.mean()
    ss_between = sum(len(g) * (g.mean() - overall) ** 2 for _, g in vals.groupby(labels))
    return float(ss_between / ((vals - overall) ** 2).sum())


def adjusted_homophily(g: ig.Graph, y: np.ndarray) -> float:
    same = np.mean([y[e.source] == y[e.target] for e in g.es])
    p2 = float(np.sum((np.bincount(y) / len(y)) ** 2))
    return (same - p2) / (1 - p2)


def distance_curve(g: ig.Graph, y: np.ndarray, rng: np.random.RandomState) -> pd.DataFrame:
    baseline = float(np.sum((np.bincount(y) / len(y)) ** 2))
    sources = rng.choice(g.vcount(), size=min(N_SOURCES, g.vcount()), replace=False)
    same = np.zeros(MAX_D + 1)
    total = np.zeros(MAX_D + 1)
    dists = np.array(g.shortest_paths(source=list(sources)))
    for row, s in zip(dists, sources):
        for d in range(1, MAX_D + 1):
            idx = np.nonzero(row == d)[0]
            total[d] += len(idx)
            same[d] += int((y[idx] == y[s]).sum())
    rows = [
        {"d": d, "p_same": same[d] / total[d], "excess": same[d] / total[d] - baseline, "n_pairs": int(total[d])}
        for d in range(1, MAX_D + 1)
        if total[d] > 0
    ]
    return pd.DataFrame(rows)


def first_zero_crossing(curve: pd.DataFrame) -> float:
    """Reach = where the excess first crosses zero (linear interp between hops)."""
    prev_d, prev_e = 0, None
    for d, e in zip(curve["d"], curve["excess"]):
        if prev_e is not None and prev_e > 0 >= e:
            return prev_d + prev_e / (prev_e - e)
        prev_d, prev_e = d, e
    return float(curve["d"].iloc[-1])  # never crossed in range


def main():
    # Optional argv: explicit cell ids (e.g. replicate ids SYNTH_HR_L2_B1_S101).
    # Results are merged into the output CSVs, replacing rows for the same id.
    cells = sys.argv[1:] or CELLS
    outputs = SESSION_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    rng = np.random.RandomState(SEED)
    summary, curves = [], []
    for cell in cells:
        edges_df, nodes_df = load_single_graph_dataset(cell, label_column="field_class")
        g = ig.Graph(edges=list(zip(edges_df.src_node_id, edges_df.dest_node_id)), n=len(nodes_df))
        y = nodes_df["field_class"].to_numpy()
        deg = pd.Series(g.degree(), index=nodes_df.node_id)
        curve = distance_curve(g, y, rng)
        curve.insert(0, "dataset", cell)
        curves.append(curve)
        summary.append(
            {
                "dataset": cell,
                "wavelength": WAVELENGTHS[cell.split("_")[2]],
                "coupling": COUPLINGS[cell.split("_")[3]],
                "deg_eta2": round(eta2(np.log1p(deg), pd.Series(y, index=deg.index)), 4),
                "h_adj": round(adjusted_homophily(g, y), 3),
                "reach_zero_crossing": round(first_zero_crossing(curve), 2),
            }
        )
        print(summary[-1])
    summary_df, curves_df = pd.DataFrame(summary), pd.concat(curves)
    for path, new in ((outputs / "manipulation_checks.csv", summary_df),
                      (outputs / "distance_homophily_curves.csv", curves_df)):
        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old[~old.dataset.isin(new.dataset)], new], ignore_index=True)
        new.to_csv(path, index=False)
    print("written:", outputs / "manipulation_checks.csv")


if __name__ == "__main__":
    main()
