"""Addendum: persist each network's edge list for map drawing.

The compute runs persist per-node data but not the edge list; the generators
are fully seeded, so regenerating the graph yields the identical network. We
verify that claim against the saved node_meta (node count + degree sequence)
before writing edges.csv — a mismatch aborts loudly rather than drawing a
wrong graph under a right field.
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment.landscape_synthetic import (  # noqa: E402
    make_calibration_tails,
    make_fraud_rings,
    make_infiltrators,
)

NETWORKS = {
    "calibration_tails": make_calibration_tails,
    "fraud_rings": make_fraud_rings,
    "infiltrators": make_infiltrators,
}
OUTPUTS = SESSION_ROOT / "outputs"


def main():
    for network, generator in NETWORKS.items():
        run_dir = OUTPUTS / f"{network}_k1"
        meta_path = run_dir / "node_meta.csv"
        if not meta_path.exists():
            print(f"[skip] {network}: no node_meta yet")
            continue
        meta = pd.read_csv(meta_path)
        edges_df, nodes_df, _ = generator(n=1500, seed=7)
        G = nx.Graph()
        G.add_nodes_from(nodes_df["node_id"])
        G.add_edges_from(edges_df.itertuples(index=False, name=None))
        assert G.number_of_nodes() == len(meta), f"{network}: node count mismatch"
        degrees = meta.sort_values("node_id")["degree"].to_numpy()
        regen = pd.Series([G.degree(v) for v in sorted(G.nodes())]).to_numpy()
        assert (degrees == regen).all(), f"{network}: degree sequence mismatch"
        for k in (1, 2):
            out = OUTPUTS / f"{network}_k{k}" / "edges.csv"
            if out.parent.exists():
                edges_df.to_csv(out, index=False)
        print(f"[ok] {network}: {G.number_of_edges()} edges persisted (verified)")


if __name__ == "__main__":
    main()
