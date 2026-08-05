"""Score-landscape study: embed EVERY node's egonet and persist the field data.

Per run (network x k): egonet around every node -> structural features ->
one Wasserstein embedding over all bags (single fit, so the space is
consistent by construction) + pooled features. Persists everything the maps
need; plotting is pure post-processing on these CSVs.

Outputs per run (outputs/<run_id>/):
  embeddings.csv   graph_id (bag id) + emb_* columns
  pooled.csv       graph_id + pooled feature columns
  bag_table.csv    graph_id, center_node, labels, sizes
  node_meta.csv    node_id, is_anomaly, plant metadata, degree,
                   hop distance to nearest anomaly, layout x/y
  config.json      generator + pipeline params, git SHA, NEExT version

Resumable: runs with an existing embeddings.csv are skipped.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import build_bags, git_sha, neext_version, pooled_features, wasserstein_embedding  # noqa: E402
from lib.containment.landscape_synthetic import (  # noqa: E402
    make_calibration_tails,
    make_fraud_rings,
    make_infiltrators,
)

from NEExT import NEExT  # noqa: E402

NETWORKS = {
    "calibration_tails": make_calibration_tails,
    "fraud_rings": make_fraud_rings,
    "infiltrators": make_infiltrators,
}
K_HOPS = [1, 2]
N = 1500
SEED = 7
OUTPUTS = SESSION_ROOT / "outputs"


def node_meta(collection, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Degree, hop-distance-to-nearest-anomaly (multi-source BFS), FR layout."""
    graph = collection.graphs[0]
    G = graph.G  # igraph
    anomalies = [v for v in graph.nodes if graph.node_attributes[v]["is_anomaly"] == 1]
    dist_rows = np.array(G.distances(source=anomalies))  # anomalies x nodes
    nearest = dist_rows.min(axis=0)
    # Seeded start grid so re-running compute reproduces identical coords
    # (audit hardening; existing runs keep their persisted layout).
    rng = np.random.default_rng(0)
    seed_coords = rng.random((len(graph.nodes), 2)).tolist()
    layout = G.layout_fruchterman_reingold(niter=500, seed=seed_coords)
    coords = np.array(layout.coords)
    meta = pd.DataFrame(
        {
            "node_id": graph.nodes,
            "degree": G.degree(),
            "dist_to_anomaly": nearest,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    return meta.merge(nodes_df, on="node_id", how="left")


def run_one(nxt, network: str, k: int) -> None:
    run_id = f"{network}_k{k}"
    run_dir = OUTPUTS / run_id
    if (run_dir / "embeddings.csv").exists():
        print(f"[skip] {run_id}", flush=True)
        return
    t0 = time.time()

    edges_df, nodes_df, gen_config = NETWORKS[network](n=N, seed=SEED)
    structural_nodes = nodes_df[["node_id", "is_anomaly"]]
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=structural_nodes, graph_type="igraph", filter_largest_component=True
    )
    bags = build_bags(nxt, collection, label_column="is_anomaly", k_hop=k, n_centers=None)
    features = nxt.compute_node_features(
        bags.egonets, feature_list=["all"], feature_vector_length=3, show_progress=False, n_jobs=-1
    )
    embeddings = wasserstein_embedding(nxt, bags.egonets, features, dimension=16, seed=42)
    pooled = pooled_features(features, stats=("mean", "max", "p90"))

    run_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(run_dir / "embeddings.csv", index=False)
    pooled.to_csv(run_dir / "pooled.csv", index=False)
    bags.table.to_csv(run_dir / "bag_table.csv", index=False)
    node_meta(collection, nodes_df).to_csv(run_dir / "node_meta.csv", index=False)
    config = {
        **gen_config,
        "k_hop": k,
        "run_id": run_id,
        "n_bags": len(bags.table),
        "bag_positive_rate": float(bags.table["y_contains"].mean()),
        "bag_nodes_median": float(bags.table["n_nodes"].median()),
        "git_sha": git_sha(RESEARCH_ROOT.parents[0]),
        "neext_version": neext_version(),
        "seconds_total": round(time.time() - t0, 1),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(
        f"[done] {run_id}: {config['n_bags']} bags, pos_rate={config['bag_positive_rate']:.3f}, "
        f"med_size={config['bag_nodes_median']:.0f} ({config['seconds_total']:.0f}s)",
        flush=True,
    )


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    nxt = NEExT(log_level="ERROR")
    for k in K_HOPS:
        for network in NETWORKS:
            run_one(nxt, network, k)
    print("All runs complete.", flush=True)


if __name__ == "__main__":
    main()
