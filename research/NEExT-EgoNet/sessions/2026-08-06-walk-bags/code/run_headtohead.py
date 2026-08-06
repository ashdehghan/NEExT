"""C1: bag-construction head-to-head on the landscape harness.

For each of the three landscape networks and each bag variant, embed EVERY
node's bag in one consistent space and measure the three properties the walk
construction is supposed to buy:

  smoothness   |Δnovelty| across edges vs random pairs (ratio < 1 = smooth)
  spike        background percentile of the anomaly-median novelty (d=0)
  hub pull     Spearman corr(novelty, degree) — the hub-attraction confound

Variants: hop k=1, hop k=2 (the phase-1/landscape baselines, recomputed here
so every number shares one code path), walk defaults (0.15 restart), walk
low-restart (0.05), walk unweighted (membership only — the weights ablation).

Outputs per (network, variant): outputs/<run_id>/ with embeddings.csv,
bag_table.csv, node_meta.csv, edges.csv, config.json; summary metrics in
outputs/headtohead.csv. Resumable per run dir.
"""

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import build_bags, git_sha, neext_version, wasserstein_embedding  # noqa: E402
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
VARIANTS = {
    "hop_k1": {"method": "k_hop", "k_hop": 1},
    "hop_k2": {"method": "k_hop", "k_hop": 2},
    "walk": {"method": "random_walk", "restart_prob": 0.15},
    "walk_far": {"method": "random_walk", "restart_prob": 0.05},
    "walk_unweighted": {"method": "random_walk", "restart_prob": 0.15, "weight_by_visits": False},
    "walk_mv3": {"method": "random_walk", "restart_prob": 0.15, "min_visits": 3},
}
N = 1500
SEED = 7
K_REFS = 25
REF_SEED = 123
OUTPUTS = SESSION_ROOT / "outputs"


def node_meta(collection, nodes_df):
    graph = collection.graphs[0]
    G = graph.G
    anomalies = [v for v in graph.nodes if graph.node_attributes[v]["is_anomaly"] == 1]
    nearest = np.array(G.distances(source=anomalies)).min(axis=0)
    return pd.DataFrame({"node_id": graph.nodes, "degree": G.degree(), "dist_to_anomaly": nearest}).merge(
        nodes_df, on="node_id", how="left"
    )


def run_one(nxt, network, variant):
    run_id = f"{network}__{variant}"
    run_dir = OUTPUTS / run_id
    if (run_dir / "embeddings.csv").exists():
        print(f"[skip] {run_id}", flush=True)
        return
    t0 = time.time()

    edges_df, nodes_df, gen_config = NETWORKS[network](n=N, seed=SEED)
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df[["node_id", "is_anomaly"]], graph_type="igraph", filter_largest_component=True
    )
    bags = build_bags(nxt, collection, label_column="is_anomaly", n_centers=None, **VARIANTS[variant])
    features = nxt.compute_node_features(
        bags.egonets, feature_list=["all"], feature_vector_length=3, show_progress=False, n_jobs=-1
    )
    embeddings = wasserstein_embedding(nxt, bags.egonets, features, dimension=16, seed=42)

    run_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(run_dir / "embeddings.csv", index=False)
    bags.table.to_csv(run_dir / "bag_table.csv", index=False)
    node_meta(collection, nodes_df).to_csv(run_dir / "node_meta.csv", index=False)
    edges_df.to_csv(run_dir / "edges.csv", index=False)
    config = {
        **gen_config,
        "variant": variant,
        **VARIANTS[variant],
        "run_id": run_id,
        "bag_nodes_median": float(bags.table["n_nodes"].median()),
        "bag_positive_rate": float(bags.table["y_contains"].mean()),
        "git_sha": git_sha(RESEARCH_ROOT.parents[0]),
        "neext_version": neext_version(),
        "seconds_total": round(time.time() - t0, 1),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"[done] {run_id}: med_size={config['bag_nodes_median']:.0f} ({config['seconds_total']:.0f}s)", flush=True)


def analyze():
    rows = []
    rng = np.random.default_rng(0)
    for run_dir in sorted(OUTPUTS.iterdir()):
        if not (run_dir / "embeddings.csv").exists():
            continue
        emb = pd.read_csv(run_dir / "embeddings.csv")
        bags = pd.read_csv(run_dir / "bag_table.csv")
        meta = pd.read_csv(run_dir / "node_meta.csv")
        edges = pd.read_csv(run_dir / "edges.csv")
        config = json.loads((run_dir / "config.json").read_text())

        df = bags.merge(emb, on="graph_id").merge(meta, left_on="center_node", right_on="node_id")
        emb_cols = [c for c in emb.columns if c != "graph_id"]
        X = df[emb_cols].to_numpy(float)
        refs = df.sample(frac=1.0, random_state=REF_SEED).head(K_REFS)
        D = np.linalg.norm(X[:, None, :] - refs[emb_cols].to_numpy(float)[None, :, :], axis=2)
        df["novelty"] = D.mean(axis=1)

        pos_of = {c: i for i, c in enumerate(df["center_node"])}
        nov = df["novelty"].to_numpy()
        pairs = [(pos_of[u], pos_of[v]) for u, v in edges.itertuples(index=False, name=None) if u in pos_of and v in pos_of]
        anoms = set(df.loc[df["is_anomaly"] == 1, "center_node"])
        bg_pairs = [
            (i, j) for (i, j) in pairs
            if df["center_node"].iat[i] not in anoms and df["center_node"].iat[j] not in anoms
        ]
        edge_diff = np.array([abs(nov[i] - nov[j]) for i, j in bg_pairs])
        ri = rng.integers(0, len(nov), size=(len(bg_pairs), 2))
        rand_diff = np.abs(nov[ri[:, 0]] - nov[ri[:, 1]])

        at0 = df.loc[df["dist_to_anomaly"] == 0, "novelty"]
        far = df.loc[df["dist_to_anomaly"] >= 1, "novelty"]
        rows.append(
            {
                "network": config["network"] if "network" in config else run_dir.name.split("__")[0],
                "variant": config["variant"],
                "median_bag_size": config["bag_nodes_median"],
                "smoothness_ratio": round(float(np.median(edge_diff) / np.median(rand_diff)), 3),
                "spike_pctile": round(float((far < at0.median()).mean()), 3),
                "hub_pull_rho": round(float(spearmanr(df["novelty"], df["degree"]).statistic), 3),
                "d1_pctile": round(float((far < df.loc[df["dist_to_anomaly"] == 1, "novelty"].median()).mean()), 3),
            }
        )
    out = pd.DataFrame(rows).sort_values(["network", "variant"])
    out.to_csv(OUTPUTS / "headtohead.csv", index=False)
    print(out.to_string(index=False))


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    nxt = NEExT(log_level="ERROR")
    for network in NETWORKS:
        for variant in VARIANTS:
            run_one(nxt, network, variant)
    analyze()


if __name__ == "__main__":
    main()
