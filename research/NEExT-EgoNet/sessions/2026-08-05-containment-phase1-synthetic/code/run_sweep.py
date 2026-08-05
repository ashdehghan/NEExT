"""Phase 1: synthetic detectability sweep for containment search.

For each cell (family x anomaly x prevalence x k): build a planted-anomaly
graph, decompose into k-hop bags around sampled centers, compute structural
features ONCE, and evaluate every representation on the containment label
under identical splits. Each cell writes outputs/<run_id>/ (config, metrics,
per-bag predictions, bag table); the sweep is resumable (existing cells are
skipped) and aggregates to outputs/../results.csv incrementally.

Usage:
    uv run python code/run_sweep.py            # full sweep
    uv run python code/run_sweep.py --smoke    # one tiny cell end-to-end
"""

import sys
import time
from itertools import product
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import (  # noqa: E402
    aggregate,
    build_bags,
    evaluate_representation,
    git_sha,
    make_synthetic,
    neext_version,
    node_oracle_scores,
    pooled_features,
    run_complete,
    size_only,
    wasserstein_embedding,
    write_run,
)

import pandas as pd  # noqa: E402

from NEExT import NEExT  # noqa: E402

SWEEP = {
    "families": ["er", "ba"],
    "anomalies": ["hub", "clique", "tail"],
    "prevalences": [0.005, 0.01, 0.02, 0.05, 0.10],
    "k_hops": [1, 2],
    "n": 3000,
    "n_centers": 800,
    "graph_seed": 7,
    "center_seed": 13,
    "ml_seed": 42,
    "feature_list": ["all"],
    "feature_vector_length": 3,
    "embedding_dimension": 16,
    "n_splits": 10,
    "test_size": 0.3,
}

SMOKE_OVERRIDES = {
    "families": ["er"],
    "anomalies": ["hub"],
    "prevalences": [0.05],
    "k_hops": [1],
    "n": 400,
    "n_centers": 200,
    "n_splits": 3,
}

CONFIG_KEYS = ["family", "anomaly", "prevalence", "k_hop", "n", "n_centers", "bag_nodes_median", "bag_nodes_p90"]
OUTPUTS = SESSION_ROOT / "outputs"


def run_cell(nxt, params, family, anomaly, prevalence, k_hop) -> None:
    run_id = f"{family}_{anomaly}_p{prevalence:g}_k{k_hop}"
    if run_complete(OUTPUTS, run_id):
        print(f"[skip] {run_id} (already complete)", flush=True)
        return
    t0 = time.time()

    edges_df, nodes_df, gen_config = make_synthetic(
        family, anomaly, n=params["n"], prevalence=prevalence, seed=params["graph_seed"]
    )
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )
    bags = build_bags(
        nxt,
        collection,
        label_column="is_anomaly",
        k_hop=k_hop,
        n_centers=params["n_centers"],
        seed=params["center_seed"],
    )
    t_bags = time.time() - t0

    features = nxt.compute_node_features(
        bags.egonets,
        feature_list=params["feature_list"],
        feature_vector_length=params["feature_vector_length"],
        show_progress=False,
        n_jobs=-1,
    )
    t_features = time.time() - t0 - t_bags

    reps = {
        "wasserstein": wasserstein_embedding(
            nxt, bags.egonets, features, dimension=params["embedding_dimension"], seed=params["ml_seed"]
        ),
        "pooled_all": pooled_features(features, stats=("mean", "max", "p90")),
        "pooled_max": pooled_features(features, stats=("max",)),
        "size_only": size_only(bags.table),
        "node_oracle": node_oracle_scores(nxt, collection, bags, seed=params["ml_seed"]),
    }

    metrics_rows, score_frames = [], []
    for rep_name, rep_df in reps.items():
        result = evaluate_representation(
            rep_name, rep_df, bags.table, n_splits=params["n_splits"], test_size=params["test_size"], seed=params["ml_seed"]
        )
        metrics_rows.extend(result["metrics_rows"])
        if not result["bag_scores"].empty:
            score_frames.append(result["bag_scores"])

    bag_predictions = (
        pd.concat(score_frames, ignore_index=True).merge(bags.table, on="graph_id") if score_frames else bags.table.copy()
    )

    config = {
        **params,
        **gen_config,
        "k_hop": k_hop,
        "run_id": run_id,
        "git_sha": git_sha(RESEARCH_ROOT.parents[0]),
        "neext_version": neext_version(),
        "bag_nodes_median": float(bags.table["n_nodes"].median()),
        "bag_nodes_p90": float(bags.table["n_nodes"].quantile(0.9)),
        "bag_positive_rate": float(bags.table["y_contains"].mean()),
        "seconds_bags": round(t_bags, 1),
        "seconds_features": round(t_features, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    write_run(OUTPUTS, run_id, config, metrics_rows, bag_predictions, bags.table)
    aggregate(OUTPUTS, CONFIG_KEYS)
    ok_rows = [r for r in metrics_rows if r.get("status") == "ok"]
    aucs = {}
    for r in ok_rows:
        aucs.setdefault(r["representation"], []).append(r["roc_auc"])
    summary = ", ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in aucs.items()) or "DEGENERATE"
    print(
        f"[done] {run_id}: pos_rate={config['bag_positive_rate']:.3f} "
        f"med_size={config['bag_nodes_median']:.0f} auc[{summary}] ({config['seconds_total']:.0f}s)",
        flush=True,
    )


def main():
    smoke = "--smoke" in sys.argv
    params = {**SWEEP, **(SMOKE_OVERRIDES if smoke else {})}
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    nxt = NEExT(log_level="ERROR")

    cells = list(product(params["k_hops"], params["families"], params["anomalies"], params["prevalences"]))
    print(f"{'SMOKE ' if smoke else ''}sweep: {len(cells)} cells", flush=True)
    t0 = time.time()
    for k_hop, family, anomaly, prevalence in cells:  # k=1 cells first
        run_cell(nxt, params, family, anomaly, prevalence, k_hop)
    print(f"Sweep complete in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
