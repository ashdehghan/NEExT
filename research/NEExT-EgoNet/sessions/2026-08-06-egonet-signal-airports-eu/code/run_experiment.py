"""Experiment 1: is there class signal in the 2-hop egonet? (Air Traffic Europe)

Pipeline: catalog dataset -> 2-hop egonet per node (label = center's class) ->
4 structural features per member (feature_vector_length=1, computed inside the
egonet) -> approx-Wasserstein embedding (dim 4) -> XGBoost under 10x stratified
70/30 shared splits, against two floors:

  - majority: train-majority constant predictor
  - permuted: same pipeline, training labels shuffled (the honest random model)

Artifacts follow lib/containment/runio conventions: one dir per method under
outputs/, plus a session-root results.csv aggregate. Figures read artifacts
only (code/plot_results.py).

Usage: uv run python code/run_experiment.py   (from the session directory)
"""

import json
import sys
import time
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
REPO_ROOT = RESEARCH_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import runio  # noqa: E402
from lib.containment.representations import wasserstein_embedding  # noqa: E402
from lib.nodeclass import (  # noqa: E402
    build_node_bags,
    build_node_table,
    egonet_rep_to_node_frame,
    evaluate_node_representation,
    filter_rare_classes,
    khop_reach,
    majority_floor,
    permutation_floor,
    summarize_node_metrics,
)
from NEExT import NEExT  # noqa: E402

import config as C  # noqa: E402
from datasets import load_single_graph_dataset  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"


def main():
    cfg = C.CONFIG
    t0 = time.time()

    edges_df, nodes_df = load_single_graph_dataset(
        cfg["dataset"], label_column=cfg["label_column"], structural_only=True
    )
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type=cfg["graph_type"], filter_largest_component=True
    )
    node_table, table_report = build_node_table(collection, cfg["label_column"])
    node_table, filter_report = filter_rare_classes(node_table, min_count=cfg["min_class_count"])
    centers = node_table["node_id"].tolist()  # every node is a center
    print(f"[{time.time() - t0:6.1f}s] graph loaded: {table_report['n_nodes']} nodes, "
          f"{table_report['n_classes']} classes, {len(centers)} centers")

    reach = khop_reach(collection, centers, k=cfg["k_hop"])
    (OUTPUTS / f"{cfg['dataset']}__k{cfg['k_hop']}_reach.json").write_text(json.dumps(reach, indent=2))
    print(f"[{time.time() - t0:6.1f}s] k={cfg['k_hop']} reach: median {reach['median']:.0f} nodes "
          f"({reach['median_frac_of_graph']:.0%} of graph), p90 {reach['p90']:.0f}, max {reach['max']:.0f}")

    bags = build_node_bags(
        nxt, collection, centers, cfg["label_column"], method="k_hop",
        k_hop=cfg["k_hop"], seed=cfg["egonet_seed"],
    )
    print(f"[{time.time() - t0:6.1f}s] {len(bags.egonets.graphs)} egonets built "
          f"(median size {bags.table['n_nodes'].median():.0f})")

    features = nxt.compute_node_features(
        bags.egonets,
        feature_list=list(cfg["feature_list"]),
        feature_vector_length=cfg["feature_vector_length"],
        show_progress=False,
        n_jobs=-1,
    )
    rep_df = wasserstein_embedding(nxt, bags.egonets, features, dimension=cfg["wasserstein_dim"], seed=cfg["embed_seed"])
    rep_node = egonet_rep_to_node_frame(rep_df, bags.table)
    print(f"[{time.time() - t0:6.1f}s] features + embedding done ({cfg['wasserstein_dim']} dims)")

    eval_kwargs = dict(n_splits=cfg["n_splits"], test_size=cfg["test_size"], seed=cfg["ml_seed"])
    results = {
        "egonet_k2_wass": evaluate_node_representation("egonet_k2_wass", rep_node, node_table, **eval_kwargs),
        "majority": majority_floor(node_table, **eval_kwargs),
        "permuted": permutation_floor("permuted", rep_node, node_table, **eval_kwargs),
    }

    run_config = {
        **cfg,
        "n_centers": len(centers),
        "table_report": table_report,
        "filter_report": filter_report,
        "reach": reach,
        "git_sha": runio.git_sha(REPO_ROOT),
        "neext_version": runio.neext_version(),
    }
    all_rows = []
    for method, out in results.items():
        run_config_m = {**run_config, "method": method, "status": out["status"]}
        runio.write_run(
            OUTPUTS,
            f"{cfg['dataset']}__{method}",
            run_config_m,
            out["metrics_rows"],
            out["node_predictions"],
            bags.table,
        )
        all_rows.extend(out["metrics_rows"])
    rep_node.to_parquet(OUTPUTS / f"{cfg['dataset']}__egonet_k2_wass" / "representation.parquet", index=False)
    runio.aggregate(OUTPUTS, ["dataset", "method", "status", "k_hop", "git_sha"])

    print(f"[{time.time() - t0:6.1f}s] done\n")
    print(summarize_node_metrics(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
