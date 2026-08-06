"""Experiment 1: is there class signal in the k-hop egonet? (Air Traffic Europe)

Pipeline: catalog dataset -> k-hop egonet per node for k in {1,2,3} (label =
center's class) -> 4 structural features per member (feature_vector_length=1,
computed inside the egonet) -> approx-Wasserstein embedding (dim 4) -> XGBoost
under 10x stratified 70/30 shared splits, against two floors:

  - permuted: same pipeline, training labels shuffled (the honest random
    model, the figure baseline)
  - majority: train-majority constant predictor (ledger only, off the figure)

Artifacts follow lib/containment/runio conventions: one dir per method under
outputs/, plus a session-root results.csv aggregate. Figures read artifacts
only (code/plot_results.py).

Usage (from the session directory):
    uv run python code/run_experiment.py                  # local feature scope
    uv run python code/run_experiment.py --scope global   # global feature scope
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["local", "global"], default="local", help="feature_scope for compute_node_features")
    parser.add_argument("--family", choices=["khop", "walk"], default="khop", help="bag construction family")
    args = parser.parse_args()

    cfg = dict(C.CONFIG)
    cfg["feature_scope"] = args.scope
    cfg["family"] = args.family
    if args.scope == "global":
        cfg["feature_tag"] = f"{cfg['feature_tag']}-glob"
    t0 = time.time()
    print(f"family={cfg['family']}  feature_scope={cfg['feature_scope']}  feature_tag={cfg['feature_tag']}")

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

    eval_kwargs = dict(n_splits=cfg["n_splits"], test_size=cfg["test_size"], seed=cfg["ml_seed"])
    if cfg["family"] == "khop":
        constructions = {f"egonet_k{k}_wass": ("k_hop", {"k_hop": k}) for k in cfg["k_hops"]}
    else:
        constructions = {f"{name}_wass": ("random_walk", {**params, "n_walks": C.N_WALKS}) for name, params in C.WALK_CONSTRUCTIONS.items()}

    results, bag_tables, reaches, reps = {}, {}, {}, {}
    for method, (bag_method, bag_params) in constructions.items():
        if bag_method == "k_hop":
            k = bag_params["k_hop"]
            reach = khop_reach(collection, centers, k=k)
            reaches[k] = reach
            (OUTPUTS / f"{cfg['dataset']}__k{k}_reach.json").write_text(json.dumps(reach, indent=2))
            print(f"[{time.time() - t0:6.1f}s] k={k} reach: median {reach['median']:.0f} nodes "
                  f"({reach['median_frac_of_graph']:.0%} of graph), p90 {reach['p90']:.0f}, max {reach['max']:.0f}")

        bags = build_node_bags(
            nxt, collection, centers, cfg["label_column"], method=bag_method,
            seed=cfg["egonet_seed"], **bag_params,
        )
        features = nxt.compute_node_features(
            bags.egonets,
            feature_list=list(cfg["feature_list"]),
            feature_vector_length=cfg["feature_vector_length"],
            show_progress=False,
            n_jobs=-1,
            feature_scope=cfg["feature_scope"],
        )
        rep_df = wasserstein_embedding(nxt, bags.egonets, features, dimension=cfg["wasserstein_dim"], seed=cfg["embed_seed"])
        rep_node = egonet_rep_to_node_frame(rep_df, bags.table)
        results[method] = evaluate_node_representation(method, rep_node, node_table, **eval_kwargs)
        bag_tables[method], reps[method] = bags.table, rep_node
        print(f"[{time.time() - t0:6.1f}s] {method}: {len(bags.egonets.graphs)} bags "
              f"(median size {bags.table['n_nodes'].median():.0f}) evaluated")

    # Floors ride with the khop family only: permuted on the k=2 representation
    # (any would do — it exists to show the pipeline scores at chance without
    # true labels); the walk family reuses those floor rows at plot time.
    # Majority is recorded for the ledger but stays off the figure.
    if cfg["family"] == "khop":
        results["permuted"] = permutation_floor("permuted", reps["egonet_k2_wass"], node_table, **eval_kwargs)
        results["majority"] = majority_floor(node_table, **eval_kwargs)

    run_config = {
        **cfg,
        "n_centers": len(centers),
        "table_report": table_report,
        "filter_report": filter_report,
        "reach": {str(k): r for k, r in reaches.items()},
        "git_sha": runio.git_sha(REPO_ROOT),
        "neext_version": runio.neext_version(),
    }
    all_rows = []
    fallback_table = next(iter(bag_tables.values()))
    for method, out in results.items():
        run_config_m = {**run_config, "method": method, "status": out["status"]}
        run_id = f"{cfg['dataset']}__{method}__{cfg['feature_tag']}"
        runio.write_run(
            OUTPUTS,
            run_id,
            run_config_m,
            out["metrics_rows"],
            out["node_predictions"],
            bag_tables.get(method, fallback_table),
        )
        all_rows.extend(out["metrics_rows"])
        if method in reps:
            reps[method].to_parquet(OUTPUTS / run_id / "representation.parquet", index=False)
    runio.aggregate(OUTPUTS, ["dataset", "method", "status", "feature_tag", "feature_scope", "git_sha"])

    print(f"[{time.time() - t0:6.1f}s] done\n")
    print(summarize_node_metrics(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
