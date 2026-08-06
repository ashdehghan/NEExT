"""Node-classification benchmark: egonet embeddings vs classic node embeddings.

One cell = (dataset, method). Methods:
  egonet side  — 5 constructions (hop_k1, hop_k2 guarded, 3 walk variants),
                 each as wass16 + pooled (+ a free sizeonly diagnostic)
  baselines    — 6 karateclub methods, center_struct, degree_only, majority

Every cell is resumable (skipped when its metrics.csv exists) and persists
everything post-hoc analysis could need: the full representation matrix
(parquet), per-split predictions with class probabilities, per-stage
timings, and every parameter. Figures and any future analysis read ONLY
these artifacts.

Usage:
    uv run python code/run_benchmark.py                # full matrix
    uv run python code/run_benchmark.py --smoke        # 5-min end-to-end
    uv run python code/run_benchmark.py --datasets CORA,BOOKS
    uv run python code/run_benchmark.py --methods 'hop_k1*,deepwalk'
    uv run python code/run_benchmark.py --list         # cell ledger
"""

import argparse
import fnmatch
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
REPO_ROOT = RESEARCH_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import runio  # noqa: E402
from lib.containment.representations import pooled_features, wasserstein_embedding  # noqa: E402
from lib.nodeclass import (  # noqa: E402
    KC_METHODS,
    build_node_bags,
    build_node_table,
    center_structural_features,
    degree_only,
    egonet_rep_to_node_frame,
    evaluate_node_representation,
    filter_rare_classes,
    kc_embed,
    khop_reach,
    majority_floor,
    sample_centers,
    to_networkx,
)
from NEExT import NEExT  # noqa: E402

import config as C  # noqa: E402
from datasets import load_single_graph_dataset  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"


def kc_version() -> str:
    try:
        from importlib.metadata import version

        return version("karateclub")
    except Exception:
        return "unknown"


def base_config(ds_id, label_column, label_type, method, family, cfg, extra=None) -> dict:
    return {
        "dataset": ds_id,
        "label_column": label_column,
        "label_type": label_type,
        "method": method,
        "family": family,
        "n_centers": cfg["n_centers"],
        "n_splits": cfg["n_splits"],
        "test_size": cfg["test_size"],
        "min_class_count": cfg["min_class_count"],
        "center_seed": cfg["center_seed"],
        "egonet_seed": cfg["egonet_seed"],
        "ml_seed": cfg["ml_seed"],
        "git_sha": runio.git_sha(REPO_ROOT),
        "neext_version": runio.neext_version(),
        "karateclub_version": kc_version(),
        **(extra or {}),
    }


def run_cell(ds_id, method, rep_df, node_table, cell_config, cfg, timings=None) -> None:
    """Evaluate one node-keyed representation and persist the full cell."""
    run_id = f"{ds_id}__{method}"
    if runio.run_complete(OUTPUTS, run_id):
        print(f"[skip] {run_id}", flush=True)
        return
    t0 = time.time()
    if rep_df is None:  # majority floor
        result = majority_floor(node_table, n_splits=cfg["n_splits"], test_size=cfg["test_size"], seed=cfg["ml_seed"])
    else:
        result = evaluate_node_representation(
            method, rep_df, node_table, n_splits=cfg["n_splits"], test_size=cfg["test_size"], seed=cfg["ml_seed"]
        )
    cell_config = dict(cell_config)
    cell_config["timings"] = {**(timings or {}), "eval": round(time.time() - t0, 2)}
    cell_config["status"] = result["status"]
    run_dir = runio.write_run(OUTPUTS, run_id, cell_config, result["metrics_rows"], result["node_predictions"], node_table)
    if rep_df is not None:
        rep_df.to_parquet(run_dir / "representation.parquet", index=False)
    print(f"[done] {run_id} ({result['status']}, {time.time() - t0:.0f}s eval)", flush=True)


def write_error(ds_id, method, cell_config, err: str, node_table) -> None:
    """An errored cell is recorded (status='error' + traceback tail), not fatal.

    NOTE: error cells count as complete for resumability — delete the run dir
    to retry after fixing the cause.
    """
    run_id = f"{ds_id}__{method}"
    if runio.run_complete(OUTPUTS, run_id):
        return
    cell_config = dict(cell_config)
    cell_config["status"] = "error"
    cell_config["error"] = err
    rows = [{"representation": method, "split": "", "status": "error"}]
    runio.write_run(OUTPUTS, run_id, cell_config, rows, pd.DataFrame(), node_table)
    print(f"[ERROR] {run_id}: {err.splitlines()[-1] if err else 'unknown'}", flush=True)


def write_guarded(ds_id, method, cell_config, reach, node_table) -> None:
    run_id = f"{ds_id}__{method}"
    if runio.run_complete(OUTPUTS, run_id):
        print(f"[skip] {run_id}", flush=True)
        return
    cell_config = dict(cell_config)
    cell_config["status"] = "guarded"
    cell_config["k2_reach"] = reach
    rows = [{"representation": method, "split": "", "status": "guarded"}]
    runio.write_run(OUTPUTS, run_id, cell_config, rows, pd.DataFrame(), node_table)
    print(f"[guard] {run_id} (median k=2 reach {reach['median']:.0f} nodes)", flush=True)


def prepare_dataset(nxt, ds_id, label_column, cfg):
    """Load graph, build + persist the canonical node table and graph summary."""
    t0 = time.time()
    edges_df, nodes_df = load_single_graph_dataset(ds_id, label_column, structural_only=True)
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )
    load_seconds = round(time.time() - t0, 2)

    full_table, class_report = build_node_table(collection, label_column)
    filtered, filter_report = filter_rare_classes(full_table, min_count=cfg["min_class_count"])
    node_table = sample_centers(filtered, n_centers=cfg["n_centers"], seed=cfg["center_seed"])

    graph = collection.graphs[0]
    degs = np.array(graph.G.degree())
    summary = {
        "dataset": ds_id,
        "n_nodes": len(graph.nodes),
        "n_edges": int(graph.G.ecount()),
        "degree_mean": round(float(degs.mean()), 2),
        "degree_median": float(np.median(degs)),
        "degree_max": int(degs.max()),
        "n_sampled_centers": len(node_table),
        "n_classes_sampled": int(node_table["y"].nunique()),
        "load_seconds": load_seconds,
        "class_report": class_report,
        "filter_report": filter_report,
    }
    node_table.to_csv(OUTPUTS / f"{ds_id}__node_table.csv", index=False)
    (OUTPUTS / f"{ds_id}__dataset.json").write_text(json.dumps(summary, indent=2))
    return collection, node_table, summary


def egonet_cells(nxt, ds_id, label_column, label_type, collection, node_table, cfg, method_filter):
    centers = node_table["node_id"].tolist()
    for cons_name, params in C.CONSTRUCTIONS.items():
        cell_names = [f"{cons_name}__wass16", f"{cons_name}__pooled", f"{cons_name}__sizeonly"]
        wanted = [m for m in cell_names if method_filter(m)]
        if not wanted:
            continue
        if all(runio.run_complete(OUTPUTS, f"{ds_id}__{m}") for m in wanted):
            for m in wanted:
                print(f"[skip] {ds_id}__{m}", flush=True)
            continue

        family = "egonet_hop" if params["method"] == "k_hop" else "egonet_walk"
        # "method" in cell config is the CELL name; the construction's own
        # method key would clobber it, so it rides as bag_method.
        cons_extra = {"construction": cons_name, "bag_method": params["method"],
                      **{k: v for k, v in params.items() if k != "method"}}

        if cons_name == "hop_k2":
            reach = khop_reach(collection, centers, k=2)
            guard = cfg["k2_guard"]
            limit = max(guard["abs_nodes"], guard["frac"] * len(collection.graphs[0].nodes))
            (OUTPUTS / f"{ds_id}__hop_k2_reach.json").write_text(json.dumps(reach, indent=2))
            if reach["median"] > limit:
                for m in wanted:
                    write_guarded(
                        ds_id, m, base_config(ds_id, label_column, label_type, m, family, cfg, cons_extra), reach, node_table
                    )
                continue

        try:
            _run_construction(
                nxt, ds_id, label_column, label_type, collection, node_table, cfg,
                method_filter, cons_name, params, family, cons_extra, centers,
            )
        except Exception:
            err = traceback.format_exc()
            for m in wanted:
                write_error(ds_id, m, base_config(ds_id, label_column, label_type, m, family, cfg, cons_extra), err, node_table)
        gc.collect()


def _run_construction(nxt, ds_id, label_column, label_type, collection, node_table, cfg,
                      method_filter, cons_name, params, family, cons_extra, centers):
        t0 = time.time()
        bags = build_node_bags(
            nxt, collection, centers, label_column, seed=cfg["egonet_seed"], **params
        )
        t_construct = round(time.time() - t0, 2)
        bags.table.to_csv(OUTPUTS / f"{ds_id}__{cons_name}__bag_table.csv", index=False)

        t0 = time.time()
        features = nxt.compute_node_features(
            bags.egonets,
            feature_list=list(cfg["feature_list"]),
            feature_vector_length=cfg["feature_vector_length"],
            show_progress=False,
            n_jobs=-1,
        )
        t_features = round(time.time() - t0, 2)
        shared_t = {"construct": t_construct, "features": t_features, "shared_across_cells": True}
        size_stats = {
            "n_nodes_median": float(bags.table["n_nodes"].median()),
            "n_nodes_p90": float(bags.table["n_nodes"].quantile(0.9)),
            "n_nodes_max": int(bags.table["n_nodes"].max()),
        }

        name = f"{cons_name}__wass16"
        if method_filter(name):
            t0 = time.time()
            rep = wasserstein_embedding(nxt, bags.egonets, features, dimension=cfg["wasserstein_dim"], seed=cfg["ml_seed"])
            t_embed = round(time.time() - t0, 2)
            run_cell(
                ds_id, name, egonet_rep_to_node_frame(rep, bags.table), node_table,
                base_config(ds_id, label_column, label_type, name, family, cfg,
                            {**cons_extra, "representation": "wass16", "dim": cfg["wasserstein_dim"], **size_stats}),
                cfg, {**shared_t, "embed": t_embed},
            )

        name = f"{cons_name}__pooled"
        if method_filter(name):
            t0 = time.time()
            rep = pooled_features(features, stats=cfg["pool_stats"], egonets=bags.egonets)
            t_pool = round(time.time() - t0, 2)
            run_cell(
                ds_id, name, egonet_rep_to_node_frame(rep, bags.table), node_table,
                base_config(ds_id, label_column, label_type, name, family, cfg,
                            {**cons_extra, "representation": "pooled", "pool_stats": list(cfg["pool_stats"]), **size_stats}),
                cfg, {**shared_t, "embed": t_pool},
            )

        name = f"{cons_name}__sizeonly"
        if method_filter(name):
            rep = bags.table[["node_id", "n_nodes", "n_edges"]].copy()
            run_cell(
                ds_id, name, rep, node_table,
                base_config(ds_id, label_column, label_type, name, "floor", cfg,
                            {**cons_extra, "representation": "sizeonly", **size_stats}),
                cfg, shared_t,
            )

        del bags, features
        gc.collect()


def baseline_cells(nxt, ds_id, label_column, label_type, collection, node_table, cfg, method_filter):
    G = None
    for method in C.KC_BASELINES:
        if not method_filter(method):
            continue
        if runio.run_complete(OUTPUTS, f"{ds_id}__{method}"):
            print(f"[skip] {ds_id}__{method}", flush=True)
            continue
        try:
            if G is None:
                G = to_networkx(collection)
            t0 = time.time()
            rep, record = kc_embed(method, G, seed=cfg["kc_seed"])
            t_embed = round(time.time() - t0, 2)
            rep_path = OUTPUTS / f"{ds_id}__{method}__fullgraph.parquet"
            rep.to_parquet(rep_path, index=False)  # full-graph embedding, reusable beyond the sample
            rep_sampled = rep[rep["node_id"].isin(node_table["node_id"])]
            run_cell(
                ds_id, method, rep_sampled, node_table,
                base_config(ds_id, label_column, label_type, method, "kc_baseline", cfg, record),
                cfg, {"embed": t_embed},
            )
        except Exception:
            write_error(
                ds_id, method, base_config(ds_id, label_column, label_type, method, "kc_baseline", cfg),
                traceback.format_exc(), node_table,
            )

    if method_filter("center_struct") and not runio.run_complete(OUTPUTS, f"{ds_id}__center_struct"):
        t0 = time.time()
        rep = center_structural_features(
            nxt, collection, feature_list=tuple(cfg["feature_list"]), feature_vector_length=cfg["feature_vector_length"]
        )
        t_feat = round(time.time() - t0, 2)
        rep.to_parquet(OUTPUTS / f"{ds_id}__center_struct__fullgraph.parquet", index=False)
        rep_sampled = rep[rep["node_id"].isin(node_table["node_id"])]
        run_cell(
            ds_id, "center_struct", rep_sampled, node_table,
            base_config(ds_id, label_column, label_type, "center_struct", "structural", cfg,
                        {"feature_vector_length": cfg["feature_vector_length"]}),
            cfg, {"features": t_feat},
        )
    elif method_filter("center_struct"):
        print(f"[skip] {ds_id}__center_struct", flush=True)

    if method_filter("degree_only"):
        run_cell(
            ds_id, "degree_only", degree_only(collection).merge(node_table[["node_id"]], on="node_id"), node_table,
            base_config(ds_id, label_column, label_type, "degree_only", "floor", cfg), cfg,
        )
    if method_filter("majority"):
        run_cell(ds_id, "majority", None, node_table, base_config(ds_id, label_column, label_type, "majority", "floor", cfg), cfg)


def all_method_names():
    names = []
    for cons in C.CONSTRUCTIONS:
        names += [f"{cons}__wass16", f"{cons}__pooled", f"{cons}__sizeonly"]
    names += C.KC_BASELINES + ["center_struct", "degree_only", "majority"]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="mini end-to-end: AIRPORTS_USA, 300 centers, 3 splits")
    ap.add_argument("--datasets", default=None, help="comma-separated catalog ids")
    ap.add_argument("--methods", default=None, help="comma-separated method patterns (fnmatch, e.g. 'hop_k1*,deepwalk')")
    ap.add_argument("--list", action="store_true", help="print the cell ledger and exit")
    args = ap.parse_args()

    global OUTPUTS
    cfg = dict(C.CONFIG)
    datasets = C.DATASETS
    if args.smoke:
        cfg.update({"n_centers": 300, "n_splits": 3})
        datasets = [d for d in C.DATASETS if d[0] == "AIRPORTS_USA"]
        OUTPUTS = SESSION_ROOT / "outputs-smoke"  # never pollutes the real matrix
    if args.datasets:
        wanted = {d.strip().upper() for d in args.datasets.split(",")}
        datasets = [d for d in datasets if d[0] in wanted]

    patterns = [p.strip() for p in args.methods.split(",")] if args.methods else ["*"]

    def method_filter(name):
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    if args.list:
        for ds_id, *_ in datasets:
            for m in all_method_names():
                if method_filter(m):
                    state = "done" if runio.run_complete(OUTPUTS, f"{ds_id}__{m}") else "pending"
                    print(f"{state:8} {ds_id}__{m}")
        return

    OUTPUTS.mkdir(exist_ok=True)
    t_start = time.time()
    nxt = NEExT(log_level="WARNING")
    for ds_id, label_column, label_type, _ in datasets:
        print(f"\n=== {ds_id} ({label_type}) ===", flush=True)
        t0 = time.time()
        collection, node_table, summary = prepare_dataset(nxt, ds_id, label_column, cfg)
        print(
            f"[data] {summary['n_nodes']} nodes / {summary['n_edges']} edges, "
            f"{summary['n_sampled_centers']} centers, {summary['n_classes_sampled']} classes "
            f"({summary['load_seconds']}s load)",
            flush=True,
        )
        egonet_cells(nxt, ds_id, label_column, label_type, collection, node_table, cfg, method_filter)
        baseline_cells(nxt, ds_id, label_column, label_type, collection, node_table, cfg, method_filter)
        print(f"[dataset done] {ds_id} in {(time.time() - t0) / 60:.1f} min", flush=True)
        del collection
        gc.collect()

    if args.smoke:
        print(f"\nSmoke done in {(time.time() - t_start) / 60:.1f} min (outputs-smoke/, no aggregate).", flush=True)
    else:
        runio.aggregate(OUTPUTS, ["dataset", "label_type", "method", "family", "construction", "status"])
        print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min. results.csv written.", flush=True)


if __name__ == "__main__":
    main()
