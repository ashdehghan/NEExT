"""Fit external node embeddings on the full graph and evaluate them on the
canonical center set of the paper's signal sessions (same 10 stratified splits,
same XGBoost, own permutation floor per method).

Per (dataset, method, dim, seed):
  outputs/<DS>__<method>__d<dim>-s<seed>__fullgraph.parquet   cached embedding (node_id + cols)
  outputs/<DS>__<method>__d<dim>-s<seed>/                      runio run dir (+ representation.parquet
                                                                for run_separation.py)
  outputs/<DS>__permuted_<method>__d<dim>-s<seed>/             paired permutation floor
  outputs/<DS>__node_table.csv, <DS>__center_gate.json         the canonical centers + gate proof
and results.csv at the session root (runio.aggregate).

Usage (from the repo root, always --no-sync):
  uv run --no-sync python research/NEExT-EgoNet/sessions/2026-08-21-embedding-baselines/code/run_embeddings.py \
      --dataset chameleon --methods all --probe      # embed + cache + timings only
  ... --dataset roman --methods struc2vec,graphwave --dims 16 --seeds 42
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
REPO_ROOT = RESEARCH_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import runio  # noqa: E402
from lib.nodeclass import (  # noqa: E402
    FAMILY_OF,
    build_node_table,
    embed,
    evaluate_node_representation,
    filter_rare_classes,
    permutation_floor,
    sample_centers,
    summarize_node_metrics,
    to_networkx,
)
from NEExT import NEExT  # noqa: E402

import config as C  # noqa: E402
from datasets import load_single_graph_dataset  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
SESSIONS = RESEARCH_ROOT / "sessions"


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def canonical_node_table(ds):
    """The signal session's node_struct bag_table.csv IS its canonical node table."""
    path = SESSIONS / ds["signal_session"] / "outputs" / f"{ds['catalog_id']}__node_struct__fall-vl1" / "bag_table.csv"
    return pd.read_csv(path) if path.exists() else None, path


def assert_center_gate(node_table, ds, dataset_key):
    ref, path = canonical_node_table(ds)
    if ref is None:
        raise FileNotFoundError(f"center gate: canonical node table missing at {path}")
    mine = node_table[["node_id", "y"]].reset_index(drop=True)
    theirs = ref[["node_id", "y"]].reset_index(drop=True).astype(mine.dtypes.to_dict())
    pd.testing.assert_frame_equal(mine, theirs)
    sha = hashlib.sha256(mine.to_csv(index=False).encode()).hexdigest()
    gate = {"dataset": dataset_key, "n_centers": int(len(mine)), "reference": str(path.relative_to(RESEARCH_ROOT)), "sha256": sha}
    (OUTPUTS / f"{ds['catalog_id']}__center_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--methods", default="all", help="'all' (core methods) or comma-separated registry ids")
    parser.add_argument("--dims", default=str(C.CONFIG["dim"]), help="comma-separated embedding dims")
    parser.add_argument("--seeds", default=",".join(str(s) for s in C.CONFIG["embed_seeds"]))
    parser.add_argument("--probe", action="store_true", help="embed + cache + timing table only, no evaluation")
    parser.add_argument("--force", action="store_true", help="recompute even if the run is complete")
    args = parser.parse_args()

    ds = C.DATASETS[args.dataset]
    methods = C.CORE_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    OUTPUTS.mkdir(exist_ok=True)
    t0 = time.time()

    edges_df, nodes_df = load_single_graph_dataset(ds["catalog_id"], label_column=ds["label_column"], structural_only=True)
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type=C.CONFIG["graph_type"], filter_largest_component=True
    )
    node_table, table_report = build_node_table(collection, ds["label_column"])
    node_table, filter_report = filter_rare_classes(node_table, min_count=C.CONFIG["min_class_count"])
    if ds["n_centers"]:
        node_table = sample_centers(node_table, n_centers=ds["n_centers"], seed=C.CONFIG["center_seed"])
    gate = assert_center_gate(node_table, ds, args.dataset)
    node_table.to_csv(OUTPUTS / f"{ds['catalog_id']}__node_table.csv", index=False)
    centers = node_table["node_id"].to_numpy()
    G = to_networkx(collection)
    n = G.number_of_nodes()
    print(f"[{time.time() - t0:6.1f}s] {ds['catalog_id']}: {n} nodes, {G.number_of_edges()} edges, "
          f"{len(centers)} centers (gate OK, sha {gate['sha256'][:8]})")

    eval_kwargs = {"n_splits": C.CONFIG["n_splits"], "test_size": C.CONFIG["test_size"], "seed": C.CONFIG["ml_seed"]}
    timing_rows = []
    for dim in dims:
        for seed in seeds:
            for method in methods:
                tag = f"d{dim}-s{seed}"
                run_id = f"{ds['catalog_id']}__{method}__{tag}"
                cache = OUTPUTS / f"{run_id}__fullgraph.parquet"
                if not args.probe and not args.force and runio.run_complete(OUTPUTS, run_id):
                    print(f"[{time.time() - t0:6.1f}s] {run_id}: complete, skipping")
                    continue
                rss0 = peak_rss_mb()
                if cache.exists() and not args.force:
                    rep = pd.read_parquet(cache)
                    record = json.loads(cache.with_suffix(".json").read_text())
                    print(f"[{time.time() - t0:6.1f}s] {run_id}: embedding loaded from cache")
                else:
                    t1 = time.time()
                    nodes = centers if method.startswith("graphwave") else None  # exact per-column for GraphWave
                    rep, record = embed(method, G, seed=seed, dim=dim, nodes=nodes)
                    record["embed_seconds"] = time.time() - t1
                    record["peak_rss_mb"] = peak_rss_mb()
                    record["peak_rss_before_mb"] = rss0
                    # contract checks before anything is cached
                    if rep["node_id"].dtype != np.int64:
                        raise AssertionError(f"{method}: node_id dtype {rep['node_id'].dtype}")
                    expected = len(centers) if nodes is not None else n
                    if len(rep) != expected or rep["node_id"].is_unique is False:
                        raise AssertionError(f"{method}: {len(rep)} rows, expected {expected}")
                    if not np.isfinite(rep.drop(columns='node_id').to_numpy()).all():
                        raise AssertionError(f"{method}: non-finite values")
                    rep.to_parquet(cache, index=False)
                    cache.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
                    print(f"[{time.time() - t0:6.1f}s] {run_id}: embedded in {record['embed_seconds']:.1f}s "
                          f"({record['n_features']} features, peak RSS {record['peak_rss_mb']:.0f} MB)")
                timing_rows.append({"dataset": ds["catalog_id"], "method": method, "dim": dim, "seed": seed,
                                    "embed_seconds": record.get("embed_seconds"), "n_features": record.get("n_features"),
                                    "peak_rss_mb": record.get("peak_rss_mb")})
                if args.probe:
                    continue

                rep_c = rep.merge(node_table[["node_id"]], on="node_id", how="inner")
                if len(rep_c) != len(centers):
                    raise AssertionError(f"{method}: embedding covers {len(rep_c)} of {len(centers)} centers")
                out = evaluate_node_representation(method, rep_c, node_table, **eval_kwargs)
                floor = permutation_floor(f"permuted_{method}", rep_c, node_table, **eval_kwargs)
                base_cfg = {
                    "dataset": ds["catalog_id"],
                    "dataset_key": args.dataset,
                    "label_column": ds["label_column"],
                    "n_centers": int(len(centers)),
                    "family": FAMILY_OF[method],
                    "dim": dim,
                    "embed_seed": seed,
                    "feature_tag": tag,
                    "feature_scope": "fullgraph",
                    "embed_record": record,
                    "n_features": record.get("n_features"),
                    "embed_seconds": record.get("embed_seconds"),
                    "peak_rss_mb": record.get("peak_rss_mb"),
                    "table_report": table_report,
                    "filter_report": filter_report,
                    "center_gate": gate,
                    "protocol": {**eval_kwargs, **{k: C.CONFIG[k] for k in ("graph_type", "min_class_count", "center_seed", "workers")}},
                    "git_sha": runio.git_sha(REPO_ROOT),
                    "neext_version": runio.neext_version(),
                    "lib_versions": _lib_versions(),
                }
                runio.write_run(OUTPUTS, run_id, {**base_cfg, "method": method, "status": out["status"]},
                                out["metrics_rows"], out["node_predictions"], node_table)
                rep_c.to_parquet(OUTPUTS / run_id / "representation.parquet", index=False)
                runio.write_run(OUTPUTS, f"{ds['catalog_id']}__permuted_{method}__{tag}",
                                {**base_cfg, "method": f"permuted_{method}", "status": floor["status"]},
                                floor["metrics_rows"], floor["node_predictions"], node_table)
                ok = [r for r in out["metrics_rows"] if r["status"] == "ok"]
                fl = [r for r in floor["metrics_rows"] if r["status"] == "ok"]
                print(f"[{time.time() - t0:6.1f}s] {run_id}: macro-F1 {np.mean([r['f1_macro'] for r in ok]):.3f} "
                      f"(floor {np.mean([r['f1_macro'] for r in fl]):.3f}) over {len(ok)} splits")
                runio.aggregate(OUTPUTS, ["dataset", "dataset_key", "method", "family", "dim", "embed_seed", "feature_tag",
                                          "n_features", "embed_seconds", "peak_rss_mb", "status", "git_sha", "neext_version"])

    print(f"\n[{time.time() - t0:6.1f}s] done")
    if timing_rows:
        print(pd.DataFrame(timing_rows).to_string(index=False))
    if not args.probe:
        res = pd.read_csv(SESSION_ROOT / "results.csv")
        res = res[(res.dataset == ds["catalog_id"]) & (res.status == "ok")]
        print(summarize_node_metrics(res.to_dict("records")).to_string(index=False))


def _lib_versions():
    import importlib.metadata as m

    out = {}
    for name in ("gensim", "numba", "networkx", "pecanpy", "karateclub", "xgboost", "scikit-learn"):
        try:
            out[name] = m.version(name)
        except Exception:
            out[name] = None
    return out


if __name__ == "__main__":
    main()
