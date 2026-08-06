"""C2: containment margins with walk bags on the phase-1 subset.

Reruns 12 phase-1 cells (tail + clique on er/ba at pi in {.01,.02,.05}) with
random-walk bags under the audited protocol (shared splits, uniform
degeneracy rule, leak-free oracle skipped here — the fair-method margin over
size_only is the compared quantity). Representations: weighted Wasserstein
(walk weights flow through the incidence matrix), pooled_all extended with
min/p10 (audit follow-up), pooled_max, size_only. Hop-bag numbers are NOT
rerun — they are read from the frozen phase-1 results.csv for comparison.

Outputs: outputs/containment/<run_id>/ per cell + outputs/containment_results.csv
+ outputs/containment_comparison.csv (walk vs frozen hop margins).
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

from lib.containment import (  # noqa: E402
    build_bags,
    evaluate_representation,
    git_sha,
    make_synthetic,
    neext_version,
    pooled_features,
    size_only,
    wasserstein_embedding,
)

from NEExT import NEExT  # noqa: E402

CELLS = [
    (family, anomaly, prevalence)
    for family in ("er", "ba")
    for anomaly in ("tail", "clique")
    for prevalence in (0.01, 0.02, 0.05)
]
PARAMS = {
    "n": 3000,
    "n_centers": 800,
    "graph_seed": 7,
    "center_seed": 13,
    "ml_seed": 42,
    "n_splits": 10,
    "test_size": 0.3,
    "walk_length": 10,
    "n_walks": 100,
    "restart_prob": 0.15,
    "min_visits": 3,
}
PHASE1_RESULTS = SESSION_ROOT.parents[0] / "2026-08-05-containment-phase1-synthetic" / "results.csv"
OUTPUTS = SESSION_ROOT / "outputs" / "containment"


def run_cell(nxt, family, anomaly, prevalence):
    run_id = f"{family}_{anomaly}_p{prevalence:g}_walk"
    run_dir = OUTPUTS / run_id
    if (run_dir / "metrics.csv").exists():
        print(f"[skip] {run_id}", flush=True)
        return
    t0 = time.time()

    edges_df, nodes_df, gen_config = make_synthetic(family, anomaly, n=PARAMS["n"], prevalence=prevalence, seed=PARAMS["graph_seed"])
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )
    bags = build_bags(
        nxt,
        collection,
        label_column="is_anomaly",
        n_centers=PARAMS["n_centers"],
        seed=PARAMS["center_seed"],
        method="random_walk",
        walk_length=PARAMS["walk_length"],
        n_walks=PARAMS["n_walks"],
        restart_prob=PARAMS["restart_prob"],
        min_visits=PARAMS["min_visits"],
    )
    features = nxt.compute_node_features(bags.egonets, feature_list=["all"], feature_vector_length=3, show_progress=False, n_jobs=-1)

    reps = {
        "wasserstein": wasserstein_embedding(nxt, bags.egonets, features, dimension=16, seed=PARAMS["ml_seed"]),
        "pooled_all": pooled_features(features, stats=("mean", "max", "p90", "min", "p10"), egonets=bags.egonets),
        "pooled_max": pooled_features(features, stats=("max",)),
        "size_only": size_only(bags.table),
    }
    metrics_rows = []
    for rep_name, rep_df in reps.items():
        result = evaluate_representation(
            rep_name, rep_df, bags.table, n_splits=PARAMS["n_splits"], test_size=PARAMS["test_size"], seed=PARAMS["ml_seed"]
        )
        metrics_rows.extend(result["metrics_rows"])

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    bags.table.to_csv(run_dir / "bag_table.csv", index=False)
    config = {
        **gen_config,
        **PARAMS,
        "run_id": run_id,
        "bag_positive_rate": float(bags.table["y_contains"].mean()),
        "bag_nodes_median": float(bags.table["n_nodes"].median()),
        "git_sha": git_sha(RESEARCH_ROOT.parents[0]),
        "neext_version": neext_version(),
        "seconds_total": round(time.time() - t0, 1),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    ok = pd.DataFrame([r for r in metrics_rows if r.get("status") == "ok"])
    summary = (
        ", ".join(f"{k}={v:.3f}" for k, v in ok.groupby("representation").roc_auc.mean().items()) if not ok.empty else "DEGENERATE"
    )
    print(f"[done] {run_id}: pos_rate={config['bag_positive_rate']:.3f} auc[{summary}] ({config['seconds_total']:.0f}s)", flush=True)


def compare():
    frames = []
    for run_dir in sorted(OUTPUTS.iterdir()):
        if not (run_dir / "metrics.csv").exists():
            continue
        m = pd.read_csv(run_dir / "metrics.csv")
        cfg = json.loads((run_dir / "config.json").read_text())
        m["run_id"] = run_dir.name
        m["family"], m["anomaly"], m["prevalence"] = cfg["family"], cfg["anomaly"], cfg["prevalence"]
        m["n_bags_cfg"], m["pos_rate"] = 800, cfg["bag_positive_rate"]
        frames.append(m)
    walk = pd.concat(frames, ignore_index=True)
    walk.to_csv(SESSION_ROOT / "outputs" / "containment_results.csv", index=False)
    ok = walk[walk["status"] == "ok"].copy()
    minority = (np.minimum(ok["positive_rate"], 1 - ok["positive_rate"]) * ok["n_bags"]).round()
    ok = ok[minority >= 10]
    wagg = ok.groupby(["family", "anomaly", "prevalence", "representation"]).roc_auc.mean().unstack()
    wagg["walk_margin"] = wagg[["wasserstein", "pooled_all", "pooled_max"]].max(axis=1) - wagg["size_only"]

    p1 = pd.read_csv(PHASE1_RESULTS)
    p1ok = p1[p1["status"] == "ok"].copy()
    p1min = (np.minimum(p1ok["positive_rate"], 1 - p1ok["positive_rate"]) * p1ok["n_bags"]).round()
    p1ok = p1ok[p1min >= 10]
    p1ok = p1ok[p1ok["anomaly"].isin(["tail", "clique"]) & p1ok["prevalence"].isin([0.01, 0.02, 0.05])]
    hop = p1ok.groupby(["family", "anomaly", "prevalence", "k_hop", "representation"]).roc_auc.mean().unstack()
    hop["hop_margin"] = hop[["wasserstein", "pooled_all", "pooled_max"]].max(axis=1) - hop["size_only"]
    hop_best = hop.reset_index().groupby(["family", "anomaly", "prevalence"])["hop_margin"].max()

    comparison = wagg[["walk_margin"]].join(hop_best.rename("best_hop_margin"))
    comparison["delta"] = (comparison["walk_margin"] - comparison["best_hop_margin"]).round(3)
    comparison = comparison.round(3).reset_index()
    comparison.to_csv(SESSION_ROOT / "outputs" / "containment_comparison.csv", index=False)
    print(comparison.to_string(index=False))


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    nxt = NEExT(log_level="ERROR")
    for family, anomaly, prevalence in CELLS:
        run_cell(nxt, family, anomaly, prevalence)
    compare()


if __name__ == "__main__":
    main()
