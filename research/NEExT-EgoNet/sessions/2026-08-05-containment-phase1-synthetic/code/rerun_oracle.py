"""Regenerate ONLY the node_oracle rows of every phase-1 cell, leak-free.

The audit (2026-08-05) found the original oracle trained on all bag centers
(including test bags). This script rebuilds each cell's graph and bags
deterministically (same seeds), VERIFIES the rebuilt bag table matches the
stored one exactly (abort on mismatch — never write numbers against a graph
we can't prove is the same), evaluates the fixed `evaluate_node_oracle`, and
rewrites only the node_oracle rows in metrics.csv / bag_predictions.csv.
Fair-representation rows are asserted untouched. Re-aggregates results.csv.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import aggregate, build_bags, evaluate_node_oracle, git_sha, make_synthetic  # noqa: E402

from NEExT import NEExT  # noqa: E402

from run_sweep import CONFIG_KEYS, OUTPUTS, SWEEP  # noqa: E402

BAG_COLS = ["graph_id", "center_node", "y_contains", "y_center", "n_positive_members", "n_nodes", "n_edges"]


def rerun_cell(nxt, run_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("oracle_rerun"):
        print(f"[skip] {run_dir.name} (already rerun)", flush=True)
        return
    t0 = time.time()

    edges_df, nodes_df, _ = make_synthetic(
        config["family"], config["anomaly"], n=config["n"], prevalence=config["prevalence"], seed=config["seed"]
    )
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )
    bags = build_bags(
        nxt, collection, label_column="is_anomaly", k_hop=config["k_hop"],
        n_centers=config["n_centers"], seed=config["center_seed"],
    )
    stored = pd.read_csv(run_dir / "bag_table.csv")
    rebuilt = bags.table[BAG_COLS].reset_index(drop=True)
    if not rebuilt.equals(stored[BAG_COLS].reset_index(drop=True)):
        raise AssertionError(f"{run_dir.name}: rebuilt bag table differs from stored — aborting, nothing written")

    oracle = evaluate_node_oracle(
        nxt, collection, bags, n_splits=SWEEP["n_splits"], test_size=SWEEP["test_size"], seed=SWEEP["ml_seed"]
    )

    metrics = pd.read_csv(run_dir / "metrics.csv")
    fair_before = metrics[metrics["representation"] != "node_oracle"].copy()
    new_metrics = pd.concat([fair_before, pd.DataFrame(oracle["metrics_rows"])], ignore_index=True)
    # Oracle rows may add columns (n_excluded_bags) and concat can coerce
    # dtypes on degenerate cells (empty-string metrics); compare fair rows on
    # their own columns, value-wise.
    fair_after = new_metrics[new_metrics["representation"] != "node_oracle"][fair_before.columns]
    same = fair_before.reset_index(drop=True).astype(str).equals(fair_after.reset_index(drop=True).astype(str))
    assert same, "fair rows changed"
    new_metrics.to_csv(run_dir / "metrics.csv", index=False)

    preds_path = run_dir / "bag_predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        if "representation" in preds.columns:
            preds = preds[preds["representation"] != "node_oracle"]
            if not oracle["bag_scores"].empty:
                oracle_preds = oracle["bag_scores"].merge(stored, on="graph_id")
                preds = pd.concat([preds, oracle_preds], ignore_index=True)
            preds.to_csv(preds_path, index=False)

    config["oracle_rerun"] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(RESEARCH_ROOT.parents[0]),
        "protocol": "per-split training on train-bag centers; test bags scored on non-training-center members only",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    ok = [r["roc_auc"] for r in oracle["metrics_rows"] if r.get("status") == "ok"]
    summary = f"auc={sum(ok)/len(ok):.3f} over {len(ok)} ok splits" if ok else f"status={oracle['status']}"
    print(f"[done] {run_dir.name}: oracle {summary} ({time.time()-t0:.0f}s)", flush=True)


def main():
    nxt = NEExT(log_level="ERROR")
    run_dirs = sorted(d for d in OUTPUTS.iterdir() if (d / "metrics.csv").exists())
    print(f"Rerunning oracle for {len(run_dirs)} cells", flush=True)
    for run_dir in run_dirs:
        rerun_cell(nxt, run_dir)
    aggregate(OUTPUTS, CONFIG_KEYS)
    print("Oracle rerun complete; results.csv re-aggregated.", flush=True)


if __name__ == "__main__":
    main()
