"""Region-holdout (separated-split) evaluation over cached representations.

Phase 3 Part A: how much of each representation's performance survives when
train and test centers cannot share neighborhoods? Reuses the cached
representation.parquet artifacts from the source sessions — no feature or
embedding recomputation. For each (dataset, representation, buffer B):
region-holdout splits with 10 region-seed repeats, plus a permutation floor
under the same splits.

Targets (--target):
  synth   : SYNTH_HR_L2_B0 / SYNTH_HR_L2_B1 originals, global khop k1..6
            (the calibration anchors; P-A1 gate)
  amazon  : AMAZON_RATINGS, khop k1..5 both scopes + node_struct
  chameleon: WIKIPEDIA_ARTICLES_CHAMELEON, khop k1..3 both scopes + node_struct

Run from the repo root.
"""

import argparse
import sys
import time
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

import numpy as np
import pandas as pd

from lib.nodeclass import build_node_table, filter_rare_classes, sample_centers
from lib.nodeclass.evaluate import evaluate_node_representation, permutation_floor
from lib.nodeclass.separated_splits import adjacency_from_edges, region_holdout_splits

from NEExT import NEExT

SESSIONS = RESEARCH_ROOT / "sessions"
BUFFERS = [0, 1, 2, 4, 6, 8, 10]
N_REPEATS = 10
SPLIT_SEED = 42

TARGETS = {
    "synth": {
        "session": "2026-08-12-synthetic-homophily-reach",
        "datasets": ["SYNTH_HR_L2_B0", "SYNTH_HR_L2_B1"],
        "label_column": "field_class",
        "n_centers": 2000,
        "reps": [(f"egonet_k{k}_wass", "fall-vl1-glob") for k in range(1, 7)],
        "node_struct": False,
    },
    "amazon": {
        "session": "2026-08-06-egonet-signal-amazon-ratings",
        "datasets": ["AMAZON_RATINGS"],
        "label_column": "rating_class",
        "n_centers": 3000,
        "reps": [(f"egonet_k{k}_wass", tag) for k in range(1, 6) for tag in ("fall-vl1", "fall-vl1-glob")],
        "node_struct": True,
    },
    "chameleon": {
        "session": "2026-08-06-egonet-signal-wiki-chameleon",
        "datasets": ["WIKIPEDIA_ARTICLES_CHAMELEON"],
        "label_column": "traffic_quintile",
        "n_centers": 1000,
        "reps": [(f"egonet_k{k}_wass", tag) for k in range(1, 4) for tag in ("fall-vl1", "fall-vl1-glob")],
        "node_struct": True,
    },
    "roman": {
        "session": "2026-08-06-egonet-signal-roman-empire",
        "datasets": ["ROMAN_EMPIRE"],
        "label_column": "syntactic_role",
        "n_centers": 3000,
        "reps": [(f"egonet_k{k}_wass", tag) for k in range(1, 5) for tag in ("fall-vl1", "fall-vl1-glob")],
        "node_struct": True,
    },
}


def load_dataset(session: str, dataset: str, label_column: str):
    sys.path.insert(0, str(SESSIONS / session / "code"))
    import datasets as ds_mod  # each session ships a loader with this name

    edges_df, nodes_df = ds_mod.load_single_graph_dataset(dataset, label_column=label_column, structural_only=True)
    sys.path.pop(0)
    for mod in ("datasets", "config"):
        sys.modules.pop(mod, None)
    return edges_df, nodes_df


def rep_path(session: str, dataset: str, method: str, tag: str) -> Path:
    return SESSIONS / session / "outputs" / f"{dataset}__{method}__{tag}" / "representation.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    args = parser.parse_args()
    cfg = TARGETS[args.target]

    out_dir = SESSION_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    t0 = time.time()
    for dataset in cfg["datasets"]:
        edges_df, nodes_df = load_dataset(cfg["session"], dataset, cfg["label_column"])
        nxt = NEExT(log_level="ERROR")
        collection = nxt.load_single_graph_from_dfs(
            edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
        )
        node_table, _ = build_node_table(collection, cfg["label_column"])
        node_table, _ = filter_rare_classes(node_table, min_count=10)
        node_table = sample_centers(node_table, n_centers=cfg["n_centers"], seed=13)

        g = collection.graphs[0].G
        adj = adjacency_from_edges(g.vcount(), *zip(*[(e.source, e.target) for e in g.es]))
        print(f"[{time.time() - t0:6.1f}s] {dataset}: {g.vcount()} nodes, {len(node_table)} centers")

        methods = list(cfg["reps"])
        if cfg["node_struct"]:
            methods.append(("node_struct", "fall-vl1"))
        for method, tag in methods:
            if method == "node_struct":
                # The nodefeat family never persisted a representation.parquet;
                # its representation IS the cached full-graph feature table.
                src = SESSIONS / cfg["session"] / "outputs" / f"{dataset}__source_features__fall-vl1.parquet"
                if not src.exists():
                    raise FileNotFoundError(f"missing cached source features: {src}")
                feats = pd.read_parquet(src)
                keep = [c for c in feats.columns if c != "graph_id"]
                rep_df = feats[keep].merge(node_table[["node_id"]], on="node_id", how="inner")
            else:
                path = rep_path(cfg["session"], dataset, method, tag)
                if not path.exists():
                    raise FileNotFoundError(f"missing cached representation: {path}")
                rep_df = pd.read_parquet(path)
            for B in BUFFERS:
                splits, stats_rows = [], []
                for name, idx_train, idx_test, stats in region_holdout_splits(
                    adj, node_table, buffer_hops=B, n_repeats=N_REPEATS, seed=SPLIT_SEED
                ):
                    splits.append((name, idx_train, idx_test))
                    stats_rows.append(stats)
                out = evaluate_node_representation(method, rep_df, node_table, splits=splits)
                floor = permutation_floor("permuted", rep_df, node_table, splits=splits)
                stats_df = pd.DataFrame(stats_rows)
                for row in out["metrics_rows"] + floor["metrics_rows"]:
                    row.update(
                        dataset=dataset,
                        feature_tag=tag,
                        method=method if row["representation"] != "permuted" else f"permuted_{method}",
                        buffer_hops=B,
                        n_train_mean=float(stats_df.n_train.mean()),
                        n_test_mean=float(stats_df.n_test.mean()),
                    )
                    all_rows.append(row)
                ok = [r for r in out["metrics_rows"] if r["status"] == "ok"]
                mean_f1 = np.mean([r["f1_macro"] for r in ok]) if ok else float("nan")
                print(f"[{time.time() - t0:6.1f}s] {dataset} {method} {tag} B={B}: "
                      f"f1={mean_f1:.3f} over {len(ok)}/{N_REPEATS} splits "
                      f"(train~{stats_df.n_train.mean():.0f})")
            # incremental checkpoint: a late failure must not discard finished work
            pd.DataFrame(all_rows).to_csv(out_dir / f"separation_{args.target}.csv", index=False)
    result = pd.DataFrame(all_rows)
    out_path = out_dir / f"separation_{args.target}.csv"
    result.to_csv(out_path, index=False)
    print(f"[{time.time() - t0:6.1f}s] written {out_path} ({len(result)} rows)")


if __name__ == "__main__":
    main()
