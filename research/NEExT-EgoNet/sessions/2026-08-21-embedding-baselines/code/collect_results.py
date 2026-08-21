"""Combine this session's embedding rows with the four signal sessions' bag /
node_struct / permuted rows into one frame, tagged by family.

Writes outputs/combined_results.csv with columns:
  dataset, dataset_key, method, feature_tag, family, split, f1_macro, accuracy, auc_ovr_macro,
  status, source_session, n_features, dim, embed_seed
and asserts the best bag per dataset matches config.DATASETS[...]["expected_best_bag"].
Run from the repo root.
"""

import sys
from pathlib import Path

import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import config as C  # noqa: E402

SESSIONS = RESEARCH_ROOT / "sessions"
KEEP = ["dataset", "method", "feature_tag", "split", "f1_macro", "accuracy", "auc_ovr_macro", "status", "n_features"]


def family_of_signal_row(method):
    if method == "permuted":
        return "floor"
    if method == "majority":
        return "majority"
    if method == "node_struct":
        return "node"
    if method.startswith("egonet_k"):
        return "bag_hop"
    if method.startswith("walk_"):
        return "bag_walk"
    return "other"


def main():
    frames = []
    for key, ds in C.DATASETS.items():
        res = SESSIONS / ds["signal_session"] / "results.csv"
        if not res.exists():
            print(f"skip {key}: {res} missing")
            continue
        df = pd.read_csv(res)
        df = df[(df.dataset == ds["catalog_id"]) & (df.feature_tag.isin(["fall-vl1", "fall-vl1-glob"]))]
        df = df[KEEP].copy()
        df["dataset_key"] = key
        df["family"] = df["method"].map(family_of_signal_row)
        df["source_session"] = ds["signal_session"]
        df["dim"] = None
        df["embed_seed"] = None
        frames.append(df)
    mine = pd.read_csv(SESSION_ROOT / "results.csv")
    mine = mine[KEEP + ["dataset_key", "family", "dim", "embed_seed"]].copy()
    mine.loc[mine.method.str.startswith("permuted_"), "family"] = "floor_emb"
    mine["source_session"] = SESSION_ROOT.name
    frames.append(mine)
    out = pd.concat(frames, ignore_index=True)
    out = out[out.status == "ok"]

    # best bag per dataset = argmax mean f1 over hop+walk rows; must match config
    for key, ds in C.DATASETS.items():
        if ds["expected_best_bag"] is None:
            continue
        bags = out[(out.dataset_key == key) & (out.family.isin(["bag_hop", "bag_walk"]))]
        if bags.empty:
            continue
        means = bags.groupby(["method", "feature_tag"])["f1_macro"].mean()
        best = means.idxmax()
        if tuple(best) != tuple(ds["expected_best_bag"]):
            raise AssertionError(f"{key}: best bag in results is {best}, config says {ds['expected_best_bag']}")
    path = SESSION_ROOT / "outputs" / "combined_results.csv"
    out.to_csv(path, index=False)
    summary = out.groupby(["dataset_key", "family", "method", "feature_tag"])["f1_macro"].agg(["mean", "std", "count"]).round(3)
    print(summary.to_string())
    print(f"\nwritten {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
