"""Scripted statistics for the baselines section.

outputs/stats_shared_splits.csv — per (dataset, method, tag): mean/std macro-F1,
accuracy, and paired tests over the 10 shared splits vs node_struct, vs the best
bag, vs the method's own permutation floor (scipy ttest_rel + wilcoxon).

outputs/stats_region_holdout.csv — per (dataset, method, buffer): lift over the
paired floor on every B{B}_r{r} split, one-sample t vs 0, paired t vs the best
bag's lift, retention vs the shared-split lift.
Run from the repo root, after collect_results.py (and run_separation.py for the
holdout part).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(SESSION_ROOT / "code"))

import config as C  # noqa: E402

OUT = SESSION_ROOT / "outputs"
SEP_DIR = RESEARCH_ROOT / "sessions" / "2026-08-13-leakage-control" / "outputs"
HOLDOUT_BUFFERS = {"amazon": [0, 2, 4], "roman": [0, 2, 4], "chameleon": [0, 2], "synth_b0": [0, 2, 4], "synth_b1": [0, 2, 4]}


def _paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 3:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    except ValueError:
        w = np.nan
    return float(t), float(p), float(w)


def _series(df, key, method, tag):
    d = df[(df.dataset_key == key) & (df.method == method) & (df.feature_tag == tag)].sort_values("split")
    return d["f1_macro"].to_numpy(), d["accuracy"].to_numpy(), d["split"].to_numpy()


def shared_split_stats(comb):
    rows = []
    for key, ds in C.DATASETS.items():
        d = comb[comb.dataset_key == key]
        if d.empty:
            continue
        node_f1, _, node_splits = _series(comb, key, "node_struct", "fall-vl1")
        best = ds["expected_best_bag"]
        if best is None:
            bags = d[d.family.isin(["bag_hop", "bag_walk"])]
            best = tuple(bags.groupby(["method", "feature_tag"])["f1_macro"].mean().idxmax()) if not bags.empty else None
        bag_f1 = _series(comb, key, *best)[0] if best else np.array([])
        emb = d[d.family.isin(["proximity", "structural"])]
        for (method, tag), g in emb.groupby(["method", "feature_tag"]):
            g = g.sort_values("split")
            f1 = g["f1_macro"].to_numpy()
            floor = _series(comb, key, f"permuted_{method}", tag)[0]
            row = {
                "dataset_key": key, "dataset": ds["catalog_id"], "method": method, "feature_tag": tag,
                "family": g["family"].iloc[0], "n_splits": len(f1),
                "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1) if len(f1) > 1 else np.nan,
                "acc_mean": g["accuracy"].mean(), "n_features": g["n_features"].iloc[0],
                "best_bag": f"{best[0]}__{best[1]}" if best else None,
            }
            for name, ref in (("node", node_f1), ("bestbag", bag_f1), ("floor", floor)):
                if len(ref) == len(f1) and len(f1) >= 3:
                    t, p, w = _paired(f1, ref)
                    row[f"diff_{name}"] = float(f1.mean() - ref.mean())
                    row[f"t_{name}"] = t
                    row[f"p_{name}"] = p
                    row[f"wilcoxon_p_{name}"] = w
                else:
                    row[f"diff_{name}"] = row[f"t_{name}"] = row[f"p_{name}"] = row[f"wilcoxon_p_{name}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _lifts(sep, method, tag, B):
    ok = sep[(sep.status == "ok") & (sep.buffer_hops == B)]
    m = ok[(ok.method == method) & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    f = ok[(ok.method == f"permuted_{method}") & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    idx = m.index.intersection(f.index)
    return (m.loc[idx] - f.loc[idx]).sort_index()


def holdout_stats(comb, shared):
    rows = []
    for key, buffers in HOLDOUT_BUFFERS.items():
        target = {"synth_b0": "synth_emb", "synth_b1": "synth_emb"}.get(key, f"{key}_emb")
        path = SEP_DIR / f"separation_{target}.csv"
        base_path = SEP_DIR / f"separation_{'synth' if key.startswith('synth') else key}.csv"
        if not path.exists():
            continue
        sep = pd.read_csv(path)
        base = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()
        ds = C.DATASETS[key]
        if key.startswith("synth"):
            sep = sep[sep.dataset == ds["catalog_id"]]
            base = base[base.dataset == ds["catalog_id"]] if not base.empty else base
        best = ds["expected_best_bag"]
        sh = shared[shared.dataset_key == key]
        for (method, tag), _ in sep[~sep.method.str.startswith("permuted_")].groupby(["method", "feature_tag"]):
            shared_lift = sh.loc[(sh.method == method) & (sh.feature_tag == tag), "diff_floor"]
            shared_lift = float(shared_lift.iloc[0]) if len(shared_lift) else np.nan
            for B in buffers:
                lifts = _lifts(sep, method, tag, B)
                row = {"dataset_key": key, "dataset": ds["catalog_id"], "method": method, "feature_tag": tag, "buffer_hops": B,
                       "n_ok": int(len(lifts)), "lift_mean": float(lifts.mean()) if len(lifts) else np.nan,
                       "lift_std": float(lifts.std(ddof=1)) if len(lifts) > 1 else np.nan,
                       "shared_lift": shared_lift,
                       "retention": float(lifts.mean() / shared_lift) if len(lifts) and shared_lift and shared_lift > 0 else np.nan}
                if len(lifts) >= 3:
                    t, p = stats.ttest_1samp(lifts.to_numpy(), 0.0)
                    row["t_vs_zero"], row["p_vs_zero"] = float(t), float(p)
                else:
                    row["t_vs_zero"] = row["p_vs_zero"] = np.nan
                if best and not base.empty:
                    bag = _lifts(base, best[0], best[1], B)
                    idx = lifts.index.intersection(bag.index)
                    if len(idx) >= 3:
                        t, p, w = _paired(lifts.loc[idx], bag.loc[idx])
                        row["diff_bestbag_lift"] = float((lifts.loc[idx] - bag.loc[idx]).mean())
                        row["p_bestbag_lift"] = p
                    else:
                        row["diff_bestbag_lift"] = row["p_bestbag_lift"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    comb = pd.read_csv(OUT / "combined_results.csv")
    shared = shared_split_stats(comb)
    shared.to_csv(OUT / "stats_shared_splits.csv", index=False)
    print(shared[["dataset_key", "method", "f1_mean", "f1_std", "diff_node", "p_node", "diff_bestbag", "p_bestbag", "diff_floor"]].round(3).to_string(index=False))
    hold = holdout_stats(comb, shared)
    if not hold.empty:
        hold.to_csv(OUT / "stats_region_holdout.csv", index=False)
        print(hold[["dataset_key", "method", "buffer_hops", "n_ok", "lift_mean", "shared_lift", "retention", "p_vs_zero"]].round(3).to_string(index=False))
    else:
        print("no region-holdout CSVs yet")


if __name__ == "__main__":
    main()
