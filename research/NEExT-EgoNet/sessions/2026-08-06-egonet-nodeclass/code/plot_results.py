"""Benchmark figures — regenerated from outputs/ artifacts ONLY.

F1  fig_per_dataset      3x3 panels: per-method macro-F1 dot+CI, one panel per dataset
F2  fig_margin_heatmap   dataset x method margin over best non-egonet baseline
F3  fig_cost_frontier    wall-clock vs normalized quality, Pareto front
F4  fig_taxonomy_verdict best-of-family dumbbells grouped by label type

Run: uv run --with matplotlib python code/plot_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps  # noqa: E402

ps.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"
METRIC = "f1_macro"

# Display order + labels. sizeonly/majority are floors used as reference lines.
METHOD_ORDER = [
    ("hop_k1__wass16", "hop $k{=}1$ · Wass"),
    ("hop_k1__pooled", "hop $k{=}1$ · pooled"),
    ("hop_k2__wass16", "hop $k{=}2$ · Wass"),
    ("hop_k2__pooled", "hop $k{=}2$ · pooled"),
    ("walk_default__wass16", "walk · Wass"),
    ("walk_default__pooled", "walk · pooled"),
    ("walk_tight__wass16", "walk tight · Wass"),
    ("walk_tight__pooled", "walk tight · pooled"),
    ("walk_wide__wass16", "walk wide · Wass"),
    ("walk_wide__pooled", "walk wide · pooled"),
    ("deepwalk", "DeepWalk"),
    ("node2vec", "node2vec"),
    ("node2vec_bfs", "node2vec (BFS)"),
    ("walklets", "Walklets"),
    ("graphwave", "GraphWave"),
    ("role2vec", "Role2Vec"),
    ("center_struct", "Structural (full graph)"),
    ("degree_only", "Degree + core"),
]
FAMILY_OF = {}
for m, _ in METHOD_ORDER:
    if m.startswith("hop_"):
        FAMILY_OF[m] = "egonet_hop"
    elif m.startswith("walk_"):
        FAMILY_OF[m] = "egonet_walk"
    elif m in ("center_struct",):
        FAMILY_OF[m] = "structural"
    elif m in ("degree_only",):
        FAMILY_OF[m] = "floor"
    else:
        FAMILY_OF[m] = "kc_baseline"
KC_METHODS = [m for m, _ in METHOD_ORDER if FAMILY_OF[m] == "kc_baseline"]
EGONET_METHODS = [m for m, _ in METHOD_ORDER if FAMILY_OF[m].startswith("egonet")]
LABEL_TYPE_ORDER = ["structural-role", "local-property", "community", "outlier"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    return df[df["status"] == "ok"].copy()


def load_configs() -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(OUTPUTS.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        timings = cfg.get("timings", {})
        rows.append(
            {
                "run_id": cfg_path.parent.name,
                "dataset": cfg.get("dataset"),
                "method": cfg.get("method"),
                "family": cfg.get("family"),
                "status": cfg.get("status"),
                "label_type": cfg.get("label_type"),
                "seconds_total": float(sum(v for k, v in timings.items() if isinstance(v, (int, float)))),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method): mean/std of the primary metric + majority floor."""
    agg = (
        df.groupby(["dataset", "label_type", "method"])
        .agg(mean=(METRIC, "mean"), std=(METRIC, "std"), n_features=("n_features", "first"))
        .reset_index()
    )
    return agg


def dataset_order(summary: pd.DataFrame) -> list:
    ds = summary[["dataset", "label_type"]].drop_duplicates()
    ds["type_rank"] = ds["label_type"].map({t: i for i, t in enumerate(LABEL_TYPE_ORDER)})
    return ds.sort_values(["type_rank", "dataset"])["dataset"].tolist()


def guarded_cells(configs: pd.DataFrame) -> set:
    return set(configs.loc[configs["status"] == "guarded", "run_id"])


def fig_per_dataset(summary, datasets, guarded):
    n = len(datasets)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ps.FULL_W, 2.3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for idx, ds in enumerate(datasets):
        ax = axes.flat[idx]
        sub = summary[summary["dataset"] == ds].set_index("method")
        ys, xs, errs, colors = [], [], [], []
        labels = []
        for row_i, (m, label) in enumerate(METHOD_ORDER):
            labels.append(label)
            ys.append(len(METHOD_ORDER) - 1 - row_i)
            if m in sub.index:
                xs.append(sub.loc[m, "mean"])
                errs.append(sub.loc[m, "std"])
                colors.append(ps.FAMILY_COLOR[FAMILY_OF[m]])
            elif f"{ds}__{m}" in guarded:
                xs.append(np.nan)
                errs.append(0)
                colors.append(ps.MUTED)
                ax.text(0.02, len(METHOD_ORDER) - 1 - row_i, "guarded", fontsize=5.5, color=ps.MUTED, va="center")
            else:
                xs.append(np.nan)
                errs.append(0)
                colors.append(ps.MUTED)
        xs, errs = np.array(xs), np.array(errs)
        for y, x, e, c in zip(ys, xs, errs, colors):
            if np.isfinite(x):
                ax.errorbar(x, y, xerr=e, fmt="o", color=c, ecolor=c, elinewidth=0.8, capsize=1.5, markersize=2.8)
        maj = summary[(summary["dataset"] == ds) & (summary["method"] == "majority")]
        if len(maj):
            ax.axvline(maj["mean"].iloc[0], color=ps.MUTED, linestyle=(0, (1, 1.6)), linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels(reversed(labels) if False else [label for _, label in reversed(METHOD_ORDER)], fontsize=5.5)
        if idx % ncols != 0:
            ax.set_yticklabels([])
        label_type = summary.loc[summary["dataset"] == ds, "label_type"].iloc[0]
        ax.set_title(f"({chr(97 + idx)}) {ds}\n{label_type}", loc="left", fontsize=6.5, pad=3)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color=ps.GRID, linewidth=0.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("macro-F1")
    handles = [
        Line2D([], [], color=ps.FAMILY_COLOR[f], marker="o", linestyle="", markersize=3, label=ps.FAMILY_LABEL[f])
        for f in ("egonet_hop", "egonet_walk", "kc_baseline", "structural", "floor")
    ]
    handles.append(Line2D([], [], color=ps.MUTED, linestyle=(0, (1, 1.6)), label="Majority floor"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 0.99))
    ps.save(fig, FIGURES / "fig_per_dataset")


def fig_margin_heatmap(summary, datasets, guarded):
    methods = EGONET_METHODS + KC_METHODS + ["center_struct"]
    mat = np.full((len(datasets), len(methods)), np.nan)
    best_base = {}
    for i, ds in enumerate(datasets):
        sub = summary[summary["dataset"] == ds].set_index("method")
        baselines = [m for m in KC_METHODS + ["center_struct", "degree_only"] if m in sub.index]
        if not baselines:
            continue
        best = max(baselines, key=lambda m: sub.loc[m, "mean"])
        best_base[ds] = (best, sub.loc[best, "mean"])
        for j, m in enumerate(methods):
            if m in sub.index:
                mat[i, j] = sub.loc[m, "mean"] - sub.loc[best, "mean"]
    lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 0.1
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 0.34 * len(datasets) + 1.4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i, ds in enumerate(datasets):
        for j, m in enumerate(methods):
            if not np.isfinite(mat[i, j]):
                hatch_reason = "guarded" if f"{ds}__{m}" in guarded else "—"
                ax.text(j, i, "▨" if hatch_reason == "guarded" else "", ha="center", va="center", fontsize=6, color=ps.MUTED)
            else:
                ax.text(
                    j, i, f"{mat[i, j]:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=5,
                    color="white" if abs(mat[i, j]) > 0.55 * lim else ps.INK,
                )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([dict(METHOD_ORDER)[m] for m in methods], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(datasets)))
    labels = []
    for ds in datasets:
        lt = summary.loc[summary["dataset"] == ds, "label_type"].iloc[0]
        best, val = best_base.get(ds, ("?", np.nan))
        labels.append(f"{ds}  [{lt}]  (best base: {dict(METHOD_ORDER).get(best, best)} {val:.2f})")
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(visible=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("macro-F1 margin over best non-egonet baseline", fontsize=6, labelpad=2)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 0.98, 1))
    ps.save(fig, FIGURES / "fig_margin_heatmap")


def fig_cost_frontier(summary, configs, datasets):
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 3.2))
    merged = summary.merge(configs[["dataset", "method", "seconds_total", "family"]], on=["dataset", "method"], how="left")
    pts = []
    for ds in datasets:
        sub = merged[merged["dataset"] == ds]
        maj = sub.loc[sub["method"] == "majority", "mean"]
        floor = maj.iloc[0] if len(maj) else 0.0
        best = sub["mean"].max()
        if best <= floor:
            continue
        for _, r in sub.iterrows():
            if r["method"] in ("majority",) or not np.isfinite(r.get("seconds_total", np.nan)):
                continue
            q = (r["mean"] - floor) / (best - floor)
            pts.append((r["seconds_total"], q, r["family"], r["method"]))
            ax.scatter(r["seconds_total"], q, s=8, color=ps.FAMILY_COLOR.get(r["family"], ps.MUTED), alpha=0.75, linewidths=0)
    if pts:
        pts.sort()
        front_x, front_y = [], []
        best_q = -np.inf
        for x, q, *_ in pts:
            if q > best_q:
                best_q = q
                front_x.append(x)
                front_y.append(q)
        ax.step(front_x, front_y, where="post", color=ps.INK, linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("total cell wall-clock (s, log)")
    ax.set_ylabel("normalized quality  $(F_1 - \\mathrm{floor})/(\\mathrm{best} - \\mathrm{floor})$")
    handles = [
        Line2D([], [], color=ps.FAMILY_COLOR[f], marker="o", linestyle="", markersize=3, label=ps.FAMILY_LABEL[f])
        for f in ("egonet_hop", "egonet_walk", "kc_baseline", "structural", "floor")
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_cost_frontier")


def fig_taxonomy_verdict(summary, datasets):
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 0.42 * len(datasets) + 1.2))
    rows = []
    for ds in datasets:
        sub = summary[summary["dataset"] == ds].set_index("method")
        ego = [m for m in EGONET_METHODS if m in sub.index]
        kc = [m for m in KC_METHODS if m in sub.index]
        if not ego or not kc:
            continue
        best_ego = max(ego, key=lambda m: sub.loc[m, "mean"])
        best_kc = max(kc, key=lambda m: sub.loc[m, "mean"])
        struct = sub.loc["center_struct", "mean"] if "center_struct" in sub.index else np.nan
        rows.append(
            {
                "dataset": ds,
                "label_type": summary.loc[summary["dataset"] == ds, "label_type"].iloc[0],
                "ego": sub.loc[best_ego, "mean"],
                "ego_m": best_ego,
                "kc": sub.loc[best_kc, "mean"],
                "kc_m": best_kc,
                "struct": struct,
            }
        )
    for i, r in enumerate(reversed(rows)):
        lo, hi = min(r["ego"], r["kc"]), max(r["ego"], r["kc"])
        ax.plot([lo, hi], [i, i], color=ps.AXIS, linewidth=1.0, zorder=1)
        ax.scatter(r["kc"], i, color=ps.FAMILY_COLOR["kc_baseline"], s=34, zorder=3, marker="D")
        ax.scatter(r["ego"], i, color=ps.FAMILY_COLOR["egonet_hop" if r["ego_m"].startswith("hop") else "egonet_walk"],
                   s=16, zorder=4, marker="o")
        if np.isfinite(r["struct"]):
            ax.scatter(r["struct"], i, color=ps.FAMILY_COLOR["structural"], s=26, zorder=2, marker="|")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['dataset']}  [{r['label_type']}]" for r in reversed(rows)], fontsize=6.5)
    ax.set_xlabel("macro-F1 (best of family)")
    handles = [
        Line2D([], [], color=ps.FAMILY_COLOR["egonet_hop"], marker="o", linestyle="", markersize=4, label="Best egonet"),
        Line2D([], [], color=ps.FAMILY_COLOR["kc_baseline"], marker="D", linestyle="", markersize=4, label="Best node embedding"),
        Line2D([], [], color=ps.FAMILY_COLOR["structural"], marker="|", linestyle="", markersize=6, label="Structural features"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    ax.grid(axis="x", color=ps.GRID, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_taxonomy_verdict")


def main():
    FIGURES.mkdir(exist_ok=True)
    df = load_results()
    configs = load_configs()
    summary = summarize(df)
    datasets = dataset_order(summary)
    guarded = guarded_cells(configs)
    fig_per_dataset(summary, datasets, guarded)
    fig_margin_heatmap(summary, datasets, guarded)
    fig_cost_frontier(summary, configs, datasets)
    fig_taxonomy_verdict(summary, datasets)
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
