"""Phase-1 figures, regenerated exclusively from saved CSVs.

Reads outputs/results.csv (per-split rows + config columns) and the per-run
bag tables; writes figures/*.png. Rerunnable at any time; no experiment
state is consulted.

Figures:
  detectability_<family>_<anomaly>.png  AUC vs prevalence, k=1 vs k=2 lines,
                                        one line per representation
  dilution.png                          AUC vs median bag size (all cells)
  saturation.png                        empirical bag positive rate vs the
                                        1-(1-pi)^s uniform-placement curve
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"

REPS = ["wasserstein", "pooled_all", "pooled_max", "size_only", "node_oracle"]
REP_STYLE = {
    "wasserstein": dict(color="#4269d0", marker="o"),
    "pooled_all": dict(color="#efb118", marker="s"),
    "pooled_max": dict(color="#ff725c", marker="^"),
    "size_only": dict(color="#9c9c9c", marker="x", linestyle="--"),
    "node_oracle": dict(color="#3ca951", marker="d", linestyle=":"),
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    ok = df[df["status"] == "ok"].copy()
    return (
        ok.groupby(["family", "anomaly", "prevalence", "k_hop", "representation"])
        .agg(
            roc_auc=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            positive_rate=("positive_rate", "first"),
            bag_nodes_median=("bag_nodes_median", "first"),
        )
        .reset_index()
    )


def detectability_panels(agg: pd.DataFrame):
    for (family, anomaly), cell in agg.groupby(["family", "anomaly"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, k in zip(axes, sorted(cell["k_hop"].unique())):
            sub = cell[cell["k_hop"] == k]
            for rep in REPS:
                line = sub[sub["representation"] == rep].sort_values("prevalence")
                if line.empty:
                    continue
                ax.errorbar(
                    line["prevalence"], line["roc_auc"], yerr=line["roc_auc_std"],
                    label=rep, capsize=2, **REP_STYLE[rep],
                )
            ax.axhline(0.5, color="k", lw=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("prevalence π")
            ax.set_title(f"{family.upper()} / {anomaly}, k={k}")
        axes[0].set_ylabel("bag-level ROC-AUC")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / f"detectability_{family}_{anomaly}.png", dpi=150)
        plt.close(fig)


def dilution_plot(agg: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rep in ["wasserstein", "pooled_max", "size_only"]:
        sub = agg[agg["representation"] == rep]
        ax.scatter(sub["bag_nodes_median"], sub["roc_auc"], label=rep, alpha=0.7, **{
            k: v for k, v in REP_STYLE[rep].items() if k in ("color", "marker")
        })
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("median bag size (nodes)")
    ax.set_ylabel("bag-level ROC-AUC")
    ax.set_title("Detection vs bag size across all cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "dilution.png", dpi=150)
    plt.close(fig)


def saturation_plot():
    rows = []
    for run_dir in sorted(OUTPUTS.iterdir()):
        cfg_path, table_path = run_dir / "config.json", run_dir / "bag_table.csv"
        if not (cfg_path.exists() and table_path.exists()):
            continue
        import json

        cfg = json.loads(cfg_path.read_text())
        table = pd.read_csv(table_path)
        rows.append(
            {
                "anomaly": cfg["anomaly"],
                "prevalence": cfg["prevalence"],
                "mean_size": table["n_nodes"].mean(),
                "pos_rate": table["y_contains"].mean(),
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"hub": "o", "clique": "s", "tail": "^"}
    for anomaly, sub in df.groupby("anomaly"):
        x = sub["prevalence"] * sub["mean_size"]
        ax.scatter(x, sub["pos_rate"], label=f"{anomaly} (empirical)", marker=markers[anomaly], alpha=0.75)
    grid = np.logspace(-2, 1.5, 100)
    ax.plot(grid, 1 - np.exp(-grid), "k--", lw=1, label="uniform placement: $1-e^{-\\pi s}$")
    ax.set_xscale("log")
    ax.set_xlabel("π × mean bag size")
    ax.set_ylabel("bag positive rate")
    ax.set_title("Saturation: empirical vs uniform-placement prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "saturation.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    agg = load_results()
    detectability_panels(agg)
    dilution_plot(agg)
    saturation_plot()
    print(f"Figures written to {FIGURES}")
    if "--summary" in sys.argv:
        pivot = agg.pivot_table(index=["family", "anomaly", "prevalence", "k_hop"], columns="representation", values="roc_auc")
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
