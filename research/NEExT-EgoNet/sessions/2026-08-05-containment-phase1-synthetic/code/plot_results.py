"""Phase-1 figures, regenerated exclusively from saved CSVs.

Publication style (lib.containment.plotstyle): vector PDF sized to the
manuscript text width, serif type, validated colors, no in-figure titles
(captions carry them). A 300-dpi PNG twin of each figure is written for
quick visual inspection.

Figures:
  detectability_<family>_<anomaly>.{pdf,png}  two panels (k=1 | k=2):
      AUC vs prevalence per representation, +-1 sd bands
  dilution.{pdf,png}    AUC vs median bag size; cluster medians joined
  saturation.{pdf,png}  bag positive rate vs pi*s and the uniform curve
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

OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"
REPS = ["wasserstein", "pooled_all", "pooled_max", "size_only", "node_oracle"]
PREVALENCES = [0.005, 0.01, 0.02, 0.05, 0.10]


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


def _prevalence_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(PREVALENCES)
    ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"prevalence $\pi$")


def _plot_series(ax, sub, rep, band=True):
    line = sub[sub["representation"] == rep].sort_values("prevalence")
    if line.empty:
        return
    kwargs = dict(
        color=ps.COLOR[rep],
        marker=ps.MARKER[rep] or None,
        linestyle=ps.LINESTYLE[rep],
        markersize=3.5,
        markeredgewidth=0.9 if rep == "size_only" else 0,
        linewidth=1.0 if rep in ("size_only", "node_oracle") else 1.2,
        zorder=3 if rep not in ("size_only", "node_oracle") else 2,
    )
    ax.plot(line["prevalence"], line["roc_auc"], **kwargs)
    if band and rep not in ("node_oracle", "size_only"):
        sd = line["roc_auc_std"].fillna(0)
        ax.fill_between(
            line["prevalence"], line["roc_auc"] - sd, line["roc_auc"] + sd,
            color=ps.COLOR[rep], alpha=0.12, linewidth=0, zorder=1,
        )


def detectability_panels(agg: pd.DataFrame):
    for (family, anomaly), cell in agg.groupby(["family", "anomaly"]):
        fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.55), sharey=True)
        ymin = 1.0
        for ax, k, tag in zip(axes, sorted(cell["k_hop"].unique()), "ab"):
            sub = cell[cell["k_hop"] == k]
            for rep in REPS:
                _plot_series(ax, sub, rep)
            shown = sub[sub["representation"] != "node_oracle"]["roc_auc"]
            if len(shown):
                ymin = min(ymin, float(shown.min()) - 0.04)
            ax.axhline(0.5, color=ps.AXIS, lw=0.6, zorder=0)
            _prevalence_axis(ax)
            ps.panel_tag(ax, f"({tag}) $k={k}$")
        ymin = min(ymin, 0.47)
        axes[0].set_ylim(ymin, 1.03)
        axes[0].set_ylabel("bag-level ROC-AUC")
        axes[1].text(
            PREVALENCES[-1], 0.502, "chance", ha="right", va="bottom", fontsize=6.5, color=ps.MUTED
        )
        handles = [
            plt.Line2D(
                [], [], color=ps.COLOR[r], marker=ps.MARKER[r] or None,
                linestyle=ps.LINESTYLE[r], markersize=3.5,
                markeredgewidth=0.9 if r == "size_only" else 0, label=ps.LABEL[r],
            )
            for r in REPS
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=5, columnspacing=1.1, handlelength=1.9)
        fig.subplots_adjust(wspace=0.08)
        ps.save(fig, FIGURES / f"detectability_{family}_{anomaly}")


def dilution_plot(agg: pd.DataFrame):
    """Tail (point-anomaly) cells only: dilution is the claim, and mixing in
    cliques (which improve with radius) would average it away. Each thin line
    joins one (family, prevalence) cell's k=1 point to its k=2 point; bold
    lines are per-representation means."""
    reps = ["wasserstein", "pooled_max", "size_only"]
    tail = agg[agg["anomaly"] == "tail"]
    fig, ax = plt.subplots(figsize=(ps.HALF_W, 2.5))
    for rep in reps:
        sub = tail[tail["representation"] == rep]
        for _, pair in sub.groupby(["family", "prevalence"]):
            pair = pair.sort_values("k_hop")
            if len(pair) == 2:
                ax.plot(
                    pair["bag_nodes_median"], pair["roc_auc"], color=ps.COLOR[rep],
                    lw=0.6, alpha=0.28, zorder=1,
                )
        mean = sub.groupby("k_hop").agg(x=("bag_nodes_median", "median"), y=("roc_auc", "mean")).sort_values("x")
        ax.plot(
            mean["x"], mean["y"], color=ps.COLOR[rep], marker=ps.MARKER[rep],
            markersize=5, linewidth=1.6, label=ps.LABEL[rep],
            markeredgewidth=1.4 if rep == "size_only" else 0,
            zorder=3,
        )
    ax.axhline(0.5, color=ps.AXIS, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100])
    ax.set_xticklabels(["10", "30", "100"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("median bag size (nodes)")
    ax.set_ylabel("bag-level ROC-AUC")
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="lower left", handlelength=1.8, borderaxespad=0.2)
    ps.save(fig, FIGURES / "dilution")


def saturation_plot():
    rows = []
    for run_dir in sorted(OUTPUTS.iterdir()):
        cfg_path, table_path = run_dir / "config.json", run_dir / "bag_table.csv"
        if not (cfg_path.exists() and table_path.exists()):
            continue
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
    fig, ax = plt.subplots(figsize=(ps.HALF_W, 2.5))
    grid = np.logspace(-1.5, 1.8, 200)
    ax.plot(grid, 1 - np.exp(-grid), color=ps.INK, linestyle=(0, (4, 2)), lw=0.9, zorder=1)
    ax.annotate(
        r"$1-e^{-\pi s}$", xy=(0.16, 1 - np.exp(-0.16)), xytext=(0.05, 0.42),
        fontsize=7, color=ps.INK,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.MUTED, shrinkA=2, shrinkB=1),
    )
    style = {"hub": ("#2a78d6", "o"), "clique": ("#eb6834", "s"), "tail": ("#1baf7a", "^")}
    for anomaly in ["hub", "clique", "tail"]:
        sub = df[df["anomaly"] == anomaly]
        color, marker = style[anomaly]
        ax.scatter(
            sub["prevalence"] * sub["mean_size"], sub["pos_rate"], s=11, alpha=0.8,
            color=color, marker=marker, linewidths=0, label=anomaly, zorder=2,
        )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$\pi \times$ mean bag size")
    ax.set_ylabel("bag positive rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.2)
    ps.save(fig, FIGURES / "saturation")


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
