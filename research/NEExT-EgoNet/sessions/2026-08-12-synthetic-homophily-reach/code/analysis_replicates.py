"""Replicate-averaged mechanism test (Ash-approved 10-replicate study).

Per wavelength: 10 independent worlds (seeds 101-110), each with its own
measured agreement curve and its own global whisper-ON sweep. The figure
shows, in REAL macro-F1 lift units (observed minus that replicate's own
permutation floor):

- orange line + band: mean lift across replicates, ±1 std across replicates
- black dashed: the adaptive-polling prediction sqrt(sum n_d e_d^2) built
  from the replicate-averaged agreement curves, affine-fitted to the lift
  scale (r is affine-invariant; the fit only places the curve on the axis)
- gray dotted: the naive-averaging prediction, same treatment

Also prints the small-patch dip verdict: paired test across replicates of
lift(k4) - lift(k3).

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

N_SOURCES = 800
SEEDS = list(range(101, 111))
WORLDS = {"L1": "small patches ($\\lambda$=0.10)", "L2": "medium patches ($\\lambda$=0.20)", "L3": "large patches ($\\lambda$=0.35)"}
KS = np.arange(1, 7)


def predictions(curves, cells):
    """Replicate-averaged ring populations/excess -> (naive, adaptive) shapes."""
    n_stack, e_stack = [], []
    for cell in cells:
        cv = curves[curves.dataset == cell].sort_values("d")
        n_stack.append((cv.n_pairs / N_SOURCES).to_numpy()[:10])
        e_stack.append(cv.excess.to_numpy()[:10])
    n_d = np.mean(n_stack, axis=0)
    e_d = np.mean(e_stack, axis=0)
    naive = np.array([np.sum(n_d[:k] * e_d[:k]) / np.sqrt(np.sum(n_d[:k])) for k in KS])
    adapt = np.array([np.sqrt(np.sum(n_d[:k] * e_d[:k] ** 2)) for k in KS])
    return naive, adapt


def affine_fit(pred, target):
    a, b = np.polyfit(pred, target, 1)
    return a * pred + b


def main():
    curves = pd.read_csv(SESSION_ROOT / "outputs" / "distance_homophily_curves.csv")
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df.status == "ok"]

    ps.use_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    report = []
    lifts_by_L = {}
    for ax, (L, title) in zip(axes, WORLDS.items()):
        cells = [f"SYNTH_HR_{L}_B1_S{s}" for s in SEEDS]
        lifts = []
        for cell in cells:
            rows = df[(df.dataset == cell) & (df.feature_tag == "fall-vl1-glob")]
            floor = rows[rows.method == "permuted"]["f1_macro"].mean()
            lifts.append([rows[rows.method == f"egonet_k{k}_wass"]["f1_macro"].mean() - floor for k in KS])
        lifts = np.array(lifts)
        lifts_by_L[L] = lifts
        mean, std = lifts.mean(axis=0), lifts.std(axis=0)

        naive, adapt = predictions(curves, cells)
        r_naive = float(np.corrcoef(naive, mean)[0, 1])
        r_adapt = float(np.corrcoef(adapt, mean)[0, 1])
        report.append({"L": L, "r_naive": round(r_naive, 3), "r_adaptive": round(r_adapt, 3)})

        ax.fill_between(KS, mean - std, mean + std, color=ps.FAMILY_COLOR["structural"], alpha=0.18, linewidth=0)
        ax.plot(KS, mean, color=ps.FAMILY_COLOR["structural"], lw=1.8, marker="o", ms=4.5,
                label="Observed mean (10 worlds) $\\pm$1 std")
        ax.plot(KS, affine_fit(adapt, mean), color=ps.INK, lw=1.4, linestyle=(0, (5, 2)),
                marker="s", ms=4, fillstyle="none", label="Predicted: adaptive weighting")
        ax.plot(KS, affine_fit(naive, mean), color=ps.MUTED, lw=1.2, linestyle=(0, (2, 2)),
                marker="^", ms=4, fillstyle="none", label="Predicted: naive averaging")
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_title(f"{title}\n$r$: adaptive {r_adapt:.2f}, naive {r_naive:.2f}", fontsize=10)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel("macro-F1 lift over floor")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    w, h = fig.get_size_inches()
    ps.save(fig, SESSION_ROOT / "figures" / "polling_prediction_replicated")
    print(f"aspect {w / h:.3f}")
    print(pd.DataFrame(report).to_string(index=False))

    d = lifts_by_L["L1"][:, 3] - lifts_by_L["L1"][:, 2]  # k4 - k3 per replicate
    t, p = stats.ttest_rel(lifts_by_L["L1"][:, 3], lifts_by_L["L1"][:, 2])
    print(f"\nL1 dip verdict (k4 vs k3 across 10 worlds): mean diff {d.mean():+.4f}, std {d.std():.4f}, p={p:.3f}")


if __name__ == "__main__":
    main()
