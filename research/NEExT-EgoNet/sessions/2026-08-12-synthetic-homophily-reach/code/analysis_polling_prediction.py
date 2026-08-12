"""Ring-by-ring test of the polling mechanism.

From the measured distance-homophily curve alone (no classifier data), predict
the SHAPE of the k-profile: a bag that averages its members' whispers has
polling signal

    S(k) = sum_{d<=k} n_d * e_d / sqrt(sum_{d<=k} n_d)

where n_d = mean ring population at distance d and e_d = measured label-
agreement excess at d (signed — anti-correlated rings partially cancel, which
is what naive averaging does). Compare S(k), normalized per world, against the
observed global whisper-ON profile (mean macro-F1 minus the permutation
floor), also normalized. If polling is the mechanism, the shapes should match
— including where they flatten.

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

N_SOURCES = 800  # manipulation_checks.py sampled this many BFS sources
WORLDS = {
    "SYNTH_HR_L1_B1": "small patches ($\\lambda$=0.10)",
    "SYNTH_HR_L2_B1": "medium patches ($\\lambda$=0.20)",
    "SYNTH_HR_L3_B1": "large patches ($\\lambda$=0.35)",
}


def normalize(v):
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min())


def main():
    curves = pd.read_csv(SESSION_ROOT / "outputs" / "distance_homophily_curves.csv")
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df.status == "ok"]

    ps.use_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    report = []
    for ax, (cell, title) in zip(axes, WORLDS.items()):
        cv = curves[curves.dataset == cell].sort_values("d")
        n_d = (cv.n_pairs / N_SOURCES).to_numpy()
        e_d = cv.excess.to_numpy()
        ks = np.arange(1, 7)
        # Naive polling: equal-weight averaging of every bag member — dilution
        # (sqrt N) fights the saturating signal, so this predicts a DECLINE.
        pred_naive = np.array([np.sum(n_d[:k] * e_d[:k]) / np.sqrt(np.sum(n_d[:k])) for k in ks])
        # Adaptive polling: an aggregator that weights rings by informativeness
        # (matched filter) — extra uninformative mass is ignored, so the curve
        # saturates where e_d dies instead of declining.
        pred_adapt = np.array([np.sqrt(np.sum(n_d[:k] * e_d[:k] ** 2)) for k in ks])

        obs_rows = df[(df.dataset == cell) & (df.feature_tag == "fall-vl1-glob")]
        obs = np.array([obs_rows[obs_rows.method == f"egonet_k{k}_wass"]["f1_macro"].mean() for k in ks])
        floor = df[(df.dataset == cell) & (df.method == "permuted") & (df.feature_tag == "fall-vl1")]["f1_macro"].mean()
        obs_lift = obs - floor

        r_naive = float(np.corrcoef(pred_naive, obs_lift)[0, 1])
        r_adapt = float(np.corrcoef(pred_adapt, obs_lift)[0, 1])
        report.append({"cell": cell, "r_naive": round(r_naive, 3), "r_adaptive": round(r_adapt, 3)})

        ax.plot(ks, normalize(pred_naive), color=ps.MUTED, lw=1.4, linestyle=(0, (2, 2)),
                marker="^", ms=4, fillstyle="none", label="Predicted: naive averaging")
        ax.plot(ks, normalize(pred_adapt), color=ps.INK, lw=1.4, linestyle=(0, (5, 2)),
                marker="s", ms=4, fillstyle="none", label="Predicted: adaptive weighting")
        ax.plot(ks, normalize(obs_lift), color=ps.FAMILY_COLOR["structural"], lw=1.8,
                marker="o", ms=4.5, label="Observed (global, whisper ON)")
        ax.set_title(f"{title}\n$r$: adaptive {r_adapt:.2f}, naive {r_naive:.2f}", fontsize=10)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel("profile shape\n(normalized 0–1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    w, h = fig.get_size_inches()
    ps.save(fig, SESSION_ROOT / "figures" / "polling_prediction")
    print(f"aspect {w / h:.3f}")
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    main()
