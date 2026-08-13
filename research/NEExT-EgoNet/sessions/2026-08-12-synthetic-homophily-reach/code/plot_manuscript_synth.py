"""Manuscript figures for the Synthetic Mechanism Validation section.

1. synth_construction — one row: the three wavelength fields (real generator
   fields, RdBu, scattered nodes) + the beta mechanism zoom (exaggerated,
   labeled as such).
2. synth_prediction — the replicated ring-by-ring prediction figure (same
   content as the board version; regenerated here so the manuscript figure
   is produced by one script).
3. synth_profiles — six-world side-by-side k-profiles; each box = the 10
   replicate means (world-level variance). Requires the full battery; the
   script skips it with a notice until all four series have 10 replicates.

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps
from lib.nodeclass.synthetic import _smooth_field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from datasets import WAVELENGTHS

SEEDS = list(range(101, 111))
KS = np.arange(1, 7)
N_SOURCES = 800
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
WORLD_TITLES = {"L1": "small patches ($\\lambda$=0.10)", "L2": "medium patches ($\\lambda$=0.20)", "L3": "large patches ($\\lambda$=0.35)"}


def save(fig, name):
    ps.save(fig, SESSION_ROOT / "figures" / name)
    print("saved", name)


def construction_figure():
    rng_pts = np.random.RandomState(7)
    pts = rng_pts.uniform(0, 1, (420, 2))
    gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (L, lam) in zip(axes[:3], WAVELENGTHS.items()):
        rng = np.random.RandomState(13)
        f_grid = _smooth_field(grid, lam, rng).reshape(gx.shape)
        rng = np.random.RandomState(13)
        f_pts = _smooth_field(pts, lam, rng)
        ax.imshow(f_grid, extent=(0, 1, 0, 1), origin="lower", cmap="RdBu_r", alpha=0.5)
        ax.scatter(pts[:, 0], pts[:, 1], c=f_pts, cmap="RdBu_r", s=5, edgecolors="k", linewidths=0.15)
        ax.set_title(WORLD_TITLES[L], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # beta mechanism zoom: exaggerated coupling so the radius difference is visible
    ax = axes[3]
    lam, BETA_SHOW, r0 = 0.35, 0.45, 0.075
    rng = np.random.RandomState(13)
    f_grid = _smooth_field(grid, lam, rng).reshape(gx.shape)
    rng = np.random.RandomState(13)
    f_pts = _smooth_field(pts, lam, rng)
    radii = r0 * (1 + BETA_SHOW * np.clip(f_pts, -1.5, 1.5))
    ax.imshow(f_grid, extent=(0, 1, 0, 1), origin="lower", cmap="RdBu_r", alpha=0.25)
    ax.scatter(pts[:, 0], pts[:, 1], color="0.55", s=4, zorder=1)
    inner = (pts[:, 0] > 0.2) & (pts[:, 0] < 0.8) & (pts[:, 1] > 0.2) & (pts[:, 1] < 0.8)
    hi = np.where(inner)[0][np.argmax(f_pts[inner])]
    lo = np.where(inner)[0][np.argmin(f_pts[inner])]
    for idx, col in ((hi, "#c0392b"), (lo, "#2471a3")):
        r = radii[idx]
        d = np.hypot(pts[:, 0] - pts[idx, 0], pts[:, 1] - pts[idx, 1])
        for j in np.where((d < (r + radii) / 2) & (d > 0))[0]:
            ax.plot([pts[idx, 0], pts[j, 0]], [pts[idx, 1], pts[j, 1]], color=col, lw=0.7, zorder=2)
        ax.add_patch(plt.Circle(pts[idx], r, fill=False, color=col, lw=1.2, linestyle=(0, (4, 2)), zorder=3))
        ax.scatter(*pts[idx], color=col, s=32, zorder=4, edgecolors="k", linewidths=0.4)
    ax.set_title("coupling $r_i = r_0(1+\\beta f_i)$\n($\\beta$ exaggerated for display)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "synth_construction")


def _load_results():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    return df[df.status == "ok"]


def _lift_profiles(df, L, B, scope):
    """Per-replicate mean lift profile (mean over splits, minus that replicate's floor)."""
    tag = "fall-vl1-glob" if scope == "global" else "fall-vl1"
    out = []
    for s in SEEDS:
        rows = df[(df.dataset == f"SYNTH_HR_{L}_{B}_S{s}") & (df.feature_tag == tag)]
        if rows.empty:
            return None
        floor = rows[rows.method == "permuted"]["f1_macro"].mean()
        prof = [rows[rows.method == f"egonet_k{k}_wass"]["f1_macro"].mean() for k in KS]
        if any(pd.isna(v) for v in prof) or pd.isna(floor):
            return None
        out.append([v - floor for v in prof])
    return np.array(out)


def prediction_figure(df):
    curves = pd.read_csv(SESSION_ROOT / "outputs" / "distance_homophily_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (L, title) in zip(axes, WORLD_TITLES.items()):
        n_stack, e_stack = [], []
        for s in SEEDS:
            cv = curves[curves.dataset == f"SYNTH_HR_{L}_B1_S{s}"].sort_values("d")
            n_stack.append((cv.n_pairs / N_SOURCES).to_numpy()[:10])
            e_stack.append(cv.excess.to_numpy()[:10])
        n_d, e_d = np.mean(n_stack, axis=0), np.mean(e_stack, axis=0)
        naive = np.array([np.sum(n_d[:k] * e_d[:k]) / np.sqrt(np.sum(n_d[:k])) for k in KS])
        adapt = np.array([np.sqrt(np.sum(n_d[:k] * e_d[:k] ** 2)) for k in KS])

        lifts = _lift_profiles(df, L, "B1", "global")
        mean, std = lifts.mean(axis=0), lifts.std(axis=0)

        def place(pred):
            a = np.cov(pred, mean)[0, 1] / np.var(pred)
            a = max(0.0, a)
            return a * pred + (mean.mean() - a * pred.mean())

        r_naive = float(np.corrcoef(naive, mean)[0, 1])
        r_adapt = float(np.corrcoef(adapt, mean)[0, 1])
        ax.fill_between(KS, mean - std, mean + std, color=SCOPE_COLOR["global"], alpha=0.18, linewidth=0)
        ax.plot(KS, mean, color=SCOPE_COLOR["global"], lw=1.8, marker="o", ms=4.5, label="Observed mean (10 worlds) $\\pm$1 std")
        ax.plot(KS, place(adapt), color=ps.INK, lw=1.4, linestyle=(0, (5, 2)), marker="s", ms=4, fillstyle="none", label="Predicted: adaptive weighting")
        ax.plot(KS, place(naive), color=ps.MUTED, lw=1.2, linestyle=(0, (2, 2)), marker="^", ms=4, fillstyle="none", label="Predicted: naive averaging")
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_title(f"{title}\n$r$: adaptive {r_adapt:.2f}, naive {r_naive:.2f}", fontsize=10)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel("macro-F1 lift over floor")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "synth_prediction")


def profiles_figure(df):
    checks = pd.read_csv(SESSION_ROOT / "outputs" / "manipulation_checks.csv")
    reach = {
        L: checks[checks.dataset.str.match(f"SYNTH_HR_{L}_B1_S1")]["reach_zero_crossing"].mean()
        for L in WORLD_TITLES
    }
    offsets = {("global", "B1"): -0.27, ("global", "B0"): -0.09, ("local", "B1"): +0.09, ("local", "B0"): +0.27}
    style = {"B1": "-", "B0": (0, (4, 2))}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (L, title) in zip(axes, WORLD_TITLES.items()):
        for (scope, B), off in offsets.items():
            lifts = _lift_profiles(df, L, B, scope)
            if lifts is None:
                print(f"profiles_figure: missing replicates for {L} {B} {scope} — battery not finished; skipping figure")
                plt.close(fig)
                return False
            color = SCOPE_COLOR[scope]
            alpha = 0.4 if B == "B1" else 0.18
            for i, k in enumerate(KS):
                b = ax.boxplot([lifts[:, i]], positions=[k + off], widths=0.15, showfliers=False, patch_artist=True, zorder=3)
                b["boxes"][0].set_facecolor(color)
                b["boxes"][0].set_alpha(alpha)
                b["boxes"][0].set_edgecolor(color)
                b["medians"][0].set_color(ps.INK)
                for part in ("whiskers", "caps"):
                    for line in b[part]:
                        line.set_color(color)
                        line.set_alpha(0.4 if B == "B1" else 0.25)
            med = np.median(lifts, axis=0)
            ax.plot(KS + off, med, color=color, lw=1.2, linestyle=style[B], alpha=1.0 if B == "B1" else 0.55, zorder=2)
        ax.axhline(0, color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)))
        ax.axvline(reach[L], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([f"$k$={k}" for k in KS])
        ax.set_xlim(0.45, 6.55)
        ax.set_title(f"{title}, reach {reach[L]:.1f}", fontsize=10)
    axes[0].set_ylabel("macro-F1 lift over floor")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.4, edgecolor=SCOPE_COLOR["global"], label="Global (full-graph)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.4, edgecolor=SCOPE_COLOR["local"], label="Local (in-bag)"),
        plt.Line2D([], [], color=ps.INK, lw=1.4, label="coupling on ($\\beta$=0.03)"),
        plt.Line2D([], [], color=ps.INK, lw=1.4, linestyle=(0, (4, 2)), alpha=0.55, label="coupling off ($\\beta$=0)"),
        plt.Line2D([], [], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55, label="Measured reach"),
        plt.Line2D([], [], color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)), label="Permutation floor"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "synth_profiles")
    return True


def main():
    ps.use_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9})
    df = _load_results()
    construction_figure()
    prediction_figure(df)
    profiles_figure(df)


if __name__ == "__main__":
    main()
