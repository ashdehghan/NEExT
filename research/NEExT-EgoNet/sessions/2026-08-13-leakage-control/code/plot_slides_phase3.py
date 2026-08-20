"""Slide-size variants of the phase-2/3 figures for the beamer deck
(p_014c00f9), following the exp*s pattern: same house style and data as the
manuscript figures, physically smaller with larger fonts so they stay
legible at half/full-slide placement.

Renders into this session's figures/: synth_consts, synth_profs, morphs,
anchors, realdatas. Run from the repo root.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "sessions" / "2026-08-12-synthetic-homophily-reach" / "code"))

from lib.containment import plotstyle as ps
from lib.nodeclass.synthetic import _smooth_field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import WAVELENGTHS

SYNTH_SESSION = RESEARCH_ROOT / "sessions" / "2026-08-12-synthetic-homophily-reach"
FIG_DIR = SESSION_ROOT / "figures"
SEEDS = list(range(101, 111))
KS = np.arange(1, 7)
SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
WORLD_TITLES = {"L1": "small ($\\lambda$=0.10)", "L2": "medium ($\\lambda$=0.20)", "L3": "large ($\\lambda$=0.35)"}
BETA = {"B0": 0.0, "B1": 0.03, "B2": 0.10, "B3": 0.20, "B4": 0.40}
K_COLORS = {2: "#4c72b0", 4: "#dd8452", 6: "#55a868"}
BUFFERS = [0, 1, 2, 4, 6, 8, 10]


def save(fig, name):
    ps.save(fig, FIG_DIR / name)
    print("saved", name)


def _box(ax, values, position, color, width, alpha=0.4):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True, zorder=3, manage_ticks=False)
    b["boxes"][0].set_facecolor(color)
    b["boxes"][0].set_alpha(alpha)
    b["boxes"][0].set_edgecolor(color)
    b["medians"][0].set_color(ps.INK)
    for part in ("whiskers", "caps"):
        for line in b[part]:
            line.set_color(color)
            line.set_alpha(alpha)


def synth_consts():
    rng_pts = np.random.RandomState(7)
    pts = rng_pts.uniform(0, 1, (420, 2))
    gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))
    for ax, (L, lam) in zip(axes[:3], WAVELENGTHS.items()):
        rng = np.random.RandomState(13)
        f_grid = _smooth_field(grid, lam, rng).reshape(gx.shape)
        rng = np.random.RandomState(13)
        f_pts = _smooth_field(pts, lam, rng)
        ax.imshow(f_grid, extent=(0, 1, 0, 1), origin="lower", cmap="RdBu_r", alpha=0.5)
        ax.scatter(pts[:, 0], pts[:, 1], c=f_pts, cmap="RdBu_r", s=4, edgecolors="k", linewidths=0.1)
        ax.set_title(WORLD_TITLES[L], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[3]
    lam, BETA_SHOW, r0 = 0.35, 0.45, 0.075
    rng = np.random.RandomState(13)
    f_grid = _smooth_field(grid, lam, rng).reshape(gx.shape)
    rng = np.random.RandomState(13)
    f_pts = _smooth_field(pts, lam, rng)
    radii = r0 * (1 + BETA_SHOW * np.clip(f_pts, -1.5, 1.5))
    ax.imshow(f_grid, extent=(0, 1, 0, 1), origin="lower", cmap="RdBu_r", alpha=0.25)
    ax.scatter(pts[:, 0], pts[:, 1], color="0.55", s=3, zorder=1)
    inner = (pts[:, 0] > 0.2) & (pts[:, 0] < 0.8) & (pts[:, 1] > 0.2) & (pts[:, 1] < 0.8)
    hi = np.where(inner)[0][np.argmax(f_pts[inner])]
    lo = np.where(inner)[0][np.argmin(f_pts[inner])]
    for idx, col in ((hi, "#c0392b"), (lo, "#2471a3")):
        r = radii[idx]
        d = np.hypot(pts[:, 0] - pts[idx, 0], pts[:, 1] - pts[idx, 1])
        for j in np.where((d < (r + radii) / 2) & (d > 0))[0]:
            ax.plot([pts[idx, 0], pts[j, 0]], [pts[idx, 1], pts[j, 1]], color=col, lw=0.7, zorder=2)
        ax.add_patch(plt.Circle(pts[idx], r, fill=False, color=col, lw=1.2, linestyle=(0, (4, 2)), zorder=3))
        ax.scatter(*pts[idx], color=col, s=28, zorder=4, edgecolors="k", linewidths=0.4)
    ax.set_title("$r_i = r_0(1+\\beta f_i)$ ($\\beta$ exagg.)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "synth_consts")


def _lift_profiles(df, L, B, scope):
    tag = "fall-vl1-glob" if scope == "global" else "fall-vl1"
    out = []
    for s in SEEDS:
        rows = df[(df.dataset == f"SYNTH_HR_{L}_{B}_S{s}") & (df.feature_tag == tag)]
        if rows.empty:
            return None
        floor = rows[rows.method == "permuted"]["f1_macro"].mean()
        prof = [rows[rows.method == f"egonet_k{k}_wass"]["f1_macro"].mean() for k in KS]
        out.append([v - floor for v in prof])
    return np.array(out)


def synth_profs():
    df = pd.read_csv(SYNTH_SESSION / "results.csv")
    df = df[df.status == "ok"]
    checks = pd.read_csv(SYNTH_SESSION / "outputs" / "manipulation_checks.csv")
    reach = {L: checks[checks.dataset.str.match(f"SYNTH_HR_{L}_B1_S1")]["reach_zero_crossing"].mean() for L in WORLD_TITLES}
    offsets = {("global", "B1"): -0.27, ("global", "B0"): -0.09, ("local", "B1"): +0.09, ("local", "B0"): +0.27}
    style = {"B1": "-", "B0": (0, (4, 2))}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)
    for ax, (L, title) in zip(axes, WORLD_TITLES.items()):
        for (scope, B), off in offsets.items():
            lifts = _lift_profiles(df, L, B, scope)
            color = SCOPE_COLOR[scope]
            alpha = 0.4 if B == "B1" else 0.18
            for i, k in enumerate(KS):
                _box(ax, lifts[:, i], k + off, color, width=0.15, alpha=alpha)
            ax.plot(KS + off, np.median(lifts, axis=0), color=color, lw=1.1,
                    linestyle=style[B], alpha=1.0 if B == "B1" else 0.55, zorder=2)
        ax.axhline(0, color=ps.MUTED, lw=0.8, linestyle=(0, (2, 2)))
        ax.axvline(reach[L], color=ps.INK, lw=1.0, linestyle=(0, (5, 2)), alpha=0.55)
        ax.set_xlabel("$k$ (hops)", fontsize=12)
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.45, 6.55)
        ax.set_title(f"{title}, reach {reach[L]:.1f}", fontsize=12)
    axes[0].set_ylabel("lift over floor", fontsize=12)
    fig.tight_layout()
    save(fig, "synth_profs")


def morphs():
    res = pd.read_csv(SYNTH_SESSION / "results.csv")
    res = res[res.status == "ok"]
    rows = []
    for B, beta in BETA.items():
        for s in range(101, 106):
            cell = f"SYNTH_HR_L2_{B}_S{s}"
            sub = res[res.dataset == cell]
            glob = sub[sub.feature_tag == "fall-vl1-glob"]
            ns = sub[sub.method == "node_struct"]
            if glob.empty or ns.empty:
                continue
            floor = glob[glob.method == "permuted"]["f1_macro"].mean()
            best_bag = max(glob[glob.method == f"egonet_k{k}_wass"]["f1_macro"].mean() for k in range(1, 7))
            node = ns["f1_macro"].mean()
            rows.append({"beta": beta, "node": node - floor, "bag_adv": best_bag - node})
    part = pd.DataFrame(rows)
    betas = sorted(part.beta.unique())
    series = [("node", ps.FAMILY_COLOR["egonet_walk"]), ("bag_adv", ps.FAMILY_COLOR["structural"])]
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    for col, color in series:
        meds = []
        for b in betas:
            vals = part[part.beta == b][col].to_numpy()
            _box(ax, vals, b, color, width=0.014)
            meds.append(float(np.median(vals)))
        ax.plot(betas, meds, color=color, lw=1.3, zorder=2)
    ax.axhline(0, color=ps.MUTED, lw=0.9)
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas], fontsize=11)
    ax.set_xlim(-0.025, 0.425)
    ax.set_xlabel("coupling $\\beta$", fontsize=12)
    ax.set_ylabel("macro-F1", fontsize=12)
    fig.tight_layout()
    save(fig, "morphs")


def anchors():
    df = pd.read_csv(SESSION_ROOT / "outputs" / "separation_synth.csv")
    ok = df[df.status == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), sharey=True)
    xpos = {B: i for i, B in enumerate(BUFFERS)}
    offsets = {2: -0.25, 4: 0.0, 6: +0.25}
    for ax, ds, title in zip(axes, ("SYNTH_HR_L2_B0", "SYNTH_HR_L2_B1"),
                             ("$\\beta=0$ (leakage only)", "$\\beta=0.03$")):
        sub = ok[ok.dataset == ds]
        for k, color in K_COLORS.items():
            meds = []
            for B in BUFFERS:
                lifts = []
                for r in range(10):
                    split = f"B{B}_r{r}"
                    m = sub[(sub.method == f"egonet_k{k}_wass") & (sub.buffer_hops == B) & (sub.split == split)]["f1_macro"]
                    fl = sub[(sub.method == f"permuted_egonet_k{k}_wass") & (sub.buffer_hops == B) & (sub.split == split)]["f1_macro"]
                    if len(m) and len(fl):
                        lifts.append(float(m.iloc[0]) - float(fl.iloc[0]))
                _box(ax, lifts, xpos[B] + offsets[k], color, width=0.18)
                meds.append(float(np.median(lifts)))
            ax.plot([xpos[B] + offsets[k] for B in BUFFERS], meds, color=color, lw=1.1, zorder=2)
        ax.axhline(0, color=ps.MUTED, lw=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(B) for B in BUFFERS], fontsize=10)
        ax.set_xlabel("buffer $B$ (hops)", fontsize=12)
    axes[0].set_ylabel("lift over floor", fontsize=12)
    fig.tight_layout()
    save(fig, "anchors")


def realdatas():
    cfgs = [
        ("separation_amazon.csv", "Amazon Ratings", BUFFERS,
         [("egonet_k4_wass", "fall-vl1-glob", ps.FAMILY_COLOR["structural"], 0.103),
          ("node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], 0.088)]),
        ("separation_roman.csv", "Roman Empire", BUFFERS,
         [("egonet_k2_wass", "fall-vl1", ps.FAMILY_COLOR["egonet_hop"], 0.106),
          ("node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], 0.085)]),
        ("separation_chameleon.csv", "Chameleon", [0, 1, 2],
         [("egonet_k2_wass", "fall-vl1-glob", ps.FAMILY_COLOR["structural"], 0.445),
          ("node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], 0.442)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    for ax, (csv, title, buffers, series) in zip(axes, cfgs):
        df = pd.read_csv(SESSION_ROOT / "outputs" / csv)
        ok = df[df.status == "ok"]
        xpos = {B: i for i, B in enumerate(buffers)}
        offsets = np.linspace(-0.18, 0.18, len(series))
        for (method, tag, color, shared_ref), off in zip(series, offsets):
            meds, xs = [], []
            for B in buffers:
                lifts = []
                for r in range(10):
                    split = f"B{B}_r{r}"
                    m = ok[(ok.method == method) & (ok.feature_tag == tag) & (ok.buffer_hops == B) & (ok.split == split)]["f1_macro"]
                    f = ok[(ok.method == f"permuted_{method}") & (ok.feature_tag == tag) & (ok.buffer_hops == B) & (ok.split == split)]["f1_macro"]
                    if len(m) and len(f):
                        lifts.append(float(m.iloc[0]) - float(f.iloc[0]))
                if len(lifts) >= 5:
                    _box(ax, lifts, xpos[B] + off, color, width=0.15)
                    meds.append(float(np.median(lifts)))
                    xs.append(xpos[B] + off)
            ax.plot(xs, meds, color=color, lw=1.1, zorder=2)
            ax.axhline(shared_ref, color=color, lw=1.0, linestyle=(0, (2, 2)), alpha=0.65)
        ax.axhline(0, color=ps.MUTED, lw=0.9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(B) for B in buffers], fontsize=10)
        ax.set_xlabel("buffer $B$ (hops)", fontsize=12)
    axes[0].set_ylabel("lift over floor", fontsize=12)
    fig.tight_layout()
    save(fig, "realdatas")


def predicts():
    """Slide variant of the ring-by-ring prediction figure (replicated)."""
    from scipy import stats  # noqa: F401  (parity with analysis module)
    curves = pd.read_csv(SYNTH_SESSION / "outputs" / "distance_homophily_curves.csv")
    df = pd.read_csv(SYNTH_SESSION / "results.csv")
    df = df[df.status == "ok"]
    N_SOURCES = 800
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    for ax, (L, title) in zip(axes, WORLD_TITLES.items()):
        n_stack, e_stack = [], []
        for sd in SEEDS:
            cv = curves[curves.dataset == f"SYNTH_HR_{L}_B1_S{sd}"].sort_values("d")
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

        r_n = float(np.corrcoef(naive, mean)[0, 1])
        r_a = float(np.corrcoef(adapt, mean)[0, 1])
        ax.fill_between(KS, mean - std, mean + std, color=SCOPE_COLOR["global"], alpha=0.18, linewidth=0)
        ax.plot(KS, mean, color=SCOPE_COLOR["global"], lw=1.8, marker="o", ms=4)
        ax.plot(KS, place(adapt), color=ps.INK, lw=1.4, linestyle=(0, (5, 2)), marker="s", ms=4, fillstyle="none")
        ax.plot(KS, place(naive), color=ps.MUTED, lw=1.2, linestyle=(0, (2, 2)), marker="^", ms=4, fillstyle="none")
        ax.axhline(0, color=ps.MUTED, lw=0.8)
        ax.set_title(f"{title}\n$r$: adaptive {r_a:.2f}, naive {r_n:.2f}", fontsize=11)
        ax.set_xlabel("$k$ (hops)", fontsize=12)
        ax.set_xticks(range(1, 7))
    axes[0].set_ylabel("lift over floor", fontsize=12)
    fig.tight_layout()
    save(fig, "predicts")


def main():
    ps.use_style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11})
    FIG_DIR.mkdir(exist_ok=True)
    synth_consts()
    synth_profs()
    morphs()
    anchors()
    realdatas()
    predicts()


if __name__ == "__main__":
    main()
