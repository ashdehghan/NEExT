"""Phase-3 board figures in the house style: small box plots per point over
the repeats/worlds, median trend lines through them (board_profiles_box
pattern). No marker+errorbar series.

1. anchor_separation_box — lift vs buffer B; boxes over the 10 region
   repeats per (k, B), one panel per coupling.
2. morphing_box — node-baseline lift and bag advantage vs beta; boxes over
   the replicate worlds available so far (partial until the sweep lands).

Run from the repo root.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYNTH_SESSION = RESEARCH_ROOT / "sessions" / "2026-08-12-synthetic-homophily-reach"
K_COLORS = {2: "#4c72b0", 4: "#dd8452", 6: "#55a868"}
BUFFERS = [0, 1, 2, 4, 6, 8, 10]
BETA = {"B0": 0.0, "B1": 0.03, "B2": 0.10, "B3": 0.20, "B4": 0.40}


def _box(ax, values, position, color, width, alpha=0.4):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True, zorder=3)
    b["boxes"][0].set_facecolor(color)
    b["boxes"][0].set_alpha(alpha)
    b["boxes"][0].set_edgecolor(color)
    b["medians"][0].set_color(ps.INK)
    for part in ("whiskers", "caps"):
        for line in b[part]:
            line.set_color(color)
            line.set_alpha(alpha)


def anchor_separation_box():
    df = pd.read_csv(SESSION_ROOT / "outputs" / "separation_synth.csv")
    ok = df[df.status == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    xpos = {B: i for i, B in enumerate(BUFFERS)}
    offsets = {2: -0.25, 4: 0.0, 6: +0.25}
    for ax, ds, title in zip(
        axes,
        ("SYNTH_HR_L2_B0", "SYNTH_HR_L2_B1"),
        ("$\\beta=0$ control (leakage only)", "$\\beta=0.03$ (leakage + structure)"),
    ):
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
            ax.plot([xpos[B] + offsets[k] for B in BUFFERS], meds, color=color, lw=1.3, zorder=2)
        ax.axhline(0, color=ps.MUTED, lw=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([str(B) for B in BUFFERS])
        ax.set_xlabel("separation buffer $B$ (hops)")
    axes[0].set_ylabel("macro-F1 lift over floor")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.4, edgecolor=c, label=f"$k$={k}")
        for k, c in K_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    w, h = fig.get_size_inches()
    ps.save(fig, SESSION_ROOT / "figures" / "anchor_separation_box")
    print(f"anchor_separation_box aspect {w / h:.3f}")


def morphing_box():
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
    if part.empty:
        print("morphing_box: no worlds with node_struct yet")
        return
    betas = sorted(part.beta.unique())
    xpos = {b: i for i, b in enumerate(betas)}
    series = [("node", ps.FAMILY_COLOR["egonet_walk"], "node-baseline lift over floor", -0.15),
              ("bag_adv", ps.FAMILY_COLOR["structural"], "bag advantage (best bag $-$ node)", +0.15)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for col, color, label, off in series:
        meds = []
        for b in betas:
            vals = part[part.beta == b][col].to_numpy()
            _box(ax, vals, xpos[b] + off, color, width=0.22)
            meds.append(float(np.median(vals)))
        ax.plot([xpos[b] + off for b in betas], meds, color=color, lw=1.3, zorder=2, label=label)
    ax.axhline(0, color=ps.MUTED, lw=0.9)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel("coupling $\\beta$ (whisper volume)")
    ax.set_ylabel("macro-F1")
    counts = part.groupby("beta").size()
    ax.set_title("worlds per $\\beta$: " + ", ".join(f"{b:g}:{counts[b]}" for b in betas), fontsize=9, color=ps.MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.4, edgecolor=c, label=lbl)
               for _, c, lbl, _ in series]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    w, h = fig.get_size_inches()
    ps.save(fig, SYNTH_SESSION / "figures" / "morphing_box")
    print(f"morphing_box aspect {w / h:.3f}")
    print(part.groupby("beta").agg(n=("node", "size"), node=("node", "median"), bag_adv=("bag_adv", "median")).round(3).to_string())


def main():
    ps.use_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    anchor_separation_box()
    morphing_box()


if __name__ == "__main__":
    main()
