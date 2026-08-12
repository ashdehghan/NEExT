"""Synthetic grid figures: k-profiles per wavelength + the money figure.

Profiles: one panel per wavelength, four series (scope x coupling), the
permutation floor, and the measured homophily reach as a vertical line —
the plateau should form at the line. Money figure: last k with a
significant paired gain (p<.05 over the 10 shared splits) vs measured
reach, one point per (wavelength, scope, coupling) series.

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

SCOPE_COLOR = {"local": ps.FAMILY_COLOR["egonet_hop"], "global": ps.FAMILY_COLOR["structural"]}
COUPLING_STYLE = {"B1": "-", "B0": (0, (4, 2))}
WAVELENGTH_OF = {"L1": 0.10, "L2": 0.20, "L3": 0.35}


def load():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    kh = df[(df.status == "ok") & df.method.str.startswith("egonet_k")].copy()
    kh["k"] = kh.method.str.extract(r"k(\d)").astype(int)
    kh["scope"] = kh.feature_tag.map(lambda t: "global" if t.endswith("glob") else "local")
    kh["L"] = kh.dataset.str.split("_").str[2]
    kh["B"] = kh.dataset.str.split("_").str[3]
    floors = df[(df.status == "ok") & (df.method == "permuted") & (df.feature_tag == "fall-vl1")]
    reach = pd.read_csv(SESSION_ROOT / "outputs" / "manipulation_checks.csv").set_index("dataset")["reach_zero_crossing"]
    return kh, floors, reach


def last_significant_gain(g: pd.DataFrame) -> int:
    piv = g.pivot_table(index="split", columns="k", values="f1_macro")
    last = 1
    for k in range(2, 7):
        diff = piv[k] - piv[k - 1]
        _, p = stats.ttest_rel(piv[k], piv[k - 1])
        if p < 0.05 and diff.mean() > 0:
            last = k
    return last


def profiles_figure(kh, floors, reach):
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.7), sharey=True)
    for ax, L in zip(axes, ("L1", "L2", "L3")):
        for B in ("B1", "B0"):
            cell = f"SYNTH_HR_{L}_{B}"
            for scope in ("global", "local"):
                g = kh[(kh.dataset == cell) & (kh.scope == scope)].groupby("k")["f1_macro"].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], yerr=g["std"], color=SCOPE_COLOR[scope],
                            linestyle=COUPLING_STYLE[B], marker="o", ms=2.5, lw=1.1, capsize=1.5,
                            alpha=1.0 if B == "B1" else 0.65)
            floor = floors[floors.dataset == cell]["f1_macro"].mean()
            if B == "B0":
                ax.axhline(floor, color=ps.MUTED, lw=0.7, linestyle=(0, (2, 2)))
        ax.axvline(min(float(reach[f"SYNTH_HR_{L}_B1"]), 6.45), color=ps.INK, lw=0.9,
                   linestyle=(0, (5, 2)), alpha=0.55)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.6, 6.45)
        ax.set_title(f"$\\lambda$={WAVELENGTH_OF[L]}  (reach {reach[f'SYNTH_HR_{L}_B1']:.1f})", fontsize=9)
    axes[0].set_ylabel("macro-F1")
    handles = [
        plt.Line2D([], [], color=SCOPE_COLOR["global"], lw=1.1, label="Global scope"),
        plt.Line2D([], [], color=SCOPE_COLOR["local"], lw=1.1, label="Local scope"),
        plt.Line2D([], [], color=ps.INK, lw=1.1, linestyle="-", label="$\\beta{=}0.03$ (solid)"),
        plt.Line2D([], [], color=ps.INK, lw=1.1, linestyle=(0, (4, 2)), alpha=0.65, label="$\\beta{=}0$ (dashed)"),
        plt.Line2D([], [], color=ps.INK, lw=0.9, linestyle=(0, (5, 2)), alpha=0.55, label="Measured reach"),
        plt.Line2D([], [], color=ps.MUTED, lw=0.7, linestyle=(0, (2, 2)), label="Permutation floor"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    ps.save(fig, SESSION_ROOT / "figures" / "synth_profiles_grid")


def money_figure(kh, reach):
    fig, ax = plt.subplots(figsize=(ps.FULL_W * 0.55, 2.6))
    markers = {"B1": "o", "B0": "s"}
    for scope in ("global", "local"):
        for B in ("B1", "B0"):
            xs, ys = [], []
            for L in ("L1", "L2", "L3"):
                cell = f"SYNTH_HR_{L}_{B}"
                g = kh[(kh.dataset == cell) & (kh.scope == scope)]
                xs.append(float(reach[cell]))
                ys.append(last_significant_gain(g))
            ax.plot(xs, ys, marker=markers[B], ms=5, lw=1.0, color=SCOPE_COLOR[scope],
                    linestyle="-" if B == "B1" else (0, (4, 2)),
                    fillstyle="full" if B == "B1" else "none",
                    label=f"{scope}, $\\beta{{=}}{0.03 if B == 'B1' else 0}$")
    lim = [1.5, 9]
    ax.plot(lim, lim, color=ps.MUTED, lw=0.8, zorder=0)
    ax.text(4.35, 4.05, "$k^*{=}$reach", fontsize=7, color=ps.MUTED, rotation=40)
    ax.axhline(6, color=ps.GRID, lw=0.6)
    ax.text(1.7, 6.1, "sweep limit $k{=}6$", fontsize=7, color=ps.MUTED)
    ax.set_xlabel("measured homophily reach (hops)")
    ax.set_ylabel("last $k$ with significant\npaired gain")
    ax.set_xlim(*lim)
    ax.set_ylim(1.5, 6.8)
    ax.set_yticks(range(2, 7))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    ps.save(fig, SESSION_ROOT / "figures" / "synth_money_kstar_vs_reach")


def main():
    kh, floors, reach = load()
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    profiles_figure(kh, floors, reach)
    money_figure(kh, reach)


if __name__ == "__main__":
    main()
