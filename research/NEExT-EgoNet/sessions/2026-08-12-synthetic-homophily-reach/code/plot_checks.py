"""Distance-homophily curve figure for the B1 grid (manipulation-check artifact).

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"L1": "#4c72b0", "L2": "#dd8452", "L3": "#55a868"}


def main():
    curves = pd.read_csv(SESSION_ROOT / "outputs" / "distance_homophily_curves.csv")
    checks = pd.read_csv(SESSION_ROOT / "outputs" / "manipulation_checks.csv").set_index("dataset")
    ps.use_style()
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.9))
    for L, c in COLORS.items():
        cell = f"SYNTH_HR_{L}_B1"
        cv = curves[curves.dataset == cell]
        lam = checks.loc[cell, "wavelength"]
        reach = checks.loc[cell, "reach_zero_crossing"]
        ax.plot(cv.d, cv.excess, marker="o", ms=3, lw=1.2, color=c, label=f"$\\lambda$={lam} (reach {reach:.1f})")
        ax.axvline(reach, color=c, lw=0.7, linestyle=(0, (3, 3)), alpha=0.7)
    ax.axhline(0, color=ps.MUTED, lw=0.8)
    ax.set_xlabel("hop distance $d$")
    ax.set_ylabel("homophily excess\n$P(\\mathrm{same}\\mid d) - \\sum_c p_c^2$")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
    fig.tight_layout()
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    ps.save(fig, SESSION_ROOT / "figures" / "synth_distance_homophily_v2")


if __name__ == "__main__":
    main()
