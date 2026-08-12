"""Per-world overview figures in the shared manuscript style (Layout A).

One figure per grid cell, identical to the real-dataset experiment figures:
random floor + node-features reference boxes, then local/global box pairs
for k = 1..6. No walk family in this session (walk_methods empty).

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.nodeclass.figures import dataset_overview_figure

import pandas as pd

KHOP_METHODS = {f"egonet_k{k}_wass": f"$k$={k}" for k in range(1, 7)}
CELLS = [f"SYNTH_HR_{L}_{B}" for L in ("L1", "L2", "L3") for B in ("B1", "B0")]


def main():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    df = df[df.status == "ok"]
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    for cell in CELLS:
        sub = df[df.dataset == cell]
        dataset_overview_figure(sub, KHOP_METHODS, {}, SESSION_ROOT / "figures" / f"overview_{cell}", height=2.5)
        print("rendered", cell)


if __name__ == "__main__":
    main()
