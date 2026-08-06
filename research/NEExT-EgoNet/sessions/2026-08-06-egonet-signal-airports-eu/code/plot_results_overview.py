"""Experiment 1 overview figure (Air Traffic Europe) — merged-axis layout.

Thin wrapper over lib.nodeclass.figures.dataset_overview_figure.
Output: figures/exp1_overview.{pdf,png}

Usage: uv run --with matplotlib python code/plot_results_overview.py
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.nodeclass.figures import dataset_overview_figure  # noqa: E402

import pandas as pd  # noqa: E402

df = pd.read_csv(SESSION_ROOT / "results.csv")
df = df[(df["status"] == "ok") & df["feature_tag"].isin(["fall-vl1", "fall-vl1-glob"])]
stem = dataset_overview_figure(
    df,
    khop_methods={f"egonet_k{k}_wass": f"$k{{=}}{k}$" for k in [1, 2, 3]},
    walk_methods={"walk_tight_wass": "tight", "walk_default_wass": "default", "walk_wide_wass": "wide"},
    stem=SESSION_ROOT / "figures" / "exp1_overview",
)
print(f"wrote {stem}.{{pdf,png}}")
