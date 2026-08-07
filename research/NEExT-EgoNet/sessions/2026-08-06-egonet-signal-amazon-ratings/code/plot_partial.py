"""Incremental overview render (tolerant mode) — draws whatever stages have landed."""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.nodeclass.figures import dataset_overview_figure  # noqa: E402

import pandas as pd  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "exp4_partial"
df = pd.read_csv(SESSION_ROOT / "results.csv")
df = df[df["status"] == "ok"]
stem = dataset_overview_figure(
    df,
    khop_methods={f"egonet_k{k}_wass": f"$k{{=}}{k}$" for k in [1, 2, 3]},
    walk_methods={"walk_tight_wass": "tight", "walk_default_wass": "default", "walk_wide_wass": "wide"},
    stem=SESSION_ROOT / "figures" / out,
    strict=False,
)
print(f"wrote {stem}.png")
