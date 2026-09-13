"""Roman Empire figures for the short (conference) paper, at LNCS text width.

The library renderers hard-code the letter-page width (plotstyle.FULL_W = 6.3 in).
The Springer svproc text width is 12.2 cm = 4.80 in, so this wrapper narrows
FULL_W before rendering so the 8 pt figure fonts print at size instead of
being scaled down by \\includegraphics. Two figures, both Roman Empire only:

  figures/short/exp2_overview     -- ego-net k-hop + walk bags, both scopes
                                     (same data/layout as the long paper's
                                     fig:exp2-overview)
  figures/short/baselines_roman   -- the six node-embedding methods vs the
                                     references, single panel (the Roman
                                     panel of fig:baselines-overview)

Run from the repo root: uv run --no-sync python <this file>
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib.containment import plotstyle as ps  # noqa: E402
from lib.nodeclass.figures import baselines_overview_figure, dataset_overview_figure  # noqa: E402
from plot_baselines_overview import METHODS  # noqa: E402

LNCS_TEXT_WIDTH_IN = 12.2 / 2.54
ROMAN_SESSION = RESEARCH_ROOT / "sessions" / C.DATASETS["roman"]["signal_session"]


def main(tag="d16-s42"):
    ps.FULL_W = LNCS_TEXT_WIDTH_IN
    out = SESSION_ROOT / "figures" / "short"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROMAN_SESSION / "results.csv")
    df = df[df["status"] == "ok"]
    stem = dataset_overview_figure(
        df,
        khop_methods={f"egonet_k{k}_wass": f"$k{{=}}{k}$" for k in [1, 2, 3, 4, 5]},
        walk_methods={"walk_tight_wass": "tight", "walk_default_wass": "default", "walk_wide_wass": "wide"},
        stem=out / "exp2_overview",
        height=2.4,
    )
    print(f"written {stem}.pdf/.png")

    comb = pd.read_csv(SESSION_ROOT / "outputs" / "combined_results.csv")
    panels = [{"key": "roman", "title": "Roman Empire", "best_bag": C.DATASETS["roman"]["expected_best_bag"], "emb_tag": tag}]
    stem = baselines_overview_figure(comb, panels, METHODS, out / "baselines_roman", ncols=1, height=2.6)
    print(f"written {stem}.pdf/.png")


if __name__ == "__main__":
    main()
