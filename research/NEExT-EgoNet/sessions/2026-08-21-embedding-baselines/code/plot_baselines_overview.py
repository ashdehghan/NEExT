"""Figure: embedding baselines vs the paper's references, one panel per dataset.
Reads outputs/combined_results.csv (collect_results.py). Run from the repo root."""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib.nodeclass.figures import baselines_overview_figure  # noqa: E402

TITLES = {"airports": "(a) Air Traffic -- Europe", "roman": "(b) Roman Empire", "chameleon": "(c) Wikipedia -- Chameleon", "amazon": "(d) Amazon Ratings"}
METHODS = {
    "deepwalk": ("proximity", "DeepWalk"),
    "node2vec_bfs": ("proximity", "n2v BFS"),
    "node2vec_dfs": ("proximity", "n2v DFS"),
    "struc2vec": ("structural", "struc2vec"),
    "role2vec": ("structural", "Role2Vec"),
    "graphwave": ("structural", "GraphWave"),
}


def main(tag="d16-s42"):
    comb = pd.read_csv(SESSION_ROOT / "outputs" / "combined_results.csv")
    panels = [{"key": k, "title": TITLES[k], "best_bag": C.DATASETS[k]["expected_best_bag"], "emb_tag": tag}
              for k in ("airports", "roman", "chameleon", "amazon") if (comb.dataset_key == k).any()]
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    stem = baselines_overview_figure(comb, panels, METHODS, SESSION_ROOT / "figures" / "baselines_overview")
    print(f"written {stem}.pdf/.png")


if __name__ == "__main__":
    main()
