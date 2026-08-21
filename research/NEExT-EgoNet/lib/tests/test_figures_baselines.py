"""Smoke tests for the embedding-baselines figure renderers (synthetic frames)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("matplotlib")

from lib.containment import plotstyle as ps  # noqa: E402
from lib.nodeclass.figures import baselines_overview_figure, separation_families_figure  # noqa: E402


def _combined():
    rng = np.random.RandomState(0)
    rows = []
    for key in ("airports", "roman"):
        for method, tag, fam, mu in [("permuted", "fall-vl1", "floor", 0.25), ("node_struct", "fall-vl1", "node", 0.5),
                                     ("egonet_k1_wass", "fall-vl1", "bag_hop", 0.55), ("deepwalk", "d16-s42", "proximity", 0.35),
                                     ("struc2vec", "d16-s42", "structural", 0.58)]:
            for split in range(10):
                rows.append({"dataset_key": key, "method": method, "feature_tag": tag, "family": fam, "split": split,
                             "f1_macro": mu + rng.normal(0, 0.02), "accuracy": mu, "status": "ok"})
    return pd.DataFrame(rows)


def test_overview_figure_renders(tmp_path):
    comb = _combined()
    panels = [{"key": "airports", "title": "(a) Airports", "best_bag": ("egonet_k1_wass", "fall-vl1"), "emb_tag": "d16-s42"},
              {"key": "roman", "title": "(b) Roman", "best_bag": ("egonet_k1_wass", "fall-vl1"), "emb_tag": "d16-s42"}]
    methods = {"deepwalk": ("proximity", "DeepWalk"), "struc2vec": ("structural", "struc2vec")}
    stem = baselines_overview_figure(comb, panels, methods, tmp_path / "overview")
    assert Path(str(stem) + ".pdf").exists() and Path(str(stem) + ".png").exists()


def test_separation_figure_renders(tmp_path):
    rng = np.random.RandomState(1)
    rows = []
    for method in ("deepwalk", "permuted_deepwalk", "struc2vec", "permuted_struc2vec"):
        for B in (0, 2):
            for r in range(10):
                base = 0.25 if method.startswith("permuted") else 0.4
                rows.append({"method": method, "feature_tag": "d16-s42", "buffer_hops": B, "split": f"B{B}_r{r}",
                             "f1_macro": base + rng.normal(0, 0.02), "status": "ok"})
    sep = pd.DataFrame(rows)
    legend = [(ps.FAMILY_COLOR["emb_proximity"], "proximity"), (ps.FAMILY_COLOR["emb_structural"], "structural")]
    panels = [{"title": "toy", "buffers": [0, 2], "legend": legend,
               "series": [(sep, "deepwalk", "d16-s42", ps.FAMILY_COLOR["emb_proximity"], "deepwalk", 0.15),
                          (sep, "struc2vec", "d16-s42", ps.FAMILY_COLOR["emb_structural"], "struc2vec", 0.15)]}]
    stem = separation_families_figure(panels, tmp_path / "sep")
    assert Path(str(stem) + ".pdf").exists()
