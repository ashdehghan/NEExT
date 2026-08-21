"""Registry contract: every method returns a node_id-keyed, finite frame that
the node-classification evaluator accepts; pecanpy routes are deterministic."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.nodeclass import available_methods, embed
from lib.nodeclass.evaluate import evaluate_node_representation


def _graph():
    G = nx.barbell_graph(7, 5)  # 19 nodes, two roles + a path
    return G


def _node_table(G):
    role = [0] * 7 + [1] * 5 + [0] * 7
    return pd.DataFrame({"node_id": range(G.number_of_nodes()), "y": role, "y_raw": role})


@pytest.mark.parametrize("method", [m for m in available_methods() if m != "graphwave"])
def test_learned_methods_contract(method, tmp_path):
    pytest.importorskip("gensim")
    G = _graph()
    kw = {"n_walks": 4, "walk_length": 20}
    if method == "struc2vec":
        kw.update(k_star=2, n_candidates=None, epochs=1)
    df, rec = embed(method, G, seed=3, dim=8, **kw)
    assert list(df.columns) == ["node_id"] + [f"{method}_{j}" for j in range(8)]
    assert df["node_id"].tolist() == list(range(G.number_of_nodes()))
    assert rec["family"] in {"proximity", "structural"} and rec["n_features"] == 8 and rec["dim"] == 8
    assert np.isfinite(df.drop(columns="node_id").values).all()
    df2, _ = embed(method, G, seed=3, dim=8, **kw)
    pd.testing.assert_frame_equal(df, df2)


def test_graphwave_contract_and_subset():
    G = _graph()
    df, rec = embed("graphwave", G, scales=[1.0, 4.0], time_pnts=np.linspace(0, 10, 4))
    assert rec["fixed_width"] and rec["dim"] is None and rec["n_features"] == 16
    sub, _ = embed("graphwave", G, nodes=np.array([2, 9]), scales=[1.0, 4.0], time_pnts=np.linspace(0, 10, 4))
    assert sub["node_id"].tolist() == [2, 9]
    assert np.allclose(sub.drop(columns="node_id").values, df.drop(columns="node_id").values[[2, 9]])


def test_frames_feed_the_evaluator(tmp_path):
    pytest.importorskip("gensim")
    G = _graph()
    table = _node_table(G)
    df, _ = embed("role2vec", G, seed=1, dim=4, n_walks=3, walk_length=10)
    out = evaluate_node_representation("role2vec", df, table, n_splits=2, test_size=0.3, seed=0)
    assert out["status"] == "ok" and len(out["metrics_rows"]) == 2


def test_unknown_method():
    with pytest.raises(KeyError):
        embed("nope", _graph())


def test_pecanpy_route_runs_when_available(tmp_path):
    pytest.importorskip("pecanpy")
    pytest.importorskip("gensim")
    G = _graph()
    df, rec = embed("deepwalk", G, seed=1, dim=4, n_walks=2, walk_length=10, route="pecanpy", workdir=str(tmp_path))
    assert rec["route"] == "pecanpy" and df.shape == (G.number_of_nodes(), 5)
