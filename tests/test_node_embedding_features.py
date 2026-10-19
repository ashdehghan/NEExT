"""struc2vec / Role2Vec / GraphWave as NEExT node features, and feature groups in greedy selection."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.features import Features


def _collection(graph_type="networkx"):
    G = nx.connected_caveman_graph(4, 6)
    edges_df = pd.DataFrame(list(G.edges()), columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": list(G.nodes()), "lbl": [n % 2 for n in G.nodes()]})
    nxt = NEExT(log_level="WARNING")
    return nxt, nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type)


@pytest.mark.parametrize("name, width", [("graphwave", 100), ("role2vec", 8), ("struc2vec", 8)])
def test_embedding_block_and_group(name, width):
    if name != "graphwave":
        pytest.importorskip("gensim")
    nxt, collection = _collection()
    params = {} if name == "graphwave" else {name: {"dim": 8, "seed": 1}}
    features = nxt.compute_node_features(
        collection, feature_list=["degree_centrality", name], feature_vector_length=1, show_progress=False, node_embedding_params=params
    )
    block = [f"{name}_{i}" for i in range(width)]
    assert features.feature_columns == ["degree_centrality_0"] + block
    assert features.feature_groups == {name: block}
    assert len(features.features_df) == 24
    units = features.selection_units()
    assert list(units) == ["degree_centrality_0", name]


def test_embedding_is_deterministic_for_a_seed():
    pytest.importorskip("gensim")
    nxt, collection = _collection()
    runs = [
        nxt.compute_node_features(
            collection, feature_list=["role2vec"], show_progress=False, node_embedding_params={"role2vec": {"dim": 4, "seed": 3}}
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0].features_df.to_numpy(), runs[1].features_df.to_numpy())


def test_all_excludes_embeddings():
    nxt, collection = _collection()
    features = nxt.compute_node_features(collection, feature_list=["all"], feature_vector_length=1, show_progress=False)
    assert not any(col.startswith(("struc2vec", "role2vec", "graphwave")) for col in features.feature_columns)
    assert features.feature_groups == {}


def test_local_scope_on_egonets_is_refused_and_global_projects_the_group():
    nxt, collection = _collection("igraph")
    egonets = nxt.compute_k_hop_egonets(collection, k_hop=1, egonet_feature_target="lbl")
    with pytest.raises(ValueError, match="feature_scope='global'"):
        nxt.compute_node_features(egonets, feature_list=["graphwave"], show_progress=False)
    features = nxt.compute_node_features(egonets, feature_list=["graphwave"], feature_scope="global", show_progress=False)
    assert features.feature_groups == {"graphwave": [f"graphwave_{i}" for i in range(100)]}
    assert len(features.features_df) == sum(len(g.nodes) for g in egonets.graphs)


def test_unknown_embedding_params_rejected():
    nxt, collection = _collection()
    with pytest.raises(ValueError, match="unknown embeddings"):
        nxt.compute_node_features(collection, feature_list=["page_rank"], show_progress=False, node_embedding_params={"deepwalk": {}})


def test_greedy_selection_moves_groups_as_blocks():
    nxt, collection = _collection("igraph")
    egonets = nxt.compute_k_hop_egonets(collection, k_hop=1, egonet_feature_target="lbl")
    features = nxt.compute_node_features(
        egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=2, feature_scope="global", show_progress=False
    )
    grouped = Features(features.features_df, features.feature_columns, {"pr": ["page_rank_0", "page_rank_1"]})
    results = nxt.compute_feature_importance(
        egonets, grouped, feature_importance_algorithm="supervised_greedy", n_iterations=2, n_jobs=1, parallel_backend="thread", holdout_size=0.3
    )
    assert sorted(results["feature_name"]) == ["degree_centrality_0", "degree_centrality_1", "pr"]
    assert dict(zip(results["feature_name"], results["n_columns"]))["pr"] == 2
    assert results["heldout_performance"].notna().all()
