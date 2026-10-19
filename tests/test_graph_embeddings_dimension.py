"""The optimal-transport vectorizers cannot emit more dimensions than the node feature space has.

Before this guard, asking for embedding_dimension above the feature width failed deep inside
the output assembly with a bare IndexError; now it is a ValueError that says what to change.
"""

import pandas as pd
import pytest

from NEExT import NEExT


def two_cliques():
    left, right = list(range(6)), list(range(6, 12))
    edges = [(u, v) for i, u in enumerate(left) for v in left[i + 1 :]] + [(u, v) for i, u in enumerate(right) for v in right[i + 1 :]] + [(5, 6)]
    edges_df = pd.DataFrame(edges, columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": left + right, "lbl": [0] * 6 + [1] * 6})
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    collection = nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph")
    egonets = nxt.compute_k_hop_egonets(collection, k_hop=1, egonet_feature_target="lbl")
    features = nxt.compute_node_features(egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=2, show_progress=False)
    return nxt, egonets, features  # 4 feature columns


def test_dimension_at_feature_width_works():
    nxt, egonets, features = two_cliques()
    embeddings = nxt.compute_graph_embeddings(egonets, features, embedding_algorithm="approx_wasserstein", embedding_dimension=4)
    assert len(embeddings.embedding_columns) == 4
    assert embeddings.embeddings_df[embeddings.embedding_columns].shape == (12, 4)


def test_dimension_above_feature_width_raises_clearly():
    nxt, egonets, features = two_cliques()
    with pytest.raises(ValueError, match="embedding_dimension=8 exceeds"):
        nxt.compute_graph_embeddings(egonets, features, embedding_algorithm="approx_wasserstein", embedding_dimension=8)
