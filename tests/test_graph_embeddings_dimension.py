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


def test_rank_deficient_components_are_zeroed_and_seed_free():
    """Columns that rescale another leave the bag-mean matrix rank-deficient; the components past its rank are
    round-off in a seed-dependent direction and must come out as exact zeros, identical for every seed."""
    import numpy as np

    from NEExT.features import Features

    nxt, egonets, features = two_cliques()
    df = features.features_df.copy()
    base = features.feature_columns[0]
    df["copy_a"] = 2.0 * df[base]
    df["copy_b"] = -0.5 * df[base] + 1.0
    columns = features.feature_columns + ["copy_a", "copy_b"]
    widened = Features(df, columns)
    rank = np.linalg.matrix_rank(df.groupby("graph_id")[columns].mean().to_numpy())
    assert rank < len(columns)
    runs = [
        nxt.compute_graph_embeddings(egonets, widened, embedding_algorithm="approx_wasserstein", embedding_dimension=len(columns), random_state=seed)
        .embeddings_df.drop(columns="graph_id")
        .to_numpy()
        for seed in (1, 2)
    ]
    for emb in runs:
        zero = (emb == 0.0).all(axis=0)
        assert zero.sum() == len(columns) - rank
        assert zero[rank:].all() and not zero[:rank].any()
    assert abs(abs(runs[0]) - abs(runs[1])).max() < 1e-6
