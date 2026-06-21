"""Quality guard for GNN graph embeddings.

These tests prove the embeddings carry real signal (not just the right shape):
a simple classifier trained on the GNN embedding vectors of a separable
synthetic dataset must score well above chance. The dataset is generated offline
(no network), and we use a lightweight scikit-learn classifier directly on the
embedding vectors rather than the heavy XGBoost pipeline — that keeps the test
fast and avoids running torch's and XGBoost's OpenMP runtimes in one process.

Floors are deliberately conservative relative to observed benchmark numbers
(er_vs_ba: GCN ~1.0, GraphSAGE/GIN ~0.98) so the test is a real regression guard
without being flaky on small samples. A collapsed embedding (e.g. the pre-fix
GIN) scores at chance and fails this guard.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

from NEExT.framework import NEExT  # noqa: E402

ARCHITECTURES = ["GCN", "GraphSAGE", "GIN"]
ACCURACY_FLOOR = 0.80  # er_vs_ba is highly separable; observed GNN acc >= 0.98


def _labeled_collection_and_features():
    """Build a separable labeled synthetic collection (ER vs BA) + features."""
    nxt = NEExT(log_level="ERROR")
    collection = nxt.generate_synthetic_graphs(preset="er_vs_ba", seed=42, n_per_class=30, n_nodes=40)
    features = nxt.compute_node_features(collection, feature_list=["page_rank", "degree_centrality"], feature_vector_length=3)
    return nxt, collection, features


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_gnn_embeddings_carry_signal(architecture):
    """A classifier on GNN embeddings must beat chance on a separable dataset."""
    nxt, collection, features = _labeled_collection_and_features()
    embeddings = nxt.compute_graph_embeddings(
        collection,
        features,
        embedding_algorithm="gnn",
        architecture=architecture,
        embedding_dimension=6,
    )

    df = embeddings.embeddings_df
    labels_by_graph = {graph.graph_id: graph.graph_label for graph in collection.graphs}
    y = np.array([labels_by_graph[graph_id] for graph_id in df["graph_id"]])
    X = df[embeddings.embedding_columns].to_numpy()

    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=3)
    accuracy = float(scores.mean())
    assert accuracy >= ACCURACY_FLOOR, (
        f"{architecture} cross-val accuracy {accuracy:.3f} below floor " f"{ACCURACY_FLOOR} — embeddings may have collapsed or lost signal"
    )
