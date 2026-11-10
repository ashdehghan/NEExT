"""basic_expansion on an edgeless graph (a patch made of one isolated node) must not divide by zero."""

import networkx as nx
import pytest

from NEExT import NEExT


@pytest.mark.parametrize("graph_type", ["networkx", "igraph"])
def test_basic_expansion_edgeless_graph_is_zero(graph_type):
    edgeless = nx.Graph(label=0)
    edgeless.add_nodes_from([0, 1])
    path = nx.path_graph(3)
    path.graph["label"] = 1
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    coll = nxt.load_from_networkx([edgeless, path], graph_type=graph_type, filter_largest_component=False)
    feats = nxt.compute_node_features(coll, feature_list=["basic_expansion"], feature_vector_length=2, show_progress=False, normalize_features=False)
    df = feats.features_df
    cols = [c for c in df.columns if c.startswith("basic_expansion")]
    edgeless_rows = df[df["graph_id"] == coll.graphs[0].graph_id]
    assert len(edgeless_rows) == 2
    assert (edgeless_rows[cols].to_numpy() == 0).all()
    path_rows = df[df["graph_id"] == coll.graphs[1].graph_id]
    assert path_rows[cols[0]].max() > 0
