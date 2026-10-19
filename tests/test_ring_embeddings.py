"""Ring-by-ring egonet embeddings: rings on egonets, blocks, nesting, empty rings, weights, importance maps."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from NEExT import NEExT
from NEExT.embeddings import member_rings, ring_importance_map


def _setup(graph_type="networkx", k_hop=3):
    G = nx.barabasi_albert_graph(60, 2, seed=4)
    edges_df = pd.DataFrame(list(G.edges()), columns=["src_node_id", "dest_node_id"])
    nodes_df = pd.DataFrame({"node_id": list(G.nodes()), "lbl": [n % 3 for n in G.nodes()]})
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(edges_df=edges_df, nodes_df=nodes_df, graph_type=graph_type)
    egonets = nxt.compute_k_hop_egonets(collection, k_hop=k_hop, egonet_feature_target="lbl")
    return nxt, collection, egonets


def _parent_distances(collection):
    G = collection.graphs[0].G
    G = G if isinstance(G, nx.Graph) else nx.Graph(G.get_edgelist())
    return dict(nx.all_pairs_shortest_path_length(G))


@pytest.mark.parametrize("graph_type", ["networkx", "igraph"])
def test_khop_rings_are_parent_hop_distances(graph_type):
    nxt, collection, egonets = _setup(graph_type)
    dist = _parent_distances(collection)
    for egonet in egonets.graphs:
        centre = egonet.original_node_id
        original = {internal: node for node, internal in egonet.node_mapping.items()}
        assert set(egonet.node_rings) == set(egonet.nodes)
        for internal, ring in egonet.node_rings.items():
            assert ring == dist[centre][original[internal]]


def test_walk_rings_are_parent_hop_distances():
    nxt, collection, _ = _setup()
    walks = nxt.compute_random_walk_egonets(collection, walk_length=10, n_walks=50, egonet_feature_target="lbl", random_seed=2)
    dist = _parent_distances(collection)
    for egonet in walks.graphs:
        original = {internal: node for node, internal in egonet.node_mapping.items()}
        assert egonet.node_rings[egonet.node_mapping[egonet.original_node_id]] == 0
        for internal, ring in egonet.node_rings.items():
            assert ring == dist[egonet.original_node_id][original[internal]]


def test_approx_global_ring_blocks_match_hop_shell_features():
    """Under approx_wasserstein + global scope, ring r's block is a linear map of the ring's mean feature vector,
    which is what feature_vector_length > r computes as the hop-r shell mean (standardised: affine)."""
    nxt, collection, egonets = _setup(k_hop=2)
    feats = nxt.compute_node_features(
        egonets,
        feature_list=["degree_centrality", "page_rank", "clustering_coefficient"],
        feature_vector_length=1,
        feature_scope="global",
        show_progress=False,
    )
    rings = nxt.compute_ring_embeddings(egonets, feats, max_ring=2)
    shells = nxt.compute_node_features(
        collection,
        feature_list=["degree_centrality", "page_rank", "clustering_coefficient"],
        feature_vector_length=3,
        normalize_features=False,
        show_progress=False,
    ).features_df
    centre_of = {g.graph_id: g.original_node_id for g in egonets.graphs}
    emb = rings.embeddings_df.assign(node_id=lambda d: d["graph_id"].map(centre_of)).merge(shells, on="node_id")
    for ring in (0, 1, 2):
        X = emb[[f"{f}_{ring}" for f in ("degree_centrality", "page_rank", "clustering_coefficient")]].to_numpy()
        X = np.column_stack([X, np.ones(len(X))])
        for col in [c for c in rings.embedding_columns if c.endswith(f"_ring{ring}") and c.startswith("emb_")]:
            y = emb[col].to_numpy()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            assert np.allclose(X @ coef, y, atol=1e-8)


def test_blocks_are_nested_in_max_ring():
    nxt, _, egonets = _setup()
    feats = nxt.compute_node_features(
        egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=1, feature_scope="global", show_progress=False
    )
    small = nxt.compute_ring_embeddings(egonets, feats, max_ring=1).embeddings_df
    large = nxt.compute_ring_embeddings(egonets, feats, max_ring=3).embeddings_df
    for col in small.columns:
        np.testing.assert_allclose(small[col].to_numpy(), large[col].to_numpy())


def test_empty_rings_are_zero_and_sizes_match():
    nxt, _, egonets = _setup(k_hop=1)
    feats = nxt.compute_node_features(
        egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=1, feature_scope="global", show_progress=False
    )
    emb = nxt.compute_ring_embeddings(egonets, feats, max_ring=3).embeddings_df
    assert (emb["ring0_size"] == 1).all()
    assert (emb[["ring2_size", "ring3_size"]] == 0).all().all()
    assert (emb[[c for c in emb.columns if c.endswith(("_ring2", "_ring3"))]] == 0).all().all()
    rings = member_rings(egonets)
    expected = rings[rings["ring"] == 1].groupby("graph_id").size()
    assert emb.set_index("graph_id")["ring1_size"].equals(expected.reindex(emb["graph_id"]).fillna(0).astype(int).rename("ring1_size"))


def test_ring_weights_scale_blocks():
    nxt, _, egonets = _setup()
    feats = nxt.compute_node_features(
        egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=1, feature_scope="global", show_progress=False
    )
    base = nxt.compute_ring_embeddings(egonets, feats, max_ring=2).embeddings_df
    weighted = nxt.compute_ring_embeddings(egonets, feats, max_ring=2, ring_weights={1: 0.0, 2: 2.0}).embeddings_df
    ring1 = [c for c in base.columns if c.endswith("_ring1") and c.startswith("emb_")]
    ring2 = [c for c in base.columns if c.endswith("_ring2") and c.startswith("emb_")]
    assert (weighted[ring1] == 0).all().all()
    np.testing.assert_allclose(weighted[ring2].to_numpy(), 2.0 * base[ring2].to_numpy())


def test_local_scope_and_exact_lot_run():
    nxt, _, egonets = _setup(k_hop=2)
    feats = nxt.compute_node_features(egonets, feature_list=["degree_centrality", "page_rank"], feature_vector_length=1, show_progress=False)
    emb = nxt.compute_ring_embeddings(egonets, feats, max_ring=2, embedding_algorithm="wasserstein", embedding_dimension=2)
    assert emb.embeddings_df.shape == (len(egonets.graphs), 1 + 3 * 2 + 3)


def test_importance_map():
    nxt, collection, _ = _setup()
    walks = nxt.compute_random_walk_egonets(collection, walk_length=10, n_walks=50, egonet_feature_target="lbl", random_seed=2)
    egonet = walks.graphs[0]
    imp = ring_importance_map(walks, egonet.graph_id, {0: 1.0, 1: 0.5})
    assert abs(imp["importance"].sum() - 1.0) < 1e-9
    assert (imp.loc[imp["ring"] >= 2, "importance"] == 0).all()
    assert set(imp["node_id"]) == set(egonet.node_mapping)


def test_draw_importance_map_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from NEExT.embeddings import draw_importance_map

    nxt, collection, egonets = _setup(k_hop=2)
    imp = ring_importance_map(egonets, egonets.graphs[0].graph_id, {0: 1.0, 1: 1.0, 2: 0.2})
    ax = draw_importance_map(collection.graphs[0], imp)
    assert ax is not None
