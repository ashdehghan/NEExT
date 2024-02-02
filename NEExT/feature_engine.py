"""
This class contains methods which allows the user
to compute various features on the graph, which
could capture various properties of the graph
including structural, density, ...
"""

# External Libraries
import copy
import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

# Internal Libraries
from NEExT.helper_functions import get_nodes_x_hops_away, get_all_in_community_degrees, \
      get_own_in_community_degree, get_specific_in_community_degree, community_volume

from NEExT import node_embedding_engine


def get_list_of_graph_features():
    return list(features.keys())


def compute_feature(g_obj, feat_name, feat_vect_len):
    if feat_name not in features:
        raise ValueError("The selected feature is not supported.")
    g_obj.computed_features.add(feat_name)
    features[feat_name](g_obj, feat_vect_len, feat_name)


def compute_lsme(g_obj, feat_vect_len, func_name):
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
    else:
        selected_nodes = list(G.nodes)

    feats = node_embedding_engine.run_lsme_embedding(G, feat_vect_len, selected_nodes)
    feat_cols = []
    for col in feats.columns.tolist():
        if "feat" in col:
            feat_cols.append(col)
    feats.insert(1, "graph_id", graph_id)
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def compute_self_walk(g_obj, feat_vect_len, func_name):
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
    else:
        selected_nodes = list(G.nodes)
    iG = ig.Graph.from_networkx(G)
    A = np.array(iG.get_adjacency().data)
    Ao = copy.deepcopy(A)
    feats = {}
    for i in range(2, feat_vect_len+2):
        A = np.linalg.matrix_power(Ao, i)
        diag_elem = np.diag(A)
        feats["feat_selfwalk_"+str(i-2)] = list(diag_elem)
    feats = pd.DataFrame(feats)
    feat_cols = list(feats.columns)
    feats.insert(0, "node_id", list(G.nodes))
    feats.insert(1, "graph_id", graph_id)
    # This is not a cleanest way of doing this, but for now, it is ok.
    feats = feats[feats["node_id"].isin(selected_nodes)]
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def compute_basic_expansion(g_obj, feat_vect_len, func_name):
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
    else:
        selected_nodes = list(G.nodes)
    feats = node_embedding_engine.run_expansion_embedding(G, feat_vect_len, selected_nodes)
    feat_cols = []
    for col in feats.columns.tolist():
        if "feat" in col:
            feat_cols.append(col)
    feats.insert(1, "graph_id", graph_id)
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def compute_basic_node_features(g_obj, feat_vect_len, func_name):
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
    else:
        selected_nodes = list(G.nodes)
    node_feature_list = []
    feat_cols = None
    for node in selected_nodes:
        feat_cols = list(G.nodes[node].keys())
        node_feature_list.append(list(G.nodes[node].values()))
    feats = pd.DataFrame(node_feature_list)
    feat_cols = ["feat_"+i for i in feat_cols]
    feats.columns = feat_cols
    feats.insert(0, "node_id", list(G.nodes))
    feats.insert(1, "graph_id", graph_id)
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def compute_structural_node_features(g_obj, feat_vect_len, func_name):
    """
    This method will compute structural node feature
    for every node up to emb_dim-hops-away-neighbors.
    """
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
    else:
        selected_nodes = list(G.nodes)
    if func_name == "page_rank":
        srtct_feat = nx.pagerank(G, alpha=0.9, max_iter=1000)
    elif func_name == "degree_centrality":
        srtct_feat = nx.degree_centrality(G)
    elif func_name == "closeness_centrality":
        srtct_feat = nx.closeness_centrality(G)
    elif func_name == "load_centrality":
        srtct_feat = nx.load_centrality(G)
    elif func_name == "eigenvector_centrality":
        srtct_feat = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-03)
    else:
        raise ValueError("The selected structural feature is not supported.")
    nodes = []
    features = []
    for node in selected_nodes:
        nodes.append(node)
        feat_vect = []
        nbs = get_nodes_x_hops_away(G, node, max_hop_length=feat_vect_len)
        feat_vect.append(srtct_feat[node])
        for i in range(1, feat_vect_len):
            if len(nbs[i]) > 0:
                nbs_pr = [srtct_feat[j] for j in nbs[i]]
                feat_vect.append(sum(nbs_pr)/len(nbs_pr))
            else:
                feat_vect.append(0.0)
        features.append(feat_vect)
    feats = pd.DataFrame(features)
    feat_cols = ["feat_"+func_name+"_"+str(i) for i in range(feats.shape[1])]
    feats.columns = feat_cols
    feats.insert(0, "node_id", nodes)
    feats.insert(1, "graph_id", graph_id)
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def compute_community_aware_features(g_obj, feat_vect_len, func_name):
    """
    Calculates node features
    that use the modularity-optimizing Leiden community detection algorithm.

    Node that the community detection algorithm uses all nodes in the graph,
    not just the sample. This might have computational implications.
    """
    graph_id = g_obj.graph_id
    G = g_obj.graph
    if g_obj.graph_node_source == "sample":
        selected_nodes = g_obj.node_samples
        # get neighbors of selected nodes for feat_vect_len hops
        calculated_nodes = list(
            set([
                n for node in selected_nodes
                for n in get_nodes_x_hops_away(G, node, feat_vect_len)
                ])
            )
    else:
        selected_nodes = list(G.nodes)
        calculated_nodes = selected_nodes

    # get the Leiden community partition
    # TODO repeat this? to guarantee stable result
    resolution_lambda = 1
    iG = ig.Graph.from_networkx(G)
    partition = iG.community_leiden(objective_function="modularity",
                                    resolution_parameter=resolution_lambda
                                    )

    if func_name == 'anomaly_score_CADA':
        comm_feat = {
            node: G.degree(node) / max(get_all_in_community_degrees(G, node, partition))
            for node in calculated_nodes
        }
    elif func_name == 'normalized_anomaly_score_CADA':
        comm_feat = {
            node: get_own_in_community_degree(G, node, partition) / G.degree(node)
            for node in calculated_nodes
        }
    elif func_name == 'community_association_strength':
        comm_feat = {
            node: 2*(
                (get_own_in_community_degree(G, node, partition)/G.degree(node))
                -
                resolution_lambda*(
                    (community_volume(G, node, partition) - G.degree(node))
                    /
                    (2*G.number_of_edges())
                )
            )
            for node in calculated_nodes
        }
    elif func_name == 'normalized_within_module_degree':
        # the mean and std of the indegree of nodes in the same community
        mu_per_community = {}
        sigma_per_community = {}

        comm_feat = {}
        for node in calculated_nodes:
            # get index of the community the node belongs to
            comm_index = [i for i, community in enumerate(partition) if node in community][0]
            if node not in mu_per_community:
                in_community_degrees = [
                    get_specific_in_community_degree(G, v, partition, comm_index)
                    for v in partition[comm_index]
                ]
                mu_per_community[comm_index] = np.mean(in_community_degrees)
                sigma_per_community[comm_index] = np.std(in_community_degrees)

            if sigma_per_community[comm_index] == 0:
                comm_feat[node] = 0
            else:
                comm_feat[node] = (
                    (get_specific_in_community_degree(G, node, partition, comm_index)
                     - mu_per_community[comm_index]) / sigma_per_community[comm_index]
                )
    elif func_name == 'participation_coefficient':
        comm_feat = {
            node: 1 - sum(
                (in_degree / G.degree(node))**2
                for in_degree in get_all_in_community_degrees(G, node, partition)
            )
            for node in calculated_nodes
        }
    else:
        raise ValueError("The selected structural feature is not supported.")

    features = []
    for i, node in enumerate(selected_nodes):
        feat_vect = []
        nbs = get_nodes_x_hops_away(G, node, max_hop_length=feat_vect_len)
        feat_vect.append(comm_feat[node])
        for i in range(1, feat_vect_len):
            if len(nbs[i]) > 0:
                nbs_pr = [comm_feat[j] for j in nbs[i]]
                feat_vect.append(sum(nbs_pr)/len(nbs_pr))
            else:
                feat_vect.append(0.0)
        features.append(feat_vect)
    feats = pd.DataFrame(features)
    feat_cols = ["feat_"+func_name+"_"+str(i) for i in range(feats.shape[1])]
    feats.columns = feat_cols
    feats.insert(0, "node_id", selected_nodes)
    feats.insert(1, "graph_id", graph_id)
    g_obj.feature_collection["features"][func_name] = {}
    g_obj.feature_collection["features"][func_name]["feats"] = feats
    g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


def pool_features(g_obj, pool_method="concat"):
    """
    This method will use the features built on the graph to construct
    a global embedding for the nodes of the graph.
    """
    pooled_features = pd.DataFrame()
    if pool_method == "concat":
        for feat_name in list(g_obj.computed_features):
            features = g_obj.feature_collection["features"][feat_name]["feats"]
            if pooled_features.empty:
                pooled_features = features.copy(deep=True)
            else:
                pooled_features = pooled_features.merge(
                    features, on=["node_id", "graph_id"], how="inner"
                )
    else:
        raise ValueError("Pooling type is not supported.")
    g_obj.feature_collection["pooled_features"] = pooled_features


features = {
    "lsme": compute_lsme,
    "self_walk": compute_self_walk,
    "basic_expansion": compute_basic_expansion,
    "basic_node_features": compute_basic_node_features,
    "page_rank": compute_structural_node_features,
    "degree_centrality": compute_structural_node_features,
    "closeness_centrality": compute_structural_node_features,
    "load_centrality": compute_structural_node_features,
    "eigenvector_centrality": compute_structural_node_features
}
