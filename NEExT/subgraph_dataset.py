from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from NEExT.embeddings import Embeddings
from NEExT.graph_collection import GraphCollection
from NEExT.node_features import NodeFeatures
from NEExT.subgraph_collection import SubGraphCollection


def egonet_features(
    subgraph_collection: SubGraphCollection,
    features: List[str] = ["all"],
):
    out_features = []

    for graph in subgraph_collection.graphs:
        egonet_features_df = pd.DataFrame([node_attributes for _, node_attributes in graph.node_attributes.items()])
        features = egonet_features_df.columns if features == ["all"] else features

        egonet_features_df = egonet_features_df[features]
        egonet_features_df["node_id"] = graph.nodes
        egonet_features_df["graph_id"] = graph.graph_id

        out_features.append(egonet_features_df)

    return pd.concat(out_features, axis=0, ignore_index=True)


def egonet_node_role_features(
    subgraph_collection: SubGraphCollection,
):
    out_features = []
    for graph in subgraph_collection.graphs:
        # graph_id, node_id = subgraph_collection.subgraph_to_graph_node_mapping[graph.graph_id]
        # egonet_features = {"graph_id": graph_id, "node_id": node_id}

        egonet_features_df = pd.DataFrame()
        egonet_features_df["node_id"] = graph.nodes
        egonet_features_df["graph_id"] = graph.graph_id


        out_features.append(egonet_features_df)

    return pd.concat(out_features, axis=0, ignore_index=True)


def combine_structural_with_egonet_features(
    features: NodeFeatures,
    egonet_features_df: pd.DataFrame,
):
    features.features_df = features.features_df.merge(egonet_features_df, on=["graph_id", "node_id"])
    features.feature_columns += [feature for feature in egonet_features_df.columns if feature not in ["node_id", "graph_id"]]
    return features


def create_data_df(
    graph_collection: GraphCollection,
    subgraph_collection: SubGraphCollection,
    embeddings: Embeddings,
    target: str,
):
    """
    Mapping values of known features from a (graph_id, node_id) to (subgraph_id)
    """
    node_features_df = []
    data_df = embeddings.embeddings_df.copy().rename(columns={"graph_id": "subgraph_id"})
    data_df[["graph_id", "node_id"]] = data_df["subgraph_id"].map(subgraph_collection.subgraph_to_graph_node_mapping).to_list()

    for graph in graph_collection.graphs:
        for node in graph.nodes:
            node_variables = {"graph_id": graph.graph_id, "node_id": node}
            if graph.graph_label:
                node_variables["graph_label"] = graph.graph_label

            node_variables = node_variables | graph.node_attributes[node]
            node_variables.pop(target)
            node_features_df.append(node_variables)

    node_features_df = pd.DataFrame(node_features_df)
    data_df = data_df.merge(node_features_df).drop(columns=["graph_id", "node_id"]).rename(columns={"subgraph_id": "graph_id"})
    return data_df
