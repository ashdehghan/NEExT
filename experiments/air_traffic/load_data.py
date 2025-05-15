import igraph as ig
import pandas as pd

from NEExT.builders import EmbeddingBuilder
from NEExT.collections import EgonetCollection
from NEExT.datasets.graph_dataset import GraphDataset
from NEExT.features import NodeFeatures, StructuralNodeFeatures
from NEExT.io import GraphIO


def load_dataframes(config) -> pd.DataFrame:
    edges_df = pd.read_csv(config.edge_list_url, delimiter=" ", header=None)
    edges_df.columns = ["src_node_id", "dest_node_id"]
    node_mapping_dict = {}
    i = 0
    for node_id in set(edges_df["src_node_id"].tolist() + edges_df["dest_node_id"].tolist()):
        node_mapping_dict[node_id] = i
        i += 1
    edges_df["src_node_id"] = edges_df["src_node_id"].map(node_mapping_dict)
    edges_df["dest_node_id"] = edges_df["dest_node_id"].map(node_mapping_dict)

    features_df = pd.read_csv(config.label_url, delimiter=" ")
    features_df.rename(columns={"node": "node_id"}, inplace=True)
    features_df["node_id"] = features_df["node_id"].map(node_mapping_dict)

    G = ig.Graph(edges=edges_df.values.tolist())
    features_df["degree"] = G.degree()

    nodes = sorted(list(set(edges_df["src_node_id"].tolist() + edges_df["dest_node_id"].tolist())))
    mapping_df = pd.DataFrame({"node_id": nodes, "graph_id": 0})

    return edges_df, features_df, mapping_df


def generate_egonets(config, edges_df, features_df, mapping_df):
    graph_io = GraphIO()
    graph_collection = graph_io.load_from_dfs(
        edges_df=edges_df,
        node_graph_df=mapping_df,
        node_features_df=features_df,
        graph_type=config.graph_type,
        filter_largest_component=config.filter_largest_component,
    )

    # Global features
    global_structural_features = StructuralNodeFeatures(
        graph_collection=graph_collection,
        show_progress=False,
        suffix="global",
        feature_list=config.global_structural_features["feature_list"],
        feature_vector_length=config.global_structural_features["feature_vector_length"],
    ).compute()
    graph_collection.add_node_features(global_structural_features.features_df)

    # Egonet collection
    egonet_collection = EgonetCollection(egonet_feature_target=config.egonet_feature_target, skip_features=config.skip_features)
    egonet_collection.compute_k_hop_egonets(graph_collection, config.k_hop)

    return graph_collection, egonet_collection, global_structural_features


def generate_embeddings(config, global_structural_features, graph_collection, egonet_collection):
    # Local structural features
    structural_node_features = StructuralNodeFeatures(
        graph_collection=egonet_collection,
        show_progress=False,
        suffix="local",
        feature_list=config.egonet_structural_features["feature_list"],
        feature_vector_length=config.egonet_structural_features["feature_vector_length"],
    )
    structural_features = structural_node_features.compute()

    # Node features
    node_features = NodeFeatures(
        egonet_collection,
        feature_list=global_structural_features.feature_columns + config.egonet_node_features["feature_list"],
        show_progress=False,
    )
    features = node_features.compute()

    # Embeddings
    emb_builder = EmbeddingBuilder(
        graph_collection=egonet_collection,
        structural_features=structural_features,
        features=features,
        embeddings_dimension=config.embeddings["embeddings_dimension"],
    )
    embeddings = emb_builder.compute(config.embeddings["strategy"])

    # Dataset
    dataset = GraphDataset(egonet_collection, embeddings, standardize=False)

    return embeddings, dataset
