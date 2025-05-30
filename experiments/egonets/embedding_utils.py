import time

import mlflow
import pandas as pd
from NEExT.builders import EmbeddingBuilder
from NEExT.collections.egonet_collection import EgonetCollection
from NEExT.features import NodeFeatures, StructuralNodeFeatures


def node2vec_embedding(graph_collection):
    from fastnode2vec import Graph, Node2Vec

    from NEExT.embeddings.embeddings import Embeddings

    embeddings = []
    for graph in graph_collection.graphs:
        graph_n2v = Graph(graph.G.get_edgelist(), directed=False, weighted=False)
        n2v = Node2Vec(graph_n2v, dim=4, walk_length=100, window=10, p=2.0, q=0.5, workers=2)
        n2v.train(epochs=2)

        embedding_columns = [f"n2v_dim_{i}" for i in range(4)]
        embedding_df = pd.DataFrame(n2v.wv[graph.nodes], columns=embedding_columns)
        embeddings.append(embedding_df)

    embeddings = pd.concat(embeddings, axis=0, ignore_index=True)
    embeddings["graph_id"] = list(range(len(embeddings)))
    embeddings = Embeddings(embeddings, "n2v", embedding_columns)
    return embeddings


def compute_global_features(params, partial_metrics, graph_collection):
    start = time.time()
    global_structural_node_features = StructuralNodeFeatures(
        graph_collection=graph_collection,
        show_progress=params["show_progress"],
        suffix="global",
        feature_list=params["global_structural_feature_list"],
        feature_vector_length=params["global_feature_vector_length"],
        n_jobs=1,
    ).compute()
    global_structural_time = time.time() - start
    graph_collection.add_node_features(global_structural_node_features.features_df)
    partial_metrics["global_structural_time"] = global_structural_time
    # mlflow.log_metric("global_structural_time", global_structural_time)
    return global_structural_node_features


def compute_local_structural_features(params, partial_metrics, egonet_collection):
    start = time.time()
    local_structural_node_features = StructuralNodeFeatures(
        graph_collection=egonet_collection,
        show_progress=params["show_progress"],
        suffix="local",
        feature_list=params["local_structural_feature_list"],
        feature_vector_length=params["local_feature_vector_length"],
        n_jobs=1,
    )
    structural_features = local_structural_node_features.compute()
    local_structural_time = time.time() - start
    partial_metrics["local_structural_time"] = local_structural_time
    # mlflow.log_metric("local_structural_time", local_structural_time)
    return structural_features


def compute_local_node_features(params, egonet_collection, global_structural_node_features):
    node_features = NodeFeatures(
        egonet_collection,
        feature_list=global_structural_node_features.feature_columns + params["local_node_features"],
        show_progress=params["show_progress"],
        n_jobs=1,
    )
    features = node_features.compute()
    return features


def add_positional_features(
    params,
    egonet_collection: EgonetCollection,
    structural_features: StructuralNodeFeatures,
    features: NodeFeatures,
):
    egonet_position_features = egonet_collection.compute_egonet_positionaL_features(params["egonet_position"], params["position_one_hot"])

    if not params["position_as_vector"]:
        # mode that encodes position into egonet features
        if len(features.feature_columns) > 0:
            features.features_df = features.features_df.merge(egonet_position_features.features_df, on=["graph_id", "node_id"])
            for column in features.feature_columns:
                features.features_df[column] = features.features_df[column] * features.features_df["egonet_position"]

            features.features_df = features.features_df.drop(columns=["egonet_position"])
        if len(structural_features.feature_columns) > 0:
            structural_features.features_df = structural_features.features_df.merge(egonet_position_features.features_df, on=["graph_id", "node_id"])
            for column in structural_features.feature_columns:
                structural_features.features_df[column] = structural_features.features_df[column] * structural_features.features_df["egonet_position"]

            structural_features.features_df = structural_features.features_df.drop(columns=["egonet_position"])
    else:
        if len(features.feature_columns) > 0:
            features += egonet_position_features
        elif len(structural_features.feature_columns) > 0:
            structural_features += egonet_position_features

    return structural_features, features


def compute_embedding(params, egonet_collection, structural_features, features):
    emb_builder = EmbeddingBuilder(
        graph_collection=egonet_collection,
        structural_features=structural_features,
        features=features,
        embeddings_dimension=params["embeddings_dimension"],
    )
    embeddings = emb_builder.compute(params["embeddings_strategy"])
    return embeddings
