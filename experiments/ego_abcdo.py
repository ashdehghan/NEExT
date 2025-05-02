from functools import partial
import multiprocessing as mp
import logging
import signal
import time

import colorcet as cc
import igraph as ig
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import umap
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from NEExT.builders import EmbeddingBuilder
from NEExT.collections import EgonetCollection
from NEExT.datasets import GraphDataset
from NEExT.features import NodeFeatures, StructuralNodeFeatures
from NEExT.io import GraphIO
from NEExT.outliers.benchmark_utils.data_loading import load_abcdo_data


def load_and_preprocess_data(dataset_name: str):
    """Loads and preprocesses the ABCD dataset."""
    edges_df, mapping_df, features_df, _ = load_abcdo_data(dataset_name, hide_frac={0: 0, 1: 0})
    community_id = features_df["community_id"]
    features_df = features_df.drop(columns=["random_community_feature", "community_id"])
    return edges_df, mapping_df, features_df, community_id


# def timeout_handler(signum, frame):
#     raise Exception("Timeout exception")


# if __name__ == "__main__":
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(int(60 * 3))
#     try:
#         main()
#     finally:
#         signal.alarm(0)


def run_training_loop(params, data_path):
    with mlflow.start_run():
        # with mlflow.start_run(nested=True):
        print(params)
        graph_io = GraphIO()
        edges_df, mapping_df, features_df, community_id = load_and_preprocess_data(data_path)
        graph_collection = graph_io.load_from_dfs(
            edges_df=edges_df,
            node_graph_df=mapping_df,
            node_features_df=features_df,
            graph_type="igraph",
            filter_largest_component=params["filter_largest_component"],
        )
        if params["random"]:
            train_random_model(params, features_df)
            return None

        elif params["n2v"]:
            embeddings = node2vec_embedding(graph_collection)
            egonet_collection = EgonetCollection(egonet_feature_target=params["egonet_target"], skip_features=params["egonet_skip_features"])
            egonet_collection.compute_k_hop_egonets(graph_collection, 0)

        else:
            print("global")
            global_structural_node_features = compute_global_features(params, graph_collection)

            print("Egonet building")
            egonet_collection = EgonetCollection(
                egonet_feature_target=params["egonet_target"],
                skip_features=params["egonet_skip_features"],
            )
            egonet_collection.compute_k_hop_egonets(graph_collection, params["egonet_k_hop"])

            print("local structural")
            structural_features = compute_local_structural_features(params, egonet_collection)

            print("local features")
            features = compute_local_node_features(params, egonet_collection, global_structural_node_features)

            print("positions")
            structural_features, features = add_positional_features(params, egonet_collection, structural_features, features)

            print("embedding")
            embeddings = compute_embedding(params, egonet_collection, structural_features, features)

        dataset = GraphDataset(egonet_collection, embeddings)

        signature = infer_signature(dataset.X_labeled, dataset.y_labeled)

        run_experiments(params, dataset, signature)

        make_charts(params, features_df, community_id, dataset)


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


def make_charts(params, features_df, community_id, dataset):
    print('charts?')
    palette = sns.color_palette(cc.glasbey, n_colors=25)
    palette_short = [palette[-1], palette[0]]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.grid(True)
    x1, x2 = dataset.X_labeled.iloc[:, 0], dataset.X_labeled.iloc[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
    sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

    fig.tight_layout()
    mlflow.log_figure(fig, "embedding.png")

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.grid(True)

    reducer = umap.UMAP()
    umap_embed = reducer.fit_transform(dataset.X_labeled)

    x1, x2 = umap_embed[:, 0], umap_embed[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
    sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

    fig.tight_layout()
    mlflow.log_figure(fig, "embedding_umap.png")


def train_random_model(params, features_df):
    model = DummyClassifier(strategy="stratified", random_state=13)
    X, y = np.ones((len(features_df["is_outlier"]), 1)), features_df["is_outlier"]
    signature = infer_signature(X, y)

    extended_params = params | {"model_name": "random"}
    mlflow.log_params(extended_params)
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=i,
            stratify=y,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)[:, 1]

        mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_prob, average="micro"), step=i)
        mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_prob, average="micro"), step=i)
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_prob, average="micro"), step=i)


def run_experiments(params, dataset, signature):
    models = [
        ("lr", LogisticRegression(max_iter=1000, random_state=13)),
        ("lgbm", LGBMClassifier(random_state=13, verbose=-1)),
    ]

    for model_name, model in models:
        with mlflow.start_run(nested=True):
            extended_params = params | {"model_name": model_name}
            mlflow.log_params(extended_params)
            for i in range(10):
                print(f"Experiment run id: {i}")
                x_train, x_test, y_train, y_test = train_test_split(
                    dataset.X_labeled,
                    dataset.y_labeled,
                    test_size=0.2,
                    random_state=i,
                    stratify=dataset.y_labeled,
                )
                
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_pred_prob = model.predict_proba(x_test)[:, 1]

                mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_prob, average="micro"), step=i)
                mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_prob, average="micro"), step=i)
                mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="micro"), step=i)
    print('done?')


def compute_embedding(params, egonet_collection, structural_features, features):
    emb_builder = EmbeddingBuilder(
        graph_collection=egonet_collection,
        structural_features=structural_features,
        features=features,
        embeddings_dimension=params["embeddings_dimension"],
    )
    embeddings = emb_builder.compute(params["embeddings_strategy"])
    return embeddings


def add_positional_features(params, egonet_collection, structural_features, features):
    egonet_position_features = egonet_collection.compute_egonet_positionaL_features(params["position_one_hot"])
    if params["egonet_position"] and len(features.feature_columns) > 0:
        features += egonet_position_features
    elif params["egonet_position"] and len(structural_features.feature_columns) > 0:
        structural_features += egonet_position_features
    return structural_features, features


def compute_local_node_features(params, egonet_collection, global_structural_node_features):
    node_features = NodeFeatures(
        egonet_collection,
        feature_list=global_structural_node_features.feature_columns + params["local_node_features"],
        show_progress=params["show_progress"],
        n_jobs=1,
    )
    features = node_features.compute()
    return features


def compute_local_structural_features(params, egonet_collection):
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
    mlflow.log_metric("local_structural_time", local_structural_time)
    return structural_features


def compute_global_features(params, graph_collection):
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
    mlflow.log_metric("global_structural_time", global_structural_time)
    return global_structural_node_features


params = [
    {
        "egonet_target": "is_outlier",
        "egonet_skip_features": [],
        "filter_largest_component": True,
        "show_progress": False,
        "comment": "asd",
        #
        "egonet_k_hop": 1,
        "global_structural_feature_list": ["all"],
        "global_feature_vector_length": 4,
        "local_structural_feature_list": [],
        "local_feature_vector_length": 1,
        "local_node_features": [],
        "embeddings_dimension": 5,
        "embeddings_strategy": "feature_embeddings",
        "egonet_position": True,
        "position_one_hot": True,
        #
        "n2v": False,
        "random": True,
    },
    {
        "egonet_target": "is_outlier",
        "egonet_skip_features": [],
        "filter_largest_component": True,
        "show_progress": False,
        "comment": "asd",
        #
        "egonet_k_hop": 1,
        "global_structural_feature_list": ["all"],
        "global_feature_vector_length": 4,
        "local_structural_feature_list": [],
        "local_feature_vector_length": 1,
        "local_node_features": [],
        "embeddings_dimension": 5,
        "embeddings_strategy": "feature_embeddings",
        "egonet_position": True,
        "position_one_hot": True,
        #
        "n2v": True,
        "random": False,
    },
    {
        "egonet_target": "is_outlier",
        "egonet_skip_features": [],
        "filter_largest_component": True,
        "show_progress": False,
        "comment": "asd",
        #
        "egonet_k_hop": 1,
        "global_structural_feature_list": ["all"],
        "global_feature_vector_length": 4,
        "local_structural_feature_list": [],
        "local_feature_vector_length": 1,
        "local_node_features": [],
        "embeddings_dimension": 5,
        "embeddings_strategy": "feature_embeddings",
        "egonet_position": True,
        "position_one_hot": True,
        #
        "n2v": False,
        "random": False,
    },
]

# mlflow.config.enable_async_logging(True)

if __name__ == "__main__":
    data_path = "abcdo_data_1000_200_0.3"
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(f"/{data_path}_test")

    # fun = partial(run_training_loop, data_path=data_path)
    # for param in params:
    #     fun(param)

    fun = partial(run_training_loop, data_path=data_path)
    with mp.Pool(processes=4) as pool:
        pool.map(fun, params)