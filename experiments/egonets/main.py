import argparse
import importlib
import multiprocessing as mp
from functools import partial
import traceback

import mlflow
from embedding_utils import (
    add_positional_features,
    compute_embedding,
    compute_global_features,
    compute_local_node_features,
    compute_local_structural_features,
    node2vec_embedding,
)
from modeling_utils import make_charts, run_experiments, train_random_model

from NEExT.collections import EgonetCollection
from NEExT.datasets import GraphDataset
from NEExT.io import GraphIO
from NEExT.outliers.benchmark_utils.data_loading import load_abcdo_data


def load_and_preprocess_data(dataset_name: str):
    """Loads and preprocesses the ABCD dataset."""
    edges_df, mapping_df, features_df, _ = load_abcdo_data(dataset_name, hide_frac={0: 0, 1: 0})
    community_id = features_df["community_id"]
    features_df = features_df.drop(columns=["random_community_feature", "community_id"])
    return edges_df, mapping_df, features_df, community_id


def evaluation_loop(params, data_path):
    with mlflow.start_run():
        try:
            print(params)
            partial_metrics = {}
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
                global_structural_node_features = compute_global_features(params, partial_metrics, graph_collection)

                print("Egonet building")
                egonet_collection = EgonetCollection(
                    egonet_feature_target=params["egonet_target"],
                    skip_features=params["egonet_skip_features"],
                )
                egonet_collection.compute_k_hop_egonets(graph_collection, params["egonet_k_hop"])

                print("local structural")
                structural_features = compute_local_structural_features(params, partial_metrics,egonet_collection)

                print("local features")
                features = compute_local_node_features(params, egonet_collection, global_structural_node_features)

                print("positions")
                structural_features, features = add_positional_features(params, egonet_collection, structural_features, features)

                print("embedding")
                embeddings = compute_embedding(params, egonet_collection, structural_features, features)

            dataset = GraphDataset(egonet_collection, embeddings)

            run_experiments(params, partial_metrics, dataset)

            make_charts(params, features_df, community_id, dataset)
        except Exception as e:
            print('RUN FAILED!')
            print(traceback.format_exc())
            mlflow.log_params(params | {"status": "FAILURE"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="myprogram")
    parser.add_argument("--exp", type=str, help="Name of experiment configuration file")
    parser.add_argument("--workers", type=int, default=4, help="Name of experiment configuration file")
    args = parser.parse_args()

    params_module = importlib.import_module(f"parameters.{args.exp}")

    params = [dict(param) for param in params_module.params]

    data_path = "abcdo_data_1000_200_0.3"
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    experiment_name = f"/{data_path}_{params_module.experiment}"
    mlflow.set_experiment(experiment_name)

    fun = partial(evaluation_loop, data_path=data_path)
    if args.workers == 1:
        for param in params:
            fun(param)
    else:
        with mp.Pool(processes=args.workers) as pool:
            pool.map(fun, params)
