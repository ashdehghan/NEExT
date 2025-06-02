from functools import partial

import mlflow
from config import Params
from joblib import Parallel, delayed

from experiment import evaluation_loop
from NEExT.outliers.benchmark_utils.data_loading import load_abcdo_data


def prepare_dataset(data: str):
    edges_df, mapping_df, features_df, _ = load_abcdo_data(data, hide_frac={0: 0, 1: 0})
    community_id = features_df["community_id"]
    features_df = features_df.drop(columns=["random_community_feature", "community_id"])
    return edges_df,mapping_df,features_df,community_id


def main():
    workers = 4
    data = "abcdo_data_1000_200_0.3"
    params = [
        Params(
            comment="random_model",
            random=True,
        ),
        Params(
            comment="node2vec",
            n2v=True,
        ),
        Params(
            comment="neext_global_all_short",
            global_structural_feature_list=["all"],
            global_feature_vector_length=1,
            embeddings_strategy="feature_embeddings",
        ),
        Params(
            comment="neext_global_all_long",
            global_structural_feature_list=["all"],
            global_feature_vector_length=5,
            embeddings_strategy="feature_embeddings",
        ),
        Params(
            comment="neext_local_betastar_short",
            local_structural_feature_list=["betastar"],
            local_feature_vector_length=1,
            egonet_k_hop=1,
            embeddings_strategy="structural_embeddings",
        ),
    ]

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    experiment_name = f"/{data}"
    mlflow.set_experiment(experiment_name)

    data_fun = partial(prepare_dataset, data=data)
    if workers == 1:
        for param in params:
            evaluation_loop(param, data_fun)
    else:
        Parallel(n_jobs=workers)(delayed(evaluation_loop)(param, data_fun) for param in params)


if __name__ == "__main__":
    main()
