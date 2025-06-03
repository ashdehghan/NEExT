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
    return edges_df, mapping_df, features_df, community_id


def main():
    workers = 4
    data = "abcdo_data_1000_200_0.3"
    # default models
    params = [
        Params(
            comment="random_model",
            random=True,
        ),
        Params(
            comment="node2vec",
            n2v=True,
        ),
    ]
    # just ego node features
    params += [
        Params(
            comment=f"node_features_{feature}",
            global_structural_feature_list=[feature],
            global_feature_vector_length=1,
            embeddings_strategy="only_egonet_node_features",
            egonet_k_hop=0,
        )
        for feature in ["all", "betastar", "page_rank", "degree_centrality", "clustering_coefficient"]
    ]

    # global models
    MAX_K_HOP = 3

    params += [
        Params(
            comment=f"global_structural_features_{feature}_{k_hop}_{vec_len}",
            global_structural_feature_list=[feature],
            global_feature_vector_length=vec_len,
            embeddings_strategy="feature_embeddings",
            egonet_k_hop=k_hop,
        )
        for feature in [
            "all",
            "degree_centrality",
            "betastar",
            "load_centrality",
            "lsme",
            "local_efficiency",
            "betweenness_centrality",
            "closeness_centrality",
        ]
        for k_hop in range(1, MAX_K_HOP + 1)
        for vec_len in range(1, k_hop + 1)
    ]

    # local node features
    features = [
        "degree_centrality",
        "betastar",
        "load_centrality",
        "lsme",
        "local_efficiency",
        "betweenness_centrality",
        "closeness_centrality",
    ]

    k_and_length = [(1, 1), (2, 1), (2, 2)]

    params += [
        Params(
            comment=f"local_features_{k}_{i}",
            global_structural_feature_list=[],
            local_structural_feature_list=features,
            local_feature_vector_length=i,
            egonet_k_hop=k,
            embeddings_strategy="structural_embeddings",
        )
        for k, i in k_and_length
    ]
    params += [
        Params(
            comment=f"{strategy}_local_{k_hop}_{local_i}_global_{global_i}",
            global_structural_feature_list=features,
            global_feature_vector_length=global_i,
            local_structural_feature_list=features,
            local_feature_vector_length=local_i,
            egonet_k_hop=k_hop,
            embeddings_strategy=strategy,
        )
        for k_hop, local_i in k_and_length
        for strategy in ["separate_embeddings", "combined_embeddings"]
        for global_i in [1, 2]
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
