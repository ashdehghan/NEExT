import sys
import numpy as np
import pandas as pd
from experiment_utils.data_loading import load_abcdo_data, load_pygod_data
from experiment_utils.embed import build_embeddings, build_features
from experiment_utils.supervised import supervised_eval
from experiment_utils.unsupervised import unsupervised_eval

from NEExT.collections import EgonetCollection
from NEExT.io import GraphIO
from NEExT.outliers import GraphDataset


# MODE = "supervised"
# MODEL = "lgbm"
# HIDE_FRAC = {0: 0.8, 1: 0.8}


def load_data(data, outlier_mode, hide_frac):
    graph_io = GraphIO()

    if data.startswith("abcd"):
        edges_df, mapping_df, features_df, ground_truth_df = load_abcdo_data(name=data, hide_frac=hide_frac)
        graph_data = {
            "target": "is_outlier",
            "skip_features": ["random_community_feature", "community_id"],
            "feature_list": [],
        }
    else:
        edges_df, mapping_df, features_df, ground_truth_df = load_pygod_data(name=data, outlier_mode=outlier_mode, hide_frac=hide_frac)
        graph_data = {
            "target": "is_outlier",
            "skip_features": [],
            "feature_list": [i for i in features_df.columns[1:-1]],
        }

    return graph_io, edges_df, mapping_df, features_df, ground_truth_df, graph_data


def data_processing(graph_io, edges_df, mapping_df, features_df, graph_data):
    graph_collection = graph_io.load_from_dfs(
        edges_df=edges_df,
        node_graph_df=mapping_df,
        node_features_df=features_df,
        graph_type="igraph",
        filter_largest_component=False,
    )
    subgraph_collection = EgonetCollection()
    subgraph_collection.create_egonets_from_graphs(
        graph_collection=graph_collection,
        egonet_target=graph_data["target"],
        egonet_algorithm="k_hop_egonet",
        skip_features=graph_data["skip_features"],
        max_hop_length=1,
    )
    structural_features, features = build_features(subgraph_collection, feature_vector_length=5, feature_list=graph_data["feature_list"])

    embeddings = build_embeddings(
        subgraph_collection,
        structural_features,
        features,
        strategy="structural_embedding",
        structural_embedding_dimension=5,
        feature_embedding_dimension=5,
        embedding_algorithm="approx_wasserstein",
        # approx_wasserstein, wasserstein, sinkhornvectorizer
    )
    dataset = GraphDataset(subgraph_collection, embeddings, standardize=False)
    return dataset

datasets = [
    # abcdo benchmark
    # "abcdo_data_1000_50_0.1",
    # "abcdo_data_1000_50_0.3",
    # "abcdo_data_1000_50_0.7",
    # "abcdo_data_1000_100_0.1",
    # "abcdo_data_1000_100_0.3",
    # "abcdo_data_1000_100_0.7",
    # "abcdo_data_1000_200_0.5",
    # "abcdo_data_1000_200_0.1",
    # "abcdo_data_5000_50_0.1",
    # "abcdo_data_5000_50_0.3",
    # "abcdo_data_5000_50_0.7",
    # "abcdo_data_5000_100_0.1",
    # "abcdo_data_5000_100_0.3",
    # "abcdo_data_5000_100_0.7",
    # "abcdo_data_5000_200_0.5",
    # "abcdo_data_5000_200_0.1",
    # # pygod benchmark
    # "gen_100",
    # "gen_500",
    # "gen_1000",
    # "gen_5000",
    # "inj_cora",
    "inj_amazon",
    "books"    
    "reddit",
    "weibo",
]
outliers = ["any", 'structural', 'contextual']

models = {"unsupervised": ["IF", "LOF"], "supervised": ["cosine", "knn", "lgbm"]}

fracs = {
    "unsupervised": [{0: 1, 1: 1}],
    "supervised": [{0: 0.95, 1: 0.95}, {0: 0.9, 1: 0.9}, {0: 0.8, 1: 0.8}, {0: 0.5, 1: 0.5}, {0: 0.25, 1: 0.25}, {0: 0.1, 1: 0.1}],
}


def main():
    results = []
    columns = ["data", "outlier_mode", "model", "score", "score_std", "score_max", "hide_frac_0", "hide_frac_1", "comment"]

    for DATA in datasets:
        for OUTLIER_MODE in outliers:
            for MODE in ["unsupervised", "supervised"]:
                for HIDE_FRAC in fracs[MODE]:
                    graph_io, edges_df, mapping_df, features_df, ground_truth_df, graph_data = load_data(DATA, OUTLIER_MODE, HIDE_FRAC)
                    dataset = data_processing(graph_io, edges_df, mapping_df, features_df, graph_data)
                    for MODEL in models[MODE]:
                        try:
                            print(DATA, OUTLIER_MODE, MODEL, HIDE_FRAC)

                            if MODE == "supervised":
                                if dataset.y_labeled.sum() <= 4:
                                    _data = (DATA, OUTLIER_MODE, MODEL, np.nan, np.nan, np.nan, HIDE_FRAC[0], HIDE_FRAC[1], "not enough negative samples")
                                    print(_data)
                                    results.append(_data)
                                    continue

                                _, score = supervised_eval(MODEL, ground_truth_df, dataset)
                                _data = (DATA, OUTLIER_MODE, MODEL, score, np.nan, np.nan, HIDE_FRAC[0], HIDE_FRAC[1], "")
                                print(_data)
                                results.append(_data)
                            elif MODE == "unsupervised":
                                _, score_mean, score_std, score_max = unsupervised_eval(MODEL, ground_truth_df, dataset)
                                _data = (DATA, OUTLIER_MODE, MODEL, score_mean, score_std, score_max, HIDE_FRAC[0], HIDE_FRAC[1], "")
                                print(_data)
                                results.append(_data)
                        except:
                            _data = (DATA, OUTLIER_MODE, MODEL, np.nan, np.nan, np.nan, HIDE_FRAC[0], HIDE_FRAC[1], "model train failed")
                            print(_data)
                            results.append(_data)
                        pd.DataFrame(results, columns=columns).to_parquet("results.parquet")


if __name__ == "__main__":
    main()
