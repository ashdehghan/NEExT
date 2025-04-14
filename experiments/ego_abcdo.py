import argparse
import logging
from collections import Counter

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import umap
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from NEExT.builders import EmbeddingBuilder
from NEExT.collections import EgonetCollection
from NEExT.datasets import GraphDataset
from NEExT.features import NodeFeatures, StructuralNodeFeatures
from NEExT.io import GraphIO
from NEExT.outliers.benchmark_utils.data_loading import load_abcdo_data


def main():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)  # Defaults to console (stderr)
    logging.info("Logging to console")

    parsed_arguments = parse_args()
    # You can now access the arguments like this:
    for key, value in vars(parsed_arguments).items():
        logging.info(f"{key}: {value}")

    local_feature_vector_length = parsed_arguments.egonet_k_hop
    
    graph_io = GraphIO()
    edges_df, mapping_df, features_df = load_and_preprocess_data(parsed_arguments.data_path)

    # Compute global structural node features, add them as graph node features
    logging.info("Started computing global structural node features")
    graph_collection = graph_io.load_from_dfs(
        edges_df=edges_df,
        node_graph_df=mapping_df,
        node_features_df=features_df,
        graph_type="igraph",
        filter_largest_component=parsed_arguments.filter_largest_component,
    )

    global_structural_node_features = StructuralNodeFeatures(
        graph_collection=graph_collection,
        show_progress=parsed_arguments.show_progress,
        suffix="global",
        feature_list=parsed_arguments.global_structural_feature_list,
        feature_vector_length=parsed_arguments.global_feature_vector_length,
        n_jobs=1,
    ).compute()
    graph_collection.add_node_features(global_structural_node_features.features_df)
    logging.info("Finished computing global structural node features")

    egonet_collection = EgonetCollection(egonet_feature_target=parsed_arguments.egonet_target, skip_features=parsed_arguments.egonet_skip_features)
    logging.info("Started building egonets")
    egonet_collection.compute_k_hop_egonets(graph_collection, parsed_arguments.egonet_k_hop)
    logging.info("Finished building egonets")

    logging.info("Started computing local strutural node features")
    local_structural_node_features = StructuralNodeFeatures(
        graph_collection=egonet_collection,
        show_progress=parsed_arguments.show_progress,
        suffix="local",
        feature_list=parsed_arguments.local_structural_feature_list,
        feature_vector_length=local_feature_vector_length,
        n_jobs=1,
    )
    structural_features = local_structural_node_features.compute()
    logging.info("Finished computing local strutural node features")

    logging.info("Started computing local node features")
    node_features = NodeFeatures(
        egonet_collection,
        feature_list=global_structural_node_features.feature_columns + parsed_arguments.local_node_features,
        show_progress=parsed_arguments.show_progress,
        n_jobs=1,
    )
    features = node_features.compute()
    logging.info("Finished computing local node features")

    logging.info("Started computing embeddings")
    emb_builder = EmbeddingBuilder(
        graph_collection=egonet_collection,
        structural_features=structural_features,
        features=features,
        embeddings_dimension=parsed_arguments.embeddings_dimension,
    )
    embeddings = emb_builder.compute(parsed_arguments.embeddings_strategy)
    logging.info("Finished computing embeddings")

    dataset = GraphDataset(egonet_collection, embeddings)

    logging.info("Started running experiment")
    results = run_experiment(dataset)
    results = results.groupby("name").agg(
        auc_mean=pd.NamedAgg(column="auc", aggfunc=aggfunc),
        auc_std=pd.NamedAgg(column="auc", aggfunc=aggfunc),
        precision_mean=pd.NamedAgg(column="precision", aggfunc=aggfunc),
        precision_std=pd.NamedAgg(column="precision", aggfunc=aggfunc),
    ).reset_index()
    results = pd.concat(
        [pd.DataFrame([dict(vars(parsed_arguments).items()) for _ in range(len(results))]),results], axis=1
    )
    print(results)
    logging.info("Finished running experiment")

    pq.write_to_dataset(results, root_path=parsed_arguments.output_path)


# --- Argument Parsing Function ---
def parse_args():
    """Parses command-line arguments for the graph embedding experiment."""
    parser = argparse.ArgumentParser(description="Run graph embedding and egonet analysis.")

    # Input/Output Arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="abcdo_data_1000_200_0.3",
        help="Path or identifier for the input dataset. (Default: abcdo_data_1000_200_0.3)",
    )
    parser.add_argument(
        "--output-path", type=str, default="results/ego_abcdo.parquet", help="Path to save the aggregated results parquet file. (Default: ego_abcdo.parquet)"
    )

    # Egonet Arguments
    parser.add_argument(
        "--egonet-target", type=str, default="is_outlier", help="Name of the target feature column for egonet analysis. (Default: is_outlier)"
    )
    parser.add_argument(
        "--egonet-skip-features",
        nargs="*",  # 0 or more arguments, space-separated
        default=[],
        help="List of feature names to skip during egonet creation. (Default: empty list)",
    )
    parser.add_argument("--egonet-k-hop", type=int, default=1, help="Number of hops (k) to define the neighborhood for egonets. (Default: 1)")

    # Global Structural Feature Arguments
    parser.add_argument(
        "--global-structural-feature-list",
        nargs="+",  # 1 or more arguments, space-separated
        default=["all"],
        help="List of global structural features to compute ('all' or specific names). (Default: ['all'])",
    )
    parser.add_argument(
        "--global-feature-vector-length", type=int, default=3, help="Dimensionality/length of the global structural feature vector. (Default: 3)"
    )

    # Local Structural Feature Arguments
    parser.add_argument(
        "--local-structural-feature-list",
        nargs="+",  # 1 or more arguments, space-separated
        default=["all"],
        help="List of local structural features to compute ('all' or specific names). (Default: ['all'])",
    )
    # Note: local_feature_vector_length is derived from egonet_k_hop in the original code,
    # so it's not included as a separate argument here. It would be set inside main().

    # Local Node Feature Arguments
    parser.add_argument(
        "--local-node-features",
        nargs="*",  # 0 or more arguments, space-separated
        default=[],
        help="List of original node features to include locally in egonets. (Default: empty list)",
    )

    # Embedding Arguments
    parser.add_argument("--embeddings-dimension", type=int, default=5, help="Dimension of the computed node embeddings. (Default: 5)")
    parser.add_argument(
        "--embeddings-strategy",
        type=str,
        default="feature_embeddings",
        # Example choices - adjust if needed
        choices=["feature_embeddings", "structural_embeddings", "combined_embeddings"],
        help="Strategy to use for computing node embeddings. (Default: feature_embeddings)",
    )

    # Processing Flags
    parser.add_argument(
        "--filter-largest-component",
        action="store_true",  # Makes it a boolean flag, default is False when not present
        help="If set, only process the largest connected component of the main graph.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",  # Boolean flag, default is False when not present
        help="If set, display progress bars during long computations.",
    )
    # Comment id Arguments
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Comment for analysis",
    )
    # Parse the arguments from the command line
    args = parser.parse_args()
    return args


def aggfunc(x: np.ndarray):
    return np.round(np.mean(x), 3)


def run_experiment(dataset: GraphDataset, n_runs=10):
    """
    Run experiment n_runs times, create train test splits, train a logistic regression and lgbm classifier using dataset class to predict the outliers, evaluate using auc
    """
    results = []
    for i in range(n_runs):
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.X_labeled,
            dataset.y_labeled,
            test_size=0.2,
            random_state=i,
            stratify=dataset.y_labeled,
        )

        # Logistic Regression
        models = [
            ("lr", LogisticRegression(max_iter=1000, random_state=i)),
            ("lgbm", LGBMClassifier(random_state=i, verbose=0)),
        ]
        for name, model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)[:, 1]

            results.append(
                {
                    "run": i,
                    "name": name,
                    "auc": roc_auc_score(y_test, y_pred_prob),
                    "precision": precision_score(y_test, y_pred),
                }
            )

    return pd.DataFrame(results)


def load_and_preprocess_data(dataset_name: str):
    """Loads and preprocesses the ABCD dataset."""
    edges_df, mapping_df, features_df, _ = load_abcdo_data(dataset_name, hide_frac={0: 0, 1: 0})
    features_df = features_df.drop(columns=["random_community_feature", "community_id"])
    return edges_df, mapping_df, features_df


if __name__ == "__main__":
    main()
