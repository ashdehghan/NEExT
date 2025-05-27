from NEExT import NEExT
import numpy as np
import time
import pandas as pd

# Define your custom metric function at the TOP LEVEL of the module
def my_custom_metric(graph):
    # Example: return a DataFrame with node_id, graph_id, and a constant feature
    df = pd.DataFrame({
        'node_id': graph.nodes,
        'graph_id': graph.graph_id,
        'my_custom_feature_0': [1.0] * len(graph.nodes)
    })
    # Ensure correct column order for consistency, though your example was already correct
    return df[["node_id", "graph_id", "my_custom_feature_0"]]

def main():
    # Define data URLs

    dataset = "BZR"

    if dataset in ["IMDB", "MUTAG", "NCI1", "BZR", "PROTEINS"]:
      dataset_source = "real_world_networks"
    else:
      dataset_source = "synthetic_networks"

    edges = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/edges.csv"
    node_graph_mapping = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/node_graph_mapping.csv"
    graph_labels = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/graph_labels.csv"
    node_features = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/node_features.csv"

    # Initialize NEExT and set logging level
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Load data with node reindexing and largest component filtering
    graph_collection = nxt.read_from_csv(
        edges_path=edges,
        node_graph_mapping_path=node_graph_mapping,
        graph_label_path=graph_labels,
        node_features_path=node_features,
        reindex_nodes=True,
        filter_largest_component=True,
        graph_type="networkx",
        node_sample_rate=1.0
    )

    # Print collection info using the new describe method
    print("\nGraph Collection Info:")
    print(graph_collection.describe())
    
    # Time the node feature computation
    start_time = time.time()
    
    # The custom function is now defined globally
    my_feature_methods = [
        {"feature_name": "my_custom_metric", "feature_function": my_custom_metric}
    ]

    # Compute node features, including the custom metric
    features = nxt.compute_node_features(
        graph_collection=graph_collection,
        feature_list=["all", "my_custom_metric"],
        feature_vector_length=3,
        show_progress=True,
        my_feature_methods=my_feature_methods
    )
    
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"\nNode feature computation took {computation_time:.2f} seconds")


    # Print feature information
    print("\nComputed Node Features:")
    # Get only the feature columns (excluding node_id and graph_id)
    print(f"Number of computed features: {len(features.feature_columns)}")
    print(f"Features computed: {features.feature_columns}")
    print("\nSample of computed features:")
    print(features.features_df.head())
    # Normalize features if desired
    features.normalize(type="StandardScaler")

    # Compute graph embeddings
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=graph_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=len(features.feature_columns),
        random_state=42
    )

    # Print embedding information
    print("\nComputed Graph Embeddings:")
    print(f"Number of graphs: {len(embeddings.embeddings_df)}")
    print(f"Embedding dimensions: {len(embeddings.embedding_columns)}")
    print(f"Embedding algorithm: {embeddings.embedding_name}")
    print("\nSample of computed embeddings:")
    print(embeddings.embeddings_df.head())
    
    # Train a classification model
    model_results = nxt.train_ml_model(
        graph_collection=graph_collection,
        embeddings=embeddings,
        model_type="classifier",
        sample_size=50,
        balance_dataset=False
    )
    
    # Print model results
    print("\nClassification Model Results:")
    print("\nMetrics Summary:")
    print("-" * 50)
    metrics = ['accuracy', 'recall', 'precision', 'f1_score']
    for metric in metrics:
        mean_val = np.mean(model_results[metric])
        std_val = np.std(model_results[metric])
        print(f"{metric.capitalize():<10} Mean: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nClass Information:")
    print("-" * 50)
    print(f"Classes: {model_results['classes']}")
    print(f"Number of classes: {len(model_results['classes'])}")

    # # Compute feature importance
    # importance_df = nxt.compute_feature_importance(
    #     graph_collection=graph_collection,
    #     features=features,
    #     feature_importance_algorithm="supervised_fast",
    #     embedding_algorithm="approx_wasserstein",
    #     n_iterations=5
    # )

    # # Print feature importance results
    # print("\nFeature Importance Results:")
    # print(importance_df)

if __name__ == '__main__':
    main()
