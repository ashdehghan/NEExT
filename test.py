from NEExT import NEExT
import numpy as np

def main():
    # Define data URLs
    edge_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/edges.csv"
    node_graph_mapping_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/node_graph_mapping.csv"
    graph_label_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/graph_labels.csv"

    # Initialize NEExT and set logging level
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Load data with node reindexing and largest component filtering
    graph_collection = nxt.read_from_csv(
        edges_path=edge_file,
        node_graph_mapping_path=node_graph_mapping_file,
        graph_label_path=graph_label_file,
        reindex_nodes=True,
        filter_largest_component=True,
        graph_type="igraph",
        node_sample_rate=1.0
    )

    # Print collection info using the new describe method
    print("\nGraph Collection Info:")
    print(graph_collection.describe())
    
    # Compute node features
    features = nxt.compute_node_features(
        graph_collection=graph_collection,
        feature_list=["all"],
        feature_vector_length=3,
        show_progress=True
    )

    # Normalize features if desired
    features.normalize(type="StandardScaler")

    # Print feature information
    print("\nComputed Node Features:")
    print(f"Number of nodes: {len(features.features_df)}")
    print(f"Features computed: {list(features.features_df.columns)}")
    print("\nSample of computed features:")
    print(features.features_df.head())

    # Compute graph embeddings
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=graph_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=3,
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
    print(f"Average Accuracy: {np.mean(model_results['accuracy']):.4f}")
    print(f"Average F1 Score: {np.mean(model_results['f1_score']):.4f}")
    print(f"Classes: {model_results['classes']}")

    # Compute feature importance
    importance_df = nxt.compute_feature_importance(
        graph_collection=graph_collection,
        features=features,
        feature_importance_algorithm="supervised_fast",
        embedding_algorithm="approx_wasserstein",
        n_iterations=5
    )

    # Print feature importance results
    print("\nFeature Importance Results:")
    print(importance_df)

if __name__ == '__main__':
    main()
