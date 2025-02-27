Quick Start
==========

This guide demonstrates how to use NEExT for graph analysis using a real-world dataset.

Basic Example
------------

.. code-block:: python

    from NEExT import NEExT
    import numpy as np

    def main():
        # Define data URLs - using the NCI1 dataset
        # NCI1 is a chemical compound dataset where each graph represents a molecule,
        # labeled as either active or inactive against non-small cell lung cancer
        edge_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/edges.csv"
        node_graph_mapping_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/node_graph_mapping.csv"
        graph_label_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/graph_labels.csv"

        # Initialize NEExT framework
        nxt = NEExT()
        nxt.set_log_level("INFO")  # Set logging level for detailed progress information

        # Load graph data from CSV files
        # - reindex_nodes=True: Ensures consistent node indexing across graphs
        # - filter_largest_component=True: Keeps only the largest connected component of each graph
        # - graph_type="networkx": Uses NetworkX as the backend (alternatively can use "igraph")
        graph_collection = nxt.read_from_csv(
            edges_path=edge_file,
            node_graph_mapping_path=node_graph_mapping_file,
            graph_label_path=graph_label_file,
            reindex_nodes=True,
            filter_largest_component=True,
            graph_type="networkx"
        )

        # Print collection info
        print("\nGraph Collection Info:")
        print(graph_collection.describe())

        # Compute node features
        # - feature_list=["all"]: Computes all available node features including:
        #   * Degree centrality: Measures node connectivity
        #   * Betweenness centrality: Measures node importance in information flow
        #   * Closeness centrality: Measures node's average distance to all others
        #   * Page rank: Measures node importance based on neighbor importance
        #   * Clustering coefficient: Measures local clustering around node
        # - feature_vector_length=3: Aggregates features from 3-hop neighborhoods
        features = nxt.compute_node_features(
            graph_collection=graph_collection,
            feature_list=["all"],
            feature_vector_length=3,
            show_progress=True
        )

        # Normalize features using StandardScaler
        # This ensures all features are on the same scale
        features.normalize(type="StandardScaler")

        # Print feature information
        print("\nComputed Node Features:")
        print(f"Number of nodes: {len(features.features_df)}")
        print(f"Features computed: {list(features.features_df.columns)}")

        # Compute graph embeddings using the Wasserstein distance
        # This creates fixed-size vector representations for each graph
        # - embedding_algorithm="approx_wasserstein": Uses approximate Wasserstein distance
        # - embedding_dimension=3: Creates 3-dimensional embeddings
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

        # Train a classification model
        # - model_type="classifier": For classification tasks (use "regressor" for regression)
        # - sample_size=50: Number of train/test splits for robust evaluation
        # - balance_dataset=False: Use original class distribution
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

        # Compute feature importance using supervised fast algorithm
        # This determines which node features are most important for the classification task
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

Understanding the Output
----------------------

The code above will produce several outputs:

1. Graph Collection Info:
   - Number of graphs in the dataset
   - Graph backend being used
   - Whether graphs have labels

2. Node Features:
   - Number of nodes across all graphs
   - List of computed features for each node
   - Features are normalized using StandardScaler

3. Graph Embeddings:
   - Number of embedded graphs
   - Dimensionality of embeddings
   - Algorithm used for embedding

4. Model Results:
   - Average accuracy across multiple train/test splits
   - Average F1 score for classification performance
   - List of unique classes in the dataset

5. Feature Importance:
   - Ranking of node features by importance
   - Performance scores for each feature
   - Total computation time

This example demonstrates the complete workflow from loading graph data to analyzing
feature importance, using NEExT's high-level interface while maintaining flexibility
and configurability at each step. 