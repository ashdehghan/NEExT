Basic Graph Analysis
==================

This example demonstrates the basic workflow of using NEExT to analyze graph data, including:
loading data, computing node features, creating graph embeddings, and analyzing feature importance.

Loading Graph Data
----------------

First, we'll load some graph data from CSV files. We're using the NCI1 dataset, which is a collection
of chemical compounds represented as graphs, where each graph is labeled as either active or inactive
against non-small cell lung cancer.

.. code-block:: python

    from NEExT import NEExT
    import numpy as np

    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Define paths to data files
    edge_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/edges.csv"
    node_graph_mapping_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/node_graph_mapping.csv"
    graph_label_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/NCI1/graph_labels.csv"

    # Load data with node reindexing and largest component filtering
    graph_collection = nxt.read_from_csv(
        edges_path=edge_file,
        node_graph_mapping_path=node_graph_mapping_file,
        graph_label_path=graph_label_file,
        reindex_nodes=True,
        filter_largest_component=True,
        graph_type="networkx"
    )

Computing Node Features
---------------------

Next, we'll compute various node-level features for each graph. These features capture both local
and global structural properties of the nodes.

.. code-block:: python

    # Compute node features
    features = nxt.compute_node_features(
        graph_collection=graph_collection,
        feature_list=["all"],  # Compute all available features
        feature_vector_length=3,  # Number of hops for neighborhood aggregation
        show_progress=True
    )

    # Normalize features for better model performance
    features.normalize(type="StandardScaler")

Creating Graph Embeddings
-----------------------

Now we'll create graph-level embeddings using the computed node features. These embeddings
will represent each graph as a fixed-size vector, making them suitable for machine learning.

.. code-block:: python

    # Compute graph embeddings
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=graph_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=3,
        random_state=42
    )

Training and Evaluating Models
----------------------------

With our graph embeddings, we can now train a machine learning model to classify the graphs.

.. code-block:: python

    # Train a classification model
    model_results = nxt.train_ml_model(
        graph_collection=graph_collection,
        embeddings=embeddings,
        model_type="classifier",
        sample_size=50,  # Number of train/test splits
        balance_dataset=False
    )

    # Print model results
    print(f"Average Accuracy: {np.mean(model_results['accuracy']):.4f}")
    print(f"Average F1 Score: {np.mean(model_results['f1_score']):.4f}")

Analyzing Feature Importance
-------------------------

Finally, we'll analyze which node features are most important for the classification task.
We'll use the fast supervised method which is more efficient than the greedy approach.

.. code-block:: python

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

The feature importance results show which node features contribute most to the model's
performance, ranked from most to least important. This can help in feature selection
and understanding which structural properties are most relevant for the task. 