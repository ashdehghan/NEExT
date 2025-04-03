from NEExT import NEExT
import numpy as np
import time
import pandas as pd
import plotly.express as px
import umap

def main():
    # Define data URLs
    dataset = "NCI1"
    dataset_source = "real_world_networks"

    edges = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/edges.csv"
    node_graph_mapping = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/node_graph_mapping.csv"
    graph_labels = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset}/graph_labels.csv"

    # Initialize NEExT and set logging level
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Load data with node reindexing and largest component filtering
    print("\nLoading data...")
    graph_collection = nxt.read_from_csv(
        edges_path=edges,
        node_graph_mapping_path=node_graph_mapping,
        graph_label_path=graph_labels,
        reindex_nodes=True,
        filter_largest_component=True,
        graph_type="networkx",
        node_sample_rate=1.0
    )

    # Print collection info using the describe method
    print("\nGraph Collection Info:")
    print(graph_collection.describe())
    
    # Time the node feature computation
    start_time = time.time()
    
    # Compute node features
    print("\nComputing node features...")
    features = nxt.compute_node_features(
        graph_collection=graph_collection,
        feature_list=["all"],
        feature_vector_length=3,
        show_progress=True
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
    
    # Normalize features
    print("\nNormalizing features...")
    features.normalize(type="StandardScaler")

    # Compute graph embeddings
    print("\nComputing graph embeddings...")
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
    
    # Use UMAP to reduce the embeddings to 2D for visualization
    print("\nReducing embeddings to 2D using UMAP...")
    # Extract embeddings as numpy array for UMAP
    embedding_data = embeddings.embeddings_df[embeddings.embedding_columns].values
    
    # Create and fit UMAP model
    umap_model = umap.UMAP(n_neighbors=15, 
                          min_dist=0.1, 
                          n_components=2, 
                          random_state=42)
    embedding_2d = umap_model.fit_transform(embedding_data)
    
    # Create a DataFrame with the 2D embeddings and labels
    viz_df = pd.DataFrame({
        'UMAP1': embedding_2d[:, 0],
        'UMAP2': embedding_2d[:, 1],
        'graph_id': embeddings.embeddings_df['graph_id']
    })
    
    # Add class labels from graph collection
    graph_labels_dict = {g.graph_id: g.graph_label for g in graph_collection.graphs}
    viz_df['label'] = viz_df['graph_id'].map(graph_labels_dict)
    
    # Plot the 2D embeddings with plotly
    print("\nCreating 2D visualization of embeddings...")
    fig = px.scatter(
        viz_df, 
        x='UMAP1', 
        y='UMAP2', 
        color='label',
        title=f'UMAP Visualization of {dataset} Graph Embeddings',
        labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2', 'label': 'Class'},
        hover_data=['graph_id']
    )
    
    # Improve the layout
    fig.update_layout(
        legend_title_text='Class',
        plot_bgcolor='white',
        width=900,
        height=700
    )
    
    # Show the plot
    fig.show()
    
    # Train a classification model
    print("\nTraining classification model...")
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

    # Compute feature importance
    print("\nComputing feature importance...")
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