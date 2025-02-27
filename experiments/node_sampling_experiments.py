#!/usr/bin/env python3
"""
Node Sampling Experiment

This script investigates the effect of node sampling rate on classifier accuracy
when using graph embeddings. It:
1. Loads graph data from the BZR dataset
2. Creates graph collections with different node sampling rates (1.0 to 0.1)
3. Computes node features and graph embeddings
4. Trains and evaluates classifier models
5. Visualizes the relationship between sampling rate and accuracy
"""

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from NEExT import NEExT

def run_experiment():
    """Run the node sampling experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define data URLs
    edge_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/edges.csv"
    node_graph_mapping_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/node_graph_mapping.csv"
    graph_label_file = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format/BZR/graph_labels.csv"

    # Initialize NEExT framework
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Define sampling rates to test
    sampling_rates = np.arange(1.0, 0.5, -0.1)  # From 1.0 to 0.1
    sample_size = 50  # Number of train/test splits for each rate

    # Store results
    results = {
        'sampling_rate': [],
        'accuracy': []
    }

    # Run experiment for each sampling rate
    for rate in tqdm(sampling_rates, desc="Testing sampling rates"):
        # Load data with current sampling rate
        graph_collection = nxt.read_from_csv(
            edges_path=edge_file,
            node_graph_mapping_path=node_graph_mapping_file,
            graph_label_path=graph_label_file,
            reindex_nodes=True,
            filter_largest_component=True,
            graph_type="igraph",
            node_sample_rate=rate
        )
        
        # Compute node features
        features = nxt.compute_node_features(
            graph_collection=graph_collection,
            feature_list=["all"],
            feature_vector_length=3,
            show_progress=False
        )
        
        # Get number of features
        num_features = len([col for col in features.features_df.columns 
                          if col not in ['node_id', 'graph_id']])
        
        # Compute graph embeddings
        embeddings = nxt.compute_graph_embeddings(
            graph_collection=graph_collection,
            features=features,
            embedding_algorithm="approx_wasserstein",
            embedding_dimension=num_features,
            random_state=42
        )
        
        # Train and evaluate classifier
        model_results = nxt.train_ml_model(
            graph_collection=graph_collection,
            embeddings=embeddings,
            model_type="classifier",
            sample_size=sample_size,
            balance_dataset=False
        )
        
        # Store results
        results['sampling_rate'].extend([rate] * len(model_results['accuracy']))
        results['accuracy'].extend(model_results['accuracy'])

    return results

def visualize_results(results):
    """Create visualizations of the experiment results."""
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    fig = px.box(results_df, x='sampling_rate', y='accuracy', title='Effect of Node Sampling Rate on Classifier Accuracy')
    fig.show()

def main():
    """Main function to run the experiment and create visualizations."""
    print("Starting Node Sampling Experiment...")
    results = run_experiment()
    print("\nGenerating visualizations and statistics...")
    visualize_results(results)
    print("\nExperiment complete. Results saved to:")
    print("- node_sampling_results.png")
    print("- node_sampling_summary.csv")

if __name__ == '__main__':
    main() 