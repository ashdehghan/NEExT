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
import time

def run_experiment(dataset_name, dataset_source):
    """Run the node sampling experiment."""
    
    # Define data URLs
    edge_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/edges.csv"
    node_graph_mapping_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/node_graph_mapping.csv"
    graph_label_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/graph_labels.csv"

    # Initialize NEExT framework once for all iterations
    nxt = NEExT()
    nxt.set_log_level("ERROR")

    # Define sampling rates to test
    sampling_rates = np.arange(1.0, 0.3, -0.1)  # From 1.0 to 0.1
    sample_size = 20  # Number of train/test splits for each rate
    sample_rates_loop = 5
    # Store results
    results = {
        'sampling_rate': [],
        'accuracy': [],
        'computation_time': [],  # Add computation time tracking
        'feature_computation_time': [],  # Specifically track feature computation time
        'embedding_computation_time': [],  # Specifically track embedding computation time
        'model_computation_time': []  # Specifically track model training time
    }

    # Run experiment for each sampling rate
    for rate in tqdm(sampling_rates, desc="Testing sampling rates"):
        for i in range(sample_rates_loop):
            np.random.seed(i)
            
            start_time_total = time.time()
            
            # Load data with current sampling rate
            graph_collection = nxt.read_from_csv(
                edges_path=edge_file,
                node_graph_mapping_path=node_graph_mapping_file,
                graph_label_path=graph_label_file,
                reindex_nodes=True,
                filter_largest_component=True,
                graph_type="networkx",
                node_sample_rate=rate
            )
            
            # Time feature computation
            start_time_features = time.time()
            features = nxt.compute_node_features(
                graph_collection=graph_collection,
                feature_list=["all"],
                feature_vector_length=4,
                show_progress=False
            )
            feature_time = time.time() - start_time_features
            
            # Get number of features
            num_features = len([col for col in features.features_df.columns 
                            if col not in ['node_id', 'graph_id']])
            
            # Time embedding computation
            start_time_embeddings = time.time()
            embeddings = nxt.compute_graph_embeddings(
                graph_collection=graph_collection,
                features=features,
                embedding_algorithm="approx_wasserstein",
                embedding_dimension=num_features,
                random_state=42
            )
            embedding_time = time.time() - start_time_embeddings
            
            # Time model training and evaluation
            start_time_model = time.time()
            model_results = nxt.train_ml_model(
                graph_collection=graph_collection,
                embeddings=embeddings,
                model_type="classifier",
                sample_size=sample_size,
                balance_dataset=False
            )
            model_time = time.time() - start_time_model
            
            total_time = time.time() - start_time_total
            
            # Store results
            num_accuracies = len(model_results['accuracy'])
            results['sampling_rate'].extend([rate] * num_accuracies)
            results['accuracy'].extend(model_results['accuracy'])
            results['computation_time'].extend([total_time] * num_accuracies)
            results['feature_computation_time'].extend([feature_time] * num_accuracies)
            results['embedding_computation_time'].extend([embedding_time] * num_accuracies)
            results['model_computation_time'].extend([model_time] * num_accuracies)

    return results


def main():
    """Main function to run the experiment and create visualizations."""
    all_results = pd.DataFrame()
    datasets = ["BZR", "MUTAG", "NCI1", "IMDB", "PROTEINS"]
    
    for dataset in datasets:
        print(f"\nRunning experiment for {dataset}")
        results = run_experiment(dataset_name=dataset, dataset_source="real_world_networks")
        results_df = pd.DataFrame(results)
        results_df["dataset"] = dataset
        if  all_results.empty:
            all_results = results_df
        else:
            all_results = pd.concat([all_results, results_df], ignore_index=True)
        # Combine all results and save
        all_results.to_csv("./results/sampling_rate_experiment_results.csv", index=False)

if __name__ == '__main__':
    main() 