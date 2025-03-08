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

def run_experiment(dataset_name, dataset_source):
    """Run the node sampling experiment."""
    
    # Define data URLs
    edge_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/edges.csv"
    node_graph_mapping_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/node_graph_mapping.csv"
    graph_label_file = f"https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/{dataset_source}/csv_format/{dataset_name}/graph_labels.csv"

    # Initialize NEExT framework
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Define sampling rates to test
    sampling_rates = np.arange(1.0, 0.1, -0.1)  # From 1.0 to 0.1
    sample_size = 3  # Number of train/test splits for each rate
    sample_rates_loop = 2
    # Store results
    results = {
        'sampling_rate': [],
        'accuracy': []
    }

    # Run experiment for each sampling rate
    for rate in tqdm(sampling_rates, desc="Testing sampling rates"):
        for i in range(sample_rates_loop):
            np.random.seed(i)
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
            
            # Compute node features
            features = nxt.compute_node_features(
                graph_collection=graph_collection,
                feature_list=["page_rank"],
                feature_vector_length=4,
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

def visualize_results(results, base_line):
    """Create visualizations of the experiment results."""
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    fig = px.box(results_df, x='sampling_rate', y='accuracy')
    fig.update_layout(paper_bgcolor='white')
    fig.update_layout(plot_bgcolor='white')
    fig.update_yaxes(color='black')
    fig.update_layout(
        yaxis = dict(
            title = "Accuracy",
            zeroline=True,
            showline = True,
            linecolor = 'black',
            mirror=True,
            linewidth = 2
        ),
        xaxis = dict(
            title = "Datasets",
            mirror=True,
            zeroline=True,
            showline = True,
            linecolor = 'black',
            linewidth = 2,
        ),
        width=500,
        height=500,
        font=dict(
        size=15,
        color="black")
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='#e3e1e1')
    fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='grey', range=[0, 1])
    fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=1.0)
    fig.add_hline(y=base_line, line_width=3, line_dash="dash", line_color="red")
    fig.show()

def main():
    """Main function to run the experiment and create visualizations."""
    datasets = ["BZR", "MUTAG", "NCI1", "IMDB", "PROTEINS"]
    for dataset in datasets:
        results = run_experiment(dataset_name=dataset, dataset_source="real_world_networks")
        results["dataset"] = dataset
        # visualize_results(results, base_line=0.74)


if __name__ == '__main__':
    main() 