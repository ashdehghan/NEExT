from NEExT import NEExT
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def run_experiment(dataset_name):
    # Define data URLs
    base_url = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format"
    edges = f"{base_url}/{dataset_name}/edges.csv"
    node_graph_mapping = f"{base_url}/{dataset_name}/node_graph_mapping.csv"
    graph_labels = f"{base_url}/{dataset_name}/graph_labels.csv"

    # Initialize NEExT and set logging level
    nxt = NEExT()
    nxt.set_log_level("INFO")

    # Load data
    graph_collection = nxt.read_from_csv(
        edges_path=edges,
        node_graph_mapping_path=node_graph_mapping,
        graph_label_path=graph_labels,
        reindex_nodes=True,
        filter_largest_component=True,
        graph_type="networkx",
        node_sample_rate=1.0
    )

    # Compute node features
    features = nxt.compute_node_features(
        graph_collection=graph_collection,
        feature_list=["all"],
        feature_vector_length=4,
        show_progress=True
    )

    # Normalize features
    features.normalize(type="StandardScaler")

    # Compute graph embeddings
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=graph_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=len(features.feature_columns),
        random_state=42
    )

    # Train classification model
    model_results = nxt.train_ml_model(
        graph_collection=graph_collection,
        embeddings=embeddings,
        model_type="classifier",
        sample_size=50,
        balance_dataset=False
    )

    return model_results['accuracy']

def create_visualizations(results):
    # Create a figure with two subplots
    results["Source"] = "NEExT"
    external_results = pd.read_csv("./external_datasets/leaderboard_real_world_datasets_model_accuracies.csv")
    results = pd.concat([results, external_results])
    fig = px.box(results, x="Dataset", y="Accuracy", color="Source")
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
        width=900,
        height=500,
        font=dict(
        size=15,
        color="black")
            
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.78,
        xanchor="left",
        x=1.01,
        bordercolor="Black",
        borderwidth=1
    ))
    fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='#e3e1e1')
    fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='grey')
    fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=1.0)
    fig.show()

def main():
    # List of datasets to analyze
    datasets = ['IMDB', 'MUTAG', 'NCI1', 'BZR', 'PROTEINS']
    
    # Store results for each dataset
    dfs = pd.DataFrame()
    
    # Run experiments for each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        accuracies = run_experiment(dataset)
        df = pd.DataFrame()
        df["Accuracy"] = accuracies
        df["Dataset"] = dataset
        if dfs.empty:
            dfs = df
        else:
            dfs = pd.concat([dfs, df])
        print(f"Average accuracy for {dataset}: {np.mean(accuracies):.4f}")

    # Create visualizations
    create_visualizations(dfs)


if __name__ == '__main__':
    main() 