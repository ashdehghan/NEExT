#!/usr/bin/env python3
"""
Embedding Dimension Size Experiment on Reddit 5% Dataset

This experiment tests the effect of Wasserstein embedding dimension size on classification performance.
Tests dimensions: 2, 5, 10, 15, 25, 50, and max (full feature size)

Saves intermediate results and creates scientific plots.
"""

import pickle
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
import json
from tqdm import tqdm

# Try to import plotly, fallback to basic plotting if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available, will save results as CSV only")
    PLOTLY_AVAILABLE = False

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection
from NEExT.features import Features


def extract_node_features(egonet_collection, feature_prefix='feature_'):
    """Extract original Reddit node features from egonets."""
    print(f"      Extracting Reddit node features...")
    
    feature_dfs = []
    
    # Add progress bar for egonet processing
    for egonet in tqdm(egonet_collection.graphs, desc="Processing egonets", leave=False):
        node_features_dict = {}
        
        # Get feature columns
        sample_node = list(egonet.node_attributes.keys())[0] if egonet.node_attributes else None
        if sample_node is None:
            continue
            
        feature_cols = [col for col in egonet.node_attributes[sample_node].keys() 
                       if col.startswith(feature_prefix)]
        
        # Extract features for all nodes
        for node_id in egonet.nodes:
            if node_id in egonet.node_attributes:
                node_attrs = egonet.node_attributes[node_id]
                node_features_dict[node_id] = {
                    col: node_attrs.get(col, 0.0) for col in feature_cols
                }
        
        # Aggregate features
        if node_features_dict:
            nodes_df = pd.DataFrame.from_dict(node_features_dict, orient='index')
            
            # Mean aggregation
            aggregated_features = {}
            agg_values = nodes_df.mean()
            for col in feature_cols:
                aggregated_features[f"{col}_mean"] = agg_values[col]
            
            # Create single-row DataFrame
            egonet_df = pd.DataFrame([aggregated_features])
            egonet_df['graph_id'] = egonet.graph_id
            egonet_df['node_id'] = 0
            
            feature_dfs.append(egonet_df)
    
    # Combine all
    if feature_dfs:
        combined_df = pd.concat(feature_dfs, ignore_index=True)
        feature_columns = [col for col in combined_df.columns 
                          if col not in ['graph_id', 'node_id']]
        combined_df = combined_df[['node_id', 'graph_id'] + feature_columns]
    else:
        combined_df = pd.DataFrame(columns=['node_id', 'graph_id'])
        feature_columns = []
    
    print(f"      Extracted {len(feature_columns)} Reddit features")
    
    return Features(combined_df, feature_columns)


def save_intermediate_results(results_df, timestamp):
    """Save intermediate results to CSV."""
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / f"embedding_dimensions_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"      Saved intermediate results to: {csv_path}")
    return csv_path


def create_scientific_plot(results_df, timestamp):
    """Create scientific plot using Plotly."""
    
    if not PLOTLY_AVAILABLE:
        print("      Plotly not available, skipping plot generation")
        return None
    
    # Set up the plot with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy vs Embedding Dimension",
                       "F1 Score vs Embedding Dimension",
                       "Precision vs Embedding Dimension",
                       "Recall vs Embedding Dimension"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Define colors for each model
    colors = {
        'combined': '#2E86AB',  # Blue
        'node_only': '#A23B72',  # Purple
        'structural_only': '#F18F01',  # Orange
        'random_baseline': '#C73E1D'  # Red
    }
    
    # Get unique dimensions (excluding 'max' for now)
    numeric_dims = results_df[results_df['dimension'] != 'max']['dimension'].unique()
    numeric_dims = sorted([int(d) for d in numeric_dims])
    
    # Prepare data for each model
    for model in ['combined', 'node_only', 'structural_only', 'random_baseline']:
        model_data = results_df[results_df['model'] == model].copy()
        
        # Sort by dimension (handle 'max' separately)
        numeric_data = model_data[model_data['dimension'] != 'max'].copy()
        numeric_data['dimension'] = numeric_data['dimension'].astype(int)
        numeric_data = numeric_data.sort_values('dimension')
        
        # Get max dimension data if exists
        max_data = model_data[model_data['dimension'] == 'max']
        
        # Prepare x and y values
        x_values = list(numeric_data['dimension'])
        
        # Add max dimension point if exists
        if not max_data.empty:
            max_dim_value = {
                'combined': 606,
                'node_only': 602,
                'structural_only': 4,
                'random_baseline': 100  # arbitrary for baseline
            }[model]
            x_values.append(max_dim_value)
        
        # Plot Accuracy (top-left)
        y_accuracy = list(numeric_data['accuracy'])
        y_accuracy_std = list(numeric_data['accuracy_std'])
        if not max_data.empty:
            y_accuracy.append(max_data['accuracy'].iloc[0])
            y_accuracy_std.append(max_data['accuracy_std'].iloc[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_accuracy,
                error_y=dict(type='data', array=y_accuracy_std, visible=True),
                mode='lines+markers',
                name=model.replace('_', ' ').title(),
                line=dict(color=colors[model], width=2),
                marker=dict(size=8),
                legendgroup=model,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot F1 Score (top-right)
        y_f1 = list(numeric_data['f1_score'])
        if not max_data.empty:
            y_f1.append(max_data['f1_score'].iloc[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_f1,
                mode='lines+markers',
                name=model.replace('_', ' ').title(),
                line=dict(color=colors[model], width=2),
                marker=dict(size=8),
                legendgroup=model,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot Precision (bottom-left)
        y_precision = list(numeric_data['precision'])
        if not max_data.empty:
            y_precision.append(max_data['precision'].iloc[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_precision,
                mode='lines+markers',
                name=model.replace('_', ' ').title(),
                line=dict(color=colors[model], width=2),
                marker=dict(size=8),
                legendgroup=model,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot Recall (bottom-right)
        y_recall = list(numeric_data['recall'])
        if not max_data.empty:
            y_recall.append(max_data['recall'].iloc[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_recall,
                mode='lines+markers',
                name=model.replace('_', ' ').title(),
                line=dict(color=colors[model], width=2),
                marker=dict(size=8),
                legendgroup=model,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout for scientific appearance
    fig.update_layout(
        title={
            'text': "Effect of Embedding Dimension on Classification Performance<br><sub>Reddit 5% Dataset (200 Egonets)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        template='plotly_white',
        width=1200,
        height=900,
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                title_text="Embedding Dimension" if i == 2 else "",
                gridcolor='lightgray',
                showgrid=True,
                zeroline=False,
                row=i, col=j
            )
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(title_text="Precision", row=2, col=1, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(title_text="Recall", row=2, col=2, gridcolor='lightgray', showgrid=True)
    
    # Set y-axis range [0, 1] for all metrics
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 1], row=i, col=j)
    
    # Save the plot
    results_dir = Path("experiment_results")
    html_path = results_dir / f"embedding_dimensions_plot_{timestamp}.html"
    fig.write_html(str(html_path))
    print(f"\nPlot saved to: {html_path}")
    
    # Also save as static image if kaleido is installed
    try:
        png_path = results_dir / f"embedding_dimensions_plot_{timestamp}.png"
        fig.write_image(str(png_path), width=1200, height=900, scale=2)
        print(f"Static plot saved to: {png_path}")
    except:
        print("Note: Install kaleido to save static images: pip install kaleido")
    
    # Show the plot
    fig.show()
    
    return fig


def run_embedding_dimension_experiment():
    """Run the embedding dimension experiment."""
    
    print("\n" + "="*80)
    print("EMBEDDING DIMENSION EXPERIMENT")
    print("="*80)
    print("Testing effect of Wasserstein embedding dimension on performance")
    print("Dimensions to test: 2, 5, 10, 15, 25, 50, max")
    print(f"Total experiments: 3 models × 7 dimensions = 21 experiments")
    print("="*80)
    
    # Configuration
    GRAPH_FILE = "reddit_binary_5pct.pkl"
    NUM_EGONETS = 200
    RANDOM_SEED = 42
    
    # Dimensions to test
    DIMENSIONS = [2, 5, 10, 15, 25, 50, 'max']
    
    # Random walk parameters
    WALK_LENGTH = 8
    NUM_WALKS = 4
    RESTART_PROB = 0.2
    MAX_NODES_PER_EGONET = 30
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Storage for results
    all_results = []
    
    # Overall progress tracking
    total_experiments = len(DIMENSIONS) * 3  # 3 models (excluding random baseline)
    experiment_counter = 0
    
    # Cache file paths
    cache_dir = Path("experiment_cache")
    cache_dir.mkdir(exist_ok=True)
    egonets_cache = cache_dir / "egonets_200.pkl"
    features_cache = cache_dir / "features_200.pkl"
    
    # ========== Step 1: Load/Create Egonets ==========
    if egonets_cache.exists():
        print("\n[1/3] Loading cached egonets...")
        with open(egonets_cache, 'rb') as f:
            egonet_collection = pickle.load(f)
        print(f"      Loaded {len(egonet_collection.graphs)} egonets from cache")
    else:
        print("\n[1/3] Creating egonets (will be cached for future runs)...")
        
        # Load graph
        print("      Loading Reddit 5% graph...")
        with open(GRAPH_FILE, 'rb') as f:
            graph = pickle.load(f)
        print(f"      Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}")
        
        # Create GraphCollection
        graph_collection = nxt.load_from_networkx(
            [graph],
            reindex_nodes=False,
            filter_largest_component=False,
            node_sample_rate=1.0
        )
        
        # Create Egonets
        print(f"      Creating {NUM_EGONETS} egonets with random walk...")
        start = time.time()
        
        sample_fraction = NUM_EGONETS / graph.number_of_nodes()
        
        egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
        egonet_collection.compute_k_hop_egonets(
            graph_collection=graph_collection,
            k_hop=2,
            sample_fraction=sample_fraction,
            sampling_strategy='random_walk',
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            restart_prob=RESTART_PROB,
            max_nodes_per_egonet=MAX_NODES_PER_EGONET,
            random_seed=RANDOM_SEED
        )
        
        print(f"      Created {len(egonet_collection.graphs)} egonets in {time.time() - start:.1f}s")
        
        # Cache egonets
        with open(egonets_cache, 'wb') as f:
            pickle.dump(egonet_collection, f)
        print(f"      Cached egonets to: {egonets_cache}")
    
    # Check class balance
    egonet_labels = [g.graph_label for g in egonet_collection.graphs if g.graph_label is not None]
    label_counts = Counter(egonet_labels)
    print(f"      Class balance: Serious={label_counts.get(0, 0)}, Entertainment={label_counts.get(1, 0)}")
    
    # ========== Step 2: Load/Compute Features ==========
    if features_cache.exists():
        print("\n[2/3] Loading cached features...")
        with open(features_cache, 'rb') as f:
            features_dict = pickle.load(f)
        structural_features = features_dict['structural']
        node_features = features_dict['node']
        combined_features = features_dict['combined']
        print(f"      Loaded features from cache")
    else:
        print("\n[2/3] Computing features (will be cached for future runs)...")
        
        # Structural features
        print("      Computing structural features...")
        start = time.time()
        structural_features = nxt.compute_node_features(
            graph_collection=egonet_collection,
            feature_list=[
                "degree_centrality",
                "clustering_coefficient",
                "page_rank",
                "betweenness_centrality"
            ],
            feature_vector_length=1,
            show_progress=False,
            n_jobs=-1
        )
        print(f"      Computed {len(structural_features.feature_columns)} structural features in {time.time() - start:.1f}s")
        
        # Node features
        print("      Extracting Reddit node features...")
        start = time.time()
        node_features = extract_node_features(egonet_collection)
        print(f"      Extracted {len(node_features.feature_columns)} Reddit features in {time.time() - start:.1f}s")
        
        # Combined features
        combined_features = structural_features + node_features
        print(f"      Total combined features: {len(combined_features.feature_columns)}")
        
        # Clean features
        for features_obj in [combined_features, node_features, structural_features]:
            feature_cols = [col for col in features_obj.features_df.columns 
                           if col not in ['node_id', 'graph_id']]
            for col in feature_cols:
                features_obj.features_df[col] = features_obj.features_df[col].replace([np.inf, -np.inf], np.nan)
                features_obj.features_df[col] = features_obj.features_df[col].fillna(0)
        
        # Cache features
        features_dict = {
            'structural': structural_features,
            'node': node_features,
            'combined': combined_features
        }
        with open(features_cache, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"      Cached features to: {features_cache}")
    
    print(f"\n      Feature counts:")
    print(f"        Structural: {len(structural_features.feature_columns)}")
    print(f"        Node: {len(node_features.feature_columns)}")
    print(f"        Combined: {len(combined_features.feature_columns)}")
    
    # ========== Step 3: Test Different Embedding Dimensions ==========
    print(f"\n[3/3] Testing embedding dimensions: {DIMENSIONS}")
    print("="*80)
    
    # Test each model type
    model_configs = [
        ('combined', combined_features),
        ('node_only', node_features),
        ('structural_only', structural_features)
    ]
    
    # Create progress bar for models
    model_pbar = tqdm(model_configs, desc="Testing models", position=0)
    
    for model_name, features_obj in model_pbar:
        model_pbar.set_description(f"Model: {model_name}")
        print(f"\n  Model: {model_name.upper()}")
        print(f"  Total features: {len(features_obj.feature_columns)}")
        print("-"*60)
        
        # Create progress bar for dimensions
        dim_pbar = tqdm(DIMENSIONS, desc=f"  Dimensions", position=1, leave=False)
        
        for dim in dim_pbar:
            # Determine actual dimension
            if dim == 'max':
                actual_dim = len(features_obj.feature_columns)
                dim_label = f"max ({actual_dim})"
            else:
                actual_dim = min(dim, len(features_obj.feature_columns))
                dim_label = str(dim)
            
            # Update progress bar description
            dim_pbar.set_description(f"  Dim: {dim_label}")
            
            print(f"\n    Testing dimension: {dim_label}")
            start = time.time()
            
            # Normalize features
            features_norm = features_obj.copy()
            features_norm.normalize(type="StandardScaler")
            
            # Generate embeddings
            embeddings = nxt.compute_graph_embeddings(
                graph_collection=egonet_collection,
                features=features_norm,
                embedding_algorithm="approx_wasserstein",
                embedding_dimension=actual_dim,
                random_state=RANDOM_SEED
            )
            
            # Train model
            results = nxt.train_ml_model(
                graph_collection=egonet_collection,
                embeddings=embeddings,
                model_type="classifier",
                sample_size=100,
                balance_dataset=False
            )
            
            # Store results
            result_entry = {
                'model': model_name,
                'dimension': str(dim),
                'actual_dimension': actual_dim,
                'num_features': len(features_obj.feature_columns),
                'accuracy': np.mean(results['accuracy']),
                'accuracy_std': np.std(results['accuracy']),
                'f1_score': np.mean(results['f1_score']),
                'precision': np.mean(results['precision']),
                'recall': np.mean(results['recall']),
                'time': time.time() - start
            }
            
            all_results.append(result_entry)
            
            # Update overall progress
            experiment_counter += 1
            progress_pct = (experiment_counter / total_experiments) * 100
            
            print(f"      Accuracy: {result_entry['accuracy']:.3f} ± {result_entry['accuracy_std']:.3f}")
            print(f"      F1 Score: {result_entry['f1_score']:.3f}")
            print(f"      Time: {result_entry['time']:.1f}s")
            print(f"      Overall progress: {experiment_counter}/{total_experiments} ({progress_pct:.1f}%)")
            
            # Save intermediate results
            intermediate_df = pd.DataFrame(all_results)
            save_intermediate_results(intermediate_df, timestamp)
    
    # Add random baseline
    print("\n  Adding random baseline results...")
    for dim in tqdm(DIMENSIONS, desc="  Random baseline", leave=False):
        all_results.append({
            'model': 'random_baseline',
            'dimension': str(dim),
            'actual_dimension': 0,
            'num_features': 0,
            'accuracy': 0.5,
            'accuracy_std': 0.0,
            'f1_score': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'time': 0
        })
    
    # Final results DataFrame
    results_df = pd.DataFrame(all_results)
    final_csv = save_intermediate_results(results_df, timestamp)
    
    # ========== Display Summary ==========
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Find best performing configuration
    best_result = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest Configuration:")
    print(f"  Model: {best_result['model']}")
    print(f"  Dimension: {best_result['dimension']} (actual: {best_result['actual_dimension']})")
    print(f"  Accuracy: {best_result['accuracy']:.3f}")
    print(f"  F1 Score: {best_result['f1_score']:.3f}")
    
    # Show performance by dimension for combined model
    combined_results = results_df[results_df['model'] == 'combined'].sort_values('actual_dimension')
    print(f"\nCombined Model Performance by Dimension:")
    print(f"{'Dimension':<12} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*36)
    for _, row in combined_results.iterrows():
        dim_str = f"{row['dimension']}" + (f" ({row['actual_dimension']})" if row['dimension'] == 'max' else "")
        print(f"{dim_str:<12} {row['accuracy']:.3f}        {row['f1_score']:.3f}")
    
    print(f"\nResults saved to: {final_csv}")
    
    # ========== Create Scientific Plot ==========
    if PLOTLY_AVAILABLE:
        print("\nGenerating scientific plot...")
        fig = create_scientific_plot(results_df, timestamp)
    else:
        print("\nSkipping plot generation (plotly not available)")
        fig = None
    
    return results_df


if __name__ == "__main__":
    print(f"\nStarting experiment at {datetime.now()}")
    results = run_embedding_dimension_experiment()
    print(f"\nExperiment completed at {datetime.now()}")