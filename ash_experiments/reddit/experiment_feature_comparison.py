#!/usr/bin/env python3
"""
Feature Comparison Experiment on Reddit 5% Dataset

This experiment compares four models:
1. Combined (structural + node features)
2. Node features only
3. Structural features only  
4. Random baseline

Using 200 egonets sampled via random walk.
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

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection
from NEExT.features import Features


def extract_node_features(egonet_collection, feature_prefix='feature_'):
    """Extract original Reddit node features from egonets."""
    print(f"      Extracting Reddit node features...")
    
    feature_dfs = []
    
    for egonet in egonet_collection.graphs:
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


def run_experiment():
    """Run the feature comparison experiment."""
    
    print("\n" + "="*80)
    print("REDDIT FEATURE COMPARISON EXPERIMENT")
    print("="*80)
    print("Dataset: Reddit 5% sample")
    print("Models: Combined, Node-only, Structural-only, Random baseline")
    print("Sampling: 200 egonets via random walk")
    print("="*80)
    
    # Configuration
    GRAPH_FILE = "reddit_binary_5pct.pkl"
    NUM_EGONETS = 200
    EMBEDDING_DIM = 20  # <-- THIS IS THE WASSERSTEIN DIMENSION
    RANDOM_SEED = 42
    
    # Random walk parameters
    WALK_LENGTH = 8
    NUM_WALKS = 4
    RESTART_PROB = 0.2
    MAX_NODES_PER_EGONET = 30
    
    print(f"\nConfiguration:")
    print(f"  Egonets: {NUM_EGONETS}")
    print(f"  Wasserstein embedding dimension: {EMBEDDING_DIM}")
    print(f"  Random walk: length={WALK_LENGTH}, walks={NUM_WALKS}, restart={RESTART_PROB}")
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # Storage for results
    results = {}
    
    # ========== Load Graph ==========
    print("\n[1/6] Loading Reddit 5% graph...")
    start = time.time()
    
    with open(GRAPH_FILE, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"      Loaded in {time.time() - start:.1f}s")
    print(f"      Nodes: {graph.number_of_nodes():,}")
    print(f"      Edges: {graph.number_of_edges():,}")
    
    # Check features
    sample_node = list(graph.nodes())[0]
    reddit_feature_count = sum(1 for k in graph.nodes[sample_node] if k.startswith('feature_'))
    print(f"      Reddit features per node: {reddit_feature_count}")
    
    # ========== Create GraphCollection ==========
    print("\n[2/6] Creating GraphCollection...")
    graph_collection = nxt.load_from_networkx(
        [graph],
        reindex_nodes=False,
        filter_largest_component=False,
        node_sample_rate=1.0
    )
    
    # ========== Create Egonets ==========
    print(f"\n[3/6] Creating {NUM_EGONETS} egonets with random walk sampling...")
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
    
    num_egonets = len(egonet_collection.graphs)
    print(f"      Created {num_egonets} egonets in {time.time() - start:.1f}s")
    
    # Check class balance
    egonet_labels = [g.graph_label for g in egonet_collection.graphs if g.graph_label is not None]
    label_counts = Counter(egonet_labels)
    print(f"      Class balance: Serious={label_counts.get(0, 0)}, Entertainment={label_counts.get(1, 0)}")
    
    # ========== Compute Features ==========
    print("\n[4/6] Computing features...")
    
    # Structural features
    print("   a) Structural features...")
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
    
    # Node features (Reddit features)
    print("   b) Reddit node features...")
    start = time.time()
    node_features = extract_node_features(egonet_collection)
    print(f"      Extracted {len(node_features.feature_columns)} Reddit features in {time.time() - start:.1f}s")
    
    # Combined features
    print("   c) Combining features...")
    combined_features = structural_features + node_features
    print(f"      Total combined features: {len(combined_features.feature_columns)}")
    
    # Clean features (handle NaN/Inf)
    for features_obj in [combined_features, node_features, structural_features]:
        feature_cols = [col for col in features_obj.features_df.columns 
                       if col not in ['node_id', 'graph_id']]
        for col in feature_cols:
            features_obj.features_df[col] = features_obj.features_df[col].replace([np.inf, -np.inf], np.nan)
            features_obj.features_df[col] = features_obj.features_df[col].fillna(0)
    
    # ========== Train Models ==========
    print(f"\n[5/6] Training models (embedding dim={EMBEDDING_DIM})...")
    
    # Model 1: Combined features
    print("\n   Model 1: Combined (structural + node) features")
    start = time.time()
    
    combined_features_norm = combined_features.copy()
    combined_features_norm.normalize(type="StandardScaler")
    
    embeddings_combined = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=combined_features_norm,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(combined_features.feature_columns)),
        random_state=RANDOM_SEED
    )
    
    results_combined = nxt.train_ml_model(
        graph_collection=egonet_collection,
        embeddings=embeddings_combined,
        model_type="classifier",
        sample_size=100,
        balance_dataset=False
    )
    
    results['combined'] = {
        'accuracy': np.mean(results_combined['accuracy']),
        'accuracy_std': np.std(results_combined['accuracy']),
        'f1_score': np.mean(results_combined['f1_score']),
        'precision': np.mean(results_combined['precision']),
        'recall': np.mean(results_combined['recall']),
        'num_features': len(combined_features.feature_columns),
        'time': time.time() - start
    }
    print(f"      Accuracy: {results['combined']['accuracy']:.3f} ± {results['combined']['accuracy_std']:.3f}")
    
    # Model 2: Node features only
    print("\n   Model 2: Node features only")
    start = time.time()
    
    node_features_norm = node_features.copy()
    node_features_norm.normalize(type="StandardScaler")
    
    embeddings_node = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=node_features_norm,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(node_features.feature_columns)),
        random_state=RANDOM_SEED
    )
    
    results_node = nxt.train_ml_model(
        graph_collection=egonet_collection,
        embeddings=embeddings_node,
        model_type="classifier",
        sample_size=100,
        balance_dataset=False
    )
    
    results['node_only'] = {
        'accuracy': np.mean(results_node['accuracy']),
        'accuracy_std': np.std(results_node['accuracy']),
        'f1_score': np.mean(results_node['f1_score']),
        'precision': np.mean(results_node['precision']),
        'recall': np.mean(results_node['recall']),
        'num_features': len(node_features.feature_columns),
        'time': time.time() - start
    }
    print(f"      Accuracy: {results['node_only']['accuracy']:.3f} ± {results['node_only']['accuracy_std']:.3f}")
    
    # Model 3: Structural features only
    print("\n   Model 3: Structural features only")
    start = time.time()
    
    structural_features_norm = structural_features.copy()
    structural_features_norm.normalize(type="StandardScaler")
    
    embeddings_structural = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=structural_features_norm,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(structural_features.feature_columns)),
        random_state=RANDOM_SEED
    )
    
    results_structural = nxt.train_ml_model(
        graph_collection=egonet_collection,
        embeddings=embeddings_structural,
        model_type="classifier",
        sample_size=100,
        balance_dataset=False
    )
    
    results['structural_only'] = {
        'accuracy': np.mean(results_structural['accuracy']),
        'accuracy_std': np.std(results_structural['accuracy']),
        'f1_score': np.mean(results_structural['f1_score']),
        'precision': np.mean(results_structural['precision']),
        'recall': np.mean(results_structural['recall']),
        'num_features': len(structural_features.feature_columns),
        'time': time.time() - start
    }
    print(f"      Accuracy: {results['structural_only']['accuracy']:.3f} ± {results['structural_only']['accuracy_std']:.3f}")
    
    # Model 4: Random baseline
    print("\n   Model 4: Random baseline")
    # For binary classification with balanced classes, random baseline is 0.5
    results['random_baseline'] = {
        'accuracy': 0.5,
        'accuracy_std': 0.0,
        'f1_score': 0.5,
        'precision': 0.5,
        'recall': 0.5,
        'num_features': 0,
        'time': 0
    }
    print(f"      Accuracy: {results['random_baseline']['accuracy']:.3f}")
    
    # ========== Display Results ==========
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Features':<10} {'Accuracy':<15} {'F1':<10} {'Time (s)':<10}")
    print("-"*70)
    
    for model_name, res in results.items():
        print(f"{model_name:<20} {res['num_features']:<10} "
              f"{res['accuracy']:.3f} ± {res['accuracy_std']:.3f}  "
              f"{res['f1_score']:.3f}      {res['time']:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_feature_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    improvement_over_random = (results['combined']['accuracy'] - 0.5) / 0.5 * 100
    improvement_over_structural = (results['combined']['accuracy'] - results['structural_only']['accuracy']) / results['structural_only']['accuracy'] * 100
    
    print(f"Combined model improvement over random baseline: {improvement_over_random:.1f}%")
    print(f"Combined model improvement over structural only: {improvement_over_structural:.1f}%")
    print(f"Node features alone achieve: {results['node_only']['accuracy']:.3f} accuracy")
    
    return results


if __name__ == "__main__":
    print(f"\nStarting experiment at {datetime.now()}")
    results = run_experiment()
    print(f"\nExperiment completed at {datetime.now()}")