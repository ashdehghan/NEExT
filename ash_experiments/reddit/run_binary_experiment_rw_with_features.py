#!/usr/bin/env python3
"""
Reddit Binary Classification Experiment with Random Walk Sampling + Reddit Features

Task: Predict if a Reddit post is from a SERIOUS (news/science/tech) vs 
      ENTERTAINMENT (fun/media/art) subreddit based on local graph structure 
      AND original Reddit features.

This experiment uses:
- Random walk sampling for efficient egonet generation
- Original 602 Reddit features combined with structural features
"""

import pickle
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from collections import Counter
import sys
from datetime import datetime

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection
from NEExT.features import Features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_original_node_features(egonet_collection: EgonetCollection, 
                                  aggregation_types: list = ['mean'],
                                  feature_prefix: str = 'feature_') -> Features:
    """
    Extract original Reddit features from egonets and aggregate them.
    
    Args:
        egonet_collection: Collection of egonets with node attributes
        aggregation_types: List of aggregation methods ('mean', 'max', 'std', 'ego')
        feature_prefix: Prefix of feature attributes to extract (default: 'feature_')
        
    Returns:
        Features object containing aggregated Reddit features
    """
    print(f"\n      Extracting original Reddit features...")
    start_time = time.time()
    
    feature_dfs = []
    
    for egonet in egonet_collection.graphs:
        # Get all nodes in this egonet
        node_features_dict = {}
        
        # Identify feature columns (feature_0 through feature_601)
        sample_node = list(egonet.node_attributes.keys())[0] if egonet.node_attributes else None
        if sample_node is None:
            continue
            
        feature_cols = [col for col in egonet.node_attributes[sample_node].keys() 
                       if col.startswith(feature_prefix)]
        
        # Extract features for all nodes in the egonet
        for node_id in egonet.nodes:
            if node_id in egonet.node_attributes:
                node_attrs = egonet.node_attributes[node_id]
                node_features_dict[node_id] = {
                    col: node_attrs.get(col, 0.0) for col in feature_cols
                }
        
        # Create DataFrame for this egonet's nodes
        if node_features_dict:
            nodes_df = pd.DataFrame.from_dict(node_features_dict, orient='index')
            
            # Aggregate features across the egonet
            aggregated_features = {}
            
            for agg_type in aggregation_types:
                if agg_type == 'mean':
                    agg_values = nodes_df.mean()
                    for col in feature_cols:
                        aggregated_features[f"{col}_mean"] = agg_values[col]
                        
                elif agg_type == 'max':
                    agg_values = nodes_df.max()
                    for col in feature_cols:
                        aggregated_features[f"{col}_max"] = agg_values[col]
                        
                elif agg_type == 'std':
                    agg_values = nodes_df.std().fillna(0)  # Fill NaN for single-node egonets
                    for col in feature_cols:
                        aggregated_features[f"{col}_std"] = agg_values[col]
                        
                elif agg_type == 'ego':
                    # Get features from ego center node only
                    ego_node_id = egonet.original_node_id
                    # Map original node ID to egonet internal node ID
                    ego_internal_id = egonet.node_mapping.get(ego_node_id, 0)
                    if ego_internal_id in node_features_dict:
                        ego_features = node_features_dict[ego_internal_id]
                        for col in feature_cols:
                            aggregated_features[f"{col}_ego"] = ego_features.get(col, 0.0)
            
            # Create single-row DataFrame for this egonet
            egonet_df = pd.DataFrame([aggregated_features])
            egonet_df['graph_id'] = egonet.graph_id
            egonet_df['node_id'] = 0  # Single row per graph for graph-level features
            
            feature_dfs.append(egonet_df)
    
    # Combine all egonet features
    if feature_dfs:
        combined_df = pd.concat(feature_dfs, ignore_index=True)
        # Reorder columns to have graph_id and node_id first
        feature_columns = [col for col in combined_df.columns 
                          if col not in ['graph_id', 'node_id']]
        combined_df = combined_df[['node_id', 'graph_id'] + feature_columns]
    else:
        # Empty DataFrame if no features found
        combined_df = pd.DataFrame(columns=['node_id', 'graph_id'])
        feature_columns = []
    
    elapsed = time.time() - start_time
    print(f"      Extracted {len(feature_columns)} Reddit features in {elapsed:.1f}s")
    
    return Features(combined_df, feature_columns)


def run_binary_experiment_rw():
    """Run the binary classification experiment with random walk sampling and Reddit features."""
    
    print("\n" + "="*80)
    print("REDDIT BINARY CLASSIFICATION - RANDOM WALK + REDDIT FEATURES")
    print("="*80)
    print("\nTask: Predict SERIOUS vs ENTERTAINMENT subreddits")
    print("Dataset: 7,315 nodes, 97K edges (binary 5% sample)")
    print("Classes: 0=Serious (news/tech), 1=Entertainment (fun/media)")
    print("Sampling: Random walk with restart")
    print("Features: 602 Reddit features + structural features")
    print("Expected runtime: 10-20 seconds (100 egonets)")
    print("="*80)
    
    total_start = time.time()
    
    # Configuration - Optimized for random walk sampling
    GRAPH_FILE = "reddit_binary_5pct.pkl"
    SAMPLE_FRACTION = 200 / 7315  # ~200 egonets for better accuracy
    EMBEDDING_DIM = 20  # Increased for more features
    
    # Random walk parameters
    WALK_LENGTH = 8
    NUM_WALKS = 4
    RESTART_PROB = 0.2
    MAX_NODES_PER_EGONET = 30  # Bounded neighborhood size
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # ========== STEP 1: Load Binary Graph ==========
    print("\n[1/7] Loading binary graph...")
    start = time.time()
    
    with open(GRAPH_FILE, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"      Loaded in {time.time() - start:.1f}s")
    print(f"      Nodes: {graph.number_of_nodes():,}")
    print(f"      Edges: {graph.number_of_edges():,}")
    
    # Check for Reddit features
    sample_node = list(graph.nodes())[0]
    feature_count = sum(1 for k in graph.nodes[sample_node] if k.startswith('feature_'))
    print(f"      Reddit features per node: {feature_count}")
    
    # Count binary labels
    binary_counts = Counter()
    for _, attrs in graph.nodes(data=True):
        label = attrs.get('binary_label', -1)
        if label >= 0:
            binary_counts[label] += 1
    
    print(f"      Serious: {binary_counts[0]:,} nodes")
    print(f"      Entertainment: {binary_counts[1]:,} nodes")
    print(f"      Balance: {min(binary_counts.values())/max(binary_counts.values()):.2f}")
    
    # ========== STEP 2: Create GraphCollection ==========
    print("\n[2/7] Creating GraphCollection...")
    start = time.time()
    
    graph_collection = nxt.load_from_networkx(
        [graph],
        reindex_nodes=False,
        filter_largest_component=False,
        node_sample_rate=1.0
    )
    
    print(f"      Created in {time.time() - start:.1f}s")
    
    # ========== STEP 3: Create Egonets with Random Walk Sampling ==========
    print(f"\n[3/7] Creating egonets with Random Walk sampling (~200 egonets)...")
    print(f"      Walk length: {WALK_LENGTH}, Num walks: {NUM_WALKS}, Restart prob: {RESTART_PROB}")
    print(f"      Max nodes per egonet: {MAX_NODES_PER_EGONET}")
    start = time.time()
    
    # Use binary_label as target
    egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
    egonet_collection.compute_k_hop_egonets(
        graph_collection=graph_collection,
        k_hop=2,  # Used as hint for adaptive parameters
        sample_fraction=SAMPLE_FRACTION,
        # Random walk sampling parameters
        sampling_strategy='random_walk',
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        restart_prob=RESTART_PROB,
        max_nodes_per_egonet=MAX_NODES_PER_EGONET,
        random_seed=42
    )
    
    num_egonets = len(egonet_collection.graphs)
    egonet_times = time.time() - start
    print(f"      Created {num_egonets:,} egonets in {egonet_times:.1f}s")
    
    # Analyze egonet sizes
    egonet_sizes = [len(egonet.nodes) for egonet in egonet_collection.graphs]
    print(f"      Avg egonet size: {np.mean(egonet_sizes):.1f} nodes")
    print(f"      Max egonet size: {max(egonet_sizes)} nodes")
    
    # Check label distribution in egonets
    egonet_labels = [g.graph_label for g in egonet_collection.graphs if g.graph_label is not None]
    label_counts = Counter(egonet_labels)
    print(f"      Serious egonets: {label_counts.get(0, 0)}")
    print(f"      Entertainment egonets: {label_counts.get(1, 0)}")
    
    # ========== STEP 4: Compute Structural Features ==========
    print("\n[4/7] Computing structural features...")
    start = time.time()
    
    # Simplified feature set for speed
    feature_list = [
        "degree_centrality",
        "clustering_coefficient", 
        "page_rank",
        "betweenness_centrality"
    ]
    
    structural_features = nxt.compute_node_features(
        graph_collection=egonet_collection,
        feature_list=feature_list,
        feature_vector_length=1,  # No aggregation for speed
        show_progress=False,
        n_jobs=-1
    )
    
    print(f"      Computed {len(structural_features.feature_columns)} structural features in {time.time() - start:.1f}s")
    
    # ========== STEP 5: Extract Original Reddit Features ==========
    print("\n[5/7] Extracting original Reddit features...")
    start = time.time()
    
    reddit_features = extract_original_node_features(
        egonet_collection,
        aggregation_types=['mean'],  # Just mean aggregation to start
        feature_prefix='feature_'
    )
    
    print(f"      Extraction complete in {time.time() - start:.1f}s")
    
    # ========== STEP 6: Combine All Features ==========
    print("\n[6/7] Combining structural and Reddit features...")
    
    # Combine features using NEExT's built-in addition
    combined_features = structural_features + reddit_features
    
    print(f"      Total features: {len(combined_features.feature_columns)}")
    print(f"      - Structural: {len(structural_features.feature_columns)}")
    print(f"      - Reddit: {len(reddit_features.feature_columns)}")
    
    # Handle any NaN or Inf values before normalization
    feature_cols = [col for col in combined_features.features_df.columns 
                   if col not in ['node_id', 'graph_id']]
    
    # Replace NaN with 0 and Inf with large finite values
    for col in feature_cols:
        combined_features.features_df[col] = combined_features.features_df[col].replace([np.inf, -np.inf], np.nan)
        combined_features.features_df[col] = combined_features.features_df[col].fillna(0)
    
    # Normalize combined features
    combined_features.normalize(type="StandardScaler")
    
    # ========== STEP 7: Compute Embeddings and Train ==========
    print("\n[7/7] Computing embeddings and training classifier...")
    start = time.time()
    
    # Compute embeddings with combined features
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=combined_features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(combined_features.feature_columns)),
        random_state=42
    )
    
    print(f"      Generated {len(embeddings.embeddings_df)} embeddings")
    print(f"      Embedding dimensions: {len(embeddings.embedding_columns)}")
    
    # Train classifier
    try:
        model_results = nxt.train_ml_model(
            graph_collection=egonet_collection,
            embeddings=embeddings,
            model_type="classifier",
            sample_size=100,  # Larger sample size for better estimates
            balance_dataset=False  # Already balanced
        )
        
        print(f"      Training complete in {time.time() - start:.1f}s")
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS - WITH REDDIT FEATURES")
        print("="*60)
        
        if 'accuracy' in model_results:
            acc_scores = model_results['accuracy']
            acc_mean = np.mean(acc_scores)
            acc_std = np.std(acc_scores)
            print(f"Accuracy:  {acc_mean:.3f} ± {acc_std:.3f}")
            print(f"          (min: {min(acc_scores):.3f}, max: {max(acc_scores):.3f})")
        
        if 'f1_score' in model_results:
            f1_scores = model_results['f1_score']
            f1_mean = np.mean(f1_scores)
            f1_std = np.std(f1_scores)
            print(f"F1 Score:  {f1_mean:.3f} ± {f1_std:.3f}")
        
        if 'precision' in model_results:
            prec_mean = np.mean(model_results['precision'])
            print(f"Precision: {prec_mean:.3f}")
        
        if 'recall' in model_results:
            rec_mean = np.mean(model_results['recall'])
            print(f"Recall:    {rec_mean:.3f}")
        
        print(f"\nClasses: {model_results.get('classes', [])}")
        
        # Feature importance analysis
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60)
        print(f"Used {len(combined_features.feature_columns)} total features:")
        print(f"- {len(structural_features.feature_columns)} structural graph features")
        print(f"- {len(reddit_features.feature_columns)} Reddit content features")
        
        # Interpretation
        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        
        if acc_mean > 0.7:
            print("✅ SUCCESS! Reddit features dramatically improve performance!")
            print("   The model can now distinguish serious vs entertainment posts")
            print("   based on BOTH content features AND graph structure.")
            print("\nKey insights:")
            print("- Reddit's 602 features capture semantic content patterns")
            print("- Combined with graph structure provides robust classification")
            print("- Random walk sampling preserves discriminative neighborhoods")
        elif acc_mean > 0.55:
            print("✓ Moderate improvement with Reddit features")
            print("  Performance is better than random but could be improved")
            print("  Consider: larger sample size, different aggregations, or longer walks")
        else:
            print("⚠ Limited improvement despite Reddit features")
            print("  Possible issues: small sample size, aggregation method, or feature quality")
        
        # Compare to baseline
        print(f"\nBaseline (structural only): ~43% accuracy")
        print(f"With Reddit features: {acc_mean:.1%} accuracy")
        improvement = (acc_mean - 0.43) / 0.43 * 100
        if improvement > 0:
            print(f"Relative improvement: +{improvement:.0f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"      Training failed: {e}")
        model_results = None
    
    # Final summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Processed: {num_egonets} egonets")
    print(f"Features used: {len(combined_features.feature_columns)} total")
    print(f"Task: Binary classification with Reddit + structural features")
    
    return model_results, {
        'num_egonets': num_egonets,
        'avg_egonet_size': np.mean(egonet_sizes),
        'total_features': len(combined_features.feature_columns),
        'reddit_features': len(reddit_features.feature_columns),
        'structural_features': len(structural_features.feature_columns),
        'total_time': total_time
    }


def main():
    """Main entry point."""
    print("Starting Reddit binary classification with Random Walk + Reddit Features...")
    
    results, metrics = run_binary_experiment_rw()
    
    if results:
        print("\n✅ Experiment completed successfully!")
        print(f"\n📊 Summary:")
        print(f"   - {metrics['num_egonets']} egonets processed")
        print(f"   - {metrics['total_features']} total features")
        print(f"     • {metrics['reddit_features']} Reddit features")
        print(f"     • {metrics['structural_features']} structural features")
        print(f"   - Runtime: {metrics['total_time']:.1f}s")
        
        if 'accuracy' in results:
            acc_mean = np.mean(results['accuracy'])
            if acc_mean > 0.7:
                print(f"\n🎉 BREAKTHROUGH: {acc_mean:.1%} accuracy achieved!")
                print("   Reddit features unlock the discriminative power!")
    else:
        print("\n⚠️ Experiment had issues - check logs")
    
    return results, metrics


if __name__ == "__main__":
    results, metrics = main()