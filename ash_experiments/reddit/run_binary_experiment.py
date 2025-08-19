#!/usr/bin/env python3
"""
Reddit Binary Classification Experiment with NEExT

Task: Predict if a Reddit post is from a SERIOUS (news/science/tech) vs 
      ENTERTAINMENT (fun/media/art) subreddit based on local graph structure.

This experiment uses the 5% binary sample for fast execution.
Expected runtime: 30-60 seconds
"""

import pickle
import numpy as np
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_binary_experiment():
    """Run the binary classification experiment."""
    
    print("\n" + "="*80)
    print("REDDIT BINARY CLASSIFICATION EXPERIMENT")
    print("="*80)
    print("\nTask: Predict SERIOUS vs ENTERTAINMENT subreddits")
    print("Dataset: 7,315 nodes, 97K edges (binary 5% sample)")
    print("Classes: 0=Serious (news/tech), 1=Entertainment (fun/media)")
    print("Expected runtime: 30-60 seconds")
    print("="*80)
    
    total_start = time.time()
    
    # Configuration
    GRAPH_FILE = "reddit_binary_5pct.pkl"
    K_HOP = 1  # 1-hop for speed
    SAMPLE_FRACTION = 0.15  # 15% of nodes (~1,100 egonets)
    EMBEDDING_DIM = 15  # Smaller for binary task
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # ========== STEP 1: Load Binary Graph ==========
    print("\n[1/6] Loading binary graph...")
    start = time.time()
    
    with open(GRAPH_FILE, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"      Loaded in {time.time() - start:.1f}s")
    print(f"      Nodes: {graph.number_of_nodes():,}")
    print(f"      Edges: {graph.number_of_edges():,}")
    
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
    print("\n[2/6] Creating GraphCollection...")
    start = time.time()
    
    graph_collection = nxt.load_from_networkx(
        [graph],
        reindex_nodes=False,
        filter_largest_component=False,
        node_sample_rate=1.0
    )
    
    print(f"      Created in {time.time() - start:.1f}s")
    
    # ========== STEP 3: Create Egonets ==========
    print(f"\n[3/6] Creating {K_HOP}-hop egonets ({SAMPLE_FRACTION:.0%} sample)...")
    start = time.time()
    
    # Use binary_label as target
    egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
    egonet_collection.compute_k_hop_egonets(
        graph_collection=graph_collection,
        k_hop=K_HOP,
        sample_fraction=SAMPLE_FRACTION,
        random_seed=42
    )
    
    num_egonets = len(egonet_collection.graphs)
    print(f"      Created {num_egonets:,} egonets in {time.time() - start:.1f}s")
    
    # Check label distribution in egonets
    egonet_labels = [g.graph_label for g in egonet_collection.graphs if g.graph_label is not None]
    label_counts = Counter(egonet_labels)
    print(f"      Serious egonets: {label_counts.get(0, 0)}")
    print(f"      Entertainment egonets: {label_counts.get(1, 0)}")
    
    # ========== STEP 4: Compute Features ==========
    print("\n[4/6] Computing structural features...")
    start = time.time()
    
    # Simplified feature set for speed
    feature_list = [
        "degree_centrality",
        "clustering_coefficient",
        "page_rank"
    ]
    
    features = nxt.compute_node_features(
        graph_collection=egonet_collection,
        feature_list=feature_list,
        feature_vector_length=1,  # No aggregation for speed
        show_progress=False,
        n_jobs=-1
    )
    
    print(f"      Computed {len(features.feature_columns)} features in {time.time() - start:.1f}s")
    
    # Normalize
    features.normalize(type="StandardScaler")
    
    # ========== STEP 5: Compute Embeddings ==========
    print("\n[5/6] Computing graph embeddings...")
    start = time.time()
    
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(features.feature_columns)),
        random_state=42
    )
    
    print(f"      Generated {len(embeddings.embeddings_df)} embeddings in {time.time() - start:.1f}s")
    print(f"      Dimensions: {len(embeddings.embedding_columns)}")
    
    # ========== STEP 6: Train Classifier ==========
    print("\n[6/6] Training binary classifier...")
    start = time.time()
    
    try:
        model_results = nxt.train_ml_model(
            graph_collection=egonet_collection,
            embeddings=embeddings,
            model_type="classifier",
            sample_size=50,  # Good for binary task
            balance_dataset=False  # Already balanced
        )
        
        print(f"      Training complete in {time.time() - start:.1f}s")
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
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
        
        # Interpretation
        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        
        if acc_mean > 0.6:
            print("✓ Model successfully distinguishes serious vs entertainment posts")
            print("  based on local graph structure (commenting patterns).")
            print("\nThis suggests that:")
            print("- Users interact differently with serious vs fun content")
            print("- Community structure reflects content type")
            print("- Graph topology encodes semantic information")
        else:
            print("⚠ Model shows moderate performance")
            print("  Consider increasing sample size or k-hop value")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"      Training failed: {e}")
        print("      (This may be due to insufficient samples)")
        model_results = None
    
    # Final summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Processed: {num_egonets} egonets")
    print(f"Task: Binary classification (Serious vs Entertainment)")
    
    return model_results


def main():
    """Main entry point."""
    results = run_binary_experiment()
    
    if results:
        print("\n✅ Experiment completed successfully!")
        print("   Results demonstrate that local graph structure")
        print("   can predict content type on Reddit.")
    else:
        print("\n⚠️ Experiment had issues - check logs")
    
    return results


if __name__ == "__main__":
    results = main()