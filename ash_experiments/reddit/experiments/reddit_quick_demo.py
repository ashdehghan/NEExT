#!/usr/bin/env python3
"""
Quick demonstration of Reddit node classification with NEExT.
Uses minimal parameters for fast execution.
"""

import pickle
import numpy as np
import time
import logging
import sys
from pathlib import Path
from collections import Counter

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Quick demo with minimal parameters."""
    
    logger.info("REDDIT NODE CLASSIFICATION - QUICK DEMO")
    logger.info("="*50)
    
    # Configuration for quick demo
    K_HOP = 1  # Only 1-hop neighborhoods (faster)
    SAMPLE_FRACTION = 0.05  # 5% of nodes (~580 nodes)
    EMBEDDING_DIM = 10  # Smaller embeddings
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # Load graph
    logger.info("Loading Reddit 5% sample graph...")
    with open("reddit_networkx_5pct.pkl", 'rb') as f:
        reddit_graph = pickle.load(f)
    logger.info(f"Loaded: {reddit_graph.number_of_nodes()} nodes, {reddit_graph.number_of_edges()} edges")
    
    # Check class distribution
    labels = [attrs.get('subreddit_label', -1) for _, attrs in reddit_graph.nodes(data=True)]
    valid_labels = [l for l in labels if l != -1]
    logger.info(f"Classes: {len(set(valid_labels))} unique subreddits")
    
    # Create GraphCollection
    logger.info("\nCreating GraphCollection...")
    start = time.time()
    graph_collection = nxt.load_from_networkx(
        [reddit_graph],
        reindex_nodes=False,
        filter_largest_component=False,
        node_sample_rate=1.0
    )
    logger.info(f"Time: {time.time() - start:.1f}s")
    
    # Create egonets
    logger.info(f"\nCreating {K_HOP}-hop egonets ({SAMPLE_FRACTION:.0%} sample)...")
    start = time.time()
    
    egonet_collection = EgonetCollection(egonet_feature_target='subreddit_label')
    egonet_collection.compute_k_hop_egonets(
        graph_collection=graph_collection,
        k_hop=K_HOP,
        sample_fraction=SAMPLE_FRACTION,
        random_seed=42
    )
    
    logger.info(f"Created {len(egonet_collection.graphs)} egonets in {time.time() - start:.1f}s")
    
    # Quick stats on egonets
    egonet_sizes = [len(g.nodes) for g in egonet_collection.graphs]
    logger.info(f"Egonet sizes: min={min(egonet_sizes)}, max={max(egonet_sizes)}, avg={np.mean(egonet_sizes):.1f}")
    
    # Compute minimal features
    logger.info("\nComputing features...")
    start = time.time()
    
    features = nxt.compute_node_features(
        graph_collection=egonet_collection,
        feature_list=["degree_centrality", "clustering_coefficient"],  # Just 2 features
        feature_vector_length=1,  # No aggregation
        show_progress=False,
        n_jobs=1  # Single thread for simplicity
    )
    
    logger.info(f"Computed {len(features.feature_columns)} features in {time.time() - start:.1f}s")
    features.normalize(type="StandardScaler")
    
    # Compute embeddings
    logger.info("\nComputing embeddings...")
    start = time.time()
    
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(EMBEDDING_DIM, len(features.feature_columns)),
        random_state=42
    )
    
    logger.info(f"Generated {len(embeddings.embeddings_df)} embeddings in {time.time() - start:.1f}s")
    
    # Train model
    logger.info("\nTraining classifier...")
    start = time.time()
    
    try:
        model_results = nxt.train_ml_model(
            graph_collection=egonet_collection,
            embeddings=embeddings,
            model_type="classifier",
            sample_size=50,  # Increased sample size
            balance_dataset=False  # Disable balancing due to rare classes
        )
        logger.info(f"Training complete in {time.time() - start:.1f}s")
    except ValueError as e:
        logger.warning(f"Model training failed due to class imbalance: {e}")
        logger.info("This is expected with small samples - increase SAMPLE_FRACTION for better results")
        model_results = None
    
    # Results
    logger.info("\n" + "="*50)
    logger.info("RESULTS")
    logger.info("="*50)
    
    if model_results:
        if 'accuracy' in model_results:
            acc = np.mean(model_results['accuracy'])
            logger.info(f"Accuracy: {acc:.2%}")
        
        if 'f1_score' in model_results:
            f1 = np.mean(model_results['f1_score'])
            logger.info(f"F1 Score: {f1:.3f}")
        
        logger.info(f"Classes: {len(model_results.get('classes', []))}")
    else:
        logger.info("Model training skipped due to insufficient samples per class")
    
    logger.info("\n" + "="*50)
    logger.info("DEMO COMPLETE!")
    logger.info("="*50)
    logger.info("This was a minimal demo with:")
    logger.info(f"- {len(egonet_collection.graphs)} egonets (1% sample)")
    logger.info(f"- {K_HOP}-hop neighborhoods")
    logger.info(f"- {len(features.feature_columns)} features")
    logger.info("\nFor full results, increase SAMPLE_FRACTION and add more features.")
    
    return egonet_collection, embeddings, model_results


if __name__ == "__main__":
    egonet_collection, embeddings, results = main()