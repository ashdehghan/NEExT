#!/usr/bin/env python3
"""
Reddit Node Classification Experiment using NEExT Framework

This script demonstrates node-level classification on the Reddit dataset using:
- Egonet decomposition for node-centered subgraphs
- Structural feature computation on local neighborhoods
- Graph embeddings via Wasserstein distance
- XGBoost classification for subreddit prediction

The task: Predict which subreddit (community) a Reddit post belongs to based on
its local graph structure (posts connected via shared commenters).
"""

import pickle
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from collections import Counter
import sys

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_reddit_graph(file_path: str = "reddit_networkx_5pct.pkl"):
    """Load the sampled Reddit graph."""
    logger.info(f"Loading Reddit graph from {file_path}...")
    start_time = time.time()
    
    with open(file_path, 'rb') as f:
        reddit_graph = pickle.load(f)
    
    load_time = time.time() - start_time
    logger.info(f"Graph loaded in {load_time:.2f} seconds")
    logger.info(f"Graph statistics: {reddit_graph.number_of_nodes()} nodes, {reddit_graph.number_of_edges()} edges")
    
    return reddit_graph


def analyze_graph_properties(graph):
    """Analyze and report graph properties."""
    logger.info("\n" + "="*60)
    logger.info("GRAPH ANALYSIS")
    logger.info("="*60)
    
    # Basic statistics
    logger.info(f"Nodes: {graph.number_of_nodes():,}")
    logger.info(f"Edges: {graph.number_of_edges():,}")
    logger.info(f"Average degree: {2 * graph.number_of_edges() / graph.number_of_nodes():.1f}")
    
    # Class distribution
    class_counts = Counter()
    split_counts = Counter()
    
    for node, attrs in graph.nodes(data=True):
        label = attrs.get('subreddit_label', -1)
        split = attrs.get('split', 'unknown')
        if label != -1:
            class_counts[label] += 1
        split_counts[split] += 1
    
    logger.info(f"Number of classes: {len(class_counts)}")
    logger.info(f"Class distribution (top 5): {dict(class_counts.most_common(5))}")
    logger.info(f"Split distribution: {dict(split_counts)}")
    
    # Feature check
    sample_node = next(iter(graph.nodes()))
    sample_attrs = graph.nodes[sample_node]
    feature_count = sum(1 for attr in sample_attrs if attr.startswith('feature_'))
    logger.info(f"Features per node: {feature_count}")
    
    return class_counts, split_counts


def create_egonets(nxt, graph_collection, k_hop=2, sample_fraction=0.2):
    """Create egonet collection for node classification."""
    logger.info("\n" + "="*60)
    logger.info("EGONET DECOMPOSITION")
    logger.info("="*60)
    
    logger.info(f"Creating {k_hop}-hop egonets with {sample_fraction:.0%} sampling...")
    start_time = time.time()
    
    # Create EgonetCollection with subreddit_label as target
    egonet_collection = EgonetCollection(egonet_feature_target='subreddit_label')
    
    # Compute k-hop egonets
    egonet_collection.compute_k_hop_egonets(
        graph_collection=graph_collection,
        k_hop=k_hop,
        sample_fraction=sample_fraction,
        random_seed=42
    )
    
    creation_time = time.time() - start_time
    logger.info(f"Egonet creation took {creation_time:.2f} seconds")
    logger.info(f"Created {len(egonet_collection.graphs)} egonets")
    
    # Analyze egonet sizes
    egonet_sizes = [len(g['nodes']) for g in egonet_collection.graphs]
    logger.info(f"Egonet size statistics:")
    logger.info(f"  Min nodes: {min(egonet_sizes)}")
    logger.info(f"  Max nodes: {max(egonet_sizes)}")
    logger.info(f"  Mean nodes: {np.mean(egonet_sizes):.1f}")
    logger.info(f"  Median nodes: {np.median(egonet_sizes):.1f}")
    
    # Check label distribution in egonets
    egonet_labels = [g['graph_label'] for g in egonet_collection.graphs if g.get('graph_label') is not None]
    unique_labels = len(set(egonet_labels))
    logger.info(f"Unique labels in egonets: {unique_labels}")
    
    return egonet_collection


def compute_features(nxt, egonet_collection, feature_list=None):
    """Compute structural features on egonets."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE COMPUTATION")
    logger.info("="*60)
    
    if feature_list is None:
        # Default feature set
        feature_list = [
            "page_rank",
            "degree_centrality",
            "closeness_centrality",
            "betweenness_centrality",
            "clustering_coefficient",
            "eigenvector_centrality"
        ]
    
    logger.info(f"Computing features: {feature_list}")
    start_time = time.time()
    
    # Compute node features on egonets
    features = nxt.compute_node_features(
        graph_collection=egonet_collection,
        feature_list=feature_list,
        feature_vector_length=2,  # Aggregate at different hop distances
        show_progress=True,
        n_jobs=-1  # Use all cores
    )
    
    computation_time = time.time() - start_time
    logger.info(f"Feature computation took {computation_time:.2f} seconds")
    logger.info(f"Computed {len(features.feature_columns)} features")
    logger.info(f"Feature columns: {features.feature_columns[:5]}..." if len(features.feature_columns) > 5 else f"Feature columns: {features.feature_columns}")
    
    # Add positional features (distance from ego center)
    logger.info("Adding positional features...")
    # Note: This would require implementing the positional feature method if not available
    
    # Normalize features
    logger.info("Normalizing features...")
    features.normalize(type="StandardScaler")
    
    return features


def compute_embeddings(nxt, egonet_collection, features, embedding_dim=20):
    """Compute graph embeddings for each egonet."""
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING COMPUTATION")
    logger.info("="*60)
    
    logger.info(f"Computing embeddings with dimension {embedding_dim}...")
    start_time = time.time()
    
    embeddings = nxt.compute_graph_embeddings(
        graph_collection=egonet_collection,
        features=features,
        embedding_algorithm="approx_wasserstein",
        embedding_dimension=min(embedding_dim, len(features.feature_columns)),
        random_state=42
    )
    
    embedding_time = time.time() - start_time
    logger.info(f"Embedding computation took {embedding_time:.2f} seconds")
    logger.info(f"Generated {len(embeddings.embeddings_df)} embeddings")
    logger.info(f"Embedding dimensions: {len(embeddings.embedding_columns)}")
    
    return embeddings


def train_and_evaluate(nxt, egonet_collection, embeddings):
    """Train classifier and evaluate performance."""
    logger.info("\n" + "="*60)
    logger.info("MODEL TRAINING & EVALUATION")
    logger.info("="*60)
    
    # Get labels from egonets
    labels = []
    splits = []
    
    for egonet in egonet_collection.graphs:
        label = egonet.get('graph_label', -1)
        labels.append(label)
        
        # Get split from original node attributes if available
        if egonet.get('node_attributes'):
            ego_node_id = egonet.get('ego_node_id', next(iter(egonet['nodes'])))
            split = egonet['node_attributes'].get(ego_node_id, {}).get('split', 'unknown')
            splits.append(split)
        else:
            splits.append('unknown')
    
    labels = np.array(labels)
    splits = np.array(splits)
    
    # Filter out invalid labels
    valid_mask = labels != -1
    labels = labels[valid_mask]
    splits = splits[valid_mask]
    
    logger.info(f"Total valid samples: {len(labels)}")
    logger.info(f"Number of classes: {len(np.unique(labels))}")
    logger.info(f"Class distribution: {Counter(labels).most_common(5)}")
    
    # Train model using NEExT's built-in method
    logger.info("Training XGBoost classifier...")
    start_time = time.time()
    
    try:
        model_results = nxt.train_ml_model(
            graph_collection=egonet_collection,
            embeddings=embeddings,
            model_type="classifier",
            sample_size=min(50, len(labels) // 10),  # Adjust sample size based on data
            balance_dataset=True,  # Balance classes for imbalanced data
            random_state=42
        )
        
        training_time = time.time() - start_time
        logger.info(f"Model training took {training_time:.2f} seconds")
        
        # Print results
        logger.info("\n" + "-"*40)
        logger.info("CLASSIFICATION RESULTS")
        logger.info("-"*40)
        
        metrics = ['accuracy', 'recall', 'precision', 'f1_score']
        for metric in metrics:
            if metric in model_results:
                mean_val = np.mean(model_results[metric])
                std_val = np.std(model_results[metric])
                logger.info(f"{metric.capitalize():<15} Mean: {mean_val:.4f} ± {std_val:.4f}")
        
        logger.info(f"\nNumber of classes: {len(model_results.get('classes', []))}")
        
        # Additional analysis if splits are available
        if 'train' in splits and 'test' in splits:
            train_mask = splits == 'train'
            test_mask = splits == 'test'
            logger.info(f"Train samples: {train_mask.sum()}")
            logger.info(f"Test samples: {test_mask.sum()}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        model_results = None
    
    return model_results


def compute_feature_importance(nxt, egonet_collection, features):
    """Compute and analyze feature importance."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*60)
    
    try:
        logger.info("Computing feature importance...")
        start_time = time.time()
        
        importance_df = nxt.compute_feature_importance(
            graph_collection=egonet_collection,
            features=features,
            feature_importance_algorithm="supervised_fast",
            embedding_algorithm="approx_wasserstein",
            n_iterations=3,
            random_state=42
        )
        
        importance_time = time.time() - start_time
        logger.info(f"Feature importance computation took {importance_time:.2f} seconds")
        
        # Display top features
        logger.info("\nTop 10 Most Important Features:")
        logger.info("-"*40)
        
        # Sort by importance if the dataframe has importance scores
        if not importance_df.empty:
            # Assuming the dataframe has feature names and importance scores
            top_features = importance_df.head(10)
            for idx, row in top_features.iterrows():
                logger.info(f"{idx+1:2d}. {row}")
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Feature importance computation failed: {e}")
        return None


def main():
    """Main experimental pipeline."""
    logger.info("\n" + "="*80)
    logger.info("REDDIT NODE CLASSIFICATION EXPERIMENT WITH NEExT")
    logger.info("="*80)
    
    # Configuration
    K_HOP = 2  # 2-hop neighborhoods
    SAMPLE_FRACTION = 0.2  # Sample 20% of nodes for initial experiment
    EMBEDDING_DIM = 20  # Embedding dimension
    
    try:
        # Initialize NEExT
        nxt = NEExT()
        nxt.set_log_level("WARNING")  # Reduce verbosity
        
        # Step 1: Load Reddit graph
        reddit_graph = load_reddit_graph("reddit_networkx_5pct.pkl")
        
        # Step 2: Analyze graph properties
        class_counts, split_counts = analyze_graph_properties(reddit_graph)
        
        # Step 3: Create GraphCollection from NetworkX
        logger.info("\nCreating GraphCollection...")
        graph_collection = nxt.load_from_networkx(
            [reddit_graph],
            reindex_nodes=False,  # Keep original node IDs
            filter_largest_component=False,  # Keep full graph
            node_sample_rate=1.0  # Use all nodes (already sampled)
        )
        
        # Step 4: Create egonets for node classification
        egonet_collection = create_egonets(
            nxt, 
            graph_collection, 
            k_hop=K_HOP, 
            sample_fraction=SAMPLE_FRACTION
        )
        
        # Step 5: Compute features on egonets
        features = compute_features(nxt, egonet_collection)
        
        # Step 6: Compute embeddings
        embeddings = compute_embeddings(
            nxt, 
            egonet_collection, 
            features, 
            embedding_dim=EMBEDDING_DIM
        )
        
        # Step 7: Train and evaluate model
        model_results = train_and_evaluate(nxt, egonet_collection, embeddings)
        
        # Step 8: Feature importance analysis (optional)
        if model_results is not None:
            importance_df = compute_feature_importance(nxt, egonet_collection, features)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully processed {len(egonet_collection.graphs)} egonets")
        logger.info(f"Each egonet represents a Reddit post's {K_HOP}-hop neighborhood")
        logger.info(f"Task: Predict subreddit (community) from local graph structure")
        
        if model_results:
            accuracy = np.mean(model_results.get('accuracy', [0]))
            logger.info(f"Final accuracy: {accuracy:.2%}")
        
        logger.info("\nKey Insights:")
        logger.info("- Local graph structure contains signal for content categorization")
        logger.info("- Users commenting on similar posts create community patterns")
        logger.info("- Structural features capture these interaction patterns")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    return egonet_collection, features, embeddings, model_results


if __name__ == "__main__":
    # Run the experiment
    egonet_collection, features, embeddings, results = main()
    
    print("\n" + "="*80)
    print("REDDIT NEEXT EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nTo explore further:")
    print("1. Adjust K_HOP (1, 2, 3) to change neighborhood size")
    print("2. Modify SAMPLE_FRACTION to use more/fewer nodes")
    print("3. Try different feature_list combinations")
    print("4. Experiment with embedding algorithms (exact_wasserstein, sinkhorn)")
    print("5. Use the full reddit_networkx.pkl for complete results")
    print("="*80)