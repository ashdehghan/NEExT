#!/usr/bin/env python3
"""
Load Reddit dataset and convert to NetworkX format compatible with NEExT framework.

This script loads the Reddit GraphSAGE dataset (.npz files) and creates a single
NetworkX graph with proper node attributes for use with NEExT's load_from_networkx() method.

Dataset Info:
- 232,965 Reddit posts (nodes)
- ~11.6M edges (posts connected if same user commented on both)
- 602-dimensional features per node (GloVe embeddings + metadata)
- 41 subreddit classes for node classification
- Pre-defined train/val/test splits
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_reddit_data(data_dir: str = "data") -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load Reddit dataset from .npz files.
    
    Args:
        data_dir: Directory containing reddit.npz and reddit_adj.npz
        
    Returns:
        adj_matrix: Sparse adjacency matrix (232965 x 232965)
        features: Node features (232965 x 602)
        splits_and_labels: Dictionary with train/val/test indices and labels
    """
    data_path = Path(data_dir)
    
    logger.info("Loading Reddit data files...")
    
    # Load node features and split information
    reddit_data = np.load(data_path / "reddit.npz")
    features = reddit_data['feats']
    
    splits_and_labels = {
        'train_index': reddit_data['train_index'],
        'val_index': reddit_data['val_index'], 
        'test_index': reddit_data['test_index'],
        'y_train': reddit_data['y_train'],
        'y_val': reddit_data['y_val'],
        'y_test': reddit_data['y_test']
    }
    
    # Load adjacency matrix
    adj_data = np.load(data_path / "reddit_adj.npz")
    adj_matrix = sp.csr_matrix(
        (adj_data['data'], adj_data['indices'], adj_data['indptr']),
        shape=adj_data['shape']
    )
    
    logger.info(f"Loaded graph: {adj_matrix.shape[0]} nodes, {adj_matrix.nnz} edges")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Train/Val/Test sizes: {len(splits_and_labels['train_index'])}/{len(splits_and_labels['val_index'])}/{len(splits_and_labels['test_index'])}")
    
    return adj_matrix, features, splits_and_labels


def create_networkx_graph(adj_matrix: sp.csr_matrix, 
                         features: np.ndarray, 
                         splits_and_labels: Dict[str, np.ndarray]) -> nx.Graph:
    """
    Create NetworkX graph with proper node attributes for NEExT.
    
    Args:
        adj_matrix: Sparse adjacency matrix
        features: Node features array
        splits_and_labels: Train/val/test splits and labels
        
    Returns:
        NetworkX Graph object formatted for NEExT
    """
    logger.info("Creating NetworkX graph...")
    
    # Create graph from adjacency matrix
    G = nx.from_scipy_sparse_array(adj_matrix)
    
    # Set graph-level label (required by NEExT for graph classification)
    G.graph['label'] = 0  # Single graph, arbitrary label
    
    logger.info("Adding node attributes...")
    
    # Create label mapping for all nodes
    node_labels = {}
    node_splits = {}
    
    # Map train labels
    for i, node_idx in enumerate(splits_and_labels['train_index']):
        node_labels[node_idx] = splits_and_labels['y_train'][i]
        node_splits[node_idx] = 'train'
    
    # Map val labels  
    for i, node_idx in enumerate(splits_and_labels['val_index']):
        node_labels[node_idx] = splits_and_labels['y_val'][i]
        node_splits[node_idx] = 'val'
        
    # Map test labels
    for i, node_idx in enumerate(splits_and_labels['test_index']):
        node_labels[node_idx] = splits_and_labels['y_test'][i]
        node_splits[node_idx] = 'test'
    
    # Add node attributes
    for node_id in G.nodes():
        # Store individual features as separate attributes (NEExT-compatible)
        for feat_idx in range(features.shape[1]):
            G.nodes[node_id][f'feature_{feat_idx}'] = float(features[node_id, feat_idx])
        
        # Store subreddit label for node classification
        if node_id in node_labels:
            G.nodes[node_id]['subreddit_label'] = int(node_labels[node_id])
            G.nodes[node_id]['split'] = node_splits[node_id]
        else:
            # Handle any nodes not in train/val/test (shouldn't happen but be safe)
            G.nodes[node_id]['subreddit_label'] = -1
            G.nodes[node_id]['split'] = 'unknown'
    
    logger.info(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Log some statistics
    train_nodes = sum(1 for _, data in G.nodes(data=True) if data['split'] == 'train')
    val_nodes = sum(1 for _, data in G.nodes(data=True) if data['split'] == 'val') 
    test_nodes = sum(1 for _, data in G.nodes(data=True) if data['split'] == 'test')
    
    unique_labels = set(data['subreddit_label'] for _, data in G.nodes(data=True) if data['subreddit_label'] != -1)
    
    logger.info(f"Split distribution - Train: {train_nodes}, Val: {val_nodes}, Test: {test_nodes}")
    logger.info(f"Number of unique subreddit labels: {len(unique_labels)}")
    logger.info(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
    
    return G


def save_networkx_graph(G: nx.Graph, output_path: str = "reddit_networkx.pkl") -> None:
    """
    Save NetworkX graph to pickle file.
    
    Args:
        G: NetworkX graph
        output_path: Output file path
    """
    logger.info(f"Saving graph to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size for logging
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Graph saved successfully. File size: {file_size_mb:.1f} MB")


def validate_neext_compatibility(G: nx.Graph) -> bool:
    """
    Validate that the graph is properly formatted for NEExT.
    
    Args:
        G: NetworkX graph
        
    Returns:
        True if compatible, False otherwise
    """
    logger.info("Validating NEExT compatibility...")
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Graph-level label exists
    if 'label' in G.graph:
        logger.info("✓ Graph-level label found")
        checks_passed += 1
    else:
        logger.error("✗ Missing graph-level label")
    
    # Check 2: Node features present
    sample_node = next(iter(G.nodes()))
    if any(attr.startswith('feature_') for attr in G.nodes[sample_node]):
        feature_count = sum(1 for attr in G.nodes[sample_node] if attr.startswith('feature_'))
        logger.info(f"✓ Node features found ({feature_count} features per node)")
        checks_passed += 1
    else:
        logger.error("✗ Missing node features")
    
    # Check 3: Subreddit labels present
    if 'subreddit_label' in G.nodes[sample_node]:
        logger.info("✓ Subreddit labels found")
        checks_passed += 1
    else:
        logger.error("✗ Missing subreddit labels")
    
    # Check 4: Split information present
    if 'split' in G.nodes[sample_node]:
        logger.info("✓ Split information found")
        checks_passed += 1
    else:
        logger.error("✗ Missing split information")
    
    # Check 5: Graph is connected (or at least has edges)
    if G.number_of_edges() > 0:
        logger.info(f"✓ Graph has edges ({G.number_of_edges()} edges)")
        checks_passed += 1
    else:
        logger.error("✗ Graph has no edges")
    
    success = checks_passed == total_checks
    logger.info(f"Validation: {checks_passed}/{total_checks} checks passed")
    
    return success


def main():
    """Main function to load Reddit data and create NetworkX graph."""
    logger.info("Starting Reddit to NetworkX conversion...")
    
    try:
        # Load raw data
        adj_matrix, features, splits_and_labels = load_reddit_data()
        
        # Create NetworkX graph
        G = create_networkx_graph(adj_matrix, features, splits_and_labels)
        
        # Validate compatibility
        if not validate_neext_compatibility(G):
            logger.error("Graph validation failed!")
            return
        
        # Save graph
        save_networkx_graph(G, "reddit_networkx.pkl")
        
        logger.info("Reddit to NetworkX conversion completed successfully!")
        
        # Print usage example
        print("\n" + "="*60)
        print("USAGE EXAMPLE:")
        print("="*60)
        print("# Load the graph for use with NEExT:")
        print("import pickle")
        print("from NEExT.framework import NEExT")
        print("")
        print("# Load the saved graph")
        print("with open('reddit_networkx.pkl', 'rb') as f:")
        print("    reddit_graph = pickle.load(f)")
        print("")
        print("# Use with NEExT")
        print("nxt = NEExT()")
        print("collection = nxt.load_from_networkx([reddit_graph])")
        print("")
        print("# For node classification via egonets:")
        print("from NEExT.collections import EgonetCollection")
        print("egonet_collection = EgonetCollection(egonet_feature_target='subreddit_label')")
        print("egonet_collection.compute_k_hop_egonets(collection, k_hop=2)")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()