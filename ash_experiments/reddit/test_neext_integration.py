#!/usr/bin/env python3
"""
Test loading the Reddit NetworkX graph into NEExT framework.
"""

import pickle
import sys
from pathlib import Path

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_neext_loading():
    """Test loading Reddit graph into NEExT."""
    logger.info("Testing NEExT integration...")
    
    # Load the NetworkX graph
    logger.info("Loading Reddit NetworkX graph...")
    with open('reddit_networkx.pkl', 'rb') as f:
        reddit_graph = pickle.load(f)
    
    logger.info(f"Loaded graph: {reddit_graph.number_of_nodes()} nodes, {reddit_graph.number_of_edges()} edges")
    
    # Test basic NEExT loading
    logger.info("Loading graph into NEExT...")
    nxt = NEExT()
    collection = nxt.load_from_networkx(
        [reddit_graph],
        reindex_nodes=False,  # Keep original node IDs for now
        filter_largest_component=False,  # Keep full graph
        node_sample_rate=0.01  # Use 1% sample for testing
    )
    
    logger.info(f"NEExT collection created with {len(collection.graphs)} graph(s)")
    
    # Get first graph from collection
    graph = collection.graphs[0]
    logger.info(f"Graph in collection: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    
    # Test EgonetCollection for node classification
    logger.info("Testing EgonetCollection for node classification...")
    egonet_collection = EgonetCollection(egonet_feature_target='subreddit_label')
    
    # Use a smaller k_hop for testing
    egonet_collection.compute_k_hop_egonets(collection, k_hop=1)
    
    logger.info(f"EgonetCollection created with {len(egonet_collection.graphs)} egonets")
    
    # Sample some egonets to verify structure
    if egonet_collection.graphs:
        sample_egonet = egonet_collection.graphs[0]
        logger.info(f"Sample egonet: {len(sample_egonet['nodes'])} nodes, {len(sample_egonet['edges'])} edges")
        logger.info(f"Sample egonet label: {sample_egonet.get('graph_label', 'No label')}")
    
    logger.info("NEExT integration test completed successfully!")
    
    return collection, egonet_collection


def inspect_node_attributes():
    """Inspect node attributes in the loaded graph."""
    logger.info("Inspecting node attributes...")
    
    with open('reddit_networkx.pkl', 'rb') as f:
        reddit_graph = pickle.load(f)
    
    # Get a sample node
    sample_node = next(iter(reddit_graph.nodes()))
    node_attrs = reddit_graph.nodes[sample_node]
    
    logger.info(f"Sample node {sample_node} attributes:")
    logger.info(f"  - subreddit_label: {node_attrs.get('subreddit_label', 'Missing')}")
    logger.info(f"  - split: {node_attrs.get('split', 'Missing')}")
    
    # Count feature attributes
    feature_attrs = [attr for attr in node_attrs if attr.startswith('feature_')]
    logger.info(f"  - Number of features: {len(feature_attrs)}")
    
    if feature_attrs:
        logger.info(f"  - Sample features: {feature_attrs[:5]}...")
        logger.info(f"  - Sample feature values: {[node_attrs[attr] for attr in feature_attrs[:3]]}")
    
    # Check split distribution
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    for node, data in reddit_graph.nodes(data=True):
        split = data.get('split', 'unknown')
        split_counts[split] += 1
    
    logger.info(f"Split distribution: {split_counts}")
    
    # Check label distribution
    label_counts = {}
    for node, data in reddit_graph.nodes(data=True):
        label = data.get('subreddit_label', -1)
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Number of unique labels: {len([l for l in label_counts.keys() if l != -1])}")
    logger.info(f"Nodes with missing labels: {label_counts.get(-1, 0)}")


if __name__ == "__main__":
    try:
        # Inspect the loaded graph
        inspect_node_attributes()
        
        # Test NEExT integration
        collection, egonet_collection = test_neext_loading()
        
        print("\n" + "="*60)
        print("SUCCESS: Reddit graph successfully loaded into NEExT!")
        print("="*60)
        print(f"Graph Collection: {len(collection.graphs)} graph(s)")
        print(f"Egonet Collection: {len(egonet_collection.graphs)} egonet(s)")
        print("Ready for graph embeddings, feature computation, and ML experiments!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise