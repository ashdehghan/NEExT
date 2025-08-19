#!/usr/bin/env python3
"""
Create a 20% stratified sample of the Reddit dataset for robust experiments.

This creates a larger sample than the 5% version, providing enough data for
reliable classification while still being manageable in size.
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict, deque
import logging
from typing import Set, Dict, List, Tuple
import random
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parents[1]))
from scripts.create_sampled_reddit import StratifiedGraphSampler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Create 20% sampled Reddit graph."""
    
    # Configuration
    INPUT_FILE = "../reddit_networkx.pkl"  # Full graph
    OUTPUT_FILE = "../reddit_networkx_20pct.pkl"  # 20% sample
    SAMPLE_RATE = 0.20  # 20% sample
    
    logger.info("="*80)
    logger.info("CREATING 20% REDDIT SAMPLE")
    logger.info("="*80)
    
    try:
        # Load original graph
        logger.info(f"Loading original graph from {INPUT_FILE}...")
        with open(INPUT_FILE, 'rb') as f:
            original_graph = pickle.load(f)
        
        logger.info(f"Loaded graph: {original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges")
        logger.info(f"Target sample size: {int(original_graph.number_of_nodes() * SAMPLE_RATE)} nodes")
        
        # Create sampler and perform sampling
        logger.info(f"\nStarting stratified sampling at {SAMPLE_RATE:.0%}...")
        sampler = StratifiedGraphSampler(original_graph, sample_rate=SAMPLE_RATE, random_seed=42)
        sampled_graph = sampler.sample()
        
        # Validate the sampled graph
        logger.info("\n" + "="*60)
        logger.info("VALIDATION RESULTS")
        logger.info("="*60)
        
        # Size statistics
        logger.info(f"Sample size: {sampled_graph.number_of_nodes()} nodes ({sampled_graph.number_of_nodes()/original_graph.number_of_nodes():.1%})")
        logger.info(f"Edges: {sampled_graph.number_of_edges()} ({sampled_graph.number_of_edges()/original_graph.number_of_edges():.1%} of original)")
        
        # Class distribution
        def get_label_dist(graph):
            labels = defaultdict(int)
            for _, attrs in graph.nodes(data=True):
                label = attrs.get('subreddit_label', -1)
                if label != -1:
                    labels[label] += 1
            return labels
        
        orig_labels = get_label_dist(original_graph)
        samp_labels = get_label_dist(sampled_graph)
        
        logger.info(f"Classes represented: {len(samp_labels)} of {len(orig_labels)} ({len(samp_labels)/len(orig_labels):.0%})")
        
        # Split distribution
        def get_split_dist(graph):
            splits = defaultdict(int)
            for _, attrs in graph.nodes(data=True):
                split = attrs.get('split', 'unknown')
                splits[split] += 1
            return splits
        
        samp_splits = get_split_dist(sampled_graph)
        total_nodes = sampled_graph.number_of_nodes()
        
        logger.info("\nSplit distribution in sample:")
        logger.info(f"  Train: {samp_splits['train']} nodes ({samp_splits['train']/total_nodes:.1%})")
        logger.info(f"  Val: {samp_splits['val']} nodes ({samp_splits['val']/total_nodes:.1%})")
        logger.info(f"  Test: {samp_splits['test']} nodes ({samp_splits['test']/total_nodes:.1%})")
        
        # Check connectivity
        if sampled_graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(sampled_graph), key=len)
            connectivity = len(largest_cc) / sampled_graph.number_of_nodes()
            logger.info(f"\nLargest connected component: {len(largest_cc)} nodes ({connectivity:.1%})")
        
        # Feature check
        sample_node = next(iter(sampled_graph.nodes()))
        sample_attrs = sampled_graph.nodes[sample_node]
        feature_count = sum(1 for attr in sample_attrs if attr.startswith('feature_'))
        logger.info(f"Features preserved: {feature_count} per node")
        
        # Class balance check for ML
        min_class_size = min(samp_labels.values())
        max_class_size = max(samp_labels.values())
        logger.info(f"\nClass sizes: min={min_class_size}, max={max_class_size}")
        
        if min_class_size < 2:
            logger.warning("Some classes have <2 samples - may cause issues with stratified splitting")
        else:
            logger.info("✓ All classes have sufficient samples for stratified splitting")
        
        # Save sampled graph
        logger.info(f"\nSaving 20% sample to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(sampled_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Check file size
        file_size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
        logger.info(f"File saved: {file_size_mb:.1f} MB")
        
        # Summary
        print("\n" + "="*80)
        print("20% SAMPLE CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"Nodes: {original_graph.number_of_nodes()} → {sampled_graph.number_of_nodes()}")
        print(f"Edges: {original_graph.number_of_edges()} → {sampled_graph.number_of_edges()}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Classes: {len(samp_labels)} represented")
        print(f"Connectivity: {connectivity:.1%} in largest component")
        print("\nThis sample is ideal for:")
        print("- Full NEExT experiments with reliable results")
        print("- Testing with all 41 subreddit classes")
        print("- Sufficient samples for train/val/test splits")
        print("="*80)
        
    except FileNotFoundError:
        logger.error(f"File {INPUT_FILE} not found. Please run load_reddit_to_networkx.py first.")
        raise
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        raise


if __name__ == "__main__":
    main()