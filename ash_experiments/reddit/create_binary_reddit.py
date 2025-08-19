#!/usr/bin/env python3
"""
Create a binary classification version of the Reddit dataset.
Groups subreddits into SERIOUS (news/science/tech) vs ENTERTAINMENT (fun/media/art).
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define binary groupings based on analysis
SERIOUS_SUBREDDITS = [
    0,   # AskReddit (Q&A discussions)
    1,   # worldnews
    8,   # news
    18,  # science
    19,  # technology
    20,  # space
    21,  # gadgets
    23,  # politics
    25,  # history
    26,  # economics
    27,  # philosophy
    30,  # programming
    31,  # dataisbeautiful
    32,  # Futurology
]

ENTERTAINMENT_SUBREDDITS = [
    2,   # videos
    3,   # funny
    5,   # pics
    6,   # gaming
    9,   # gifs
    10,  # aww
    11,  # Music
    12,  # television
    13,  # books
    14,  # sports
    22,  # Art
    24,  # anime
    28,  # photography
    33,  # OldSchoolCool
    34,  # GetMotivated
    37,  # Showerthoughts
]


def create_binary_dataset(input_file, output_file, keep_unknown=False):
    """
    Convert multi-class Reddit dataset to binary classification.
    
    Args:
        input_file: Path to input NetworkX graph
        output_file: Path to save binary graph
        keep_unknown: If False, remove nodes not in either category
    """
    logger.info("="*80)
    logger.info("CREATING BINARY REDDIT DATASET")
    logger.info("="*80)
    
    # Load graph
    logger.info(f"Loading graph from {input_file}...")
    with open(input_file, 'rb') as f:
        graph = pickle.load(f)
    
    original_nodes = graph.number_of_nodes()
    original_edges = graph.number_of_edges()
    
    logger.info(f"Original graph: {original_nodes:,} nodes, {original_edges:,} edges")
    
    # Create mapping for binary labels
    serious_set = set(SERIOUS_SUBREDDITS)
    entertainment_set = set(ENTERTAINMENT_SUBREDDITS)
    
    # Statistics
    stats = {
        'serious': 0,
        'entertainment': 0,
        'unknown': 0,
        'removed': 0
    }
    
    # Track nodes to remove if not keeping unknown
    nodes_to_remove = []
    
    # Convert labels
    for node, attrs in graph.nodes(data=True):
        original_label = attrs.get('subreddit_label', -1)
        
        if original_label in serious_set:
            # SERIOUS = 0
            attrs['binary_label'] = 0
            attrs['binary_category'] = 'serious'
            attrs['original_subreddit'] = original_label
            stats['serious'] += 1
            
        elif original_label in entertainment_set:
            # ENTERTAINMENT = 1
            attrs['binary_label'] = 1
            attrs['binary_category'] = 'entertainment'
            attrs['original_subreddit'] = original_label
            stats['entertainment'] += 1
            
        else:
            # Unknown category
            attrs['original_subreddit'] = original_label
            stats['unknown'] += 1
            
            if keep_unknown:
                attrs['binary_label'] = -1
                attrs['binary_category'] = 'unknown'
            else:
                nodes_to_remove.append(node)
    
    # Remove unknown nodes if requested
    if not keep_unknown and nodes_to_remove:
        logger.info(f"Removing {len(nodes_to_remove):,} nodes not in binary categories...")
        graph.remove_nodes_from(nodes_to_remove)
        stats['removed'] = len(nodes_to_remove)
    
    # Update graph label for NEExT (graph-level classification)
    graph.graph['label'] = 0  # Single graph
    graph.graph['task'] = 'binary_node_classification'
    graph.graph['classes'] = ['serious', 'entertainment']
    
    # Final statistics
    final_nodes = graph.number_of_nodes()
    final_edges = graph.number_of_edges()
    
    logger.info("\n" + "="*60)
    logger.info("CONVERSION STATISTICS")
    logger.info("="*60)
    logger.info(f"Serious nodes: {stats['serious']:,} ({stats['serious']/original_nodes*100:.1f}%)")
    logger.info(f"Entertainment nodes: {stats['entertainment']:,} ({stats['entertainment']/original_nodes*100:.1f}%)")
    logger.info(f"Unknown nodes: {stats['unknown']:,} ({stats['unknown']/original_nodes*100:.1f}%)")
    
    if stats['removed'] > 0:
        logger.info(f"Removed nodes: {stats['removed']:,}")
    
    logger.info(f"\nFinal graph: {final_nodes:,} nodes, {final_edges:,} edges")
    logger.info(f"Node reduction: {(1 - final_nodes/original_nodes)*100:.1f}%")
    logger.info(f"Edge reduction: {(1 - final_edges/original_edges)*100:.1f}%")
    
    # Check balance
    if stats['serious'] > 0 and stats['entertainment'] > 0:
        balance = min(stats['serious'], stats['entertainment']) / max(stats['serious'], stats['entertainment'])
        logger.info(f"Class balance: {balance:.2f} (1.0 = perfect balance)")
    
    # Analyze splits for binary task
    split_stats = {'train': {'serious': 0, 'entertainment': 0},
                   'val': {'serious': 0, 'entertainment': 0},
                   'test': {'serious': 0, 'entertainment': 0}}
    
    for node, attrs in graph.nodes(data=True):
        if 'binary_label' in attrs and attrs['binary_label'] >= 0:
            split = attrs.get('split', 'unknown')
            category = attrs['binary_category']
            if split in split_stats and category in ['serious', 'entertainment']:
                split_stats[split][category] += 1
    
    logger.info("\n" + "="*60)
    logger.info("SPLIT DISTRIBUTION")
    logger.info("="*60)
    
    for split in ['train', 'val', 'test']:
        s = split_stats[split]['serious']
        e = split_stats[split]['entertainment']
        total = s + e
        if total > 0:
            logger.info(f"{split.upper():5}: Total={total:5,} | Serious={s:5,} ({s/total*100:.1f}%) | Entertainment={e:5,} ({e/total*100:.1f}%)")
    
    # Save the binary graph
    logger.info(f"\nSaving binary graph to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"File saved: {file_size_mb:.1f} MB")
    
    return graph, stats


def main():
    """Create binary versions of Reddit datasets."""
    
    print("\n" + "="*80)
    print("REDDIT BINARY CLASSIFICATION DATASET CREATOR")
    print("="*80)
    print("\nThis will create binary versions with:")
    print("- Class 0: SERIOUS (news, science, tech, politics)")
    print("- Class 1: ENTERTAINMENT (funny, videos, gaming, art)")
    print("="*80)
    
    # Process different sample sizes
    datasets = [
        ("reddit_networkx_5pct.pkl", "reddit_binary_5pct.pkl", "5% Sample"),
        ("reddit_networkx_20pct.pkl", "reddit_binary_20pct.pkl", "20% Sample"),
    ]
    
    for input_file, output_file, description in datasets:
        if Path(input_file).exists():
            print(f"\nProcessing {description}...")
            graph, stats = create_binary_dataset(
                input_file, 
                output_file,
                keep_unknown=False  # Remove nodes not in binary categories
            )
        else:
            print(f"\nSkipping {description} - file not found")
    
    print("\n" + "="*80)
    print("BINARY DATASETS CREATED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Use reddit_binary_5pct.pkl for quick experiments")
    print("2. Binary task is much simpler - should train faster")
    print("3. More balanced classes - better for ML")
    print("="*80)


if __name__ == "__main__":
    main()