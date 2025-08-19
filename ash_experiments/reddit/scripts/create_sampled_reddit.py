#!/usr/bin/env python3
"""
Create a stratified sample of the Reddit dataset preserving class distribution and graph structure.

This script creates a smaller version (5% by default) of the Reddit graph while:
- Maintaining class distribution across 41 subreddits
- Preserving train/val/test split proportions
- Keeping graph connectivity through BFS expansion
- Retaining all node features and attributes
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict, deque
import logging
from typing import Set, Dict, List, Tuple
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StratifiedGraphSampler:
    """Stratified sampling of large graphs preserving class distribution and connectivity."""
    
    def __init__(self, graph: nx.Graph, sample_rate: float = 0.05, random_seed: int = 42):
        """
        Initialize the sampler.
        
        Args:
            graph: NetworkX graph with node attributes
            sample_rate: Fraction of nodes to sample (0.05 = 5%)
            random_seed: Random seed for reproducibility
        """
        self.graph = graph
        self.sample_rate = sample_rate
        self.target_size = int(graph.number_of_nodes() * sample_rate)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info(f"Initialized sampler: {graph.number_of_nodes()} nodes → {self.target_size} target nodes")
        
    def _get_class_distribution(self) -> Dict:
        """Analyze class and split distribution in the graph."""
        class_dist = defaultdict(lambda: {'train': [], 'val': [], 'test': [], 'unknown': []})
        
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get('subreddit_label', -1)
            split = attrs.get('split', 'unknown')
            if label != -1:
                class_dist[label][split].append(node)
        
        return class_dist
    
    def _select_stratified_seeds(self, class_dist: Dict, num_seeds: int) -> Set[int]:
        """
        Select seed nodes maintaining class and split distribution.
        
        Args:
            class_dist: Dictionary of class labels to split to node lists
            num_seeds: Total number of seeds to select
            
        Returns:
            Set of seed node IDs
        """
        seeds = set()
        
        # Calculate seeds per class (proportional to original distribution)
        class_sizes = {label: sum(len(nodes) for nodes in splits.values()) 
                      for label, splits in class_dist.items()}
        total_labeled = sum(class_sizes.values())
        
        seeds_per_class = {}
        for label, size in class_sizes.items():
            # At least 1 seed per class, proportional allocation for the rest
            min_seeds = 1
            proportional_seeds = int((size / total_labeled) * num_seeds)
            seeds_per_class[label] = max(min_seeds, proportional_seeds)
        
        # Adjust if we have too many seeds
        while sum(seeds_per_class.values()) > num_seeds:
            # Reduce from largest classes
            max_label = max(seeds_per_class.items(), key=lambda x: x[1])[0]
            if seeds_per_class[max_label] > 1:
                seeds_per_class[max_label] -= 1
        
        logger.info(f"Seeds per class distribution: {len(seeds_per_class)} classes")
        
        # Select seeds from each class maintaining train/val/test proportions
        for label, num_class_seeds in seeds_per_class.items():
            splits = class_dist[label]
            
            # Calculate split proportions for this class
            split_sizes = {split: len(nodes) for split, nodes in splits.items()}
            class_total = sum(split_sizes.values())
            
            if class_total == 0:
                continue
                
            # Allocate seeds per split
            split_seeds = {
                'train': max(1, int(num_class_seeds * split_sizes['train'] / class_total)),
                'val': max(0, int(num_class_seeds * split_sizes['val'] / class_total)),
                'test': max(0, int(num_class_seeds * split_sizes['test'] / class_total))
            }
            
            # Adjust to match exact number
            current_total = sum(split_seeds.values())
            if current_total < num_class_seeds and split_sizes['train'] > 0:
                split_seeds['train'] += num_class_seeds - current_total
            elif current_total > num_class_seeds and split_seeds['train'] > 1:
                split_seeds['train'] -= current_total - num_class_seeds
            
            # Select random nodes from each split
            for split, num_split_seeds in split_seeds.items():
                if num_split_seeds > 0 and len(splits[split]) > 0:
                    selected = np.random.choice(
                        splits[split], 
                        size=min(num_split_seeds, len(splits[split])),
                        replace=False
                    )
                    seeds.update(selected)
        
        logger.info(f"Selected {len(seeds)} seed nodes across all classes")
        return seeds
    
    def _expand_via_bfs(self, seeds: Set[int], target_size: int) -> Set[int]:
        """
        Expand from seed nodes using BFS to maintain connectivity.
        
        Args:
            seeds: Initial seed nodes
            target_size: Target number of nodes
            
        Returns:
            Expanded set of nodes
        """
        sampled_nodes = set(seeds)
        
        # Priority queue: nodes to explore
        queue = deque(seeds)
        
        # Track visited to avoid revisiting
        visited = set(seeds)
        
        logger.info(f"Starting BFS expansion from {len(seeds)} seeds to {target_size} nodes")
        
        while queue and len(sampled_nodes) < target_size:
            current = queue.popleft()
            
            # Get neighbors and shuffle for randomness
            neighbors = list(self.graph.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
                    
                    if len(sampled_nodes) >= target_size:
                        break
            
            # Progress update
            if len(sampled_nodes) % 1000 == 0:
                logger.info(f"  Expanded to {len(sampled_nodes)} nodes...")
        
        logger.info(f"BFS expansion complete: {len(sampled_nodes)} nodes sampled")
        return sampled_nodes
    
    def _create_subgraph(self, sampled_nodes: Set[int]) -> nx.Graph:
        """
        Create subgraph with all attributes preserved.
        
        Args:
            sampled_nodes: Set of nodes to include
            
        Returns:
            Subgraph with all attributes
        """
        logger.info("Creating subgraph with preserved attributes...")
        
        # Create subgraph
        subgraph = self.graph.subgraph(sampled_nodes).copy()
        
        # Ensure it's not a view but a proper copy
        subgraph = nx.Graph(subgraph)
        
        # Preserve graph-level attributes
        subgraph.graph = self.graph.graph.copy()
        
        logger.info(f"Subgraph created: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def sample(self) -> nx.Graph:
        """
        Perform stratified sampling of the graph.
        
        Returns:
            Sampled subgraph
        """
        logger.info("Starting stratified sampling...")
        
        # Get class distribution
        class_dist = self._get_class_distribution()
        logger.info(f"Found {len(class_dist)} classes in the graph")
        
        # Select stratified seeds (start with ~20% of target as seeds for good coverage)
        num_seeds = max(len(class_dist) * 3, int(self.target_size * 0.2))
        seeds = self._select_stratified_seeds(class_dist, num_seeds)
        
        # Expand via BFS to target size
        sampled_nodes = self._expand_via_bfs(seeds, self.target_size)
        
        # Create subgraph
        subgraph = self._create_subgraph(sampled_nodes)
        
        return subgraph


def validate_sampled_graph(original: nx.Graph, sampled: nx.Graph) -> None:
    """
    Validate the sampled graph maintains key properties.
    
    Args:
        original: Original full graph
        sampled: Sampled subgraph
    """
    logger.info("Validating sampled graph...")
    
    # Size validation
    sample_rate = sampled.number_of_nodes() / original.number_of_nodes()
    logger.info(f"Sample rate: {sample_rate:.1%} ({sampled.number_of_nodes()} / {original.number_of_nodes()} nodes)")
    logger.info(f"Edges: {sampled.number_of_edges()} (original: {original.number_of_edges()})")
    
    # Class distribution
    def get_label_dist(graph):
        labels = defaultdict(int)
        for _, attrs in graph.nodes(data=True):
            label = attrs.get('subreddit_label', -1)
            if label != -1:
                labels[label] += 1
        return labels
    
    orig_labels = get_label_dist(original)
    samp_labels = get_label_dist(sampled)
    
    logger.info(f"Classes in original: {len(orig_labels)}, in sample: {len(samp_labels)}")
    
    # Check coverage (what % of classes are represented)
    coverage = len(samp_labels) / len(orig_labels) if len(orig_labels) > 0 else 0
    logger.info(f"Class coverage: {coverage:.1%}")
    
    # Split distribution
    def get_split_dist(graph):
        splits = defaultdict(int)
        for _, attrs in graph.nodes(data=True):
            split = attrs.get('split', 'unknown')
            splits[split] += 1
        return splits
    
    orig_splits = get_split_dist(original)
    samp_splits = get_split_dist(sampled)
    
    logger.info("Split distribution:")
    for split in ['train', 'val', 'test']:
        if split in orig_splits and orig_splits[split] > 0:
            orig_pct = orig_splits[split] / original.number_of_nodes() * 100
            samp_pct = samp_splits.get(split, 0) / sampled.number_of_nodes() * 100 if sampled.number_of_nodes() > 0 else 0
            logger.info(f"  {split}: original={orig_pct:.1f}%, sampled={samp_pct:.1f}%")
    
    # Check connectivity
    if sampled.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(sampled), key=len)
        connectivity = len(largest_cc) / sampled.number_of_nodes()
        logger.info(f"Largest connected component: {connectivity:.1%} of sampled nodes")
    
    # Feature preservation
    if sampled.number_of_nodes() > 0:
        sample_node = next(iter(sampled.nodes()))
        sample_attrs = sampled.nodes[sample_node]
        feature_count = sum(1 for attr in sample_attrs if attr.startswith('feature_'))
        logger.info(f"Features per node: {feature_count}")
        logger.info(f"Sample node attributes: subreddit_label={sample_attrs.get('subreddit_label')}, split={sample_attrs.get('split')}")


def main():
    """Main function to create sampled Reddit graph."""
    
    # Configuration
    INPUT_FILE = "reddit_networkx.pkl"
    OUTPUT_FILE = "reddit_networkx_5pct.pkl"
    SAMPLE_RATE = 0.05  # 5% sample
    
    try:
        # Load original graph
        logger.info(f"Loading original graph from {INPUT_FILE}...")
        with open(INPUT_FILE, 'rb') as f:
            original_graph = pickle.load(f)
        
        logger.info(f"Loaded graph: {original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges")
        
        # Create sampler and perform sampling
        sampler = StratifiedGraphSampler(original_graph, sample_rate=SAMPLE_RATE)
        sampled_graph = sampler.sample()
        
        # Validate the sampled graph
        validate_sampled_graph(original_graph, sampled_graph)
        
        # Save sampled graph
        logger.info(f"Saving sampled graph to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(sampled_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Check file size
        file_size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
        logger.info(f"Sampled graph saved: {file_size_mb:.1f} MB")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("SAMPLED GRAPH CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"Original: {original_graph.number_of_nodes()} nodes → Sampled: {sampled_graph.number_of_nodes()} nodes")
        print(f"File size: {file_size_mb:.1f} MB")
        print()
        print("Usage:")
        print("```python")
        print("import pickle")
        print("from NEExT.framework import NEExT")
        print("")
        print(f"# Load the sampled graph")
        print(f"with open('{OUTPUT_FILE}', 'rb') as f:")
        print("    reddit_sample = pickle.load(f)")
        print("")
        print("# Use with NEExT (no sampling needed, already small)")
        print("nxt = NEExT()")
        print("collection = nxt.load_from_networkx([reddit_sample], node_sample_rate=1.0)")
        print("```")
        print("="*60)
        
    except FileNotFoundError:
        logger.error(f"File {INPUT_FILE} not found. Please run load_reddit_to_networkx.py first.")
        raise
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        raise


if __name__ == "__main__":
    main()