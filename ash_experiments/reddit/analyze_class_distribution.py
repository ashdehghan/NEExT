#!/usr/bin/env python3
"""
Analyze and visualize the class distribution in Reddit datasets.
Shows distribution for full dataset and various sample sizes.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_graph(file_path, title):
    """Analyze class distribution in a graph."""
    logger.info(f"\nAnalyzing: {title}")
    logger.info("="*60)
    
    try:
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        
        # Count class distribution
        class_counts = Counter()
        split_counts = Counter()
        
        for _, attrs in graph.nodes(data=True):
            label = attrs.get('subreddit_label', -1)
            split = attrs.get('split', 'unknown')
            if label != -1:
                class_counts[label] += 1
            split_counts[split] += 1
        
        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Statistics
        total_nodes = graph.number_of_nodes()
        labeled_nodes = sum(class_counts.values())
        num_classes = len(class_counts)
        
        logger.info(f"Total nodes: {total_nodes:,}")
        logger.info(f"Labeled nodes: {labeled_nodes:,} ({labeled_nodes/total_nodes*100:.1f}%)")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Split distribution: {dict(split_counts)}")
        
        if sorted_classes:
            counts = [c for _, c in sorted_classes]
            logger.info(f"Min class size: {min(counts):,}")
            logger.info(f"Max class size: {max(counts):,}")
            logger.info(f"Mean class size: {np.mean(counts):.1f}")
            logger.info(f"Median class size: {np.median(counts):.1f}")
            
            logger.info("\nTop 10 classes:")
            for i, (label, count) in enumerate(sorted_classes[:10], 1):
                logger.info(f"  {i:2d}. Class {label:2d}: {count:5,} nodes ({count/labeled_nodes*100:5.2f}%)")
            
            logger.info("\nBottom 10 classes:")
            for i, (label, count) in enumerate(sorted_classes[-10:], 1):
                logger.info(f"  {i:2d}. Class {label:2d}: {count:5,} nodes ({count/labeled_nodes*100:5.2f}%)")
        
        return sorted_classes, total_nodes
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None, 0


def plot_distributions(distributions):
    """Create visualization of class distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reddit Dataset Class Distributions', fontsize=16, fontweight='bold')
    
    for idx, (title, sorted_classes, total_nodes) in enumerate(distributions):
        if sorted_classes is None:
            continue
            
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data
        labels = [f"C{label}" for label, _ in sorted_classes]
        counts = [count for _, count in sorted_classes]
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, counts, color='steelblue', alpha=0.8)
        
        # Highlight small classes (< 100 samples)
        for i, count in enumerate(counts):
            if count < 100:
                bars[i].set_color('red')
            elif count < 500:
                bars[i].set_color('orange')
        
        ax.set_xlabel('Subreddit Class', fontsize=10)
        ax.set_ylabel('Number of Nodes', fontsize=10)
        ax.set_title(f'{title}\n({total_nodes:,} total nodes, {len(sorted_classes)} classes)', 
                    fontsize=11, fontweight='bold')
        
        # Rotate labels for readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at key thresholds
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Min for ML (100)')
        ax.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='Recommended (500)')
        
        if idx == 0:  # Only show legend on first plot
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'reddit_class_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved to: {output_file}")
    
    # Also create a log-scale version for better visibility of small classes
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    
    # Use the full dataset for detailed view
    if distributions[0][1] is not None:
        sorted_classes = distributions[0][1]
        labels = [str(label) for label, _ in sorted_classes]
        counts = [count for _, count in sorted_classes]
        
        x_pos = np.arange(len(labels))
        bars = ax2.bar(x_pos, counts, color='steelblue', alpha=0.8)
        
        # Color code by size
        for i, count in enumerate(counts):
            if count < 50:
                bars[i].set_color('darkred')
            elif count < 100:
                bars[i].set_color('red')
            elif count < 500:
                bars[i].set_color('orange')
            elif count < 1000:
                bars[i].set_color('yellow')
        
        ax2.set_yscale('log')
        ax2.set_xlabel('Subreddit Class ID', fontsize=12)
        ax2.set_ylabel('Number of Nodes (log scale)', fontsize=12)
        ax2.set_title('Reddit Full Dataset - Class Distribution (Log Scale)', fontsize=14, fontweight='bold')
        
        ax2.set_xticks(x_pos[::2])  # Show every other label
        ax2.set_xticklabels(labels[::2], rotation=45, ha='right', fontsize=10)
        
        ax2.grid(axis='y', alpha=0.3, which='both')
        
        # Add threshold lines
        ax2.axhline(y=50, color='darkred', linestyle='--', alpha=0.5, label='Critical (<50)')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Minimum (100)')
        ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='Recommended (500)')
        ax2.axhline(y=1000, color='yellow', linestyle='--', alpha=0.5, label='Good (1000)')
        
        ax2.legend(loc='upper right')
        
        # Add value labels for smallest classes
        for i, (label, count) in enumerate(sorted_classes[-10:]):
            ax2.annotate(str(count), 
                        xy=(x_pos[len(sorted_classes) - 10 + i], count),
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=8,
                        color='darkred')
    
    plt.tight_layout()
    plt.savefig('reddit_class_distribution_log.png', dpi=150, bbox_inches='tight')
    logger.info(f"Log-scale plot saved to: reddit_class_distribution_log.png")
    
    plt.show()


def main():
    """Main analysis function."""
    logger.info("REDDIT DATASET CLASS DISTRIBUTION ANALYSIS")
    logger.info("="*60)
    
    # Files to analyze
    files = [
        ("reddit_networkx.pkl", "Full Dataset (100%)"),
        ("reddit_networkx_20pct.pkl", "20% Sample"),
        ("reddit_networkx_5pct.pkl", "5% Sample"),
        ("reddit_networkx_1pct.pkl", "1% Sample (if exists)")
    ]
    
    distributions = []
    
    for file_path, title in files:
        if Path(file_path).exists():
            sorted_classes, total_nodes = analyze_graph(file_path, title)
            distributions.append((title, sorted_classes, total_nodes))
        else:
            logger.info(f"\nSkipping {title} - file not found")
            if len(distributions) < 4:
                distributions.append((title, None, 0))
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY - MINIMUM CLASS SIZES")
    logger.info("="*60)
    
    for title, sorted_classes, total_nodes in distributions:
        if sorted_classes:
            counts = [c for _, c in sorted_classes]
            min_count = min(counts)
            classes_under_100 = sum(1 for c in counts if c < 100)
            classes_under_500 = sum(1 for c in counts if c < 500)
            
            logger.info(f"\n{title}:")
            logger.info(f"  Minimum class size: {min_count}")
            logger.info(f"  Classes with <100 samples: {classes_under_100}")
            logger.info(f"  Classes with <500 samples: {classes_under_500}")
    
    # Create visualizations
    if any(d[1] is not None for d in distributions):
        plot_distributions(distributions)
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info("\nKey insights:")
    logger.info("- Class imbalance is severe (some classes have <100 samples)")
    logger.info("- For robust ML, need at least 100 samples per class")
    logger.info("- Consider focusing on top 20 classes with sufficient samples")
    logger.info("- Or use stratified sampling with minimum class size guarantees")


if __name__ == "__main__":
    main()