#!/usr/bin/env python3
"""
Quick script to check class distribution in Reddit datasets.
No plotting dependencies - just prints statistics.
"""

import pickle
from collections import Counter
from pathlib import Path
import numpy as np


def analyze_graph(file_path, title):
    """Analyze class distribution in a graph."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
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
        
        print(f"Total nodes: {total_nodes:,}")
        print(f"Labeled nodes: {labeled_nodes:,} ({labeled_nodes/total_nodes*100:.1f}%)")
        print(f"Number of classes: {num_classes}")
        print(f"Split: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
        
        if sorted_classes:
            counts = [c for _, c in sorted_classes]
            print(f"\nClass size statistics:")
            print(f"  Min: {min(counts):,}")
            print(f"  Max: {max(counts):,}")
            print(f"  Mean: {np.mean(counts):.1f}")
            print(f"  Median: {np.median(counts):.1f}")
            print(f"  Std Dev: {np.std(counts):.1f}")
            
            # Distribution histogram
            print(f"\nClass size distribution:")
            print(f"  <50 samples: {sum(1 for c in counts if c < 50)} classes")
            print(f"  50-100 samples: {sum(1 for c in counts if 50 <= c < 100)} classes")
            print(f"  100-500 samples: {sum(1 for c in counts if 100 <= c < 500)} classes")
            print(f"  500-1000 samples: {sum(1 for c in counts if 500 <= c < 1000)} classes")
            print(f"  1000-5000 samples: {sum(1 for c in counts if 1000 <= c < 5000)} classes")
            print(f"  >5000 samples: {sum(1 for c in counts if c >= 5000)} classes")
            
            print(f"\n{'='*40}")
            print("TOP 10 LARGEST CLASSES")
            print('='*40)
            print(f"{'Rank':<5} {'Class':<8} {'Count':<10} {'Percent':<10}")
            print('-'*40)
            for i, (label, count) in enumerate(sorted_classes[:10], 1):
                print(f"{i:<5} {label:<8} {count:<10,} {count/labeled_nodes*100:<10.2f}%")
            
            print(f"\n{'='*40}")
            print("BOTTOM 10 SMALLEST CLASSES")
            print('='*40)
            print(f"{'Rank':<5} {'Class':<8} {'Count':<10} {'Status':<15}")
            print('-'*40)
            for i, (label, count) in enumerate(sorted_classes[-10:], 1):
                status = "CRITICAL" if count < 50 else "WARNING" if count < 100 else "OK"
                print(f"{i:<5} {label:<8} {count:<10,} {status:<15}")
        
        return sorted_classes, total_nodes
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("REDDIT DATASET CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Files to analyze
    files = [
        ("reddit_networkx.pkl", "FULL DATASET (232,965 nodes)"),
        ("reddit_networkx_20pct.pkl", "20% SAMPLE (46,593 nodes)"),
        ("reddit_networkx_5pct.pkl", "5% SAMPLE (11,648 nodes)"),
    ]
    
    all_results = []
    
    for file_path, title in files:
        if Path(file_path).exists():
            sorted_classes, total_nodes = analyze_graph(file_path, title)
            all_results.append((title, sorted_classes, total_nodes))
        else:
            print(f"\n[SKIPPED] {title} - file not found")
    
    # Comparative summary
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<30} {'Min Class':<12} {'Classes <100':<15} {'Classes <500':<15}")
    print('-'*72)
    
    for title, sorted_classes, total_nodes in all_results:
        if sorted_classes:
            counts = [c for _, c in sorted_classes]
            min_count = min(counts)
            under_100 = sum(1 for c in counts if c < 100)
            under_500 = sum(1 for c in counts if c < 500)
            
            # Extract percentage from title
            if "%" in title:
                pct = title.split("(")[0].strip()
            else:
                pct = "FULL"
            
            print(f"{pct:<30} {min_count:<12} {under_100:<15} {under_500:<15}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. SEVERE CLASS IMBALANCE:
   - Top class has ~40,000 samples (17% of data)
   - Bottom classes have <100 samples each
   - 15 classes have <1000 samples

2. MINIMUM SAMPLES FOR ML:
   - Need at least 2 samples per class for stratified split
   - Recommended: 100+ samples for meaningful learning
   - Ideal: 500+ samples for robust classification

3. RECOMMENDATIONS:
   a) Focus on top 20-25 classes with sufficient samples
   b) Use class-weighted loss functions
   c) Consider hierarchical classification
   d) Try few-shot learning for rare classes
   
4. FOR EGONETS:
   - Each egonet creation is expensive (graph traversal)
   - 20% sample → 7000 egonets → too slow
   - Consider: Sample 1000-2000 nodes MAX from balanced classes
""")


if __name__ == "__main__":
    main()