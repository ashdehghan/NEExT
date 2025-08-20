#!/usr/bin/env python3
"""
Export Reddit Binary Graph to Gephi Format

This script converts the Reddit binary 5% NetworkX graph to GEXF format
for visualization in Gephi (https://gephi.org/).

The exported graph will include:
- Node attributes: binary_label, binary_category, subreddit_label
- Edge structure preserved
- Metadata for coloring/filtering in Gephi
"""

import pickle
import networkx as nx
from pathlib import Path
import time
from collections import Counter

def prepare_graph_for_gephi(graph):
    """
    Prepare the graph for Gephi export by cleaning attributes.
    
    Gephi works best with simple attribute types and clear labels.
    """
    print("\n[2/4] Preparing graph for Gephi export...")
    
    # Create a copy to avoid modifying original
    G_export = graph.copy()
    
    # Add graph metadata
    G_export.graph['name'] = 'Reddit Binary Classification Network'
    G_export.graph['description'] = 'Reddit 5% sample - Serious vs Entertainment subreddits'
    
    # Process node attributes for better Gephi visualization
    nodes_processed = 0
    for node in G_export.nodes():
        attrs = G_export.nodes[node]
        
        # Keep only essential attributes for visualization
        # Remove the 602 feature attributes to keep file size manageable
        attrs_to_keep = {
            'binary_label': attrs.get('binary_label', -1),
            'binary_category': attrs.get('binary_category', 'unknown'),
            'original_subreddit': attrs.get('original_subreddit', -1),
            'split': attrs.get('split', 'unknown')
        }
        
        # Clear all attributes
        attrs.clear()
        
        # Add back only visualization-relevant attributes
        attrs.update(attrs_to_keep)
        
        # Add visualization hints
        if attrs['binary_label'] == 0:
            attrs['color'] = 'blue'  # Serious - blue
            attrs['viz_label'] = 'Serious'
        elif attrs['binary_label'] == 1:
            attrs['color'] = 'red'   # Entertainment - red
            attrs['viz_label'] = 'Entertainment'
        else:
            attrs['color'] = 'gray'
            attrs['viz_label'] = 'Unknown'
        
        nodes_processed += 1
        if nodes_processed % 1000 == 0:
            print(f"      Processed {nodes_processed:,} nodes...")
    
    print(f"      Processed {nodes_processed:,} total nodes")
    
    # Add edge weights (all 1.0 for unweighted graph)
    for u, v in G_export.edges():
        G_export[u][v]['weight'] = 1.0
    
    return G_export

def analyze_graph_structure(graph):
    """
    Analyze and print graph statistics.
    """
    print("\n[3/4] Analyzing graph structure...")
    
    # Basic statistics
    print(f"      Nodes: {graph.number_of_nodes():,}")
    print(f"      Edges: {graph.number_of_edges():,}")
    print(f"      Density: {nx.density(graph):.6f}")
    
    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees)
    print(f"      Average degree: {avg_degree:.2f}")
    print(f"      Max degree: {max(degrees)}")
    print(f"      Min degree: {min(degrees)}")
    
    # Component analysis
    num_components = nx.number_connected_components(graph)
    largest_cc = max(nx.connected_components(graph), key=len)
    print(f"      Connected components: {num_components}")
    print(f"      Largest component: {len(largest_cc):,} nodes ({len(largest_cc)/graph.number_of_nodes()*100:.1f}%)")
    
    # Class distribution
    class_counts = Counter()
    for node, attrs in graph.nodes(data=True):
        class_counts[attrs.get('binary_category', 'unknown')] += 1
    
    print("\n      Class distribution:")
    for category, count in class_counts.most_common():
        print(f"        {category}: {count:,} nodes ({count/graph.number_of_nodes()*100:.1f}%)")

def export_to_gexf(graph, output_path):
    """
    Export the graph to GEXF format for Gephi.
    """
    print(f"\n[4/4] Exporting to GEXF format...")
    print(f"      Output file: {output_path}")
    
    start_time = time.time()
    
    # Write GEXF file
    nx.write_gexf(graph, output_path)
    
    elapsed = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"      Export complete in {elapsed:.1f} seconds")
    print(f"      File size: {file_size_mb:.1f} MB")

def main():
    """
    Main function to load and export Reddit binary graph.
    """
    print("\n" + "="*80)
    print("REDDIT BINARY GRAPH EXPORT TO GEPHI")
    print("="*80)
    print("\nThis script will export the Reddit binary 5% graph to GEXF format")
    print("for visualization in Gephi (https://gephi.org/)")
    print("="*80)
    
    # File paths
    input_file = "reddit_binary_5pct.pkl"
    output_file = "reddit_binary_5pct_gephi.gexf"
    
    # Load the graph
    print(f"\n[1/4] Loading Reddit binary graph...")
    start_time = time.time()
    
    with open(input_file, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"      Loaded in {time.time() - start_time:.1f} seconds")
    
    # Prepare for Gephi
    graph_prepared = prepare_graph_for_gephi(graph)
    
    # Analyze structure
    analyze_graph_structure(graph_prepared)
    
    # Export to GEXF
    export_to_gexf(graph_prepared, output_file)
    
    # Final instructions
    print("\n" + "="*80)
    print("EXPORT COMPLETE - GEPHI INSTRUCTIONS")
    print("="*80)
    print("\n1. Open Gephi")
    print("2. File -> Open -> Select 'reddit_binary_5pct_gephi.gexf'")
    print("3. In Overview tab:")
    print("   - Run 'Force Atlas 2' layout algorithm")
    print("   - Color nodes by 'binary_category' attribute")
    print("   - Size nodes by degree")
    print("4. In Data Laboratory tab:")
    print("   - View node attributes (binary_label, category, etc.)")
    print("5. In Preview tab:")
    print("   - Adjust visual settings and export final visualization")
    print("\nNode colors in file:")
    print("  - Blue nodes: Serious subreddits")
    print("  - Red nodes: Entertainment subreddits")
    print("  - Gray nodes: Unknown category")
    print("="*80)

if __name__ == "__main__":
    main()