#!/usr/bin/env python3
"""
Create a 2% sample of the Reddit binary graph for Gephi visualization.

This script creates a smaller sample that should be more manageable for visualization.
"""

import pickle
import networkx as nx
import random
import time
from pathlib import Path

def create_2pct_sample():
    """Create a 2% sample of the Reddit binary graph."""
    
    print("\n" + "="*60)
    print("CREATING 2% SAMPLE FOR GEPHI")
    print("="*60)
    
    # Load the 5% graph
    print("\n[1/4] Loading 5% Reddit binary graph...")
    start_time = time.time()
    
    with open('reddit_binary_5pct.pkl', 'rb') as f:
        graph_5pct = pickle.load(f)
    
    print(f"      Loaded in {time.time() - start_time:.1f}s")
    print(f"      Original (5%): {graph_5pct.number_of_nodes():,} nodes, {graph_5pct.number_of_edges():,} edges")
    
    # Calculate target size (2% means 40% of the 5% graph)
    sample_rate = 0.4  # 40% of 5% = 2% of original
    target_nodes = int(graph_5pct.number_of_nodes() * sample_rate)
    
    print(f"\n[2/4] Creating 2% sample...")
    print(f"      Target: ~{target_nodes:,} nodes (2% of original dataset)")
    start_time = time.time()
    
    # Get balanced sample of nodes
    serious_nodes = [n for n, d in graph_5pct.nodes(data=True) if d.get('binary_label') == 0]
    entertainment_nodes = [n for n, d in graph_5pct.nodes(data=True) if d.get('binary_label') == 1]
    
    # Sample equally from both classes
    nodes_per_class = target_nodes // 2
    sampled_serious = random.sample(serious_nodes, min(nodes_per_class, len(serious_nodes)))
    sampled_entertainment = random.sample(entertainment_nodes, min(nodes_per_class, len(entertainment_nodes)))
    
    # Combine sampled nodes
    sampled_nodes = sampled_serious + sampled_entertainment
    
    # Create subgraph
    graph_2pct = graph_5pct.subgraph(sampled_nodes).copy()
    
    # Remove isolated nodes
    isolated = list(nx.isolates(graph_2pct))
    graph_2pct.remove_nodes_from(isolated)
    
    print(f"      Created in {time.time() - start_time:.1f}s")
    print(f"      Final (2%): {graph_2pct.number_of_nodes():,} nodes, {graph_2pct.number_of_edges():,} edges")
    print(f"      Removed {len(isolated)} isolated nodes")
    
    # Check class balance
    serious_count = sum(1 for _, d in graph_2pct.nodes(data=True) if d.get('binary_label') == 0)
    entertainment_count = sum(1 for _, d in graph_2pct.nodes(data=True) if d.get('binary_label') == 1)
    print(f"      Class balance: Serious={serious_count:,}, Entertainment={entertainment_count:,}")
    
    # Save the 2% sample
    print("\n[3/4] Saving 2% sample...")
    start_time = time.time()
    
    with open('reddit_binary_2pct.pkl', 'wb') as f:
        pickle.dump(graph_2pct, f)
    
    file_size = Path('reddit_binary_2pct.pkl').stat().st_size / (1024 * 1024)
    print(f"      Saved reddit_binary_2pct.pkl ({file_size:.1f} MB)")
    
    # Export to GEXF for Gephi
    print("\n[4/4] Exporting to GEXF format for Gephi...")
    start_time = time.time()
    
    # Clean attributes for Gephi
    G_export = graph_2pct.copy()
    
    # Add graph metadata
    G_export.graph['name'] = 'Reddit Binary Classification Network (2% Sample)'
    G_export.graph['description'] = 'Reddit 2% sample - Serious vs Entertainment subreddits'
    
    # Clean node attributes (remove feature columns for smaller file)
    for node in G_export.nodes():
        attrs = G_export.nodes[node]
        
        # Keep only essential attributes
        attrs_to_keep = {
            'binary_label': attrs.get('binary_label', -1),
            'binary_category': attrs.get('binary_category', 'unknown'),
            'original_subreddit': attrs.get('original_subreddit', -1),
            'split': attrs.get('split', 'unknown')
        }
        
        # Clear all attributes
        attrs.clear()
        attrs.update(attrs_to_keep)
        
        # Add color for visualization
        if attrs['binary_label'] == 0:
            attrs['viz:color'] = {'r': 52, 'g': 152, 'b': 219, 'a': 1.0}  # Blue
            attrs['viz_label'] = 'Serious'
        elif attrs['binary_label'] == 1:
            attrs['viz:color'] = {'r': 231, 'g': 76, 'b': 60, 'a': 1.0}  # Red
            attrs['viz_label'] = 'Entertainment'
        else:
            attrs['viz:color'] = {'r': 149, 'g': 165, 'b': 166, 'a': 1.0}  # Gray
            attrs['viz_label'] = 'Unknown'
    
    # Add edge weights
    for u, v in G_export.edges():
        G_export[u][v]['weight'] = 1.0
    
    # Write GEXF file
    output_path = 'reddit_binary_2pct_gephi.gexf'
    nx.write_gexf(G_export, output_path)
    
    gexf_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"      Saved {output_path} ({gexf_size:.1f} MB)")
    print(f"      Export complete in {time.time() - start_time:.1f}s")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Created 2% sample with {graph_2pct.number_of_nodes():,} nodes")
    print(f"Graph density: {nx.density(graph_2pct):.6f}")
    
    # Component analysis
    num_components = nx.number_connected_components(graph_2pct)
    largest_cc = max(nx.connected_components(graph_2pct), key=len)
    print(f"Connected components: {num_components}")
    print(f"Largest component: {len(largest_cc):,} nodes ({len(largest_cc)/graph_2pct.number_of_nodes()*100:.1f}%)")
    
    print("\n" + "="*60)
    print("FILES CREATED")
    print("="*60)
    print("1. reddit_binary_2pct.pkl - NetworkX graph (2% sample)")
    print("2. reddit_binary_2pct_gephi.gexf - GEXF file for Gephi")
    print("\nTo visualize in Gephi:")
    print("1. Open Gephi")
    print("2. File -> Open -> Select 'reddit_binary_2pct_gephi.gexf'")
    print("3. Run Force Atlas 2 layout")
    print("4. Color by 'binary_category' attribute")
    print("="*60)
    
    return graph_2pct

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    graph = create_2pct_sample()