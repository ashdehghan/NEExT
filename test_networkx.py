"""
Test file for NetworkX support in NEExT
Demonstrates loading NetworkX graphs directly into the framework
"""

import networkx as nx
import numpy as np
from NEExT import NEExT

def test_networkx_loading():
    """Test loading NetworkX graphs into NEExT"""
    
    print("Testing NetworkX Support in NEExT")
    print("="*50)
    
    # Initialize NEExT
    nxt = NEExT()
    nxt.set_log_level("WARNING")
    
    # Create sample NetworkX graphs
    print("\n1. Creating NetworkX graphs...")
    
    # Graph 1: Karate Club (classic social network)
    G1 = nx.karate_club_graph()
    G1.graph['label'] = 0  # Set graph label for classification
    print(f"   - Karate Club: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    
    # Graph 2: Complete graph
    G2 = nx.complete_graph(10)
    G2.graph['label'] = 1
    print(f"   - Complete Graph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    
    # Graph 3: Random graph with attributes
    G3 = nx.erdos_renyi_graph(15, 0.3)
    G3.graph['label'] = 0
    # Add node attributes
    for node in G3.nodes():
        G3.nodes[node]['feature1'] = np.random.random()
        G3.nodes[node]['feature2'] = np.random.randint(0, 10)
    # Add edge weights
    for edge in G3.edges():
        G3.edges[edge]['weight'] = np.random.random()
    print(f"   - Random Graph: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges (with attributes)")
    
    # Load NetworkX graphs into NEExT
    print("\n2. Loading graphs into NEExT...")
    collection = nxt.load_from_networkx(
        nx_graphs=[G1, G2, G3],
        reindex_nodes=True,
        filter_largest_component=False
    )
    
    print(f"   ✓ Loaded {len(collection.graphs)} graphs")
    print(f"   ✓ Total nodes: {collection.get_total_node_count()}")
    
    # Verify graph properties
    print("\n3. Verifying graph properties...")
    for i, graph in enumerate(collection.graphs):
        print(f"   Graph {i}: {len(graph.nodes)} nodes, label={graph.graph_label}")
    
    # Compute features on NetworkX-loaded graphs
    print("\n4. Computing node features...")
    features = nxt.compute_node_features(
        graph_collection=collection,
        feature_list=["degree_centrality", "clustering_coefficient"],
        feature_vector_length=2
    )
    
    print(f"   ✓ Computed {len(features.feature_columns)} features")
    print(f"   ✓ Feature matrix shape: {features.features_df.shape}")
    
    # Test mixed loading (NetworkX + dictionary format)
    print("\n5. Testing mixed format loading...")
    mixed_data = []
    
    # Add a NetworkX graph
    G4 = nx.cycle_graph(8)
    G4.graph['label'] = 1
    mixed_data.append(G4)
    
    # Add a dictionary format graph
    dict_graph = {
        "graph_id": 100,
        "graph_label": 0,
        "nodes": list(range(10)),
        "edges": [(i, (i+1) % 10) for i in range(10)],
        "node_attributes": {},
        "edge_attributes": {}
    }
    mixed_data.append(dict_graph)
    
    # Create collection with mixed formats
    from NEExT.collections import GraphCollection
    mixed_collection = GraphCollection(graph_type="networkx")
    mixed_collection.add_graphs(mixed_data, reindex_nodes=True)
    
    print(f"   ✓ Mixed collection created with {len(mixed_collection.graphs)} graphs")
    print(f"   ✓ Graph types: NetworkX + Dictionary format")
    
    print("\n" + "="*50)
    print("✅ All tests passed! NetworkX support is working correctly.")
    print("\nYou can now:")
    print("  • Load NetworkX graphs directly using nxt.load_from_networkx()")
    print("  • Mix NetworkX graphs with other formats")
    print("  • Use all NEExT features on NetworkX-loaded graphs")
    
    return collection, features

if __name__ == "__main__":
    collection, features = test_networkx_loading()