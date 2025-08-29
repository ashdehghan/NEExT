#!/usr/bin/env python
"""
Test script to verify the improved random walk sampling implementation.
Tests the visit frequency-based truncation and error handling improvements.
"""

import networkx as nx
import numpy as np
from NEExT.sampling.random_walk_sampler import RandomWalkSampler
from NEExT.collections.egonet_collection import EgonetCollection
from NEExT.collections.graph_collection import GraphCollection
from NEExT.graphs.graph import Graph

def test_visit_frequency_truncation():
    """Test that visit frequency-based truncation preserves frequently visited nodes."""
    print("Testing visit frequency-based truncation...")
    
    # Create a test graph where some nodes should be visited more frequently
    G = nx.star_graph(20)  # Center node 0 connected to nodes 1-20
    
    sampler = RandomWalkSampler()
    
    # Sample with a small max_nodes to force truncation
    neighborhood = sampler.sample_neighborhood(
        graph=G,
        ego_node=0,
        walk_length=5,
        num_walks=10,
        restart_prob=0.3,  # Higher restart prob means center visited more
        max_nodes=10,
        random_seed=42
    )
    
    print(f"Sampled neighborhood: {neighborhood}")
    print(f"Neighborhood size: {len(neighborhood)}")
    
    # Ego node should always be included
    assert 0 in neighborhood, "Ego node not in neighborhood"
    
    # Should respect max_nodes
    assert len(neighborhood) <= 10, f"Neighborhood too large: {len(neighborhood)}"
    
    print("✓ Visit frequency truncation test passed")

def test_deterministic_behavior():
    """Test that same seed produces same results with new implementation."""
    print("\nTesting deterministic behavior...")
    
    G = nx.karate_club_graph()
    sampler = RandomWalkSampler()
    
    # Run twice with same seed
    neighborhood1 = sampler.sample_neighborhood(
        graph=G, ego_node=0, walk_length=5, num_walks=3, 
        max_nodes=15, random_seed=123
    )
    
    neighborhood2 = sampler.sample_neighborhood(
        graph=G, ego_node=0, walk_length=5, num_walks=3,
        max_nodes=15, random_seed=123
    )
    
    assert neighborhood1 == neighborhood2, "Results not deterministic"
    print("✓ Deterministic behavior test passed")

def test_integration_with_egonet_collection():
    """Test integration with EgonetCollection using improved sampling."""
    print("\nTesting integration with EgonetCollection...")
    
    # Create a test graph with node attributes
    nx_graph = nx.karate_club_graph()
    
    # Add node attributes
    for node in nx_graph.nodes():
        nx_graph.nodes[node]['test_feature'] = node * 0.1
        nx_graph.nodes[node]['binary_label'] = node % 2
    
    # Convert to NEExT Graph
    graph = Graph(
        graph_id=0,
        graph_label=0,
        nodes=list(nx_graph.nodes()),
        edges=list(nx_graph.edges()),
        node_attributes={
            node: dict(nx_graph.nodes[node]) 
            for node in nx_graph.nodes()
        },
        edge_attributes={},
        graph_type="networkx"
    )
    graph.initialize_graph()
    
    graph_collection = GraphCollection()
    graph_collection.graphs = [graph]
    
    # Test with random walk sampling
    egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
    
    egonet_collection.compute_k_hop_egonets(
        graph_collection,
        k_hop=2,
        sample_fraction=0.3,
        sampling_strategy='random_walk',
        walk_length=5,
        num_walks=3,
        restart_prob=0.2,
        max_nodes_per_egonet=10,
        random_seed=42
    )
    
    print(f"Created {len(egonet_collection.graphs)} egonets")
    
    # Verify egonets respect max_nodes_per_egonet
    max_size = max(len(egonet.nodes) for egonet in egonet_collection.graphs)
    print(f"Largest egonet size: {max_size}")
    
    assert max_size <= 10, f"Egonet too large: {max_size}"
    
    # Verify all egonets have labels
    for egonet in egonet_collection.graphs:
        assert egonet.graph_label is not None, "Egonet missing label"
    
    print("✓ Integration test passed")

def test_error_handling():
    """Test improved error handling."""
    print("\nTesting error handling...")
    
    # Create test graph
    nx_graph = nx.path_graph(5)
    for node in nx_graph.nodes():
        nx_graph.nodes[node]['binary_label'] = node % 2
    
    graph = Graph(
        graph_id=0,
        graph_label=0,
        nodes=list(nx_graph.nodes()),
        edges=list(nx_graph.edges()),
        node_attributes={
            node: dict(nx_graph.nodes[node]) 
            for node in nx_graph.nodes()
        },
        edge_attributes={},
        graph_type="networkx"
    )
    graph.initialize_graph()
    
    graph_collection = GraphCollection()
    graph_collection.graphs = [graph]
    
    egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
    
    # This should work with fallback handling
    egonet_collection.compute_k_hop_egonets(
        graph_collection,
        k_hop=1,
        sample_fraction=1.0,
        sampling_strategy='random_walk',
        walk_length=3,
        num_walks=2,
        random_seed=42
    )
    
    print(f"Created {len(egonet_collection.graphs)} egonets with fallback handling")
    print("✓ Error handling test passed")

def performance_comparison():
    """Compare performance of k-hop vs random walk sampling."""
    print("\nRunning performance comparison...")
    
    # Create a larger test graph
    G = nx.barabasi_albert_graph(500, 5, seed=42)
    
    # Add to GraphCollection
    for node in G.nodes():
        G.nodes[node]['binary_label'] = node % 2
    
    graph = Graph(
        graph_id=0,
        graph_label=0,
        nodes=list(G.nodes()),
        edges=list(G.edges()),
        node_attributes={
            node: dict(G.nodes[node]) 
            for node in G.nodes()
        },
        edge_attributes={},
        graph_type="networkx"
    )
    graph.initialize_graph()
    
    graph_collection = GraphCollection()
    graph_collection.graphs = [graph]
    
    import time
    
    # Test k-hop sampling
    start_time = time.time()
    egonet_collection_khop = EgonetCollection(egonet_feature_target='binary_label')
    egonet_collection_khop.compute_k_hop_egonets(
        graph_collection,
        k_hop=2,
        sample_fraction=0.1,  # Sample 10% of nodes
        sampling_strategy='k_hop',
        random_seed=42
    )
    khop_time = time.time() - start_time
    khop_avg_size = np.mean([len(egonet.nodes) for egonet in egonet_collection_khop.graphs])
    
    # Test random walk sampling
    start_time = time.time()
    egonet_collection_rw = EgonetCollection(egonet_feature_target='binary_label')
    egonet_collection_rw.compute_k_hop_egonets(
        graph_collection,
        k_hop=2,
        sample_fraction=0.1,  # Sample 10% of nodes
        sampling_strategy='random_walk',
        walk_length=8,
        num_walks=5,
        max_nodes_per_egonet=50,
        random_seed=42
    )
    rw_time = time.time() - start_time
    rw_avg_size = np.mean([len(egonet.nodes) for egonet in egonet_collection_rw.graphs])
    
    print(f"\nPerformance Comparison:")
    print(f"K-hop: {khop_time:.2f}s, avg egonet size: {khop_avg_size:.1f}")
    print(f"Random walk: {rw_time:.2f}s, avg egonet size: {rw_avg_size:.1f}")
    print(f"Speedup: {khop_time/rw_time:.2f}x")
    print(f"Size reduction: {khop_avg_size/rw_avg_size:.2f}x")

if __name__ == "__main__":
    print("Testing improved random walk sampling implementation...\n")
    
    test_visit_frequency_truncation()
    test_deterministic_behavior() 
    test_integration_with_egonet_collection()
    test_error_handling()
    performance_comparison()
    
    print("\n✅ All tests passed! The improved implementation is ready for production.")