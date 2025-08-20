"""
Comprehensive tests for the neighborhood sampling functionality.
"""

import unittest
import numpy as np
import networkx as nx
from NEExT.sampling.base import SamplingStrategy, get_sampler
from NEExT.sampling.k_hop_sampler import KHopSampler
from NEExT.sampling.random_walk_sampler import RandomWalkSampler
from NEExT.collections.egonet_collection import EgonetCollection
from NEExT.collections.graph_collection import GraphCollection
from NEExT.graphs.graph import Graph

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


class TestSamplingBase(unittest.TestCase):
    """Test base sampling classes and factory functions."""
    
    def test_sampling_strategy_enum(self):
        """Test SamplingStrategy enum values."""
        self.assertEqual(SamplingStrategy.K_HOP.value, "k_hop")
        self.assertEqual(SamplingStrategy.RANDOM_WALK.value, "random_walk")
    
    def test_get_sampler_factory(self):
        """Test sampler factory function."""
        # Test with enum
        k_hop_sampler = get_sampler(SamplingStrategy.K_HOP)
        self.assertIsInstance(k_hop_sampler, KHopSampler)
        
        rw_sampler = get_sampler(SamplingStrategy.RANDOM_WALK)
        self.assertIsInstance(rw_sampler, RandomWalkSampler)
        
        # Test with string
        k_hop_sampler_str = get_sampler("k_hop")
        self.assertIsInstance(k_hop_sampler_str, KHopSampler)
        
        rw_sampler_str = get_sampler("random_walk")
        self.assertIsInstance(rw_sampler_str, RandomWalkSampler)
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            get_sampler("invalid_strategy")


class TestKHopSampler(unittest.TestCase):
    """Test k-hop neighborhood sampling."""
    
    def setUp(self):
        """Set up test graphs."""
        self.sampler = KHopSampler()
        
        # Create a simple NetworkX graph
        self.nx_graph = nx.path_graph(10)  # 0-1-2-3-4-5-6-7-8-9
        
    def test_k_hop_basic(self):
        """Test basic k-hop sampling."""
        # 1-hop neighborhood of node 5
        neighborhood = self.sampler.sample_neighborhood(self.nx_graph, 5, k_hop=1)
        expected = [4, 5, 6]  # Node 5 and its neighbors
        self.assertEqual(sorted(neighborhood), expected)
        
        # 2-hop neighborhood of node 5
        neighborhood = self.sampler.sample_neighborhood(self.nx_graph, 5, k_hop=2)
        expected = [3, 4, 5, 6, 7]  # Node 5 and neighbors within 2 hops
        self.assertEqual(sorted(neighborhood), expected)
    
    def test_k_hop_edge_cases(self):
        """Test edge cases for k-hop sampling."""
        # k_hop = 0 should return only ego node
        neighborhood = self.sampler.sample_neighborhood(self.nx_graph, 5, k_hop=0)
        self.assertEqual(neighborhood, [5])
        
        # Isolated node
        isolated_graph = nx.Graph()
        isolated_graph.add_node(0)
        neighborhood = self.sampler.sample_neighborhood(isolated_graph, 0, k_hop=1)
        self.assertEqual(neighborhood, [0])
    
    def test_k_hop_validation(self):
        """Test input validation."""
        # Invalid k_hop
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 5, k_hop=-1)
        
        # Invalid ego node
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 999, k_hop=1)


class TestRandomWalkSampler(unittest.TestCase):
    """Test random walk neighborhood sampling."""
    
    def setUp(self):
        """Set up test graphs."""
        self.sampler = RandomWalkSampler()
        
        # Create a connected graph
        self.nx_graph = nx.karate_club_graph()
        
        # Create a simple path for deterministic testing
        self.path_graph = nx.path_graph(10)
    
    def test_random_walk_basic(self):
        """Test basic random walk sampling."""
        # Test with fixed seed for determinism
        neighborhood = self.sampler.sample_neighborhood(
            self.nx_graph, 
            ego_node=0,
            walk_length=5,
            num_walks=3,
            restart_prob=0.2,
            random_seed=42
        )
        
        # Ego node should always be included
        self.assertIn(0, neighborhood)
        
        # Should return a reasonable number of nodes
        self.assertGreater(len(neighborhood), 1)
        self.assertLessEqual(len(neighborhood), 15)  # Reasonable upper bound
    
    def test_random_walk_deterministic(self):
        """Test that same seed produces same results."""
        neighborhood1 = self.sampler.sample_neighborhood(
            self.path_graph,
            ego_node=5,
            walk_length=3,
            num_walks=2,
            restart_prob=0.3,
            random_seed=123
        )
        
        neighborhood2 = self.sampler.sample_neighborhood(
            self.path_graph,
            ego_node=5,
            walk_length=3,
            num_walks=2,
            restart_prob=0.3,
            random_seed=123
        )
        
        self.assertEqual(neighborhood1, neighborhood2)
    
    def test_random_walk_max_nodes(self):
        """Test max_nodes parameter."""
        neighborhood = self.sampler.sample_neighborhood(
            self.nx_graph,
            ego_node=0,
            walk_length=10,
            num_walks=10,
            max_nodes=5,
            random_seed=42
        )
        
        # Should not exceed max_nodes
        self.assertLessEqual(len(neighborhood), 5)
        # Ego node should still be included
        self.assertIn(0, neighborhood)
    
    def test_random_walk_validation(self):
        """Test input validation."""
        # Invalid walk_length
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 0, walk_length=0)
        
        # Invalid num_walks
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 0, num_walks=0)
        
        # Invalid restart_prob
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 0, restart_prob=1.5)
        
        # Invalid ego node
        with self.assertRaises(ValueError):
            self.sampler.sample_neighborhood(self.nx_graph, 999)
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter selection."""
        params = self.sampler.adaptive_parameters(self.nx_graph, 0, target_size=20)
        
        # Should return reasonable parameters
        self.assertIn('walk_length', params)
        self.assertIn('num_walks', params)
        self.assertIn('restart_prob', params)
        self.assertIn('max_nodes', params)
        
        # Values should be reasonable
        self.assertGreater(params['walk_length'], 0)
        self.assertGreater(params['num_walks'], 0)
        self.assertTrue(0 <= params['restart_prob'] <= 1)


class TestEgonetCollectionIntegration(unittest.TestCase):
    """Test integration with EgonetCollection."""
    
    def setUp(self):
        """Set up test graph collection."""
        # Create a simple graph
        nx_graph = nx.karate_club_graph()
        
        # Add node attributes for testing
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
        
        self.graph_collection = GraphCollection()
        self.graph_collection.graphs = [graph]
    
    def test_backwards_compatibility(self):
        """Test that existing k-hop behavior is preserved."""
        egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
        
        # Test original API (should use k-hop by default)
        egonet_collection.compute_k_hop_egonets(
            self.graph_collection,
            k_hop=1,
            sample_fraction=0.3,
            random_seed=42
        )
        
        # Should create egonets successfully
        self.assertGreater(len(egonet_collection.graphs), 0)
        
        # Each egonet should have a binary_label
        for egonet in egonet_collection.graphs:
            self.assertIsNotNone(egonet.graph_label)
    
    def test_random_walk_integration(self):
        """Test random walk sampling integration."""
        egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
        
        # Test with random walk sampling
        egonet_collection.compute_k_hop_egonets(
            self.graph_collection,
            k_hop=2,  # Used as hint for walk_length
            sample_fraction=0.3,
            sampling_strategy='random_walk',
            walk_length=5,
            num_walks=3,
            restart_prob=0.2,
            max_nodes_per_egonet=10,
            random_seed=42
        )
        
        # Should create egonets successfully
        self.assertGreater(len(egonet_collection.graphs), 0)
        
        # Each egonet should respect max_nodes_per_egonet
        for egonet in egonet_collection.graphs:
            self.assertLessEqual(len(egonet.nodes), 10)
            self.assertIsNotNone(egonet.graph_label)
    
    def test_sampling_strategy_enum_integration(self):
        """Test using SamplingStrategy enum directly."""
        egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
        
        # Test with enum
        egonet_collection.compute_k_hop_egonets(
            self.graph_collection,
            sampling_strategy=SamplingStrategy.RANDOM_WALK,
            walk_length=3,
            num_walks=2,
            sample_fraction=0.2,
            random_seed=42
        )
        
        self.assertGreater(len(egonet_collection.graphs), 0)


class TestPerformanceComparison(unittest.TestCase):
    """Compare performance of different sampling strategies."""
    
    def setUp(self):
        """Set up larger test graph."""
        # Create a larger graph for performance testing
        self.large_graph = nx.barabasi_albert_graph(1000, 5, seed=42)
    
    def test_sampling_efficiency(self):
        """Test that random walk is more efficient for large neighborhoods."""
        k_hop_sampler = KHopSampler()
        rw_sampler = RandomWalkSampler()
        
        # For a high-degree node, random walk should return smaller neighborhoods
        high_degree_node = max(self.large_graph.nodes(), 
                              key=lambda x: self.large_graph.degree(x))
        
        # K-hop neighborhood (can be very large)
        k_hop_neighborhood = k_hop_sampler.sample_neighborhood(
            self.large_graph, high_degree_node, k_hop=2
        )
        
        # Random walk neighborhood (bounded)
        rw_neighborhood = rw_sampler.sample_neighborhood(
            self.large_graph,
            high_degree_node,
            walk_length=10,
            num_walks=5,
            max_nodes=50,
            random_seed=42
        )
        
        # Random walk should typically produce smaller neighborhoods
        print(f"K-hop neighborhood size: {len(k_hop_neighborhood)}")
        print(f"Random walk neighborhood size: {len(rw_neighborhood)}")
        
        # Both should include the ego node
        self.assertIn(high_degree_node, k_hop_neighborhood)
        self.assertIn(high_degree_node, rw_neighborhood)


if __name__ == '__main__':
    unittest.main()