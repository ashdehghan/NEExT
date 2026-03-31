"""Tests for the NEExT synthetic graph generation module."""

import networkx as nx
import numpy as np
import pytest

from NEExT.generators import (
    AnomalyConfig,
    AnomalyInjector,
    AttributeConfig,
    AttributeGenerator,
    GeneratorSpec,
    GraphAdapter,
    GraphBuilder,
    GraphGenerator,
    SyntheticPresets,
)


# ═══════════════════════════════════════════════════════════════════════
# Config models
# ═══════════════════════════════════════════════════════════════════════


class TestGeneratorSpec:
    def test_basic_spec(self):
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"n": 50, "p": 0.1}, label=0)
        assert spec.generator_type == "erdos_renyi"
        assert spec.label == 0
        assert spec.n_range is None

    def test_spec_with_n_range(self):
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"p": 0.1}, label=1, n_range=(10, 50))
        assert spec.n_range == (10, 50)

    def test_invalid_n_range(self):
        with pytest.raises(ValueError):
            GeneratorSpec(generator_type="erdos_renyi", n_range=(50, 10))

    def test_invalid_n_range_zero(self):
        with pytest.raises(ValueError):
            GeneratorSpec(generator_type="erdos_renyi", n_range=(0, 10))


class TestAttributeConfig:
    def test_default(self):
        cfg = AttributeConfig()
        assert cfg.strategy == "random_normal"
        assert cfg.n_features == 5

    def test_custom_requires_fn(self):
        with pytest.raises(ValueError):
            AttributeConfig(strategy="custom", custom_fn=None)

    def test_custom_with_fn(self):
        cfg = AttributeConfig(strategy="custom", custom_fn=lambda G, rng: G)
        assert cfg.custom_fn is not None


class TestAnomalyConfig:
    def test_default(self):
        cfg = AnomalyConfig()
        assert cfg.anomaly_type == "structural_hub"
        assert 0 < cfg.fraction < 1

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            AnomalyConfig(fraction=0.0)
        with pytest.raises(ValueError):
            AnomalyConfig(fraction=1.0)


# ═══════════════════════════════════════════════════════════════════════
# Adapters
# ═══════════════════════════════════════════════════════════════════════


class TestGraphAdapter:
    def test_ensure_networkx_passthrough(self):
        G = nx.complete_graph(5)
        result = GraphAdapter.ensure_networkx(G)
        assert isinstance(result, nx.Graph)
        assert result is G

    def test_from_adjacency_matrix(self):
        adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        G = GraphAdapter.from_adjacency_matrix(adj)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_from_edge_list(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        G = GraphAdapter.from_edge_list(edges)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3

    def test_from_edge_list_with_attrs(self):
        edges = [(0, 1), (1, 2)]
        attrs = {0: {"color": "red"}, 1: {"color": "blue"}}
        G = GraphAdapter.from_edge_list(edges, node_attrs=attrs)
        assert G.nodes[0]["color"] == "red"

    def test_from_adjacency_matrix_non_square(self):
        with pytest.raises(ValueError):
            GraphAdapter.from_adjacency_matrix(np.zeros((3, 4)))

    def test_ensure_networkx_unsupported_type(self):
        with pytest.raises(TypeError):
            GraphAdapter.ensure_networkx("not a graph")

    def test_from_igraph(self):
        try:
            import igraph as ig
        except ImportError:
            pytest.skip("igraph not installed")
        ig_g = ig.Graph.Famous("Petersen")
        ig_g.vs["name"] = [f"v{i}" for i in range(ig_g.vcount())]
        G = GraphAdapter.from_igraph(ig_g)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() == 15
        assert G.nodes[0]["name"] == "v0"


# ═══════════════════════════════════════════════════════════════════════
# GraphGenerator
# ═══════════════════════════════════════════════════════════════════════


class TestGraphGenerator:
    def test_available_generators(self):
        gen = GraphGenerator(seed=42)
        available = gen.available_generators
        assert "erdos_renyi" in available
        assert "barabasi_albert" in available
        assert len(available) >= 16

    @pytest.mark.parametrize(
        "gen_type,params",
        [
            ("erdos_renyi", {"n": 30, "p": 0.15}),
            ("barabasi_albert", {"n": 30, "m": 2}),
            ("watts_strogatz", {"n": 30, "k": 4, "p": 0.1}),
            ("stochastic_block_model", {"n_communities": 2, "community_size": 15, "p_in": 0.3, "p_out": 0.05}),
            ("planted_partition", {"l": 3, "k": 10, "p_in": 0.3, "p_out": 0.05}),
            ("random_geometric", {"n": 30, "radius": 0.3}),
            ("powerlaw_cluster", {"n": 30, "m": 2, "p": 0.1}),
            ("configuration_model", {"n": 30}),
            ("caveman", {"l": 3, "k": 5}),
            ("connected_caveman", {"l": 3, "k": 5}),
            ("cycle", {"n": 20}),
            ("star", {"n": 10}),
            ("complete", {"n": 10}),
            ("tree", {"n": 30}),
            ("grid_2d", {"m": 4, "n": 5}),
        ],
    )
    def test_generate_one(self, gen_type, params):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type=gen_type, params=params, label=7)
        G = gen.generate_one(spec)
        assert isinstance(G, nx.Graph)
        assert G.graph["label"] == 7
        assert G.number_of_nodes() > 0
        # Verify all nodes are integers (not tuples)
        for node in G.nodes():
            assert isinstance(node, (int, np.integer))

    def test_generate_many(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"n": 20, "p": 0.1}, label=1)
        graphs = gen.generate_many(spec, 10)
        assert len(graphs) == 10
        assert all(isinstance(G, nx.Graph) for G in graphs)
        assert all(G.graph["label"] == 1 for G in graphs)

    def test_generate_collection_mixed(self):
        gen = GraphGenerator(seed=42)
        specs = [
            (GeneratorSpec(generator_type="erdos_renyi", params={"n": 20, "p": 0.1}, label=0), 5),
            (GeneratorSpec(generator_type="barabasi_albert", params={"n": 20, "m": 2}, label=1), 5),
        ]
        graphs = gen.generate_collection(specs, shuffle=True)
        assert len(graphs) == 10
        labels = [G.graph["label"] for G in graphs]
        assert labels.count(0) == 5
        assert labels.count(1) == 5

    def test_generate_collection_no_shuffle(self):
        gen = GraphGenerator(seed=42)
        specs = [
            (GeneratorSpec(generator_type="erdos_renyi", params={"n": 20, "p": 0.1}, label=0), 3),
            (GeneratorSpec(generator_type="barabasi_albert", params={"n": 20, "m": 2}, label=1), 3),
        ]
        graphs = gen.generate_collection(specs, shuffle=False)
        labels = [G.graph["label"] for G in graphs]
        assert labels == [0, 0, 0, 1, 1, 1]

    def test_variable_size_via_n_range(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"p": 0.15}, label=0, n_range=(10, 100))
        graphs = gen.generate_many(spec, 20)
        sizes = [G.number_of_nodes() for G in graphs]
        assert min(sizes) >= 10
        assert max(sizes) <= 100
        # Should have some variation
        assert len(set(sizes)) > 1

    def test_deterministic_seeding(self):
        gen1 = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"n": 30, "p": 0.2}, label=0)
        g1 = gen1.generate_one(spec)

        gen2 = GraphGenerator(seed=42)
        g2 = gen2.generate_one(spec)

        assert list(g1.edges()) == list(g2.edges())
        assert list(g1.nodes()) == list(g2.nodes())

    def test_unknown_generator_type(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type="nonexistent_type", params={}, label=0)
        with pytest.raises(ValueError, match="Unknown generator type"):
            gen.generate_one(spec)

    def test_custom_generator(self):
        gen = GraphGenerator(seed=42)

        def my_generator(params):
            n = params.get("n", 10)
            return nx.path_graph(n)

        gen.register_generator("my_path", my_generator)
        assert "my_path" in gen.available_generators

        spec = GeneratorSpec(generator_type="my_path", params={"n": 15}, label=3)
        G = gen.generate_one(spec)
        assert isinstance(G, nx.Graph)
        assert G.graph["label"] == 3
        assert G.number_of_nodes() == 15

    def test_custom_generator_type_in_spec(self):
        gen = GraphGenerator(seed=42)

        def my_wheel(params):
            return nx.wheel_graph(params.get("n", 8))

        spec = GeneratorSpec(generator_type="custom", params={"fn": my_wheel, "n": 8}, label=5)
        G = gen.generate_one(spec)
        assert G.number_of_nodes() == 8
        assert G.graph["label"] == 5

    def test_register_igraph_generator(self):
        try:
            import igraph as ig
        except ImportError:
            pytest.skip("igraph not installed")

        gen = GraphGenerator(seed=42)

        def ig_forest(params):
            n = params.get("n", 20)
            return ig.Graph.Tree(n, 2)

        gen.register_generator("ig_tree", ig_forest, output_format="igraph")
        spec = GeneratorSpec(generator_type="ig_tree", params={"n": 15}, label=0)
        G = gen.generate_one(spec)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 15

    def test_sbm_with_explicit_p_matrix(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(
            generator_type="stochastic_block_model",
            params={"sizes": [10, 10], "p_matrix": [[0.5, 0.05], [0.05, 0.5]]},
            label=0,
        )
        G = gen.generate_one(spec)
        assert G.number_of_nodes() == 20

    def test_configuration_model_explicit_degree_seq(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(
            generator_type="configuration_model",
            params={"degree_sequence": [2, 2, 2, 2, 2, 2]},
            label=0,
        )
        G = gen.generate_one(spec)
        assert isinstance(G, nx.Graph)
        # Should be a simple graph (no self-loops)
        assert len(list(nx.selfloop_edges(G))) == 0

    def test_to_graph_collection(self):
        gen = GraphGenerator(seed=42)
        spec = GeneratorSpec(generator_type="erdos_renyi", params={"n": 20, "p": 0.15}, label=0)
        graphs = gen.generate_many(spec, 5)
        collection = gen.to_graph_collection(graphs)
        assert len(collection.graphs) == 5


# ═══════════════════════════════════════════════════════════════════════
# GraphBuilder
# ═══════════════════════════════════════════════════════════════════════


class TestGraphBuilder:
    def test_start_from_and_build(self):
        b = GraphBuilder(seed=42)
        G = b.start_from("erdos_renyi", n=30, p=0.1).set_label(1).build()
        assert isinstance(G, nx.Graph)
        assert G.graph["label"] == 1
        assert G.number_of_nodes() == 30

    def test_empty_and_plant_community(self):
        b = GraphBuilder(seed=42)
        G = b.empty(10).plant_community(size=5, p_internal=0.9, p_bridge=0.3).set_label(0).build()
        assert G.number_of_nodes() == 15
        assert G.number_of_edges() > 0

    def test_from_graph(self):
        original = nx.cycle_graph(10)
        b = GraphBuilder(seed=42)
        G = b.from_graph(original).set_label(2).build()
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() == 10
        assert G.graph["label"] == 2
        # Original should be unmodified
        assert "label" not in original.graph

    def test_attach_motif(self):
        b = GraphBuilder(seed=42)
        G = b.start_from("erdos_renyi", n=20, p=0.1).attach_motif("clique", size=5, n_bridges=2).set_label(0).build()
        assert G.number_of_nodes() == 25

    def test_attach_motif_types(self):
        for motif in ["clique", "cycle", "star", "path", "tree"]:
            b = GraphBuilder(seed=42)
            G = b.empty(10).attach_motif(motif, size=5, n_bridges=1).build()
            assert G.number_of_nodes() == 15

    def test_invalid_motif(self):
        b = GraphBuilder(seed=42)
        with pytest.raises(ValueError, match="Unknown motif_type"):
            b.empty(10).attach_motif("hexagon", size=5)

    def test_add_hub(self):
        b = GraphBuilder(seed=42)
        G = b.start_from("erdos_renyi", n=20, p=0.1).add_hub(degree=10).build()
        assert G.number_of_nodes() == 21
        hub_node = 20
        assert G.degree(hub_node) <= 10  # May be less if duplicates in random choice

    def test_bridge_subgraphs(self):
        b = GraphBuilder(seed=42)
        G = b.empty(10).bridge_subgraphs(list(range(5)), list(range(5, 10)), n_bridges=3).build()
        assert G.number_of_edges() == 3

    def test_rewire_edges(self):
        b = GraphBuilder(seed=42)
        G = b.start_from("cycle", n=20).rewire_edges(fraction=0.3).build()
        assert G.number_of_nodes() == 20

    def test_merge_graph(self):
        b = GraphBuilder(seed=42)
        other = nx.complete_graph(5)
        G = b.start_from("erdos_renyi", n=10, p=0.2).merge_graph(other, bridge_edges=2).build()
        assert G.number_of_nodes() == 15

    def test_remove_edges(self):
        b = GraphBuilder(seed=42)
        G = b.start_from("complete", n=10).remove_edges(fraction=0.5).build()
        original_edges = 10 * 9 // 2  # 45
        assert G.number_of_edges() < original_edges

    def test_set_node_attribute(self):
        b = GraphBuilder(seed=42)
        G = b.empty(5).set_node_attribute(0, "color", "red").build()
        assert G.nodes[0]["color"] == "red"

    def test_set_node_attribute_invalid_node(self):
        b = GraphBuilder(seed=42)
        with pytest.raises(ValueError):
            b.empty(5).set_node_attribute(99, "color", "red")

    def test_chaining(self):
        b = GraphBuilder(seed=42)
        G = (
            b.start_from("erdos_renyi", n=30, p=0.1)
            .plant_community(size=10, p_internal=0.8, p_bridge=0.05)
            .attach_motif("clique", size=5, attach_to="random", n_bridges=2)
            .add_hub(degree=15)
            .set_label(1)
            .build()
        )
        assert isinstance(G, nx.Graph)
        assert G.graph["label"] == 1
        assert G.number_of_nodes() == 46  # 30 + 10 + 5 + 1

    def test_build_many(self):
        builder = GraphBuilder(seed=42)

        def recipe(b):
            return b.start_from("erdos_renyi", n=20, p=0.1).set_label(0).build()

        graphs = builder.build_many(recipe, count=10)
        assert len(graphs) == 10
        assert all(isinstance(G, nx.Graph) for G in graphs)

    def test_build_many_with_labels(self):
        builder = GraphBuilder(seed=42)

        def recipe(b):
            return b.start_from("erdos_renyi", n=20, p=0.1).build()

        labels = [0, 1, 0, 1, 0]
        graphs = builder.build_many(recipe, count=5, labels=labels)
        result_labels = [G.graph["label"] for G in graphs]
        assert result_labels == labels

    def test_build_without_init_raises(self):
        b = GraphBuilder(seed=42)
        with pytest.raises(RuntimeError, match="No graph initialized"):
            b.build()


# ═══════════════════════════════════════════════════════════════════════
# AttributeGenerator
# ═══════════════════════════════════════════════════════════════════════


class TestAttributeGenerator:
    def _make_graph(self):
        G = nx.barabasi_albert_graph(30, 2, seed=42)
        G.graph["label"] = 0
        return G

    def test_random_uniform(self):
        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        config = AttributeConfig(strategy="random_uniform", n_features=3)
        ag.add_node_attributes(G, config)
        for node in G.nodes():
            assert "feature_0" in G.nodes[node]
            assert "feature_1" in G.nodes[node]
            assert "feature_2" in G.nodes[node]
            assert 0 <= G.nodes[node]["feature_0"] <= 1

    def test_random_normal(self):
        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        config = AttributeConfig(strategy="random_normal", n_features=5)
        ag.add_node_attributes(G, config)
        assert "feature_4" in G.nodes[0]

    def test_community_correlated(self):
        ag = AttributeGenerator(seed=42)
        G = nx.stochastic_block_model([15, 15], [[0.3, 0.05], [0.05, 0.3]], seed=42)
        G.graph["label"] = 0
        config = AttributeConfig(strategy="community_correlated", n_features=3)
        ag.add_node_attributes(G, config)
        assert "feature_0" in G.nodes[0]

    def test_label_informative(self):
        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        config = AttributeConfig(strategy="label_informative", n_features=4)
        ag.add_node_attributes(G, config)
        assert "feature_0" in G.nodes[0]

    def test_degree_correlated(self):
        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        config = AttributeConfig(strategy="degree_correlated", n_features=3)
        ag.add_node_attributes(G, config)
        assert "feature_0" in G.nodes[0]

    def test_custom_strategy(self):
        def my_fn(G, rng):
            for n in G.nodes():
                G.nodes[n]["custom_feat"] = rng.random()
            return G

        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        config = AttributeConfig(strategy="custom", custom_fn=my_fn)
        ag.add_node_attributes(G, config)
        assert "custom_feat" in G.nodes[0]

    def test_add_edge_attributes(self):
        ag = AttributeGenerator(seed=42)
        G = self._make_graph()
        ag.add_edge_attributes(G, n_features=2, strategy="random_uniform")
        edge = list(G.edges())[0]
        assert "edge_feature_0" in G.edges[edge]
        assert "edge_feature_1" in G.edges[edge]

    def test_add_attributes_to_collection(self):
        ag = AttributeGenerator(seed=42)
        graphs = [nx.barabasi_albert_graph(20, 2, seed=i) for i in range(5)]
        for i, G in enumerate(graphs):
            G.graph["label"] = i % 2
        config = AttributeConfig(strategy="random_normal", n_features=3)
        ag.add_attributes_to_collection(graphs, config)
        for G in graphs:
            assert "feature_0" in G.nodes[0]


# ═══════════════════════════════════════════════════════════════════════
# AnomalyInjector
# ═══════════════════════════════════════════════════════════════════════


class TestAnomalyInjector:
    def _make_graph(self, n=100):
        G = nx.barabasi_albert_graph(n, 3, seed=42)
        G.graph["label"] = 0
        return G

    def test_structural_hub(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        config = AnomalyConfig(anomaly_type="structural_hub", fraction=0.1)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 10  # 10% of 100

    def test_structural_clique(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        config = AnomalyConfig(anomaly_type="structural_clique", fraction=0.05)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 5

    def test_structural_bridge(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        config = AnomalyConfig(anomaly_type="structural_bridge", fraction=0.05)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 5

    def test_structural_rewire(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        config = AnomalyConfig(anomaly_type="structural_rewire", fraction=0.05)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 5

    def test_contextual_feature(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        # Add features first
        ag = AttributeGenerator(seed=42)
        ag.add_node_attributes(G, AttributeConfig(strategy="random_normal", n_features=3))
        config = AnomalyConfig(anomaly_type="contextual_feature", fraction=0.05)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 5

    def test_contextual_mixed(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        ag = AttributeGenerator(seed=42)
        ag.add_node_attributes(G, AttributeConfig(strategy="random_normal", n_features=3))
        config = AnomalyConfig(anomaly_type="contextual_mixed", fraction=0.05)
        injector.inject(G, config)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) == 5

    def test_inject_multiple(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph()
        configs = [
            AnomalyConfig(anomaly_type="structural_hub", fraction=0.05),
            AnomalyConfig(anomaly_type="structural_clique", fraction=0.05),
        ]
        injector.inject_multiple(G, configs)
        outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
        assert len(outliers) > 0

    def test_all_nodes_labeled(self):
        injector = AnomalyInjector(seed=42)
        G = self._make_graph(50)
        config = AnomalyConfig(anomaly_type="structural_hub", fraction=0.1)
        injector.inject(G, config)
        for n in G.nodes():
            assert "is_outlier" in G.nodes[n]
            assert G.nodes[n]["is_outlier"] in (0, 1)


# ═══════════════════════════════════════════════════════════════════════
# SyntheticPresets
# ═══════════════════════════════════════════════════════════════════════


class TestSyntheticPresets:
    def test_binary_classification_er_vs_ba(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.binary_classification_er_vs_ba(n_per_class=10, n_nodes=30)
        assert len(graphs) == 20
        labels = [G.graph["label"] for G in graphs]
        assert labels.count(0) == 10
        assert labels.count(1) == 10

    def test_multiclass_topology(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.multiclass_topology(n_per_class=5, n_nodes=30)
        assert len(graphs) == 20
        labels = set(G.graph["label"] for G in graphs)
        assert labels == {0, 1, 2, 3}

    def test_community_strength_gradient(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.community_strength_gradient(n_per_level=5, n_nodes=30)
        assert len(graphs) == 25
        labels = set(G.graph["label"] for G in graphs)
        assert labels == {0, 1, 2, 3, 4}

    def test_scalability_test(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.scalability_test(sizes=[20, 50], n_per_size=3)
        assert len(graphs) == 6

    def test_variable_size_classification(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.variable_size_classification(n_per_class=10, class_a_range=(10, 30), class_b_range=(40, 60))
        assert len(graphs) == 20
        labels = [G.graph["label"] for G in graphs]
        assert labels.count(0) == 10
        assert labels.count(1) == 10

    def test_egonet_anomaly_detection(self):
        presets = SyntheticPresets(seed=42)
        graphs = presets.egonet_anomaly_detection(n_graphs=3, n_nodes=40, anomaly_fraction=0.1)
        assert len(graphs) == 3
        for G in graphs:
            outliers = [n for n in G.nodes() if G.nodes[n].get("is_outlier") == 1]
            assert len(outliers) > 0


# ═══════════════════════════════════════════════════════════════════════
# Integration: framework.generate_synthetic_graphs()
# ═══════════════════════════════════════════════════════════════════════


class TestFrameworkIntegration:
    def test_generate_via_preset(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        collection = nxt.generate_synthetic_graphs(preset="er_vs_ba", n_per_class=5, n_nodes=20)
        assert len(collection.graphs) == 10

    def test_generate_via_specs(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        specs = [
            (GeneratorSpec(generator_type="erdos_renyi", params={"n": 20, "p": 0.15}, label=0), 5),
            (GeneratorSpec(generator_type="barabasi_albert", params={"n": 20, "m": 2}, label=1), 5),
        ]
        collection = nxt.generate_synthetic_graphs(specs=specs, seed=42)
        assert len(collection.graphs) == 10

    def test_invalid_preset(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        with pytest.raises(ValueError, match="Unknown preset"):
            nxt.generate_synthetic_graphs(preset="nonexistent")

    def test_no_preset_or_specs_raises(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        with pytest.raises(ValueError, match="Either"):
            nxt.generate_synthetic_graphs()


# ═══════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_synthetic_to_features(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        presets = SyntheticPresets(seed=42)
        graphs = presets.binary_classification_er_vs_ba(n_per_class=10, n_nodes=30)
        collection = nxt.load_from_networkx(graphs)
        assert len(collection.graphs) == 20

        features = nxt.compute_node_features(collection, feature_list=["all"])
        assert len(features.features_df) > 0

    def test_builder_to_collection(self):
        from NEExT import NEExT

        nxt = NEExT(log_level="WARNING")
        graphs = []
        for i in range(10):
            b = GraphBuilder(seed=i)
            G = b.start_from("erdos_renyi", n=20, p=0.15).plant_community(size=5, p_internal=0.8).set_label(i % 2).build()
            graphs.append(G)
        collection = nxt.load_from_networkx(graphs)
        assert len(collection.graphs) == 10
        labels = [g.graph_label for g in collection.graphs]
        assert 0 in labels
        assert 1 in labels


# ═══════════════════════════════════════════════════════════════════════
# Imports from top-level
# ═══════════════════════════════════════════════════════════════════════


class TestTopLevelImports:
    def test_import_from_neext(self):
        from NEExT import GraphBuilder, GraphGenerator, SyntheticPresets

        assert GraphGenerator is not None
        assert GraphBuilder is not None
        assert SyntheticPresets is not None

    def test_import_from_generators(self):
        from NEExT.generators import (
            AnomalyConfig,
            AnomalyInjector,
            AttributeConfig,
            AttributeGenerator,
            CollectionSpec,
            GeneratorSpec,
            GraphAdapter,
            GraphBuilder,
            GraphGenerator,
            SyntheticPresets,
        )

        assert all(
            cls is not None
            for cls in [
                AnomalyConfig,
                AnomalyInjector,
                AttributeConfig,
                AttributeGenerator,
                CollectionSpec,
                GeneratorSpec,
                GraphAdapter,
                GraphBuilder,
                GraphGenerator,
                SyntheticPresets,
            ]
        )
