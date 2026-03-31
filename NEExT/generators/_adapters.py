"""Adapter layer for converting various graph formats to NetworkX graphs.

Generators can internally use any library (igraph, graph-tool, raw matrices, etc.).
This module normalizes everything to nx.Graph at the output boundary.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class GraphAdapter:
    """Converts graphs from various formats to NetworkX graphs.

    All methods preserve node attributes, edge attributes, and edge weights.
    Nodes are relabeled to consecutive integers when needed.
    """

    @staticmethod
    def from_igraph(ig_graph: Any) -> nx.Graph:
        """Convert an igraph.Graph to nx.Graph, preserving attributes."""
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("igraph is required for this conversion. Install with: pip install python-igraph")

        if not isinstance(ig_graph, ig.Graph):
            raise TypeError(f"Expected igraph.Graph, got {type(ig_graph)}")

        G = nx.Graph()
        n = ig_graph.vcount()
        G.add_nodes_from(range(n))

        # Copy vertex attributes
        for attr_name in ig_graph.vs.attributes():
            for v in ig_graph.vs:
                G.nodes[v.index][attr_name] = v[attr_name]

        # Copy edges and edge attributes
        edge_attr_names = ig_graph.es.attributes()
        for e in ig_graph.es:
            attrs = {name: e[name] for name in edge_attr_names}
            G.add_edge(e.source, e.target, **attrs)

        # Copy graph-level attributes
        for key in ig_graph.attributes():
            G.graph[key] = ig_graph[key]

        return G

    @staticmethod
    def from_graph_tool(gt_graph: Any) -> nx.Graph:
        """Convert a graph-tool Graph to nx.Graph, preserving attributes."""
        try:
            import graph_tool as gt
        except ImportError:
            raise ImportError("graph-tool is required for this conversion. Install graph-tool separately.")

        G = nx.Graph()
        n = gt_graph.num_vertices()
        G.add_nodes_from(range(n))

        # Copy vertex properties
        for prop_name in gt_graph.vertex_properties.keys():
            prop = gt_graph.vertex_properties[prop_name]
            for v in gt_graph.vertices():
                G.nodes[int(v)][prop_name] = prop[v]

        # Copy edges and edge properties
        edge_prop_names = list(gt_graph.edge_properties.keys())
        for e in gt_graph.edges():
            attrs = {}
            for prop_name in edge_prop_names:
                attrs[prop_name] = gt_graph.edge_properties[prop_name][e]
            G.add_edge(int(e.source()), int(e.target()), **attrs)

        # Copy graph properties
        for prop_name in gt_graph.graph_properties.keys():
            G.graph[prop_name] = gt_graph.graph_properties[prop_name]

        return G

    @staticmethod
    def from_adjacency_matrix(
        matrix: Any,
        node_attrs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> nx.Graph:
        """Convert a numpy/scipy adjacency matrix to nx.Graph."""
        # Handle scipy sparse matrices
        try:
            import scipy.sparse as sp

            if sp.issparse(matrix):
                matrix = matrix.toarray()
        except ImportError:
            pass

        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

        G = nx.from_numpy_array(matrix)

        if node_attrs:
            for node, attrs in node_attrs.items():
                if node in G:
                    G.nodes[node].update(attrs)

        return G

    @staticmethod
    def from_edge_list(
        edges: List[Tuple[int, int]],
        n_nodes: Optional[int] = None,
        node_attrs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> nx.Graph:
        """Convert a raw edge list to nx.Graph."""
        G = nx.Graph()

        if n_nodes is not None:
            G.add_nodes_from(range(n_nodes))

        G.add_edges_from(edges)

        if node_attrs:
            for node, attrs in node_attrs.items():
                if node in G:
                    G.nodes[node].update(attrs)

        # Relabel to consecutive integers if needed
        nodes = sorted(G.nodes())
        if nodes and (nodes[0] != 0 or nodes[-1] != len(nodes) - 1):
            mapping = {old: new for new, old in enumerate(nodes)}
            G = nx.relabel_nodes(G, mapping)

        return G

    @staticmethod
    def ensure_networkx(graph: Any) -> nx.Graph:
        """Auto-detect input type and convert to nx.Graph.

        Supports: nx.Graph, igraph.Graph, graph-tool Graph, numpy array, scipy sparse.
        """
        if isinstance(graph, nx.Graph):
            return graph

        type_name = type(graph).__module__ + "." + type(graph).__name__

        # igraph
        if "igraph" in type_name:
            return GraphAdapter.from_igraph(graph)

        # graph-tool
        if "graph_tool" in type_name:
            return GraphAdapter.from_graph_tool(graph)

        # numpy array
        if isinstance(graph, np.ndarray):
            return GraphAdapter.from_adjacency_matrix(graph)

        # scipy sparse
        try:
            import scipy.sparse as sp

            if sp.issparse(graph):
                return GraphAdapter.from_adjacency_matrix(graph)
        except ImportError:
            pass

        raise TypeError(f"Cannot convert {type(graph)} to nx.Graph. Supported types: nx.Graph, igraph.Graph, graph_tool.Graph, numpy.ndarray, scipy.sparse")


# Mapping from output_format string to adapter function
_FORMAT_ADAPTERS = {
    "networkx": lambda g: g if isinstance(g, nx.Graph) else GraphAdapter.ensure_networkx(g),
    "igraph": GraphAdapter.from_igraph,
    "graph_tool": GraphAdapter.from_graph_tool,
    "adjacency_matrix": GraphAdapter.from_adjacency_matrix,
    "edge_list": lambda g: GraphAdapter.from_edge_list(g),
}


def adapt_output(graph: Any, output_format: str = "networkx") -> nx.Graph:
    """Convert generator output to nx.Graph using the specified format adapter."""
    adapter = _FORMAT_ADAPTERS.get(output_format)
    if adapter is None:
        raise ValueError(f"Unknown output_format '{output_format}'. Supported: {list(_FORMAT_ADAPTERS.keys())}")
    return adapter(graph)
