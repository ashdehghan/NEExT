from typing import List
from NEExT.graph import Graph
import networkx as nx


def build_egonet(
    graph: Graph,
    egonet_nodes: List[int],
    graph_id: int,
    graph_label: float,
    skip_features: List[str] = None,
):
    if skip_features is None:
        skip_features = []

    node_mapping = {n: i for i, n in enumerate(egonet_nodes)}
    sub_node_attributes = {
        node_mapping[node_id]: {
            key: value
            for key, value in feature_dict.items()
            if key not in skip_features
        }
        for node_id, feature_dict in graph.node_attributes.items()
        if node_id in egonet_nodes
    }
    sub_edge_attributes = {
        node_mapping[node_id]: {
            key: value
            for key, value in feature_dict.items()
            if key not in skip_features
        }
        for node_id, feature_dict in graph.edge_attributes.items()
        if node_id in egonet_nodes
    }

    G_sub = graph.G.subgraph(list(set(egonet_nodes)))
    nodes = list(range(G_sub.vcount()))
    edges = G_sub.get_edgelist()

    return Graph(
        graph_id=graph_id,
        graph_label=graph_label,
        nodes=nodes,
        edges=edges,
        node_attributes=sub_node_attributes,
        edge_attributes=sub_edge_attributes,
        graph_type="networkx" if isinstance(G_sub, nx.Graph) else "igraph",
        node_mapping=node_mapping,
    )


def one_hop_algorithm(graph: Graph, node_id: int) -> List[int]:
    subgraph_nodes = [node_id] + graph.G.neighbors(node_id)
    return list(set(subgraph_nodes))


def two_hop_algorithm(graph: Graph, node_id: int) -> List[int]:
    subgraph_nodes = [node_id] + graph.G.neighbors(node_id)
    for neigh in graph.G.neighbors(node_id):
        subgraph_nodes += graph.G.neighbors(neigh)
    return list(set(subgraph_nodes))
