import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union, get_args

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings
from NEExT.features import Features
from NEExT.graphs import Egonet, Graph
from NEExT.helper_functions import build_adjacency_lists, get_nodes_x_hops_away, hop_distances, random_walk_visit_counts
from NEExT.patches.leiden_patches import compute_patch_label, merge_small_communities, relabel_contiguous


def _python_scalar(value: Any) -> Any:
    """numpy scalars (as node tables deliver them) -> plain Python, so they fit the pydantic label types."""
    return value.item() if hasattr(value, "item") else value


class EgonetCollection(GraphCollection):
    """
    A collection of egonets derived from a GraphCollection.

    This class extends GraphCollection to specifically handle collections of
    egonets. Egonets are subgraphs centered around a specific node (the ego)
    and include its neighbors up to a certain distance (k-hop) or within a
    community.

    Attributes:
        egonet_feature_target (Optional[str]): The name of the node feature to
            be used as the target variable for the egonets. Defaults to None.
        skip_features (List[str]): A list of node feature names to be excluded
            when building egonets. Defaults to an empty list.
        egonet_to_graph_node_mapping (Dict[int, Tuple[int, int]]): A dictionary
            mapping egonet IDs to tuples of (original graph ID, original node ID).
            Defaults to an empty dictionary.
        egonet_node_features (Embeddings): An Embeddings object containing the
            features of a central node of each egonet. Defaults to None.
        source_graph_collection (Optional[GraphCollection]): The collection the
            egonets were built from, stashed by the builder methods so
            feature_scope="global" can compute features on the source graphs.
            Held by reference (keeps the source alive) and excluded from
            serialization. Defaults to None.
    """

    egonet_feature_target: Optional[str] = Field(default=None)
    skip_features: List[str] = Field(default_factory=list)
    egonet_to_graph_node_mapping: Dict[int, Tuple[int, int]] = Field(default_factory=dict)
    egonet_node_features: Embeddings = Field(default=None)
    source_graph_collection: Optional[GraphCollection] = Field(default=None, exclude=True, repr=False)
    # Patch tiling (compute_leiden_patches): one egonet per community rather than per node.
    patch_memberships: Dict[Union[int, str], List[int]] = Field(default_factory=dict)
    patch_label_fraction: Dict[int, float] = Field(default_factory=dict)
    patch_label_histogram: Dict[int, Dict[Union[int, float, str], int]] = Field(default_factory=dict)
    patch_merge_log: Dict[Union[int, str], List[Tuple[int, int]]] = Field(default_factory=dict)

    def _build_parent_index(self, graph: Graph) -> dict:
        """
        Precompute per-parent-graph lookup structures shared by every egonet of
        that graph, so each egonet build touches only its own neighborhood
        instead of scanning the whole parent graph.

        The position maps preserve the parent dicts' iteration order so egonet
        contents come out identical to a full-scan filter of those dicts.
        """
        skip_keys = frozenset(self.skip_features + ([self.egonet_feature_target] if self.egonet_feature_target else []))
        node_attr_pos = {nid: i for i, nid in enumerate(graph.node_attributes)}
        edge_attr_by_node = {}
        if graph.edge_attributes:
            edge_attr_by_node = defaultdict(list)
            for pos, (edge, attrs) in enumerate(graph.edge_attributes.items()):
                edge_attr_by_node[edge[0]].append((pos, edge, attrs))
        index = {
            "skip_keys": skip_keys,
            "node_attr_pos": node_attr_pos,
            "edge_attr_by_node": edge_attr_by_node,
        }
        if graph.graph_type == "igraph":
            # Attribute-free clone: igraph's subgraph() copies every vertex/edge
            # attribute of the parent, all of which the egonet build discards.
            # Same topology in the same order -> identical subgraph structure.
            import igraph as ig

            bare = ig.Graph(n=graph.G.vcount())
            bare.add_edges(graph.G.get_edgelist())
            index["bare_graph"] = bare
        return index

    def _build_egonet(
        self,
        graph: Graph,
        node_id: int,
        egonet_nodes: List[int],
        egonet_id: int,
        egonet_label: float,
        parent_index: Optional[dict] = None,
    ) -> Egonet:
        """
        This method constructs an Egonet object from a given graph.
        It extracts the relevant subgraph, node and
        edge attributes, and creates the necessary mappings.

        Args:
            graph (Graph): The original graph from which to extract the egonet.
            node_id (int): The ID of the center node (ego) of the egonet.
            egonet_nodes (List[int]): The list of node IDs that belong to the egonet.
            egonet_id (int): The unique ID to assign to the egonet.
            egonet_label (float): The label to assign to the egonet.
            parent_index (Optional[dict]): Precomputed per-parent lookups from
                _build_parent_index; computed on the fly when omitted.

        Returns:
            Egonet: The constructed egonet object.

        """

        # Sort nodes for deterministic ordering
        egonet_nodes_sorted = sorted(set(egonet_nodes))  # Remove duplicates and sort for determinism

        # build internal egonet node mapping and extract the features
        # Use sorted nodes for deterministic mapping
        node_mapping = {n: i for i, n in enumerate(egonet_nodes_sorted)}
        if parent_index is None:
            parent_index = self._build_parent_index(graph)
        skip_keys = parent_index["skip_keys"]
        egonet_nodes_set = set(egonet_nodes_sorted)

        # node attributes: visit only egonet members, in the parent dict's order
        node_attr_pos = parent_index["node_attr_pos"]
        node_attrs_src = graph.node_attributes
        attr_nodes = [nid for nid in egonet_nodes_sorted if nid in node_attr_pos]
        attr_nodes.sort(key=node_attr_pos.__getitem__)
        egonet_node_attributes = {
            node_mapping[nid]: {key: value for key, value in node_attrs_src[nid].items() if key not in skip_keys} for nid in attr_nodes
        }

        # edge attributes: gather via the per-node index, restore parent dict order
        egonet_edge_attributes = {}
        edge_attr_by_node = parent_index["edge_attr_by_node"]
        if edge_attr_by_node:
            matches = []
            for nid in egonet_nodes_sorted:
                for entry in edge_attr_by_node.get(nid, ()):
                    if entry[1][1] in egonet_nodes_set:
                        matches.append(entry)
            matches.sort(key=lambda entry: entry[0])
            for _, (src, dst), attrs in matches:
                egonet_edge_attributes[(node_mapping[src], node_mapping[dst])] = {key: value for key, value in attrs.items() if key not in skip_keys}

        # extract egonet subgraph (nx: cheap view; igraph: attribute-free clone
        # so the C subgraph routine yields identical structure without copying
        # the parent's attributes)
        if graph.graph_type == "networkx":
            G_egonet = graph.G.subgraph(egonet_nodes_sorted)
            nodes = list(range(G_egonet.number_of_nodes()))
            edges = [(node_mapping[u], node_mapping[v]) for u, v in G_egonet.edges()]
        else:
            G_egonet = parent_index["bare_graph"].subgraph(egonet_nodes_sorted)
            nodes = list(range(G_egonet.vcount()))
            edges = G_egonet.get_edgelist()

        egonet = Egonet(
            graph_id=egonet_id,
            graph_label=egonet_label,
            nodes=nodes,
            edges=edges,
            node_attributes=egonet_node_attributes,
            edge_attributes=egonet_edge_attributes,
            graph_type=graph.graph_type,
            node_mapping=node_mapping,
            original_graph_id=graph.graph_id,
            original_node_id=node_id,
        )
        egonet.initialize_graph()
        return egonet

    def compute_k_hop_egonets(
        self,
        graph_collection: GraphCollection,
        k_hop: int = 1,
        nodes_to_sample: Optional[Dict[int, List[int]]] = None,
        sample_fraction: Optional[float] = 1.0,
        random_seed: int = 13,
    ):
        """
        Computes egonets based on k-hop neighborhood.

        This method iterates through each node in each graph of the input
        GraphCollection and creates an egonet centered around that node,
        including all nodes within k-hop distance.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            k_hop (int): The maximum distance (in hops) from the center node
                to include in the egonet. Defaults to 1.

        """
        rng = np.random.RandomState(random_seed)
        if nodes_to_sample is None:
            nodes_to_sample = {}

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        valid_nodes = {}
        # draw nodes to sample from for each graph
        for graph in graph_collection.graphs:
            nodes = graph.nodes
            random_nodes = rng.choice(nodes, int(len(nodes) * sample_fraction), replace=False).tolist()
            forced_nodes = nodes_to_sample.get(graph.graph_id, [])
            valid_nodes[graph.graph_id] = list(set(random_nodes + forced_nodes))

        for graph in graph_collection.graphs:
            parent_index = self._build_parent_index(graph)
            for node_id in valid_nodes[graph.graph_id]:
                ring_of = {node_id: 0}
                if k_hop > 0:
                    egonet_nodes_dict = get_nodes_x_hops_away(graph.G, node_id, k_hop)
                    egonet_nodes = {node_id}
                    for hop, v in egonet_nodes_dict.items():
                        egonet_nodes.update(v)
                        ring_of.update({member: hop for member in v})
                else:
                    egonet_nodes = [node_id]
                # egonet_nodes = sorted(graph.G.neighborhood(node_id, order=k_hop))
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                    parent_index=parent_index,
                )
                egonet.node_rings = {egonet.node_mapping[member]: ring for member, ring in ring_of.items()}

                self.graphs.append(egonet)
                # Update graph_id_node_array with this egonet's id
                self.graph_id_node_array.extend([egonet_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)
        self.source_graph_collection = graph_collection

    def compute_random_walk_egonets(
        self,
        graph_collection: GraphCollection,
        walk_length: int = 10,
        n_walks: int = 100,
        restart_prob: float = 0.15,
        min_visits: int = 3,
        max_egonet_size: Optional[int] = None,
        weight_by_visits: bool = True,
        nodes_to_sample: Optional[Dict[int, List[int]]] = None,
        sample_fraction: Optional[float] = 1.0,
        random_seed: int = 13,
    ):
        """
        Computes egonets from random walks with restart.

        For each sampled center, runs one random walk with restart of
        `n_walks * walk_length` steps: at every step the walker jumps back to
        the center with probability `restart_prob`, otherwise it moves to a
        uniformly random neighbor. The visited nodes form the egonet and the
        visit frequencies (L1-normalized per egonet) become the egonet's
        membership weights, stored as `Egonet.node_weights` keyed by internal
        node ID; they estimate the center's personalized PageRank vector with
        teleport probability `restart_prob`. Unlike k-hop balls, these
        neighborhoods conform to community structure, and adjacent centers
        produce gradually-differing bags.

        Walks are simulated over precomputed adjacency lists with a
        self-contained RNG (igraph's own random_walk uses a process-global C
        RNG). Each center's walk RNG is seeded from (random_seed, graph
        position, center node ID), so a center's egonet is reproducible and
        does not depend on which other centers are sampled. The center is
        always a member.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            walk_length (int): Steps per walk segment; the walk takes
                n_walks * walk_length steps in total (default: 10)
            n_walks (int): Number of walk segments per center; only the
                product n_walks * walk_length matters (default: 100)
            restart_prob (float): Per-step probability of returning to the
                center; 0.0 gives pure walks (default: 0.15)
            min_visits (int): Minimum visit events for membership; the center
                is always kept. The default of 3 is a noise floor over the
                n_walks*walk_length + 1 visit events that cuts the one-visit
                fringe (which balloons bag size without adding signal); set
                1 to keep every visited node (default: 3)
            max_egonet_size (Optional[int]): Keep only the top-N most visited
                members (center always kept). None keeps all (default: None)
            weight_by_visits (bool): Attach normalized visit frequencies as
                `node_weights`; False builds unweighted membership egonets
                (default: True)
            nodes_to_sample: Dict mapping graph_id to nodes to always include
            sample_fraction: Fraction of nodes to sample egonets for (default: 1.0)
            random_seed: Random seed for center sampling and walks (default: 13)
        """
        if walk_length < 1:
            raise ValueError(f"walk_length must be >= 1, got {walk_length}")
        if n_walks < 1:
            raise ValueError(f"n_walks must be >= 1, got {n_walks}")
        if not 0.0 <= restart_prob < 1.0:
            raise ValueError(f"restart_prob must be in [0, 1), got {restart_prob}")
        if min_visits < 1:
            raise ValueError(f"min_visits must be >= 1, got {min_visits}")
        if max_egonet_size is not None and max_egonet_size < 1:
            raise ValueError(f"max_egonet_size must be >= 1, got {max_egonet_size}")

        rng = np.random.RandomState(random_seed)
        if nodes_to_sample is None:
            nodes_to_sample = {}

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        valid_nodes = {}
        # draw nodes to sample from for each graph
        for graph in graph_collection.graphs:
            nodes = graph.nodes
            random_nodes = rng.choice(nodes, int(len(nodes) * sample_fraction), replace=False).tolist()
            forced_nodes = nodes_to_sample.get(graph.graph_id, [])
            valid_nodes[graph.graph_id] = list(set(random_nodes + forced_nodes))

        for graph_position, graph in enumerate(graph_collection.graphs):
            parent_index = self._build_parent_index(graph)
            adjacency = build_adjacency_lists(graph.G)
            for node_id in valid_nodes[graph.graph_id]:
                # Per-center seed from (seed, graph, center): independent of the sampled center set.
                walk_seed = np.random.SeedSequence([random_seed, graph_position, int(node_id)]).generate_state(1)[0]
                walk_rng = np.random.RandomState(walk_seed)
                counts = random_walk_visit_counts(adjacency, node_id, n_walks * walk_length, restart_prob, walk_rng)

                if min_visits > 1:
                    counts = {n: c for n, c in counts.items() if c >= min_visits or n == node_id}
                if max_egonet_size is not None and len(counts) > max_egonet_size:
                    kept = sorted(counts, key=lambda n: (n == node_id, counts[n]), reverse=True)[:max_egonet_size]
                    counts = {n: counts[n] for n in kept}

                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None
                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=list(counts),
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                    parent_index=parent_index,
                )
                if weight_by_visits:
                    total = sum(counts.values())
                    egonet.node_weights = {egonet.node_mapping[n]: c / total for n, c in counts.items()}
                egonet.node_rings = {egonet.node_mapping[n]: ring for n, ring in hop_distances(adjacency, node_id, counts).items()}

                self.graphs.append(egonet)
                self.graph_id_node_array.extend([egonet_id] * len(egonet.nodes))
                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)
        self.source_graph_collection = graph_collection

    def compute_leiden_egonets(self, graph_collection: GraphCollection, n_iterations: int = 10, resolution: float = 1.0, random_seed: int = 13):
        """
        Computes egonets based on Leiden community detection.

        This method iterates through each graph in the input GraphCollection,
        performs Leiden community detection, and then creates an egonet for
        each node, including all nodes in the same community.

        Args:
            graph_collection (GraphCollection): The collection of graphs from
                which to derive egonets.
            n_iterations (int): The number of iterations for the Leiden
                algorithm. Defaults to 10.
            resolution (float): The resolution parameter for the Leiden
                algorithm. Defaults to 1.0.
            random_seed (int): Seed for Leiden, set on igraph's RNG for the
                call and restored afterwards, as in compute_leiden_patches.
                Defaults to 13.

        """

        if any(g.graph_type == "networkx" for g in graph_collection.graphs):
            raise NotImplementedError("Leiden egonets require iGraph backend. Use graph_type='igraph' when loading graphs.")

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        egonet_id = 0

        import igraph as ig

        seed_rng = random.Random(random_seed)
        for graph in graph_collection.graphs:
            # Leiden under a private RNG; restore the module default so nothing else observes the change.
            ig.set_random_number_generator(random.Random(seed_rng.randint(0, 2**31 - 1)))
            try:
                community_detection = graph.G.community_leiden(objective_function="modularity", n_iterations=n_iterations, resolution=resolution)
            finally:
                ig.set_random_number_generator(random)
            membership = community_detection.membership
            community_members = defaultdict(list)
            for n_id, com_id in enumerate(membership):
                community_members[com_id].append(n_id)
            parent_index = self._build_parent_index(graph)

            for node_id in range(graph.G.vcount()):
                egonet_nodes = community_members[membership[node_id]]
                egonet_label = graph.node_attributes[node_id][self.egonet_feature_target] if self.egonet_feature_target else None

                egonet = self._build_egonet(
                    graph=graph,
                    node_id=node_id,
                    egonet_nodes=egonet_nodes,
                    egonet_id=egonet_id,
                    egonet_label=egonet_label,
                    parent_index=parent_index,
                )
                self.graphs.append(egonet)
                # Update graph_id_node_array with this egonet's id
                self.graph_id_node_array.extend([egonet_id] * len(egonet.nodes))

                self.egonet_to_graph_node_mapping[egonet_id] = (graph.graph_id, node_id)
                egonet_id += 1

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)
        self.source_graph_collection = graph_collection

    def compute_leiden_patches(
        self,
        graph_collection: GraphCollection,
        resolution: float = 1.0,
        n_iterations: int = 10,
        min_patch_size: int = 5,
        expand_one_hop: bool = False,
        random_seed: int = 13,
        patch_label: Literal["majority", "centre", "purity", "distribution"] = "majority",
        purity_threshold: float = 0.8,
    ):
        """
        Tile every graph into patches: one subgraph per Leiden community.

        Unlike ``compute_leiden_egonets`` (one egonet per NODE, equal to that node's
        community), this builds one ``Egonet`` per COMMUNITY. Communities smaller than
        ``min_patch_size`` are merged into the neighbouring community they share the
        most edges with (smallest first, repeated until none remain; a small community
        with no neighbour is kept). With ``expand_one_hop`` each patch also takes its
        one-hop neighbours from the source graph, so adjacent patches overlap at their
        boundary; memberships are always recorded pre-expansion, as a partition.

        The patch's centre (``original_node_id``) is its highest-degree member in the
        source graph (ties: lowest id), so code that expects a centre keeps working.
        When ``egonet_feature_target`` names a node attribute, each patch receives a
        label computed from its members' labels by ``patch_label``: "majority" (most
        frequent; ties to the lowest label), "centre" (the centre's label), "purity"
        (1 if the majority fraction is at least ``purity_threshold``) or "distribution"
        (the histogram; ``graph_label`` then holds the majority). The majority fraction
        and the histogram are stored for every patch regardless of the scheme.

        Leiden is seeded from ``random_seed`` through igraph's RNG, which is set for
        the call and restored to Python's ``random`` module afterwards; the global
        ``random`` stream is not consumed.

        Requires the igraph backend.

        Args:
            graph_collection: Source graphs (igraph backend).
            resolution: Leiden resolution; higher gives more, smaller patches.
            n_iterations: Leiden iterations.
            min_patch_size: Communities below this size are merged (default: 5).
            expand_one_hop: Add one-hop neighbours to every patch (default: False).
            random_seed: Seed for Leiden (default: 13).
            patch_label: Label scheme, used when ``egonet_feature_target`` is set.
            purity_threshold: Threshold for the "purity" scheme.
        """
        if any(g.graph_type == "networkx" for g in graph_collection.graphs):
            raise NotImplementedError("Leiden patches require iGraph backend. Use graph_type='igraph' when loading graphs.")
        if min_patch_size < 1:
            raise ValueError(f"min_patch_size must be >= 1, got {min_patch_size}")

        import igraph as ig

        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        self.patch_memberships = {}
        self.patch_label_fraction = {}
        self.patch_label_histogram = {}
        self.patch_merge_log = {}
        patch_id = 0
        seed_rng = random.Random(random_seed)

        for graph in graph_collection.graphs:
            # Leiden under a private RNG; restore the module default so nothing else observes the change.
            ig.set_random_number_generator(random.Random(seed_rng.randint(0, 2**31 - 1)))
            try:
                raw_membership = graph.G.community_leiden(
                    objective_function="modularity", n_iterations=n_iterations, resolution=resolution
                ).membership
            finally:
                ig.set_random_number_generator(random)

            membership, merge_log = merge_small_communities(graph.edges, raw_membership, min_patch_size)
            membership = relabel_contiguous(membership)
            self.patch_merge_log[graph.graph_id] = merge_log
            patch_id = self._add_patches(graph, membership, expand_one_hop, patch_label, purity_threshold, patch_id)

        self.egonet_node_features = self._create_egonet_features_df(graph_collection)
        self.source_graph_collection = graph_collection

    def _add_patches(self, graph, membership, expand_one_hop, patch_label, purity_threshold, patch_id):
        """Build one patch per membership class of `graph` (centre = highest-degree member, ties to the lowest id),
        optionally expanded by one hop, append them to this collection and return the next free patch id."""
        self.patch_memberships[graph.graph_id] = membership
        members_by_patch: Dict[int, List[int]] = defaultdict(list)
        for node_id, com_id in enumerate(membership):
            members_by_patch[com_id].append(node_id)
        degrees = graph.G.degree()
        parent_index = self._build_parent_index(graph)

        for com_id in sorted(members_by_patch):
            members = members_by_patch[com_id]
            centre = min(members, key=lambda n: (-degrees[n], n))
            patch_nodes = list(members)
            if expand_one_hop:
                expanded = set(members)
                for neighbourhood in graph.G.neighborhood(members, order=1):
                    expanded.update(neighbourhood)
                patch_nodes = sorted(expanded)

            label = None
            if self.egonet_feature_target:
                node_labels = [_python_scalar(graph.node_attributes[n][self.egonet_feature_target]) for n in members]
                centre_label = _python_scalar(graph.node_attributes[centre][self.egonet_feature_target])
                label, fraction, histogram = compute_patch_label(
                    node_labels, patch_label, centre_label=centre_label, purity_threshold=purity_threshold
                )
                self.patch_label_fraction[patch_id] = fraction
                self.patch_label_histogram[patch_id] = histogram

            egonet = self._build_egonet(
                graph=graph,
                node_id=centre,
                egonet_nodes=patch_nodes,
                egonet_id=patch_id,
                egonet_label=label,
                parent_index=parent_index,
            )
            self.graphs.append(egonet)
            self.graph_id_node_array.extend([patch_id] * len(egonet.nodes))
            self.egonet_to_graph_node_mapping[patch_id] = (graph.graph_id, centre)
            patch_id += 1
        return patch_id

    def compute_membership_patches(
        self,
        graph_collection: GraphCollection,
        memberships: Dict[int, List[int]],
        expand_one_hop: bool = False,
        patch_label: Literal["majority", "centre", "purity", "distribution"] = "majority",
        purity_threshold: float = 0.8,
    ):
        """
        Tile every graph into patches from a given partition: one patch per class of ``memberships[graph_id]``
        (a list giving each node's part, node ids 0..n-1 of the source graph), e.g. METIS parts. Patches are built
        exactly as in ``compute_leiden_patches`` (centre, optional one-hop expansion, labels), without merging.
        Requires the igraph backend.
        """
        if any(g.graph_type == "networkx" for g in graph_collection.graphs):
            raise NotImplementedError("Membership patches require iGraph backend. Use graph_type='igraph' when loading graphs.")
        self.graphs = []
        self.graph_id_node_array = []
        self.egonet_to_graph_node_mapping = {}
        self.patch_memberships = {}
        self.patch_label_fraction = {}
        self.patch_label_histogram = {}
        self.patch_merge_log = {}
        patch_id = 0
        for graph in graph_collection.graphs:
            membership = memberships[graph.graph_id]
            if len(membership) != len(graph.nodes):
                raise ValueError(f"graph {graph.graph_id}: membership has {len(membership)} entries for {len(graph.nodes)} nodes")
            membership = relabel_contiguous(list(membership))
            self.patch_merge_log[graph.graph_id] = []
            patch_id = self._add_patches(graph, membership, expand_one_hop, patch_label, purity_threshold, patch_id)
        self.egonet_node_features = self._create_egonet_features_df(graph_collection)
        self.source_graph_collection = graph_collection

    def project_source_features(self, source_features: Features) -> Features:
        """
        Project per-node features computed on the SOURCE graphs onto egonet members.

        Expands every egonet's node_mapping (original -> internal id) into one row
        per member and joins each member's feature vector from `source_features`
        (keyed by the source graph's (graph_id, node_id)). The result has the same
        shape a local (within-egonet) computation produces — one row per
        (internal node_id, egonet graph_id), columns
        ["node_id", "graph_id", <feature columns>] — so downstream embeddings
        consume it unchanged. Rows are emitted in egonet order with ascending
        internal ids. Feature values are taken as-is; no re-normalization.

        Hard-fails if any egonet member has no feature row in `source_features`
        (typically caused by node sampling on the source collection) — a silent
        drop here would de-pair members from their bags.

        Args:
            source_features (Features): Features computed on the source
                GraphCollection the egonets were built from.

        Returns:
            Features: Projected features for every egonet member.
        """
        node_ids, graph_ids, src_gids, src_nids = [], [], [], []
        for egonet in self.graphs:
            for orig, internal in sorted(egonet.node_mapping.items(), key=lambda item: item[1]):
                node_ids.append(internal)
                graph_ids.append(egonet.graph_id)
                src_gids.append(egonet.original_graph_id)
                src_nids.append(orig)
        members = pd.DataFrame({"node_id": node_ids, "graph_id": graph_ids, "_src_graph_id": src_gids, "_src_node_id": src_nids})

        src = source_features.features_df.rename(columns={"graph_id": "_src_graph_id", "node_id": "_src_node_id"})
        merged = members.merge(src, on=["_src_graph_id", "_src_node_id"], how="left", indicator=True)

        missing = merged[merged["_merge"] != "both"]
        if len(missing):
            examples = list(missing[["_src_graph_id", "_src_node_id"]].itertuples(index=False, name=None))[:5]
            raise ValueError(
                f"{len(missing)} egonet member(s) have no feature row in the source features, "
                f"e.g. (graph_id, node_id) = {examples}. Likely cause: the source collection was "
                f"node-sampled (node_sample_rate < 1.0) or the features were computed on a "
                f"different collection than the egonets were built from."
            )

        projected = merged[["node_id", "graph_id"] + list(source_features.feature_columns)].reset_index(drop=True)
        return Features(projected, list(source_features.feature_columns), source_features.feature_groups)

    def _create_egonet_features_df(self, graph_collection: GraphCollection):
        """
        Creates a DataFrame containing node features for each egonet.

        This method extracts node features from the original graph collection and
        organizes them into a DataFrame where each row represents an egonet and
        its associated node features.

        Args:
            graph_collection (GraphCollection): The original collection of graphs
                from which the egonets were derived.

        Returns:
            Embeddings: An Embeddings object containing the egonet node features DataFrame.
        """
        egonet_node_features_df = pd.DataFrame().from_dict(self.egonet_to_graph_node_mapping, orient="index").reset_index()
        egonet_node_features_df.columns = ["subgraph_id", "graph_id", "node_id"]

        skip_keys = frozenset(self.skip_features + ([self.egonet_feature_target] if self.egonet_feature_target else []))
        raw_features = {}

        for graph in graph_collection.graphs:
            graph_id = graph.graph_id
            for node_id, features in graph.node_attributes.items():
                kept = {feature: value for feature, value in features.items() if feature not in skip_keys}
                if kept:
                    raw_features[graph_id, node_id] = kept

        raw_features = (
            pd.DataFrame.from_dict(raw_features, orient="index").reset_index().rename(columns={"level_0": "graph_id", "level_1": "node_id"})
        )

        egonet_node_features_df = (
            (
                egonet_node_features_df.merge(raw_features, on=["graph_id", "node_id"])
                .drop(columns=["graph_id", "node_id"])
                .rename(columns={"subgraph_id": "graph_id"})
            )
            if raw_features is not None and len(raw_features) > 0
            else (egonet_node_features_df.drop(columns=["graph_id", "node_id"]).rename(columns={"subgraph_id": "graph_id"}))
        )
        return Embeddings(egonet_node_features_df, "egonet_node_features", [col for col in egonet_node_features_df.columns if col != "graph_id"])

    def compute_egonet_positional_features(
        self,
        strategy: Literal["distance", "inv_distance", "inv_exp_distance"],
        one_hot_encode: bool = False,
    ):
        """
        Compute egonet positional features that can be used to encode central node
        position in the egonet. The positional features have to be added independently
        to features before embedding if you want to include it.
        """

        node_ids: List[int] = []
        graph_ids: List[int] = []
        positions: List[float] = []
        for egonet in self.graphs:
            _, central_node = self.egonet_to_graph_node_mapping[egonet.graph_id]
            mapped_central = egonet.node_mapping[central_node]

            if egonet.graph_type == "igraph":
                egonet_positions = egonet.G.distances(mapped_central)[0]
            else:
                lengths = nx.single_source_shortest_path_length(egonet.G, mapped_central)
                egonet_positions = [lengths.get(n, float("inf")) for n in egonet.nodes]
            node_ids.extend(egonet.nodes)
            graph_ids.extend([egonet.graph_id] * len(egonet.nodes))
            positions.extend(egonet_positions)

        df_position = pd.DataFrame({"node_id": node_ids, "graph_id": graph_ids, "egonet_position": positions})
        if strategy == "inv_distance":
            df_position["egonet_position"] = 1 / (df_position["egonet_position"] + 1)
        elif strategy == "inv_exp_distance":
            df_position["egonet_position"] = 1 / np.exp(df_position["egonet_position"] + 1)

        if one_hot_encode and strategy == "distance":
            df_position = pd.get_dummies(df_position, columns=["egonet_position"], dtype=np.int8)
        elif not one_hot_encode and strategy == "distance":
            df_position["egonet_position"] = MinMaxScaler().fit_transform(df_position[["egonet_position"]])

        positional_features = Features(df_position, [i for i in df_position.columns if i not in ["node_id", "graph_id"]])
        return positional_features
