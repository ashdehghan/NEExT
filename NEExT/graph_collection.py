"""
	Author : Ash Dehghan
	Description: This class uses the Networkx library as a base to
	build a class that handles a collection of graphs, to be used
	for graph analysis.
"""

# External Libraries
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Internal Modules
from NEExT.graph_object import Graph_Object


class Graph_Collection:

	def __init__(self, global_config):
		self.global_config = global_config
		self.graph_collection = []
		self.total_numb_of_nodes = None
		self.graph_id_node_array = None
		self.graph_label_list_unique = None
		self.grpah_labels_df = None
		self.global_embeddings_cols = None
		self.global_embeddings = None
		self.global_embeddings_cols_arc = None
		self.global_embeddings_arc = None


	def load_graphs_from_csv(self, edge_file, node_graph_mapping_file, node_features_file=None):
		"""
			This method uses the user configuration to build a collection
			of graphs object.
		"""
		edges = pd.read_csv(edge_file)
		src_nodes = [int(i) for i in edges["node_a"].tolist()]
		dst_nodes = [int(i) for i in edges["node_b"].tolist()]
		edgelist = list(zip(src_nodes, dst_nodes))
		G = nx.from_edgelist(edgelist)
		# Add node features if exists
		if node_features_file:
			node_features = pd.read_csv(node_features_file)
			node_features_map = {}
			for idx in tqdm(range(len(node_features)), desc="Loading node features:", disable=self.global_config.quiet_mode):
				feats = dict(node_features.iloc[idx])
				node_id = int(feats["node_id"])
				del feats["node_id"]
				node_features_map[node_id] = feats
			nx.set_node_attributes(G, node_features_map)
		node_graph_map = pd.read_csv(node_graph_mapping_file)
		node_graph_map["node_id"] = node_graph_map["node_id"].astype(int)
		node_graph_map["graph_id"] = node_graph_map["graph_id"].astype(int)
		graph_ids = node_graph_map["graph_id"].unique().tolist()
		self.total_numb_of_nodes = 0
		self.graph_id_node_array = []
		for graph_id in tqdm(graph_ids, desc="Building subgraphs:", disable=self.global_config.quiet_mode):
			node_list = node_graph_map[node_graph_map["graph_id"] == graph_id]["node_id"].tolist()
			g = nx.Graph(G.subgraph(node_list))
			cc = list(nx.connected_components(g))
			g_obj = Graph_Object()
			g_obj.graph_id = graph_id
			g_obj.graph = g
			g_obj.numb_of_nodes = len(g.nodes)
			g_obj.numb_of_edges = len(g.edges)
			g_obj.numb_of_connected_components = len(cc)
			g_obj.connected_components = sorted(cc, key=len, reverse=True)
			self.graph_collection.append(g_obj)
			self.total_numb_of_nodes += len(g.nodes)
			self.graph_id_node_array.extend(np.repeat(g_obj.graph_id, len(g.nodes)))


	def filter_collection_for_largest_connected_component(self):
		"""
			This method will go through all the sub-graphs and if the number
			of component of the sub-graph is greater than 1, it will only keep the largest component.
		"""
		self.total_numb_of_nodes = 0
		self.graph_id_node_array = []
		for g_obj in tqdm(self.graph_collection, desc="Filtering graphs:", disable=self.global_config.quiet_mode):
			largest_cc = list(g_obj.connected_components[0])
			g = nx.Graph(g_obj.graph.subgraph(largest_cc))
			cc = list(nx.connected_components(g))
			g_obj.graph = g
			g_obj.numb_of_nodes = len(g.nodes)
			g_obj.numb_of_edges = len(g.edges)
			g_obj.diameter = nx.diameter(g)
			g_obj.numb_of_connected_components = len(cc)
			g_obj.connected_components = sorted(cc, key=len, reverse=True)
			self.total_numb_of_nodes += len(g.nodes)
			self.graph_id_node_array.extend(np.repeat(g_obj.graph_id, len(g.nodes)))


	def reset_node_indices(self):
		"""
			This method will reset the node indices to start from 0.
			It will keep a mapping between old and new node indices.
		"""
		for g_obj in tqdm(self.graph_collection, desc="Resrting node indices:", disable=self.global_config.quiet_mode):
			g = g_obj.graph
			mapping = {}
			current_nodes = list(g.nodes)
			for idx, node in enumerate(current_nodes):
				mapping[node] = idx
			g_obj.graph = nx.relabel_nodes(g, mapping)
			g_obj.re_index_map = mapping


	def export_graph_collection_stats(self):
		"""
			This method export a collection of basic statistics about the graph collection.
		"""
		
		stat_numb_of_nodes = []
		stat_avg_node_degree = []
		for g_obj in tqdm(self.graph_collection, desc="Building stats:", disable=self.global_config.quiet_mode):
			g = g_obj.graph
			stat_numb_of_nodes.append(len(g.nodes))
			stat_avg_node_degree.append(np.mean(np.array(g.degree)[:,1]))
		stat_obj = {}
		stat_obj["numb_node_dist"] = stat_numb_of_nodes
		stat_obj["avg_node_degree"] = stat_avg_node_degree
		return stat_obj


	def assign_graph_labels_from_csv(self, graph_label_file):
		"""
			This function will take as input graph label csv path.
			It will load the labels and assigns it to graphs in the collection.
		"""
		graph_labels = pd.read_csv(graph_label_file)
		self.grpah_labels_df = graph_labels.copy(deep=True)
		self.graph_label_list_unique = graph_labels["graph_label"].unique().tolist()
		graph_labels = graph_labels.set_index("graph_id")["graph_label"].to_dict()
		no_label_counter = 0
		for g_obj in tqdm(self.graph_collection, desc="Assigning graph labels:", disable=self.global_config.quiet_mode):
			graph_id = g_obj.graph_id
			if graph_id in graph_labels:
				g_obj.graph_label = graph_labels[graph_id]
			else:
				g_obj.graph_label = "unknown"
				no_label_counter += 1
