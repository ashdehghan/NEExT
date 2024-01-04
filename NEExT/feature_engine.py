"""
	This class contains methods which allows the user
	to compute various features on the graph, which 
	could capture various properties of the graph
	including structural, density, ...
"""

# External Libraries
import copy
import umap
import json
import random
import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA

# Internal Libraries
# from ugaf.global_config import Global_Config
from ugaf.helper_functions import get_nodes_x_hops_away
from ugaf.node_embedding_engine import Node_Embedding_Engine


class Feature_Engine:


	def __init__(self, global_config):
		self.global_config = global_config
		self.node_emb_engine = Node_Embedding_Engine()
		self.feature_functions = {}
		self.feature_functions["basic_node_features"] = self.build_basic_node_features
		self.feature_functions["lsme"] = self.build_lsme
		self.feature_functions["basic_expansion"] = self.build_basic_expansion
		self.feature_functions["self_walk"] = self.build_self_walk
		self.feature_functions["structural_node_feature"] = self.build_structural_node_features
		self.supported_structural_node_features = ["page_rank", "degree_centrality", "closeness_centrality", "load_centrality", "eigenvector_centrality"]


	def build_features(self, G, graph_id):
		config = self.global_config.config["graph_features"]
		node_samples = self.sample_graph(G, config)
		feature_collection = {}
		feature_collection["features"] = {}
		for feature_obj in config["features"]:
			feature_collection = self.feature_functions[feature_obj["feature_name"]](feature_collection, G, feature_obj, feature_obj["type"], node_samples, graph_id)
		feature_collection = self.build_gloabal_embedding(feature_collection, node_samples, config)
		return feature_collection


	def sample_graph(self, G, config):
		"""
			This method will sample the graph based on the information
			given in the configuration.
		"""
		if self.global_config.config["graph_sample"]["flag"] == "no":
			node_samples = list(G.nodes)
		else:
			sample_size = int(len(G.nodes) * config["graph_sample"]["sample_fraction"])
			graph_nodes = list(G.nodes)[:]
			random.shuffle(graph_nodes)
			node_samples = graph_nodes[0:sample_size]
		return node_samples


	def build_gloabal_embedding(self, feature_collection, node_samples, config):
		"""
			This method will use the features built on the graph to construct
			a global embedding for the nodes of the graph.
		"""
		global_emb_df = pd.DataFrame()
		if config["gloabl_embedding"]["type"] == "concat":
			for func_name in feature_collection["features"]:
				if "embs" in feature_collection["features"][func_name]:
					embs = feature_collection["features"][func_name]["embs"]
					if global_emb_df.empty:
						global_emb_df = embs.copy(deep=True)
					else:
						global_emb_df = global_emb_df.merge(embs, on=["node_id", "graph_id"], how="inner")
			feature_collection["global_embedding"] = global_emb_df
		else:
			raise ValueError("Gloabl embedding type is not supported.")
		return feature_collection



	def build_self_walk(self, feature_collection, G, config, func_name, node_samples, graph_id):
		iG = ig.Graph.from_networkx(G)
		A = np.array(iG.get_adjacency().data)
		emb_dim = config["emb_dim"]
		Ao = copy.deepcopy(A)
		embs = {}
		for i in range(2, emb_dim+2):
			A = np.linalg.matrix_power(Ao, i)
			diag_elem = np.diag(A)
			embs["emb_selfwalk_"+str(i-2)] = list(diag_elem)
		embs = pd.DataFrame(embs)
		emb_cols = list(embs.columns)
		embs.insert(0, "node_id", list(G.nodes))
		embs.insert(1, "graph_id", graph_id)
		feature_collection["features"][func_name] = {}
		feature_collection["features"][func_name]["embs"] = embs
		feature_collection["features"][func_name]["embs_cols"] = emb_cols
		return feature_collection
			

	def build_basic_node_features(self, feature_collection, G, config, func_name, node_samples, graph_id):
		node_feature_list = []
		emb_cols = None
		for node in G.nodes:
			emb_cols = list(G.nodes[node].keys())
			node_feature_list.append(list(G.nodes[node].values()))
		embs = pd.DataFrame(node_feature_list)
		emb_cols = ["emb_"+i for i in emb_cols]
		embs.columns = emb_cols
		embs.insert(0, "node_id", list(G.nodes))
		embs.insert(1, "graph_id", graph_id)
		feature_collection["features"][func_name] = {}
		feature_collection["features"][func_name]["embs"] = embs
		feature_collection["features"][func_name]["embs_cols"] = emb_cols
		return feature_collection


	def build_structural_node_features(self, feature_collection, G, config, func_name, node_samples, graph_id):
		"""
			This method will compute structural node feature for every node up to 
			emb_dim hops away neighbors.
		"""
		structural_feature_type = config["type"]
		if structural_feature_type == "page_rank":
			srtct_feat = nx.pagerank(G, alpha=0.9)
		elif structural_feature_type == "degree_centrality":
			srtct_feat = nx.degree_centrality(G)
		elif structural_feature_type == "closeness_centrality":
			srtct_feat = nx.closeness_centrality(G)
		elif structural_feature_type == "load_centrality":
			srtct_feat = nx.load_centrality(G)
		elif structural_feature_type == "eigenvector_centrality":
			srtct_feat = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-03)
		else:
			raise ValueError("The selected structural feature is not supported.")
		emb_dim = int(config["emb_dim"])
		nodes = []
		features = []
		for node in list(G.nodes):
			if node in node_samples:
				nodes.append(node)
				feat_vect = []
				nbs = get_nodes_x_hops_away(G, node, max_hop_length=emb_dim)
				feat_vect.append(srtct_feat[node])
				for i in range(1, emb_dim):
					if i in nbs:
						nbs_pr = [srtct_feat[j] for j in nbs[i]]
						feat_vect.append(sum(nbs_pr)/len(nbs_pr))
					else:
						feat_vect.append(0.0)
				features.append(feat_vect)
		# Construct embedding df
		embs = pd.DataFrame(features)
		emb_cols = ["emb_"+func_name+"_"+str(i) for i in range(embs.shape[1])]
		embs.columns = emb_cols
		embs.insert(0, "node_id", nodes)
		embs.insert(1, "graph_id", graph_id)
		feature_collection["features"][func_name] = {}
		feature_collection["features"][func_name]["embs"] = embs
		feature_collection["features"][func_name]["embs_cols"] = emb_cols
		return feature_collection


	def build_lsme(self, feature_collection, G, config, func_name, node_samples, graph_id):
		emb_dim = config["emb_dim"]
		embs = self.node_emb_engine.run_lsme_embedding(G, emb_dim, node_samples)
		emb_cols = []
		for col in embs.columns.tolist():
			if "emb" in col:
				emb_cols.append(col)
		embs.insert(1, "graph_id", graph_id)
		feature_collection["features"][func_name] = {}
		feature_collection["features"][func_name]["embs"] = embs
		feature_collection["features"][func_name]["embs_cols"] = emb_cols
		return feature_collection


	def build_basic_expansion(self, feature_collection, G, config, func_name, node_samples, graph_id):
		emb_dim = config["emb_dim"]
		embs = self.node_emb_engine.run_expansion_embedding(G, emb_dim, node_samples)
		emb_cols = []
		for col in embs.columns.tolist():
			if "emb" in col:
				emb_cols.append(col)
		embs.insert(1, "graph_id", graph_id)
		feature_collection["features"][func_name] = {}
		feature_collection["features"][func_name]["embs"] = embs
		feature_collection["features"][func_name]["embs_cols"] = emb_cols
		return feature_collection



		

		

