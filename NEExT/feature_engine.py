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
from NEExT.helper_functions import get_nodes_x_hops_away
from NEExT.node_embedding_engine import Node_Embedding_Engine


class Feature_Engine:


	def __init__(self, global_config):
		self.global_config = global_config
		self.node_emb_engine = Node_Embedding_Engine()
		# Feature functions
		self.features = {}
		self.features["lsme"] = self.compute_lsme
		self.features["self_walk"] = self.compute_self_walk
		self.features["basic_expansion"] = self.compute_basic_expansion
		self.features["basic_node_features"] = self.compute_basic_node_features
		self.features["page_rank"] = self.compute_structural_node_features
		self.features["degree_centrality"] = self.compute_structural_node_features
		self.features["closeness_centrality"] = self.compute_structural_node_features
		self.features["load_centrality"] = self.compute_structural_node_features
		self.features["eigenvector_centrality"] = self.compute_structural_node_features


	def get_list_of_graph_features(self):
		return list(self.features.keys())


	def compute_feature(self, g_obj, feat_name, feat_vect_len):
		g_obj.computed_features.add(feat_name)
		self.features[feat_name](g_obj, feat_vect_len, feat_name)


	def compute_lsme(self, g_obj, feat_vect_len, func_name):
		graph_id = g_obj.graph_id
		feats = self.node_emb_engine.run_lsme_embedding(G, feat_vect_len)
		feat_cols = []
		for col in feats.columns.tolist():
			if "feat" in col:
				feat_cols.append(col)
		feats.insert(1, "graph_id", graph_id)
		g_obj.feature_collection["features"][func_name] = {}
		g_obj.feature_collection["features"][func_name]["feats"] = feats
		g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


	def compute_self_walk(self, g_obj, feat_vect_len, func_name):
		graph_id = g_obj.graph_id
		G = g_obj.graph
		iG = ig.Graph.from_networkx(G)
		A = np.array(iG.get_adjacency().data)
		Ao = copy.deepcopy(A)
		feats = {}
		for i in range(2, feat_vect_len+2):
			A = np.linalg.matrix_power(Ao, i)
			diag_elem = np.diag(A)
			feats["feat_selfwalk_"+str(i-2)] = list(diag_elem)
		feats = pd.DataFrame(feats)
		feat_cols = list(feats.columns)
		feats.insert(0, "node_id", list(G.nodes))
		feats.insert(1, "graph_id", graph_id)
		g_obj.feature_collection["features"][func_name] = {}
		g_obj.feature_collection["features"][func_name]["feats"] = feats
		g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


	def compute_basic_expansion(self, g_obj, feat_vect_len, func_name):
		graph_id = g_obj.graph_id
		G = g_obj.graph
		feats = self.node_emb_engine.run_expansion_embedding(G, feat_vect_len)
		feat_cols = []
		for col in feats.columns.tolist():
			if "feat" in col:
				feat_cols.append(col)
		feats.insert(1, "graph_id", graph_id)
		g_obj.feature_collection["features"][func_name] = {}
		g_obj.feature_collection["features"][func_name]["feats"] = feats
		g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


	def compute_basic_node_features(self, g_obj, feat_vect_len, func_name):
		graph_id = g_obj.graph_id
		G = g_obj.graph
		node_feature_list = []
		feat_cols = None
		for node in G.nodes:
			feat_cols = list(G.nodes[node].keys())
			node_feature_list.append(list(G.nodes[node].values()))
		feats = pd.DataFrame(node_feature_list)
		feat_cols = ["feat_"+i for i in feat_cols]
		feats.columns = feat_cols
		feats.insert(0, "node_id", list(G.nodes))
		feats.insert(1, "graph_id", graph_id)
		g_obj.feature_collection["features"][func_name] = {}
		g_obj.feature_collection["features"][func_name]["feats"] = feats
		g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


	def compute_structural_node_features(self, g_obj, feat_vect_len, func_name):
		"""
			This method will compute structural node feature for every node up to 
			emb_dim hops away neighbors.
		"""
		graph_id = g_obj.graph_id
		G = g_obj.graph
		if func_name == "page_rank":
			srtct_feat = nx.pagerank(G, alpha=0.9)
		elif func_name == "degree_centrality":
			srtct_feat = nx.degree_centrality(G)
		elif func_name == "closeness_centrality":
			srtct_feat = nx.closeness_centrality(G)
		elif func_name == "load_centrality":
			srtct_feat = nx.load_centrality(G)
		elif func_name == "eigenvector_centrality":
			srtct_feat = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-03)
		else:
			raise ValueError("The selected structural feature is not supported.")
		nodes = []
		features = []
		for node in list(G.nodes):
			nodes.append(node)
			feat_vect = []
			nbs = get_nodes_x_hops_away(G, node, max_hop_length=feat_vect_len)
			feat_vect.append(srtct_feat[node])
			for i in range(1, feat_vect_len):
				if i in nbs:
					nbs_pr = [srtct_feat[j] for j in nbs[i]]
					feat_vect.append(sum(nbs_pr)/len(nbs_pr))
				else:
					feat_vect.append(0.0)
			features.append(feat_vect)
		feats = pd.DataFrame(features)
		feat_cols = ["feat_"+func_name+"_"+str(i) for i in range(feats.shape[1])]
		feats.columns = feat_cols
		feats.insert(0, "node_id", nodes)
		feats.insert(1, "graph_id", graph_id)
		g_obj.feature_collection["features"][func_name] = {}
		g_obj.feature_collection["features"][func_name]["feats"] = feats
		g_obj.feature_collection["features"][func_name]["feats_cols"] = feat_cols


	def pool_features(self, g_obj, pool_method="concat"):
		"""
			This method will use the features built on the graph to construct
			a global embedding for the nodes of the graph.
		"""
		pooled_features = pd.DataFrame()		
		if pool_method == "concat":
			for feat_name in list(g_obj.computed_features):
				features = g_obj.feature_collection["features"][feat_name]["feats"]
				if pooled_features.empty:
					pooled_features = features.copy(deep=True)
				else:
					pooled_features = pooled_features.merge(features, on=["node_id", "graph_id"], how="inner")
		else:
			raise ValueError("Pooling type is not supported.")
		g_obj.feature_collection["pooled_features"] = pooled_features






			














		

		

