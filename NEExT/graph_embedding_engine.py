"""
	Author: Ash Dehghan
"""

# External Libraries
import scipy
import random
import vectorizers
import numpy as np
import pandas as pd
import networkx as nx


class Graph_Embedding_Engine:


	def __init__(self, global_config):
		self.global_config = global_config
		self.embedding_engines = {}
		self.embedding_engines["approx_wasserstein"] = self.build_approx_wasserstein_graph_embedding
		self.embedding_engines["wasserstein"] = self.build_wasserstein_graph_embedding
		self.embedding_engines["sinkhornvectorizer"] = self.build_sinkhornvectorizer_graph_embedding


	def get_list_of_graph_embedding_engines(self):
		return list(self.embedding_engines.keys())


	def build_graph_embedding(self, emb_dim_len, emb_engine, graph_c, graph_feat_cols=[]):
		if emb_engine in self.embedding_engines:
			if len(graph_feat_cols) == 0:
				graph_feat_cols = graph_c.global_feature_vector_cols
			graphs_embed, graph_embedding_df = self.embedding_engines[emb_engine](graph_c, emb_dim_len, graph_feat_cols)
		else:
			raise ValueError("Graph embedding type selected is not supported.")
		return graphs_embed, graph_embedding_df


	def build_approx_wasserstein_graph_embedding(self, graph_c, emb_dim_len, graph_feat_cols):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		if graph_c.use_sampled_nodes:
			n = graph_c.total_numb_of_sampled_nodes
		else:
			n = graph_c.total_numb_of_nodes
		
		# since the graph_id_node_array is not necessarily a valid index, we need to map it to a valid index
		rows = graph_c.graph_id_node_array
		mapper = {i : j for j, i in enumerate(np.sort(np.unique(rows)))}
		rows = np.array([mapper[i] for i in rows])

		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_feature_vector[graph_feat_cols].values
		graph_ids = graph_c.global_feature_vector["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
			n_components=emb_dim_len
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df


	def build_wasserstein_graph_embedding(self, graph_c, emb_dim_len, graph_feat_cols):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		if graph_c.use_sampled_nodes:
			n = graph_c.total_numb_of_sampled_nodes
		else:
			n = graph_c.total_numb_of_nodes

		# since the graph_id_node_array is not necessarily a valid index, we need to map it to a valid index
		rows = graph_c.graph_id_node_array
		mapper = {i : j for j, i in enumerate(np.sort(np.unique(rows)))}
		rows = np.array([mapper[i] for i in rows])

		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_feature_vector[graph_feat_cols].values
		graph_ids = graph_c.global_feature_vector["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.WassersteinVectorizer(
			memory_size="4G",
			random_state=42,
			n_components=emb_dim_len
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df


	def build_sinkhornvectorizer_graph_embedding(self, graph_c, emb_dim_len, graph_feat_cols):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		if graph_c.use_sampled_nodes:
			n = graph_c.total_numb_of_sampled_nodes
		else:
			n = graph_c.total_numb_of_nodes

		# since the graph_id_node_array is not necessarily a valid index, we need to map it to a valid index
		rows = graph_c.graph_id_node_array
		mapper = {i : j for j, i in enumerate(np.sort(np.unique(rows)))}
		rows = np.array([mapper[i] for i in rows])

		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_feature_vector[graph_feat_cols].values
		graph_ids = graph_c.global_feature_vector["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.SinkhornVectorizer(
			memory_size="4G",
			random_state=42,
			n_components=emb_dim_len
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df

