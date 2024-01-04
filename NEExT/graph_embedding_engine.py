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

# Internal Modules
# from ugaf.global_config import Global_Config

class Graph_Embedding_Engine:


	def __init__(self, global_config):
		self.global_config = global_config


	def build_graph_embedding(self, graph_c):

		graph_embedding_type = self.global_config.config["graph_embedding"]["embedding_type"]

		if graph_embedding_type == "approx_wasserstein":
			graphs_embed, graph_embedding_df = self.build_approx_wasserstein_graph_embedding(graph_c)
		elif graph_embedding_type == "wasserstein":
			graphs_embed, graph_embedding_df = self.build_wasserstein_graph_embedding(graph_c)
		elif graph_embedding_type == "sinkhornvectorizer":
			graphs_embed, graph_embedding_df = self.build_sinkhornvectorizer_graph_embedding(graph_c)
		else:
			raise ValueError("Graph embedding type selected is not supported.")
		return graphs_embed, graph_embedding_df


	def build_approx_wasserstein_graph_embedding(self, graph_c):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		n = graph_c.total_numb_of_nodes
		rows = graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_embeddings[graph_c.global_embeddings_cols].values
		if self.global_config.config["graph_embedding"]["graph_emb_dim"] == "auto":
			graph_emb_dim = graph_c.global_embeddings[graph_c.global_embeddings_cols].shape[1]
		else:
			graph_emb_dim = self.global_config.config["graph_embedding"]["graph_emb_dim"]
		graph_ids = graph_c.global_embeddings["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
			n_components=graph_emb_dim
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df


	def build_wasserstein_graph_embedding(self, graph_c):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		n = graph_c.total_numb_of_nodes
		rows = graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_embeddings[graph_c.global_embeddings_cols].values
		if self.global_config.config["graph_embedding"]["graph_emb_dim"] == "auto":
			graph_emb_dim = graph_c.global_embeddings[graph_c.global_embeddings_cols].shape[1]
		else:
			graph_emb_dim = self.global_config.config["graph_embedding"]["graph_emb_dim"]
		graph_ids = graph_c.global_embeddings["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.WassersteinVectorizer(
			memory_size="4G",
			random_state=42,
			n_components=graph_emb_dim
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df


	def build_sinkhornvectorizer_graph_embedding(self, graph_c):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		n = graph_c.total_numb_of_nodes
		rows = graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		embedding_collection = graph_c.global_embeddings[graph_c.global_embeddings_cols].values
		if self.global_config.config["graph_embedding"]["graph_emb_dim"] == "auto":
			graph_emb_dim = graph_c.global_embeddings[graph_c.global_embeddings_cols].shape[1]
		else:
			graph_emb_dim = self.global_config.config["graph_embedding"]["graph_emb_dim"]
		graph_ids = graph_c.global_embeddings["graph_id"].unique().tolist()
		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.SinkhornVectorizer(
			memory_size="4G",
			random_state=42,
			n_components=graph_emb_dim
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df

# se_vectorizer = vectorizers.SinkhornVectorizer(
#     n_components=160,
#     memory_size="4G",
#     random_state=42,
# )
# se_vectors = se_vectorizer.fit_transform(info_doc_matrix, vectors=word_vectors)

