"""
	Author : Ash Dehghan
"""

# External Libraries
import umap
import scipy
import vectorizers
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import plotly.express as px
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Internal Modules
from NEExT.ml_models import ML_Models
from NEExT.global_config import Global_Config
from NEExT.feature_engine import Feature_Engine
from NEExT.helper_functions import divide_chunks
from NEExT.graph_collection import Graph_Collection
from NEExT.graph_embedding_engine import Graph_Embedding_Engine


class NEExT:


	def __init__(self, quiet_mode="off"):
		self.global_config = Global_Config()
		self.global_config.set_output_mode(quiet_mode)
		self.graph_c = Graph_Collection(self.global_config)
		self.feat_eng = Feature_Engine(self.global_config)
		self.ml_model = ML_Models(self.global_config)
		self.g_emb = Graph_Embedding_Engine(self.global_config)
		self.graph_embedding = {}
		self.similarity_matrix_stats = {}
		self.ml_model_results = None


	def load_data_from_csv(self, edge_file, node_graph_mapping_file, node_features_file=None, graph_label_file=None, filter_for_largest_cc=True, reset_node_indices=True):
		"""
			This method uses the Graph Collection class to build an object
			which handels a set of graphs.
		"""
		self.graph_c.load_graphs_from_csv(edge_file, node_graph_mapping_file, node_features_file)
		if filter_for_largest_cc:
			self.graph_c.filter_collection_for_largest_connected_component()
		if reset_node_indices:
			self.graph_c.reset_node_indices()
		if graph_label_file:
			self.graph_c.assign_graph_labels_from_csv(graph_label_file)
		

	def get_global_feature_vector(self):
		return self.graph_c.global_feature_vector


	def get_archived_global_feature_vector(self):
		return self.graph_c.global_feature_vector_arc

		
	def get_list_of_graph_features(self):
		return self.feat_eng.get_list_of_graph_features()


	def compute_graph_feature(self, feat_name, feat_vect_len):
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features", disable=self.global_config.quiet_mode):			
			self.feat_eng.compute_feature(g_obj, feat_name, feat_vect_len)


	def pool_graph_features(self, pool_method="concat"):
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Pooling features", disable=self.global_config.quiet_mode):
			self.feat_eng.pool_features(g_obj, pool_method)
		self.standardize_graph_features_globaly()


	def standardize_graph_features_globaly(self):
		"""
			This method will standardize the graph features across all graphs.
		"""
		all_graph_feats = pd.DataFrame()
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Standardizing features", disable=self.global_config.quiet_mode):
			df = g_obj.feature_collection["pooled_features"]
			if all_graph_feats.empty:
				all_graph_feats = df.copy(deep=True)
			else:
				all_graph_feats = pd.concat([all_graph_feats, df])

		# Grab graph feature embedding columns
		feat_cols = []
		for col in all_graph_feats.columns.tolist():
			if "feat_" in col:
				feat_cols.append(col)
		# Standardize the embedding
		node_ids = all_graph_feats["node_id"].tolist()
		graph_ids = all_graph_feats["graph_id"].tolist()
		feat_df = all_graph_feats[feat_cols].copy(deep=True)
		# Normalize data
		scaler = StandardScaler()
		feat_df = pd.DataFrame(scaler.fit_transform(feat_df))
		feat_df.columns = feat_cols
		feat_df.insert(0, "node_id", node_ids)
		feat_df.insert(1, "graph_id", graph_ids)
		# Keep a collective global embedding
		self.graph_c.global_feature_vector = feat_df.copy(deep=True)
		self.graph_c.global_feature_vector_cols = feat_cols
		# Re-assign global embeddings to each graph
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Updating features", disable=self.global_config.quiet_mode):
			df = feat_df[feat_df["graph_id"] == g_obj.graph_id].copy(deep=True)
			g_obj.feature_collection["pooled_features"] = df


	def apply_dim_reduc_to_graph_feats(self, dim_size, reducer_type):
		"""
			This method will apply dimensionality reduction to the gloabl feature embeddings.
			Since many of the processses in UGAF require the use of the global feat embeddings
			and their embedding columns, this function makes a copy of those to keep for record
			and will replace the main feat embedding DataFrame and columns with the reduced ones.
		"""
		if dim_size >= len(self.graph_c.global_feature_vector_cols):
			if not self.global_config.quiet_mode:
				print("The number of reduced dimension is > to actual dimensions.")
			return
		# Make copies
		self.graph_c.global_feature_vector_cols_arc = self.graph_c.global_feature_vector_cols[:]
		self.graph_c.global_feature_vector_arc = self.graph_c.global_feature_vector.copy(deep=True)
		data = self.graph_c.global_feature_vector[self.graph_c.global_feature_vector_cols]
		if reducer_type == "umap":		
			reducer = umap.UMAP(n_components=dim_size)
			data = reducer.fit_transform(data)
		elif reducer_type == "pca":
			reducer = PCA(n_components=dim_size)
			data = reducer.fit_transform(data)
		else:
			raise ValueError("Wrong reducer selected.")
		scaler = StandardScaler()
		data = scaler.fit_transform(data)
		data = pd.DataFrame(data)
		emb_cols = ["emb_"+str(i) for i in range(data.shape[1])]
		data.columns = emb_cols
		data.insert(0, "node_id", self.graph_c.global_feature_vector["node_id"])
		data.insert(1, "graph_id", self.graph_c.global_feature_vector["graph_id"])
		self.graph_c.global_feature_vector_cols = emb_cols[:]
		self.graph_c.global_feature_vector = data.copy(deep=True)


	def get_list_of_graph_embedding_engines(self):
		return self.g_emb.get_list_of_graph_embedding_engines()


	def get_graph_embeddings(self):
		return self.graph_embedding["graph_embedding_df"]


	def build_graph_embedding(self, emb_dim_len, emb_engine):
		"""
			This method uses the Graph Embedding Engine object to 
			build a graph embedding for every graph in the graph collection.
		"""
		graph_embedding, graph_embedding_df = self.g_emb.build_graph_embedding(emb_dim_len, emb_engine, self.graph_c)
		self.graph_embedding["graph_embedding"] = graph_embedding
		self.graph_embedding["graph_embedding_df"] = graph_embedding_df


	def compute_similarity_matrix_stats(self):
		"""
			This method will run through the features computes on the graph and computes
			similarity matrices on those features per graph.
		"""
		res_df = pd.DataFrame()
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Computing similarity stats", disable=self.global_config.quiet_mode):
			feature_list = []
			eigen_val_list = []
			sim_mean_list = []			
			for feature in g_obj["graph_features"]["features"]:
				
				data = self.graph_c.global_feature_vector[self.graph_c.global_feature_vector["graph_id"] == g_obj["graph_id"]]
				embs_cols = g_obj["graph_features"]["features"][feature]["embs_cols"]
				
				data = data[embs_cols].values
			
				sim_matrix = cosine_similarity(data, data)

				eigenvalues, eigenvectors = LA.eig(sim_matrix)
				eigenvalues = [i.real for i in eigenvalues]

				max_ei = max(eigenvalues)

				feature_list.append(feature)
				eigen_val_list.append(max_ei)
				sim_mean_list.append(sim_matrix.mean())

			df = pd.DataFrame()
			df["feature"] = feature_list
			df["largest_eigen_value"] = eigen_val_list
			df["similarity_matrix_mean"] = sim_mean_list
			df.insert(0, "graph_id", g_obj["graph_id"])

			if res_df.empty:
				res_df = df.copy(deep=True)
			else:
				res_df = pd.concat([res_df, df])

		self.similarity_matrix_stats["data"] = res_df
		self.similarity_matrix_stats["metrics"] = ["largest_eigen_value", "similarity_matrix_mean"]
		self.similarity_matrix_stats["metrics_pretty_name"] = ["Largest EigenValue", "Similarity Matrix Mean"]



	def build_model(self):
		data_obj = self.format_data_for_classification()
		self.ml_model_results = self.ml_model.build_model(data_obj)


	def format_data_for_classification(self):
		graph_emb = self.graph_embedding["graph_embedding_df"]
		data = self.graph_c.grpah_labels_df.merge(graph_emb, on="graph_id")
		x_cols = []
		for col in data.columns.tolist():
			if "emb" in col:
				x_cols.append(col)
		data_obj = {}
		data_obj["data"] = data
		data_obj["x_cols"] = x_cols
		data_obj["y_col"] = "graph_label"
		return data_obj


	def visualize_graph_embedding(self, color_by="nothing", color_target_type="classes"):
		"""
			This method uses the the graph embedding and UMAP to
			visulize the embeddings in two dimensions. It can also color the
			points if there are labels available for the graph.
		"""
		if color_by == "graph_label":
			data = self.graph_embedding["graph_embedding_df"].merge(self.graph_c.grpah_labels_df, on="graph_id", how="inner")
			data.rename(columns={"graph_label":"Graph Label"}, inplace=True)
			if color_target_type == "classes":
				data["Graph Label"] = data["Graph Label"].astype(str)
		elif color_by == "similarity_matrix_mean":
			data = self.graph_embedding["graph_embedding_df"].merge(self.similarity_matrix_stats["data"], on="graph_id", how="inner")
			data.rename(columns={"similarity_matrix_mean":"Similarity Matrix Mean"}, inplace=True)
			if color_target_type == "classes":
				data["Graph Label"] = data["Graph Label"].astype(str)
		elif color_by == "nothing":
			data = self.graph_embedding["graph_embedding_df"].copy(deep=True)
		else:
			raise ValueError("Selected coloring is not supported.")
			
		# Identify embedding colomns
		emb_cols = []
		for col in data.columns.tolist():
			if "emb" in col:
				emb_cols.append(col)
		# Perform dimensionality reduction
		reducer = umap.UMAP()
		redu_emb = reducer.fit_transform(data[emb_cols])
		data["x"] = redu_emb[:,0]
		data["y"] = redu_emb[:,1]
		# Generate plotly figures
		if color_by == "graph_label":
			fig = px.scatter(data, x="x", y="y", color="Graph Label", size=[4]*len(data))		
		elif color_by == "similarity_matrix_mean":
			fig = px.scatter(data, x="x", y="y", color="Similarity Matrix Mean", size=[4]*len(data))
		elif color_by == "nothing":
			fig = px.scatter(data, x="x", y="y", size=[4]*len(data))
		else:
			raise ValueError("Selected coloring is not supported.")

		# Update figure layout
		fig.update_layout(paper_bgcolor='white')
		fig.update_layout(plot_bgcolor='white')
		fig.update_yaxes(color='black')
		fig.update_layout(
			yaxis = dict(
				title = "Dim-1",
				zeroline=True,
				showline = True,
				linecolor = 'black',
				mirror=True,
				linewidth = 2
			),
			xaxis = dict(
				title = 'Dim-2',
				mirror=True,
				zeroline=True,
				showline = True,
				linecolor = 'black',
				linewidth = 2,
			),
			width=600,
			height=500,
			font=dict(
			size=15,
			color="black")
				
		)
		fig.update_layout(showlegend=True)
		fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='#e3e1e1')
		fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='grey')
		fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.6)
		fig.show()


	def visualize_similarity_matrix_stats(self, color_by_label=False):

		for idx, metric in enumerate(self.similarity_matrix_stats["metrics"]):

			y_name = self.similarity_matrix_stats["metrics_pretty_name"][idx]

			if color_by_label:
				data = self.similarity_matrix_stats["data"].merge(self.graph_c.grpah_labels_df, on="graph_id", how="inner")
			else:
				data = self.similarity_matrix_stats["data"].copy(deep=True)

			# Generate eigen value plotly figures
			if color_by_label:
				fig = px.scatter(data, x="graph_label", y=metric)
				x_name = "Graph Label"
			else:
				fig = px.scatter(data, x="graph_id", y=metric)
				x_name = "Graph ID"

			# Update figure layout
			fig.update_layout(paper_bgcolor='white')
			fig.update_layout(plot_bgcolor='white')
			fig.update_yaxes(color='black')
			fig.update_layout(
				yaxis = dict(
					title = y_name,
					zeroline=True,
					showline = True,
					linecolor = 'black',
					mirror=True,
					linewidth = 2
				),
				xaxis = dict(
					title = x_name,
					mirror=True,
					zeroline=True,
					showline = True,
					linecolor = 'black',
					linewidth = 2,
				),
				width=600,
				height=500,
				font=dict(
				size=15,
				color="black")
					
			)
			fig.update_layout(showlegend=True)
			fig.update_layout(legend=dict(
				yanchor="bottom",
				y=0.01,
				xanchor="left",
				x=0.78,
				bordercolor="Black",
				borderwidth=1
			))
			fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='#e3e1e1')
			fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='grey')
			fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.6)
			return fig, data


