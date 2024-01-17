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
from scipy.stats import wasserstein_distance
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

	def discard_all_graph_features(self):
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Discarding features", disable=self.global_config.quiet_mode):
			for feat in g_obj.computed_features:
				g_obj.feature_collection["features"].pop(feat)
			g_obj.computed_features = set()	
	
	def discard_graph_feature(self, feat_name):
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Discarding features", disable=self.global_config.quiet_mode):
			g_obj.feature_collection["features"].pop(feat_name)
			g_obj.computed_features.remove(feat_name)

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
		feat_cols = ["feat_"+str(i) for i in range(data.shape[1])]
		data.columns = feat_cols
		data.insert(0, "node_id", self.graph_c.global_feature_vector["node_id"])
		data.insert(1, "graph_id", self.graph_c.global_feature_vector["graph_id"])
		self.graph_c.global_feature_vector_cols = feat_cols[:]
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


	def get_feature_importance_classification_technique(self, emb_engine="approx_wasserstein", sample_size=15, balance_classes=True):
		res_df = pd.DataFrame()
		for col in self.graph_c.global_feature_vector_cols:
			graph_feat_cols = [col]
			graph_embedding, graph_embedding_df = self.g_emb.build_graph_embedding(1, emb_engine, self.graph_c, graph_feat_cols)
			data_obj = self.format_data_for_classification(graph_embedding_df)
			ml_model_results = self.ml_model.build_classification_model(data_obj, sample_size, balance_classes)
			df = pd.DataFrame(ml_model_results)
			df["feature"] = col
			if res_df.empty:
				res_df = df.copy(deep=True)
			else:
				res_df = pd.concat([res_df, df])
		

	def build_classification_model(self, sample_size=50, balance_classes=False):
		graph_emb = self.graph_embedding["graph_embedding_df"]
		data_obj = self.format_data_for_classification(graph_emb)
		self.ml_model_results = self.ml_model.build_classification_model(data_obj, sample_size, balance_classes)


	def format_data_for_classification(self, graph_emb):
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


	def compute_feat_variability(self, plot_results=False):
		graph_ids = self.graph_c.global_feature_vector["graph_id"].unique().tolist()
		wd_list = []
		feat_list = []
		for feat_col_name in self.graph_c.global_feature_vector_cols:
			ref_feat = self.graph_c.global_feature_vector[self.graph_c.global_feature_vector["graph_id"] == graph_ids[0]][feat_col_name].values
			for i in range(1, len(graph_ids)):
				feat = self.graph_c.global_feature_vector[self.graph_c.global_feature_vector["graph_id"] == graph_ids[i]][feat_col_name].values
				wd = wasserstein_distance(ref_feat, feat)
				wd_list.append(wd)
				feat_list.append(feat_col_name)
		res_df = pd.DataFrame()
		res_df["Feature Name"] = feat_list
		res_df["Feature Variability Score"] = wd_list
		res_df = pd.DataFrame(res_df.groupby(by=["Feature Name"])["Feature Variability Score"].std()).reset_index()
		res_df.sort_values(by=["Feature Variability Score"], ascending=False, inplace=True)
		if plot_results:
			x = res_df["Feature Name"].values
			y = res_df["Feature Variability Score"].values
			plt.bar(x, y)
			plt.xlabel("Feature Name")
			plt.ylabel("Variability Score")
			plt.xticks(rotation=45)
			plt.show()
		return res_df