"""
    Author : Ash Dehghan
"""

import copy

import numpy as np
import pandas as pd
import plotly.express as px

# External Libraries
import umap
from bayes_opt import BayesianOptimization
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from NEExT import feature_engine, graph_embedding_engine
from NEExT.global_config import Global_Config
from NEExT.graph_collection import Graph_Collection

# Internal Modules
from NEExT.ml_models import ML_Models


class NEExT:

    def __init__(self, quiet_mode="off"):
        self.global_config = Global_Config()
        self.global_config.set_output_mode(quiet_mode)
        self.graph_c = Graph_Collection(self.global_config)
        self.ml_model = ML_Models(self.global_config)
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

    def build_node_sample_collection(self, sample_rate):
        self.graph_c.build_node_sample_collection(sample_rate=sample_rate)

    def get_global_feature_vector(self):
        return self.graph_c.global_feature_vector

    def get_archived_global_feature_vector(self):
        return self.graph_c.global_feature_vector_arc

    def get_list_of_graph_features(self):
        return feature_engine.get_list_of_graph_features()

    def compute_graph_feature(self, feat_name, feat_vect_len):
        for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features", disable=self.global_config.quiet_mode):
            feature_engine.compute_feature(g_obj, feat_name, feat_vect_len)

    def load_custom_node_feature_function(self, function, function_name):
        feature_engine.load_function(function, function_name)

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
            feature_engine.pool_features(g_obj, pool_method)
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
        self.graph_c.global_feature_vector_original = self.graph_c.global_feature_vector.copy(deep=True)
        self.graph_c.global_feature_vector_cols = feat_cols
        # Re-assign global embeddings to each graph
        for g_obj in tqdm(self.graph_c.graph_collection, desc="Updating features", disable=self.global_config.quiet_mode):
            df = feat_df[feat_df["graph_id"] == g_obj.graph_id].copy(deep=True)
            g_obj.feature_collection["pooled_features"] = df

    def apply_dim_reduc_to_graph_feats(self, dim_size, reducer_type):
        """
        This method will apply dimensionality reduction to the
        global feature embeddings. Since many of the processses in UGAF
        require the use of the global feature embeddings and their embedding
        columns, this function makes a copy of those to keep for record
        and will replace the main feat embedding DataFrame and columns
        with the reduced ones.
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
        feat_cols = ["feat_" + str(i) for i in range(data.shape[1])]
        data.columns = feat_cols
        data.insert(0, "node_id", self.graph_c.global_feature_vector["node_id"])
        data.insert(1, "graph_id", self.graph_c.global_feature_vector["graph_id"])
        self.graph_c.global_feature_vector_cols = feat_cols[:]
        self.graph_c.global_feature_vector = data.copy(deep=True)
        self.graph_c.global_feature_vector_original = self.graph_c.global_feature_vector.copy(deep=True)

    def get_list_of_graph_embedding_engines(self):
        return graph_embedding_engine.get_list_of_graph_embedding_engines()

    def get_graph_embeddings(self):
        return self.graph_embedding["graph_embedding_df"]

    def build_graph_embedding(self, emb_dim_len, emb_engine):
        """
        This method uses the Graph Embedding Engine object to
        build a graph embedding for every graph in the graph collection.
        """
        graph_embedding, graph_embedding_df = graph_embedding_engine.build_graph_embedding(emb_dim_len, emb_engine, self.graph_c)

        self.graph_embedding["graph_embedding"] = graph_embedding
        self.graph_embedding["graph_embedding_df"] = graph_embedding_df

    # def compute_feat_variability(self):
    #     graph_ids = self.graph_c.global_feature_vector["graph_id"].unique().tolist()
    #     wd_list = []
    #     feat_list = []
    #     graph_id_i = []
    #     graph_id_j = []
    #     for feat_col_name in tqdm(self.graph_c.global_feature_vector_cols,
    #                               desc="Standardizing features",
    #                               disable=self.global_config.quiet_mode):

    #         for i in range(0, len(graph_ids)):
    #             ref_feat = self.graph_c.global_feature_vector[
    #                 self.graph_c.global_feature_vector["graph_id"] == graph_ids[i]
    #                 ][feat_col_name].values

    #             for j in range(0, len(graph_ids)):
    #                 feat = self.graph_c.global_feature_vector[
    #                     self.graph_c.global_feature_vector["graph_id"] == graph_ids[j]
    #                     ][feat_col_name].values
    #                 wd = wasserstein_distance(ref_feat, feat)
    #                 wd_list.append(wd)
    #                 feat_list.append(feat_col_name)
    #                 graph_id_i.append(i)
    #                 graph_id_j.append(j)

    #     res_df = pd.DataFrame()
    #     res_df["graph_id_i"] = graph_id_i
    #     res_df["graph_id_j"] = graph_id_j
    #     res_df["feature_name"] = feat_list
    #     res_df["score"] = wd_list
    #     # res_df = pd.DataFrame(
    #     #     res_df.groupby(by=["Feature Name"])[
    #     #         "Feature Variability Score"
    #     #         ].std()
    #     #     ).reset_index()
    #     # res_df.sort_values(by=["Feature Variability Score"], ascending=False, inplace=True)

    #     return res_df

    # {'target': 0.8447368421052632, 'params': {'feat_basic_expansion_0': 0.39676747423066994, 'feat_page_rank_0': 0.538816734003357}}
    # {'target': 0.8421052631578949, 'params': {'feat_basic_expansion_0': 0.20050017663542263, 'feat_page_rank_0': 0.6165392570147974}}

    def black_box_function(self, **kwargs):
        # sum_of_w = sum(list(kwargs.values()))
        self.graph_c.global_feature_vector = self.graph_c.global_feature_vector_original.copy(deep=True)
        for col in self.graph_c.global_feature_vector_cols:
            w = kwargs[col]
            # if w <= 0.5:
            #     w = 0
            # else:
            #     w = 1
            self.graph_c.global_feature_vector[col] *= w

        emb_dim_len = len(self.graph_c.global_feature_vector_cols)
        _, graph_embedding_df = graph_embedding_engine.build_graph_embedding(graph_c=self.graph_c, emb_dim_len=1, emb_engine="approx_wasserstein")
        data_obj = self.format_data_for_classification(graph_embedding_df)
        ml_model_results = self.ml_model.build_classification_model(data_obj, 10, True)
        metric = np.mean(np.array(ml_model_results["accuracy"]))
        return metric

    def get_supervised_feature_importance(self):
        pbounds = {}
        for col in self.graph_c.global_feature_vector_cols:
            pbounds[col] = (0, 1)
        optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(
            init_points=2,
            n_iter=20,
        )
        print(optimizer.max)

    def get_feature_importance_classification_technique(self, emb_engine="approx_wasserstein", sample_size=20, balance_classes=True, inverse=False):
        """
            The way this function finds the feature importance is in a greety way.
            Only if the added feature improves the model accuracy, then it gets added.
        """
        accuracy_contribution = []
        accuracy_contribution_std = []
        selected_features = []

        while len(selected_features) < len(self.graph_c.global_feature_vector_cols):

            col_list = []
            accuracy_mean_list = []
            accuracy_std_list = []
            for col in self.graph_c.global_feature_vector_cols:

                if col not in selected_features:
                    feats = selected_features[:]
                    feats.append(col)

                    graph_embedding, graph_embedding_df = graph_embedding_engine.build_graph_embedding(
                        emb_dim_len=len(feats),
                        emb_engine=emb_engine,
                        graph_c=self.graph_c,
                        graph_feat_cols=feats)


                    data_obj = self.format_data_for_classification(graph_embedding_df)
                    ml_model_results = self.ml_model.build_classification_model(data_obj, sample_size, balance_classes)
                    
                    mean_accuracy = np.mean(np.array(ml_model_results["accuracy"]))
                    std_accuracy = np.std(np.array(ml_model_results["accuracy"]))
                    col_list.append(col)
                    accuracy_mean_list.append(mean_accuracy)
                    accuracy_std_list.append(std_accuracy)

            if inverse:
                max_accuracy_val = min(accuracy_mean_list)
            else:
                max_accuracy_val = max(accuracy_mean_list)
            selected_feat = col_list[accuracy_mean_list.index(max_accuracy_val)]
            std_accuracy_val = accuracy_std_list[accuracy_mean_list.index(max_accuracy_val)]
            print(selected_feat, max_accuracy_val)
            
            accuracy_contribution.append(max_accuracy_val)
            accuracy_contribution_std.append(std_accuracy_val)
            selected_features.append(selected_feat)
            
        return selected_features, accuracy_contribution, accuracy_contribution_std
      



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

    def visualize_graph_embedding(self, color_by="nothing", color_target_type="classes", dim_reduction="UMAP"):
        """
        This method uses the the graph embedding and UMAP to
        visualize the embeddings in two dimensions. It can also color the
        points if there are labels available for the graph.
        """
        if color_by == "graph_label":
            data = self.graph_embedding["graph_embedding_df"].merge(self.graph_c.grpah_labels_df, on="graph_id", how="inner")
            data.rename(columns={"graph_label": "Graph Label"}, inplace=True)
            if color_target_type == "classes":
                data["Graph Label"] = data["Graph Label"].astype(str)
            elif color_target_type == "continuous":
                data["Graph Label"] = data["Graph Label"].astype(float)
        elif color_by == "similarity_matrix_mean":
            data = self.graph_embedding["graph_embedding_df"].merge(self.similarity_matrix_stats["data"], on="graph_id", how="inner")
            data.rename(columns={"similarity_matrix_mean": "Similarity Matrix Mean"}, inplace=True)
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
        if dim_reduction == "UMAP":
            reducer = umap.UMAP()
            redu_emb = reducer.fit_transform(data[emb_cols])
            data["x"] = redu_emb[:, 0]
            data["y"] = redu_emb[:, 1]
        elif dim_reduction == "first_two":
            data["x"] = data[emb_cols[0]]
            data["y"] = data[emb_cols[1]]
        # TODO add t-sne
        # TODO add PCA
        # Generate plotly figures
        if color_by == "graph_label":
            fig = px.scatter(data, x="x", y="y", color="Graph Label", size=[4] * len(data))
        elif color_by == "similarity_matrix_mean":
            fig = px.scatter(data, x="x", y="y", color="Similarity Matrix Mean", size=[4] * len(data))
        elif color_by == "nothing":
            fig = px.scatter(data, x="x", y="y", size=[4] * len(data))
        else:
            raise ValueError("Selected coloring is not supported.")

        # Update figure layout
        fig.update_layout(paper_bgcolor="white")
        fig.update_layout(plot_bgcolor="white")
        fig.update_yaxes(color="black")
        fig.update_layout(
            yaxis=dict(title="Dim-1", zeroline=True, showline=True, linecolor="black", mirror=True, linewidth=2),
            xaxis=dict(title="Dim-2", mirror=True, zeroline=True, showline=True, linecolor="black", linewidth=2),
            width=600,
            height=500,
            font=dict(size=15, color="black"),
        )
        fig.update_layout(showlegend=True)
        fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor="#e3e1e1")
        fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor="grey")
        fig.update_traces(marker_line_color="black", marker_line_width=1.5, opacity=0.6)
        return fig, data
