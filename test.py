from NEExT.NEExT import NEExT

edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/edge_file.csv"
graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/graph_label_mapping_file.csv"
node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/node_graph_mapping_file.csv"

nxt = NEExT(quiet_mode="on")

nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)

nxt.compute_graph_feature(feat_name="basic_expansion", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="self_walk", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="closeness_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="load_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="eigenvector_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="lsme", feat_vect_len=4)


nxt.pool_graph_features(pool_method="concat")

selected_features, accuracy_contribution, accuracy_contribution_std = nxt.get_feature_importance_classification_technique()

res_df = pd.DataFrame()
res_df["selected_features"] = selected_features
res_df["accuracy_contribution"] = accuracy_contribution
res_df["accuracy_contribution_std"] = accuracy_contribution_std

res_df.to_csv("../results.csv", index=False)

# nxt.build_graph_embedding(emb_dim_len=2, emb_engine="approx_wasserstein")

# print(nxt.graph_embedding["graph_embedding_df"])

# nxt.get_supervised_feature_importance()