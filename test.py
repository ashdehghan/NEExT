from NEExT.NEExT import NEExT


edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/BZR/processed_data/edge_file.csv"
graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/BZR/processed_data/graph_label_mapping_file.csv"
node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/BZR/processed_data/node_graph_mapping_file.csv"
node_features_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/BZR/processed_data/node_feature_file.csv"

nxt = NEExT()

nxt.load_data_from_csv(edge_file, node_graph_mapping_file, node_features_file, graph_label_file)



nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=4)

nxt.pool_graph_features(pool_method="concat")
nxt.apply_dim_reduc_to_graph_feats(dim_size=4, reducer_type="pca")