import pandas as pd
import numpy as np
from NEExT.NEExT import NEExT


exp_name = "NCI1_Feat_Imp"


edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/edge_file.csv"
graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/graph_label_mapping_file.csv"
node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/NCI1/processed_data/node_graph_mapping_file.csv"

nxt = NEExT(quiet_mode="off")

nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)


nxt.compute_graph_feature(feat_name="lsme", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="basic_expansion", feat_vect_len=4)
# nxt.compute_graph_feature(feat_name="self_walk", feat_vect_len=4)
# nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="closeness_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="load_centrality", feat_vect_len=4)
nxt.compute_graph_feature(feat_name="eigenvector_centrality", feat_vect_len=4)

nxt.pool_graph_features(pool_method="concat")


nxt.build_graph_embedding(emb_dim_len=24, emb_engine="wasserstein")

nxt.build_classification_model(sample_size=100, balance_classes=False)

print(np.mean(np.array(nxt.ml_model_results["accuracy"])))