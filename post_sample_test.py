import pandas as pd
from tqdm import tqdm
from NEExT.NEExT import NEExT

edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/edge_file.csv"
graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/graph_label_mapping_file.csv"
node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/node_graph_mapping_file.csv"


nxt = NEExT(quiet_mode="on")

nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)

# nxt.build_node_sample_collection(sample_rate=0.1)

nxt.compute_graph_feature(feat_name="self_walk", feat_vect_len=3)
nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="lsme", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="basic_expansion", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="closeness_centrality", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="load_centrality", feat_vect_len=3)
# nxt.compute_graph_feature(feat_name="eigenvector_centrality", feat_vect_len=3)

nxt.pool_graph_features(pool_method="concat")



gloab_feat_vect_copy = nxt.graph_c.global_feature_vector.copy(deep=True)

def g_feat_fix():
	df = pd.DataFrame()
	for g_obj in nxt.graph_c.graph_collection:
		node_samples = g_obj.node_samples

		dfdf = nxt.graph_c.global_feature_vector[
			(nxt.graph_c.global_feature_vector["graph_id"] == g_obj.graph_id)
			&
			(nxt.graph_c.global_feature_vector["node_id"].isin(node_samples))
			].copy(deep=True)
		if df.empty:
			df = dfdf.copy(deep=True)
		else:
			df = pd.concat([df, dfdf])

	nxt.graph_c.global_feature_vector = df.copy(deep=True)

res = pd.DataFrame()
for samp_rate in tqdm(range(5, 101, 1)):

	for batch in range(1, 30):

		nxt.build_node_sample_collection(sample_rate=samp_rate/100)
		g_feat_fix()
		nxt.build_graph_embedding(emb_dim_len=6, emb_engine="approx_wasserstein")

		df = nxt.get_graph_embeddings()
		df["sample_rate"] = samp_rate/100
		df["batch"] = batch
		if res.empty:
			res = df.copy(deep=True)
		else:
			res = pd.concat([res, df])
		nxt.graph_c.global_feature_vector = gloab_feat_vect_copy.copy(deep=True)

	
res.to_csv("post_feature_creation_sampling_synthetic.csv", index=False)
