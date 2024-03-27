import pandas as pd
from NEExT.NEExT import NEExT
from tqdm import tqdm

res = pd.DataFrame()
for samp_rate in tqdm(range(5, 105, 5)):

	edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/edge_file.csv"
	graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/graph_label_mapping_file.csv"
	node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/t1_classes/info/node_graph_mapping_file.csv"


	nxt = NEExT(quiet_mode="on")

	nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)

	nxt.build_node_sample_collection(sample_rate=samp_rate/100)

	nxt.compute_graph_feature(feat_name="self_walk", feat_vect_len=3)
	nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="lsme", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="basic_expansion", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="closeness_centrality", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="load_centrality", feat_vect_len=3)
	# nxt.compute_graph_feature(feat_name="eigenvector_centrality", feat_vect_len=3)

	nxt.pool_graph_features(pool_method="concat")

	nxt.build_graph_embedding(emb_dim_len=6, emb_engine="approx_wasserstein")

	df = nxt.get_graph_embeddings()

	df["sample_rate"] = samp_rate/100
	if res.empty:
		res = df.copy(deep=True)
	else:
		res = pd.concat([res, df])
	
res.to_csv("pre_feature_creation_sampling_synthetic.csv", index=False)