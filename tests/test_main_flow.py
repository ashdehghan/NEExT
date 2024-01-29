import sys
sys.path.append("../")

from NEExT.NEExT import NEExT


def test_readme_code():
    """ 
    Test the code in the README.md file. Does not check for correctness, just that the code runs without errors.
    """
    edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/edge_file.csv"
    graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/graph_label_mapping_file.csv"
    node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/node_graph_mapping_file.csv"
    nxt = NEExT(quiet_mode="on")
    nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)
    nxt.get_list_of_graph_features()
    nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=4)
    nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=4)
    nxt.pool_graph_features(pool_method="concat")
    df = nxt.get_global_feature_vector()
    nxt.apply_dim_reduc_to_graph_feats(dim_size=4, reducer_type="pca")
    df = nxt.get_global_feature_vector()
    df = nxt.get_archived_global_feature_vector()
    nxt.get_list_of_graph_embedding_engines()
    nxt.build_graph_embedding(emb_dim_len=3, emb_engine="approx_wasserstein")
    df = nxt.get_graph_embeddings()

    return True
