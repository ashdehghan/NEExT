from template import Param


experiment = "embedding_combinations"

features = [
    "degree_centrality",
    "betastar",
    "load_centrality",
    "lsme",
    "local_efficiency",
    "betweenness_centrality",
    "closeness_centrality",
]

k_and_length = [(1, 1), (2, 1), (2, 2)]


params = [
    Param(
        comment=f"{strategy}_local_{k_hop}_{local_i}_global_{global_i}",
        global_structural_feature_list=['all'],
        global_feature_vector_length=2,
        local_structural_feature_list=features,
        local_feature_vector_length=local_i,
        egonet_k_hop=k_hop,
        embeddings_strategy=strategy,
    )
    for k_hop, local_i in k_and_length
    for strategy in ['separate_embeddings', 'combined_embeddings']
    for global_i in [1, 2]
]
