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
        comment=f"{experiment}_{k}_{i}_{strategy}",
        global_structural_feature_list=['all'],
        global_feature_vector_length=2,
        local_structural_feature_list=features,
        local_feature_vector_length=i,
        egonet_k_hop=k,
        embeddings_strategy=strategy,
    )
    for k, i in k_and_length
    for strategy in ['separate_embeddings', 'combined_embeddings']
]
