from template import Param


experiment = "local_tuning"
data_path = "abcdo_data_1000_200_0.3" 

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
        comment=f"{experiment}_{k}_{i}",
        global_structural_feature_list=[],
        local_structural_feature_list=features,
        local_feature_vector_length=i,
        egonet_k_hop=k,
        embeddings_strategy="structural_embeddings",
    )
    for k, i in k_and_length
]
