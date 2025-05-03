from template import Param

features = [
    "page_rank",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "clustering_coefficient",
    "local_efficiency",
    "lsme",
    "load_centrality",
    "basic_expansion",
    "betastar",
]


experiment = "local_structural_features_time"
params = [
    Param(
        comment=f"local_structural_{feature}_{i}",
        local_structural_feature_list=[feature],
        local_feature_vector_length=i,
        k_hop=k_hop,
        embeddings_strategy='structural_embeddings',
    )
    for feature in features
    for k_hop in range(0, 5)
    for i in range(1, k_hop + 1)
]
