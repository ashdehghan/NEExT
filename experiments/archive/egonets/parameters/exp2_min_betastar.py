from template import Param

experiment = "global_features_effect_min_betastar"
data_path = "abcdo_data_1000_200_0.3" 

params = [
    Param(
        comment=f"global_structural_features_{i}",
        global_structural_feature_list=[
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
        ],
        global_feature_vector_length=i,
        embeddings_strategy="feature_embeddings",
    )
    for i in range(1, 7)
]
