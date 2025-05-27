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


experiment = "local_structural_features_time_v1"
data_path = "abcdo_data_1000_200_0.3" 


combinations = []
for feature in features:
    combinations.append((feature, 0, 1))

for j in [
        (feature, k_hop, i) 
        for feature in features 
        for k_hop in range(1, 5) 
        for i in range(1, k_hop + 1)
    ]:
    combinations.append(j)

params = [
    Param(
        comment=f"local_structural_{feature}_{k_hop}_{i}",
        local_structural_feature_list=[feature],
        local_feature_vector_length=i,
        egonet_k_hop=k_hop,
        embeddings_strategy="structural_embeddings",
    )
    for feature, k_hop, i in combinations
]
