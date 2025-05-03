from template import Param


experiment = "embeddings_combinations"
params = [
    Param(
        comment="embeddings_combinations",
        global_structural_feature_list=['all'],
        global_feature_vector_length=4,
        local_structural_feature_list=['betastar'],
        local_feature_vector_length=2,
        k_hop=2,
        embeddings_strategy='combined_embeddings',
    ),
    Param(
        comment="embeddings_combinations",
        global_structural_feature_list=['all'],
        global_feature_vector_length=4,
        local_structural_feature_list=['betastar'],
        local_feature_vector_length=2,
        k_hop=2,
        embeddings_strategy='separate_embeddings',
    ),
]
