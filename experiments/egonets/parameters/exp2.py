from template import Param

experiment = "global_features_effect"
params = [
    Param(
        comment=f"global_structural_features_{i}",
        global_structural_feature_list=["all"],
        global_feature_vector_length=i,
        embeddings_strategy="feature_embeddings",
    )
    for i in range(1, 7)
]
