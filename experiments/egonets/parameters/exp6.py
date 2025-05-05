from NEExT.graphs import egonet
from template import Param


experiment = "tuning_positional_encoding"

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
k_and_length = [(1, 1)]


params = [
    Param(
        comment=f"{strategy}_local_{k_hop}_{local_i}_global_{global_i}_position_{one_hot}",
        global_structural_feature_list=['all'],
        global_feature_vector_length=global_i,
        local_structural_feature_list=features,
        local_feature_vector_length=local_i,
        egonet_k_hop=k_hop,
        embeddings_strategy=strategy,
        egonet_position=True,
        position_one_hot=one_hot,
    )
    for k_hop, local_i in k_and_length
    for strategy in ['combined_embeddings']
    for global_i in [2]
    for one_hot in [False, True]
]
