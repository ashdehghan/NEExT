from NEExT.graphs import egonet
from template import Param


experiment = "test"
data_path = "abcdo_data_10000_500_0.3" 

features = [
    "betastar",
]

combinations = []
for feature in features:
    combinations.append((feature, 0, 1))

for j in [
        (feature, k_hop, i) 
        for feature in features 
        for k_hop in range(4, 5) 
        for i in range(1, k_hop + 1)
    ]:
    combinations.append(j)


params = [
    Param(
        comment=f'exp1_{feature}_{k_hop}_{glob_len}',
        global_structural_feature_list=[feature],
        global_feature_vector_length=glob_len,
        egonet_k_hop=k_hop,
        embeddings_strategy='feature_embeddings',
    )
    for feature, k_hop, glob_len in combinations
]
