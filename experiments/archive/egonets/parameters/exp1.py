from template import Param

experiment = "benchmark_models"
data_path = "abcdo_data_1000_200_0.3" 
params = [
    Param(
        comment="random model",
        random=True,
    ),
    Param(
        comment="node2vec embedding",
        n2v=True,
    ),
]
