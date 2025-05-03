from template import Param

experiment = "benchmark_models"
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
