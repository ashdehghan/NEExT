from NEExT.graphs import egonet
from template import Param


experiment = "brazil_benchmark"

# abcdo graphs from https://github.com/CptQuak/graph_data
# airports from https://github.com/leoribeiro/struc2vec/tree/master/graph
data_path = "airports_brazil"

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
