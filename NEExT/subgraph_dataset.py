import pandas as pd

from NEExT.embeddings import Embeddings
from NEExT.graph_collection import GraphCollection
from NEExT.subgraph_collection import SubGraphCollection

def create_data_df(
    graph_collection: GraphCollection,
    subgraph_collection: SubGraphCollection,
    embeddings: Embeddings,
    target: str
):
    """
    Mapping values of known features from a (graph_id, node_id) to (subgraph_id)
    """
    node_features_df = []
    data_df = embeddings.embeddings_df.copy().rename(columns={'graph_id': "subgraph_id"})
    data_df[['graph_id', 'node_id']] = data_df['subgraph_id'].map(subgraph_collection.subgraph_to_graph_node_mapping).to_list()

    for graph in graph_collection.graphs:
        for node in graph.nodes:
            node_variables = {'graph_id': graph.graph_id, 'node_id': node
            }
            if graph.graph_label:
                node_variables['graph_label'] = graph.graph_label
                
            node_variables = node_variables | graph.node_attributes[node]
            node_variables.pop(target)
            node_features_df.append(node_variables)
            
    node_features_df = pd.DataFrame(node_features_df)
    data_df = data_df.merge(node_features_df).drop(columns=['graph_id', 'node_id']).rename(columns={'subgraph_id': "graph_id"})
    return data_df