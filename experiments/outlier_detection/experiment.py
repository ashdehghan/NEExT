import traceback
from functools import partial
from typing import Callable

import mlflow
from config import Params
from embedding_utils import (
    add_positional_features,
    compute_embedding,
    compute_global_features,
    compute_local_node_features,
    compute_local_structural_features,
    node2vec_embedding,
)
from joblib import Parallel, delayed
from modeling_utils import make_charts, run_experiments, train_random_model

from NEExT.collections import EgonetCollection
from NEExT.datasets import GraphDataset
from NEExT.io import GraphIO



def evaluation_loop(param: Params, prepare_dataset: Callable):
    with mlflow.start_run(run_name=param.comment):
        try:
            partial_metrics = {}
            edges_df, mapping_df, features_df, community_id = prepare_dataset()
            graph_io = GraphIO()
            graph_collection = graph_io.load_from_dfs(
                edges_df=edges_df,
                node_graph_df=mapping_df,
                node_features_df=features_df,
                graph_type="igraph",
                filter_largest_component=param.filter_largest_component,
            )
            if param.random:
                train_random_model(param, features_df)
                return None

            elif param.n2v:
                embeddings = node2vec_embedding(graph_collection)
                
                egonet_collection = EgonetCollection(egonet_feature_target=param.egonet_target, skip_features=param.egonet_skip_features)
                egonet_collection.compute_k_hop_egonets(graph_collection, 0)
            else:
                global_structural_node_features = compute_global_features(param, partial_metrics, graph_collection)
                
                egonet_collection = EgonetCollection(egonet_feature_target=param.egonet_target, skip_features=param.egonet_skip_features)
                egonet_collection.compute_k_hop_egonets(graph_collection, param.egonet_k_hop)

                structural_features = compute_local_structural_features(param, partial_metrics, egonet_collection)
                features = compute_local_node_features(param, egonet_collection, global_structural_node_features)

                structural_features, features = add_positional_features(param, egonet_collection, structural_features, features)

                embeddings = compute_embedding(param, egonet_collection, structural_features, features)

            dataset = GraphDataset(egonet_collection, embeddings)
            run_experiments(param, partial_metrics, dataset)
            make_charts(param, features_df, community_id, dataset)

        except Exception as e:
            print("RUN FAILED!")
            print(traceback.format_exc())
            mlflow.log_params(dict(param) | {"status": "FAILURE"})