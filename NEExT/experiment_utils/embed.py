from typing import List

from NEExT.builders import EmbeddingBuilder
from NEExT.collections import EgonetCollection
from NEExT.features import NodeFeatures, StructuralNodeFeatures


def build_features(
    egonet_collection: EgonetCollection,
    feature_vector_length: int,
    feature_list: List[str],
):
    structural_node_features = StructuralNodeFeatures(
        graph_collection=egonet_collection,
        feature_list=["all"],
        feature_vector_length=feature_vector_length,
        n_jobs=8,
        show_progress=False,
    )
    node_features = NodeFeatures(
        egonet_collection,
        feature_list=feature_list,
        show_progress=False,
    )
    structural_features = structural_node_features.compute()
    features = node_features.compute()
    return structural_features, features


def build_embeddings(
    egonet_collection: EgonetCollection,
    structural_features: List[str],
    features: List[str],
    strategy: str,
    structural_embedding_dimension: int,
    feature_embedding_dimension: int,
    embedding_algorithm: str = 'approx_wasserstein',
):
    emb_builder = EmbeddingBuilder(
        egonet_collection,
        strategy=strategy,
        structural_features=structural_features,
        features=features,
    )
    embeddings = emb_builder.compute(structural_embedding_dimension, feature_embedding_dimension, embedding_algorithm=embedding_algorithm)
    return embeddings
