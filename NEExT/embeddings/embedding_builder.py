from typing import Dict, List, Literal, Optional

import pandas as pd

from NEExT.collections import GraphCollection
from NEExT.collections.egonet_collection import EgonetCollection
from NEExT.embeddings import GraphEmbeddings
from NEExT.features import NodeFeatures, StructuralNodeFeatures


class EmbeddingBuilder:
    def __init__(
        self,
        graph_collection: GraphCollection,
        strategy: Literal[
            "separate_embedding",
            "combined_embedding",
            "structural_embedding",
            "merge_node_features",
            "only_node_features",
        ] = "structural_embedding",
        structural_features: Optional[StructuralNodeFeatures] = None,
        features: Optional[NodeFeatures] = None,
    ):
        if structural_features is None and features is None:
            raise ValueError("At least one of structural_features or features must be provided.")

        self.graph_collection = graph_collection
        self.strategy = strategy
        self.structural_features = structural_features
        self.features = features

        self.available_algorithms = {
            "separate_embedding": self._separate_embedding,
            "combined_embedding": self._combined_embedding,
            "structural_embedding": self._structural_embedding,
            "merge_egonet_node_features": self._merge_egonet_node_features,
            "only_egonet_node_features": self._only_egonet_node_features,
        }

    def compute(
        self,
        structural_embedding_dimension: int,
        feature_embedding_dimension: int,
        feature_columns: List[str] = None,
        random_state: int = 42,
        memory_size: str = "4G",
        embedding_algorithm: str = "approx_wasserstein",
    ):
        structural_embedding_dimension = min(len(self.structural_features.feature_columns), structural_embedding_dimension)
        feature_embedding_dimension = min(len(self.features.feature_columns), feature_embedding_dimension)

        configs = self._build_configs(
            structural_embedding_dimension,
            feature_embedding_dimension,
            feature_columns,
            random_state,
            memory_size,
            embedding_algorithm,
        )

        embeddings = self.available_algorithms[self.strategy](**configs)

        return embeddings

    def _only_egonet_node_features(self, **kwargs):
        if not isinstance(self.graph_collection, EgonetCollection):
            raise Exception('Not graph collection is not an EgonetCollection!')
        embeddings =  self.graph_collection.egonet_node_features
        return embeddings

    def _merge_egonet_node_features(self, structural_config, **kwargs):
        if not isinstance(self.graph_collection, EgonetCollection):
            raise Exception('Not graph collection is not an EgonetCollection!')
        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        embeddings = graph_structural_embeddings.compute()
        embeddings = embeddings + self.graph_collection.egonet_node_features
        return embeddings

    def _separate_embedding(self, structural_config, features_config, **kwargs):
        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        graph_feature_embeddings = GraphEmbeddings(**features_config)

        structural_embeddings = graph_structural_embeddings.compute()
        feature_embeddings = graph_feature_embeddings.compute()
        embeddings = structural_embeddings + feature_embeddings
        return embeddings

    def _combined_embedding(self, combined_config, **kwargs):
        graph_embeddings = GraphEmbeddings(**combined_config)
        embeddings = graph_embeddings.compute()
        return embeddings

    def _structural_embedding(self, structural_config, **kwargs):
        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        embeddings = graph_structural_embeddings.compute()
        return embeddings

    def _build_configs(
        self,
        structural_embedding_dimension: int,
        feature_embedding_dimension: int,
        feature_columns: List[str],
        random_state: int,
        memory_size: str,
        embedding_algorithm: str,
    ) -> Dict[str, Dict]:
        if self.structural_features:
            structural_config = dict(
                graph_collection=self.graph_collection,
                features=self.structural_features,
                embedding_algorithm=embedding_algorithm,
                embedding_dimension=structural_embedding_dimension,
                feature_columns=feature_columns,
                random_state=random_state,
                memory_size=memory_size,
                suffix="struct",
            )
        else:
            structural_config = None

        if self.features:
            features_config = dict(
                graph_collection=self.graph_collection,
                features=self.features,
                embedding_algorithm=embedding_algorithm,
                embedding_dimension=feature_embedding_dimension,
                feature_columns=feature_columns,
                random_state=random_state,
                memory_size=memory_size,
                suffix="feat",
            )
        else:
            combined_config = None

        if self.structural_features and self.features:
            combined_features = self.structural_features + self.features
            combined_config = dict(
                graph_collection=self.graph_collection,
                features=combined_features,
                embedding_algorithm=embedding_algorithm,
                embedding_dimension=structural_embedding_dimension,
                feature_columns=feature_columns,
                random_state=random_state,
                memory_size=memory_size,
                suffix="feat",
            )

        return dict(
            structural_config=structural_config, 
            features_config=features_config, 
            combined_config=combined_config
        )
