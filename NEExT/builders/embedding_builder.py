from typing import Dict, List, Literal, Optional

import pandas as pd

from NEExT.collections import EgonetCollection, GraphCollection
from NEExT.embeddings import Embeddings, GraphEmbeddings
from NEExT.features import NodeFeatures, StructuralNodeFeatures


class EmbeddingBuilder:
    def __init__(
        self,
        graph_collection: GraphCollection,
        structural_features: Optional[StructuralNodeFeatures] = None,
        features: Optional[NodeFeatures] = None,
        embeddings_dimension: int = 1,
        feature_columns: List[str] = None,
        random_state: int = 42,
        memory_size: str = "4G",
        embeddings_algorithm: str = "approx_wasserstein",
    ):
        if structural_features is None and features is None:
            raise ValueError("At least one of structural_features or features must be provided.")

        self.graph_collection = graph_collection
        self.structural_features = structural_features
        self.features = features
        self.feature_columns = feature_columns
        self.random_state = random_state
        self.memory_size = memory_size
        self.embedding_algorithm = embeddings_algorithm

        self.structural_embeddings_dimension = min(len(self.structural_features.feature_columns), embeddings_dimension)
        self.feature_embeddings_dimension = min(len(self.features.feature_columns), embeddings_dimension)
        self.combined_embeddings_dimension = self.structural_embeddings_dimension + self.feature_embeddings_dimension

        self.available_algorithms = {
            "structural_embeddings": self._structural_embeddings,
            "feature_embeddings": self._feature_embeddings,
            "separate_embeddings": self._separate_embeddings,
            "combined_embeddings": self._combined_embeddings,
            "structural_with_node_features": self._structural_with_node_features,
            "only_egonet_node_features": self._only_egonet_node_features,
        }

    def compute(
        self,
        strategy: Literal[
            "structural_embeddings",
            "feature_embeddings",
            "separate_embeddings",
            "combined_embeddings",
            "structural_with_node_features",
            "only_egonet_node_features",
        ] = "structural_embeddings",
    ) -> "Embeddings":
        embeddings = self.available_algorithms[strategy]()

        return embeddings

    def _get_structural_config(self):
        return dict(
            graph_collection=self.graph_collection,
            features=self.structural_features,
            embedding_algorithm=self.embedding_algorithm,
            embedding_dimension=self.structural_embeddings_dimension,
            feature_columns=self.feature_columns,
            random_state=self.random_state,
            memory_size=self.memory_size,
            suffix="struct",
        )

    def _get_node_feature_config(self):
        return dict(
            graph_collection=self.graph_collection,
            features=self.features,
            embedding_algorithm=self.embedding_algorithm,
            embedding_dimension=self.feature_embeddings_dimension,
            feature_columns=self.feature_columns,
            random_state=self.random_state,
            memory_size=self.memory_size,
            suffix="feat",
        )

    def get_combined_config(self):
        combined_features = self.structural_features + self.features
        return dict(
            graph_collection=self.graph_collection,
            features=combined_features,
            embedding_algorithm=self.embedding_algorithm,
            embedding_dimension=self.combined_embeddings_dimension,
            feature_columns=self.feature_columns,
            random_state=self.random_state,
            memory_size=self.memory_size,
            suffix="feat",
        )

    def _structural_embeddings(self):
        structural_config = self._get_structural_config()

        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        embeddings = graph_structural_embeddings.compute()

        return embeddings

    def _feature_embeddings(self):
        features_config = self._get_node_feature_config()

        graph_feature_embeddings = GraphEmbeddings(**features_config)
        embeddings = graph_feature_embeddings.compute()

        return embeddings

    def _separate_embeddings(self):
        structural_config = self._get_structural_config()
        features_config = self._get_node_feature_config()

        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        graph_feature_embeddings = GraphEmbeddings(**features_config)

        structural_embeddings = graph_structural_embeddings.compute()

        # compute feature embedding and combine them only if node features are specified
        if self.feature_columns is not None:
            feature_embeddings = graph_feature_embeddings.compute()
            embeddings = structural_embeddings + feature_embeddings

        return embeddings

    def _combined_embeddings(self):
        combined_config = self.get_combined_config()
        graph_embeddings = GraphEmbeddings(**combined_config)
        embeddings = graph_embeddings.compute()
        return embeddings

    def _only_egonet_node_features(self):
        if not isinstance(self.graph_collection, EgonetCollection):
            raise Exception("Graph collection is not an EgonetCollection!")

        embeddings = self.graph_collection.egonet_node_features
        return embeddings

    def _structural_with_node_features(self):
        if not isinstance(self.graph_collection, EgonetCollection):
            raise Exception("Graph collection is not an EgonetCollection!")

        structural_config = self._get_structural_config()

        graph_structural_embeddings = GraphEmbeddings(**structural_config)
        embeddings = graph_structural_embeddings.compute()
        embeddings = embeddings + self.graph_collection.egonet_node_features
        return embeddings
