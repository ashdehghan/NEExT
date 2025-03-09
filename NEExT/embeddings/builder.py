from typing import List, Literal, Optional

from NEExT.collections import GraphCollection
from NEExT.embeddings import GraphEmbeddings
from NEExT.features import NodeFeatures, StructuralNodeFeatures


class EmbeddingBuilder:
    def __init__(
        self,
        graph_collection: GraphCollection,
        strategy: Literal["separate_embedding", "combined_embedding", "structural_embedding"] = "structural_embedding",
        structural_features: Optional[StructuralNodeFeatures] = None,
        features: Optional[NodeFeatures] = None,
    ):
        if structural_features is None and features is None:
            raise ValueError("At least one of structural_features or features must be provided.")

        self.graph_collection = graph_collection
        self.strategy = strategy
        self.structural_features = structural_features
        self.features = features

        self.available_algorithms = [
            "separate_embedding",
            "combined_embedding",
            "structural_embedding",
        ]

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

        structural_config, feature_config, combined_config = self._build_configs(
            structural_embedding_dimension,
            feature_embedding_dimension,
            feature_columns,
            random_state,
            memory_size,
            embedding_algorithm,
        )

        if self.strategy == "structural_embedding":
            graph_structural_embeddings = GraphEmbeddings(**structural_config)
            embeddings = graph_structural_embeddings.compute()
        elif self.strategy == "combined_embedding":
            graph_embeddings = GraphEmbeddings(**combined_config)
            embeddings = graph_embeddings.compute()
        elif self.strategy == "separate_embedding":
            graph_structural_embeddings = GraphEmbeddings(**structural_config)
            graph_feature_embeddings = GraphEmbeddings(**feature_config)

            structural_embeddings = graph_structural_embeddings.compute()
            feature_embeddings = graph_feature_embeddings.compute()
            embeddings = structural_embeddings + feature_embeddings

        return embeddings

    def _build_configs(
        self,
        structural_embedding_dimension: int,
        feature_embedding_dimension: int,
        feature_columns: List[str],
        random_state,
        memory_size,
        embedding_algorithm,
    ):
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
            combined_config = dict(
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

        return structural_config, combined_config, combined_config
