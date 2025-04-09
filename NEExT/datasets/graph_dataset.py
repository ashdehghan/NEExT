import pandas as pd
from sklearn.preprocessing import StandardScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings


class GraphDataset:
    def __init__(
        self,
        graph_collection: GraphCollection,
        embeddings: Embeddings,
        standardize: bool = False,
        unlabeled_marker: int = -1,
    ):
        """
        This class integrates graph data, embeddings, and labels,
        providing a structured way to access and manipulate the data for various
        graph-related tasks. It supports data standardization and the identification
        of labeled and unlabeled graphs.

        Args:
            graph_collection (GraphCollection): The collection of graphs.
            embeddings (Embeddings): The embeddings of the graphs.
            standardize (bool, optional): Whether to standardize the embedding
                features. Defaults to False.
            unlabeled_marker (int, optional): Value used to mark unlabeled
                graphs. Defaults to -1.

        Raises:
            ValueError: If the graph_collection or embeddings are not provided.
        """
        self.graph_collection = graph_collection
        self.embeddings = embeddings
        self.standardize = standardize
        self.unlabeled_marker = unlabeled_marker

        self.labels_df = self._prepare_labels_df()
        self._embeddings_df = pd.merge(self.embeddings.embeddings_df, self.labels_df, on="graph_id").sort_values("graph_id")
        self.graph_ids = self._embeddings_df["graph_id"].to_list()

        self.labeled_graphs = self._embeddings_df.query(f"label != {self.unlabeled_marker}")["graph_id"].to_list()
        self.unlabeled_graphs = self._embeddings_df.query(f"label == {self.unlabeled_marker}")["graph_id"].to_list()
        
        if self.standardize:
            self.scaler = StandardScaler().fit(self._embeddings_df[self.embeddings.embedding_columns])

        self._prepare_modeling_features()

    def _prepare_modeling_features(self):
        """Prepare features and labels for modeling.

        This method extracts the features (X) and labels (y) from the data_df,
        and separates them into labeled and unlabeled sets.
        """
        self.X = self._embeddings_df[self.embeddings.embedding_columns]
        self.y = self._embeddings_df["label"].values

        self.X_labeled = self._embeddings_df.query(f"label != {self.unlabeled_marker}")[self.embeddings.embedding_columns]
        self.X_unlabeled = self._embeddings_df.query(f"label == {self.unlabeled_marker}")[self.embeddings.embedding_columns]
        self.y_labeled = self._embeddings_df.query(f"label != {self.unlabeled_marker}")["label"].values

        if self.standardize:
            self.X_labeled = self.scaler.transform(self.X_labeled)
            self.X_unlabeled = self.scaler.transform(self.X_unlabeled)

    def _prepare_labels_df(self) -> pd.DataFrame:
        """Prepare DataFrame with graph IDs and labels.

        This method creates a DataFrame that maps graph IDs to their
        corresponding labels, based on the graph_collection.

        Returns:
            pd.DataFrame: DataFrame with 'graph_id' and 'label' columns.
        """
        graph_ids = []
        graph_labels = []

        for graph in self.graph_collection.graphs:
            graph_ids.append(graph.graph_id)
            graph_labels.append(graph.graph_label)

        return pd.DataFrame({"graph_id": graph_ids, "label": graph_labels})
