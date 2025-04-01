import pandas as pd
from sklearn.preprocessing import StandardScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings


class OutlierDataset:
    def __init__(
        self,
        graph_collection: GraphCollection,
        embedding: Embeddings,
        standardize: bool = False,
    ):
        self.graph_collection = graph_collection
        self.data_df = embedding.embeddings_df
        self.feature_cols = [col for col in self.data_df.columns if col not in ["graph_id"]]
        self.standardize = standardize

        if self.standardize:
            self.scaler = StandardScaler()
            self.data_df[self.feature_cols] = self.scaler.fit_transform(self.data_df[self.feature_cols])

        self.labels_df = self._prepare_labels_df()

        # Merge embeddings with labels
        self.data_df = pd.merge(self.data_df, self.labels_df, on="graph_id").sort_values("graph_id")
        self.graph_id = self.data_df["graph_id"].to_list()
        self.labeled_graphs = self.data_df.query("label != -1")["graph_id"].to_list()
        self.unlabeled_graphs = self.data_df.query("label == -1")["graph_id"].to_list()

        # Extracting features and labels for easy access
        self.X = self.data_df[self.feature_cols]
        self.y = self.data_df["label"].values

        self.X_labeled = self.data_df.query("label != -1")[self.feature_cols]
        self.y_labeled = self.data_df.query("label != -1")["label"].values
    
        self.X_unlabeled = self.data_df.query("label == -1")[self.feature_cols]

        if self.standardize:
            self.X_labeled = self.scaler.transform(self.data_df.query("label != -1")[self.feature_cols])
            self.X_unlabeled = self.scaler.transform(self.data_df.query("label == -1")[self.feature_cols])

    def _prepare_labels_df(self) -> pd.DataFrame:
        """
        Prepare DataFrame with graph IDs and labels.

        Returns:
            pd.DataFrame: DataFrame with graph_id and label columns
        """
        graph_ids = []
        graph_labels = []

        for graph in self.graph_collection.graphs:
            graph_ids.append(graph.graph_id)
            graph_labels.append(graph.graph_label)

        return pd.DataFrame({"graph_id": graph_ids, "label": graph_labels})
