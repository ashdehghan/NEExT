from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
        self.unlabeled_graphs = self.data_df.query("label == -1")["graph_id"].to_list()
        self.X = self.data_df[self.feature_cols].values
        self.y = self.data_df["label"].values

        self.X_labeled = self.data_df.query("label != -1")[self.feature_cols].values
        self.y_labeled = self.data_df.query("label != -1")["label"].values

        self.X_unlabeled = self.data_df.query("label == -1")[self.feature_cols].values
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


class CosineOutlierDetector(BaseEstimator):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self._vectors = None
        self._labels = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._vectors = X
        self._labels = y
        return self

    def predict_prob(self, X: np.ndarray):
        probs = []

        for unlabeled_id in range(len(X)):
            vector = X[unlabeled_id, :]
            prob = self._vector_prediction(vector)
            probs.append(prob)

        return np.array(probs)

    def predict(self, X: np.ndarray, probs: Optional[np.ndarray] = None):
        if probs is None:
            probs = self.predict_prob(X)
        preds = np.where(probs > 0.5, 1, 0)
        return preds

    def predict_full_df(self, unlabeled: np.ndarray, X: np.ndarray):
        df = []

        probs = self.predict_prob(X)
        preds = self.predict(X, probs=probs)

        df = pd.DataFrame({"graph_id": unlabeled, "prob": probs, "pred": preds})
        return df

    def _vector_prediction(self, vector: np.ndarray):
        similarities = cosine_similarity(self._vectors, [vector]).reshape(-1)
        # ind = np.argpartition(similarities, -self.top_k)[-(self.top_k+1) :]
        ind = similarities.argsort()[-(self.top_k + 1) : -1]
        similar_labels = np.array([i for i in self._labels[ind] if i != -1])
        return np.mean(similar_labels) if len(similar_labels) > 0 else 0
