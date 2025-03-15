import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from NEExT.collections.graph_collection import GraphCollection
from NEExT.embeddings.embeddings import Embeddings


class OutlierDetector(BaseEstimator):
    def __init__(
        self,
        graph_collection: GraphCollection,
        embedding: Embeddings,
        top_k: int = 10,
        standardize: bool = False,
        threshold: float = 0.0,
    ):
        self.graph_collection = graph_collection
        self.data_df = embedding.embeddings_df
        self.top_k = top_k
        self.standardize = standardize
        self.threshold = threshold

        self.labels_df = self._prepare_labels_df()

        # Merge embeddings with labels
        self.data_df = pd.merge(self.data_df, self.labels_df, on="graph_id")
        self.unlabeled = self.data_df.query("label == -1")["graph_id"].to_list()
        self.feature_cols = [col for col in self.data_df.columns if col not in ["graph_id", "label"]]

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

    def _vector_prediction(self, vector):
        vectors = self.data_df[self.feature_cols].values
        if self.standardize:
            scaler = StandardScaler()
            vectors = scaler.fit_transform(vectors)
            vector = scaler.transform(vector.reshape(1, -1)).reshape(-1)

        similarities = cosine_similarity(vectors, [vector]).reshape(-1)

        ind = np.argpartition(similarities, -self.top_k)[-self.top_k - 1 : -1]
        similar_labels = self.data_df.loc[ind, "label"].values
        similar_labels = np.array([i for i in similar_labels if i != -1])
        return np.mean(similar_labels)

    def fit(self, X=None, y=None):
        return self

    def predict_prob(self, X=None):
        probs = []

        for unlabeled_id in self.unlabeled:
            vector = self.data_df.query(f"graph_id=={unlabeled_id}")[self.feature_cols].values.flatten()
            prob = self._vector_prediction(vector)
            probs.append(prob)

        return np.array(probs)

    def predict(self, X=None, probs=None):
        if probs is None:
            probs = self.predict_prob()
        preds = np.where(probs > self.threshold, 1, 0)
        return preds

    def predict_full_df(self, X=None):
        df = []

        probs = self.predict_prob()
        preds = self.predict(probs=probs)

        df = pd.DataFrame(
            {
                "graph_id": self.unlabeled,
                "prob": probs,
                "pred": preds,
            }
        )
        return df
