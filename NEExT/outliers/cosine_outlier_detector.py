from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class CosineOutlierDetector(BaseEstimator):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self._vectors = None
        self._labels = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._vectors = X
        self._labels = y
        return self

    def predict_proba(self, X: np.ndarray):
        probs = []

        for unlabeled_id in range(len(X)):
            vector = X[unlabeled_id, :]
            prob = self._vector_sim_label_prob(vector)
            probs.append(prob)
        
        probs = np.array(probs)
        out_probs = np.ones((len(probs), 2))
        out_probs[:, 0] = 1 - probs
        out_probs[:, 1] = probs
        return out_probs

    def predict(self, X: np.ndarray, probs: Optional[np.ndarray] = None):
        if probs is None:
            probs = self.predict_proba(X)
        preds = np.where(probs[:, 1] > 0.5, 1, 0)
        return preds

    def _vector_sim_label_prob(self, vector: np.ndarray):
        similarities = cosine_similarity(self._vectors, [vector]).reshape(-1)
        # ind = np.argpartition(similarities, -self.top_k)[-(self.top_k+1) :]
        ind = similarities.argsort()[-(self.top_k + 1) : -1]
        similar_labels = np.array([i for i in self._labels[ind] if i != -1])
        return np.mean(similar_labels) if len(similar_labels) > 0 else 0
