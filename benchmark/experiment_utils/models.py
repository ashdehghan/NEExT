import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from NEExT.outliers import OutlierDataset


def score_unlabeled_gt(
    dataset: OutlierDataset,
    detector: BaseEstimator,
    ground_truth_df: pd.DataFrame = None,
):
    if ground_truth_df is not None:
        ground_truth_df = ground_truth_df.sort_values("graph_id")
        out = predict_full_df(detector, dataset.graph_id, dataset.X)
        bl_acc = roc_auc_score(ground_truth_df["is_outlier"], out["pred"])
    else:
        out = predict_full_df(detector, dataset.labeled_graphs, dataset.X_labeled)
        bl_acc = roc_auc_score(dataset.y_labeled, out["pred"])
    return out, bl_acc


def predict_full_df(detector: BaseEstimator, unlabeled: np.ndarray, X: np.ndarray):
    df = []

    probs = detector.predict_proba(X)[:, 1]
    preds = detector.predict(X)

    df = pd.DataFrame({"graph_id": unlabeled, "prob": probs, "pred": preds}).sort_values("graph_id")
    return df
