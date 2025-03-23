import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from NEExT.ml_models.outlier_detector import OutlierDataset


def score_unlabeled_gt(
    dataset: OutlierDataset,
    detector: BaseEstimator,
    ground_truth_df: pd,
):
    detector.fit(dataset.X_labeled, dataset.y_labeled)

    out = detector.predict_full_df(dataset.unlabeled_graphs, dataset.X_unlabeled)
    out_unlab = out.merge(ground_truth_df[ground_truth_df["graph_id"].isin(out["graph_id"])]).sort_values("is_outlier", ascending=False)
    bl_acc = roc_auc_score(out_unlab["is_outlier"], out_unlab["pred"])
    return bl_acc
