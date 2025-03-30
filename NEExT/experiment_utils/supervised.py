from functools import partial

import numpy as np
import optuna
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from NEExT.experiment_utils.models import score_unlabeled_gt
from NEExT.outliers import CosineOutlierDetector, LGBMOutlier, OutlierDataset


def objective(trial: optuna.Trial, model: str, dataset: OutlierDataset):
    if model == "cosine":
        top_k = trial.suggest_int("top_k", 1, 20)
        detector = CosineOutlierDetector(top_k=top_k)
    elif model == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        detector = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 75),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 1, 100),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e1, log=True),
            "reg_alpha": trial.suggest_float("learning_rate", 1e-5, 1e1, log=True),
            "reg_lambda": trial.suggest_float("learning_rate", 1e-5, 1e1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
        }
        detector = LGBMOutlier(**params)

    metric = cross_val_score(
        detector,
        dataset.X_labeled,
        dataset.y_labeled,
        cv=StratifiedKFold(min(5, int(np.sum(dataset.y_labeled)))),
        n_jobs=-1,
        scoring=make_scorer(roc_auc_score),
    )
    return np.nanmean(metric)


def supervised_eval(model, ground_truth_df, dataset):
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, model=model, dataset=dataset), n_trials=50, n_jobs=4)

    if model == "cosine":
        detector = CosineOutlierDetector(**study.best_params)
    elif model == "knn":
        detector = KNeighborsClassifier(**study.best_params)
    elif model == "lgbm":
        detector = LGBMOutlier(**study.best_params)

    detector.fit(dataset.X_labeled, dataset.y_labeled)
    out_df, score = score_unlabeled_gt(dataset, detector, ground_truth_df)
    return out_df, score
