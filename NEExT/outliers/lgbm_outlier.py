from typing import Literal

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class LGBMOutlier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 10,
        min_data_in_leaf: int = 1,
        num_leaves: int = 10,
        max_depth: int = 10,
        class_weight: Literal[None, "balanced"] = "balanced",
        learning_rate: float = 1e-1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        colsample_bytree: float = 1.0,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree

        self.hyperparameters = {
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
        } | {
            "n_estimators": self.n_estimators,
            "min_data_in_leaf": self.min_data_in_leaf,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "colsample_bytree": self.colsample_bytree,
        }

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)

        self.model_ = lgb.train(
            self.hyperparameters,
            train_set=train_set,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5),
            ],
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.where(self.model_.predict(X) >= 0.5, 1, 0)

    def predict_proba(self, X):
        check_is_fitted(self)
        out = np.zeros((len(X), 2))
        out[:, 1] = self.model_.predict(X)
        out[:, 0] = 1 - out[:, 1]
        return out
