from sklearn.ensemble import IsolationForest
import colorcet as cc
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import mlflow
import numpy as np
import seaborn as sns
import umap
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_random_model(param, features_df):
    model = DummyClassifier(strategy="stratified", random_state=13)
    X, y = np.ones((len(features_df["is_outlier"]), 1)), features_df["is_outlier"]

    run_name = param.comment + "_random"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(dict(param) | {"model_name": "random"})
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=i,
                stratify=y,
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)[:, 1]

            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred), step=i)
            mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_prob, average="micro", multi_class="ovr"), step=i)
            mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_prob, average="micro"), step=i)
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred_prob, average="micro"), step=i)


def run_experiments(param, partial_metrics, dataset):
    # supervised outlier detection
    models = [
        ("lr", LogisticRegression(max_iter=1000, random_state=13)),
        ("lgbm", LGBMClassifier(random_state=13, verbose=-1)),
    ]

    for model_name, model in models:
        run_name = param.comment + "_" + model_name
        with mlflow.start_run(run_name=run_name, nested=True):
            extended_params = dict(param) | {"model_name": model_name, "status": "OK"}
            print(f"Run_name: {run_name}")
            for i in range(10):
                x_train, x_test, y_train, y_test = train_test_split(
                    dataset.X_labeled,
                    dataset.y_labeled,
                    test_size=0.2,
                    random_state=i,
                    stratify=dataset.y_labeled,
                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_pred_prob = model.predict_proba(x_test)[:, 1]

                mlflow.log_params(extended_params)
                for key, value in partial_metrics.items():
                    mlflow.log_metric(key, value, step=i)

                accuracy_metric = accuracy_score(y_test, y_pred)
                auc_metric = roc_auc_score(y_test, y_pred_prob, average="micro", multi_class="ovr")
                ap_metric = average_precision_score(y_test, y_pred_prob, average="micro")
                f1_metric = f1_score(y_test, y_pred, average="micro")

                mlflow.log_metric("accuracy", accuracy_metric, step=i)
                mlflow.log_metric("auc", auc_metric, step=i)
                mlflow.log_metric("average_precision", ap_metric, step=i)
                mlflow.log_metric("f1_score", f1_metric, step=i)
    
    # unsupervised outlier detection
    models = [
        ("loc", LocalOutlierFactor(n_neighbors=20, contamination=0.1)),
        ("if", IsolationForest(random_state=13)),
    ]                    
    for model_name, model in models:
        run_name = param.comment + "_" + model_name
        with mlflow.start_run(run_name=run_name, nested=True):
            print(f"Run_name: {run_name}")
            extended_params = dict(param) | {"model_name": model_name, "status": "OK"}
            for i in range(10):
                y_pred = model.fit_predict(dataset.X_labeled)
                y_pred = np.where(y_pred == -1, 1 ,0)

                mlflow.log_params(extended_params)
                for key, value in partial_metrics.items():
                    mlflow.log_metric(key, value, step=i)

                accuracy_metric = accuracy_score(dataset.y_labeled, y_pred)
                auc_metric = roc_auc_score(dataset.y_labeled, y_pred, average="micro", multi_class="ovr")

                mlflow.log_metric("accuracy", accuracy_metric, step=i)
                mlflow.log_metric("auc", auc_metric, step=i)


def make_charts(param, features_df, community_id, dataset):
    palette = sns.color_palette(cc.glasbey, n_colors=25)
    palette_short = [palette[-1], palette[0]]

    if dataset.X_labeled.shape[1] >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        [a.grid(True) for a in ax]

        x1, x2 = dataset.X_labeled.iloc[:, 0], dataset.X_labeled.iloc[:, 1]
        sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
        sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

        fig.tight_layout()
        mlflow.log_figure(fig, "embedding.png")

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        [a.grid(True) for a in ax]

        reducer = umap.UMAP()
        umap_embed = reducer.fit_transform(dataset.X_labeled)

        x1, x2 = umap_embed[:, 0], umap_embed[:, 1]
        sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
        sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

        fig.tight_layout()
        mlflow.log_figure(fig, "embedding_umap.png")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        [a.grid(True) for a in ax]
        
        x1 = dataset.X_labeled.iloc[:, 0]
        
        kwargs = dict(kde=True, stat='probability', kde_kws={'cut': 3})
        sns.histplot(x=x1, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short, **kwargs)
        sns.histplot(x=x1, hue=community_id, ax=ax[1], palette=palette, **kwargs)
        fig.tight_layout()
        mlflow.log_figure(fig, "embedding.png")
