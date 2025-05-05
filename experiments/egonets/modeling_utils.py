import colorcet as cc
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
import umap
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_random_model(params, features_df):
    model = DummyClassifier(strategy="stratified", random_state=13)
    X, y = np.ones((len(features_df["is_outlier"]), 1)), features_df["is_outlier"]

    extended_params = params | {"model_name": "random"}
    mlflow.log_params(extended_params)
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
        y_pred_prob = model.predict_proba(x_test)#[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred), step=i)
        mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_prob, average="micro", multi_class="ovr"), step=i)
        mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_prob, average="micro"), step=i)
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_prob, average="micro"), step=i)


def run_experiments(params, partial_metrics, dataset):
    models = [
        ("lr", LogisticRegression(max_iter=1000, random_state=13)),
        ("lgbm", LGBMClassifier(random_state=13, verbose=-1)),
    ]

    for model_name, model in models:
        with mlflow.start_run(nested=True):
            extended_params = params | {"model_name": model_name, "status": "OK"}
            for i in range(10):
                print(f"Experiment run id: {i}")
                x_train, x_test, y_train, y_test = train_test_split(
                    dataset.X_labeled,
                    dataset.y_labeled,
                    test_size=0.2,
                    random_state=i,
                    stratify=dataset.y_labeled,
                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_pred_prob = model.predict_proba(x_test)#[:, 1]

                mlflow.log_params(extended_params)
                for key, value in partial_metrics.items():
                    mlflow.log_metric(key, value, step=i)
                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred), step=i)
                mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_prob, average="micro", multi_class="ovr"), step=i)
                mlflow.log_metric("average_precision", average_precision_score(y_test, y_pred_prob, average="micro"), step=i)
                mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="micro"), step=i)
    print("done?")


def make_charts(params, features_df, community_id, dataset):
    palette = sns.color_palette(cc.glasbey, n_colors=25)
    palette_short = [palette[-1], palette[0]]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.grid(True)
    x1, x2 = dataset.X_labeled.iloc[:, 0], dataset.X_labeled.iloc[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
    sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

    fig.tight_layout()
    mlflow.log_figure(fig, "embedding.png")

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.grid(True)

    reducer = umap.UMAP()
    umap_embed = reducer.fit_transform(dataset.X_labeled)

    x1, x2 = umap_embed[:, 0], umap_embed[:, 1]
    sns.scatterplot(x=x1, y=x2, hue=features_df["is_outlier"], ax=ax[0], palette=palette_short)
    sns.scatterplot(x=x1, y=x2, hue=community_id, ax=ax[1], palette=palette)

    fig.tight_layout()
    mlflow.log_figure(fig, "embedding_umap.png")
