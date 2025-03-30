from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor

from NEExT.collections import EgonetCollection
from NEExT.experiment_utils.data_loading import load_abcdo_data, load_pygod_data
from NEExT.experiment_utils.embed import build_embeddings, build_features
from NEExT.experiment_utils.models import score_unlabeled_gt
from NEExT.io import GraphIO
from NEExT.outliers import CosineOutlierDetector, LGBMOutlier, OutlierDataset

DATA = "gen_100"
OUTLIER_MODE = "any"

MODE = "unsupervised"
MODEL = "IF"
HIDE_FRAC = {0: 1, 1: 1}

MODE = "supervised"
MODEL = "lgbm"
HIDE_FRAC = {0: 0.8, 1: 0.8}


def load_data(DATA, OUTLIER_MODE, HIDE_FRAC):
    graph_io = GraphIO()

    if DATA.startswith("abcd"):
        edges_df, mapping_df, features_df, ground_truth_df = load_pygod_data(name=DATA, outlier_mode=OUTLIER_MODE, hide_frac=HIDE_FRAC)
        graph_data = {
            "target": "is_outlier",
            "skip_features": ["random_community_feature", "community_id"],
            "feature_list": [],
        }
    else:
        edges_df, mapping_df, features_df, ground_truth_df = load_pygod_data(name=DATA, outlier_mode=OUTLIER_MODE, hide_frac=HIDE_FRAC)
        graph_data = {
            "target": "is_outlier",
            "skip_features": [],
            "feature_list": [i for i in features_df.columns[1:-1]],
        }

    return graph_io, edges_df, mapping_df, features_df, ground_truth_df, graph_data


def data_processing(graph_io, edges_df, mapping_df, features_df, graph_data):
    graph_collection = graph_io.load_from_dfs(
        edges_df=edges_df,
        node_graph_df=mapping_df,
        node_features_df=features_df,
        graph_type="igraph",
        filter_largest_component=False,
    )
    subgraph_collection = EgonetCollection()
    subgraph_collection.create_egonets_from_graphs(
        graph_collection=graph_collection,
        egonet_target=graph_data["target"],
        egonet_algorithm="k_hop_egonet",
        skip_features=graph_data["skip_features"],
        max_hop_length=1,
    )
    structural_features, features = build_features(subgraph_collection, feature_vector_length=5, feature_list=graph_data["feature_list"])

    embeddings = build_embeddings(
        subgraph_collection,
        structural_features,
        features,
        strategy="structural_embedding",
        structural_embedding_dimension=5,
        feature_embedding_dimension=5,
        embedding_algorithm="approx_wasserstein",
        # approx_wasserstein, wasserstein, sinkhornvectorizer
    )
    dataset = OutlierDataset(subgraph_collection, embeddings, standardize=False)
    return dataset


def unsupervised_eval(MODEL, ground_truth_df, dataset):
    hyperparames = {
        "LOF": [int(i * 10) for i in [0.5, 1, 5, 10, 15]],
        "IF": [int(i * 10) for i in [0.5, 1, 5, 10, 25, 100]],
    }

    res = []
    for i in hyperparames[MODEL]:
        if MODEL == "LOF":
            detector = LocalOutlierFactor(n_neighbors=i)
        elif MODEL == "IF":
            detector = IsolationForest(n_estimators=i)
        y_pred = detector.fit_predict(dataset.X_unlabeled)
        y_pred = np.where(y_pred == 1, 0, 1)
        s = roc_auc_score(ground_truth_df["is_outlier"], y_pred)
        res.append(s)

    return np.mean(res), np.std(res), np.max(res)


def supervised_eval(MODEL, ground_truth_df, dataset):
    def objective(trial: optuna.Trial, model: str):
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

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, model=MODEL), n_trials=50, n_jobs=4)

    if MODEL == "cosine":
        detector = CosineOutlierDetector(**study.best_params)
    elif MODEL == "knn":
        detector = KNeighborsClassifier(**study.best_params)
    elif MODEL == "lgbm":
        detector = LGBMOutlier(**study.best_params)

    detector.fit(dataset.X_labeled, dataset.y_labeled)
    out_df, score = score_unlabeled_gt(dataset, detector, ground_truth_df)
    return score


graph_io, edges_df, mapping_df, features_df, ground_truth_df, graph_data = load_data(DATA, OUTLIER_MODE, HIDE_FRAC)
dataset = data_processing(graph_io, edges_df, mapping_df, features_df, graph_data)
if MODE == "supervised":
    score = supervised_eval(MODEL, ground_truth_df, dataset)
    print(DATA, OUTLIER_MODE, MODEL, score, np.nan, np.nan, HIDE_FRAC[0], HIDE_FRAC[1])

elif MODE == "unsupervised":
    score_mean, score_std, score_max = unsupervised_eval(MODEL, ground_truth_df, dataset)
    print(DATA, OUTLIER_MODE, MODEL, score_mean, score_std, score_max, HIDE_FRAC[0], HIDE_FRAC[1])
