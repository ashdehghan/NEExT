import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def objective(trial, X_train, y_train):
    lgb_params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "auc_mu",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50),
        "verbose": -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc_mu", callbacks=[lgb.early_stopping(10, verbose=False)])

        y_pred_proba = model.predict_proba(X_val)
        score = roc_auc_score(y_val, y_pred_proba, multi_class="ovr")
        auc_scores.append(score)

    return np.mean(auc_scores)


def run_experiments(config, dataset):
    scores = []
    print(dataset.X_labeled)
    for i in range(config.modeling["n_experiments"]):
        print(f"\nExperiment {i + 1}/{config.modeling['n_experiments']}")

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X_labeled,
            dataset.y_labeled,
            test_size=config.modeling["test_size"],
            random_state=i,
            stratify=dataset.y_labeled,
        )
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=config.modeling["n_trials"])

        best_params = study.best_params
        best_params.update(
            {
                "objective": "multiclass",
                "num_class": len(np.unique(y_train)),
                "metric": config.modeling["eval_metric"],
                "random_state": config.modeling["random_state"],
                "verbose": -1,
            }
        )

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        print(f"  Best Params: {best_params}")
        print(f"  Accuracy: {acc:.4f}")

    print("\nAll Results:")
    for idx, acc in enumerate(scores):
        print(f"Experiment {idx + 1}: Accuracy = {acc:.4f}")
    print(f"Average Accuracy: {np.mean(scores):.4f}")
