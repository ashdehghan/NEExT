import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor


def unsupervised_eval(model, ground_truth_df, dataset):
    hyperparames = {
        "LOF": [int(i * 10) for i in [0.5, 1, 5, 10, 15]],
        "IF": [int(i * 10) for i in [0.5, 1, 5, 10, 25, 100]],
    }

    res = []
    max_score = 0
    best_params = None
    
    for i in hyperparames[model]:
        if model == "LOF":
            detector = LocalOutlierFactor(n_neighbors=i)
        elif model == "IF":
            detector = IsolationForest(n_estimators=i)
        y_pred = detector.fit_predict(dataset.X_unlabeled)
        y_pred = np.where(y_pred == 1, 0, 1)
        s = roc_auc_score(ground_truth_df["is_outlier"], y_pred)
        res.append(s)
        
        if s > max_score:
            max_score = s
            best_params = i
        
    if model == "LOF":
        detector = LocalOutlierFactor(n_neighbors=best_params)
    elif model == "IF":
        detector = IsolationForest(n_estimators=best_params)
    y_pred = detector.fit_predict(dataset.X_unlabeled)
    y_pred = np.where(y_pred == 1, 0, 1)
    
    return y_pred, np.mean(res), np.std(res), np.max(res)
