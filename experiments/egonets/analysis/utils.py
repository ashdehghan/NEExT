import pandas as pd
import mlflow


def get_metric_names(mlflow_client, run_id):
    run_data = mlflow_client.get_run(run_id).data.to_dictionary()
    metric_names = list(run_data["metrics"].keys())
    return metric_names


def pull_run_metrics_as_df(mlflow_client, run_id, metric_names=None):
    if metric_names is None:
        metric_names = get_metric_names(mlflow_client, run_id)

    metrics_df = None
    for metric in metric_names:
        metric_history = mlflow_client.get_metric_history(run_id=run_id, key=metric)
        pd_convertible_metric_history = [
            {
                "run_id": run_id,
                "step": mm.step,
                mm.key: mm.value,
            }
            for mm in metric_history
        ]
        if metrics_df is None:
            metrics_df = pd.DataFrame(pd_convertible_metric_history)
        else:
            x = pd.DataFrame(pd_convertible_metric_history)
            metrics_df = metrics_df.merge(x)

    return metrics_df


def get_metrics(server_uri, experiment_ids):
    mlflow.set_tracking_uri(uri=server_uri)
    mlflow_client = mlflow.MlflowClient()
    df_runs = mlflow.search_runs(experiment_ids=experiment_ids)
    df_runs = df_runs.merge(pd.concat([pull_run_metrics_as_df(mlflow_client, i) for i in df_runs["run_id"]], axis=0), on="run_id")
    return df_runs
