import pickle
from typing import Any, Dict

import mlflow
from mlflow import get_experiment_by_name
from mlflow import log_artifact, log_metric
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from optuna import Study
from pandas import DataFrame, Series

from hcve_lib.custom_types import ValueWithCI


def log_pickled(data: Any, path: str) -> None:
    with TempDir() as tmp_dir:
        path = tmp_dir.path() + '/' + path
        with open(path, 'bw') as file:
            pickle.dump(data, file, protocol=5)

        log_artifact(path)


def log_text(text: str, path: str) -> None:
    with TempDir() as tmp_dir:
        path = tmp_dir.path() + '/' + path
        with open(path, 'w') as file:
            file.write(text)
        log_artifact(path)


def load_pickled_artifact(run_id: str, path: str) -> Any:
    client = MlflowClient()
    path = client.download_artifacts(run_id, path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def log_metrics_ci(metrics: Dict[Any, ValueWithCI]) -> None:
    for metric_name, metric_value in metrics.items():
        log_metric(metric_name, metric_value['mean'])
        log_metric(f'{metric_name}_l', metric_value['ci'][0])
        log_metric(f'{metric_name}_r', metric_value['ci'][1])


def is_root_run(_runs: DataFrame) -> Series:
    return _runs['tags.mlflow.parentRunId'].isna()


def get_completed_runs(experiment_name: str) -> DataFrame:
    _runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id)
    return _runs[_runs['status'] == 'FINISHED']


def get_last_run(_runs: DataFrame, condition: Any) -> Series:
    return _runs[condition].iloc[0]


def get_children_run(_runs: DataFrame, parent_id: str) -> Series:
    return _runs[_runs['tags.mlflow.parentRunId'] == parent_id]


def get_study(run_id: str) -> Study:
    return load_pickled_artifact(run_id, 'study')


def get_run_duration(run: Series) -> float:
    return run['end_time'] - run['start_time']
