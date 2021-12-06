import pickle
from typing import Any, Dict, Hashable, Tuple

import mlflow
from mlflow import get_experiment_by_name, ActiveRun, start_run
from mlflow import log_artifact, log_metric
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from optuna import Study
from pandas import DataFrame, Series

from hcve_lib.custom_types import ValueWithCI, SplitPrediction


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


def log_metrics_ci(metrics: Dict[Any, ValueWithCI],
                   drop_ci: bool = False) -> None:
    for metric_name, metric_value in metrics.items():

        if not drop_ci:
            log_metric(f'{metric_name}_l', metric_value['ci'][0])

        log_metric(metric_name, metric_value['mean'])

        if not drop_ci:
            log_metric(f'{metric_name}_r', metric_value['ci'][1])


def is_root_run(_runs: DataFrame) -> Series:
    return _runs['tags.mlflow.parentRunId'].isna()


def get_completed_runs(experiment_name: str) -> DataFrame:
    _runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id)
    return _runs[_runs['status'] == 'FINISHED']


def get_latest_run(runs: DataFrame, condition: Any) -> Series:
    return runs[condition].iloc[0]


def get_latest_root_run(runs: DataFrame) -> Series:
    return get_latest_run(runs, is_root_run(runs))


def search_latest_root_completed_run(
    experiment_name: str,
    run_name: str = None,
) -> Tuple[Run, Dict[Hashable, SplitPrediction]]:
    runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id,
        'attributes.status="FINISHED"' +
        (f' and tags.mlflow.runName="{run_name}"' if run_name else ''),
        max_results=1,
        output_format='list')
    if len(runs) > 0:
        try:
            result = load_pickled_artifact(runs[0].info.run_id, 'result')
        except OSError:
            result = None

        return runs[0], result

    else:
        raise Exception("Not applicable runs found")


def search_latest_root_completed_run_children(
    experiment_name: str,
    run_name: str = None,
) -> Tuple[Run, Dict[Hashable, SplitPrediction]]:
    root_run, _, = search_latest_root_completed_run(experiment_name, run_name)
    runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id,
        f'tags.mlflow.parentRunId = "{root_run.info.run_id}"',
        output_format='list')
    return runs


def get_children_runs(_runs: DataFrame, parent_id: str) -> Series:
    return _runs[_runs['tags.mlflow.parentRunId'] == parent_id]


def get_study(run_id: str) -> Study:
    return load_pickled_artifact(run_id, 'study')


def get_run_duration(run: Series) -> float:
    return run['end_time'] - run['start_time']


def get_active_experiment_id():
    return mlflow.active_run().info.experiment_id


def get_experiment_id():
    return mlflow.active_run().info.experiment_id


def log_study(study) -> None:
    log_text(study.best_trial.user_attrs['pipeline'], 'pipeline.txt')
    mlflow.log_param(
        'hyperparameters',
        study.best_trial.user_attrs['hyperparameters'],
    )
    log_metrics_ci(study.best_trial.user_attrs['metrics'])
    log_pickled(study, 'study')


def log_optimizer(optimizer) -> None:
    log_text(optimizer.study.best_trial.user_attrs['pipeline'], 'pipeline.txt')
    mlflow.log_param('hyperparameters',
                     optimizer.study.best_trial.user_attrs['hyperparameters'])
    log_metrics_ci(optimizer.study.best_trial.user_attrs['metrics'])
    log_pickled(optimizer.study.best_trial.user_attrs['result_slit'],
                'result_slit')
    log_pickled(optimizer.study, 'study')


def update_nested_run(run_name: str) -> ActiveRun:
    runs = mlflow.search_runs(
        filter_string=
        f'tags.mlflow.parentRunId = "{mlflow.active_run().info.run_id}" and tags.mlflow.runName = "{run_name}"',
        output_format='list',
    )

    if len(runs) > 1:
        raise Exception("Duplicated run name")

    return start_run(runs[0].info.run_id, nested=True)


def encode_run_name(identifier: Any) -> str:
    if isinstance(identifier, str):
        return identifier
    elif isinstance(identifier, tuple):
        return ','.join(map(str, identifier))


def get_subruns_of_latest(experiment_name: str, root_name: str):
    runs = get_completed_runs(experiment_name)
    root_runs = runs[is_root_run(runs)]
    run_method = root_runs.query(
        f'`tags.mlflow.runName` == "{root_name}"').iloc[0]
    return get_children_runs(runs, run_method['run_id'])
