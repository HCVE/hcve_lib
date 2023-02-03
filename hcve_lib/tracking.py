from collections import defaultdict

import io
import logging
import pickle
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from logging import Logger
from sys import stdout
from typing import Any, Dict, Hashable, Tuple, Optional, Union, List

import mlflow
from toolz import dissoc

from hcve_lib.custom_types import ValueWithCI, Result, ExceptionValue
from hcve_lib.functional import pipe, valmap_
from hcve_lib.log_output import capture_output, log_output
from hcve_lib.utils import is_noneish
from hcve_lib.visualisation import display_html
from mlflow import get_experiment_by_name, ActiveRun, start_run, get_run, get_experiment, set_tag, create_experiment, \
    set_experiment, MlflowException
from mlflow import log_artifact
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from optuna import Study
from pandas import DataFrame, Series


def log_pickled(data: Any, path: str) -> None:
    with TempDir() as tmp_dir:
        path = tmp_dir.path() + '/' + path
        with open(path, 'bw') as file:
            pickle.dump(data, file, protocol=5)

        log_artifact(path)


def load_pickled_artifact(run_id: str, path: str) -> Any:
    client = MlflowClient()
    path = client.download_artifacts(run_id, path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_run_results(run_id: str, load_models: bool = False) -> Result:
    try:
        result = load_pickled_artifact(run_id, 'result')
        if load_models:
            models = load_pickled_artifact(run_id, 'result_models')
            return {key: {**result[key], 'model': models[key]} for key in result.keys()}
        else:
            return result

    except OSError:
        raise Exception(f'Run {run_id} does not have artifact "result"')


def load_subrun_results(run_id: str) -> Dict[Hashable, Result]:
    return valmap_(
        get_subruns(run_id),
        lambda run: load_run_results(run.info.run_id),
    )


def get_subruns(root_run_id: str) -> Dict[Hashable, Run]:
    root_run = get_run(root_run_id)

    children_runs = mlflow.search_runs(
        root_run.info.experiment_id, f'tags.mlflow.parentRunId = "{root_run.info.run_id}"', output_format='list'
    )

    return pipe(
        ((run.data.tags['mlflow.runName'], run) for run in children_runs),
        dict,
    )


def load_group_results(group_id: str, **kwargs) -> Dict[Hashable, Result]:
    group_runs = mlflow.search_runs(
        filter_string=f'tags.group_id="{group_id}"',
        output_format='list',
        search_all_experiments=True,
    )
    return pipe(
        ((index, load_run_results(run.info.run_id, **kwargs)) for index, run in enumerate(group_runs)),
        dict,
    )


def display_run_info(run_id: str):
    run = get_run(run_id)
    display_html(
        wrap_table(
            '    <tr>' \
            '        <td style="text-align: left">Name</td>' \
            f'        <th  style="text-align: left">{run.data.tags["mlflow.runName"]}</th>' \
            '    </tr>' +
            get_context_run_info(run)
        )
    )


def display_run_info_context(run_id: str) -> None:
    run = get_run(run_id)
    display_html(wrap_table(get_context_run_info(run)))


def get_context_run_info(run: Run) -> str:
    return \
        '    <tr>' \
        '        <td>Start time</td>' \
        f'       <th>{datetime.utcfromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M")}</th>' \
        '    </tr>' \
        '    <tr>' \
        '        <td>Experiment</td>' \
        f'       <th>{get_experiment(run.info.experiment_id).name}</th>' \
        '    </tr>' \
        '    <tr>' \
        '        <td>Run ID</td>' \
        f'        <th>{run.info.run_id}</th>' \
        '    </tr>'


def wrap_table(content: str) -> str:
    return f'<table style="text-align: left">{content}</table>'


def log_metrics(
    metrics: Dict[Any, ValueWithCI],
    drop_ci: bool = False,
    prefix: str = '',
) -> None:
    for metric_name, metric_value in metrics.items():
        metric_name_ = prefix + metric_name
        if isinstance(metric_value, ExceptionValue):
            log_exception_value(metric_name_, metric_value)
        else:
            if not drop_ci:
                log_metric(f'{metric_name_}_l', metric_value['ci'][0])

            log_metric(metric_name_, metric_value['mean'])

            if not drop_ci:
                log_metric(f'{metric_name_}_r', metric_value['ci'][1])


def log_metrics_single(metrics: Dict[str, Union[float, ExceptionValue]], ) -> None:
    for name, value in metrics.items():
        log_metric(name, value)


def log_metric(name: str, value: Union[float, ExceptionValue]) -> None:
    if isinstance(value, ExceptionValue):
        log_exception_value(name, value)
    elif not is_noneish(value):
        mlflow.log_metric(name, value)


def log_exception_value(name: str, value: ExceptionValue) -> None:
    mlflow.set_tag(
        f'{name}_exception',
        str(value.exception),
    )
    mlflow.set_tag(
        f'{name}_value',
        str(value.value),
    )


def is_root_run(_runs: DataFrame) -> Series:
    return _runs['tags.mlflow.parentRunId'].isna()


def get_completed_runs(experiment_name: str) -> DataFrame:
    _runs = mlflow.search_runs(get_experiment_by_name(experiment_name).experiment_id)
    return _runs[_runs['status'] == 'FINISHED']


def get_latest_run(runs: DataFrame, condition: Any) -> Series:
    return runs[condition].iloc[0]


def get_latest_root_run(runs: DataFrame) -> Series:
    return get_latest_run(runs, is_root_run(runs))


def search_latest_root_completed_run(
    experiment_name: str,
    run_name: str = None,
    additional_filter: Optional[str] = None,
) -> Tuple[Run, Result]:
    runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id,
        'attributes.status="FINISHED"' + (f' and tags.mlflow.runName="{run_name}"' if run_name else '') +
        (f' and {additional_filter}' if additional_filter else ""),
        max_results=1,
        output_format='list'
    )
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
    additional_filter: str = None,
) -> Tuple[Run, Result]:
    root_run, _, = search_latest_root_completed_run(experiment_name, run_name)
    search_filter = f'tags.mlflow.parentRunId = "{root_run.info.run_id}"' + (
        '' if not additional_filter else f' and {additional_filter}'
    )
    runs = mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id, search_filter, output_format='list'
    )
    return runs


def get_children_runs(_runs: DataFrame, parent_id: str) -> Series:
    return _runs[_runs['tags.mlflow.parentRunId'] == parent_id]


def get_study(run_id: str) -> Study:
    return load_pickled_artifact(run_id, 'study')


def get_run_duration(run: Series) -> float:
    return run['end_time'] - run['start_time']


def get_active_experiment_id():
    active_run = mlflow.active_run()
    if active_run:
        return active_run.info.experiment_id
    else:
        return None


def get_experiment_id(name: str = None):
    if name:
        run = mlflow.get_experiment_by_name(name)
    else:
        run = mlflow.active_run().info

    return run.experiment_id


def log_study(study) -> None:
    mlflow.log_text(study.best_trial.user_attrs['pipeline'], 'pipeline.txt')
    mlflow.log_param(
        'hyperparameters',
        study.best_trial.user_attrs['hyperparameters'],
    )
    log_metrics(study.best_trial.user_attrs['metrics'])
    log_pickled(study, 'study')


def log_optimizer(optimizer) -> None:
    mlflow.log_text(optimizer.study.best_trial.user_attrs['pipeline'], 'pipeline.txt')
    mlflow.log_param('hyperparameters', optimizer.study.best_trial.user_attrs['hyperparameters'])
    log_metrics(optimizer.study.best_trial.user_attrs['metrics'])
    log_pickled(optimizer.study.best_trial.user_attrs['result_split'], 'result_split')
    log_pickled(optimizer.study, 'study')


def update_nested_run(run_name: str) -> ActiveRun:
    active_run_id = mlflow.active_run().info.run_id
    runs = mlflow.search_runs(
        filter_string=(f'tags.mlflow.parentRunId = "{active_run_id}"'
                       f'and tags.`mlflow.runName` = "{run_name}"'),
        output_format='list',
        experiment_ids=mlflow.active_run().info.experiment_id,
    )
    if len(runs) == 0:
        raise Exception(f'Sub runs under \'{active_run_id}\' with name \'{run_name}\' not found')
    elif len(runs) > 1:
        raise Exception("Duplicated run name")

    return start_run(runs[0].info.run_id, nested=True)


def encode_run_name(identifier: Any) -> str:
    if isinstance(identifier, str):
        return identifier
    elif isinstance(identifier, tuple):
        return ','.join(map(str, identifier))
    else:
        raise Exception("Not supported type")


def get_subruns_of(
    experiment_name: str,
    root_name: str,
    additional_filter: Optional[str] = None,
):
    runs = get_completed_runs(experiment_name)
    root_runs = runs[is_root_run(runs)]
    run_method = root_runs.query(
        f'`tags.mlflow.runName` == "{root_name}"' + (f' and {additional_filter}' if additional_filter else '')
    ).iloc[0]
    return get_children_runs(runs, run_method['root_run_id'])


def log_early_stopping(logger: logging.Logger, iterations: int):
    logger.info(f'Early stopping after {iterations} iterations')


def get_run_info(run_id: str) -> Dict:
    run = get_run(run_id)
    return {'run_name': run.data.tags['mlflow.runName'], 'experiment_name': get_experiment(run.info.experiment_id).name}


def log_model(result: Result) -> None:
    result_without_models = valmap_(
        result,
        lambda prediction: dissoc(prediction, 'model'),
    )
    log_pickled(result_without_models, 'result')

    only_models = valmap_(
        result,
        lambda prediction: prediction['model'],
    )

    log_pickled(only_models, 'result_models')


def get_logger(name: str = 'default') -> Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)
    return logger


def log_to_variable(logger: Logger) -> None:
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)


@contextmanager
def get_standard_repeat_context(
    method_name: str,
    repeat_id: str,
    random_state: int,
):
    with start_run(run_name=f'repeat {repeat_id}', nested=True) as mlflow:
        with capture_output() as buffer:
            # TODO: not used now
            # for key, value in asdict(configuration).items():
            #     set_tag(key, value)
            set_tag("repeat_id", repeat_id)
            set_tag("random_state", random_state)

            yield

        log_output(buffer())


def define_experiment(name: str) -> None:
    try:
        create_experiment(name)
    except MlflowException:
        pass
    set_experiment(experiment_name=name)
