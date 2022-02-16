import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from functools import partial
from multiprocessing import Pool, cpu_count
from time import process_time
from typing import Callable, Dict, List, Union, Iterable, Hashable, Optional, Mapping, Sequence, TypeVar, Any

import toolz
import yaml
from mlflow import active_run, get_experiment, start_run, set_tag
from numpy.random import rand
from optuna import create_study, Trial
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler
import optuna
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from toolz import compose_left, merge, identity, dissoc
from toolz.curried import valmap, valfilter

from hcve_lib.custom_types import SplitPrediction, Estimator, Target, SplitInput, Splits, Result, Splitter
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index
from hcve_lib.functional import star_args, pipe, always, t, statements
from hcve_lib.splitting import filter_missing_features
from hcve_lib.tracking import log_metrics_ci, get_active_experiment_id
from hcve_lib.utils import empty_dict, list_to_dict_by_keys, list_to_dict_index, cross_validate_apply_mask, \
    partial2_args, loc, random_seed, noop_context_manager, noop
from hcve_lib.utils import NonDaemonPool

default_cv = KFold(n_splits=10).split
CROSS_VALIDATE_KEY = 'cross_validate'


def configuration_to_params(dictionary: Dict) -> Dict:
    return_value = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                return_value["%s__%s" % (key, key2)] = value2
        else:
            return_value[key] = value

    return return_value


OptimizeEvaluate = Callable[[
    Trial,
    Callable,
    int,
    DataFrame,
    Target,
], float]


class Optimize:
    fit_best_model: Optional[Estimator]

    def __init__(
        self,
        get_pipeline: Callable,
        predict: Callable,
        optuna_suggest: Callable,
        objective_evaluate: OptimizeEvaluate,
        random_state: int,
        get_splits=None,
        optimize_params=empty_dict,
        mlflow_callback=None,
        optimize_callbacks: List[Callable] = None,
        study_name: str = None,
        catch_exceptions: bool = True,
        logger: logging.Logger = None,
    ):
        random_seed(random_state)

        if optimize_callbacks is None:
            optimize_callbacks = []

        if mlflow_callback is True:
            mlflow_callback = MLflowCallback(nest_trials=True)

        if not study_name:
            if active_run():
                self.study_name = get_experiment(
                    active_run().info.experiment_id).name
            else:
                self.study_name = 'study'

        self.get_splits = get_splits
        self.get_pipeline = get_pipeline
        self.optuna_suggest = optuna_suggest
        # TODO
        warnings.filterwarnings("ignore")
        self.study = create_study(
            direction='maximize',
            study_name=self.study_name,
            sampler=TPESampler(seed=random_state),
        )

        self.objective_evaluate = objective_evaluate
        self.predict_callback = predict
        self.optimize_params = optimize_params
        self.fit_best_model = None
        self.mlflow_callback = mlflow_callback
        self.optimize_callbacks = optimize_callbacks
        self.catch_exceptions = catch_exceptions
        self.logger = logger
        self.random_state = random_state

    def fit(self, X, y):
        if self.mlflow_callback:
            decorator = self.mlflow_callback.track_in_mlflow()
        else:
            decorator = identity
        self.study.optimize(
            decorator(
                compose_left(
                    self.optuna_suggest,
                    star_args(partial(self._objective_instantiate)),
                    star_args(
                        partial(
                            self.objective_evaluate,
                            X=X,
                            y=y,
                            get_splits=self.get_splits,
                            log_mlflow=self.mlflow_callback is not None,
                            logger=self.logger,
                            predict=self.predict_callback,
                            random_state=self.random_state,
                        )),
                )),
            **merge(
                dict(n_trials=1),
                self.optimize_params,
            ),
            catch=((Exception, ArithmeticError,
                    RuntimeError) if self.catch_exceptions else ()),
            callbacks=[
                *self.optimize_callbacks,
                *([self.mlflow_callback] if self.mlflow_callback else [])
            ],
        )

    def _objective_instantiate(self, trial, hyperparameters):
        if CROSS_VALIDATE_KEY in hyperparameters:
            cv_hyperparameters = hyperparameters[CROSS_VALIDATE_KEY]
            rest_hyperparameters = dissoc(hyperparameters, CROSS_VALIDATE_KEY)
        else:
            cv_hyperparameters = {}
            rest_hyperparameters = hyperparameters

        trial.set_user_attr('cv_hyperparameters', cv_hyperparameters)
        trial.set_user_attr('hyperparameters', rest_hyperparameters)

        return trial, lambda X, _random_state: self._instantiate_from_hyperparameters(
            rest_hyperparameters,
            X,
            _random_state,
        )

    def _instantiate_from_hyperparameters(
        self,
        hyperparameters,
        X: DataFrame,
        random_state: int,
    ) -> Estimator:
        return self.get_pipeline(X, random_state).set_params(
            **configuration_to_params(hyperparameters))


def evaluate_optimize_splits(
    trial: Trial,
    get_pipeline: Callable,
    random_state: int,
    X: DataFrame,
    y: Target,
    predict: Callable,
    get_splits: Splitter,
    log_mlflow: bool = False,
    logger: logging.Logger = None,
):
    start_time = process_time()
    cv_hyperparams = trial.user_attrs['cv_hyperparameters']
    split_runs = cross_validate(
        X,
        y,
        get_pipeline,
        predict,
        random_state,
        get_splits=get_splits,
        n_jobs=1,
        train_test_filter_callback=partial(
            filter_missing_features,
            threshold=cv_hyperparams.get('missing_fraction', 1),
        ),
        logger=logger,
    )
    metrics = compute_metrics_ci(
        split_runs,
        [partial2_args(c_index, kwargs={
            'X': X,
            'y': y
        })],
    )
    trial.set_user_attr('result_split', split_runs)
    trial.set_user_attr('metrics', metrics)
    trial.set_user_attr('pipeline', str(get_pipeline(X, random_state).steps))

    duration = process_time() - start_time
    trial.set_user_attr('duration', duration)

    if log_mlflow:
        log_metrics_ci(metrics, drop_ci=True)

    return metrics['c_index']['mean']


def optimize_per_split(
    get_optimize: Callable[..., Optimize],
    get_splits: Callable[[DataFrame, Target], Dict[Hashable, SplitInput]],
    X: DataFrame,
    y: Target,
    mlflow_track: bool = False,
    n_jobs: Optional[int] = -1,
) -> Dict[Hashable, Optimize]:
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(get_splits(X, y)))

    splits = get_splits(X, y)

    fold_data = {
        fold_name: (
            fold_name,
            get_optimize(get_splits=always({'train_test': fold})),
            X,
            y,
            mlflow_track,
        )
        for fold_name, fold in splits.items()
    }

    return run_parallel(cross_validate_fit, fold_data, n_jobs)


def optimize_per_group(
    get_optimize: Callable[[], Optimize],
    X: DataFrameGroupBy,
    y: Target,
    mlflow_track: bool = False,
    n_jobs: Optional[int] = -1,
) -> Dict[Hashable, Optimize]:
    if n_jobs == -1:
        n_jobs = cpu_count()
    fold_data = {
        group_name: (
            group_name,
            get_optimize(),
            group_X,
            loc(group_X.index, y),
            mlflow_track,
        )
        for group_name, group_X in X
    }
    return run_parallel(cross_validate_fit, fold_data, n_jobs)


ExecutePerGroupT = TypeVar('ExecutePerGroupT')


def execute_per_group(
    callback: Callable[[str, DataFrame, Target], ExecutePerGroupT],
    X_group_by: DataFrameGroupBy,
    y: Target,
    n_jobs: Optional[int] = -1,
) -> Dict[Hashable, ExecutePerGroupT]:
    if n_jobs == -1:
        n_jobs = cpu_count()
    fold_data = {
        group_name: (
            group_X,
            loc(group_X.index, y),
        )
        for group_name, group_X in X_group_by
    }
    return run_parallel(callback, fold_data, n_jobs)


def run_parallel(function: Callable, fold_data: Dict, n_jobs: int) -> Dict:
    if n_jobs == 1:
        optimizers = list_to_dict_by_keys(
            itertools.starmap(
                function,
                fold_data.values(),
            ),
            fold_data.keys(),
        )
    else:
        with NonDaemonPool(min(len(fold_data), n_jobs)) as p:  # type: ignore
            optimizers = list_to_dict_by_keys(
                p.starmap(
                    function,
                    fold_data.values(),
                ),
                fold_data.keys(),
            )
    return optimizers


def apply_args_and_kwargs(function: Callable, args: List, kwargs: Dict):
    return function(*args, **kwargs)


class OptimizeEstimator(BaseEstimator, Optimize):
    X_train: DataFrame = None
    y_train: Optional[Target] = None

    def fit(self, X, y):
        super().fit(X, y)
        self.fit_best_model = self._instantiate_from_hyperparameters(
            self.study.best_trial.user_attrs['hyperparameters'],
            X,
            self.random_state,
        )
        if self.fit_best_model:
            try:
                self.fit_best_model.fit(X, y)
            except ArithmeticError:
                self.fit_best_model = None

            self.X_train = X
            self.y_train = y

    def predict(self, X):
        return self.fit_best_model.predict(X) if self.fit_best_model else None

    def predict_proba(self, X):
        return self.fit_best_model.predict_proba(
            X) if self.fit_best_model else None

    def predict_survival_function(self, X):
        return self.fit_best_model.predict_survival_function(
            X) if self.fit_best_model else None

    def score(self, X, y):
        return self.fit_best_model.score(X, y) if self.fit_best_model else None


GetRepeatContext = Callable[[int, int], AbstractContextManager]
OnRepeatResults = Callable[[Result, int], None]


def repeated_cross_validate(
    n_repeats: int,
    random_state: int,
    get_repeat_context: GetRepeatContext = noop_context_manager,
    on_repeat_result: OnRepeatResults = noop,
    *args,
    **kwargs,
) -> List[Result]:

    data = {
        repeat_index: statements(
            run_random_state := random_state + repeat_index,
            (
                partial(get_repeat_context, repeat_index, run_random_state),
                partial(on_repeat_result, random_state=run_random_state),
                args,
                toolz.merge(kwargs, {'random_state': run_random_state}),
            ),
        )
        for repeat_index in range(n_repeats)
    }

    return pipe(
        run_parallel(
            run_cross_validate_repeat,
            data,
            n_jobs=len(data),
        ).values(),
        list,
    )


def run_cross_validate_repeat(
    get_context: Callable[[], AbstractContextManager],
    on_result: Callable[[Result], None],
    args,
    kwargs,
):
    with get_context():
        result = cross_validate(*args, **kwargs)
        on_result(result)

    return result


def cross_validate(
    X: DataFrame,
    y: Target,
    get_pipeline: Callable[[DataFrame, int], Estimator],
    predict: Callable,
    random_state: int,
    get_splits: Splitter = None,
    train_test_filter_callback: Callable[[Series, Series], bool] = None,
    n_batches: int = 1,
    callbacks: Mapping[str, Callable] = empty_dict,
    n_jobs: int = None,
    logger: logging.Logger = None,
    mlflow_track: bool = False,
    split_hyperparameters: Optional[Mapping[str, Dict]] = empty_dict,
) -> Result:
    random_seed(random_state)
    if n_jobs is None:
        n_jobs = cpu_count()

    y_data = y['data'] if y is Dict else y

    if get_splits is None:
        splits = default_cv(X, y_data)
    else:
        splits = get_splits(X, y_data, random_state=random_state)

    splits_dict: Dict[Hashable, SplitInput]

    if isinstance(splits, Sequence):
        splits_dict = list_to_dict_index(splits)
    elif isinstance(splits, Dict):
        splits_dict = splits
    else:
        raise TypeError('Incompatible splits')

    models = valmap(
        lambda _: get_pipeline(X, random_state),
        splits_dict,
    )

    if split_hyperparameters:
        for split_name, model in models.items():
            model.set_params(
                **configuration_to_params(split_hyperparameters[split_name]))

    column_masks = get_column_mask(splits_dict, X, train_test_filter_callback)

    if logger:
        removed_features = pipe(
            column_masks,
            valmap(valfilter(identity)),
            valmap(lambda k: list(k.keys())),
        )
        logger.debug('\n' + yaml.dump(removed_features))

    for repeat_index in range(n_batches):
        logging.debug(f'Chunk {repeat_index}')

        models = cross_validate_train(
            X,
            y,
            models,
            splits_dict,
            column_masks,
            n_jobs=n_jobs,
            mlflow_track=mlflow_track,
            logger=logger,
            random_state=random_state,
        )

        if 'report_batch' in callbacks:
            callbacks['report_batch'](models, repeat_index + 1, n_batches)

    runs = list(
        cross_validate_predict(
            X,
            y,
            predict,
            splits_dict,
            random_state,
            column_masks,
            models,
        ))

    if isinstance(splits, Dict):
        return list_to_dict_by_keys(runs, splits.keys())
    else:
        return list_to_dict_index(runs)


def get_removed_features_from_mask(
    column_masks: Dict[Hashable, Dict[Hashable, bool]]
) -> Dict[Hashable, List[str]]:
    return pipe(
        column_masks,
        valmap(lambda masks: valfilter(identity, masks)),
        valmap(lambda k: list(k.keys())),
    )


def get_column_mask(
    splits: Splits,
    X: DataFrame,
    train_test_filter_callback: Callable[[Series, Series], bool] = None,
) -> Dict[Hashable, Dict[Hashable, bool]]:
    if train_test_filter_callback:
        return dict(
            get_column_mask_filter(
                X,
                splits,
                train_test_filter_callback,
            ))
    else:
        return get_columns_mask_default(X, splits)


def get_columns_mask_default(
    X: DataFrame,
    splits: Splits,
) -> Dict[Hashable, Dict[Hashable, bool]]:
    return valmap(lambda _: {column: False for column in X.columns}, splits)


def get_column_mask_filter(
    X: DataFrame,
    splits: Splits,
    _train_test_filter: Callable[[DataFrame, DataFrame], bool],
) -> Iterable[Dict[Hashable, bool]]:
    for fold_name, (train, test) in splits.items():
        X_train = X.loc[train]
        X_test = X.loc[test]
        if len(X_test) == 0 or len(X_train) == 0:
            yield fold_name, {}
        else:
            yield fold_name, {
                column_name: _train_test_filter(
                    X_train[column_name],
                    X_test[column_name],
                )
                for column_name in X
            }


def cross_validate_train(
    X: DataFrame,
    y: Target,
    models: List[Estimator],
    splits_dict: Splits,
    filtered_columns: Dict[Hashable, Dict[Hashable, bool]],
    random_state: int,
    n_jobs: int = -1,
    mlflow_track: bool = False,
    logger: logging.Logger = None,
) -> List[Estimator]:
    if n_jobs == -1:
        n_jobs = cpu_count()

    fold_data = [(
        fold_name,
        models[fold_name],
        cross_validate_preprocess(
            loc(train_split, X, ignore_not_present=True, logger=logger),
            filtered_columns[fold_name],
        ),
        loc(train_split, y, ignore_not_present=True, logger=logger),
        random_state,
        mlflow_track,
    ) for fold_name, (train_split, test_split) in splits_dict.items()]

    if n_jobs == 1:
        models = list_to_dict_by_keys(
            itertools.starmap(
                cross_validate_fit,
                fold_data,
            ),
            splits_dict.keys(),
        )
    else:
        with Pool(min(len(splits_dict), n_jobs)) as p:
            models = list_to_dict_by_keys(
                p.starmap(
                    cross_validate_fit,
                    fold_data,
                ),
                splits_dict.keys(),
            )
    return models


def cross_validate_preprocess(
    data: DataFrame,
    mask: Dict[Hashable, bool],
):
    return pipe(
        data,
        partial(cross_validate_apply_mask, mask),
    )


def cross_validate_predict(
    X: DataFrame,
    y: Target,
    predict: Callable,
    splits: Splits,
    random_state: int,
    filtered_columns: Dict[Hashable, Dict[Hashable, bool]],
    models: Dict[Hashable, Estimator],
) -> Iterable[SplitPrediction]:
    for name, split in splits.items():
        yield predict(
            cross_validate_preprocess(X, filtered_columns[name]),
            y,
            model=models[name],
            split=split,
            random_state=random_state,
        )


def predict_proba(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
    random_state: int,
) -> SplitPrediction:
    y_score = DataFrame(
        model.predict_proba(loc(split[1], X)),
        index=split[1],
    )
    return SplitPrediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=y_score,
        model=model,
        random_state=random_state,
    )


def predict_survival(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
    random_state: int,
) -> SplitPrediction:
    return SplitPrediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=predict_survival_(
            model,
            loc(
                split[1],
                X,
                ignore_not_present=True,
            ),
        ),
        model=model,
        random_state=random_state,
    )


def predict_survival_(model: Estimator, X_test: DataFrame) -> Series:
    return Series(
        model.predict(X_test),
        index=X_test.index,
    )


def predict_survival_dsm(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
) -> float:
    return SplitPrediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=Series(
            model.predict(X.loc[split[1]]).flatten(),
            index=X.loc[split[1]].index,
        ),
        model=model,
    )


def predict_predict(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
) -> SplitPrediction:
    return SplitPrediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=Series(
            model.predict(X.loc[split[1]]),
            index=X.loc[split[1]].index,
        ),
        model=model,
    )


def cross_validate_fit(
    fold_name: str,
    estimator: Estimator,
    X: DataFrame,
    y: Target,
    random_state: int,
    mlflow_track: bool = False,
) -> Estimator:
    random_seed(random_state)
    if mlflow_track:
        with start_run(
                run_name=str(fold_name),
                nested=True,
                experiment_id=get_active_experiment_id(),
        ):
            estimator.fit(X, y)
    else:
        estimator.fit(X, y)

    return estimator


def series_to_target(series: Series) -> Target:
    return {'name': series.name, 'data': series}
