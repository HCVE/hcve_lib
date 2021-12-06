import itertools
import logging
from abc import ABC, abstractmethod
from functools import partial
from math import inf
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Union, Iterable, Hashable, Optional, Mapping, Sequence, TypeVar

import numpy as np
import yaml
from mlflow import active_run, get_experiment, start_run, set_tag, log_metrics
from optuna import create_study, Trial
from optuna.integration import MLflowCallback
from pandas import DataFrame, Series, Index
from pandas.core.groupby import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from toolz import compose_left, merge, identity, dissoc
from toolz.curried import valmap, valfilter

from hcve_lib.custom_types import SplitPrediction, Estimator, Target, SplitInput, Splits
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index, get_1_class_y_score
from hcve_lib.functional import star_args, pipe, always, statements, t
from hcve_lib.splitting import filter_missing_features
from hcve_lib.tracking import log_metrics_ci, get_active_experiment_id
from hcve_lib.utils import empty_dict, list_to_dict_by_keys, list_to_dict_index, cross_validate_apply_mask, \
    partial2_args, loc

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


class Method(ABC):
    @abstractmethod
    def get_optuna_hyperparameters(self):
        ...


OptimizeEvaluate = Callable[[
    Trial,
    Callable,
    DataFrame,
    Target,
], float]


class Optimize:
    def __init__(
        self,
        get_pipeline: Callable,
        predict_callback: Callable,
        optuna: Callable,
        objective_evaluate: OptimizeEvaluate,
        get_splits=None,
        optimize_params=empty_dict,
        mlflow_callback=None,
        optimize_callbacks: List[Callable] = None,
        study_name: str = None,
        catch_exceptions: bool = True,
        logger: logging.Logger = None,
    ):
        if optimize_callbacks is None:
            optimize_callbacks = []

        if mlflow_callback is True:
            mlflow_callback = MLflowCallback(nest_trials=True)

        if not study_name and active_run():
            self.study_name = get_experiment(
                active_run().info.experiment_id).name
        self.get_splits = get_splits
        self.get_pipeline = get_pipeline
        self.optuna = optuna
        self.study = create_study(direction='maximize',
                                  study_name=self.study_name)
        self.objective_evaluate = objective_evaluate
        self.predict_callback = predict_callback
        self.optimize_params = optimize_params
        self.fit_best_model = None
        self.mlflow_callback = mlflow_callback
        self.optimize_callbacks = optimize_callbacks
        self.catch_exceptions = catch_exceptions
        self.logger = logger

    def fit(self, X, y):
        if self.mlflow_callback:
            decorator = self.mlflow_callback.track_in_mlflow()
        else:
            decorator = identity

        self.study.optimize(
            decorator(
                compose_left(
                    self.optuna,
                    star_args(partial(self._objective_instantiate)),
                    star_args(
                        partial(
                            self.objective_evaluate,
                            X=X,
                            y=y,
                            get_splits=self.get_splits,
                            log_mlflow=self.mlflow_callback is not None,
                            logger=self.logger,
                            predict_callback=self.predict_callback,
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
        return trial, lambda X: self._instantiate_from_hyperparameters(
            rest_hyperparameters,
            X,
        )

    def _instantiate_from_hyperparameters(
        self,
        hyperparameters,
        X: DataFrame,
    ) -> Estimator:
        return self.get_pipeline(X).set_params(
            **configuration_to_params(hyperparameters))


def evaluate_optimize_splits(
    trial: Trial,
    get_pipeline: Callable,
    X: DataFrame,
    y: Target,
    predict_callback: Callable,
    get_splits,
    log_mlflow: bool = False,
    logger: logging.Logger = None,
):
    cv_hyperparams = trial.user_attrs['cv_hyperparameters']
    split_runs = cross_validate(
        X,
        y,
        get_pipeline,
        predict_callback,
        get_splits(X, y),
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
    trial.set_user_attr('result_slit', split_runs)
    trial.set_user_attr('metrics', metrics)
    trial.set_user_attr('pipeline', str(get_pipeline(X).steps))
    if log_mlflow:
        log_metrics_ci(metrics, drop_ci=True)
        # TODO
        # set_tag("inner_cv", get_splits.__name__)
    return metrics['c_index']['mean']


def optimize_per_split(
    get_optimize: Callable[[], Optimize],
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
            statements(
                optimize := get_optimize(),
                setattr(optimize, 'get_splits', always({'tt': fold})),
                optimize,
            ),
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
            group_name,
            group_X,
            loc(group_X.index, y),
        )
        for group_name, group_X in X_group_by
    }
    return run_parallel(callback, fold_data, n_jobs)


def run_parallel(function, fold_data, n_jobs):
    if n_jobs == 1:
        optimizers = list_to_dict_by_keys(
            itertools.starmap(
                function,
                fold_data.values(),
            ),
            fold_data.keys(),
        )
    else:
        with Pool(min(len(fold_data), n_jobs)) as p:  # type: ignore
            optimizers = list_to_dict_by_keys(
                p.starmap(
                    function,
                    fold_data.values(),
                ),
                fold_data.keys(),
            )
    return optimizers


class OptimizeEstimator(BaseEstimator, Optimize):
    X_train: DataFrame = None
    y_train: Optional[Target] = None
    fit_best_model: Optional[Estimator] = None

    def fit(self, X, y):
        super().fit(X, y)
        self.fit_best_model = self._instantiate_from_hyperparameters(
            self.study.best_trial.user_attrs['hyperparameters'],
            X,
        )
        if self.fit_best_model:
            self.fit_best_model.fit(X, y)
            self.X_train = X
            self.y_train = y

    def predict(self, X):
        return self.fit_best_model.predict(X)

    def predict_proba(self, X):
        return self.fit_best_model.predict_proba(X)

    def predict_survival_function(self, X):
        return self.fit_best_model.predict_survival_function(X)


def cross_validate(
    X: DataFrame,
    y: Union[Target, Union[Series, np.recarray]],
    get_pipeline: Callable[[DataFrame], Estimator],
    predict: Callable,
    splits: Union[Iterable[SplitInput], Dict[Hashable, SplitInput]] = None,
    train_test_filter_callback: Callable[[Series, Series], bool] = None,
    n_batches: int = 1,
    callbacks: Mapping[str, Callable] = empty_dict,
    n_jobs: int = None,
    logger: logging.Logger = None,
    mlflow_track: bool = False,
    split_hyperparameters: Mapping[str, Dict] = empty_dict
) -> Dict[Hashable, SplitPrediction]:
    if train_test_filter_callback is None:
        train_test_filter_callback = filter_missing_features
    if n_jobs is None:
        n_jobs = cpu_count()
    y_data = y['data'] if y is Dict else y

    if splits is None:
        splits = default_cv(X, y_data)

    splits_dict: Dict[Hashable, SplitInput]

    if isinstance(splits, Sequence):
        splits_dict = list_to_dict_index(splits)
    elif isinstance(splits, Dict):
        splits_dict = splits
    else:
        raise TypeError('Incompatible splits')

    models = valmap(lambda _: get_pipeline(X), splits_dict)

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
        )

        if 'report_batch' in callbacks:
            callbacks['report_batch'](models, repeat_index + 1, n_batches)

    runs = list(
        cross_validate_predict(
            X,
            y,
            predict,
            splits_dict,
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
    n_jobs: int = -1,
    mlflow_track: bool = False,
) -> List[Estimator]:
    if n_jobs == -1:
        n_jobs = cpu_count()

    fold_data = [(
        fold_name,
        models[fold_name],
        cross_validate_preprocess(
            X.loc[train_split],
            filtered_columns[fold_name],
        ),
        loc(train_split, y),
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
    filtered_columns: Dict[Hashable, Dict[Hashable, bool]],
    models: Dict[Hashable, Estimator],
) -> Iterable[SplitPrediction]:
    for name, split in splits.items():
        yield predict(
            cross_validate_preprocess(X, filtered_columns[name]),
            y,
            model=models[name],
            split=split,
        )


def predict_proba(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
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
    )


def predict_survival(
    X: DataFrame,
    y: Target,
    split: SplitInput,
    model: Estimator,
) -> SplitPrediction:
    return SplitPrediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=predict_survival_(model, loc(split[1], X)),
        model=model,
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
            model.predict(X.iloc[split[1]]).flatten(),
            index=X.iloc[split[1]].index,
        ),
        model=model,
    )


def cross_validate_fit(
    fold_name: str,
    estimator: Estimator,
    X: DataFrame,
    y: Target,
    mlflow_track: bool = False,
) -> Estimator:
    if mlflow_track:
        with start_run(
                run_name=str(fold_name),
                nested=True,
                experiment_id=get_active_experiment_id(),
        ):
            estimator.fit(X, y['data'])
    else:
        estimator.fit(X, y['data'])

    return estimator


def series_to_target(series: Series) -> Target:
    return {'name': series.name, 'data': series}
