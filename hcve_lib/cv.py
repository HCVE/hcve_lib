import itertools
import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Any, Union, Iterable, Hashable

import yaml
from mlflow import active_run, get_experiment
from optuna import create_study, Trial
from optuna.integration import MLflowCallback
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from toolz import compose_left, merge, identity, dissoc
from toolz.curried import valmap, valfilter, map

from common import brier
from hcve_lib.custom_types import FoldPrediction, Estimator, EstimatorProba, Target, FoldInput
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index
from hcve_lib.functional import star_args, pipe
from hcve_lib.tracking import log_metrics_ci
from hcve_lib.utils import empty_dict, index_data, list_to_dict_by_keys, map_groups_iloc, subtract_lists, \
    list_to_dict_index, percent_missing, partial2

default_cv = KFold(n_splits=10).split
__CROSS_VALIDATE__ = '__cross_validate__'


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


class Optimize(BaseEstimator):
    def __init__(
        self,
        get_pipeline: Callable,
        objective: Callable,
        scoring: Callable,
        predict_callback: Callable,
        cv,
        optimize_params=empty_dict,
        mlflow_callback=None,
        optimize_callbacks: List[Callable] = None,
        study_name: str = None,
        logger: logging.Logger = None,
    ):
        if optimize_callbacks is None:
            optimize_callbacks = []

        if mlflow_callback is True:
            mlflow_callback = MLflowCallback(nest_trials=True)

        if not study_name and active_run():
            study_name = get_experiment(active_run().info.experiment_id).name

        self.cv = cv
        self.get_pipeline = get_pipeline
        self.objective = objective
        self.study = create_study(direction='maximize', study_name=study_name)
        self.scoring = scoring
        self.predict_callback = predict_callback
        self.optimize_params = optimize_params
        self.fit_best_model = None
        self.mlflow_callback = mlflow_callback
        self.optimize_callbacks = optimize_callbacks
        self.logger = logger

    def fit(self, X, y):
        if self.mlflow_callback:
            decorator = self.mlflow_callback.track_in_mlflow()
        else:
            decorator = identity

        self.study.optimize(
            decorator(
                compose_left(
                    self.objective,
                    star_args(partial(self._objective_instantiate)),
                    star_args(partial(self._objective_evaluate, X=X, y=y)),
                )),
            **merge(
                dict(n_trials=1),
                self.optimize_params,
            ),
            callbacks=[
                *self.optimize_callbacks,
                *([self.mlflow_callback] if self.mlflow_callback else [])
            ],
        )
        # TODO:
        # self.fit_best_model = self._instantiate_from_hyperparameters(
        #     self.study.best_trial.user_attrs['hyperparameters'])
        # self.fit_best_model.fit(X, y)

    def predict(self, X):
        return self.fit_best_model.predict(X)

    def predict_proba(self, X):
        return self.fit_best_model.predict_proba(X)

    def _objective_instantiate(self, trial, hyperparameters):
        if __CROSS_VALIDATE__ in hyperparameters:
            cv_hyperparameters = hyperparameters[__CROSS_VALIDATE__]
            rest_hyperparameters = dissoc(hyperparameters, __CROSS_VALIDATE__)
        else:
            cv_hyperparameters = {}
            rest_hyperparameters = hyperparameters

        trial.set_user_attr('cv_hyperparameters', cv_hyperparameters)
        trial.set_user_attr('hyperparameters', rest_hyperparameters)
        return trial, lambda X: self._instantiate_from_hyperparameters(
            rest_hyperparameters,
            X,
        )

    def _objective_evaluate(
        self,
        trial: Trial,
        get_pipeline,
        X: DataFrame,
        y: Target,
    ):
        cv_hyperparams = trial.user_attrs['cv_hyperparameters']
        result = cross_validate(
            X,
            y,
            get_pipeline,
            self.predict_callback,
            self.cv,
            n_jobs=1,
            train_test_filter=lambda x_train, x_test: filter_missing_features(
                x_train,
                x_test,
                threshold=cv_hyperparams.get('missing_fraction', 1)),
            logger=self.logger,
        )
        metrics = compute_metrics_ci(
            result['predictions'],
            [c_index, partial2(brier, kwargs={'time_point': 365 * 3})],
            y_true=y,
        )
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('pipeline', str(get_pipeline(X).steps))
        trial.set_user_attr(
            'mask',
            json.dumps(get_removed_features_from_mask(result['column_masks'])))
        if self.mlflow_callback:
            log_metrics_ci(metrics)
        return metrics['c_index']['mean']

    def _instantiate_from_hyperparameters(self, hyperparameters, X: DataFrame):
        return self.get_pipeline(X).set_params(
            **configuration_to_params(hyperparameters))


def cross_validate(
    X: DataFrame,
    y: Target,
    get_pipeline: Callable[[DataFrame], Estimator],
    predict: Callable,
    splits: Union[Iterable[FoldInput], Dict[Any, FoldInput]] = None,
    train_test_filter: Callable[[Series, Series], bool] = None,
    n_batches: int = 1,
    callbacks: Dict[str, Callable] = empty_dict,
    n_jobs: int = None,
    logger: logging.Logger = None,
) -> Dict[Any, FoldPrediction]:
    if n_jobs is None:
        n_jobs = cpu_count()

    if splits is None:
        splits = default_cv(X, y)

    splits_list = splits.values() if isinstance(splits, Dict) else splits
    splits_list = list(splits_list)

    models = [get_pipeline(X) for _ in range(len(splits_list))]

    column_masks = get_column_mask(splits_list, X, train_test_filter)

    if logger:
        removed_features = pipe(
            column_masks,
            map(valfilter(identity)),
            map(lambda k: list(k.keys())),
            list,
        )
        logger.debug('\n' + yaml.dump(removed_features))

    for repeat_index in range(n_batches):
        logging.debug(f'Chunk {repeat_index}')

        models = cross_validate_train(
            X,
            y,
            models,
            splits_list,
            column_masks,
            n_jobs=n_jobs,
        )

        if 'report_batch' in callbacks:
            callbacks['report_batch'](models, repeat_index + 1, n_batches)

    scores = list(
        cross_validate_predict(
            X,
            y,
            predict,
            splits_list,
            column_masks,
            models,
        ))

    if isinstance(splits, Dict):
        return {
            'predictions': list_to_dict_by_keys(scores, splits.keys()),
            'column_masks': list_to_dict_by_keys(column_masks, splits.keys()),
        }
    else:
        return {
            'predictions': list_to_dict_index(scores),
            'column_masks': list_to_dict_index(column_masks),
        }


def get_removed_features_from_mask(
        column_masks: Dict[Any, Dict[Any, bool]]) -> Dict[Hashable, List[str]]:
    return pipe(
        column_masks,
        valmap(lambda masks: valfilter(identity, masks)),
        valmap(lambda k: list(k.keys())),
    )


def get_column_mask(
    splits: List[FoldInput],
    X: DataFrame,
    train_test_filter: Callable[[Series, Series], bool] = None,
) -> List[Dict[Any, bool]]:
    if train_test_filter:
        return list(get_column_mask_filter(
            X,
            splits,
            train_test_filter,
        ))
    else:
        return get_columns_mask_default(X, splits)


def get_columns_mask_default(
    X: DataFrame,
    splits: List[FoldInput],
) -> List[Dict[Any, bool]]:
    filtered_columns = [ \
        {
            column: False
            for column in X.columns
        }
        for _ in splits
    ]
    return filtered_columns


def get_column_mask_filter(
    X: DataFrame,
    splits: List[FoldInput],
    train_test_filter: Callable[[DataFrame, DataFrame], bool],
) -> Iterable[Dict[str, bool]]:
    for train, test in splits:
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        yield {
            column_name: train_test_filter(
                X_train[column_name],
                X_test[column_name],
            )
            for column_name in X
        }


def cross_validate_train(
    X: DataFrame,
    y: Target,
    models: List[Estimator],
    splits_list: List[FoldInput],
    filtered_columns: List[Dict[str, bool]],
    n_jobs: int = -1,
) -> List[Estimator]:
    if n_jobs == -1:
        n_jobs = cpu_count()

    fold_data = [(
        models[nr],
        cross_validate_preprocess(X, train_split, filtered_columns[nr]),
        index_data(train_split, y),
        filtered_columns[nr],
    ) for nr, (train_split, test_split) in enumerate(splits_list)]

    if n_jobs == 1:
        models = list(itertools.starmap(
            cross_validate_fit,
            fold_data,
        ))
    else:
        with Pool(min(len(splits_list), n_jobs)) as p:
            models = p.starmap(
                cross_validate_fit,
                fold_data,
            )
    return models


def cross_validate_preprocess(
    data: DataFrame,
    split: List[int],
    filtered_columns: Dict[str, bool],
):
    return pipe(
        data,
        partial(index_data, split),
        partial(cross_validate_apply_mask, filtered_columns),
    )


def cross_validate_predict(
    X: DataFrame,
    y: Target,
    predict: Callable,
    splits: List[FoldInput],
    filtered_columns: List[Dict[str, bool]],
    models: List[Estimator],
) -> Iterable[FoldPrediction]:
    for index, (train_split, test_split) in enumerate(splits):
        yield predict(
            X_train=cross_validate_preprocess(
                X,
                train_split,
                filtered_columns[index],
            ),
            y_train=index_data(train_split, y),
            X_test=cross_validate_preprocess(
                X,
                test_split,
                filtered_columns[index],
            ),
            y_true=index_data(test_split, y),
            model=models[index],
        )


def cross_validate_apply_mask(
    mask: Dict[str, bool],
    data: DataFrame,
) -> DataFrame:
    new_data = data.copy()
    if set(mask.keys()) != set(data.columns):
        raise Exception("Keys do not match")

    for column_name, remove in mask.items():
        if remove:
            new_data.drop(column_name, axis=1, inplace=True)

    return new_data


# TODO:
# def cross_validate_predict_binary(
#     X: DataFrame,
#     y: Target,
#     estimators: List[EstimatorProba],
#     splits: List[FoldInput] = None,
# ) -> List[FoldPrediction]:
#     scores = []
#     for index, (train_split, test_split) in enumerate(splits):
#         X_test = X.iloc[test_split]
#         y_score = estimators[index].predict_proba(X_test)
#
#         if not isinstance(y_score, Series):
#             y_score = DataFrame(y_score, index=X_test.index)
#             y_score = get_1_class_y_score(y_score)
#
#         scores.append(
#             FoldPrediction(y_true=y.iloc[test_split], y_score=y_score))
#
#     return scores


def predict_survival(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_true: Target,
    model: Estimator,
) -> float:
    return FoldPrediction(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_score=Series(
            model.predict(X_test),
            index=X_test.index,
        ),
        y_true=y_true,
        model=model,
    )


def predict_nn(
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_true: Target,
    model: Estimator,
) -> float:
    predicted_score = model.predict(X_test)
    return FoldPrediction(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_score=Series(
            predicted_score.reshape(len(predicted_score)),
            index=X_test.index,
        ),
        y_true=y_true,
        model=model,
    )


def predict_proba(
    X_train: DataFrame,
    y_train: Target,
    X_test: DataFrame,
    y_true: Target,
    model: EstimatorProba,
) -> float:
    return FoldPrediction(
        X_train=X_train,
        y_train=y_train,
        y_score=model.predict_proba(X_test),
        y_true=y_true,
        model=model,
    )


def cross_validate_fit(estimator, X, y, columns: Dict[str, bool]) -> Estimator:
    estimator.fit(
        X,
        y,
    )
    return estimator


def lco_cv(data: DataFrameGroupBy) -> Dict[Any, FoldInput]:
    flatten_data = data.apply(identity)
    all_indexes = range(0, len(flatten_data))
    groups = map_groups_iloc(data, flatten_data)
    return {
        key: (subtract_lists(all_indexes, subset), subset)
        for key, subset in groups
    }


def lm_cv(data: DataFrameGroupBy) -> Dict[Any, FoldInput]:
    lco_splits: Dict[Any, FoldInput] = lco_cv(data)
    return {
        key: pipe(
            fold_input,
            reversed,
            list,
        )
        for key, fold_input in lco_splits.items()
    }


def kfold_cv(data: DataFrame, **kfold_args) -> Dict[Any, FoldInput]:
    return list_to_dict_index(KFold(**kfold_args).split(data))


def train_test(
    data: DataFrame,
    train_filter: Callable,
    test_filter: Callable = None,
) -> Dict[Any, FoldInput]:
    index = Series(
        pipe(
            len(data),
            range,
            list,
        ),
        index=data.index,
    )

    train_data = data[train_filter(data)]

    if not test_filter:
        test_data = index.loc[~index.index.isin(train_data.index)]
    else:
        test_data = data[test_filter(data)]

    return {
        'train_test': (
            index.loc[train_data.index].tolist(),
            index.loc[test_data.index].tolist(),
        )
    }


def filter_missing_features(
    x_train: Series,
    x_test: Series,
    threshold: float = 1,
) -> bool:
    return \
        percent_missing(x_train) >= threshold \
        or percent_missing(x_test) >= threshold
