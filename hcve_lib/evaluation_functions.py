import logging
import traceback
from collections import defaultdict
from functools import partial
from itertools import chain
from statistics import mean
from typing import Dict, cast, Iterable, DefaultDict, Hashable, Optional, TypeVar, Type, Any
from typing import Union, List, Tuple, Callable

import numpy
import numpy as np
import pandas
import toolz
from numpy import NaN, std
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.metrics import roc_auc_score
from toolz import pluck, merge, itemmap, valmap, merge_with

from hcve_lib.custom_types import ClassificationMetrics, ValueWithStatistics, \
    ClassificationMetricsWithStatistics, GenericConfusionMatrix, ConfusionMetrics, ValueWithCI, \
    ConfusionMatrix, Target, Result, ExceptionValue, Splits, TrainTestIndex, Estimator, Method
from hcve_lib.metrics import get_standard_metrics
from hcve_lib.metrics_types import Metric
from hcve_lib.custom_types import Prediction
from hcve_lib.functional import pass_args, pipe, star_args, try_except
from hcve_lib.stats import confidence_interval
from hcve_lib.tracking import log_metrics
from hcve_lib.utils import transpose_dict, map_groups_loc, split_data, get_y_split, loc
from sksurv.metrics import concordance_index_censored

from hcve_lib.wrapped_sklearn import DFStandardScaler

HashableT = TypeVar('HashableT', bound=Hashable)


def log_repeat_metrics(
    result: Result,
    y: Target,
    metrics: List[Metric] = None,
) -> Dict[HashableT, ValueWithCI]:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metrics = compute_metrics_result(
        result,
        y,
        metrics,
    )
    log_metrics(metrics)
    return metrics


def compute_metrics(
    results: List[Result],
    y: Target,
    metrics: List[Metric] = None,
    skip_metrics: List[str] = None,
) -> Dict[HashableT, ValueWithCI]:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metrics_runs = []

    for result in results:
        metrics_runs.append(compute_metrics_result_per_prediction(result, y, metrics, skip_metrics=skip_metrics))

    metrics_renamed = [
        {f'{split}_{num}': pred
         for split, pred in result.items()}
        for num, result in enumerate(metrics_runs)
    ]
    metrics_merged = toolz.merge(metrics_renamed)

    return pipe(
        metrics_merged,
        transpose_dict,
        partial(
            itemmap,
            star_args(
                lambda metric_name, metrics_values: (
                    metric_name,
                    compute_metric_statistics(metrics_values.values()),
                ),
            ),
        ),
    )


def compute_metrics_result(
    result: Result,
    y: Target,
    metrics: List[Metric],
    skip_metrics: List[str] = None,
) -> Dict[HashableT, ValueWithCI]:
    metrics_per_split: Dict = compute_metrics_result_per_prediction(
        result,
        y,
        metrics,
        skip_metrics=skip_metrics,
    )

    return pipe(
        metrics_per_split,
        transpose_dict,
        partial(
            itemmap,
            star_args(
                lambda metric_name, metrics_values: (
                    metric_name,
                    compute_metric_statistics(metrics_values.values()),
                ),
            ),
        ),
    )


def compute_metrics_result_per_prediction(
    result: Result,
    y: Target,
    metrics: List[Metric],
    skip_metrics: List[str] = None,
) -> Dict[HashableT, Dict[str, Union[float, ExceptionValue]]]:
    return {
        key: compute_metrics_prediction(
            metrics,
            y,
            prediction,
            skip_metrics,
        )
        for key, prediction in result.items()
    }


def compute_metric_statistics(values: Iterable[Optional[float]]) -> Union[ValueWithStatistics, ExceptionValue]:
    values_ = [value for value in values if isinstance(value, (float, int))]
    try:
        return ValueWithStatistics(
            mean=mean(values_) if len(values_) > 0 else np.nan,
            ci=confidence_interval(values_)[1] if len(values_) > 1 else (np.nan, np.nan),
            std=std(values_) if len(values_) > 1 else np.nan,
        )
    except TypeError as e:
        print(traceback.format_exc())
        return ExceptionValue(value=values_, exception=e)


def compute_metric_result(
    metric: Metric,
    y: Target,
    result: Result,
    skip_metrics: List[str] = None,
) -> Dict[HashableT, Union[float, ExceptionValue]]:
    return {
        key: list(compute_metrics_prediction(
            [metric],
            y,
            prediction,
            skip_metrics,
        ).values())[0]
        for key, prediction in result.items()
    }


def compute_metrics_prediction(
    metrics: List[Metric],
    y: Target,
    prediction: Prediction,
    skip_metrics: Optional[List[str]] = None,
) -> Dict[str, Union[float, ExceptionValue]]:
    metric_names: Iterable[str] = iter(())
    metric_values: Iterable[Union[float, ExceptionValue]] = iter(())
    for metric in metrics:
        new_names, new_values = compute_metric_prediction_items(
            metric,
            y,
            prediction,
            skip_metrics,
        )
        new_names = list(new_names)
        new_values = list(new_values)
        metric_names = chain(metric_names, new_names)
        metric_values = chain(metric_values, new_values)

    return dict(zip(metric_names, metric_values))


def compute_metric_prediction_items(
    metric: Metric,
    y: Target,
    prediction: Prediction,
    skip_metrics: Optional[List[str]] = None,
) -> Tuple[List, List]:

    new_names = metric.get_names(prediction, y)
    if not skip_metrics or any(n not in skip_metrics for n in new_names):
        return new_names, metric.get_values(prediction, y)


def compute_metric_prediction(
    metric: Metric,
    y: Target,
    prediction: Prediction,
    skip_metrics: Optional[List[str]] = None,
) -> Dict:
    return dict(zip(*compute_metric_prediction_items(
        metric,
        y,
        prediction,
        skip_metrics,
    )))


def get_2_level_groups(
    folds: Result,
    group_by: DataFrameGroupBy,
    data: DataFrame,
) -> Dict[Hashable, Result]:
    groups = list(map_groups_loc(group_by))
    result: DefaultDict = defaultdict(dict)

    for fold_name, fold in folds.items():
        for group_name, group_index in groups:
            group_index_test = group_index[group_index.isin(fold['y_score'].index)]
            group_iloc_train = fold['split'][0]
            group_iloc_test = group_index_test.map(lambda key: data.index.get_loc(key))

            if len(group_iloc_test) == 0:
                result[fold_name][group_name] = None
                continue

            result[fold_name][group_name] = merge(
                fold,
                {
                    'split': (group_iloc_train, group_iloc_test),
                    'y_score': fold['y_score'].loc[group_index_test],
                },
            )

    return dict(result)


def compute_metric_groups(
    metric: Callable[[Prediction], float],
    groups: Dict[Hashable, Result],
) -> Dict:
    result: DefaultDict = defaultdict(dict)
    for train_name, test_folds in groups.items():
        for test_name, test_fold in test_folds.items():
            if test_fold:
                result[train_name][test_name] = metric(test_fold)
            else:
                result[train_name][test_name] = None
    return dict(result)


def compute_ci_for_metrics_collection(metrics: List[ClassificationMetrics]) -> ClassificationMetricsWithStatistics:
    attributes = list(metrics[0].keys())
    metrics_with_ci_dict = {
        attribute: pass_args(
            confidence_interval(list(pluck(attribute, metrics))),
            lambda m, ci, std_val: ValueWithStatistics(mean=m, std=std_val, ci=ci),
        )
        for attribute in attributes
    }
    return cast(
        ClassificationMetricsWithStatistics,
        metrics_with_ci_dict,
    )


def get_1_class_y_score(y_score: Union[DataFrame, Series]) -> Series:
    if isinstance(y_score, Series):
        return y_score
    return y_score.iloc[:, 1]


ConfusionMatrixWithStatistics = GenericConfusionMatrix[ValueWithStatistics]


def get_metrics_from_confusion_matrix(confusion_matrix) -> ConfusionMetrics:
    try:
        npv = confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fn)
    except ZeroDivisionError:
        npv = 0

    return ConfusionMetrics(
        precision=(confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fp)) if
        (confusion_matrix.tp + confusion_matrix.fp) > 0 else NaN,
        recall=(confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fn)) if
        (confusion_matrix.tp + confusion_matrix.fn) > 0 else NaN,
        fpr=confusion_matrix.fp / (confusion_matrix.fp + confusion_matrix.tn),
        tnr=confusion_matrix.tn / (confusion_matrix.fp + confusion_matrix.tn),
        fnr=confusion_matrix.fn / (confusion_matrix.fn + confusion_matrix.tp),
        npv=npv,
    )


def get_confusion_from_threshold(
    y: Series,
    scores: Series,
    threshold: float = 0.5,
) -> ConfusionMatrix:
    fn = 0
    tn = 0
    tp = 0
    fp = 0

    for index, score in scores.items():
        if score < threshold:
            if y.loc[index] == 1:
                fn += 1
            elif y.loc[index] == 0:
                tn += 1
        elif score >= threshold:
            if y.loc[index] == 1:
                tp += 1
            elif y.loc[index] == 0:
                fp += 1

    matrix = ConfusionMatrix(
        fn=fn,
        tn=tn,
        tp=tp,
        fp=fp,
    )

    return matrix


def c_index(
    prediction: Prediction,
    y: Target,
    is_train: bool = False,
) -> Union[ExceptionValue, float]:
    if len(prediction['y_score']) == 0:
        return np.nan

    y_train, y_test = get_y_split(y, prediction)
    y_to_evaluate = y_train if is_train else y_test

    try:
        index: Tuple = concordance_index_censored(
            y_to_evaluate['data']['label'].loc[prediction['y_score'].index].to_numpy().astype(numpy.bool_),
            y_to_evaluate['data']['tte'].loc[prediction['y_score'].index],
            prediction['y_score'].to_numpy(),
        )
        return index[0]
    except ValueError as e:
        return ExceptionValue(exception=e)


def roc_auc(fold: Prediction, X: DataFrame, y: Target) -> float:
    if len(fold['y_score']) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    return roc_auc_score(y_test['data'], fold['y_score'], multi_class='ovo')


def c_index_inverse_score(
    fold: Prediction,
    X: DataFrame,
    y: Target,
) -> float:
    if len(fold['y_score']) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    index: Tuple = concordance_index_censored(
        y_test['data']['label'].astype(bool),
        y_test['data']['tte'],
        1 - fold['y_score'],
    )
    return index[0]


def get_splits_by_class(y: Target, labels: List = None) -> Splits:
    classes = get_target_label(y).unique()
    return {
        f'outcome_{cls}': y['data'][get_target_label(y) == cls].index.to_list()
        for cls in classes
        if (labels is None or cls in labels)
    }


def get_splits_by_age(age: Series, years: int = 10) -> Splits:
    group_by = age.groupby((np.floor(age / years) * years))
    return {f'age__{round(age)}_{round(age + years)}': group.index.to_list() for age, group in group_by}


def get_target_label(y: Target) -> Series:
    if isinstance(y['data'], Series):
        return y['data']
    elif isinstance(y['data'], DataFrame):
        return y['data']['label']
    else:
        raise TypeError("Can't extract series from target")


def predict_proba(
    X: DataFrame,
    y: Target,
    split: TrainTestIndex,
    model: Estimator,
    method: Type[Method],
    random_state: int,
) -> Prediction:
    y_score = DataFrame(
        model.predict_proba(loc(split[1], X)),
        index=split[1],
    )
    return Prediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=y_score,
        model=model,
        method=method,
        random_state=random_state,
    )


class Pipeline:

    def predict_survival_time(self):
        pass


def predict_survival(
    X: DataFrame,
    y: Target,
    split: TrainTestIndex,
    model: Estimator,
    random_state: int,
    method: Type[Method],
    time: Union[int, Iterable] = 5 * 365,
) -> Prediction:
    X_test = loc(
        split[1],
        X,
        ignore_not_present=True,
    )

    return Prediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=Series(
            model.predict(X_test),
            index=X_test.index,
        ),
        y_proba={('tte' if isinstance(time, Iterable) else time): predict_survival_proba(time, X_test, model)},
        model=model,
        random_state=random_state,
        method=method,
    )


def predict_survival_proba(
    time: Union[int, Iterable],
    X: DataFrame,
    model: Estimator,
) -> Series:
    try:
        survival_functions = list(model.predict_survival_function(X))
    except AttributeError:
        return ExceptionValue(exception=TypeError('model missing \'predict_survival_function\' method.'))

    # TODO: HACK >>
    survival_fns_valid = list(reject_exception_values(survival_functions))
    difference_valid = len(survival_functions) - len(survival_fns_valid)
    if difference_valid > 0:
        logging.warning(f'{difference_valid} individuals dropped due to out of range prediction')
    # <<

    if survival_fns_valid is None or len(survival_fns_valid) == 0:
        return ExceptionValue(exception=ValueError('"predict_survival_function" returning None'), )
    else:
        return Series(
            [
                try_except(lambda: fn(time), {Exception: lambda e: ExceptionValue(exception=e)}) for fn, time in
                zip(survival_fns_valid, (time if isinstance(time, Iterable) else ([time] * len(survival_fns_valid))))
            ],
            index=X.index,
        )


def predict_survival_dsm(
    X: DataFrame,
    y: Target,
    split: TrainTestIndex,
    model: Estimator,
    random_state: int,
    method: Type[Method],
    time: int = 3 * 365,
) -> Prediction:
    X_test = loc(
        split[1],
        X,
        ignore_not_present=True,
    )
    return Prediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=Series(
            model.predict(X_test).flatten(),
            index=X_test.index,
        ),
        y_proba={
            time: try_except(
                lambda: Series(
                    model.predict_survival(X_test, [time] * len(X_test)),
                    index=X_test.index,
                ),
                {Exception: lambda e: ExceptionValue(exception=e)},
            ),
        },
        model=model,
        method=method,
    )


def predict_predict(
    X: DataFrame,
    y: Target,
    split: TrainTestIndex,
    model: Estimator,
) -> Prediction:
    return Prediction(
        split=split,
        X_columns=X.columns.tolist(),
        y_column=y['name'],
        y_score=Series(
            model.predict(X.loc[split[1]]),
            index=X.loc[split[1]].index,
        ),
        model=model,
    )


def reject_exception_values(sequence: Iterable) -> Iterable:
    return iter(filter(is_not_exception_value, sequence))


def is_not_exception_value(that: Any) -> bool:
    return not isinstance(that, ExceptionValue)


def merge_standardize_prediction(result: Result) -> Prediction:
    standardized_result = valmap(
        lambda prediction: {
            **prediction,
            'y_score': DFStandardScaler().fit_transform(
                DataFrame({'y_score': prediction['y_score']}), DataFrame({'y_score': prediction['y_score']})
            )['y_score'],
        },
        result,
    )
    return merge_predictions(standardized_result)


def merge_predictions(result: Result) -> Prediction:
    y_pred = pandas.concat([prediction['y_pred'] for cohort, prediction in result.items()])

    return Prediction(
        y_pred=y_pred,
        split=(list(y_pred.index), list(y_pred.index)),
        # y_proba=pipe(
        #     {
        #         # TODO: throwing exceptions
        #         # time: prediction if isinstance(prediction, ExceptionValue) else prediction.get('y_proba', {})
        #         # for time,
        #         # prediction in result.items()
        #     },
        #     transpose_dict,
        #     partial(
        #         valmap,
        #         lambda y_probas: try_except(
        #             lambda: pandas.concat(y_probas.values()),
        #             {Exception: lambda e: e},
        #         ),
        #     ),
        # ),
    )


def average_group_scores(group: Dict[Hashable, Result]) -> Result:
    averaged_predictions = {}
    for test_cohort, group in transpose_dict(group).items():
        y_scores = [prediction['y_score'] for prediction in group.values()]
        y_proba = transpose_dict(group).get('y_proba')

        averaged_predictions[test_cohort] = {
            **group[0],
            'y_score': pandas.concat(y_scores, axis=1).mean(axis=1),
        }

        if y_proba:
            try:
                y_proba_joined = merge_with(
                    partial(pandas.concat, axis=1),
                    *y_proba.values(),
                )
            except TypeError as e:
                averaged_predictions[test_cohort]['y_proba'] = ExceptionValue(exception=e, value=y_proba)
            else:
                y_proba_mean = valmap(
                    partial(DataFrame.mean, axis=1),
                    y_proba_joined,
                )
                averaged_predictions[test_cohort]['y_proba'] = y_proba_mean

    return averaged_predictions


def get_inverse_weight(
    series: Series,
    proportions: Dict[Any, float] = None,
) -> Series:
    counts = series.value_counts()
    if proportions is None:
        proportions = {k: 1 / len(counts) for k, _ in counts.items()}
    proportions_s = Series(proportions)

    proportions = proportions_s / (counts / counts.sum())
    return proportions


def map_inverse_weight(
    series: Series,
    proportions: Dict[Any, float] = None,
) -> Series:
    weights = get_inverse_weight(series, proportions=proportions)
    return series.map(weights).astype('float')
