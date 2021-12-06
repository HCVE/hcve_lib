from collections import defaultdict
from statistics import mean
from typing import Any, Dict, cast, Iterable, DefaultDict, Hashable, Optional
from typing import Union, List, Tuple, Callable

import numpy as np
from numpy import NaN
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.metrics import roc_auc_score

from sksurv.metrics import concordance_index_censored
from toolz import pluck, merge
from toolz.curried import itemmap

from hcve_lib.custom_types import ClassificationMetrics, ValueWithStatistics, \
    ClassificationMetricsWithStatistics, GenericConfusionMatrix, ConfusionMetrics, ValueWithCI, ConfusionMatrix, Target
from hcve_lib.custom_types import SplitPrediction
from hcve_lib.functional import pass_args, pipe, find_index, star_args, reject_none
from hcve_lib.stats import confidence_interval
from hcve_lib.utils import transpose_dict, map_groups_loc, split_data


def get_1_class_y_score(y_score: Union[DataFrame, Series]) -> Series:
    if isinstance(y_score, Series):
        return y_score
    return y_score.iloc[:, 1]


def compute_ci_for_metrics_collection(
    metrics: List[ClassificationMetrics]
) -> ClassificationMetricsWithStatistics:
    attributes = list(metrics[0].keys())
    metrics_with_ci_dict = {
        attribute: pass_args(
            confidence_interval(list(pluck(attribute, metrics))),
            lambda m, ci, std: ValueWithStatistics(mean=m, std=std, ci=ci),
        )
        for attribute in attributes
    }
    return cast(
        ClassificationMetricsWithStatistics,
        metrics_with_ci_dict,
    )


def get_roc_point_by_threshold(
    threshold: float,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:

    first_index = find_index(
        lambda _index: _index >= threshold,
        list(thresholds),
        reverse=True,
    )

    if first_index == len(thresholds) - 1:
        second_index = first_index
    else:
        second_index = first_index + 1

    first_threshold = thresholds[first_index]
    second_threshold = thresholds[second_index]

    if second_threshold != first_threshold:
        ratio = (threshold - second_threshold) \
                / (first_threshold - second_threshold)
    else:
        ratio = 1
    return (
        ((fpr[second_index] * (1 - ratio)) + (fpr[first_index] * ratio)),
        (tpr[second_index] * (1 - ratio) + tpr[first_index] * ratio),
    )


ConfusionMatrixWithStatistics = GenericConfusionMatrix[ValueWithStatistics]


def get_metrics_from_confusion_matrix(confusion_matrix) -> ConfusionMetrics:
    try:
        npv = confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fn)
    except ZeroDivisionError:
        npv = 0

    return ConfusionMetrics(
        precision=(confusion_matrix.tp /
                   (confusion_matrix.tp + confusion_matrix.fp)) if
        (confusion_matrix.tp + confusion_matrix.fp) > 0 else NaN,
        recall=(confusion_matrix.tp /
                (confusion_matrix.tp + confusion_matrix.fn)) if
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


def c_index(fold: SplitPrediction, X: DataFrame, y: Target) -> float:
    if len(fold['y_score']) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    index: Tuple = concordance_index_censored(
        y_test['data']['label'].astype(bool),
        y_test['data']['tte'],
        fold['y_score'],
    )
    return index[0]


def roc_auc(fold: SplitPrediction, X: DataFrame, y: Target) -> float:
    if len(fold['y_score']) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    return roc_auc_score(y_test['data'], fold['y_score'], multi_class='ovo')


def c_index_inverse_score(
    fold: SplitPrediction,
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


def compute_metrics_ci(
    folds: Dict[Any, SplitPrediction],
    metrics: List[Callable[[SplitPrediction], float]],
) -> Dict[str, ValueWithCI]:
    metrics_values_per_metrics = compute_metrics_folds(folds, metrics)
    return pipe(
        metrics_values_per_metrics,
        transpose_dict,
        itemmap(
            star_args(
                lambda metric_name, metrics_values:
                (metric_name, compute_metric_ci(metrics_values.values())),
            )),
    )


def compute_metric_ci(
        metrics_values: Iterable[Optional[float]]) -> ValueWithCI:
    values = list(metrics_values)
    return ValueWithCI(
        mean=mean(list(reject_none(values))),
        ci=confidence_interval(list(reject_none(values)))[1]
        if len(values) > 1 else (0, 0),
    )


def compute_metrics_folds(
    folds: Dict[Any, SplitPrediction],
    metrics: List[Callable[[SplitPrediction], float]],
) -> Dict[Any, Dict[Any, float]]:
    return {
        study_num: compute_metrics_fold(metrics, prediction)
        for study_num, prediction in folds.items()
    }


def compute_metrics_fold(
    metrics: List[Callable[[SplitPrediction], float]],
    fold: SplitPrediction,
) -> Dict[Any, float]:

    return {metric.__name__: metric(fold) for metric in metrics}


def get_2_level_groups(
    folds: Dict[Hashable, SplitPrediction],
    group_by: DataFrameGroupBy,
    data: DataFrame,
) -> Dict[Hashable, Dict[Hashable, SplitPrediction]]:
    groups = list(map_groups_loc(group_by))
    result: DefaultDict = defaultdict(dict)

    for fold_name, fold in folds.items():
        for group_name, group_index in groups:
            group_index_test = group_index[group_index.isin(
                fold['y_score'].index)]
            group_iloc_train = fold['split'][0]
            group_iloc_test = group_index_test.map(
                lambda key: data.index.get_loc(key))

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
    metric: Callable[[SplitPrediction], float],
    groups: Dict[Hashable, Dict[Hashable, SplitPrediction]],
) -> Dict:
    result: DefaultDict = defaultdict(dict)
    for train_name, test_folds in groups.items():
        for test_name, test_fold in test_folds.items():
            if test_fold:
                result[train_name][test_name] = metric(test_fold)
            else:
                result[train_name][test_name] = None
    return dict(result)
