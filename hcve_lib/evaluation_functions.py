from collections import defaultdict

from pandas.core.groupby import DataFrameGroupBy
from statistics import mean

from functools import partial
from toolz.curried import valmap, itemmap
from typing import Any, Dict, cast, Type, Iterable, DefaultDict, Hashable, Optional
from typing import Union, List, Tuple, Callable

import numpy as np
from numpy import NaN
from pandas import DataFrame, Series
from sklearn.metrics import roc_curve, precision_score, balanced_accuracy_score, f1_score, average_precision_score, \
    accuracy_score, roc_auc_score, brier_score_loss
from sksurv.metrics import concordance_index_censored
from toolz import pluck, merge

from hcve_lib.custom_types import ClassificationMetrics, ValueWithStatistics, \
    ClassificationMetricsWithStatistics, GenericConfusionMatrix, ConfusionMetrics, ValueWithCI, ConfusionMatrix, Target
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.functional import pass_args, pipe, find_index, statements, star_args, try_except, rejectNone
from hcve_lib.statistics import confidence_interval
from hcve_lib.utils import transpose_dict, map_groups_iloc, index_data, map_groups_loc


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


def compute_classification_metrics_from_result(
    cv_pairs: List[FoldPrediction],
    ignore_warning: bool = False,
) -> Dict[str, ClassificationMetricsWithStatistics]:
    return pipe(
        cv_pairs,
        partial(
            map,
            lambda cv_pair: compute_classification_metrics_fold(
                cv_pair,
                ignore_warning=ignore_warning,
            ),
        ),
        list,
        compute_ci_for_metrics_collection,
    )


def compute_classification_metrics_fold(
    cv_pair: FoldPrediction,
    ignore_warning=False,
) -> ClassificationMetrics:
    return compute_classification_metrics(
        cv_pair['y_score'],
        cv_pair['y_true'],
        threshold=0.5,
        ignore_warning=ignore_warning,
    )


DEFAULT_THRESHOLD = 0.5


def compute_classification_metrics(
    y_score,
    y_true,
    threshold: float = DEFAULT_THRESHOLD,
    ignore_warning: bool = False,
) -> ClassificationMetrics:
    y_score_normalized = y_score.copy()
    y_score_normalized[y_score_normalized < 0] = 0
    y_score_normalized[y_score_normalized > 1] = 1

    y_predict = y_score_normalized >= threshold
    y_true_masked = y_true.loc[y_predict.index]
    roc = roc_curve(y_true_masked, y_score_normalized)
    fpr, tpr = get_roc_point_by_threshold(threshold, *roc)
    npv = get_metrics_from_confusion_matrix(
        get_confusion_from_threshold(
            y_true_masked,
            y_score_normalized,
            threshold,
        )).npv

    precision = precision_score(
        y_true_masked, y_predict,
        **({
            'zero_division': 0
        } if ignore_warning else {}))

    return ClassificationMetrics(
        recall=tpr,
        precision=precision,
        balanced_accuracy=balanced_accuracy_score(y_true_masked, y_predict),
        f1=f1_score(y_true_masked, y_predict),
        tnr=1 - fpr,
        fpr=fpr,
        fnr=1 - tpr,
        average_precision=average_precision_score(
            y_true_masked,
            y_score_normalized,
        ),
        accuracy=accuracy_score(y_true_masked, y_predict),
        roc_auc=roc_auc_score(y_true_masked, y_score_normalized),
        npv=npv,
        brier_score=brier_score_loss(y_true_masked, y_score_normalized))


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
        ration = 1
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


def c_index(fold: FoldPrediction) -> float:
    if len(fold['y_score']) == 0:
        return np.nan

    index: Tuple = concordance_index_censored(
        fold['y_true']['label'].astype(bool),
        fold['y_true']['tte'],
        fold['y_score'],
    )
    return index[0]


def compute_metrics_ci(
    folds: Dict[Any, FoldPrediction],
    metrics: List[Callable[[FoldPrediction], float]],
    y_true: Target,
) -> Dict[str, ValueWithCI]:
    metrics_values_per_metrics = compute_metrics_folds(folds, metrics, y_true)
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
        mean=mean(list(rejectNone(values))),
        ci=confidence_interval(list(rejectNone(values)))[1]
        if len(values) > 1 else (0, 0),
    )


def compute_metrics_folds(
    folds: Dict[Any, FoldPrediction],
    metrics: List[Callable[[FoldPrediction], float]],
    y_true: Target,
) -> Dict[Any, Dict[Any, float]]:
    return {
        study_num: compute_metric_fold(metrics, prediction, y_true)
        for study_num, prediction in folds.items()
    }


def compute_metric_fold(
    metrics: List[Callable[[FoldPrediction], float]],
    fold: FoldPrediction,
    y: Target,
) -> Dict[Any, float]:
    return {metric.__name__: metric(fold) for metric in metrics}


def get_2_level_groups(
    folds: Dict[Hashable, FoldPrediction],
    group_by: DataFrameGroupBy,
) -> Dict[Hashable, Dict[Hashable, FoldPrediction]]:
    groups = list(map_groups_loc(group_by))
    result: DefaultDict = defaultdict(dict)

    for fold_name, fold in folds.items():
        for group_name, group_index in groups:
            group_index_subset = group_index[group_index.isin(
                fold['y_score'].index)]

            group_iloc_subset = group_index_subset.map(
                lambda key: fold['y_score'].index.get_loc(key))

            result[fold_name][group_name] = FoldPrediction(**merge(
                fold, {
                    'X_test': index_data(group_iloc_subset, fold['X_test']),
                    'y_score': fold['y_score'].loc[group_index_subset],
                    'y_true': index_data(group_iloc_subset, fold['y_true']),
                }))

    return dict(result)


def compute_metric_groups(
    metric: Callable[[FoldPrediction], float],
    groups: Dict[Hashable, Dict[Hashable, FoldPrediction]],
) -> Dict:
    result: DefaultDict = defaultdict(dict)
    for train_name, test_folds in groups.items():
        for test_name, test_fold in test_folds.items():
            result[train_name][test_name] = metric(test_fold)
    return dict(result)
