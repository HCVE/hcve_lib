import logging
import traceback
from collections import defaultdict
from functools import partial, reduce
from itertools import chain
from statistics import mean
from typing import (
    Dict,
    cast,
    Iterable,
    DefaultDict,
    Hashable,
    Optional,
    TypeVar,
    Type,
    Any,
)
from typing import Union, List, Tuple, Callable

import numpy
import numpy as np
import pandas
import pandas as pd
import toolz
from numpy import std
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from plotly import graph_objects as go
from scipy.stats import stats
from sklearn.metrics import roc_auc_score
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from toolz import pluck, merge, itemmap, valmap, merge_with

from hcve_lib.custom_types import (
    ClassificationMetrics,
    ValueWithStatistics,
    ClassificationMetricsWithStatistics,
    GenericConfusionMatrix,
    ConfusionMetrics,
    ValueWithCI,
    ConfusionMatrix,
    Target,
    Result,
    ExceptionValue,
    Splits,
    TrainTestIndex,
    Estimator,
    Method,
    Results,
    T1,
    Metrics,
)
from hcve_lib.custom_types import Prediction
from hcve_lib.data import to_survival_y_records
from hcve_lib.functional import pass_args, pipe, star_args, try_except
from hcve_lib.metrics import get_standard_metrics
from hcve_lib.metrics_types import Metric
from hcve_lib.stats import confidence_interval
from hcve_lib.tracking import log_metrics
from hcve_lib.utils import (
    transpose_dict,
    map_groups_loc,
    split_data,
    get_y_split,
    loc,
    get_first_entry,
    get_X_y_split,
)
from hcve_lib.wrapped_sklearn import DFStandardScaler

HashableT = TypeVar("HashableT", bound=Hashable)


def compute_metrics(
    results: List[Result],
    y: Target,
    metrics: Optional[List[Metric]] = None,
    skip_metrics: Optional[List[str]] = None,
) -> Metrics:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metrics_merged = compute_metrics_per_prediction(results, y, metrics, skip_metrics)
    return compute_metrics_statistics(metrics_merged)


def compute_metrics_merged_splits(
    results: List[Result],
    y: Target,
    metrics: Optional[List[Metric]] = None,
    skip_metrics: Optional[List[str]] = None,
) -> Dict[HashableT, ValueWithStatistics]:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metric_values = {}

    for index_result, result in enumerate(results):
        prediction = merge_standardize_prediction(result)
        metric_values[index_result] = compute_metrics_prediction(
            prediction, y, metrics, skip_metrics
        )

    return compute_metrics_statistics(metric_values)


def compute_metric(
    results: Results,
    y: Target,
    metric: Metric = None,
) -> Dict[HashableT, ValueWithStatistics]:
    if metric is None:
        metric = get_standard_metrics(y)[0]

    metrics_merged = compute_metrics_per_prediction(results, y, [metric])
    return get_first_entry(compute_metrics_statistics(metrics_merged))


def compute_metrics_per_prediction(
    results: List[Result],
    y: Target,
    metrics: List[Metric] = None,
    skip_metrics: List[str] = None,
) -> Dict[Any, Dict[Any, float]]:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metrics_runs = []

    for result in results:
        metrics_runs.append(
            compute_metrics_result_per_prediction(
                result, y, metrics, skip_metrics=skip_metrics
            )
        )

    metrics_renamed = [
        {(split, num): pred for split, pred in result.items()}
        for num, result in enumerate(metrics_runs)
    ]

    metrics_merged = toolz.merge(metrics_renamed)

    return metrics_merged


def compute_metrics_per_prediction_merge_repeats(
    results: List[Result],
    y: Target,
    metrics: List[Metric] = None,
    skip_metrics: List[str] = None,
) -> Dict[Any, Dict[Any, float]]:
    if metrics is None:
        metrics = get_standard_metrics(y)

    metrics_runs = []

    for result in results:
        metrics_runs.append(
            compute_metrics_result_per_prediction(
                result, y, metrics, skip_metrics=skip_metrics
            )
        )

    metrics_renamed = [
        {f"{split}_{num}": pred for split, pred in result.items()}
        for num, result in enumerate(metrics_runs)
    ]

    metrics_merged = toolz.merge(metrics_renamed)

    return metrics_merged


def compute_metrics_statistics(
    values: Dict[Any, Dict[Any, float]],
) -> Dict[Any, ValueWithStatistics]:
    return pipe(
        values,
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
    metrics: List[Metric] = None,
    skip_metrics: List[str] = None,
) -> Metrics:
    if metrics is None:
        metrics = get_standard_metrics(y)

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
            prediction,
            y,
            metrics,
            skip_metrics,
        )
        for key, prediction in result.items()
    }


def compute_metric_statistics(
    values: Iterable[Optional[float]],
) -> Union[ValueWithStatistics, ExceptionValue]:
    values_ = [value for value in values if isinstance(value, (float, int))]
    try:
        return ValueWithStatistics(
            mean=mean(values_) if len(values_) > 0 else np.nan,
            ci=(
                confidence_interval(values_)[1]
                if len(values_) > 1
                else (np.nan, np.nan)
            ),
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
        key: list(
            compute_metrics_prediction(prediction, y, [metric], skip_metrics).values()
        )[0]
        for key, prediction in result.items()
    }


def compute_metrics_y(
    model: Estimator,
    y: Target,
    metrics: List[Metric] = None,
):
    if metrics is None:
        metrics = get_standard_metrics(y)

    for metric in metrics:
        ...


def compute_metrics_prediction(
    prediction: Prediction,
    y: Target,
    metrics: List[Metric] = None,
    skip_metrics: Optional[List[str]] = None,
) -> Dict[str, Union[float, ExceptionValue]]:
    if metrics is None:
        metrics = get_standard_metrics(y)

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
        return new_names, (
            metric.get_values(prediction, y)
            if prediction["y_pred"] is not None
            else None
        )


def compute_metric_prediction(
    metric: Metric,
    y: Target,
    prediction: Prediction,
    skip_metrics: Optional[List[str]] = None,
) -> Dict:
    return dict(
        zip(
            *compute_metric_prediction_items(
                metric,
                y,
                prediction,
                skip_metrics,
            )
        )
    )


def get_2_level_groups(
    folds: Result,
    group_by: DataFrameGroupBy,
    data: DataFrame,
) -> Dict[Hashable, Result]:
    groups = list(map_groups_loc(group_by))
    result: DefaultDict = defaultdict(dict)

    for fold_name, fold in folds.items():
        for group_name, group_index in groups:
            group_index_test = group_index[group_index.isin(fold["y_score"].index)]
            group_iloc_train = fold["split"][0]
            group_iloc_test = group_index_test.map(lambda key: data.index.get_loc(key))

            if len(group_iloc_test) == 0:
                result[fold_name][group_name] = None
                continue

            result[fold_name][group_name] = merge(
                fold,
                {
                    "split": (group_iloc_train, group_iloc_test),
                    "y_score": fold["y_score"].loc[group_index_test],
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


def compute_ci_for_metrics_collection(
    metrics: List[ClassificationMetrics],
) -> ClassificationMetricsWithStatistics:
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


ConfusionMatrixWithStatistics = GenericConfusionMatrix[ValueWithStatistics]


def get_metrics_from_confusion_matrix(confusion_matrix) -> ConfusionMetrics:
    try:
        npv = confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fn)
    except ZeroDivisionError:
        npv = 0

    return ConfusionMetrics(
        precision=(
            (confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fp))
            if (confusion_matrix.tp + confusion_matrix.fp) > 0
            else float("nan")
        ),
        recall=(
            (confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fn))
            if (confusion_matrix.tp + confusion_matrix.fn) > 0
            else float("nan")
        ),
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
    if len(prediction["y_score"]) == 0:
        return np.nan

    y_train, y_test = get_y_split(y, prediction)
    y_to_evaluate = y_train if is_train else y_test

    try:
        from sksurv.metrics import concordance_index_censored

        index: Tuple = concordance_index_censored(
            y_to_evaluate["data"]["label"]
            .loc[prediction["y_score"].index]
            .to_numpy()
            .astype(numpy.bool_),
            y_to_evaluate["data"]["tte"].loc[prediction["y_score"].index],
            prediction["y_score"].to_numpy(),
        )
        return index[0]
    except ValueError as e:
        return ExceptionValue(exception=e)


def roc_auc(fold: Prediction, X: DataFrame, y: Target) -> float:
    if len(fold["y_score"]) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    return roc_auc_score(y_test["data"], fold["y_score"], multi_class="ovo")


def c_index_inverse_score(
    fold: Prediction,
    X: DataFrame,
    y: Target,
) -> float:
    if len(fold["y_score"]) == 0:
        return np.nan
    _, _, _, y_test = split_data(X, y, fold)
    from sksurv.metrics import concordance_index_censored

    index: Tuple = concordance_index_censored(
        y_test["data"]["label"].astype(bool),
        y_test["data"]["tte"],
        1 - fold["y_score"],
    )
    return index[0]


def get_splits_by_class(y: Target, labels: List = None) -> Splits:
    classes = get_target_label(y).unique()
    return {
        f"outcome_{cls}": y["data"][get_target_label(y) == cls].index.to_list()
        for cls in classes
        if (labels is None or cls in labels)
    }


def get_splits_by_age(age: Series, years: int = 10) -> Splits:
    group_by = age.groupby((np.floor(age / years) * years))
    return {
        f"age__{round(age)}_{round(age + years)}": group.index.to_list()
        for age, group in group_by
    }


def get_target_label(y: Target) -> Series:
    if isinstance(y["data"], Series):
        return y["data"]
    elif isinstance(y["data"], DataFrame):
        return y["data"]["label"]
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
        y_column=y["name"],
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
        y_column=y["name"],
        y_score=Series(
            model.predict(X_test),
            index=X_test.index,
        ),
        y_proba={
            ("tte" if isinstance(time, Iterable) else time): predict_survival_proba(
                time, X_test, model
            )
        },
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
        return ExceptionValue(
            exception=TypeError("model missing 'predict_survival_function' method.")
        )

    # TODO: HACK >>
    survival_fns_valid = list(reject_exception_values(survival_functions))
    difference_valid = len(survival_functions) - len(survival_fns_valid)
    if difference_valid > 0:
        logging.warning(
            f"{difference_valid} individuals dropped due to out of range prediction"
        )
    # <<

    if survival_fns_valid is None or len(survival_fns_valid) == 0:
        return ExceptionValue(
            exception=ValueError('"predict_survival_function" returning None'),
        )
    else:
        return Series(
            [
                try_except(
                    lambda: fn(time), {Exception: lambda e: ExceptionValue(exception=e)}
                )
                for fn, time in zip(
                    survival_fns_valid,
                    (
                        time
                        if isinstance(time, Iterable)
                        else ([time] * len(survival_fns_valid))
                    ),
                )
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
        y_column=y["name"],
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
        y_column=y["name"],
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
            "y_pred": (
                DFStandardScaler().fit_transform(prediction["y_pred"])
                if prediction["y_pred"] is not None
                else None
            ),
        },
        result,
    )
    return merge_predictions(standardized_result)


def merge_predictions(result: Result) -> Prediction:
    try:
        y_pred = pandas.concat(
            [prediction["y_pred"] for cohort, prediction in result.items()]
        )
        split = (list(y_pred.index), list(y_pred.index))
    except ValueError:
        y_pred = None
        split = None

    return Prediction(
        y_pred=y_pred,
        split=split,
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
        y_scores = [prediction["y_score"] for prediction in group.values()]
        y_proba = transpose_dict(group).get("y_proba")

        averaged_predictions[test_cohort] = {
            **group[0],
            "y_score": pandas.concat(y_scores, axis=1).mean(axis=1),
        }

        if y_proba:
            try:
                y_proba_joined = merge_with(
                    partial(pandas.concat, axis=1),
                    *y_proba.values(),
                )
            except TypeError as e:
                averaged_predictions[test_cohort]["y_proba"] = ExceptionValue(
                    exception=e, value=y_proba
                )
            else:
                y_proba_mean = valmap(
                    partial(DataFrame.mean, axis=1),
                    y_proba_joined,
                )
                averaged_predictions[test_cohort]["y_proba"] = y_proba_mean

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
    return series.map(weights).astype("float")


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


def get_times():
    return [360 * year for year in range(1, 11)]


def predict_survival_from_prediction(X, y, times, prediction):
    X_train, X_test, y_train, y_test = get_X_y_split(X, y, prediction)
    y_pred_fns = prediction["model"].predict_survival_function(X_test)

    y_pred = np.array([[y_pred_fn(time) for time in times] for y_pred_fn in y_pred_fns])
    return {
        **prediction,
        "y_pred_survival": y_pred,
        "times": times,
    }


def compute_cumulative_dynamic_auc_on_prediction(X, y, times, prediction):
    X_train, X_test, y_train, y_test = get_X_y_split(X, y, prediction)

    y_pred = (
        prediction["y_pred_survival"]
        if "y_pred_survival" in prediction
        else -prediction["y_pred"][0]
    )

    cda = cumulative_dynamic_auc(
        to_survival_y_records(y_train),
        to_survival_y_records(y_test),
        -y_pred,
        times,
    )

    return Series(cda[0], index=times)


def compute_ibs_on_prediction(X, y, times, prediction):
    X_train, X_test, y_train, y_test = get_X_y_split(X, y, prediction)

    y_pred = prediction["y_pred_survival"]

    ibs = brier_score(
        to_survival_y_records(y_train),
        to_survival_y_records(y_test),
        np.array(y_pred),
        times,
    )

    return Series(ibs[1], index=times)


def concat_predictions(acc, cda_per_time):
    if not isinstance(acc, Series) and acc is None:
        return cda_per_time
    else:
        return pandas.concat([acc, cda_per_time], axis=1)


def compute_t_ci_columns(df, confidence=0.95):
    results = []
    for index, row in df.iterrows():
        data = row.dropna().values.astype(float)  # Drop NaN values and convert to float
        mean = np.mean(data)
        se = stats.sem(data)  # Standard error of the mean
        h = se * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)  # Margin of error
        ci_lower = mean - h
        ci_upper = mean + h
        results.append((mean, ci_lower, ci_upper))
    result_df = pd.DataFrame(
        results, columns=["Mean", "CI Lower", "CI Upper"], index=df.index
    )
    return result_df


def bootstrap_confidence_interval(data, num_samples=1000, confidence=0.95):
    means = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower_bound = np.percentile(means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower_bound, upper_bound


def compute_bootstrap_ci_columns(df, num_samples=1000, confidence=0.95):
    results = []
    for index, row in df.iterrows():
        data = row.dropna().values.astype(float)  # Drop NaN values and convert to float
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(
            data, num_samples, confidence
        )
        results.append((mean, ci_lower, ci_upper))
    result_df = pd.DataFrame(
        results, columns=["Mean", "CI Lower", "CI Upper"], index=df.index
    )
    return result_df


def compute_survival_cda_per_split_with_ci(X, y, results):
    times = get_times()
    y_pred = map_predictions(
        results,
        partial(
            predict_survival_from_prediction,
            X,
            y,
            times,
        ),
    )
    y_cda = map_predictions(
        y_pred,
        partial(
            compute_cumulative_dynamic_auc_on_prediction,
            X,
            y,
            times,
        ),
    )
    y_cda_splits_concat = reduce_splits_across_repeats(y_cda, concat_predictions)
    y_cda_splits_ci = valmap(compute_bootstrap_ci_columns, y_cda_splits_concat)
    return y_cda_splits_ci


def compute_score_cda_per_split_with_ci(X, y, results):
    times = get_times()
    y_cda = map_predictions(
        results, partial(compute_cumulative_dynamic_auc_on_prediction, X, y, times)
    )
    y_cda_splits_concat = reduce_splits_across_repeats(y_cda, concat_predictions)
    y_cda_splits_ci = valmap(compute_bootstrap_ci_columns, y_cda_splits_concat)
    return y_cda_splits_ci


def compute_ibs_per_split_with_ci(X, y, results):
    times = get_times()
    y_pred = map_predictions(
        results,
        partial(
            predict_survival_from_prediction,
            X,
            y,
            times,
        ),
    )
    y_cda = map_predictions(y_pred, partial(compute_ibs_on_prediction, X, y, times))
    y_cda_splits_concat = reduce_splits_across_repeats(y_cda, concat_predictions)
    y_cda_splits_ci = valmap(compute_bootstrap_ci_columns, y_cda_splits_concat)
    return y_cda_splits_ci


def plot_ci_dataframe(df, fig=None, name=""):
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["CI Lower"],
            mode="lines",
            line=dict(color="lightgrey"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["CI Upper"],
            mode="lines",
            line=dict(color="lightgrey"),
            fill="tonexty",
            name=f"{name} Confidence Interval",
        )
    )

    # Add the mean line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Mean"],
            mode="lines+markers",
            name=f"{name}" if name else "Mean",
        )
    )

    fig.update_layout(
        title="Mean and Confidence Intervals",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )

    return fig


def map_predictions(
    results: Results, func: Callable[[Prediction], T1]
) -> List[Dict[Hashable, T1]]:
    mapped_results = []

    for result in results:
        mapped_result = {}

        for fold, prediction in result.items():
            mapped_result[fold] = func(prediction)

        mapped_results.append(mapped_result)

    return mapped_results


def reduce_splits_across_repeats(
    results: Results,
    func: Callable[[Any, Prediction], Any],
    initializer: Optional[Any] = None,
) -> Dict[Hashable, Any]:
    split_names = {split for result in results for split in result.keys()}

    reduced_results = {}
    for split_name in split_names:
        split_predictions = [
            result[split_name] for result in results if split_name in result
        ]
        reduced_value = reduce(
            lambda acc, pred: func(acc, pred), split_predictions, initializer
        )
        reduced_results[split_name] = reduced_value

    return reduced_results
