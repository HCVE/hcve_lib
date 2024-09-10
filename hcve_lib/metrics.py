from dataclasses import dataclass
from typing import Union, Tuple, Optional, List, Callable, Any, Iterable

from prettytable import PLAIN_COLUMNS, PrettyTable
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from toolz import valmap
from typing_extensions import Literal

import dill
import numpy
import numpy as np
from itertools import product
from numpy import mean
from pandas import DataFrame, Series
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    average_precision_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils import resample
from toolz.curried import get_in

from hcve_lib.custom_types import (
    Prediction,
    Target,
    ExceptionValue,
    Splits,
    ValueWithStatistics,
    ValueWithCI,
    TargetType,
    Index,
)
from hcve_lib.data import binarize_event, to_survival_y_records
from hcve_lib.functional import flatten, pipe
from hcve_lib.metrics_types import Metric, OptimizationDirection
from hcve_lib.pipelines import get_target_type
from hcve_lib.splitting import resample_prediction_test
from hcve_lib.utils import (
    loc,
    transpose_list,
    get_y_split,
    get_1_class_y_score,
    is_noneish,
)
from hcve_lib.visualisation import display_html, p


def statistic_from_bootstrap(values):
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = numpy.percentile(values, p)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = numpy.percentile(values, p)
    return ValueWithStatistics(
        mean=mean(values),
        ci=(lower, upper),
        std=(np.sum((values - np.mean(values)) ** 2) / (len(values) - 2)) ** (1 / 2),
    )


class BootstrappedMetric(Metric):
    def __init__(
        self,
        metric: Metric,
        random_state: int,
        iterations: int = 100,
        return_summary: bool = True,
    ):
        super()
        self.metric = metric
        self.random_state = random_state
        self.iterations = iterations
        self.return_summary = return_summary

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return self.metric.get_names(prediction, y)

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        metric_values = []
        iteration_tryout = 0
        iteration_success = 0
        max_iterations = self.iterations * 5

        while iteration_success < self.iterations and iteration_tryout < max_iterations:
            sample_index = resample(
                prediction["y_pred"].index,
                n_samples=round(len(prediction["y_pred"])),
                random_state=self.random_state + iteration_tryout,
            )

            sample_prediction = resample_prediction_test(
                sample_index,
                prediction,
            )

            values = self.metric.get_values(
                sample_prediction,
                y,
            )

            iteration_tryout += 1

            if any(isinstance(value, ExceptionValue) for value in values):
                print(values)
                continue
            else:
                metric_values.append(values)
                iteration_success += 1

        values_per_names = transpose_list(metric_values)
        if not self.return_summary:
            if iteration_success == 0:
                return [ExceptionValue() for _ in self.get_names(prediction, y)]
            return values_per_names
        else:
            values_to_return = []

            for values in values_per_names:
                values_ = [
                    value for value in values if not isinstance(value, ExceptionValue)
                ]
                if len(values_) == 0:
                    values_to_return.append((ExceptionValue(value=values)))
                else:
                    values_to_return.append(statistic_from_bootstrap(values_))

        return values_to_return

    def get_direction(self) -> OptimizationDirection:
        return self.metric.get_direction()


class Maximize:
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE


class Minimize:
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE


@dataclass
class WeightedCIndex(Maximize, Metric):
    def __init__(
        self,
        weight=None,
    ):
        self.weight = weight

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["weighted_c_index"]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        from rpy2 import robjects
        from rpy2.interactive.packages import importr

        y_ = self.get_y(y, prediction)
        y_score = prediction["y_pred"]

        if isinstance(y_score, DataFrame):
            y_score = y_score[0]

        y_ = loc(y_score.index, y)

        if self.weight is not None:
            weight = self.weight.loc[y_.data.index]
        else:
            weight = None

        intsurv = importr("intsurv")

        index = intsurv.cIndex(
            robjects.FloatVector(y_.data["tte"]),
            robjects.FloatVector(y_.data["label"]),
            robjects.FloatVector(y_score),
            *([robjects.FloatVector(weight)] if weight is not None else []),
        )
        return [index[0]]
        # except ValueError as e:
        #     print(f'{e=}')
        #     return [ExceptionValue(exception=e)]

    def compute(self): ...


@dataclass
class WeightedCIndex2(Maximize, Metric):
    def __init__(
        self,
        weight=None,
    ):
        self.weight = weight

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["c_index"]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        y_ = self.get_y(y, prediction)
        y_score = prediction["y_pred"]
        y_ = loc(y_score.index, y)
        if self.weight is not None:
            weights = self.weight.loc[y_.data.index]
        else:
            weights = None

        weights = weights.to_numpy()
        concordant_sum = 0.0
        permissible_pairs_sum = 0.0
        event_times = y_.data["tte"].to_numpy()
        event_observed = y_.data["label"].to_numpy()
        predicted_scores = -y_score[0].to_numpy()
        n = len(y_)

        for i in range(n):
            for j in range(i + 1, n):
                if event_times[i] != event_times[j]:  # permissible pair
                    weight = weights[i] * weights[j]
                    if event_times[i] < event_times[j] and event_observed[i] == 1:
                        permissible_pairs_sum += weight
                        if predicted_scores[i] < predicted_scores[j]:
                            concordant_sum += weight
                        elif predicted_scores[i] == predicted_scores[j]:
                            concordant_sum += 0.5 * weight
                    elif event_times[j] < event_times[i] and event_observed[j] == 1:
                        permissible_pairs_sum += weight
                        if predicted_scores[j] < predicted_scores[i]:
                            concordant_sum += weight
                        elif predicted_scores[j] == predicted_scores[i]:
                            concordant_sum += 0.5 * weight

        if permissible_pairs_sum == 0:
            return 0.5  # If no permissible pairs, return 0.5

        return [concordant_sum / permissible_pairs_sum]

        # except ValueError as e:
        #     print(f'{e=}')
        #     return [ExceptionValue(exception=e)]

    def compute(self): ...


def get_y_proba_for_time(
    prediction: Prediction,
    X: DataFrame,
    y: Target,
    time: int,
) -> Series:
    y_proba = prediction["y_pred"].get(time)
    if len(y_proba.isna()) == len(y_proba):
        y_proba = predict_proba_for_prediction(prediction, X, y, time)
    return y_proba


def predict_proba_for_prediction(
    prediction: Prediction,
    X: DataFrame,
    y: Target,
    time: int,
) -> Prediction:
    return prediction["method"].predict(X)


class ROC_AUC(Maximize, Metric):
    def __init__(self):
        super().__init__()

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["roc_auc"]

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        try:
            if len(y_true.unique()) <= 2:
                try:
                    return [roc_auc_score(y_true, y_pred[1])]
                except KeyError:
                    return [roc_auc_score(y_true, y_pred[0])]
            else:
                return [roc_auc_score(y_true, y_pred, multi_class="ovo")]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


class FunctionMetric(Metric):
    def __init__(
        self,
        function: Callable,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.function = function
        self.direction = direction

    def get_values(
        self, prediction: Prediction, y: Target
    ) -> List[Union[ExceptionValue, float, ValueWithCI]]:
        y_ = self.get_y(y, prediction)
        return [self.function(y_, prediction["y_pred"])]

    def get_direction(self) -> OptimizationDirection:
        return self.direction

    def get_names(self, prediction: Prediction, y: Target) -> List[str]:
        return [self.function.__name__]

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        return self.function(y_true, y_pred)


class PR_AUC(Maximize, Metric):
    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["pr_auc"]

    def compute(self, y_true: Target, y_pred: DataFrame):
        try:
            try:
                return [average_precision_score(y_true, y_pred[1])]
            except KeyError:
                return [average_precision_score(y_true, y_pred[0])]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


class Accuracy(Maximize, Metric):
    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["accuracy"]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        try:
            y_ = self.get_y(y, prediction)
            return [
                accuracy_score(
                    prediction["y_pred"][1],
                    y_,
                )
            ]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


class MeanSquaredError(Maximize, Metric):
    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["mse"]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        try:
            y_ = self.get_y(y, prediction)
            return [
                mean_squared_error(
                    prediction["y_pred"][1],
                    y_,
                )
            ]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


@dataclass
class CIndex(Maximize, Metric):
    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["c_index"]

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        try:
            value = concordance_index_censored(
                y_true.data["label"].to_numpy().astype(numpy.bool_),
                y_true.data["tte"],
                get_1_class_y_score(y_pred),
            )
            return [value[0]]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


@dataclass
class IBS(Minimize, Metric):
    def __init__(
        self, X: DataFrame, times: Iterable[int] = None, target: str = "y_pred"
    ):
        super().__init__()
        self.times = times
        self.target = target
        self.X = X

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return ["ibs"]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        y_train, y_test = get_y_split(y, prediction)
        print(prediction.keys())
        y_proba_table = prediction["model"].predict_proba_table(self.X).T
        ibs = integrated_brier_score(
            to_survival_y_records(y_train),
            to_survival_y_records(y_test),
            y_proba_table.to_numpy(),
            y_proba_table.columns,
        )


def get_y_proba_for_time(
    prediction: Prediction,
    X: DataFrame,
    y: Target,
    time: int,
) -> Series:
    y_proba = prediction["y_proba"].get(time)
    if len(y_proba.isna()) == len(y_proba):
        y_proba = predict_proba_for_prediction(prediction, X, y, time)
    return y_proba


def predict_proba_for_prediction(
    prediction: Prediction,
    X: DataFrame,
    y: Target,
    time: int,
) -> Prediction:
    return prediction["method"].predict(X)


class Minimize:
    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE


class Brier(Minimize, Metric):
    X: DataFrame
    time: Optional[int]

    def __init__(
        self,
        X: DataFrame,
        time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.X = X
        self.time = time

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        y_train, y_test = get_y_split(y, prediction)
        y_train_ = target_to_survival_y_records(y_train)
        y_test_ = target_to_survival_y_records(y_test)
        values = []

        for time in self.get_times(prediction):
            try:
                from sksurv.metrics import brier_score

                values.append(
                    brier_score(
                        y_train_["data"],
                        y_test_["data"],
                        get_y_proba_for_time(
                            prediction,
                            self.X,
                            y,
                            time,
                        ).loc[y_test["data"].index],
                        time,
                    )[0]
                )
            except Exception as e:
                values.append(ExceptionValue(e))

        return values

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        raise NotImplementedError

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return [f"brier_{time}" for time in self.get_times(prediction)]

    def get_times(
        self,
        prediction: Prediction,
    ) -> List[int]:
        if self.time:
            return [self.time]
        else:
            available = list(prediction["y_pred"].keys())
            if len(available) > 0:
                return available
            else:
                raise ValueError(
                    "There are no available y_proba to compute brier score"
                )


class SimpleBrier(Minimize, Metric):
    time: Optional[int]

    def __init__(
        self,
        X: DataFrame,
        time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.X = X
        self.time = time

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        y_ = self.get_y(y, prediction)
        return [
            get_simple_brier_for_time(time, self.X, y_, prediction)
            for time in self.get_times(prediction)
        ]

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return [f"simple_brier_{time}" for time in self.get_times(prediction)]

    def get_times(self, prediction: Prediction) -> List[int]:
        if self.time:
            return [self.time]
        else:
            return list(prediction["y_proba"].keys())


def get_simple_brier_for_time(
    time: int,
    X: DataFrame,
    y: Target,
    prediction: Prediction,
):
    try:
        y_binary = (1 - binarize_event(time, y.data).dropna(),)
        return brier_score_loss(
            y_binary,
            get_y_proba_for_time(
                prediction,
                X,
                y,
                time,
            ).loc[y_binary],
        )
    except Exception as e:
        return ExceptionValue(None, e)

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return [f"brier_{time}" for time in self.get_times(prediction, y)]

    def get_times(
        self,
        prediction: Prediction,
    ) -> List[int]:
        if self.time:
            return [self.time]
        else:
            available = list(prediction["y_proba"].keys())
            if len(available) > 0:
                return available
            else:
                raise ValueError(
                    "There are no available y_proba to compute brier score"
                )


class AtTime:
    time: Optional[int] = None

    def get_times(self, prediction: Prediction, y: Target) -> List[int]:
        if self.time == "all_times":
            return y.data["tte"].to_list()
        if self.time:
            return [self.time]
        else:
            return list(prediction["y_proba"].keys())


class BinaryMetricAtTime(Metric, AtTime):
    def __init__(
        self,
        binary_metric: Callable,
        time: Optional[int] = None,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time = time
        self.binary_metric = binary_metric
        self.direction = direction

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Any]:
        y_ = self.get_y(y, prediction)
        y_binarized = binarize_event(self.time, y_.data)
        try:
            return [
                self.binary_metric(
                    y_binarized,
                    loc(
                        y_binarized.index,
                        1 - prediction["y_proba"][time],
                        ignore_not_present=True,
                    ),
                )
                for time in self.get_times(prediction, y)
            ]
        except KeyError as e:
            raise KeyError(f'Only {prediction["y_proba"].keys()} available')

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return [
            f"{self.binary_metric.__name__}_{time}"
            for time in self.get_times(prediction, y)
        ]

    def get_direction(self) -> OptimizationDirection:
        return self.direction


class BinaryMetricFromScore(Metric, AtTime):
    def __init__(
        self,
        binary_metric: Callable,
        time: Union[Optional[int], str] = None,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        target=get_in(["y_score"]),
        sample_weight: Series = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time = time
        self.binary_metric = binary_metric
        self.direction = direction
        self.sample_weight = sample_weight

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Any]:
        y_ = self.get_y(y, prediction)
        out = []
        for time in self.get_times(prediction, y):
            y_binarized = binarize_event(time, y_.data)
            if self.sample_weight is not None:
                sample_weight_ = self.sample_weight.loc[y_binarized.index]
            else:
                sample_weight_ = None

            y_score = prediction["y_score"][
                (prediction["y_score"].index.isin(y_.data.index))
                & (prediction["y_score"].index.isin(y_binarized.index))
            ]

            out.append(
                self.binary_metric(
                    y_binarized.astype(float),
                    y_score.astype(float),
                    sample_weight=sample_weight_,
                )
            )

        return out

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]:
        return [
            f"{self.binary_metric.__name__}_{time}"
            for time in self.get_times(prediction, y)
        ]

    def get_direction(self) -> OptimizationDirection:
        return self.direction


def precision_recall_curve_with_confusion(
    y_true, probas_pred, *args, sample_weight=None, **kwargs
):
    # TODO: HACK
    index_intersection = (
        probas_pred.index.drop_duplicates()
        .intersection(y_true.index.drop_duplicates())
        .drop_duplicates()
    )

    probas_pred_ = Series(
        [
            (
                probas_pred.loc[index]
                if isinstance(probas_pred.loc[index], float)
                else probas_pred.loc[index].iloc[0]
            )
            for index in probas_pred.index.drop_duplicates()
        ],
        index=index_intersection.drop_duplicates(),
    )

    y_true_ = Series(
        [
            (
                y_true.loc[index]
                if isinstance(y_true.loc[index], float)
                else y_true.loc[index].iloc[0]
            )
            for index in probas_pred_.index.drop_duplicates()
        ],
        index=probas_pred_.index.drop_duplicates(),
    )

    if sample_weight is not None:
        sample_weight_ = Series(
            [
                (
                    sample_weight.loc[index]
                    if isinstance(sample_weight.loc[index], float)
                    else sample_weight.loc[index].iloc[0]
                )
                for index in probas_pred_.index.drop_duplicates()
            ],
            index=probas_pred_.index.drop_duplicates(),
        )
    else:
        sample_weight_ = None

    with open("./output/session.plk", "wb") as f:
        dill.dump([y_true, probas_pred, y_true_, probas_pred_, index_intersection], f)

    precision, recall, thresholds = precision_recall_curve(
        y_true_,
        probas_pred_,
        *args,
        sample_weight=sample_weight_,
        **kwargs,
    )
    confusion_matrices = []
    for threshold in thresholds:
        confusion_matrices.append(
            confusion_matrix(y_true_, probas_pred_ >= threshold, *args, **kwargs)
        )

    return precision, recall, confusion_matrices, thresholds


class WeightedMetric(Metric):
    def __init__(
        self,
        metric: Metric,
        weights: Series,
    ):
        super()
        self.metric = metric
        self.weights = weights

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List:
        return ["Weighted" + name for name in self.metric.get_names(prediction, y)]

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        y_pred_resampled = y_pred.sample(frac=1, replace=True, weights=self.weights)
        y_true_resampled = loc(y_pred_resampled.index, y_true)
        return self.metric.compute(y_true_resampled, y_pred_resampled)

    def get_direction(self) -> OptimizationDirection:
        return self.metric.get_direction()


class StratifiedMetric(Metric):
    def __init__(
        self,
        metric: Metric,
        by: Splits,
    ):
        super()
        self.metric = metric
        self.by = by

    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List:
        return [
            (prefix, name)
            for prefix, name in product(
                self.by.keys(),
                self.metric.get_names(prediction, y),
            )
        ]

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        return pipe(
            self.get_values_(prediction, y),
            list,
            flatten,
            list,
        )

    def compute(
        self, y_true: Target, y_pred: DataFrame
    ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]:
        raise NotImplementedError

    def get_values_(self, prediction, y):
        for name, index in self.by.items():
            subsampled_prediction = resample_prediction_test(index, prediction)
            if len(subsampled_prediction["y_pred"]) > 0:
                yield self.metric.get_values(
                    subsampled_prediction,
                    loc(subsampled_prediction["split"][1], y, ignore_not_present=True),
                )
            else:
                yield ExceptionValue(
                    exception=ValueError(f'Missing y_pred for split "{name}"')
                )

    def get_direction(self) -> OptimizationDirection:
        return self.metric.get_direction()


def get_standard_metrics(y: Target) -> List[Metric]:
    target_type = get_target_type(y)
    return get_standard_metrics_for_target_type(target_type)


def get_standard_metrics_for_target_type(target_type: TargetType) -> List[Metric]:
    if target_type == TargetType.REGRESSION:
        return get_standard_regression_metrics()
    elif target_type == TargetType.CLASSIFICATION:
        return get_standard_classification_metrics()
    elif target_type == TargetType.TIME_TO_EVENT:
        return get_standard_time_to_event_metrics()
    else:
        raise NotImplementedError


def get_standard_regression_metrics():
    return [
        FunctionMetric(r2_score),
        FunctionMetric(mean_absolute_error),
    ]


def get_standard_classification_metrics() -> List[Metric]:
    return [
        ROC_AUC(),
        PR_AUC(),
    ]


def target_to_survival_y_records(y):
    if not isinstance(y.data, numpy.recarray):
        return {**y, "data": to_survival_y_records(y)}
    else:
        return y


def get_standard_time_to_event_metrics() -> List[Metric]:
    return [
        CIndex(),
    ]
