from IPython.core.display import display
from abc import ABC
from dataclasses import dataclass
from itertools import product

from toolz.curried import get_in
from typing import Union, Tuple, Optional, List, Callable, Any, Literal, Dict

import numpy
import numpy as np
from numpy import mean
from pandas import DataFrame, Series
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.utils import resample

from hcve_lib.custom_types import Prediction, Target, ExceptionValue, Splits, Metric, OptimizationDirection, Maximize, \
    Minimize, ValueWithStatistics, TargetData
from hcve_lib.data import binarize_event
from hcve_lib.evaluation_functions import target_to_survival_y_records
from hcve_lib.functional import flatten, pipe, t
from hcve_lib.splitting import resample_prediction_test
from hcve_lib.utils import get_y_split, loc, transpose_list, binarize, empty_dict
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
import plotly.express as px
from rpy2 import robjects
from rpy2.interactive.packages import importr
from sklearn.metrics import confusion_matrix, precision_recall_curve
import dill


@dataclass
class SubsetMetric(Metric, ABC):
    is_train: bool = False

    def get_y(
            self,
            y: Target,
            prediction: Prediction,
            both: bool = False,
    ) -> TargetData:
        y_train, y_test = get_y_split(y, prediction)
        y_train_data = y_train.data
        y_test_data = y_test.data

        if both:
            return y_train_data, y_test_data
        else:
            return y_train_data if self.is_train else y_test_data


class StratifiedMetric(Metric):

    def __init__(
            self,
            metric: Metric,
            splits: Splits,
    ):
        super()
        self.metric = metric
        self.splits = splits

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return [
            f'{prefix}__{name}' for prefix, name in product(
                self.splits.keys(),
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

    def get_values_(self, prediction, y):
        for name, index in self.splits.items():
            subsampled_prediction = resample_prediction_test(index, prediction)
            if len(subsampled_prediction['y_score']) > 0:
                yield self.metric.get_values(
                    subsampled_prediction,
                    loc(index, y, ignore_not_present=True),
                )
            else:
                yield ExceptionValue(exception=ExceptionValue(ValueError(f'Missing y_score for split "{name}"')))

    def get_direction(self) -> OptimizationDirection:
        return self.metric.get_direction()


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
                prediction['y_score'].index,
                n_samples=round(len(prediction['y_score'])),
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
                values_ = [value for value in values if not isinstance(value, ExceptionValue)]
                if len(values_) == 0:
                    values_to_return.append((ExceptionValue(value=values)))
                else:
                    values_to_return.append(statistic_from_bootstrap(values_))

        return values_to_return

    def get_direction(self) -> OptimizationDirection:
        return self.metric.get_direction()


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


@dataclass
class WeightedCIndex(Maximize, SubsetMetric):

    def __init__(
            self,
            target: Literal['y_score', 'y_proba'] = 'y_score',
            weight=None,
    ):
        self.target = target
        self.weight = weight

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return ['c_index']

    def get_values(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        if len(prediction[self.target]) == 0:
            return [ExceptionValue(
                prediction[self.target],
                ValueError('y_score empty'),
            )]

        # try:
        y_ = self.get_y(y, prediction)
        y_score = prediction[self.target]
        # TODO: HACK
        y_score = y_score[y_score.index.isin(y_['data'].index)]

        if self.weight is not None:
            weight = self.weight[y_['data'].index]
        else:
            weight = None

        intsurv = importr('intsurv')
        index = (
            intsurv.cIndex(
                robjects.FloatVector(y_['data']['tte']),
                robjects.FloatVector(y_['data']['label']),
                robjects.FloatVector(y_score),
                *([robjects.FloatVector(weight)] if weight is not None else []),
            )
        )
        return [index[0]]
        # except ValueError as e:
        #     print(f'{e=}')
        #     return [ExceptionValue(exception=e)]


def get_y_proba_for_time(
        prediction: Prediction,
        X: DataFrame,
        y: Target,
        time: int,
) -> Series:
    y_proba = prediction['y_proba'].get(time)
    if len(y_proba.isna()) == len(y_proba):
        y_proba = predict_proba_for_prediction(prediction, X, y, time)
    return y_proba


def predict_proba_for_prediction(
        prediction: Prediction,
        X: DataFrame,
        y: Target,
        time: int,
) -> Prediction:
    return prediction['method'].predict(
        X,
        y,
        prediction['split'],
        prediction['model'],
        prediction['method'],
        prediction['random_state'],
        time=time,
    )


class AUC(Maximize, SubsetMetric):
    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return ['auc']

    def get_values(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        try:
            y_ = self.get_y(y, prediction)
            return [roc_auc_score(y_, prediction['y_proba'])]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


class Accuracy(Maximize, SubsetMetric):
    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return ['auc']

    def get_values(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        try:
            y_ = self.get_y(y, prediction)
            return [accuracy_score(y_, prediction['y_proba'])]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


@dataclass
class CIndex(Maximize, SubsetMetric):

    def __init__(
            self,
            target: Literal['y_score', 'y_proba'] = 'y_score',
    ):
        self.target = target

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return ['c_index']

    def get_values(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        print(prediction)
        if len(prediction[self.target]) == 0:
            return [ExceptionValue(
                prediction[self.target],
                ValueError('y_score empty'),
            )]

        try:
            y_ = self.get_y(y, prediction)

            index: Tuple = concordance_index_censored(
                y_['data']['label'].to_numpy().astype(numpy.bool_),
                y_['data']['tte'],
                # TODO: HACK
                prediction[self.target]
                [prediction[self.target].index.isin(y_['data'].index.to_list())].to_numpy().flatten(),
            )
            return [index[0]]
        except ValueError as e:
            return [ExceptionValue(exception=e)]


def get_y_proba_for_time(
        prediction: Prediction,
        X: DataFrame,
        y: Target,
        time: int,
) -> Series:
    y_proba = prediction['y_proba'].get(time)
    if len(y_proba.isna()) == len(y_proba):
        y_proba = predict_proba_for_prediction(prediction, X, y, time)
    return y_proba


def predict_proba_for_prediction(
        prediction: Prediction,
        X: DataFrame,
        y: Target,
        time: int,
) -> Prediction:
    return prediction['method'].predict(
        X,
        y,
        prediction['split'],
        prediction['model'],
        prediction['method'],
        prediction['random_state'],
        time=time,
    )


class Brier(Minimize, SubsetMetric):
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
        y_train, y_test = self.get_y(y, prediction, both=True)
        y_train_ = target_to_survival_y_records(y_train)
        y_test_ = target_to_survival_y_records(y_test)
        values = []

        for time in self.get_times(prediction):
            try:
                values.append(
                    brier_score(
                        y_train_['data'],
                        y_test_['data'],
                        get_y_proba_for_time(
                            prediction,
                            self.X,
                            y,
                            time,
                        ).loc[y_test['data'].index],
                        time,
                    )[1][0]
                )
            except Exception as e:
                values.append(ExceptionValue(e))

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return [f'brier_{time}' for time in self.get_times(prediction)]

    def get_times(
            self,
            prediction: Prediction,
    ) -> List[int]:
        if self.time:
            return [self.time]
        else:
            available = list(prediction['y_proba'].keys())
            if len(available) > 0:
                return available
            else:
                raise ValueError('There are no available y_proba to compute brier score')


class SimpleBrier(Minimize, SubsetMetric):
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
        return [get_simple_brier_for_time(time, self.X, y_, prediction) for time in self.get_times(prediction)]

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return [f'simple_brier_{time}' for time in self.get_times(prediction)]

    def get_times(self, prediction: Prediction) -> List[int]:
        if self.time:
            return [self.time]
        else:
            return list(prediction['y_proba'].keys())


def get_simple_brier_for_time(
        time: int,
        X: DataFrame,
        y: Target,
        prediction: Prediction,
):
    try:
        return brier_score_loss(
            y_binary := 1 - binarize_event(time, y['data']).dropna(),
            get_y_proba_for_time(
                prediction,
                X,
                y,
                time,
            ).loc[y_binary.index],
        )
    except Exception as e:
        return ExceptionValue(None, e)

    # class IntegratedBrier(Minimize, SubsetMetric):
    #     X: DataFrame
    #     time: Optional[int]
    #
    #     def __init__(
    #         self,
    #         X: DataFrame,
    #         time: Optional[int] = None,
    #         *args,
    #         **kwargs,
    #     ):
    #         super().__init__(*args, **kwargs)
    #         self.X = X
    #         self.time = time
    #
    #     def get_values(
    #         self,
    #         prediction: Prediction,
    #         y: Target,
    #     ) -> List[Union[ExceptionValue, float]]:
    #         y_train, y_test = self.get_y(y, prediction, both=True)
    #         y_train_ = target_to_survival_y_records(y_train)
    #         y_test_ = target_to_survival_y_records(y_test)
    #         values = []
    #
    #         for time in self.get_times(prediction):
    #             try:
    #                 values.append(
    #                     integrated_brier_score(
    #                         y_train_['data'],
    #                         y_test_['data'],
    #                         get_y_proba_for_time(
    #                             prediction,
    #                             self.X,
    #                             y,
    #                             time,
    #                         ).loc[y_test['data'].index],
    #                         time,
    #                     )[1][0])
    #             except Exception as e:
    #                 values.append(ExceptionValue(e))

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return [f'brier_{time}' for time in self.get_times(prediction, y)]

    def get_times(
            self,
            prediction: Prediction,
    ) -> List[int]:
        if self.time:
            return [self.time]
        else:
            available = list(prediction['y_proba'].keys())
            if len(available) > 0:
                return available
            else:
                raise ValueError('There are no available y_proba to compute brier score')


class AtTime:
    time: Optional[int] = None

    def get_times(self, prediction: Prediction, y: Target) -> List[int]:
        if self.time == 'all_times':
            return y['data']['tte'].to_list()
        if self.time:
            return [self.time]
        else:
            return list(prediction['y_proba'].keys())


class BinaryMetricAtTime(SubsetMetric, AtTime):

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
        y_binarized = binarize_event(self.time, y_['data'])
        try:
            return [
                self.binary_metric(
                    y_binarized,
                    loc(
                        y_binarized.index,
                        1 - prediction['y_proba'][time],
                        ignore_not_present=True,
                    ),
                ) for time in self.get_times(prediction, y)
            ]
        except KeyError as e:
            raise KeyError(f'Only {prediction["y_proba"].keys()} available')

    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        return [f'{self.binary_metric.__name__}_{time}' for time in self.get_times(prediction, y)]

    def get_direction(self) -> OptimizationDirection:
        return self.direction


class BinaryMetricFromScore(SubsetMetric, AtTime):

    def __init__(
            self,
            binary_metric: Callable,
            time: Union[Optional[int], str] = None,
            direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
            target=get_in(['y_score']),
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
            y_binarized = binarize_event(time, y_['data'])
            if self.sample_weight is not None:
                sample_weight_ = self.sample_weight.loc[y_binarized.index]
            else:
                sample_weight_ = None

            y_score = prediction['y_score'][(prediction['y_score'].index.isin(y_['data'].index))
                                            & (prediction['y_score'].index.isin(y_binarized.index))]

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
        return [f'{self.binary_metric.__name__}_{time}' for time in self.get_times(prediction, y)]

    def get_direction(self) -> OptimizationDirection:
        return self.direction


def precision_recall_curve_with_confusion(y_true, probas_pred, *args, sample_weight=None, **kwargs):
    # TODO: HACK
    index_intersection = probas_pred.index.drop_duplicates().intersection(y_true.index.drop_duplicates()
                                                                          ).drop_duplicates()

    # print(len(index_intersection))
    probas_pred_ = Series(
        [
            v if isinstance(v := probas_pred.loc[index], float) else v.iloc[0]
            for index in probas_pred.index.drop_duplicates()
        ],
        index=index_intersection.drop_duplicates()
    )
    # print(len(probas_pred_))
    y_true_ = Series(
        [
            v if isinstance(v := y_true.loc[index], float) else v.iloc[0]
            for index in probas_pred_.index.drop_duplicates()
        ],
        index=probas_pred_.index.drop_duplicates()
    )
    # print(len(y_true_))

    if sample_weight is not None:
        sample_weight_ = Series(
            [
                v if isinstance(v := sample_weight.loc[index], float) else v.iloc[0]
                for index in probas_pred_.index.drop_duplicates()
            ],
            index=probas_pred_.index.drop_duplicates()
        )
    else:
        sample_weight_ = None

    with open('./output/session.plk', 'wb') as f:
        dill.dump([y_true, probas_pred, y_true_, probas_pred_, index_intersection], f)

    precision, recall, thresholds = precision_recall_curve(
        y_true_, probas_pred_, *args, sample_weight=sample_weight_, **kwargs,
    )
    confusion_matrices = []
    for threshold in thresholds:
        confusion_matrices.append(confusion_matrix(y_true_, probas_pred_ >= threshold, *args, **kwargs))

    return precision, recall, confusion_matrices, thresholds
