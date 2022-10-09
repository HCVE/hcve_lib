from dataclasses import dataclass
from itertools import product
from typing import Union, List, Literal

import numpy
import numpy as np
from numpy import mean
from sklearn.utils import resample

from hcve_lib.custom_types import Prediction, Target, ExceptionValue, Splits, Metric, OptimizationDirection, Maximize, \
    ValueWithStatistics
from hcve_lib.functional import flatten, pipe
from hcve_lib.splitting import resample_prediction_test
from hcve_lib.utils import get_y_split, loc, transpose_list
from rpy2 import robjects
from rpy2.interactive.packages import importr


@dataclass
class SubsetMetric(Metric):
    is_train: bool = False

    def get_y(
        self,
        y: Target,
        prediction: Prediction,
        both: bool = False,
    ):
        y_train, y_test = get_y_split(y, prediction)
        if both:
            return y_train, y_test
        else:
            return y_train if self.is_train else y_test


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
