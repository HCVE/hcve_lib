from typing import List, Union
from unittest.mock import Mock

import numpy as np
from numpy import mean
from pandas import Series, DataFrame
from pandas._testing import assert_series_equal

from hcve_lib.custom_types import Prediction, Target, ExceptionValue
from hcve_lib.metrics import StratifiedMetric, SimpleBrier, BootstrappedMetric, BinaryMetricAtTime, FunctionMetric
from hcve_lib.metrics_types import Metric, OptimizationDirection


def test_StratifiedMetric():
    class DummyMetric(Metric):

        def get_direction(self) -> OptimizationDirection:
            pass

        def get_names(
                self,
                prediction: Prediction,
                y: Target,
        ) -> List[str]:
            return ['a', 'b']

        def get_values(
                self,
                prediction: Prediction,
                y: Target,
        ) -> List[Union[ExceptionValue, float]]:
            assert list(prediction['y_score'].index) == [10, 30]
            assert list(prediction['y_proba']['x'].index) == [10, 30]
            return [0, 1]

    metric = StratifiedMetric(
        DummyMetric(),
        {0: [10, 30]},
    )
    prediction = {
        'y_proba': {
            'x': Series([10, 20, 30], index=[10, 20, 30])
        },
        'y_score': Series([10, 20, 30], index=[10, 20, 30]),
    }
    y = Series([100, 200, 300], index=[10, 20, 30])

    assert metric.get_values(
        prediction,
        y=y,
    ) == [0, 1]

    assert metric.get_names(
        prediction,
        y=y,
    ) == ['0__a', '0__b']


def test_BootstrappedMetric():
    class DummyMetric(Metric):

        def get_names(
                self,
                prediction: Prediction,
                y: Target,
        ) -> List[str]:
            return ['a']

        def get_values(
                self,
                prediction: Prediction,
                y: Target,
        ) -> List[Union[ExceptionValue, float]]:
            return [mean(prediction['y_score'])]

        def get_direction(self) -> OptimizationDirection:
            pass

    np.random.seed(1)
    metric = BootstrappedMetric(DummyMetric(), random_state=1)

    prediction = {
        'y_score': Series([np.random.normal(loc=0) for _ in range(50)])
    }
    y = Series(list(range(50)))

    assert metric.get_values(
        prediction,
        y=y,
    ) == [{
        'ci': (-0.3834587567553192, 0.3471052170739042),
        'mean': -0.025577169726534905
    }]

    # STRATIFICATION


def test_SimpleBrier():
    brier = SimpleBrier(time=100, is_train=True)
    assert brier.get_values(
        {
            'y_proba': {
                100: Series([0.5, 0, 1], index=[1, 2, 3])
            },
            'split': ([1, 2], [3]),
        },
        {
            'data':
                DataFrame(
                    {
                        'label': [1, 1, 1],
                        'tte': [10, 101, 1000]
                    },
                    index=[1, 2, 3],
                )
        },
    ) == [0.625]


def test_BinaryMetricAtTime():
    prediction = {
        'y_proba': {
            20: Series([0.1, 0.1, 0.2, 0.3])
        },
        'split': [[], [0, 1, 2, 3]],
    }

    target = {
        'data': DataFrame({
            'label': [0, 1, 1, 0],
            'tte': [10, 10, 20, 30]
        })
    }

    binary_metric = Mock()

    BinaryMetricAtTime(
        binary_metric,
        time=20,
        threshold=0.2,
    ).get_values(
        prediction,
        target,
    )

    y_true, y_pred = binary_metric.call_args[0]

    assert_series_equal(y_pred, Series([0, 1, 1], index=[1, 2, 3]))
    assert_series_equal(y_true, Series([1, 1, 0], index=[1, 2, 3]))


def test_FunctionMetric():
    def dummy_metric(y_true: Series, y_pred: Series):
        return (y_true == y_pred).sum()

    assert FunctionMetric(dummy_metric).get_names({}, Series()) == ['dummy_metric']

    prediction = Prediction(
        y_pred=Series([1, 2, 3], index=[10, 20, 30]),
        split=([], [10, 20, 30]),
    )
    y = Series([1, 3, 3], index=[10, 20, 30])
    metric = FunctionMetric(dummy_metric)
    assert metric.get_names(
        prediction,
        y,
    ) == ['dummy_metric']

    assert metric.get_values(
        prediction,
        y,
    ) == [2]
