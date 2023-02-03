from typing import Tuple, Dict, List, Union
from unittest import mock
from unittest.mock import Mock

from optuna import Trial
from pandas import DataFrame, Series
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal

from hcve_lib.custom_types import Target, Estimator, Prediction, ExceptionValue, ValueWithCI
from hcve_lib.cv import cross_validate_single_repeat_, OptimizationParams
from hcve_lib.metrics_types import Metric, OptimizationDirection
from hcve_lib.wrapped_sklearn import DFPipeline


def test_cross_validate():

    class MockEstimator(Estimator):

        def __init__(self, hyperparameter1=None):
            self.hyperparameter1 = hyperparameter1

        def predict(self, X: DataFrame):
            return X['x']

        def set_params(self, **params):
            breakpoint()

    def get_splits(X, y, random_state):
        return {'split': (['a', 'b'], ['c'])}

    X = DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
    y = Series([10, 20, 30], name='y', index=['a', 'b', 'c'])

    result = cross_validate_single_repeat_(lambda _1, _2: MockEstimator(), X, y, get_splits)
    assert_series_equal(Series([3], index=['c'], name='x'), result['split']['y_pred'])


# test whether the model was set with hyperparameters and trained on correct subsets
def test_cross_validate_optimize():

    class MockTransform(Estimator):

        def fit_transform(self, X: DataFrame, y: Target):
            return X

        def transform(self, X: DataFrame):
            return X

    class MockEstimator(Estimator):

        def __init__(self, hyperparameter1=None):
            self.hyperparameter1 = hyperparameter1

        def predict_proba(self, X: DataFrame):
            return X['x']

        def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
            return Mock(), {f'hyperparameter1': 10}

        def set_params(self, **params):
            breakpoint()

    step1 = MockTransform()
    step2 = MockEstimator()

    class MockMetric(Metric):

        def get_names(
            self,
            prediction: Prediction,
            y: Target,
        ) -> List[str]:
            return ['metric1']

        def get_values(
            self,
            prediction: Prediction,
            y: Target,
        ) -> List[Union[ExceptionValue, float, ValueWithCI]]:
            return [0.6]

        def get_direction(self) -> OptimizationDirection:
            return OptimizationDirection.MAXIMIZE

    X = DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
    y = Series([10, 20, 30], name='y', index=['a', 'b', 'c'])

    def get_splits(X, y, random_state):
        return {'split': (['a', 'b'], ['c'])}

    pipeline = DFPipeline([
        ('step1', step1),
        ('step2', step2),
    ])

    with mock.patch.object(pipeline, 'set_params') as set_params:
        with mock.patch.object(pipeline, 'fit') as fit:
            cross_validate_single_repeat_(
                lambda random_state: pipeline,
                X,
                y,
                get_splits=get_splits,
                random_state=1,
                optimize=True,
                optimize_params=OptimizationParams(
                    n_trials=1,
                    objective_metric=MockMetric(),
                    get_splits=lambda *args, **kwargs: {'split1': (['a'], ['b'])},
                ),
                n_jobs=1
            )

            # right hyperparameters set
            assert len(set_params.call_args_list) == 2
            assert set_params.call_args_list[0].kwargs == {'step2__hyperparameter1': 10}
            assert len(fit.call_args_list) == 2

            # right data subset (1. evaluating optimization and 2. fitting optimized model on the whole train set)
            assert_frame_equal(fit.call_args_list[0].args[0], DataFrame({'x': [1]}, index=['a']))
            assert_frame_equal(fit.call_args_list[1].args[0], DataFrame({'x': [1, 2]}, index=['a', 'b']))
