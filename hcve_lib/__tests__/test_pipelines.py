from functools import partial
from typing import Tuple, Dict
from unittest.mock import Mock

from optuna import Trial
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from statsmodels.compat.pandas import assert_frame_equal

from hcve_lib.custom_types import Model
from hcve_lib.cv import cross_validate_single_repeat_
from hcve_lib.pipelines import prepend_timeline, aggregate_results
from hcve_lib.splitting import get_train_test
from hcve_lib.wrapped_sklearn import DFPipeline

RANDOM_STATE = 52857613


def test_prepend_timeline():

    class DummyClassifier(BaseEstimator):

        def __init__(self):
            self.y = None

        def fit(self, X, y):
            self.y = y + X['x']

        def predict(self, X):
            return self.y

    old_pipeline = DFPipeline([('step1', DummyClassifier())])
    new_pipeline = prepend_timeline(
        old_pipeline,
        ('step0', FunctionTransformer(lambda s: s + 1)),
    )

    old_pipeline.fit(DataFrame({'x': [10, 10]}), Series([1, 2]))
    assert_series_equal(old_pipeline.predict(DataFrame()), Series([11, 12])),

    new_pipeline.fit(DataFrame({'x': [10, 10]}), Series([1, 2]))
    assert_series_equal(new_pipeline.predict(DataFrame()), Series([12, 13])),


def test_Model():
    data = DataFrame({'x': list(range(10, 20)), 'y': list(range(100, 200, 10))})

    class TestModel(Model):

        def get_estimator(self) -> BaseEstimator:
            return LinearRegression()

    result = cross_validate_single_repeat_(
        lambda random_state: TestModel(),
        data, ['x'],
        get_splits=partial(get_train_test, test_size=0.5),
        random_state='y',
        n_jobs=1
    )

    assert result['train_test']['split'] == ([0, 1, 7, 2, 5], [4, 8, 6, 3, 9])
    assert result['train_test']['X_columns'] == ['x']
    assert result['train_test']['y_column'] == 'y'

    assert_series_equal(
        result['train_test']['y_target'], Series([140.0, 180.0, 160.0, 130.0, 190.0], index=[4, 8, 6, 3, 9])
    )


def test_Model_suggest_optuna():

    class TestStep1(Model):

        @staticmethod
        def suggest_optuna(trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
            return trial, {
                f'{prefix}_hyperparameter1': trial.suggest_categorical(f'{prefix}_hyperparameter', ['a', 'b', 'c'])
            }

    class TestStep2(Model):

        @staticmethod
        def suggest_optuna(trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
            return trial, {
                f'{prefix}_hyperparameter2': trial.suggest_categorical(f'{prefix}_hyperparameter2', ['a', 'b', 'c'])
            }

    pipeline = DFPipeline([
        ('step1', TestStep1(random_state=1)),
        ('step2', TestStep2(random_state=2)),
    ])

    trial, suggested = pipeline.suggest_optuna(Mock())

    assert list(suggested.keys()) == ['step1', 'step2']
    assert list(suggested['step1'].keys()) == ['step1_hyperparameter1']
    assert list(suggested['step2'].keys()) == ['step2_hyperparameter2']


def test_Optimize_Pipeline():

    class TestStep1(Model):

        @staticmethod
        def suggest_optuna(trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
            return trial, {
                f'{prefix}_hyperparameter1': trial.suggest_categorical(f'{prefix}_hyperparameter', ['a', 'b', 'c'])
            }

    class TestStep2(Model):

        @staticmethod
        def suggest_optuna(trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
            return trial, {
                f'{prefix}_hyperparameter2': trial.suggest_categorical(f'{prefix}_hyperparameter2', ['a', 'b', 'c'])
            }

    pipeline = DFPipeline([
        ('step1', TestStep1(random_state=1)),
        ('step2', TestStep2(random_state=2)),
    ])

    trial, suggested = pipeline.suggest_optuna(Mock())

    assert list(suggested.keys()) == ['step1', 'step2']
    assert list(suggested['step1'].keys()) == ['step1_hyperparameter1']
    assert list(suggested['step2'].keys()) == ['step2_hyperparameter2']


def test_apply_to_model():
    results = [
        {
            's1': {
                'model': {
                    'internal': [1, 2, 3]
                }
            },
            's2': {
                'model': {
                    'internal': [4, 5, 6]
                }
            }
        }, {
            's1': {
                'model': {
                    'internal': [7, 8, 9]
                }
            },
        }
    ]
    # noinspection PyTypeChecker
    assert aggregate_results(
        results,
        lambda model: model['internal'],
    ) == {
        '0_s1': [1, 2, 3],
        '0_s2': [4, 5, 6],
        '1_s1': [7, 8, 9],
    }
