import numpy as np
import pandas
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal, assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from unittest.mock import Mock, call

from hcve_lib.cv import cross_validate, optimize_per_split, predict_proba, series_to_target
from hcve_lib.splitting import filter_missing_features


def _test_cross_validate():
    Xs = []
    ys = []
    Xs_test = []

    class DummyEstimator(BaseEstimator):
        def __init__(self, param):
            self.param = param

        @staticmethod
        def fit(X, y):
            Xs.append(X)
            ys.append(y)

        def predict_proba(self, X):
            Xs_test.append(X)
            return X.apply(lambda row: row['x'] * self.param, axis='columns')

    get_pipeline = Mock(side_effect=[
        DummyEstimator(2),
        DummyEstimator(2),
        DummyEstimator(3),
    ])

    X_all = DataFrame({'x': list(range(3))})
    y_all = Series(list(range(3)))

    result = cross_validate(
        X_all,
        y_all,
        get_pipeline,
        predict=predict_proba,
        splits=KFold(n_splits=3).split(X_all),
        n_jobs=1,
        train_test_filter_callback=filter_missing_features,
    )

    for X_fold, X_test_fold, y_fold in zip(Xs, Xs_test, ys):
        assert (Series(X_fold.index == y_fold.index).all())
        assert len(X_test_fold.index.intersection(X_fold.index)) == 0

    y_trues = pandas.concat([fold['y_true'] for fold in result])
    assert_series_equal(y_trues, y_all)

    y_scores = pandas.concat([fold['y_score'] for fold in result])
    assert_series_equal(y_scores, Series([0, 2, 6]))


def test_optimize_cv():
    optimize_input = [Mock(), Mock()]
    get_optimize = Mock(side_effect=optimize_input)
    get_splits = Mock(side_effect=lambda *args: {
        'a': ([0, 1, 2], [3]),
        'b': ([0, 1, 3], [2]),
    })
    optimize_per_split(
        get_optimize,
        get_splits,
        DataFrame({'a': [1, 2, 3]}),
        Series([10, 20, 30]),
        n_jobs=1,
    )
    get_optimize.assert_called()

    for optimize in optimize_input:
        optimize.fit.assert_called_once()


def test_predict_proba():
    class MockMethod:
        predict_proba = Mock(side_effect=[np.array([[0.1, 0.9]])])

    model = MockMethod()
    prediction = predict_proba(
        DataFrame({'x': [1, 2]}, index=[10, 20]),
        {
            'name': 'a',
            'data': Series([1, 2], index=[10, 20]),
        },
        ([10], [20]),
        model,
    )

    assert prediction['X_columns'] == ['x']
    assert prediction['model'] == model
    assert prediction['split'] == ([10], [20])
    assert prediction['y_column'] == 'a'
    assert_frame_equal(
        prediction['y_score'],
        DataFrame(
            {
                0: [0.1],
                1: [0.9]
            },
            index=[20],
        ),
    )
    assert len(model.predict_proba.call_args_list) == 1
    assert_frame_equal(
        model.predict_proba.call_args_list[0][0][0],
        DataFrame({'x': 2}, index=[20]),
    )


def test_series_to_target():
    series = Series([1, 2, 3], name='a')
    target = series_to_target(series)
    assert target['name'] == 'a'
    assert_series_equal(target['data'], series)
