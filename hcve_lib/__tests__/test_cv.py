import numpy as np
import pandas
from _pytest.python_api import raises
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from unittest.mock import Mock

from hcve_lib.cv import cross_validate, lco_cv, train_test, get_column_mask_filter, filter_missing_features, \
    predict_proba, cross_validate_apply_mask, get_column_mask, get_removed_features_from_mask


def test_cross_validate():
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
    )

    for X_fold, X_test_fold, y_fold in zip(Xs, Xs_test, ys):
        assert (Series(X_fold.index == y_fold.index).all())
        assert len(X_test_fold.index.intersection(X_fold.index)) == 0

    y_trues = pandas.concat([fold['y_true'] for fold in result])
    assert_series_equal(y_trues, y_all)

    y_scores = pandas.concat([fold['y_score'] for fold in result])
    assert_series_equal(y_scores, Series([0, 2, 6]))


def test_lco_cv():
    assert lco_cv(
        DataFrame(
            {
                'a': [1, 1, 1, 2, 2, 3]
            },
            index=[10, 1, 2, 3, 4, 5],
        ).groupby('a')) == {
            1: ([3, 4, 5], [0, 1, 2]),
            2: ([0, 1, 2, 5], [3, 4]),
            3: ([0, 1, 2, 3, 4], [5]),
        }


def test_train_test():
    assert train_test(
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 1, 2, 3, 4, 5],
        ),
        train_filter=lambda _data: _data['a'] == 1,
    ) == {
        'train_test': ([0, 1, 2], [3, 4, 5])
    }

    assert train_test(
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 1, 2, 3, 4, 5],
        ),
        train_filter=lambda _data: _data['a'] == 1,
        test_filter=lambda _data: _data['a'] == 3,
    ) == {
        'train_test': ([0, 1, 2], [5])
    }


def test_cross_validate_apply_filter():
    assert list(
        get_column_mask_filter(
            DataFrame({'x': [1, 2, 3, 4]}),
            [
                ([0, 1], [2, 3]),
                ([2, 3], [0, 1]),
            ],
            (lambda X_train, X_test:
             (X_train.tolist() == [3, 4]) and (X_test.tolist() == [1, 2])),
        )) == [{
            'x': False
        }, {
            'x': True
        }]


def test_filter_missing_features():
    assert filter_missing_features(
        Series([0, 1, np.nan, 2]),
        Series([0, 1, 2, 3]),
        threshold=0.25,
    ) is True

    assert filter_missing_features(
        Series([0, 1, 3, 4]),
        Series([0, 1, 2, np.nan]),
        threshold=0.25,
    ) is True

    assert filter_missing_features(
        Series([0, 1, 3, 4]),
        Series([0, 1, 2, np.nan]),
        threshold=0.30,
    ) is False


def test_cross_validate_apply_mask():
    assert_frame_equal(
        cross_validate_apply_mask(
            {
                'a': True,
                'b': False
            },
            DataFrame({
                'a': [1],
                'b': [2],
            }),
        ),
        DataFrame({
            'b': [2],
        }),
    )

    with raises(Exception):
        cross_validate_apply_mask(
            {
                'a': True,
            },
            DataFrame({
                'a': [1],
                'b': [2],
            }),
        )


def test_get_column_mask():
    # Default
    assert get_column_mask(
        X=DataFrame({'x': [10, 20, 30, 40]}),
        splits=[([0, 1, 2], [3]), ([0, 1, 3], [2])],
    ) == [
        {
            'x': False
        },
        {
            'x': False
        },
    ]

    # Filter provided
    assert get_column_mask(
        X=DataFrame({'x': [10, 20, 30, 40]}),
        splits=[([0, 1, 2], [3]), ([0, 1, 3], [2])],
        train_test_filter=Mock(side_effect=[False, True]),
    ) == [
        {
            'x': False
        },
        {
            'x': True
        },
    ]


def test_get_removed_features_from_mask():
    assert get_removed_features_from_mask({
        'x': {
            'a': True,
            'b': False
        },
        'y': {
            'a': False,
            'b': True
        },
    }) == {
        'x': ['a'],
        'y': ['b']
    }
