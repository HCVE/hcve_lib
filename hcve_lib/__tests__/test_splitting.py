from unittest.mock import Mock

import numpy as np
from _pytest.python_api import raises
from numpy.testing import assert_array_equal
from pandas import DataFrame, Series, Index
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from hcve_lib.cv import get_column_mask_filter, get_column_mask, get_removed_features_from_mask
from hcve_lib.splitting import get_lo_splits, iloc_to_loc, get_1_to_1_splits, train_test_filter, \
    filter_missing_features, get_kfold_splits, get_splitting_per_group, get_group_indexes, \
    resample_prediction_test
from hcve_lib.utils import cross_validate_apply_mask


def test_get_lo_splits():
    assert get_lo_splits(
        DataFrame(
            {'a': [1, 1, 2, 2, 3]},
            index=[10, 1, 3, 4, 5],
        ),
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 1, 2, 3, 4, 5],
        ),
        'a',
    ) == {
        1: ([3, 4, 5], [10, 1]),
        2: ([10, 1, 5], [3, 4]),
        3: ([10, 1, 3, 4], [5]),
    }


def test_iloc_to_loc():
    assert iloc_to_loc(
        DataFrame({'x': [100, 200, 300]}, index=[10, 20, 30]),
        [1, 2],
    ) == [20, 30]


def test_get_kfold_splits():
    assert get_kfold_splits(
        X=DataFrame({'x': [100, 200, 300]}, index=[10, 20, 30]),
        n_splits=3,
        random_state=1,
    ) == {
        0: [[20, 30], [10]], 1: [[10, 20], [30]], 2: [[10, 30], [20]]
    }


def test_get_1_to_1_splits():

    assert get_1_to_1_splits(
        DataFrame(
            {'a': [1, 1, 2, 2, 3]},
            index=[10, 30, 40, 50, 60],
        ),
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 20, 30, 40, 50, 60],
        ),
        'a',
    ) == {
        (1, 2): ([10, 30], [40, 50]), (1, 3): ([10, 30], [60]), (2, 1): ([40, 50], [10, 30]), (2, 3): ([40, 50], [60]),
        (3, 1): ([60], [10, 30]), (3, 2): ([60], [40, 50])
    }


def test_train_test_filter():
    print(
        train_test_filter(
            DataFrame(
                {'a': [1, 1, 1, 2, 2, 3]},
                index=[10, 1, 2, 3, 4, 5],
            ),
            train_filter=lambda _data: _data['a'] == 1,
        )
    )
    assert train_test_filter(
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 1, 2, 3, 4, 5],
        ),
        train_filter=lambda _data: _data['a'] == 1,
    ) == {
        'train_test_filter': ([10, 1, 2], [3, 4, 5])
    }

    assert train_test_filter(
        DataFrame(
            {'a': [1, 1, 1, 2, 2, 3]},
            index=[10, 1, 2, 3, 4, 5],
        ),
        train_filter=lambda _data: _data['a'] == 1,
        test_filter=lambda _data: _data['a'] == 3,
    ) == {
        'train_test_filter': ([10, 1, 2], [5])
    }


def test_cross_validate_apply_filter():
    assert list(
        get_column_mask_filter(
            DataFrame({'x': [1, 2, 3, 4]}, index=[0, 1, 2, 3]),
            {
                0: ([0, 1], [2, 3]),
                1: ([2, 3], [0, 1]),
            },
            (lambda X_train, X_test: (X_train.tolist() == [3, 4]) and (X_test.tolist() == [1, 2])),
        )
    ) == [(0, {
        'x': False
    }), (1, {
        'x': True
    })]


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
                'a': True, 'b': False
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
        splits={
            0: ([0, 1, 2], [3]), 1: ([0, 1, 3], [2])
        },
    ) == {
        0: {
            'x': False
        },
        1: {
            'x': False
        },
    }

    # Filter provided
    assert get_column_mask(
        X=DataFrame({'x': [10, 20, 30, 40]}),
        splits={
            0: ([0, 1, 2], [3]), 1: ([0, 1, 3], [2])
        },
        train_test_filter_callback=Mock(side_effect=[False, True]),
    ) == {
        0: {
            'x': False
        }, 1: {
            'x': True
        }
    }


def test_get_removed_features_from_mask():
    assert get_removed_features_from_mask({
        'x': {
            'a': True, 'b': False
        },
        'y': {
            'a': False, 'b': True
        },
    }) == {
        'x': ['a'], 'y': ['b']
    }


# TODO
def test_train_test_proportion():
    ...
    # assert train_test_proportion(
    #     DataFrame(
    #         {'x': [10, 20, 30, 40]},
    #         index=['a', 'b', 'c', 'd'],
    #     ),
    #     test_size=1,
    #     shuffle=False,
    # ) == {
    #     'train_test': ([0, 1, 2], [3])
    # }


def test_get_splitting_per_group():

    def get_splits(group):
        return {
            'x': (group.index.tolist()[0:1], group.index.tolist()[1:]),
            'y': (group.index.tolist()[1:], group.index.tolist()[0:1]),
        }

    assert get_splitting_per_group(
        DataFrame(
            {'x': [100, 110, 300, 310, 320]},
            index=[10, 20, 40, 50, 60],
        ),
        DataFrame(
            {'STUDY': [10, 10, 10, 30, 30, 30]},
            index=[10, 20, 30, 40, 50, 60],
        ),
        get_splits=get_splits,
    ) == {
        (10, 'x'): ([10], [20]), (10, 'y'): ([20], [10]), (30, 'x'): ([40], [50, 60]), (30, 'y'): ([50, 60], [40])
    }


def test_get_group_indexes():
    result = get_group_indexes(
        DataFrame(
            {'x': [1, 1, 2]},
            index=['a', 'b', 'c'],
        ),
        'x',
    )
    assert_array_equal(result[1], Index(('a', 'b')))
    assert_array_equal(result[2], Index(('c', )))


def test_resample_prediction_test():
    out = resample_prediction_test(
        [10, 30],
        dict(
            split=[(50, 60), (10, 20, 30)],
            y_score=Series([1, 2, 3], index=[10, 20, 30]),
            y_proba={'a': Series([2, 3, 4], index=[10, 20, 30])},
        ),
    )

    assert out['split'][1] == [10, 30]

    assert_series_equal(
        out['y_score'],
        Series(
            [1, 3],
            index=[10, 30],
        ),
    )
    assert_series_equal(
        out['y_proba']['a'],
        Series([2, 4], index=[10, 30]),
    )
