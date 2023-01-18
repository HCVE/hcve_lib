import sys
from typing import List, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest

from hcve_lib.custom_types import Result, Prediction
from hcve_lib.functional import itemmap_recursive, map_recursive
from hcve_lib.utils import get_class_ratios, decamelize_arguments, camelize_return, map_column_names, \
    cumulative_count, inverse_cumulative_count, key_value_swap, index_data, list_to_dict_by_keys, subtract_lists, \
    map_groups_iloc, remove_prefix, remove_column_prefix, transpose_dict, map_groups_loc, loc, split_data, \
    get_fraction_missing, get_keys, sort_columns_by_order, is_noneish, sort_index_by_order, SurvivalResample, \
    transpose_list, binarize, get_fractions, run_parallel, is_numeric, get_models_from_repeats, get_jobs
from imblearn.under_sampling import RandomUnderSampler
from numpy.testing import assert_array_equal
from pandas import Series, DataFrame, Int64Index
from pandas.testing import assert_frame_equal, assert_series_equal


def test_get_class_ratio():
    assert get_class_ratios(Series([1, 1, 1, 1, 0, 0])) == {0: 2., 1: 1.}
    assert get_class_ratios(Series([1, 1, 1, 2, 0, 0])) == {0: 1.5, 1: 1.0, 2: 3.0}


def test_decamelize_arguments():

    @decamelize_arguments
    def test_function(arg1: Dict, arg2: List):
        return arg1, arg2

    assert test_function(
        {'oneVariable': 1},
        [{
            'secondVariable': 2
        }],
    ) == (
        {
            'one_variable': 1
        },
        [{
            'second_variable': 2
        }],
    )


def test_camelize_return():

    @camelize_return
    def test_function(arg1: Dict, arg2: List):
        return arg1, arg2

    assert test_function(
        {'one_variable': 1},
        [{
            'second_variable': 2
        }],
    ) == (
        {
            'oneVariable': 1
        },
        [{
            'secondVariable': 2
        }],
    )


def test_map_columns():
    assert_frame_equal(
        map_column_names(
            DataFrame({
                'a': [1],
                'b': [2]
            }),
            lambda k: k + 'x',
        ),
        DataFrame({
            'ax': [1],
            'bx': [2]
        }),
    )


def test_cumulative_count():
    assert list(cumulative_count(Series([
        0,
        3,
        5,
        8,
    ]))) == [
        (0, 0.25),
        (3, 0.5),
        (5, 0.75),
        (8, 1.0),
    ]

    assert list(cumulative_count(Series([
        np.nan,
        3,
        5,
        8,
    ]))) == [
        (3, 0.25),
        (5, 0.5),
        (8, 0.75),
    ]


def test_inverse_cumulative_count():
    assert list(inverse_cumulative_count(Series([
        0,
        3,
        5,
        8,
    ]))) == [
        (0, 1.),
        (3, 0.75),
        (5, 0.5),
        (8, 0.25),
    ]


def test_key_value_swap():
    assert key_value_swap({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}


def test_index_data():
    assert_frame_equal(
        index_data([1, 2], DataFrame(
            {'a': [5, 6, 7]},
            index=[1, 2, 3],
        )),
        DataFrame(
            {'a': [6, 7]},
            index=[2, 3],
        ),
    )

    assert_series_equal(
        index_data([1, 2], Series(
            [5, 6, 7],
            index=[1, 2, 3],
        )),
        Series(
            [6, 7],
            index=[2, 3],
        ),
    )

    assert_array_equal(
        index_data([1, 2], np.array(
            [(1.0, 2), (2.0, 3), (3.0, 4)],
            dtype=[('x', '<f8'), ('y', '<i8')],
        )),
        np.array(
            [(2.0, 3), (3.0, 4)],
            dtype=[('x', '<f8'), ('y', '<i8')],
        ),
    )


def test_list_to_dict():
    assert list_to_dict_by_keys([1, 2], ['a', 'b']) == {'a': 1, 'b': 2}


def test_subtract_lists():
    assert subtract_lists([1, 2, 3], [2]) == [1, 3]


def test_map_groups_iloc():
    data = DataFrame(
        {'a': [1, 1, 1, 2, 2, 3]},
        index=[10, 1, 2, 3, 4, 5],
    )
    assert list(map_groups_iloc(data.groupby('a'), data)) == [
        (1, [0, 1, 2]),
        (2, [3, 4]),
        (3, [5]),
    ]


def test_remove_prefix():
    assert remove_prefix('prefix_', 'prefix_x_prefix') == 'x_prefix'
    assert remove_prefix('prefix_', 'refix_x_prefix') == 'refix_x_prefix'


def test_remove_column_prefix():
    assert_frame_equal(
        remove_column_prefix(DataFrame({
            'a': [1],
            'categorical__b': ['a'],
            'continuous__c': [3]
        })),
        DataFrame({
            'a': [1],
            'b': ['a'],
            'c': [3]
        }),
    )


def test_percent_missing():
    assert get_fraction_missing(Series([np.nan, np.nan])) == 1
    assert get_fraction_missing(Series([np.nan, 5])) == 0.5
    assert get_fraction_missing(Series([1, 5])) == 0


def test_transpose_dict():
    assert transpose_dict({
        0: {
            'a': 'x',
            'b': 1
        },
        1: {
            'a': 'y',
            'b': 2
        },
        2: {
            'a': 'z',
            'b': 3
        },
    }) == {
        'a': {
            0: 'x',
            1: 'y',
            2: 'z'
        },
        'b': {
            0: 1,
            1: 2,
            2: 3
        },
    }


def test_transpose_list():
    assert transpose_list([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_map_groups_loc():
    results = list(map_groups_loc(DataFrame(
        {
            'a': [0, 0, 1, 1, 1]
        },
        index=[10, 20, 30, 40, 50],
    ).groupby('a')))

    assert results[0][0] == 0
    assert_array_equal(results[0][1], Int64Index([10, 20], dtype='int64'))

    assert results[1][0] == 1
    assert_array_equal(results[1][1], Int64Index([30, 40, 50], dtype='int64'))


def test_loc():
    assert_frame_equal(
        loc(
            [10, 30],
            DataFrame(
                {'a': [0, 0, 1, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        ),
        DataFrame(
            {'a': [0, 1]},
            index=[10, 30],
        ),
    )

    with pytest.raises(Exception):
        loc(
            [10, 30, -100],
            DataFrame(
                {'a': [0, 0, 1, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        )

    loc(
        [10, 30, -100],
        DataFrame(
            {'a': [0, 0, 1, 1, 1]},
            index=[10, 20, 30, 40, 50],
        ),
        ignore_not_present=True,
    )


def test_split_data():
    X_train, y_train, X_test, y_test = split_data(
        X=DataFrame(
            {
                'a': [0, 1, 2, 3, 4],
                'b': [0, 1, 2, 3, 4],
            },
            index=[10, 20, 30, 40, 50],
        ),
        y={
            'data': DataFrame(
                {
                    'tte': [0, 10, 200, 30, 40],
                    'label': [0, 0, 0, 1, 1]
                },
                index=[10, 20, 30, 40, 50],
            ),
        },
        prediction={
            'split': ([10, 50], [20, 30]),
            # Skipping index 20 in y_score
            'y_score': Series([1, 3, 4, 5], index=[10, 30, 40, 50]),
            'X_columns': ['a']
        },
    )

    assert_frame_equal(
        X_train,
        DataFrame(
            {
                'a': [0, 4],
            },
            index=[10, 50],
        ),
    )
    assert_frame_equal(y_train['data'], DataFrame(
        {
            'tte': [0, 40],
            'label': [0, 1]
        },
        index=[10, 50],
    ))

    assert_frame_equal(
        X_test,
        DataFrame(
            {
                'a': [2],
            },
            index=[30],
        ),
    )
    assert_frame_equal(
        y_test['data'],
        DataFrame(
            {
                'tte': [200],
                'label': [0]
            },
            index=[30],
        ),
    )


def test_split_data_remove_extended():
    X_train, y_train, X_test, y_test = split_data(
        DataFrame(
            {
                'a': [0, 1, 2, 3, 4],
                'b': [0, 1, 2, 3, 4],
            },
            index=[10, 20, 30, 40, 50],
        ),
        {
            'data': DataFrame(
                {
                    'tte': [0, 10, 200, 30, 40],
                    'label': [0, 0, 0, 1, 1]
                },
                index=[10, 20, 30, 40, 50],
            ),
        },
        {
            'split': ([10, 50], [20, 30, 40]),
            'X_columns': ['a']
        },
        remove_extended=True,
    )
    assert_frame_equal(
        X_train,
        DataFrame(
            {
                'a': [0, 4],
            },
            index=[10, 50],
        ),
    )
    assert_frame_equal(y_train['data'], DataFrame(
        {
            'tte': [0, 40],
            'label': [0, 1]
        },
        index=[10, 50],
    ))

    assert_frame_equal(
        X_test,
        DataFrame(
            {
                'a': [1, 3],
            },
            index=[20, 40],
        ),
    )
    assert_frame_equal(y_test['data'], DataFrame(
        {
            'tte': [10, 30],
            'label': [0, 1]
        },
        index=[20, 40],
    ))


def test_fraction_missing():
    assert get_fraction_missing(Series([0, 1, 2, np.nan])) == 0.25


def test_map_recursive():
    assert map_recursive(
        {
            'a': {
                'b': [2, 3],
                'c': 4,
            },
        },
        lambda num, _: num + 1 if isinstance(num, int) else num,
    ) == {
        'a': {
            'b': [3, 4],
            'c': 5,
        }
    }


def test_get_keys():
    assert get_keys(['x'], {'x': 1, 'y': 2}) == {'x': 1}


def test_itemmap_recursive():
    with pytest.raises(TypeError):
        itemmap_recursive('x', lambda x: x + 'b')

    assert itemmap_recursive(
        {
            'x': 1,
            'y': 2
        },
        lambda k, v, l: (k + 'b', v + 1),
    ) == {
        'xb': 2,
        'yb': 3
    }

    assert itemmap_recursive(
        (1, 2, 3),
        lambda k, v, l: (None, v + 1),  # Key ignored
    ) == (2, 3, 4)

    assert itemmap_recursive(
        [1, 2, 3],
        lambda k, v, l: (None, v + 1),  # Key ignored
    ) == [2, 3, 4]

    assert itemmap_recursive(
        {
            'x': {
                'y': 1
            },
        },
        lambda k, v, l: (k + 'b', v + 1 if isinstance(v, int) else v),
    ) == {
        'xb': {
            'yb': 2
        },
    }

    assert itemmap_recursive(
        {'x': [1, 2, 3]},
        lambda k, v, l: (str(k) + 'b', v + l if isinstance(v, int) else v),
    ) == {
        'xb': [2, 3, 4],
    }


def test_sort_columns_by_order():
    assert_frame_equal(
        sort_columns_by_order(
            DataFrame({
                'a': [1],
                'b': [2],
                'c': [3]
            }),
            ['x', 'a', 'c'],
        ),
        DataFrame({
            'x': [np.nan],
            'a': [1],
            'c': [3],
        }),
    )


def test_sort_index_by_order():
    assert_frame_equal(
        sort_index_by_order(
            DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z']),
            ['w', 'z', 'y'],
        ),
        DataFrame({'a': [np.nan, 3, 2]}, index=['w', 'z', 'y']),
    )


def test_is_noneish():
    assert is_noneish(None) is True
    assert is_noneish(np.nan) is True
    assert is_noneish(False) is False
    assert is_noneish(0) is False
    assert is_noneish('0') is False
    assert is_noneish(5) is False


def subprocess(something):
    print('3')


def test_capture_output():
    # test_std = StringIO()
    # sys.stdout = test_std
    # sys.stderr = test_std

    from hcve_lib.log_output import capture_output

    with capture_output() as get_output:
        print('1')
        print('2', file=sys.stderr)
        run_parallel(subprocess, {0: ['a']}, n_jobs=2)

    assert get_output() == '1\n2\n3\n'


def test_SurvivalResample():
    X = DataFrame({'x': [1, 2, 3]}, index=[10, 20, 30])
    y = {
        'data': DataFrame(
            {
                'tte': [10, 10, 20],
                'label': [1, 1, 0],
            },
            index=[10, 20, 30],
        )
    }
    Xr, yr = SurvivalResample(RandomUnderSampler()).fit_resample(X, y)
    assert yr['data']['label'].value_counts()[1] == yr['data']['label'].value_counts()[0]


def test_binarize():
    assert_series_equal(
        binarize(Series([0.1, 0.2, 0.3]), threshold=0.2),
        Series([0, 1, 1]),
    )


def test_get_fractions():
    assert_series_equal(
        get_fractions(Series(['a', 'a', 'b', 'c'])),
        Series([0.5, 0.25, 0.25], index=['a', 'b', 'c']),
    )


def test_is_numeric():
    assert is_numeric('5') is True
    assert is_numeric('-5') is True
    assert is_numeric('5.5') is True
    assert is_numeric(100) is True
    assert is_numeric('a5') is False


# TODO
def test_get_categorical_columns():
    categories = get_categorical_columns(DataFrame({'x': [1], 'y': ['cat']}, dtypes={'x': int, 'y': 'category'}))


def test_get_models_from_repeats():
    model1 = Mock()
    model1.feature_importances_ = [0.1, 0.5]

    forests = get_models_from_repeats([{'split1': {'model': model1}}])
    print(forests)


@patch('multiprocessing.cpu_count', Mock(return_value=10))
def test_get_jobs():
    granted, residual = get_jobs(4, 6)
    assert granted == 4
    assert residual == 4

    granted, residual = get_jobs(7, 8)
    assert granted == 7
    assert residual == 3

    granted, residual = get_jobs(-1, 8)
    assert granted == 8
    assert residual == 2

    granted, residual = get_jobs(12, 8)
    assert granted == 8
    assert residual == 2

    granted, residual = get_jobs(-1)
    assert granted == 10
    assert residual == 1

    granted, residual = get_jobs(1, 8)
    assert granted == 1
    assert residual == 1
