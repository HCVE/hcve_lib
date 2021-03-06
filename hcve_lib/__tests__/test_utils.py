from typing import List, Dict

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pandas import Series, DataFrame, Int64Index

from hcve_lib.utils import get_class_ratios, decamelize_arguments, camelize_return, map_column_names, cumulative_count, \
    inverse_cumulative_count, key_value_swap, index_data, list_to_dict_by_keys, subtract_lists, map_groups_iloc, \
    remove_prefix, remove_column_prefix, get_fraction_missing, transpose_dict, map_groups_loc, loc, split_data, \
    get_fraction_missing, map_recursive, get_keys
from pandas.testing import assert_frame_equal, assert_series_equal


def test_get_class_ratio():
    assert get_class_ratios(Series([1, 1, 1, 1, 0, 0])) == {0: 2., 1: 1.}
    assert get_class_ratios(Series([1, 1, 1, 2, 0, 0])) == {
        0: 1.5,
        1: 1.0,
        2: 3.0
    }


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
        index_data([1, 2],
                   np.array(
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
        remove_column_prefix(
            DataFrame({
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


def test_map_groups_loc():
    results = list(
        map_groups_loc(
            DataFrame(
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
            'data':
            DataFrame(
                {
                    'tte': [0, 10, 200, 30, 40],
                    'label': [0, 0, 0, 1, 1]
                },
                index=[10, 20, 30, 40, 50],
            ),
        },
        fold={
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
    assert_frame_equal(
        y_train['data'],
        DataFrame(
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
            'data':
            DataFrame(
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
    assert_frame_equal(
        y_train['data'],
        DataFrame(
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
    assert_frame_equal(
        y_test['data'],
        DataFrame(
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
        lambda num: num + 1,
    ) == {
        'a': {
            'b': [3, 4],
            'c': 5,
        }
    }


def test_get_keys():
    assert get_keys(['x'], {'x': 1, 'y': 2}) == {'x': 1}
