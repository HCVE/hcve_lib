import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from hcve_lib.functional import map_columns, reject_none, always, reject_none_values, accept_extra_parameters, lagged, \
    subtract


def test_map_columns():
    assert_frame_equal(
        map_columns(
            lambda name, value: value + name,
            DataFrame({
                'a': ['1', '2'],
                'b': ['2', '3'],
            }),
        ),
        DataFrame({
            'a': ['1a', '2a'],
            'b': ['2b', '3b'],
        }),
    )


def test_reject_none():
    assert list(reject_none([1, 2, None, 4])) == [1, 2, 4]


def test_reject_none_values():
    assert reject_none_values({'a': 5, 'b': None}) == {'a': 5}


def test_always():
    func = always('a')
    assert func('x') == 'a'
    assert func() == 'a'


def test_accept_extra_parameters():
    @accept_extra_parameters
    def test_func(arg1, arg2=1):
        return arg1 * arg2

    assert test_func(5) == 5
    assert test_func(5, 2) == 10
    assert test_func(5, arg2=2) == 10
    assert test_func(5, 6, arg2=2, arg5=2) == 10

    with pytest.raises(TypeError):
        test_func(arg2=5)


def test_lagged():
    assert list(lagged([1, 2, 3])) == [(1, 2), (2, 3)]


def test_subtract():
    assert list(subtract(iter([1, 2, 3]), iter([2]))) == [1, 3]

