from pandas import DataFrame
from pandas.testing import assert_frame_equal

from hcve_lib.functional import map_columns, rejectNone


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


def test_rejectNone():
    assert list(rejectNone([1, 2, None, 4])) == [1, 2, 4]
