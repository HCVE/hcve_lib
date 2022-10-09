import numpy as np
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from hcve_lib.data import get_identifiers, sanitize_data_inplace, get_survival_y, \
    binarize_event, get_X, MetadataItemType, remove_nan_target, is_target, format_feature_value, \
    format_features_and_values, get_variables, get_available_identifiers_per_category, inverse_format_feature_value, \
    get_targets, get_age_range


def test_sanitize_data_inplace():
    df = DataFrame({'abc': [1, 2, 3], 'VISIT': ['a', 'b', 'c']})
    sanitize_data_inplace(df)
    assert_frame_equal(
        df,
        DataFrame({
            'ABC': [1, 2, 3],
            'VISIT': ['A', 'B', 'C'],
        }),
    )


def test_get_features_from_metadata():
    metadata = [
        {
            'identifier': 'Administrative',
            'children': [
                {
                    'identifier': 'AGE', 'meaning': 'Age', 'unit': 'Years'
                }, {
                    'identifier': 'BW', 'meaning': 'Body weight', 'unit': 'Kg'
                }
            ]
        },
        {
            'identifier': 'Medical history',
            'children': [
                {
                    'identifier': 'HTA', 'meaning': 'Hypertension', 'mapping': {
                        0: False, 1: True
                    }
                },
                {
                    'identifier': 'DIABETES', 'meaning': 'History of diabetes', 'mapping': {
                        0: False, 1: True
                    }
                },
            ]
        }
    ]

    features = list(get_variables(metadata))

    assert features == [
        {
            'identifier': 'AGE', 'meaning': 'Age', 'unit': 'Years'
        },
        {
            'identifier': 'BW', 'meaning': 'Body weight', 'unit': 'Kg'
        },
        {
            'identifier': 'HTA', 'meaning': 'Hypertension', 'mapping': {
                0: False, 1: True
            }
        },
        {
            'identifier': 'DIABETES', 'meaning': 'History of diabetes', 'mapping': {
                0: False, 1: True
            }
        },
    ]


def test_get_feature_identifiers():
    assert list(
        get_identifiers(
            [
                {
                    'identifier': 'AGE', 'meaning': 'Age', 'unit': 'Years'
                },
                {
                    'identifier': 'BW', 'meaning': 'Body weight', 'unit': 'Kg'
                },
            ]
        )
    ) == ['AGE', 'BW']


def test_get_survival_y():
    result = get_survival_y(
        DataFrame({
            'a': [0, 1, 0],
            'a_tte': [1, 2, 3],
            'x': [4, 5, 6],
        }),
        target_feature='a',
        metadata=[{
            'identifier': 'a', 'identifier_tte': 'a_tte'
        }],
    )

    assert result['name'] == 'a'

    assert_frame_equal(
        result['data'],
        DataFrame({
            'label': [0, 1, 0],
            'tte': [1, 2, 3],
        }),
    )


def test_binarize_survival():
    assert_series_equal(
        binarize_event(
            tte=100,
            survival_y=DataFrame(
                {
                    'label': [0, 1, 1, 1, 0],
                    'tte': [50, 60, 100, 110, 200],
                },
                index=[0, 1, 2, 3, 4],
            ),
        ),
        Series(
            [np.nan, 1, 1, 0, 0],
            index=[0, 1, 2, 3, 4],
        ),
    )


def test_get_X():

    assert_frame_equal(
        get_X(
            DataFrame({
                'a': [0, 1, 0],
                'x': [4, 5, 6],
                'y': [1, 2, 3],
            }),
            metadata=[
                {
                    'identifier': 'a', 'type': MetadataItemType.BINARY_TARGET.value
                },
                {
                    'identifier': 'x',
                },
            ],
        ),
        DataFrame({
            'x': [4, 5, 6],
        }),
    )


def test_remove_nan_target():
    X, y = remove_nan_target(
        DataFrame(
            {
                'a': [0, 1, 2],
            },
            index=[0, 1, 2],
        ),
        Series(
            [10, np.nan, 20],
            index=[0, 1, 2],
        ),
    )

    assert_frame_equal(
        X,
        DataFrame(
            {
                'a': [0, 2],
            },
            index=[0, 2],
        ),
    )

    assert_series_equal(
        y,
        Series(
            [10., 20.],
            index=[0, 2],
        ),
    )


def test_is_target():
    assert is_target({
        'type': 'survival_target',
    }) is True

    assert not is_target({
        'type': 'not',
    }) is True

    assert not is_target({}) is True


def test_format_value():
    assert format_feature_value('a', {'mapping': {'a': 'b'}}) == 'b'
    assert format_feature_value('c', {'mapping': {'a': 'b'}}) == 'c'
    assert format_feature_value('d', {}) == 'd'


def test_inverse_format_value():
    assert inverse_format_feature_value('b', {'mapping': {'a': 'b'}}) == 'a'
    assert inverse_format_feature_value('c', {'mapping': {'a': 'b'}}) == 'c'
    assert inverse_format_feature_value('d', {}) == 'd'


def test_format_dataframe():
    assert_frame_equal(
        format_features_and_values(
            DataFrame(
                {
                    'a': [0, 1],
                    'b': [2, 3],
                    'c': [2, 3],
                },
                index=[0, 1],
            ),
            metadata=[
                {
                    'identifier': 'a', 'meaning': 'Ax'
                },
                {
                    'identifier': 'b', 'mapping': {
                        2: "x"
                    }
                },
            ],
        ),
        DataFrame(
            {
                'Ax': [0, 1],
                'b': ['x', 3],
                'c': [2, 3],
            },
            index=[0, 1],
        ),
    ),


def test_get_available_identifiers_per_category():
    identifiers = list(
        get_available_identifiers_per_category(
            [
                {
                    'identifier': '1', 'children': [{
                        'identifier': 'a'
                    }]
                },
                {
                    'identifier': '2', 'children': [
                        {
                            'identifier': 'b'
                        },
                        {
                            'identifier': 'c'
                        },
                    ]
                },
            ],
            DataFrame({'a': [1, 2]}),
        )
    )

    assert len(identifiers) == 1
    assert identifiers[0][0]['identifier'] == '1'
    assert identifiers[0][1] == ['a']


def test_get_targets():
    target_features = [
        {
            'identifier': 'x1', 'meaning': 'X1', 'type': MetadataItemType.SURVIVAL_TARGET.value
        },
        {
            'identifier': 'x2', 'meaning': 'X2', 'type': MetadataItemType.BINARY_TARGET.value
        },
    ]

    other_features = [{'identifier': 'x3', 'meaning': 'X3', 'type': 'something_else'}]

    metadata = [{
        'identifier': 'category', 'children': [
            *target_features,
            *other_features,
        ]
    }]

    assert list(get_targets(metadata)) == target_features


def test_get_age_range():
    assert_frame_equal(
        get_age_range(
            DataFrame({'AGE': [10, 20, 30, 40]}, index=[1, 2, 3, 4]),
            [20, 40],
        ),
        DataFrame(
            {'AGE': [20, 30, 40]},
            index=[2, 3, 4],
        ),
    )
