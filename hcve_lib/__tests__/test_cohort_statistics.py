from copy import copy, deepcopy
from unittest.mock import patch

import pytest
from _pytest.python_api import raises
from pandas import DataFrame

from hcve_lib.cohort_statistics import get_description_column, get_value_columns


@patch(
    'hcve_lib.cohort_statistics.categorize_features',
    return_value=(['a'], ['b']),
)
def test_get_row_names(_):
    metadata = [{
        'identifier':
        'Category',
        'children': [{
            'identifier': 'a',
            'unit': 'g1',
            'mapping': {
                10: 'X',
                20: 'Y'
            }
        }, {
            'identifier': 'b',
            'unit': 'g2',
        }]
    }]

    data = DataFrame({'a': [10, 20], 'b': [1., 2.]})

    assert list(get_description_column(
        metadata,
        data,
    )) == [
        (0, 'category', 'Category'),
        (1, 'item', 'a, g1, X / Y, % (n)'),
        (1, 'item', 'b, g2'),
    ]


@patch(
    'hcve_lib.cohort_statistics.categorize_features',
    return_value=(['a', 'c'], ['b']),
)
def test_get_row_values(_):
    metadata = [{
        'identifier':
        'Category',
        'children': [
            {
                'identifier': 'a',
                'unit': 'g1',
                'mapping': {
                    10: 'X',
                    20: 'Y'
                }
            },
            {
                'identifier': 'b',
                'unit': 'g2',
            },
            {
                'identifier': 'c',
                'mapping': {
                    1: 'Yes',
                    2: 'No'
                }
            },
        ]
    }]

    data = DataFrame({'a': [10, 20], 'b': [1., 2.], 'c': [1, 2]})

    assert list(get_value_columns(
        metadata,
        data,
    )) == ['', '50 (1) / 50 (1)', '1.5 (1.1-1.9)', '50 (1)']

    metadata_no_mapping = deepcopy(metadata)
    del metadata_no_mapping[0]['children'][0]['mapping']

    with pytest.raises(Exception):
        assert list(get_value_columns(
            metadata_no_mapping,
            data,
        )) == ['50 (1) / 50 (1)', '1.5 (1.1-1.9)', '50 (1)']
