from copy import copy, deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from pandas import DataFrame

from hcve_lib.cohort_statistics import get_description_column, get_value_column, get_missing_column, \
    get_events_per_person_years, get_incidence, get_missing, get_non_missing, get_missing_fraction, get_median_follow_up

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

data = DataFrame({
    'a': [10, 10, 10, 20],
    'b': [1., 2., 3., np.NAN],
    'c': [1, 2, np.NAN, np.NAN]
})


@patch(
    'hcve_lib.cohort_statistics.categorize_features',
    return_value=(['a'], ['b']),
)
def test_get_row_names(_):
    assert list(get_description_column(
        metadata,
        data,
    )) == [
        (0, 'category', 'Category'),
        (1, 'item', 'a, g1, X / Y, % (n)'),
        (1, 'item', 'b, g2'),
        (1, 'item', 'c'),
    ]


@patch(
    'hcve_lib.cohort_statistics.categorize_features',
    return_value=(['a', 'c'], ['b']),
)
def test_get_value_column(_):

    assert list(get_value_column(
        metadata,
        data,
    )) == ['', '75 (3) / 25 (1)', '2.0 (1.2-2.8)', '50 (1)']

    metadata_no_mapping = deepcopy(metadata)
    del metadata_no_mapping[0]['children'][0]['mapping']

    with pytest.raises(Exception):
        list(get_value_column(
            metadata_no_mapping,
            data,
        ))


def test_get_missing_column():
    assert list(get_missing_column(metadata, data)) == ['', '', '25.0', '50.0']


def test_get_events_per_person_years():
    assert get_events_per_person_years(
        DataFrame({
            'EVENT': [1, 0, 0, 0],
            'FUEVENT': [365, 365, 365, 365],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
        desired_person_years=1000,
    ) == 250


def test_get_incidence():
    assert get_incidence(
        DataFrame({
            'EVENT': [1, 0, 0, 0],
            'FUEVENT': [365, 365, 365, 365],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
    ) == 0.25


def test_get_na_fu():
    assert get_missing(
        DataFrame({
            'EVENT': [1, 0, 0, np.nan],
            'FUEVENT': [365, 365, 365, 365],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
    ) == 1


def test_get_non_na_fu():
    assert get_non_missing(
        DataFrame({
            'EVENT': [1, 0, 0, np.nan],
            'FUEVENT': [365, 365, 365, 365],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
    ) == 3


def test_get_missing_fraction():
    assert get_missing_fraction(
        DataFrame({
            'EVENT': [1, 0, 0, np.nan],
            'FUEVENT': [365, 365, 365, 365],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
    ) == 0.25


def test_get_median_follow_up():

    assert get_median_follow_up(
        DataFrame({
            'EVENT': [1, 0, 0, 1, np.nan],
            'FUEVENT': [1, 10, 15, 100, 100000],
        }),
        feature={
            'identifier': 'EVENT',
            'identifier_tte': 'FUEVENT'
        },
    ) == 15
