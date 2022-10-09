from numpy import array, inf
from numpy.ma.testutils import assert_array_equal
from pandas import DataFrame

from hcve_lib.survival import survival_to_interval, get_event_probability, get_event_case_ratio


def test_survival_to_intervals():
    lower_bound, upper_bound = survival_to_interval(
        {'data': DataFrame({
            'tte': [1, 10, 20],
            'label': [0, 1, 0]
        })})

    assert_array_equal(
        lower_bound,
        array([1., 10., 20.]),
    )

    assert_array_equal(
        lower_bound,
        array([+inf, 10., +inf]),
    )


def test_get_event_probability():
    assert get_event_probability(DataFrame({'label': [0, 1, 0, 0]})) == 0.25

def test_get_event_case_ratio():
    assert get_event_case_ratio(DataFrame({'label': [0, 1, 0]})) == 0.5
