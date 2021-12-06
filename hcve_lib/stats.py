from math import sqrt
from numbers import Real
from typing import List, Tuple

from numpy import std, mean
from scipy.stats import t


def confidence_interval(
    sample_values: List[Real],
    degrees_of_freedom: int = None,
) -> Tuple[float, Tuple[float, float], float]:
    degrees_of_freedom = degrees_of_freedom if degrees_of_freedom is not None else len(
        sample_values) - 1
    sample_mean: float = mean(sample_values)  # type: ignore
    sample_standard_deviation = std(
        sample_values,  # type: ignore
        ddof=len(sample_values) - degrees_of_freedom,
    )
    standard_error = sample_standard_deviation / sqrt(degrees_of_freedom)

    # noinspection PyBroadException
    try:
        interval = t.interval(
            0.95,
            degrees_of_freedom,
            loc=sample_mean,
            scale=standard_error,
        )
    except Exception:
        interval = (0., 0.)
    return sample_mean, interval, sample_standard_deviation
