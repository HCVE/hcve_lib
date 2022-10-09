from typing import Tuple

import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame

from hcve_lib.custom_types import Target


def survival_to_interval(target: Target) -> Tuple[ndarray, ndarray]:
    new_data = target['data']\
        .apply(survival_to_interval_record, axis=1)\
        .to_numpy()

    return new_data[:, 0], new_data[:, 1]


def survival_to_interval_record(survival_row: Series) -> Series:
    if survival_row['label'] == 0:
        return Series([survival_row['tte'], +np.inf])
    elif survival_row['label'] == 1:
        return Series([survival_row['tte'], survival_row['tte']])
    else:
        raise ValueError('Event label can be only 1 and 0')


def get_event_probability(y_data: DataFrame) -> float:
    counts = y_data['label'].value_counts()
    return counts[1] / counts.sum()


def get_event_case_ratio(y_data: DataFrame) -> float:
    counts = y_data['label'].value_counts()
    return counts[1] / counts[0]
