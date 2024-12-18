from typing import Dict, Protocol

from pandas import DataFrame

from hcve_lib.custom_types import Results, Target
from hcve_lib.utils import loc


class CrossValidateCallback(Protocol):
    def __call__(self, X: DataFrame, y: Target) -> Results: ...


def get_learning_curve_data(
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    random_state: int,
    start_samples: float | int = 0.1,
    end_samples: float | int = 1.0,
    n_points: int = 10,
) -> Dict[int, Results]:
    if start_samples < 0:
        raise ValueError("start_samples has to be greate or equal to  0")

    if end_samples < 0:
        raise ValueError("end_samples has to be greate or equal to  0")

    if n_points <= 0:
        raise ValueError("n_points to be greate or equal to  1")

    if start_samples >= end_samples:
        raise ValueError("start_sample must be smaller than the end sample")

    if isinstance(start_samples, float):
        if start_samples > 1:
            raise ValueError("Fraction has to be less or equal to 1 ")
        start_samples = round(len(X) * start_samples)

    if isinstance(end_samples, float):
        if end_samples > 1:
            raise ValueError("Fraction has to be less or equal to 1 ")
        end_samples = round(len(X) * end_samples)

    sample_sizes = (
        round(start_samples + i * (end_samples - start_samples) / (n_points - 1))
        for i in range(n_points)
    )

    results_all = {}

    for sample_size in sample_sizes:
        X_sample = X.sample(n=sample_size, random_state=random_state, replace=False)
        y_sample = loc(X_sample.index, y)
        results_all[sample_size] = cross_validate_callback(X=X_sample, y=y_sample)

    return results_all



