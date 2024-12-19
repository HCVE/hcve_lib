from typing import Any
from unittest import mock
from unittest.mock import Mock

import pytest
from pytest import approx
from pandas import DataFrame, Series
from pytest import raises
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from hcve_lib.custom_types import Results
from hcve_lib.custom_types import Target
from hcve_lib.cv import cross_validate
from hcve_lib.learning_curves import (
    get_learning_curve_data,
    compute_learning_curve_metrics,
    get_mean_train_size,
    GetSplits,
)
from hcve_lib.metrics import FunctionMetric
from hcve_lib.splitting import get_bootstrap
from hcve_lib.utils import partial


@pytest.fixture
def X_mock():
    return DataFrame({"feature1": range(100)})


@pytest.fixture
def y_mock():
    return Series(range(100))


def mock_get_get_splits(X: DataFrame, y: Target, random_state: int, train_size: int):
    data_index = X.index.tolist()
    return {
        "fold": (
            data_index[:train_size],
            data_index[train_size:],
        )
    }


def mock_cross_validate_callback(
    X: DataFrame, y: Target, random_state: int, get_splits: GetSplits
) -> Any:
    splits = get_splits(X, y)
    return [{split_name: {"split": split} for split_name, split in splits.items()}]


def test_valid_learning_curve_data_float_points(X_mock, y_mock) -> None:
    learning_curve_data = get_learning_curve_data(
        X=X_mock,
        y=y_mock,
        cross_validate_callback=mock_cross_validate_callback,
        random_state=42,
        get_splits=mock_get_get_splits,
        start_samples=0.1,
        end_samples=1.0,
        n_points=5,
    )

    # actual data points are half due to splitting
    assert list(learning_curve_data.keys()) == [10, 32, 55, 78, 100]

    for n_point, result in learning_curve_data.items():
        # mock_cross_validate_callback returns all passed data in the training set (index 0 vs index 1)
        assert len(result[0]["fold"]["split"][0]) == n_point


def test_invalid_start_sample_greater_than_end_sample(X_mock, y_mock) -> None:
    with raises(ValueError, match="start_sample must be smaller than the end sample"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=100,
            end_samples=10,
            n_points=10,
        )


def test_invalid_start_samples_negative(X_mock, y_mock) -> None:
    with raises(ValueError, match="start_samples has to be greate or equal to  0"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=-2,
            end_samples=10,
            n_points=5,
        )


def test_invalid_end_samples_negative(X_mock, y_mock) -> None:
    with raises(ValueError, match="end_samples has to be greate or equal to  0"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=2,
            end_samples=-10,
            n_points=5,
        )


def test_invalid_n_points_zero_or_negative(X_mock, y_mock) -> None:
    with raises(ValueError, match="n_points to be greate or equal to  1"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=10,
            end_samples=100,
            n_points=0,
        )

    with raises(ValueError, match="n_points to be greate or equal to  1"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=10,
            end_samples=100,
            n_points=-3,
        )


def get_gaussian_nb(X: DataFrame, y: Target, random_state: int):
    return GaussianNB()


def test_integration_get_learning_curve_data() -> None:
    X, y = load_digits(return_X_y=True)
    X = DataFrame(X)
    y = Series(y)

    learning_curve_data = get_learning_curve_data(
        X=X,
        y=y,
        cross_validate_callback=partial(
            cross_validate,
            get_pipeline=get_gaussian_nb,
            n_repeats=10,
        ),
        get_splits=get_bootstrap,
        random_state=42,
        start_samples=200,
        end_samples=500,
        n_points=5,
    )

    metrics = compute_learning_curve_metrics(
        data=learning_curve_data, y=y, metrics=[FunctionMetric(accuracy_score)]
    )

    metric_values = {k: m["accuracy_score"]["mean"] for k, m in metrics.items()}

    expected_values = {
        200: 0.6112326524731425,
        275: 0.658183757787916,
        350: 0.7010074222634326,
        425: 0.7626011153805117,
        500: 0.7710780296530093,
    }

    for sample_size, expected_value in expected_values.items():
        assert metric_values[sample_size] == approx(expected_value, rel=10e-2)


def test_compute_learning_curve_metrics() -> None:
    mock_results1 = Mock()
    mock_results2 = Mock()
    mock_y = Mock()
    mock_metrics = Mock()

    with mock.patch("hcve_lib.learning_curves.compute_metrics") as compute_metrics:
        compute_metrics.side_effect = [{"metric1": 1}, {"metric1": 2}]

        learning_curve_metrics = compute_learning_curve_metrics(
            data={10: mock_results1, 100: mock_results2},
            y=mock_y,
            metrics=mock_metrics,
        )

    assert len(learning_curve_metrics) == 2
    assert learning_curve_metrics[10] == {"metric1": 1}
    assert learning_curve_metrics[100] == {"metric1": 2}


def test_get_mean_train_size_with_valid_results() -> None:
    results: Results = [
        {
            "fold_1": {"split": ([1, 2, 3, 4], [])},
            "fold_2": {"split": ([1, 2, 3, 4, 5], [])},
        },
        {
            "fold_3": {"split": ([1, 2, 3], [])},
        },
    ]
    assert get_mean_train_size(results) == 4
