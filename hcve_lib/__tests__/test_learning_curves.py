import pytest
from pandas import DataFrame, Series

from hcve_lib.custom_types import Target, Results
from hcve_lib.learning_curves import get_learning_curve_data


@pytest.fixture
def X_mock():
    return DataFrame({"feature1": range(100)})


@pytest.fixture
def y_mock():
    return Series(range(100))


def mock_cross_validate_callback(X: DataFrame, y: Target) -> Results:
    return [{"fold": {"split": (X.index.tolist(), [])}}]


def test_valid_learning_curve_data_float_points(X_mock, y_mock):
    learning_curve_data = get_learning_curve_data(
        X=X_mock,
        y=y_mock,
        cross_validate_callback=mock_cross_validate_callback,
        random_state=42,
        start_samples=0.1,
        end_samples=1.0,
        n_points=5,
    )

    assert list(learning_curve_data.keys()) == [10, 32, 55, 78, 100]

    for n_point, result in learning_curve_data.items():
        # mock_cross_validate_callback returns all passed data in the training set (index 0 vs index 1)
        assert len(result[0]["fold"]["split"][0]) == n_point


def test_invalid_start_sample_greater_than_end_sample(X_mock, y_mock):
    from pytest import raises

    with raises(ValueError, match="start_sample must be smaller than the end sample"):
        get_learning_curve_data(
            X=X_mock,
            y=y_mock,
            cross_validate_callback=mock_cross_validate_callback,
            random_state=42,
            start_samples=100,
            end_samples=10,
            n_points=5,
        )


def test_invalid_start_samples_negative(X_mock, y_mock):
    from pytest import raises

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


def test_invalid_end_samples_negative(X_mock, y_mock):
    from pytest import raises

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


def test_invalid_n_points_zero_or_negative(X_mock, y_mock):
    from pytest import raises

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
