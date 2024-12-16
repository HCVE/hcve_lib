from typing import List, Union
from unittest import mock

from pandas import Series, DataFrame
from statsmodels.compat.pandas import assert_series_equal

from hcve_lib.custom_types import Prediction, Results, Target, ExceptionValue
from hcve_lib.evaluation_functions import (
    compute_metric_groups,
    get_2_level_groups,
    compute_metrics_prediction,
    get_splits_by_class,
    get_target_label,
    get_splits_by_age,
    average_group_scores,
    get_inverse_weight,
    compute_metrics,
)
from hcve_lib.metrics import Maximize
from hcve_lib.metrics_types import Metric
from hcve_lib.utils import get_fractions


def test_compute_metrics():

    X = DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1.1, 1.2, 1.3, 1.4],
            "feature3": [2.1, 2.2, 2.3, 2.4],
        },
        index=["idx1", "idx2", "test_idx_1", "test_idx_2"],
    )

    y = Series(
        [0, 1, 0, 0],
        index=["idx1", "idx2", "test_idx_1", "test_idx_2"],
        name="target",
    )

    result: Results = [
        {
            "split_1": {
                "y_pred": Series(
                    [1, 1, 0, 0],
                    index=["test_idx_1", "test_idx_1", "test_idx_2", "test_idx_2"],
                ),
                "y_column": "target",
                "X_columns": ["feature1", "feature2", "feature3"],
                "model": "MockModel",
                "split": (
                    ["idx1", "idx1" "idx2"],
                    ["test_idx_1", "test_idx_1", "test_idx_2", "test_idx_2"],
                ),
            }
        },
    ]

    class MockMetric(Maximize, Metric):
        def __init__(self):
            super().__init__()

        def get_names(
            self,
            prediction: Prediction,
            y: Target,
        ) -> List[str]:
            return ["metric"]

        def compute(
            self, y_true: Target, y_pred: Target
        ) -> List[Union[ExceptionValue, float]]:
            return [(y_true - y_pred).mean()]

    metric = MockMetric()

    metrics = compute_metrics(
        results=result,
        y=y,
        metrics=[metric],
    )

    assert metrics["metric"]["mean"] == -0.5


def test_compute_metric_groups():
    def dummy_metric(fold: Prediction):
        return (fold["y_score"]).tolist()

    fold = Prediction(
        y_score=Series(
            [10, 20, 30],
            index=["a", "b", "c"],
        ),
        y_column="y_column",
        X_columns=["a", "b"],
        split=[([0], [1])],
        model=None,
        random_state=0,
    )
    expected_result = {0: [30], 1: [10, 20]}

    data = DataFrame(
        {"agg": [0, 1, 1, 0]},
        index=["x", "a", "b", "c"],
    )
    assert compute_metric_groups(
        dummy_metric,
        get_2_level_groups(
            {"a": fold, "b": fold},
            data.groupby("agg"),
            data,
        ),
    ) == {"a": expected_result, "b": expected_result}


def test_compute_metric_per_split():
    prediction_ = {"a": {}}
    y_ = {}

    # noinspection PyMethodMayBeStatic
    class DummyMetric:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def get_names(self, prediction, y):
            assert prediction == prediction
            assert y == y
            return [self.key]

        def get_values(self, prediction, y):
            assert prediction == prediction
            assert y == y
            return [self.value]

    assert compute_metrics_prediction(
        prediction_,
        y_,
        [
            DummyMetric("a", 1.0),
            DummyMetric("b", 2.0),
        ],
    ) == {"a": 1.0, "b": 2.0}

    assert compute_metrics_prediction(
        prediction_,
        y_,
        [
            DummyMetric("a", 1.0),
            DummyMetric("b", 2.0),
        ],
        skip_metrics=["b"],
    ) == {
        "a": 1.0,
    }


def test_get_splits_by_class():
    y = {
        "data": Series(
            [
                1,
                1,
                2,
            ],
            index=["a", "b", "c"],
        )
    }
    assert get_splits_by_class(y) == {1: ["a", "b"], 2: ["c"]}


def test_get_splits_by_age():
    assert get_splits_by_age(
        Series(
            [10, 11, 12, 18, 25],
            index=[10, 20, 30, 40, 50],
        )
    ) == {{"age_10": [10, 20, 30], "age_20": [40, 50]}}


def test_get_target_label():
    series = Series(
        [
            1,
            1,
            2,
        ],
        index=["a", "b", "c"],
        name="label",
    )

    assert_series_equal(
        get_target_label({"data": series}),
        series,
    )

    assert_series_equal(
        get_target_label({"data": DataFrame({"label": series})}),
        series,
    )


def test_average_group_y_score():
    out = average_group_scores(
        {
            0: {
                "a": {
                    "y_score": Series([1, 2, 3]),
                    "y_proba": {"x": Series([0.1, 0.2, 0.3])},
                }
            },
            1: {
                "a": {
                    "y_score": Series([2, 3, 4]),
                    "y_proba": {"x": Series([0.2, 0.3, 0.4])},
                }
            },
        }
    )

    assert_series_equal(
        out["a"]["y_score"],
        Series([1.5, 2.5, 3.5]),
    )

    assert_series_equal(
        out["a"]["y_proba"]["x"],
        Series([0.15, 0.25, 0.35]),
    )


def test_get_inverse_weight():
    # Equal weights
    data = Series(["x", "x", "y", "z"])
    assert (get_inverse_weight(data) * get_fractions(data)).tolist() == [
        1 / 3,
        1 / 3,
        1 / 3,
    ]

    # Specific weights
    data2 = Series(["x", "x", "y"])
    assert (
        get_inverse_weight(data2, proportions={"x": 0.1, "y": 0.9})
        * get_fractions(data2)
    ).tolist() == [0.1, 0.9]

    print(get_inverse_weight(data2, proportions={"x": 0.1, "y": 0.9}))
