import pytest
from sklearn.datasets import load_diabetes

from hcve_lib.cv import cross_validate
from hcve_lib.feature_selection import (
    get_importance_feature_selection_curve,
    evaluate_stepped_points,
)
from hcve_lib.pipelines import LinearModel, get_basic_pipeline
from hcve_lib.splitting import get_train_test
from hcve_lib.utils import partial


def test_integration_get_importance_feature_selection_curve():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    feature_selection_curve = get_importance_feature_selection_curve(
        X,
        y,
        partial(
            cross_validate,
            get_pipeline=partial(get_basic_pipeline, get_estimator=LinearModel),
            get_splits=get_train_test,
            random_state=423,
            return_models=True,
            n_jobs=1,
            n_repeats=10,
        ),
        get_evaluated_points=partial(evaluate_stepped_points, threshold=5, step=3),
    )

    r2s = {
        n_features: point["metrics"]["r2_score"]["mean"]
        for n_features, point in feature_selection_curve.items()
    }

    expected_values = {
        10: 0.49440127268580697,
        7: 0.47710086700847226,
        5: 0.474741625593844,
        4: 0.47329204767833405,
        3: 0.4601340417941478,
        2: 0.4357810812240352,
        1: 0.2854479626941153,
    }

    for n_features, expected_value in expected_values.items():
        assert expected_value == pytest.approx(r2s[n_features], rel=10e-2)


def test_evaluate_stepped_points():
    result = list(evaluate_stepped_points(10, 5, 2))
    assert result == [10, 8, 6, 5, 4, 3, 2, 1]

    result = list(evaluate_stepped_points(10, 5, 3))
    assert result == [10, 7, 5, 4, 3, 2, 1]

    result = list(evaluate_stepped_points(5, 3, 1))
    assert result == [5, 4, 3, 2, 1]


# Run the test
test_evaluate_stepped_points()
