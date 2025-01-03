import pytest
from sklearn.datasets import load_diabetes

from hcve_lib.cv import cross_validate
from hcve_lib.feature_selection import get_importance_feature_selection_curve
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
    )
    r2s = {
        n_features: point["metrics"]["r2_score"]["mean"]
        for n_features, point in feature_selection_curve.items()
    }

    expected_values = {
        1: 0.2854479626941153,
        2: 0.4357810812240352,
        3: 0.4601340417941478,
        4: 0.47329204767833405,
        5: 0.474741625593844,
        6: 0.47456789626649976,
        7: 0.47710086700847226,
        8: 0.4761210464456674,
        9: 0.4871193062950186,
        10: 0.4943989256884606,
    }

    for n_features, expected_value in expected_values.items():
        assert expected_value == pytest.approx(r2s[n_features], rel=10e-2)
