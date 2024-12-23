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
            n_repeats=1,
        ),
    )

    print(feature_selection_curve)
