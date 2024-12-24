from pandas import Series, DataFrame
from sklearn.datasets import load_diabetes

from hcve_lib.cv import cross_validate
from hcve_lib.feature_selection import get_importance_feature_selection_curve
from hcve_lib.pipelines import LinearModel, get_supervised_pipeline
from hcve_lib.splitting import get_train_test
from hcve_lib.utils import partial


def test_integration_get_importance_feature_selection_curve():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X = DataFrame(X)
    y = Series(y)

    feature_selection_curve = get_importance_feature_selection_curve(
        X,
        y,
        partial(
            cross_validate,
            get_pipeline=partial(get_supervised_pipeline, get_estimator=LinearModel),
            get_splits=get_train_test,
            random_state=423,
            return_models=True,
            n_jobs=1,
            n_repeats=1,
        ),
    )

    print(feature_selection_curve)
