from pandas import DataFrame, Series
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from hcve_lib.custom_types import Target, Pipeline
from hcve_lib.cv import cross_validate
from hcve_lib.feature_selection import get_importance_feature_selection_curve
from hcve_lib.splitting import get_train_test
from hcve_lib.utils import partial


def get_pipeline(X: DataFrame, y: Target, random_state: int):
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "rfe",
                LogisticRegression(),
            ),
        ]
    )


def test_integrate_get_importance_feature_selection():
    X, y = load_digits(return_X_y=True)
    X = DataFrame(X)
    y = Series(y)
    feature_selection_curve = get_importance_feature_selection_curve(
        X=X,
        y=y,
        cross_validate_callback=partial(
            cross_validate,
            get_pipeline=get_pipeline,
            get_splits=get_train_test,
            n_repeats=10,
            random_state=123,
            return_models=True,
            n_jobs=1,
        ),
    )
