from pandas import DataFrame, Series
from pandas.testing import assert_series_equal
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer

from hcve_lib.cv import cross_validate
from hcve_lib.functional import t
from hcve_lib.pipelines import prepend_timeline, Model
from hcve_lib.wrapped_sklearn import DFPipeline


def test_prepend_timeline():

    class DummyClassifier(BaseEstimator):

        def __init__(self):
            self.y = None

        def fit(self, X, y):
            self.y = y + X['x']

        def predict(self, X):
            return self.y

    old_pipeline = DFPipeline([('step1', DummyClassifier())])
    new_pipeline = prepend_timeline(
        old_pipeline,
        ('step0', FunctionTransformer(lambda s: s + 1)),
    )

    old_pipeline.fit(DataFrame({'x': [10, 10]}), Series([1, 2]))
    assert_series_equal(old_pipeline.predict(DataFrame()), Series([11, 12])),

    new_pipeline.fit(DataFrame({'x': [10, 10]}), Series([1, 2]))
    assert_series_equal(new_pipeline.predict(DataFrame()), Series([12, 13])),


def test_Model():

    X = DataFrame({'x': [1, 2, 3]})
    y = Series([0.5, 2.0, 3.0])

    class TestModel(Model):

        def get_estimator(self) -> BaseEstimator:
            return KNeighborsRegressor(n_neighbors=1)

    cross_validate(X, y)
