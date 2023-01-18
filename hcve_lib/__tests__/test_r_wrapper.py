from hcve_lib.r_wrapper import REstimator
from pandas import DataFrame, Series


def test_REstimator():
    X = DataFrame()
    y = Series()
    estimator = REstimator('hcve_lib/__tests__/test.R/test_function')
    estimator.fit(X, y)
