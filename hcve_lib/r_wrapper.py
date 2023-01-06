import re

from pandas import DataFrame

from hcve_lib.custom_types import Estimator
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


# Must be activated


class REstimator(Estimator):
    def __init__(self, r_path: str):
        pandas2ri.activate()
        path_parsed = re.search(r'(.*)/(.*)', r_path)
        self.r_file = path_parsed.group(1)
        self.r_function = path_parsed.group(2)
        robjects.r['source'](self.r_file)

    def fit(self, X, y, *args, **kwargs):
        robjects.globalenv['getname']

    def predict(self, X: DataFrame):
        ...

    def predict_proba(self, X: DataFrame):
        ...

    def predict_survival_at_time(self, X: DataFrame, time: int):
        ...
