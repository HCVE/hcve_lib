import re
from typing import Dict, Iterable, Tuple, List

import pandas
from pandas import DataFrame, Series

from hcve_lib.custom_types import Estimator, Target
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


class REstimator(Estimator):
    model = None

    def __init__(self, r_path: str):
        self.feature_names_in_ = None
        pandas2ri.activate()
        path_parsed = re.search(r'(.*)/(.*)', r_path)
        self.r_file = path_parsed.group(1)
        self.r_function = path_parsed.group(2)
        self.source()

    def fit(self, X, y, *args, **kwargs):
        data, X_column, y_column = transform_df(X, y)
        self.feature_names_in_ = X.columns
        self.model = robjects.globalenv[self.r_function](data, X_column, y_column)
        return self

    def predict(self, X: DataFrame):
        return self.predict_proba(X)

    def predict_proba(self, X: DataFrame):
        X_renamed = X.rename(columns=sanitize_name)
        self.source()
        return Series(r_to_dict(robjects.r['predict'](self.model, X_renamed))['predictions'], index=X.index)

    def get_feature_importance(self):
        self.source()
        return Series(robjects.r['importance'](self.model), index=self.feature_names_in_).sort_values(ascending=False)

    def source(self):
        robjects.r("source('" + self.r_file + "')")


def transform_df(X: DataFrame, y: Target) -> Tuple[DataFrame, List[str], str]:
    X_renamed = X.rename(columns=sanitize_name)
    y_renamed = Series(y, name=sanitize_name(y.name))
    print('y_renamed', y_renamed.name)
    return pandas.concat([X_renamed, y_renamed], axis=1), list(X_renamed.columns), str(y_renamed.name)


def sanitize_name(item: str) -> str:
    return item\
        .replace(' ', '.')\
        .replace('-', '.')


def r_to_dict(what) -> Dict:
    return dict(what.items())
