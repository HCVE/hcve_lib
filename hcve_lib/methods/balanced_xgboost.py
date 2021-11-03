import pickle

from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from hcve_lib.utils import get_class_ratios, get_class_ratio


class BalancedXGBoostClassifier(XGBClassifier):
    def fit(self, X, y, *args):
        print(get_class_ratio(y))
        self.set_params(scale_pos_weight=get_class_ratio(y))
        super().fit(X, y, *args)
