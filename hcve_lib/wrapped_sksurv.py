import numpy as np
import pandas
from pandas import Series, DataFrame
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.meta import Stacking
from sksurv.tree import SurvivalTree

from hcve_lib.custom_types import Estimator, Target
from hcve_lib.utils import configuration_to_params
from hcve_lib.wrapped_sklearn import DFWrapped, ToSurvivalRecord


class DFCoxnetSurvivalAnalysis(DFWrapped, ToSurvivalRecord, CoxnetSurvivalAnalysis):
    def predict_survival_function(self, X, alpha=None, return_array=False):
        return Series(
            super().predict_survival_function(X.to_numpy(), alpha, return_array),
            index=X.index,
        )


class DFCoxPHSurvivalAnalysis(DFWrapped, CoxPHSurvivalAnalysis):
    pass


class DFRandomSurvivalForest(DFWrapped, ToSurvivalRecord, RandomSurvivalForest):
    pass


class DFSurvivalGradientBoosting(
    DFWrapped, ToSurvivalRecord, GradientBoostingSurvivalAnalysis
):
    pass


class DFSurvivalStacking(DFWrapped, Stacking, Estimator):
    def fit(self, X: DataFrame, y: Target = None, *args, **kwargs):
        self.save_fit_features(X)
        Stacking.fit(self, X, y)
        return self

    def predict(self, X: DataFrame, **kwargs):
        Xt = self._predict_estimators(X)
        return self.final_estimator_.predict(Xt)

    def _predict_estimators(self, X):
        Xt = None
        start = 0
        n_classes = None

        for estimator in self.estimators_:
            if self.probabilities and hasattr(estimator, "predict_proba"):
                p = estimator.predict_proba(X)
            else:
                p = estimator.predict(X)

            if Xt is None:
                Xt = p
            else:
                Xt = pandas.concat([Xt, p], axis=1)

        return Xt

    def set_params(self, **params):
        if "meta_learning" in params:
            self.meta_estimator.set_params(
                **configuration_to_params(params["meta_learning"])
            )

        if "base_estimators" in params:
            for name, hyperparameters in params["base_estimators"].items():
                for name_base_estimator, base_estimator in self.base_estimators:
                    if name_base_estimator == name:
                        base_estimator.set_params(
                            **configuration_to_params(hyperparameters)
                        )

        return self


class DFSurvivalTree(DFWrapped, ToSurvivalRecord, SurvivalTree):
    pass
