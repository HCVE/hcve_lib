from sklearn.base import BaseEstimator, TransformerMixin
import miceforest as mf

from optuna import Trial
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator


# noinspection PyUnusedLocal
class MiceForest(BaseEstimator, TransformerMixin):
    def __init__(self, iterations=1, random_state=None):
        self.iterations = iterations
        self.random_state = random_state
        self.kds = None

    def fit(self, X, y=None):
        self.kds = mf.ImputationKernel(
            X,
            random_state=self.random_state,
            data_subset=0.5,
            train_nonmissing=True,
            mean_match_candidates={
                24: 0,
                26: 0
            },
        )
        self.kds.mice(
            self.iterations,
            n_estimators=100,
        )
        return self

    def transform(self, X, y=None):
        return self.kds.impute_new_data(X).complete_data(0)


class DropMissingColumns(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.5):
        self.columns_to_drop_ = None
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, DataFrame):
            self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold]
        else:
            raise ValueError("Input should be a pandas DataFrame")
        return self

    def transform(self, X):
        X_transformed = X.drop(columns=self.columns_to_drop_)
        return X_transformed

    def suggest_optuna_(self, trial: Trial, X: DataFrame, prefix: str = ''):
        return trial, {
            'threshold': trial.suggest_uniform(f'{prefix}threshold', 0.1, 1)
        }
