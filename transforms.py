from optuna import Trial
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator


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
