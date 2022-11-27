from functools import partial
from typing import Any, Tuple, Callable

from hcve_lib.custom_types import Estimator, Target, TargetTransformer, Method, ExceptionValue
from hcve_lib.wrapped_sklearn import DFPipeline
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#
# class RandomForest(Model):
#
#     # def get_optimize(self):
#     #     return Optmi
#
#     def __init__(self, random_state: int, configuration: Dict):
#         self.random_state = random_state
#         self.configuration = configuration
#
#     def get_esimator(
#         self,
#         X: DataFrame,
#         random_state: int,
#         configuration: Dict,
#         verbose: bool = False,
#     ) -> DFPipeline:
#         return make_pipeline(
#             [
#                 (
#                     'estimator',
#                     RandomSurvivalForestT(
#                         transform_callback=to_survival_y_records, random_state=RANDOM_STATE, n_jobs=1
#                     ),
#                 )
#             ],
#             X,
#             configuration=configuration,
#         )
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         hyperparameters = {
#             'estimator': {
#                 'n_estimators': trial.
#                 suggest_int('estimator_n_estimators', 5, 200),
#                 'max_depth': trial.suggest_int('estimator_max_depth', 1, 4),
#                 'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
#                 'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
#                 'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
#                 'oob_score': trial.suggest_categorical('estimator_oob_score', [True, False]),
#             }
#         }
#         return trial, hyperparameters


class EstimatorDecorator:

    def __init__(self, estimator):
        self._estimator = estimator

    def get_params(self, *args, **kwargs):
        return self._estimator.get_params(*args, **kwargs)

    def __getattr__(self, item):
        if item == '_estimator':
            return getattr(self, '_estimator')
        else:
            return getattr(self._estimator, item)

    def __setattr__(self, attr_name, attr_value):
        if attr_name == '_estimator':
            super().__setattr__('_estimator', attr_value)
        else:
            setattr(self._estimator, attr_name, attr_value)


class TransformTarget:

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))


class TransformerTarget(BaseEstimator):

    def __init__(
        self,
        inner: Estimator,
        transformer: TargetTransformer,
        inverse: bool = True,
    ):
        self.inner = inner
        self.transformer = transformer
        self.inverse = inverse

    def fit(self, X, y):
        self.transformer.fit(y)
        self.inner.fit(X, self.transformer.transform(y))
        return self

    def predict(self, X, **kwargs):
        y_pred = self.inner.predict(X, **kwargs)
        if self.inverse:
            return self.transformer.inverse_transform(y_pred)
        else:
            return y_pred

    def predict_proba(self, X, **kwargs):
        y_proba = self.inner.predict_proba(X, **kwargs)

        if self.inverse:
            columns = self.transformer.inverse_transform(range(y_proba.shape[1]))
            return DataFrame(y_proba, columns=columns, index=X.index)
        else:
            return y_proba

    def score(self, X, y):
        return self.inner.score(X, self.transformer.transform(y))


def prepend_timeline(pipeline: DFPipeline, step: Tuple[str, Estimator]) -> DFPipeline:
    return DFPipeline([step] + pipeline.steps)


def subsample_pipeline(X: DataFrame, method: Method) -> Pipeline:
    pipeline = method.get_estimator(X)
    return prepend_timeline(pipeline, ('subsample', FunctionTransformer(subsample_data)))


def subsample_data(X: DataFrame) -> DataFrame:
    return X.sample(frac=0.1)


class Callback(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        fit_callback: Callable[[DataFrame, Target], Any] = None,
        transform_callback: Callable[[DataFrame], Any] = None,
        breakpoint_fit: bool = False,
        breakpoint_transform: bool = False,
    ):
        self.breakpoint_fit = breakpoint_fit
        self.breakpoint_transform = breakpoint_transform
        self.fit_callback = fit_callback
        self.transform_callback = transform_callback
        super()

    def transform(self, X):
        if self.transform_callback:
            self.transform_callback(X)
        elif not self.fit_callback and not self.transform_callback:
            print('transform', X)
        if self.breakpoint_transform:
            breakpoint()
        return X

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X, y)
        else:
            print('fit', X)
        if self.breakpoint_fit:
            breakpoint()
        return self


class LifeTime(EstimatorDecorator, BaseEstimator):

    def fit(self, X, y, *args, **kwargs):
        y_df = y['data'].copy()
        y_df['tte'] += X['AGE'] * 365
        y_transformed = {**y, 'data': y_df}
        self._estimator.fit(X, y_transformed, *args, **kwargs)
        return self

    def add_age(self, survival_function, age: Series, t: int):
        try:
            return survival_function(t + age * 365)
        except ValueError as e:
            return ExceptionValue(e)

    def predict_survival_function(
        self,
        X: DataFrame,
    ) -> Callable[[int], float]:
        survival_functions = self._estimator.predict_survival_function(X)
        return (partial(self.add_age, fn, age) for age, fn in zip(X['AGE'], survival_functions))
