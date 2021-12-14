from collections import Callable
from typing import Any, Tuple, Callable

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from hcve_lib.custom_types import Estimator, Target, TargetTransformer, Method
from hcve_lib.wrapped_sklearn import DFPipeline


class TransformTarget(BaseEstimator):
    def __init__(
        self,
        inner: Estimator,
        transform_callback: Callable[[Target], Any],
    ):
        self.inner = inner
        self.transform_callback = transform_callback

    def fit(self, X, y):
        self.inner.fit(X, self.transform_callback(y))
        return self

    def predict(self, X, **kwargs):
        return self.inner.predict(X, **kwargs)

    def score(self, X, y):
        return self.inner.score(X, self.transform_callback(y))

    def predict_survival_function(self, X):
        return self.inner.predict_survival_function(X)

    def predict_survival(self, X):
        return self.inner.predict_survival(X)


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
            columns = self.transformer.inverse_transform(
                range(y_proba.shape[1]))
            return DataFrame(y_proba, columns=columns, index=X.index)
        else:
            return y_proba

    def score(self, X, y):
        return self.inner.score(X, self.transformer.transform(y))


def prepend_timeline(pipeline: DFPipeline,
                     step: Tuple[str, Estimator]) -> DFPipeline:
    return DFPipeline([step] + pipeline.steps)


def subsample_pipeline(X: DataFrame, method: Method) -> Pipeline:
    pipeline = method.get_estimator(X)
    return prepend_timeline(pipeline,
                            ('subsample', FunctionTransformer(subsample_data)))


def subsample_data(X: DataFrame) -> DataFrame:
    return X.sample(frac=0.1)


class Callback(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fit_callback: Callable[[DataFrame], Any] = None,
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
        else:
            print('transform', X)
        if self.breakpoint_transform:
            breakpoint()
        return X

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X)
        else:
            print('fit', X)
        if self.breakpoint_fit:
            breakpoint()
        return self
