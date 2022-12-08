from functools import partial
from typing import Any, Tuple, Callable, Dict

from optuna import Trial
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from hcve_lib.custom_types import Estimator, Target, TargetTransformer, Method, ExceptionValue, Model
from hcve_lib.wrapped_sklearn import DFPipeline, DFRandomForestRegressor


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


class XGBoost(Model):

    def get_estimator(self) -> BaseEstimator:
        return XGBRegressor()

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        hyperparameters = {
            'n_estimators': trial.suggest_int(f'{prefix}_n_estimators', 5, 200),
            'max_depth': trial.suggest_int(f'{prefix}_max_depth', 1, 10),
            'learning_rate': trial.suggest_loguniform(f'{prefix}_learning_rate', 0.001, 1),
            'subsample': trial.suggest_uniform(f'{prefix}_estimator_subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_uniform(f'{prefix}_colsample_bytree', 0.1, 1),
            'min_split_loss': trial.suggest_uniform(f'{prefix}_min_split_loss', 0.1, 10),
            'min_child_weight': trial.suggest_int(f'{prefix}_min_child_weight', 1, 100),
            'reg_alpha': trial.suggest_uniform(f'{prefix}_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_uniform(f'{prefix}_reg_alpha', 0, 10),
        }
        return trial, hyperparameters

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st
        new_config = {}

        new_config['n_estimators'] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=2000,
            value=current_config.get('n_estimators', 100),
            key='n_estimators'
        )

        max_depth = st.select_slider(
            'Tree depth (max_depth)',
            [*range(1, 20), 'Unlimited', ],
            value=current_config.get('max_depth', 'Unlimited'),
        )

        new_config['max_depth'] = None if max_depth == 'Unlimited' else max_depth

        new_config['learning_rate'] = st.select_slider(
            'Fraction of samples for each tree (subsample)',
            np.arange(0.1, 1, 0.1),
            value=current_config.get('learning_rate', 0.8),
        )

        new_config['subsample'] = st.select_slider(
            'Fraction of samples for each tree (subsample)',
            np.arange(0.1, 1, 0.1),
            value=current_config.get('subsample', 0.8),
        )

        return new_config


class RandomForest(Model):

    def get_estimator(self) -> BaseEstimator:
        return DFRandomForestRegressor()

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        hyperparameters = {
            'n_estimators': trial.suggest_int(f'{prefix}_n_estimators', 5, 200),
            'max_depth': trial.suggest_int(f'{prefix}_max_depth', 1, 10),
            'min_samples_split': trial.suggest_int(f'{prefix}_min_samples_split', 2, 100),
            'max_features': trial.suggest_categorical(f'{prefix}_max_features', ['auto', 'sqrt', 'log2']),
            'oob_score': trial.suggest_categorical(f'{prefix}_oob_score', [True, False]),
        }
        return trial, hyperparameters

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st
        new_config = {}

        new_config['n_estimators'] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=2000,
            value=current_config.get('n_estimators', 100),
            key='n_estimators'
        )

        max_depth = st.select_slider(
            'Tree depth (max_depth)',
            [*range(1, 20), 'Unlimited', ],
            value=current_config.get('max_depth', 'Unlimited'),
        )

        new_config['max_depth'] = None if max_depth == 'Unlimited' else max_depth

        new_config['min_samples_split'] = st.select_slider(
            'Minimum sample for decision (min_samples_split)',
            [*range(1, 20)],
            value=current_config.get('min_samples_split', 2),
        )

        new_config['max_features'] = st.select_slider(
            'Subset of features for decision (max_features)',
            ["log2", "sqrt", *range(1, 20)],
            value=current_config.get('max_features', 'sqrt'),
        )
        return new_config
