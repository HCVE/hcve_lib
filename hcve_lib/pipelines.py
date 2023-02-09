from functools import partial, reduce
from typing import Any, Tuple, Callable, Dict, List, Iterable, Union

import numpy as np
import pandas
from optuna import Trial
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from hcve_lib.custom_types import Estimator, Target, TargetTransformer, Method, ExceptionValue, Model, TargetType, \
    Result
from hcve_lib.utils import is_numerical, estimate_categorical_columns, remove_column_prefix
from hcve_lib.wrapped_sklearn import DFPipeline, DFRandomForestRegressor, DFRandomForestClassifier, DFXGBClassifier, \
    DFXGBRegressor, DFLogisticRegression, DFColumnTransformer, DFSimpleImputer, \
    DFOrdinalEncoder, DFStandardScaler, DFElasticNet


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

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        hyperparameters = {
            'n_estimators': trial.suggest_int(f'{prefix}_n_estimators', 5, 200),
            'max_depth': trial.suggest_int(f'{prefix}_max_depth', 1, 10),
            'learning_rate': trial.suggest_float(f'{prefix}_learning_rate', 0.001, 1, log=True),
            'subsample': trial.suggest_float(f'{prefix}_estimator_subsample', 0.1, 1),
            'colsample_bytree': trial.suggest_float(f'{prefix}_colsample_bytree', 0.1, 1),
            'min_split_loss': trial.suggest_float(f'{prefix}_min_split_loss', 0.1, 10),
            'min_child_weight': trial.suggest_int(f'{prefix}_min_child_weight', 1, 100),
            'reg_alpha': trial.suggest_float(f'{prefix}_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float(f'{prefix}_reg_alpha', 0, 10),
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
            [
                *range(1, 20),
                'Unlimited',
            ],
            value=current_config.get('max_depth', 'Unlimited'),
        )

        new_config['max_depth'] = None if max_depth == 'Unlimited' else max_depth

        new_config['learning_rate'] = st.select_slider(
            'Learning rate',
            np.arange(0.1, 1, 0.1),
            value=current_config.get('learning_rate', 0.8),
        )

        new_config['subsample'] = st.select_slider(
            'Fraction of samples for each XGB tree (subsample)',
            np.arange(0.1, 1, 0.1),
            value=current_config.get('subsample', 0.8),
        )

        new_config['colsample_bytree'] = st.select_slider(
            'Fraction of variables for each XGB tree (colsample_bytree)',
            np.arange(0.1, 1, 0.1),
            value=current_config.get('colsample_bytree', 0.8),
        )

        return new_config

    def get_estimator(self) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return DFXGBRegressor(random_state=self.random_state, seed=self.random_state)
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFXGBClassifier(random_state=self.random_state, seed=self.random_state)
        else:
            raise NotImplementedError


class RandomForest(Model):

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        hyperparameters = {
            'n_estimators': trial.suggest_int(f'{prefix}_n_estimators', 5, 2000),
            'max_depth': trial.suggest_int(f'{prefix}_max_depth', 1, 10),
            'min_samples_split': trial.suggest_int(f'{prefix}_min_samples_split', 2, 100),
            'max_features': trial.suggest_categorical(f'{prefix}_max_features', ['sqrt', 'log2', *range(1, 50)]),
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
            max_value=5000,
            value=current_config.get('n_estimators', 100),
            key='n_estimators'
        )
        max_depth = st.select_slider(
            'Tree depth (max_depth)',
            [
                *range(1, 20),
                'Unlimited',
            ],
            value=current_config.get('max_depth', 'Unlimited'),
        )

        new_config['max_depth'] = None if max_depth == 'Unlimited' else max_depth

        new_config['min_samples_split'] = st.select_slider(
            'Minimum sample for decision (min_samples_split)',
            [*range(1, 50)],
            value=current_config.get('min_samples_split', 2),
        )

        new_config['max_features'] = st.select_slider(
            'Subset of features for decision (max_features)',
            ["log2", "sqrt", *range(1, 100)],
            value=current_config.get('max_features', 'sqrt'),
        )
        return new_config

    def get_estimator(self) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return DFRandomForestRegressor(random_state=self.random_state)
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFRandomForestClassifier(random_state=self.random_state)
        else:
            raise NotImplementedError


class LinearModel(Model):

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        if self.target_type == TargetType.REGRESSION:
            hyperparameters = {
                'alpha': trial.suggest_float(f'{prefix}_alpha', 0.1, 100., log=True),
                'l1_ratio': trial.suggest_float(f'{prefix}_l1_ratio', 0, 1)
            }
        else:
            hyperparameters = {
                'penalty': trial.suggest_categorical(f'{prefix}_penalty', ['l1', 'l2', 'elasticnet']),
                'C': trial.suggest_float(f'{prefix}_C', 0.01, 10 ** 3, log=True),
            }

            if hyperparameters['penalty'] == 'elasticnet':
                hyperparameters['l1_ratio'] = trial.suggest_float(f'{prefix}_l1_ratio', 0, 1)

            if hyperparameters['penalty'] in ('elasticnet', 'l1'):
                hyperparameters['solver'] = 'saga'

        return trial, hyperparameters

    def get_estimator(self) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return DFElasticNet(random_state=self.random_state, max_iter=1000)
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFLogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise NotImplementedError

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st
        new_config = {}

        new_config['alpha'] = st.slider(
            "alpha", min_value=0, max_value=5000, value=current_config.get('alpha', 1), key='alpha'
        )

        new_config['l1_ratio'] = st.select_slider(
            'l1_ratio',
            [
                *np.arange(0, 1.1, 0.1),
            ],
            value=current_config.get('l1_ratio', 1.),
        )
        return new_config


class CoxNet(Model):

    def get_estimator(self):
        from hcve_lib.wrapped_sksurv import DFCoxnetSurvivalAnalysis
        return DFCoxnetSurvivalAnalysis(fit_baseline_model=True, n_alphas=1),

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        hyperparameters = {
            'l1_ratio': 1 - trial.suggest_loguniform(f'{prefix}_l1_ratio', 0.1, 1),
            'alphas': [trial.suggest_loguniform(f'{prefix}_alphas', 10 ** -2, 1)],
        }
        return trial, hyperparameters


class RepeatedEnsemble(Estimator):

    def __init__(self, get_pipeline: Callable, repeats: int = 10, random_state: int = None):
        self.get_pipeline = get_pipeline
        self.repeats = repeats
        self.params = {}
        self.random_state = random_state
        self.estimators = []
        for repeat in range(self.repeats):
            self.estimators.append(self.get_pipeline(random_state=self.random_state + (repeat * 10000)))

    def get_estimator(self):
        return self.get_pipeline(random_state=self.random_state)

    def fit(self, X, y, *args, **kwargs):
        self.estimators = []
        for repeat in range(self.repeats):
            pipeline = self.get_pipeline(random_state=self.random_state + (repeat * 10000), X=X)
            pipeline.set_params(**self.params)
            pipeline.fit(X, y)
            self.estimators.append(pipeline)

    def transform(self, X: DataFrame):
        return self.estimators[0].transform(X)

    def predict_proba(self, X: DataFrame):
        y_probas = self.predict(X)
        y_probas_averaged = reduce(lambda sum_df, next_df: sum_df + next_df, y_probas) / len(y_probas)
        return y_probas_averaged

    def predict(self, X: DataFrame):
        y_preds = []
        for estimator in self.estimators:
            y_preds.append(estimator.predict(X))

        y_preds_averaged = reduce(lambda sum_df, next_df: sum_df + next_df, y_preds) / len(y_preds)

        return y_preds_averaged

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        return self.estimators[0].suggest_optuna(trial, prefix)

    def set_params(self, **kwargs):
        self.params = kwargs
        for estimator in self.estimators:
            estimator.set_params(**self.params)

    def get_params(self, **kwargs):
        if self.params is not None:
            return self.params
        else:
            return self.get_estimator().get_params(**kwargs)

    def get_feature_importance(self) -> Series:
        feature_importances = []
        for estimator in self.estimators:
            feature_importances.append(estimator.get_feature_importance())
        return pandas.concat(feature_importances, axis=1)

    def get_p_value_feature_importance(self, X: DataFrame, y: Target) -> Series:
        feature_importances = []
        for estimator in self.estimators:
            feature_importances.append(estimator.get_p_value_feature_importance(X, y))
        return pandas.concat(feature_importances, axis=1)

    def __getattr__(self, item):
        if hasattr(self.estimators[0], item):
            return getattr(self.estimators[0], item)
        else:
            raise AttributeError(f'AttributeError: object has no attribute \'{item}\'')

    def __getitem__(self, item):
        return self.estimator[item]

    # TODO
    def get_final(self):
        return self.estimators[0]


def get_target_type(y: Target) -> TargetType:
    if is_numerical(y):
        return TargetType.REGRESSION
    else:
        return TargetType.CLASSIFICATION


def aggregate_results(
        results: Union[List[Result], Result],
        callback: Callable[[Estimator], Any],
) -> DataFrame:
    if isinstance(results, list):
        results_ = results
    else:
        results_ = [results]

    model_output = {}

    for repeat_n, result, in enumerate(results_):
        for split_name, prediction in result.items():
            model_output[f'{repeat_n}_{split_name}'] = callback(prediction)

    return pandas.concat(model_output, axis=1)


def get_results_feature_importance(results: Union[List[Result], Result]) -> DataFrame:
    return aggregate_results(results, lambda prediction: prediction['model'].get_feature_importance()).copy()


def get_results_p_value_feature_importance(results: Union[List[Result], Result], X: DataFrame, y: Target) -> DataFrame:
    return aggregate_results(results, lambda prediction: prediction['model'].get_p_value_feature_importance(X, y))


def get_supervised_pipeline(
        X: DataFrame,
        y: Target,
        random_state: int,
        get_estimator: Callable[..., Estimator],
) -> DFPipeline:
    categorical = estimate_categorical_columns(X)
    continuous = list(set(X.columns) - set(categorical))

    return DFPipeline(
        [
            (
                'impute',
                DFColumnTransformer(
                    [
                        ('categorical', DFSimpleImputer(strategy='most_frequent'), categorical),
                        ('continuous', DFSimpleImputer(strategy='mean'), continuous)
                    ],
                )
            ),
            ('remove_prefix', FunctionTransformer(remove_column_prefix)),
            (
                'encode',
                DFColumnTransformer(
                    [('categorical', DFOrdinalEncoder(), categorical)],
                    remainder='passthrough',
                )
            ),
            ('remove_prefix2', FunctionTransformer(remove_column_prefix)),
            ('scaler', DFStandardScaler()),
            ('estimator', get_estimator(target_type=get_target_type(y), random_state=random_state)),
        ]
    )
