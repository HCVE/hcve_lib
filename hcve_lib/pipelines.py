import json
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from functools import reduce
from logging import Logger
from math import log, exp, inf
from statistics import mean
from time import sleep
from typing import Any, Tuple, Dict, Union, Iterable, Optional
from typing import List, Callable

import numpy as np
import pandas
import ray
import toolz
import xgboost as xgb
from numpy import transpose
from optuna import Trial
from pandas import DataFrame
from pandas import Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sksurv.ensemble import RandomSurvivalForest
from toolz import dissoc
from xgboost import Booster, DMatrix

from hcve_lib.custom_types import (
    Estimator,
    Target,
    TargetTransformer,
    Method,
    TargetType,
    Result,
    TargetObject,
    TargetData,
)
from hcve_lib.custom_types import ExceptionValue
from hcve_lib.functional import t
from hcve_lib.utils import (
    is_numerical,
    estimate_continuous_columns,
    remove_column_prefix,
    loc,
    compute_classification_scores_statistics,
    average_classification_scores,
    estimate_categorical_and_continuous_columns,
)
from hcve_lib.visualisation import print_formatted
from hcve_lib.wrapped_sklearn import (
    DFPipeline,
    DFRandomForestRegressor,
    DFRandomForestClassifier,
    DFLogisticRegression,
    DFColumnTransformer,
    DFSimpleImputer,
    DFStandardScaler,
    DFElasticNet,
    DFXGBClassifier,
    DFWrapped,
    DFExtraTreesClassifier,
    DFSurvivalXGB,
    DFOneHotEncoder,
    DFGaussianMixture,
)
from hcve_lib.wrapped_sksurv import (
    DFSurvivalGradientBoosting,
    DFSurvivalStacking,
    DFSurvivalTree,
    DFRandomSurvivalForest,
    DFCoxnetSurvivalAnalysis,
)


class EstimatorDecorator:
    def __init__(self, estimator):
        self._estimator = estimator

    def get_params(self, *args, **kwargs):
        return self._estimator.get_params(*args, **kwargs)

    def __getattr__(self, item):
        if item == "_estimator":
            return getattr(self, "_estimator")
        else:
            return getattr(self._estimator, item)

    def __setattr__(self, attr_name, attr_value):
        if attr_name == "_estimator":
            super().__setattr__("_estimator", attr_value)
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
    return prepend_timeline(
        pipeline, ("subsample", FunctionTransformer(subsample_data))
    )


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
            print("transform", X)
        if self.breakpoint_transform:
            breakpoint()
        return X

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X, y)
        else:
            print("fit", X)
        if self.breakpoint_fit:
            breakpoint()
        return self


class LifeTime(EstimatorDecorator, BaseEstimator):
    def fit(self, X, y, *args, **kwargs):
        y_df = y["data"].copy()
        y_df["tte"] += X["AGE"] * 365
        y_transformed = {**y, "data": y_df}
        self._estimator.fit(X, y_transformed)
        return self

    def predict_survival_function(
        self,
        X: DataFrame,
    ) -> Callable[[int], float]:
        survival_functions = self._estimator.predict_survival_function(X)
        return (
            partial(self.add_age, fn, age)
            for age, fn in zip(X["AGE"], survival_functions)
        )

    def add_age(self, survival_function, age: Series, t: int):
        try:
            return survival_function(t + age * 365)
        except ValueError as e:
            return ExceptionValue(e)


class PredictionMethod(Estimator):
    _estimator: Any
    params: Dict
    target_type: TargetType

    def __init__(
        self,
        random_state: int,
        X: DataFrame,
        logger: Logger = None,
        log_mlflow: bool = True,
        target_type: TargetType = TargetType.NA,
        verbose: int = None,
        **kwargs,
    ):
        self.random_state = random_state
        self.logger = logger
        self.log_mlflow = log_mlflow
        self.target_type = target_type
        self.verbose = verbose
        self.kwargs = kwargs
        self._estimator = self.get_estimator_(X)
        self.params = {}

    def fit(self, X: DataFrame, y: Any, *args, **kwargs):
        self._estimator = self.get_estimator_(X)
        self._estimator.set_params(**self.params)
        self._estimator.fit(X, y, *args, **kwargs)
        self.target_type = get_target_type(y)
        return self

    def transform(self, X: DataFrame):
        return X

    def predict(self, X: DataFrame, *args, **kwargs):
        if self.target_type == TargetType.CLASSIFICATION:
            y_pred: DataFrame = self._estimator.predict_proba(X, *args, **kwargs)

            # HACK for two class prediction
            if isinstance(y_pred, Series):
                y_pred = DataFrame({0: 1 - y_pred, 1: y_pred})
            elif len(y_pred.columns) == 1:
                y_pred[1] = 1 - y_pred[0]

            return y_pred

        elif self.target_type == TargetType.REGRESSION:
            return self._estimator.predict(X)
        elif self.target_type == TargetType.TIME_TO_EVENT:
            return self._estimator.predict(X)

    def predict_proba_table(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_estimator(self, X: DataFrame = None) -> Union[Any, Iterable[Any]]:
        raise NotImplementedError

    def get_estimator_(self, X: DataFrame = None) -> Any:
        try:
            return self.get_estimator(X).set_params(
                **{
                    **self.kwargs,
                    **({"verbose": self.verbose} if self.verbose is not None else {}),
                },
            )
        except ValueError:
            # verbose not accepted
            return self.get_estimator(X).set_params(
                **self.kwargs,
            )

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        return trial, {}

    def get_feature_importance(self) -> Series:
        if not self._estimator:
            raise Exception("Must be fit")
        else:
            return self._estimator.get_feature_importance()

    def get_p_value_feature_importance(self, X: DataFrame, y: Target) -> Series:
        raise NotImplementedError

    def set_params(self, **kwargs):
        self.params = kwargs
        self._estimator.set_params(**kwargs)

    def get_params(self, **kwargs):
        return self._estimator.get_params(**kwargs)

    def __getattr__(self, item):
        if item == "estimator" or item == "_estimator":
            if "_estimator" in self.__dict__:
                return getattr(self, "_estimator")
            else:
                raise AttributeError()
        elif hasattr(self._estimator, item):
            return getattr(self._estimator, item)
        else:
            raise AttributeError(f"AttributeError: object has no attribute '{item}'")

    def __getstate__(self):
        return self.__dict__

    def __getitem__(self, item):
        return self._estimator[item]


class XGBoost(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        if self.target_type == TargetType.TIME_TO_EVENT:
            hyperparameters = {
                # "aft_loss_distribution": trial.suggest_categorical(
                #     "aft_loss_distribution", ["normal", "logistic", "extreme"]
                # ),
                # "aft_loss_distribution_scale": trial.suggest_loguniform(
                #     "aft_loss_distribution_scale", 0.1, 10.0
                # ),
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
                "learning_rate": trial.suggest_float(
                    f"{prefix}_learning_rate", 0.001, 1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 1, 8),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
                "subsample": trial.suggest_float(
                    f"{prefix}_estimator_subsample", 0.1, 1
                ),
                # "colsample_bytree": trial.suggest_float(
                #     f"{prefix}_colsample_bytree", 0.1, 1
                # ),
                # "min_split_loss": trial.suggest_float(
                #     f"{prefix}_min_split_loss", 0.1, 10
                # ),
                # "min_child_weight": trial.suggest_int(
                #     f"{prefix}_min_child_weight", 1, 100
                # ),
            }
        else:
            hyperparameters = {
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "learning_rate": trial.suggest_float(
                    f"{prefix}_learning_rate", 0.001, 1, log=True
                ),
                "subsample": trial.suggest_float(
                    f"{prefix}_estimator_subsample", 0.1, 1
                ),
                "colsample_bytree": trial.suggest_float(
                    f"{prefix}_colsample_bytree", 0.1, 1
                ),
                "min_split_loss": trial.suggest_float(
                    f"{prefix}_min_split_loss", 0.1, 10
                ),
                "min_child_weight": trial.suggest_int(
                    f"{prefix}_min_child_weight", 1, 100
                ),
                "reg_alpha": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
            }

        return trial, hyperparameters

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}

        new_config["n_estimators"] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=2000,
            value=current_config.get("n_estimators", 100),
            key="n_estimators",
        )

        max_depth = st.select_slider(
            "Tree depth (max_depth)",
            [
                *range(1, 20),
                "Unlimited",
            ],
            value=current_config.get("max_depth", "Unlimited"),
        )

        new_config["max_depth"] = None if max_depth == "Unlimited" else max_depth

        new_config["learning_rate"] = st.select_slider(
            "Learning rate",
            np.arange(0.1, 1, 0.1),
            value=current_config.get("learning_rate", 0.8),
        )

        new_config["subsample"] = st.select_slider(
            "Fraction of samples for each XGB tree (subsample)",
            np.arange(0.1, 1, 0.1),
            value=current_config.get("subsample", 0.8),
        )

        new_config["colsample_bytree"] = st.select_slider(
            "Fraction of variables for each XGB tree (colsample_bytree)",
            np.arange(0.1, 1, 0.1),
            value=current_config.get("colsample_bytree", 0.8),
        )

        return new_config

    def get_estimator(self, X=None) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            # return DFXGBRegressor(
            #     random_state=self.random_state, seed=self.random_state
            # )
            pass
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFXGBClassifier(
                random_state=self.random_state,
                seed=self.random_state,
                # enable_categorical=True,
            )
        elif self.target_type == TargetType.TIME_TO_EVENT:
            return DFSurvivalXGB(
                random_state=self.random_state,
                seed=self.random_state,
            )
        else:
            raise NotImplementedError


class GaussianMixture(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "n_components": trial.suggest_int(f"{prefix}_n_components", 1, 10),
            "covariance_type": trial.suggest_categorical(
                f"{prefix}_covariance_type", ["full", "tied", "diag", "spherical"]
            ),
        }
        return trial, hyperparameters

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}
        new_config["n_components"] = st.slider(
            "Number of components (n_components)",
            min_value=1,
            max_value=10,
            value=current_config.get("n_components", 1),
            key="n_components",
        )
        new_config["covariance_type"] = st.selectbox(
            "Covariance type (covariance_type)",
            ["full", "tied", "diag", "spherical"],
            index=["full", "tied", "diag", "spherical"].index(
                current_config.get("covariance_type", "full")
            ),
        )
        return new_config

    def get_estimator(self, X=None):
        return DFGaussianMixture(
            n_components=2, covariance_type="full", random_state=self.random_state
        )


class SurvivalStacking(PredictionMethod):
    def get_estimator(self, X=None) -> Estimator:
        return DFSurvivalStacking(
            meta_estimator=self.get_meta_learner(X),
            base_estimators=self.get_base_learners(X),
        )

    def get_meta_learner(self, X: DataFrame) -> DFPipeline:
        return DFPipeline(
            [
                # *_get_standard_scaler(),
                (
                    "estimator",
                    # LinearModel(
                    #     X=X,
                    #     random_state=self.random_state * 1000,
                    #     target_type=TargetType.TIME_TO_EVENT,
                    #     fit_baseline_model=True,
                    # ),
                    SurvivalTree(random_state=self.random_state, X=X),
                ),
            ]
        )

    def get_base_learners(self, X: DataFrame):
        categorical, continuous = estimate_categorical_and_continuous_columns(X)
        return [
            (
                "gm",
                DFPipeline(
                    [
                        *_get_one_hot(categorical),
                        *_get_standard_scaler(),
                        (
                            "estimator",
                            GaussianMixture(
                                X=X,
                                random_state=self.random_state,
                                n_components=5,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "rf",
                DFPipeline(
                    [
                        (
                            "estimator",
                            SurvivalRandomForest(
                                X=X,
                                random_state=self.random_state,
                                n_estimators=500,
                                n_jobs=-1,
                                verbose=5,
                                max_depth=3,
                            ),
                        )
                    ]
                ),
            ),
            (
                "pcp",
                DFPipeline(
                    [
                        (
                            "estimator",
                            PooledCohort(X=X, random_state=self.random_state),
                        ),
                    ]
                ),
            ),
            (
                "cn",
                DFPipeline(
                    [
                        *_get_one_hot(categorical),
                        *_get_standard_scaler(),
                        (
                            "estimator",
                            LinearModel(
                                X=X,
                                random_state=self.random_state,
                                target_type=TargetType.TIME_TO_EVENT,
                                fit_baseline_model=True,
                            ),
                        ),
                    ]
                ),
            ),
        ]

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, *args, **kwargs
    ) -> Tuple[Trial, Dict]:
        _, hyperparameters_meta = self.get_meta_learner(X).suggest_optuna(
            trial, X, "meta_learning"
        )

        hyperparameters = {
            **hyperparameters_meta,
            "base_estimators": {},
        }

        for name, model in self.get_base_learners(X):
            _, hyperparameters["base_estimators"][name] = model.suggest_optuna(trial, X)
        print("HYPERPARAMETERS", hyperparameters)
        return trial, hyperparameters


class Ensemble(PredictionMethod):
    def get_estimator(self, X=None) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return StackingRegressor(
                [
                    (
                        "rf",
                        DFRandomForestRegressor(
                            random_state=self.random_state, n_estimators=500
                        ),
                    ),
                    ("lr", DFElasticNet(random_state=self.random_state, max_iter=1000)),
                ]
            )
        elif self.target_type == TargetType.CLASSIFICATION:
            return StackingClassifier(
                [
                    (
                        "rf",
                        DFRandomForestClassifier(
                            random_state=self.random_state, n_estimators=500
                        ),
                    ),
                    (
                        "lr",
                        DFLogisticRegression(
                            random_state=self.random_state, max_iter=1000
                        ),
                    ),
                ]
            )
        elif self.target_type == TargetType.TIME_TO_EVENT:
            raise NotImplementedError
        else:
            raise NotImplementedError


class XGBSEBase(PredictionMethod):
    def fit(self, X: DataFrame, y: TargetData, *args, **kwargs):
        from xgbse.converters import convert_to_structured

        self._estimator.fit(
            X, convert_to_structured(y["tte"], y["label"]), *args, **kwargs
        )

    def predict(self, X: DataFrame, *args, **kwargs) -> Series:
        y_pred = self._estimator.predict(X)
        y_pred_ = -Series(y_pred.mean(axis=1))
        y_pred_.index = X.index
        return y_pred_


class XGBSEKNN(XGBSEBase):
    def get_estimator(self, X=None):
        from xgbse import XGBSEKaplanNeighbors
        from xgbse._kaplan_neighbors import DEFAULT_PARAMS as KNN_DEFAULT_PARAMS

        # TODO: XGB 2.
        # return XGBSEKaplanNeighbors(xgb_params=KNN_DEFAULT_PARAMS | dict(device="cuda"))
        return XGBSEKaplanNeighbors(
            xgb_params=KNN_DEFAULT_PARAMS | dict(tree_method="gpu_hist")
        )

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        from xgbse._kaplan_neighbors import DEFAULT_PARAMS as KNN_DEFAULT_PARAMS

        hyperparameters = {
            "n_neighbors": trial.suggest_int(f"{prefix}_n_neighbors", 1, 100),
            "xgb_params": KNN_DEFAULT_PARAMS
            | {
                # "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "learning_rate": trial.suggest_float(
                    f"{prefix}_learning_rate", 0.001, 1, log=True
                ),
                "subsample": trial.suggest_float(
                    f"{prefix}_estimator_subsample", 0.1, 1
                ),
                "colsample_bynode": trial.suggest_float(
                    f"{prefix}_colsample_bytree", 0.1, 1
                ),
                # "min_split_loss": trial.suggest_float(f"{prefix}_min_split_loss", 0.1, 10),
                # "min_child_weight": trial.suggest_int(f"{prefix}_min_child_weight", 1, 100),
                # "reg_alpha": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
                # "reg_lambda": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
            },
        }
        return trial, hyperparameters


class SurvivalGradientBoosting(PredictionMethod):
    def get_estimator_(self, X: DataFrame = None) -> Any:
        return DFSurvivalGradientBoosting(verbose=1, random_state=self.random_state)

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "learning_rate": trial.suggest_uniform(f"{prefix}learning_rate", 0, 1),
            "max_depth": trial.suggest_int(f"{prefix}max_depth", 1, 10),
            "n_estimators": trial.suggest_int(f"{prefix}n_estimators", 5, 200),
            "min_samples_split": trial.suggest_int(f"{prefix}min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int(f"{prefix}min_samples_leaf", 1, 200),
            "max_features": trial.suggest_discrete_uniform(
                f"{prefix}max_features", 0.1, 1, 0.1
            ),
            "subsample": trial.suggest_uniform(f"{prefix}subsample", 0.1, 1),
        }
        return trial, hyperparameters


class XGBSEBCE(XGBSEBase):
    def get_estimator(self, X=None):
        from xgbse import XGBSEKaplanNeighbors

        return XGBSEKaplanNeighbors()

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        from xgbse._kaplan_neighbors import DEFAULT_PARAMS as KNN_DEFAULT_PARAMS
        from xgbse._debiased_bce import DEFAULT_PARAMS_LR

        hyperparameters = {
            "lr_params": {
                **DEFAULT_PARAMS_LR,
                "C": trial.suggest_float(f"{prefix}_C", 0.01, 10**3, log=True),
            },
            "xgb_params": KNN_DEFAULT_PARAMS
            | {
                # "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "learning_rate": trial.suggest_float(
                    f"{prefix}_learning_rate", 0.001, 1, log=True
                ),
                "subsample": trial.suggest_float(
                    f"{prefix}_estimator_subsample", 0.1, 1
                ),
                "colsample_bynode": trial.suggest_float(
                    f"{prefix}_colsample_bytree", 0.1, 1
                ),
                # "min_split_loss": trial.suggest_float(f"{prefix}_min_split_loss", 0.1, 10),
                # "min_child_weight": trial.suggest_int(f"{prefix}_min_child_weight", 1, 100),
                # "reg_alpha": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
                # "reg_lambda": trial.suggest_float(f"{prefix}_reg_alpha", 0, 10),
            },
        }
        return trial, hyperparameters


class Federated(BaseEstimator):
    models: List[Estimator]
    params: Dict

    def __init__(
        self,
        group_by: Series,
        random_state: int,
        start_server: Callable,
        start_client: Callable,
        target_type: TargetType = None,
        X_all: DataFrame = None,
        y_all: Target = None,
        *args,
        **kwargs,
    ):
        self.group_by = group_by
        self.random_state = random_state
        self.start_server = start_server
        self.start_client = start_client
        self.X_all = X_all
        self.y_all = y_all
        self.args = args
        self.kwargs = kwargs
        self.params = {
            "server": {
                "num_rounds": 5,
            }
        }

    def get_name(self):
        return "Federated"

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "server": {
                "num_rounds": trial.suggest_int(f"{prefix}_num_rounds", 1, 100),
            },
            "clients": {},
        }
        for cohort_number in range(len(self.group_by.unique())):
            hyperparameters["clients"][cohort_number] = {
                "num_local_rounds": trial.suggest_int(
                    f"{prefix}_num_local_rounds_{cohort_number}", 1, 10
                ),
                # "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
                "max_depth": trial.suggest_int(
                    f"{prefix}_max_depth_{cohort_number}", 1, 10
                ),
                "learning_rate": trial.suggest_float(
                    f"{prefix}_learning_rate_{cohort_number}", 0.001, 1, log=True
                ),
                "subsample": trial.suggest_float(
                    f"{prefix}_estimator_subsample_{cohort_number}", 0.1, 1
                ),
                "colsample_bytree": trial.suggest_float(
                    f"{prefix}_colsample_bytree_{cohort_number}", 0.1, 1
                ),
                "min_split_loss": trial.suggest_float(
                    f"{prefix}_min_split_loss_{cohort_number}", 0.1, 10
                ),
                "min_child_weight": trial.suggest_int(
                    f"{prefix}_min_child_weight_{cohort_number}", 1, 100
                ),
                "reg_alpha": trial.suggest_float(
                    f"{prefix}_reg_alpha_{cohort_number}", 0, 10
                ),
                "reg_lambda": trial.suggest_float(
                    f"{prefix}_reg_alpha_{cohort_number}", 0, 10
                ),
            }
        return trial, hyperparameters

    def set_params(self, **kwargs):
        self.params = toolz.merge(self.params, kwargs)

    def get_params(self, deep=True):
        return toolz.merge(DFXGBClassifier().get_params(), self.params)

    def fit(
        self, X: DataFrame, y, X_validate: DataFrame = None, y_validate: Target = None
    ):
        if self.X_all is not None:
            # complement of X
            X_validate = self.X_all.loc[self.X_all.index.difference(X.index)][X.columns]
            y_validate = self.y_all.loc[X_validate.index]
        else:
            X_validate = None
            y_validate = None

        X_clients_train = list(X.groupby(self.group_by))
        y_clients_train = list(y.groupby(self.group_by))

        print(f"Training on {len(X_clients_train)} clients")

        server_future = ray.remote(self.start_server).remote(
            len(X_clients_train), self.params["server"]
        )

        sleep(0.1)

        futures = [
            ray.remote(self.start_client).remote(
                client_id,
                X_clients_train[client_id][1],
                y_clients_train[client_id][1],
                X_validate,
                y_validate,
                self.params["clients"][client_id],
                *self.args,
                **self.kwargs,
            )
            for client_id in range(len(X_clients_train))
        ]

        self.models = ray.get(futures)

        history = ray.get(server_future)
        print(history)

    def predict(self, X: DataFrame) -> Any:
        predictions = {}
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        print_formatted(compute_classification_scores_statistics(predictions))

        return average_classification_scores(predictions)


def evaluate_metrics_aggregation(eval_metrics):
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


class FederatedXGBoost(Estimator):
    def __init__(
        self,
        group_by: DataFrameGroupBy,
        random_state: int,
        target_type: TargetType = None,
        **kwargs,
    ):
        self.group_by = group_by
        self.random_state = random_state
        self.target_type = target_type
        self.model = None
        self.local_models = {}
        self.params = toolz.merge(
            kwargs,
            {
                "global_iterations": 1,
                "local_iterations": 10,
            },
        )

    def fit(
        self,
        X: DataFrame,
        y: Target,
        X_validate: DataFrame = None,
        y_validate: Target = None,
    ):
        local_X_y_train = {}
        self.local_models = {}

        for group_key, train_idx in self.group_by.groups.items():
            X_train, y_train = loc(train_idx, X, ignore_not_present=True), loc(
                train_idx, y, ignore_not_present=True
            )
            if len(X_train) == 0:
                continue

            local_X_y_train[group_key] = (X_train, y_train)

            self.local_models[group_key] = XGBoost(
                random_state=self.random_state,
                target_type=self.target_type,
                # n_estimators=10,
            )
            if "clients" in self.params:
                self.local_models[group_key].set_params(
                    **self.params["clients"][group_key]
                )

        global_model = None
        model = None

        for iteration_number in range(self.params["global_iterations"]):
            for group_key, model, (X_train, y_train) in zip(
                self.local_models.keys(),
                self.local_models.values(),
                local_X_y_train.values(),
            ):
                if iteration_number == 0:
                    model.fit(X_train, y_train)
                else:
                    model.estimator.booster.load_model(bytearray(global_model))
                    if "clients" in self.params:
                        local_iterations = self.params["clients"][group_key][
                            "local_iterations"
                        ]
                    else:
                        local_iterations = self.params["local_iterations"]

                    model.boost(X_train, y_train, rounds=local_iterations)

            for group_key, model in zip(
                self.local_models.keys(), self.local_models.values()
            ):
                local_model = model.estimator.booster.save_raw("json")
                local_model_bytes = bytes(local_model)
                global_model = aggregate(global_model, local_model_bytes)

            if model:
                self.model = model
                self.model.estimator.booster.load_model(bytearray(global_model))

    def predict(self, X: DataFrame):
        return self.model.predict(X)

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "clients": {},
            "global_iterations": trial.suggest_int("global_iterations", 1, 50),
        }
        for cohort_name in self.group_by.groups.keys():
            prefix = prefix + "_" + str(cohort_name)
            hyperparameters["clients"][cohort_name] = XGBoost(
                random_state=self.random_state,
                target_type=self.target_type,
            ).suggest_optuna(trial, X, prefix)[1]
            hyperparameters["clients"][cohort_name]["local_iterations"] = (
                trial.suggest_int(f"{prefix}_local_iterations_{cohort_name}", 1, 50)
            )

        return trial, hyperparameters

    def set_params(self, **kwargs):
        self.params = kwargs


def aggregate(
    bst_prev_org: Optional[bytes],
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


class _FederatedXGBoost(DFWrapped, Estimator):
    booster: Booster = None
    X_train: DataFrame
    y_train: Target
    sample_weights: Series = None

    def __init__(
        self,
        hyperparameters: Dict = None,
        X_validate: DataFrame = None,
        y_validate: Target = None,
        sample_weights: Series = None,
    ):
        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = toolz.merge(
            hyperparameters,
            {
                "num_local_rounds": 1,
                "objective": "binary:logistic",
                "eta": 0.05,
                "max_depth": 2,
                "eval_metric": ["auc"],
                "nthread": 16,
                "num_parallel_tree": 1,
                "subsample": 1,
                "tree_method": "hist",
            },
        )

        self.X_validate = X_validate
        self.y_validate = y_validate
        self.sample_weights = sample_weights

    def fit(self, X: DataFrame, y: Target, *args, **kwargs):
        self.save_fit_features(X)
        self.X_train = X
        self.y_train = y
        train_dmatrix = xgb.DMatrix(
            X,
            y,
            enable_categorical=True,
        )
        local_models = {}

        if self.sample_weights:
            train_dmatrix.set_info(
                feature_weights=self.sample_weights.loc[X.index].to_numpy()
            )

        evals = [(train_dmatrix, "train_xgboost")]

        if self.X_validate is not None:
            evals.append(
                (
                    xgb.DMatrix(
                        self.X_validate,
                        self.y_validate,
                        enable_categorical=True,
                    ),
                    "validate",
                )
            )

        self.booster = xgb.train(
            self.booster_hyperparameters,
            train_dmatrix,
            num_boost_round=self.hyperparameters["num_local_rounds"],
            evals=evals,
        )

    def predict(self, X: DataFrame):
        test_dmatrix = xgb.DMatrix(
            X,
            enable_categorical=True,
        )

        y_pred = self.booster.predict(test_dmatrix)

        return DataFrame(
            {
                0: 1 - y_pred,
                1: y_pred,
            },
            index=X.index,
        )

    @property
    def booster_hyperparameters(self):
        return dissoc(self.hyperparameters, "num_rounds", "num_local_rounds")

    def local_boost(self):
        train_dmatrix = xgb.DMatrix(
            self.X_train,
            self.y_train,
            enable_categorical=True,
        )

        for i in range(self.hyperparameters["num_local_rounds"]):
            self.booster.update(train_dmatrix, self.booster.num_boosted_rounds())

        return self.booster[
            self.booster.num_boosted_rounds()
            - self.hyperparameters[
                "num_local_rounds"
            ] : self.booster.num_boosted_rounds()
        ]


class FederatedSurvivalXGBoost(DFWrapped, Estimator):
    booster: Booster = None
    X_train: DataFrame
    y_train: Target
    sample_weights: Series = None

    def __init__(
        self,
        hyperparameters: Dict = None,
        X_validate: DataFrame = None,
        y_validate: Target = None,
        sample_weights: Series = None,
    ):
        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = toolz.merge(
            hyperparameters,
            {
                "num_local_rounds": 1,
                "eta": 0.05,
                "nthread": 16,
                "num_parallel_tree": 1,
                "subsample": 1,
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.20,
                "learning_rate": 0.05,
                "max_depth": 2,
                "tree_method": "hist",
                # "gpu_id": "0",
            },
        )

        self.X_validate = X_validate
        self.y_validate = y_validate
        self.sample_weights = sample_weights

    def fit(self, X: DataFrame, y: Target, *args, **kwargs):
        self.save_fit_features(X)
        self.X_train = X
        self.y_train = y
        train_dmatrix = survival_X_y_to_dmatrix(X, y)

        if self.sample_weights:
            train_dmatrix.set_info(
                feature_weights=self.sample_weights.loc[X.index].to_numpy()
            )

        evals = [(train_dmatrix, "train_xgboost")]

        # if self.X_validate is not None:
        #     evals.append(
        #         (
        #             xgb.DMatrix(
        #                 self.X_validate,
        #                 self.y_validate,
        #                 enable_categorical=True,
        #             ),
        #             "validate",
        #         )
        #     )
        self.booster = xgb.train(
            self.booster_hyperparameters,
            train_dmatrix,
            num_boost_round=self.hyperparameters["num_local_rounds"],
            evals=evals,
        )

    def predict(self, X: DataFrame):
        test_dmatrix = survival_X_y_to_dmatrix(X)
        y_pred = self.booster.predict(test_dmatrix)
        return -Series(y_pred, index=X.index)

    @property
    def booster_hyperparameters(self):
        return dissoc(self.hyperparameters, "num_rounds", "num_local_rounds")

    def local_boost(self):
        train_dmatrix = survival_X_y_to_dmatrix(self.X_train, self.y_train)

        for i in range(self.hyperparameters["num_local_rounds"]):
            self.booster.update(train_dmatrix, self.booster.num_boosted_rounds())

        return self.booster[
            self.booster.num_boosted_rounds()
            - self.hyperparameters[
                "num_local_rounds"
            ] : self.booster.num_boosted_rounds()
        ]


def survival_X_y_to_dmatrix(X: DataFrame, y: Target = None):
    dtrain = DMatrix(X, enable_categorical=True)
    if y is not None:
        y_lower_bound = y["tte"].copy()
        y_upper_bound = y["tte"].copy()
        y_upper_bound[y["label"] == 0] = +np.inf
        dtrain.set_float_info("label_lower_bound", y_lower_bound.to_numpy())
        dtrain.set_float_info("label_upper_bound", y_upper_bound.to_numpy())
    return dtrain


class FederatedForest(BaseEstimator):
    def __init__(
        self,
        group_by: DataFrameGroupBy,
        random_state: int,
        n_estimators_local=100,
        n_estimators_federated=2000,
        target_type: TargetType = None,
    ):
        self.group_by = group_by
        self.random_state = random_state
        self.n_estimators_local = n_estimators_local
        self.n_estimators_federated = n_estimators_federated
        self.global_forest = None
        self.target_type = target_type

    def get_name(self):
        return "FederatedForest"

    def fit(self, X: DataFrame, y):
        total_data = len(X)
        local_best_trees_list = []

        test_idxs = {}
        local_forests = {}

        for group_key, train_idx in self.group_by.groups.items():
            X_train, y_train = loc(train_idx, X, ignore_not_present=True), loc(
                train_idx, y, ignore_not_present=True
            )

            if len(X_train) == 0:
                continue

            # subset_size = len(X_train)
            # weight_subset = int(
            #     self.n_estimators_federated
            #     * (len(y.data["label"]) / len(y.data["label"]))
            # )
            # weight_subset = int(
            #     self.n_estimators_federated * (subset_size / total_data)
            # )
            # choose_n_estimators = max(1, weight_subset)
            subset_size = len(X_train)
            weight_subset = int(self.n_estimators_federated * (len(y_train) / len(y)))
            weight_subset = int(
                self.n_estimators_federated * (subset_size / total_data)
            )

            choose_n_estimators = max(1, weight_subset)
            choose_n_estimators = 10

            print(choose_n_estimators)

            local_forest = RandomForest(
                n_estimators=choose_n_estimators,
                n_jobs=-1,
                max_depth=2,
                random_state=self.random_state,
                target_type=get_target_type(y),
            )

            local_forest.fit(X_train, y_train)
            n_outputs_ = local_forest._estimator.n_outputs_
            # local_tree_scores = [
            #     accuracy_score(y_train, tree.predict(X_train))
            #     for tree in local_forest.estimators_
            # ]
            #
            # local_best_trees = sorted(
            #     zip(local_tree_scores, local_forest.estimators_),
            #     key=lambda x: x[0],
            #     reverse=True,
            # )
            local_forests[group_key] = local_forest
            # noinspection PyUnresolvedReferences
            local_best_trees_list.extend(
                local_forest.estimators_[
                    : min(choose_n_estimators, len(local_forest._estimator.estimators_))
                ]
            )
        print("TRAINING FINISHED")
        # local_best_trees_list = sorted(
        #     local_best_trees_list, key=lambda x: x[0], reverse=True
        # )[: self.n_estimators_federated]

        self.global_forest = RandomSurvivalForest()
        self.global_forest.estimators_ = local_best_trees_list
        self.global_forest.estimator.n_estimators = len(local_best_trees_list)
        self.global_forest.estimator.estimators_ = local_best_trees_list
        self.global_forest.n_outputs_ = len(local_best_trees_list)
        brier_per_model_per_group = defaultdict(dict)

        # for key_model, model in local_forests.items():
        #     for group_key, train_idx in self.group_by.groups.items():
        #         X_train, y_train = loc(train_idx, X, ignore_not_present=True), loc(
        #             train_idx, y, ignore_not_present=True
        #         )
        #         y_pred = model.predict(X_train)
        # `
        #         _, brier_value = brier_score(
        #             to_survival_y_records(y_train.data),
        #             to_survival_y_records(y_train.data),
        #             y_pred,
        #             4 * 365,
        #         )
        #
        #         cindex, _, _, _, _ = concordance_index_ipcw(
        #             to_survival_y_records(y_train.data),
        #             to_survival_y_records(y_train.data),
        #             y_pred,
        #             4 * 365,
        #         )
        #         print(key_model + " -> " + group_key)
        #         print(f"{cindex=}")
        #         print(f"{brier_value=}")
        #         print()
        #
        #         brier_per_model_per_group[key_model][group_key] = brier_value

        return self

    def predict(self, X: DataFrame):
        if not self.global_forest:
            raise Exception("The model has not been fitted yet.")

        survs = [
            [
                # -surv_func(x[1]["AGE"] + 1200)
                -surv_func(1200)
                for x, surv_func in zip(X.iterrows(), tree.predict_survival_function(X))
            ]
            for tree in self.global_forest.estimators_
        ]

        survs = [mean(surv_individual) for surv_individual in transpose(survs)]

        return Series(
            survs,
            index=X.index,
        )


class ExtraTrees(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        if (
            self.target_type == TargetType.REGRESSION
            or self.target_type == TargetType.CLASSIFICATION
        ):
            hyperparameters = {
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "min_samples_split": trial.suggest_int(
                    f"{prefix}_min_samples_split", 2, 100
                ),
                "max_features": trial.suggest_categorical(
                    f"{prefix}_max_features", ["sqrt", "log2", *range(1, 50)]
                ),
            }
        elif self.target_type == TargetType.TIME_TO_EVENT:
            hyperparameters = {
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "min_samples_split": trial.suggest_int(
                    f"{prefix}_min_samples_split", 2, 100
                ),
                "max_features": trial.suggest_categorical(
                    f"{prefix}_max_features",
                    ["sqrt", "log2", *range(1, len(X.columns))],
                ),
            }
        else:
            raise NotImplementedError

        return trial, hyperparameters

    def get_estimator(self, X=None):
        return DFExtraTreesClassifier(random_state=self.random_state)


class RegressionRandomForest(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
            "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
            "min_samples_split": trial.suggest_int(
                f"{prefix}_min_samples_split", 2, 100
            ),
            "max_features": trial.suggest_categorical(
                f"{prefix}_max_features", ["sqrt", "log2", *range(1, 50)]
            ),
        }
        return trial, hyperparameters

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}
        new_config["n_estimators"] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=5000,
            value=current_config.get("n_estimators", 100),
            key="n_estimators",
        )
        max_depth = st.select_slider(
            "Tree depth (max_depth)",
            [*range(1, 20), "Unlimited"],
            value=current_config.get("max_depth", "Unlimited"),
        )
        new_config["max_depth"] = None if max_depth == "Unlimited" else max_depth
        new_config["min_samples_split"] = st.select_slider(
            "Minimum sample for decision (min_samples_split)",
            [*range(1, 50)],
            value=current_config.get("min_samples_split", 2),
        )
        new_config["max_features"] = st.select_slider(
            "Subset of features for decision (max_features)",
            ["log2", "sqrt", *range(1, 100)],
            value=current_config.get("max_features", "sqrt"),
        )
        return new_config

    def get_estimator(self, X=None):
        from sklearn.ensemble import RandomForestRegressor as DFRandomForestRegressor

        return DFRandomForestRegressor(
            random_state=self.random_state, n_estimators=100, n_jobs=-1
        )


class ClassificationRandomForest(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
            "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
            "min_samples_split": trial.suggest_int(
                f"{prefix}_min_samples_split", 2, 100
            ),
            "max_features": trial.suggest_categorical(
                f"{prefix}_max_features", ["sqrt", "log2", *range(1, 50)]
            ),
        }
        return trial, hyperparameters

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}
        new_config["n_estimators"] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=5000,
            value=current_config.get("n_estimators", 100),
            key="n_estimators",
        )
        max_depth = st.select_slider(
            "Tree depth (max_depth)",
            [*range(1, 20), "Unlimited"],
            value=current_config.get("max_depth", "Unlimited"),
        )
        new_config["max_depth"] = None if max_depth == "Unlimited" else max_depth
        new_config["min_samples_split"] = st.select_slider(
            "Minimum sample for decision (min_samples_split)",
            [*range(1, 50)],
            value=current_config.get("min_samples_split", 2),
        )
        new_config["max_features"] = st.select_slider(
            "Subset of features for decision (max_features)",
            ["log2", "sqrt", *range(1, 100)],
            value=current_config.get("max_features", "sqrt"),
        )
        return new_config

    def get_estimator(self, X=None):
        from sklearn.ensemble import RandomForestClassifier as DFRandomForestClassifier

        return DFRandomForestClassifier(
            random_state=self.random_state, n_estimators=100, n_jobs=-1
        )


class SurvivalTree(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
            "min_samples_split": trial.suggest_int(
                f"{prefix}_min_samples_split", 2, 100
            ),
            "max_features": trial.suggest_discrete_uniform(
                f"{prefix}max_features", 0.1, 1, 0.1
            ),
        }
        return trial, hyperparameters

    def get_estimator(self, X=None):
        return DFSurvivalTree(
            random_state=self.random_state,
            max_depth=3,
            low_memory=True,
        )


class SurvivalRandomForest(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 1000),
            "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 5),
            "min_samples_split": trial.suggest_int(
                f"{prefix}_min_samples_split", 2, 100
            ),
            "max_features": trial.suggest_discrete_uniform(
                f"{prefix}max_features", 0.1, 1, 0.1
            ),
        }
        return trial, hyperparameters

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}
        new_config["n_estimators"] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=1000,
            value=current_config.get("n_estimators", 100),
            key="n_estimators",
        )
        print("n_estimators", new_config["n_estimators"])
        max_depth = st.select_slider(
            "Tree depth (max_depth)",
            [*range(1, 10)],
            value=current_config.get("max_depth", "Unlimited"),
        )
        new_config["max_depth"] = None if max_depth == "Unlimited" else max_depth
        new_config["min_samples_split"] = st.select_slider(
            "Minimum sample for decision (min_samples_split)",
            [*range(1, 50)],
            value=current_config.get("min_samples_split", 2),
        )
        new_config["max_features"] = st.select_slider(
            "Subset of features for decision (max_features)",
            ["log2", "sqrt", *range(1, 100)],
            value=current_config.get("max_features", "sqrt"),
        )
        return new_config

    def get_estimator(self, X=None):
        from hcve_lib.wrapped_sksurv import DFRandomSurvivalForest

        return DFRandomSurvivalForest(
            random_state=self.random_state,
            n_estimators=500,
            n_jobs=-1,
            verbose=5,
            max_depth=3,
            low_memory=True,
        )


class RandomForest(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        if (
            self.target_type == TargetType.REGRESSION
            or self.target_type == TargetType.CLASSIFICATION
        ):
            hyperparameters = {
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "min_samples_split": trial.suggest_int(
                    f"{prefix}_min_samples_split", 2, 100
                ),
                "max_features": trial.suggest_categorical(
                    f"{prefix}_max_features", ["sqrt", "log2", *range(1, 50)]
                ),
            }
        elif self.target_type == TargetType.TIME_TO_EVENT:
            hyperparameters = {
                "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 2000),
                "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
                "min_samples_split": trial.suggest_int(
                    f"{prefix}_min_samples_split", 2, 100
                ),
                "max_features": trial.suggest_categorical(
                    f"{prefix}_max_features",
                    ["sqrt", "log2", *range(1, len(X.columns))],
                ),
            }
        else:
            raise NotImplementedError

        return trial, hyperparameters

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}

        new_config["n_estimators"] = st.slider(
            "Number trees (n_tree)",
            min_value=1,
            max_value=5000,
            value=current_config.get("n_estimators", 100),
            key="n_estimators",
        )
        max_depth = st.select_slider(
            "Tree depth (max_depth)",
            [
                *range(1, 20),
                "Unlimited",
            ],
            value=current_config.get("max_depth", "Unlimited"),
        )

        new_config["max_depth"] = None if max_depth == "Unlimited" else max_depth

        new_config["min_samples_split"] = st.select_slider(
            "Minimum sample for decision (min_samples_split)",
            [*range(1, 50)],
            value=current_config.get("min_samples_split", 2),
        )

        new_config["max_features"] = st.select_slider(
            "Subset of features for decision (max_features)",
            ["log2", "sqrt", *range(1, 100)],
            value=current_config.get("max_features", "sqrt"),
        )
        return new_config

    def get_estimator(self, X=None) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return DFRandomForestRegressor(
                random_state=self.random_state,
                n_estimators=100,
                n_jobs=-1,
            )
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFRandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                n_jobs=-1,
            )
        elif self.target_type == TargetType.TIME_TO_EVENT:
            from hcve_lib.wrapped_sksurv import DFRandomSurvivalForest

            return DFRandomSurvivalForest(
                random_state=self.random_state,
                n_estimators=2000,
                n_jobs=-1,
                verbose=5,
                max_depth=3,
            )
        else:
            raise NotImplementedError


# class RandomForest(PredictionMethod):
#     def __init__(self, random_state: int, X: DataFrame, **kwargs):
#         super().__init__(random_state, X, **kwargs)
#         self.random_state = random_state
#         self.model = self._initialize_model()
#
#     def _initialize_model(self):
#         if self.target_type == TargetType.REGRESSION:
#             return RegressionRandomForest(self.target_type, self.random_state)
#         elif self.target_type == TargetType.CLASSIFICATION:
#             return ClassificationRandomForest(self.target_type, self.random_state)
#         elif self.target_type == TargetType.TIME_TO_EVENT:
#             return SurvivalRandomForest(self.target_type, self.random_state)
#         else:
#             raise NotImplementedError
#
#     def suggest_optuna(
#         self, trial: Trial, X: DataFrame, prefix: str = ""
#     ) -> Tuple[Trial, Dict]:
#         print("Suggesting optuna", self)
#         return self.model.suggest_optuna(trial, X, prefix)
#
#     def get_streamlit_configuration(self, current_config: Dict):
#         return self.model.get_streamlit_configuration(current_config)
#
#     def get_estimator(self, X=None):
#         return self.model.get_estimator(X)


class DeepSurv(PredictionMethod):
    def __init__(
        self,
        random_state: int,
        nodes_hidden: int = 32,
        n_layers: int = 1,
        batch_norm: bool = False,
        dropout: float = 0.1,
        output_bias: bool = False,
        batch_size: int = 10,
        learning_rate: float = 0.001,
        input_features=33,
        *args,
        **kwargs,
    ):
        self.nodes_hidden = nodes_hidden
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_bias = output_bias
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_features = input_features
        super().__init__(random_state, *args, **kwargs)

    def get_params(self, **kwargs):
        return {
            key: getattr(self, key)
            for key in (
                "nodes_hidden",
                "n_layers",
                "batch_norm",
                "dropout",
                "output_bias",
                "learning_rate",
                "batch_size",
            )
        }

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "nodes_hidden": trial.suggest_int(f"{prefix}nodes_hidden", 3, 50),
            "n_layers": trial.suggest_int(f"{prefix}n_layers", 1, 10),
            "batch_norm": trial.suggest_categorical(
                f"{prefix}batch_norm", [True, False]
            ),
            "dropout": trial.suggest_float(f"{prefix}dropout", 0, 0.5),
            "learning_rate": trial.suggest_float(
                f"{prefix}learning_rate", 10**-7, 10**-1, log=True
            ),
            "output_bias": trial.suggest_categorical(
                f"{prefix}output_bias", [True, False]
            ),
            "batch_size": trial.suggest_int(f"{prefix}batch_size", 32, 512, step=32),
        }
        return trial, hyperparameters

    def get_estimator_(self, X: DataFrame = None):
        return self.get_estimator(X)

    def fit(self, X: DataFrame, y: Target, **kwargs):
        y_ = (
            y["tte"].values.astype("float32"),
            y["label"].values.astype("float32"),
        )
        X_ = X.to_numpy().astype("float32")
        self.estimator = self.get_estimator_(X)
        self.estimator.optimizer.set_lr(self.learning_rate)
        self.estimator.fit(X_, y_, batch_size=self.batch_size)
        self.estimator.compute_baseline_hazards()
        return self

    def predict(self, X: DataFrame, **kwargs):
        X_test_ = X.to_numpy().astype("float32")
        y_pred = self.estimator.predict(X_test_)
        return y_pred
        # y_pred = self.predict_proba_table(X)
        # return 1 - y_pred.iloc[round(len(y_pred) / 2), :]

    def predict_proba_table(self, X: DataFrame) -> DataFrame:
        X_test_ = X.to_numpy().astype("float32")
        y_pred = self.estimator.predict_surv_df(X_test_)
        return y_pred

    def get_estimator(self, X: DataFrame = None) -> Any:
        import torch
        from torch.optim import Adam
        from pycox.models import CoxPH
        from torchtuples.practical import MLPVanilla

        if X is None:
            in_features = self.input_features
        else:
            in_features = len(X.columns)

        num_nodes = [self.nodes_hidden] * self.n_layers
        out_features = 1
        torch.manual_seed(self.random_state)
        net = MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            self.batch_norm,
            self.dropout,
            output_bias=self.output_bias,
        )

        model = CoxPH(net, Adam)
        # model.set_device(torch.device("cuda:0"))
        model.optimizer.set_lr(self.learning_rate)
        return model


class LinearModel(PredictionMethod):
    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        if self.target_type == TargetType.REGRESSION:
            hyperparameters = {
                "alpha": trial.suggest_float(f"{prefix}_alpha", 0.1, 100.0, log=True),
                "l1_ratio": trial.suggest_float(f"{prefix}_l1_ratio", 0, 1),
            }
        elif self.target_type == TargetType.CLASSIFICATION:
            hyperparameters = {
                "penalty": trial.suggest_categorical(
                    f"{prefix}_penalty", ["l1", "l2", "elasticnet"]
                ),
                "C": trial.suggest_float(f"{prefix}_C", 0.01, 10**3, log=True),
            }

            if hyperparameters["penalty"] == "elasticnet":
                hyperparameters["l1_ratio"] = trial.suggest_float(
                    f"{prefix}_l1_ratio", 0, 1
                )

            if hyperparameters["penalty"] in ("elasticnet", "l1"):
                hyperparameters["solver"] = "saga"
        elif self.target_type == TargetType.TIME_TO_EVENT:
            hyperparameters = {
                "l1_ratio": 1
                - trial.suggest_float(f"{prefix}_l1_ratio", 0.1, 1, log=True),
                "alphas": [
                    trial.suggest_float(f"{prefix}_alphas", 10**-2, 1, log=True)
                ],
            }
        else:
            raise NotImplementedError

        return trial, hyperparameters

    def get_estimator(self, X=None) -> Estimator:
        if self.target_type == TargetType.REGRESSION:
            return DFElasticNet(random_state=self.random_state, max_iter=1000)
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFLogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.target_type == TargetType.TIME_TO_EVENT:
            from hcve_lib.wrapped_sksurv import DFCoxnetSurvivalAnalysis

            return DFCoxnetSurvivalAnalysis(fit_baseline_model=True)
        else:
            raise NotImplementedError

    @staticmethod
    def get_streamlit_configuration(current_config: Dict):
        import streamlit as st

        new_config = {}

        log_space_alpha = list(np.logspace(-5, 5, 10))
        new_config["alpha"] = st.select_slider(
            "alpha",
            log_space_alpha,
            # value=current_config.get('alpha', 1),
            key="alpha",
        )

        new_config["l1_ratio"] = st.select_slider(
            "l1_ratio",
            [
                *np.arange(0, 1.1, 0.1),
            ],
            value=current_config.get("l1_ratio", 1.0),
        )
        return new_config


class CoxNet(PredictionMethod):
    def get_estimator(self, X=None):
        if self.target_type == TargetType.REGRESSION:
            return DFElasticNet(random_state=self.random_state, max_iter=1000)
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFLogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise NotImplementedError

    def suggest_optuna(self, trial: Trial, prefix: str = "") -> Tuple[Trial, Dict]:
        hyperparameters = {
            "l1_ratio": 1 - trial.suggest_loguniform(f"{prefix}_l1_ratio", 0.1, 1),
            "alphas": [trial.suggest_loguniform(f"{prefix}_alphas", 10**-2, 1)],
        }
        return trial, hyperparameters


class PooledCohort(PredictionMethod):
    def get_estimator(self, X=None):
        return PooledCohort_()

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {}
        return trial, hyperparameters

    def predict(self, X: DataFrame, **kwargs):
        return self._estimator.predict(X)


class PooledCohort_(BaseEstimator, ClassifierMixin):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        X_ = X.copy()

        def predict_(row: Series) -> float:
            # TODO
            # if row['AGE'] < 30:
            #     return np.nan
            bsug_adj = row["GLU"] * 18.018
            row["CHOL"] = row["CHOL"] * 38.67
            row["HDL"] = row["HDL"] * 38.67
            bmi = row["BMI"]
            sex = row["SEX"]
            if "qrs" in row:
                qrs = row["QRS"]
            else:
                qrs = 94.03636286848551

            trt_ht = row["TRT_AH"]

            if row["SMK"] == 0 or row["SMK"] == 2:
                csmk = 0
            elif row["SMK"] == 1:
                csmk = 1
            else:
                csmk = 0

            try:
                ln_age = log(row["AGE"])
            except ValueError:
                print(row["AGE"])

            ln_age_sq = ln_age**2
            try:
                ln_tchol = log(row["CHOL"])
            except ValueError:
                print(row["CHOL"])

            try:
                ln_hchol = log(row["HDL"])
            except ValueError:
                print(row["HDL"])

            ln_sbp = log(row["SBP"])

            hdm = row["DIABETES"]
            if hdm != 0 and hdm != 1:
                hdm = 0
            ln_agesbp = ln_age * ln_sbp
            ln_agecsmk = ln_age * csmk
            ln_bsug = log(bsug_adj)
            ln_bmi = log(bmi)
            ln_agebmi = ln_age * ln_bmi
            ln_qrs = log(qrs)

            if (sex == 1) and (trt_ht == 0):
                coeff_sbp = 0.91
            if (sex == 1) and (trt_ht == 1):
                coeff_sbp = 1.03
            if (sex == 2) and (trt_ht == 0):
                coeff_sbp = 11.86
            if (sex == 2) and (trt_ht == 1):
                coeff_sbp = 12.95
            if (sex == 2) and (trt_ht == 0):
                coeff_agesbp = -2.73
            if (sex == 2) and (trt_ht == 1):
                coeff_agesbp = -2.96

            if (sex == 1) and (hdm == 0):
                coeff_bsug = 0.78
            if (sex == 1) and (hdm == 1):
                coeff_bsug = 0.90
            if (sex == 2) and (hdm == 0):
                coeff_bsug = 0.91
            if (sex == 2) and (hdm == 1):
                coeff_bsug = 1.04

            if sex == 1:
                IndSum = (
                    ln_age * 41.94
                    + ln_age_sq * (-0.88)
                    + ln_sbp * coeff_sbp
                    + csmk * 0.74
                    + ln_bsug * coeff_bsug
                    + ln_tchol * 0.49
                    + ln_hchol * (-0.44)
                    + ln_bmi * 37.2
                    + ln_agebmi * (-8.83)
                    + ln_qrs * 0.63
                )

                HF_risk = 100 * (1 - (0.98752 ** exp(IndSum - 171.5)))
            elif sex == 2:
                IndSum = (
                    ln_age * 20.55
                    + ln_sbp * coeff_sbp
                    + ln_agesbp * coeff_agesbp
                    + csmk * 11.02
                    + ln_agecsmk * (-2.50)
                    + ln_bsug * coeff_bsug
                    + ln_hchol * (-0.07)
                    + ln_bmi * 1.33
                    + ln_qrs * 1.06
                )
                HF_risk = 100 * (1 - (0.99348 ** exp(IndSum - 99.73)))

            return HF_risk

        return X.apply(predict_, axis=1)

    def set_params(self, **params):
        return self


class RepeatedEnsemble(Estimator):
    def __init__(
        self,
        get_pipeline: Callable,
        n_repeats: int = 10,
        random_state: int = None,
        bootstrap: bool = False,
    ):
        self.get_pipeline = get_pipeline
        self.n_repeats = n_repeats
        self.params = {}
        self.random_state = random_state
        self.estimators = []
        for repeat in range(self.n_repeats):
            self.estimators.append(
                self.get_pipeline(random_state=self.random_state + (repeat * 10000))
            )
        print(self.estimators)

    def get_estimator(self, X: DataFrame) -> Estimator:
        return self.get_pipeline(random_state=self.random_state)

    def fit(self, X, y, *args, **kwargs):
        self.estimators = []
        for repeat in range(self.n_repeats):
            _random_state = self.random_state + (repeat * 10000)
            X_train = X.sample(frac=1, replace=True)
            y_train = y.loc[X_train.index]
            pipeline = self.get_pipeline(random_state=_random_state, X=X)
            pipeline.set_params(**self.params)
            pipeline.fit(X_train, y_train)
            self.estimators.append(pipeline)

    def transform(self, X: DataFrame):
        return self.estimators[0].transform(X)

    def predict_proba(self, X: DataFrame):
        y_probas = self.predict(X)
        y_probas_averaged = reduce(
            lambda sum_df, next_df: sum_df + next_df, y_probas
        ) / len(y_probas)
        return y_probas_averaged

    def predict(self, X: DataFrame):
        y_preds = []
        for estimator in self.estimators:
            y_preds.append(estimator.predict(X))

        y_preds_averaged = reduce(
            lambda sum_df, next_df: sum_df + next_df, y_preds
        ) / len(y_preds)

        return y_preds_averaged

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        return self.estimators[0].suggest_optuna(trial)

    def set_params(self, **kwargs):
        self.params = kwargs
        for estimator in self.estimators:
            estimator.set_params(**self.params)

    def get_params(self, **kwargs):
        if self.params is not None:
            return self.params
        else:
            return self.get_estimator().get_params(**kwargs)

    def get_feature_importance_per_repeat(self) -> DataFrame:
        feature_importances = []
        for estimator in self.estimators:
            feature_importances.append(estimator.get_feature_importance())
        return pandas.concat(feature_importances, axis=1)

    def get_feature_importance(self) -> DataFrame:
        return (
            self.get_feature_importance_per_repeat()
            .mean(axis=1)
            .sort_values(ascending=False)
        )

    def get_p_value_feature_importance(self, X: DataFrame, y: Target) -> Series:
        feature_importances = []
        for estimator in self.estimators:
            feature_importances.append(estimator.get_p_value_feature_importance(X, y))
        return pandas.concat(feature_importances, axis=1)

    def __getattr__(self, item):
        if hasattr(self.estimators[0], item):
            return getattr(self.estimators[0], item)
        else:
            raise AttributeError(f"AttributeError: object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.estimator[item]

    # TODO
    def get_final(self):
        return self.estimators[0]

    def get_name(self):
        return f"Repeated({self.estimators[0].get_name()})"


def get_target_type(y: Target) -> TargetType:
    if isinstance(y, TargetObject) and "tte" in y.data.columns:
        return TargetType.TIME_TO_EVENT
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

    for (
        repeat_n,
        result,
    ) in enumerate(results_):
        for split_name, prediction in result.items():
            model_output[f"{repeat_n}_{split_name}"] = callback(prediction)

    return pandas.concat(model_output, axis=1)


def get_results_feature_importance(results: Union[List[Result], Result]) -> DataFrame:
    return aggregate_results(
        results, lambda prediction: prediction["model"].get_feature_importance()
    ).copy()


def get_results_p_value_feature_importance(
    results: Union[List[Result], Result], X: DataFrame, y: Target
) -> DataFrame:
    return aggregate_results(
        results,
        lambda prediction: prediction["model"].get_p_value_feature_importance(X, y),
    )


def get_imputation(X: DataFrame, estimator) -> List[Tuple[str, Any]]:

    categorical = estimate_continuous_columns(X)
    continuous = list(set(X.columns) - set(categorical))

    return _get_imputation(categorical, continuous)


def _get_imputation(categorical, continuous):
    return [
        (
            "impute",
            DFColumnTransformer(
                [
                    (
                        "categorical",
                        DFSimpleImputer(strategy="most_frequent"),
                        categorical,
                    ),
                    ("continuous", DFSimpleImputer(strategy="mean"), continuous),
                ],
            ),
        ),
        ("remove_prefix", FunctionTransformer(remove_column_prefix)),
    ]


def _get_one_hot(categorical):
    return [
        (
            "encode",
            DFColumnTransformer(
                [("categorical", DFOneHotEncoder(), categorical)],
                remainder="passthrough",
            ),
        ),
        ("remove_prefix2", FunctionTransformer(remove_column_prefix)),
    ]


def _get_standard_scaler():
    return [("scaler", DFStandardScaler())]


def get_supervised_pipeline(
    X: DataFrame,
    y: Target,
    random_state: int,
    get_estimator: Callable[..., Estimator],
) -> DFPipeline:

    categorical = estimate_continuous_columns(X)
    continuous = list(set(X.columns) - set(categorical))

    return DFPipeline(
        [
            *_get_imputation(categorical, continuous),
            *_get_one_hot(categorical),
            *_get_standard_scaler(),
            (
                "estimator",
                get_estimator(
                    target_type=get_target_type(y), random_state=random_state
                ),
            ),
        ]
    )


def get_imputation_pipeline(
    X: DataFrame,
    y: Target,
    random_state: int,
    get_estimator: Callable[..., Estimator],
) -> DFPipeline:

    categorical = estimate_continuous_columns(X)
    continuous = list(set(X.columns) - set(categorical))

    return DFPipeline(
        [
            *_get_imputation(categorical, continuous),
            (
                "estimator",
                get_estimator(
                    X=X, target_type=get_target_type(y), random_state=random_state
                ),
            ),
        ]
    )


def get_simple_supervised_pipeline(
    X: DataFrame,
    y: Target,
    random_state: int,
    get_estimator: Callable[..., Estimator],
) -> DFPipeline:
    categorical = estimate_continuous_columns(X)
    continuous = list(set(X.columns) - set(categorical))

    return DFPipeline(
        [
            (
                "impute",
                DFColumnTransformer(
                    [
                        (
                            "categorical",
                            DFSimpleImputer(strategy="most_frequent"),
                            categorical,
                        ),
                        (
                            "continuous",
                            DFSimpleImputer(strategy="mean"),
                            continuous,
                        ),
                    ],
                ),
            ),
            ("remove_prefix", FunctionTransformer(remove_column_prefix)),
            (
                "encode",
                DFColumnTransformer(
                    [("categorical", DFOneHotEncoder(), categorical)],
                    remainder="passthrough",
                ),
            ),
            ("remove_prefix2", FunctionTransformer(remove_column_prefix)),
            (
                "estimator",
                get_estimator(
                    target_type=get_target_type(y), random_state=random_state
                ),
            ),
        ]
    )


class TargetBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, y):
        return self

    def transform(self, y):
        y_bin = np.zeros(len(y))
        y_bin[y >= self.threshold] = 1
        return Series(
            y_bin, index=y.index, name=y.name + f" binarized ({self.threshold})"
        ).astype("category")
