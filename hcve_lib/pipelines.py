from abc import abstractmethod
from collections import defaultdict
from copy import copy
from functools import partial, reduce
from logging import Logger
from math import log, exp
from statistics import mean
from typing import Any, Tuple, Dict, Union, Iterable
from typing import List, Callable

import numpy as np
import pandas
from optuna import Trial
from pandas import DataFrame
from pandas import Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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
from hcve_lib.utils import (
    is_numerical,
    estimate_categorical_columns,
    remove_column_prefix,
    loc,
)
from hcve_lib.wrapped_sklearn import (
    DFPipeline,
    DFRandomForestRegressor,
    DFRandomForestClassifier,
    DFLogisticRegression,
    DFColumnTransformer,
    DFSimpleImputer,
    DFOrdinalEncoder,
    DFStandardScaler,
    DFElasticNet,
    DFXGBClassifier,
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
        return (
            partial(self.add_age, fn, age)
            for age, fn in zip(X["AGE"], survival_functions)
        )


class Model(Estimator):
    _estimator: Any
    params: Dict
    target_type: TargetType

    def __init__(
            self,
            random_state: int,
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
        self._estimator = self.get_estimator_()
        self.params = {}

    def fit(self, X: DataFrame, y: Any, *args, **kwargs):
        self._estimator = self.get_estimator_()
        self._estimator.set_params(**self.params)
        self._estimator.fit(X, y, *args, **kwargs)
        # HACK
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
            return self._estimator.predict(X, *args, **kwargs)
        elif self.target_type == TargetType.TIME_TO_EVENT:
            return self._estimator.predict(X, *args, **kwargs)

    def predict_proba_table(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_estimator(self, X: DataFrame = None) -> Union[Any, Iterable[Any]]:
        raise NotImplementedError

    def get_estimator_(self, X: DataFrame = None) -> Any:
        try:
            return self.get_estimator(X).set_params(
                **{**self.kwargs, **({"verbose": self.verbose} if self.verbose is not None else {})},
            )
        except ValueError:
            # verbose not accepted
            return self.get_estimator(X).set_params(
                **self.kwargs,
            )

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

    def __getitem__(self, item):
        return self._estimator[item]


class XGBoost(Model):
    def suggest_optuna(
            self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        hyperparameters = {
            "n_estimators": trial.suggest_int(f"{prefix}_n_estimators", 5, 200),
            "max_depth": trial.suggest_int(f"{prefix}_max_depth", 1, 10),
            "learning_rate": trial.suggest_float(
                f"{prefix}_learning_rate", 0.001, 1, log=True
            ),
            "subsample": trial.suggest_float(f"{prefix}_estimator_subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float(
                f"{prefix}_colsample_bytree", 0.1, 1
            ),
            "min_split_loss": trial.suggest_float(f"{prefix}_min_split_loss", 0.1, 10),
            "min_child_weight": trial.suggest_int(f"{prefix}_min_child_weight", 1, 100),
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
                tree_method="gpu_hist",
                gpu_id=0,
                enable_categorical=True,
            )
        else:
            raise NotImplementedError


class Ensemble(Model):
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


class XGBSEBase(Model):
    def fit(self, X: DataFrame, y: TargetData, *args, **kwargs):
        from xgbse.converters import convert_to_structured
        self._estimator.fit(
            X, convert_to_structured(y["tte"], y["label"]), *args, **kwargs
        )

    def predict(self, X: DataFrame, *args, **kwargs) -> Series:
        y_pred = self._estimator.predict(X, *args, **kwargs)
        y_pred_ = -Series(y_pred.mean(axis=1))
        y_pred_.index = X.index
        return y_pred_


class XGBSEKNN(XGBSEBase):
    def get_estimator(self, X=None):
        from xgbse import XGBSEKaplanNeighbors
        from xgbse._kaplan_neighbors import DEFAULT_PARAMS as KNN_DEFAULT_PARAMS
        return XGBSEKaplanNeighbors(
            xgb_params=KNN_DEFAULT_PARAMS | dict(tree_method="gpu_hist", gpu_id=0)
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
                "C": trial.suggest_float(f"{prefix}_C", 0.01, 10 ** 3, log=True),
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


class FederatedForest(BaseEstimator):
    def __init__(
            self,
            group_by: DataFrameGroupBy,
            random_state: int,
            n_estimators_local=100,
            n_estimators_federated=500,
            target_type: TargetType = None,
    ):
        self.group_by = group_by
        self.random_state = random_state
        self.n_estimators_local = n_estimators_local
        self.n_estimators_federated = n_estimators_federated
        self.global_forest = None
        self.target_type = target_type

    def fit(self, X: DataFrame, y):
        total_data = len(X)
        local_best_trees_list = []

        test_idxs = {}
        local_forests = {}

        for group_key, train_idx in self.group_by.groups.items():
            X_train, y_train = loc(train_idx, X, ignore_not_present=True), loc(
                train_idx, y, ignore_not_present=True
            )

            subset_size = len(X_train)
            weight_subset = int(
                self.n_estimators_federated
                * (len(y.data["label"]) / len(y.data["label"]))
            )
            weight_subset = int(self.n_estimators_federated / len(self.group_by.groups))
            print(weight_subset)
            choose_n_estimators = max(1, weight_subset)
            local_forest = RandomForest(
                n_estimators=choose_n_estimators,
                n_jobs=-1,
                # max_depth=3,
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

        # local_best_trees_list = sorted(
        #     local_best_trees_list, key=lambda x: x[0], reverse=True
        # )[: self.n_estimators_federated]

        self.global_forest = copy(local_forest)
        self.global_forest._estimator.n_estimators = len(local_best_trees_list)
        self.global_forest._estimator.estimators_ = local_best_trees_list

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
        survival_functions = [
            tree.predict_survival_function(X)
            for tree in self.global_forest._estimator.estimators_
        ]
        survs = [
            mean((-fn[i](1200) for fn in survival_functions)) for i in range(len(X))
        ]
        return Series(
            survs,
            index=X.index,
        )


class RandomForest(Model):
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
                random_state=self.random_state, n_estimators=100
            )
        elif self.target_type == TargetType.CLASSIFICATION:
            return DFRandomForestClassifier(
                random_state=self.random_state, n_estimators=100
            )
        elif self.target_type == TargetType.TIME_TO_EVENT:
            from hcve_lib.wrapped_sksurv import DFRandomSurvivalForest

            return DFRandomSurvivalForest(
                random_state=self.random_state, n_estimators=100
            )
        else:
            raise NotImplementedError


class DeepSurv(Model):
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
            input_features=5,
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
                f"{prefix}learning_rate", 10 ** -7, 10 ** -1, log=True
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


class LinearModel(Model):
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
                "C": trial.suggest_float(f"{prefix}_C", 0.01, 10 ** 3, log=True),
            }

            if hyperparameters["penalty"] == "elasticnet":
                hyperparameters["l1_ratio"] = trial.suggest_float(
                    f"{prefix}_l1_ratio", 0, 1
                )

            if hyperparameters["penalty"] in ("elasticnet", "l1"):
                hyperparameters["solver"] = "saga"
        elif self.target_type == TargetType.TIME_TO_EVENT:
            hyperparameters = {
                "l1_ratio": 1 - trial.suggest_loguniform(f"{prefix}_l1_ratio", 0.1, 1),
                "alphas": [trial.suggest_loguniform(f"{prefix}_alphas", 10 ** -2, 1)],
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

            return DFCoxnetSurvivalAnalysis()
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


class CoxNet(Model):
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
            "alphas": [trial.suggest_loguniform(f"{prefix}_alphas", 10 ** -2, 1)],
        }
        return trial, hyperparameters


class RandomSurvivalForest(Model):
    def get_estimator(self, X=None):
        from hcve_lib.wrapped_sksurv import DFCoxnetSurvivalAnalysis

        return DFCoxnetSurvivalAnalysis(fit_baseline_model=True, n_alphas=1)

    def suggest_optuna(self, trial: Trial, prefix: str = "") -> Tuple[Trial, Dict]:
        hyperparameters = {
            "l1_ratio": 1 - trial.suggest_loguniform(f"{prefix}_l1_ratio", 0.1, 1),
            "alphas": [trial.suggest_loguniform(f"{prefix}_alphas", 10 ** -2, 1)],
        }
        return trial, hyperparameters


class PooledCohort(Model):
    def get_estimator(self, X=None):
        return PooledCohort_()

    def suggest_optuna(self, trial: Trial, prefix: str = "") -> Tuple[Trial, Dict]:
        hyperparameters = {}
        return trial, hyperparameters


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
            try:
                ln_age = log(row["AGE"])
            except ValueError:
                print(row["AGE"])

            ln_age_sq = ln_age ** 2
            ln_tchol = log(row["CHOL"])
            ln_hchol = log(row["HDL"])
            ln_sbp = log(row["SBP"])
            hdm = row["DIABETES"]
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
        return self.estimators[0].suggest_optuna(trial, X, prefix=prefix)

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
            (
                "encode",
                DFColumnTransformer(
                    [("categorical", DFOrdinalEncoder(), categorical)],
                    remainder="passthrough",
                ),
            ),
            ("remove_prefix2", FunctionTransformer(remove_column_prefix)),
            ("scaler", DFStandardScaler()),
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
