import numpy
from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture
from typing import Any, Optional, List, Tuple, Dict
import warnings
import numpy as np
import toolz
import xgboost
from optuna import Trial
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier,
    ExtraTreesClassifier,
)
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
    Binarizer,
    MinMaxScaler,
)
from sksurv.preprocessing import OneHotEncoder
from toolz import dissoc
from xgboost import XGBClassifier, XGBRegressor, XGBModel, DMatrix

from hcve_lib.custom_types import Estimator, Target
from hcve_lib.data import to_survival_y_records


class DFWrapped:
    fit_feature_names: List[str]

    def fit(self, X: DataFrame, y: Target = None, *args, **kwargs):
        print("FITTING")
        print(self)
        print(self.get_params())
        self.save_fit_features(X)

        if not isinstance(X, numpy.ndarray):
            X = DataFrame(X, columns=self.fit_feature_names)
            X = X.to_numpy()

        super().fit(X, y, *args, **kwargs)
        return self

    def fit_transform(self, X, *args, **kwargs):
        X_out = super().fit_transform(X, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)
        out = use_df_fn(X, X_out, columns=self.fit_feature_names, reuse_dtypes=False)
        return out

    def fit_predict(self, X, y, *args, **kwargs):
        print("FITTING")
        print(self)
        print(self.get_params())
        X_out = super().fit_predict(X, y, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)
        return use_df_fn(X, X_out, columns=self.fit_feature_names)

    def predict(self, X, *args, **kwargs) -> Series:
        y_pred = super().predict(X, *args, **kwargs)  # type: ignore
        if isinstance(X, DataFrame):
            return Series(y_pred, index=X.index)
        else:
            return y_pred

    def predict_proba(self, X, *args, **kwargs):
        if not hasattr(super(), "predict_proba"):
            return self.predict(X, *args, **kwargs)

        self.save_fit_features(X)
        y_proba = super().predict_proba(X, *args, **kwargs)  # type: ignore
        if isinstance(X, DataFrame):
            return DataFrame(y_proba, index=X.index)
        else:
            return X

        raise NotImplementedError

    def transform(self, X, *args, **kwargs):
        X_out = super().transform(X, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)
        out = use_df_fn(
            X, X_out, columns=self.fit_feature_names, reuse_dtypes=False
        ).copy()
        return out

    def save_fit_features(self, X):
        self.fit_feature_names = self.get_fit_features(X)

    def get_params(self, *args, **kwargs):
        try:
            return super().get_params(*args, **kwargs)
        except AttributeError:
            return {
                "random_state": None,
            }

    def get_feature_names(self):
        try:
            super().get_feature_names()
        except AttributeError:
            return self.fit_feature_names

    def get_fit_features(self, X: DataFrame, X_out: DataFrame = None):
        try:
            return X_out.columns  # type: ignore
        except AttributeError:
            try:
                return super().get_feature_names_out()  # type: ignore
            except (AttributeError, ValueError) as e:
                try:
                    return super().get_feature_names()  # type: ignore
                except (AttributeError, NotFittedError):
                    if isinstance(self, ColumnTransformer):
                        raise e
                    try:
                        return X.columns
                    except AttributeError:
                        pass


def series_count_inf(series: Series) -> int:
    distribution = np.isinf(series)
    try:
        return int(distribution[True])
    except KeyError:
        return 0


def use_df_fn(
    input_data_frame: DataFrame,
    output_data: Any,
    reuse_columns=True,
    reuse_index=True,
    reuse_dtypes=True,
    columns: Optional[List] = None,
) -> DataFrame:
    df_arguments = {}

    if reuse_columns:
        if columns is not None:
            df_arguments["columns"] = columns  # type: ignore
        else:
            df_arguments["columns"] = input_data_frame.columns

    if reuse_index:
        df_arguments["index"] = input_data_frame.index

    if isinstance(output_data, csr_matrix):
        output_data = output_data.toarray()

    dtypes = dict(
        zip(
            df_arguments["columns"],
            input_data_frame.dtypes,
        )
    )

    new_data = DataFrame(
        output_data,
        **df_arguments,
    )

    if reuse_dtypes:
        return new_data.astype(dtypes)
    else:
        return new_data


class DFColumnTransformer(DFWrapped, ColumnTransformer):
    ...
    # def fit_transform(self, X, *args, **kwargs):
    #     n_features = 0
    #     for index, transformer in enumerate(self.transformers):
    #         transformer_list = list(transformer)
    #         print(transformer_list)
    #         new_features = [column for column in transformer[2] if column in X.columns]
    #         n_features += len(new_features)
    #         transformer_list[2] = new_features
    #         self.transformers[index] = tuple(transformer_list)
    #
    #     return super().fit_transform(X, *args, **kwargs)
    #
    # def transform(self, X, *args, **kwargs):
    #     if not hasattr(self, "_name_to_fitted_passthrough"):
    #         self._name_to_fitted_passthrough = {}
    #
    #     return super().transform(X, *args, **kwargs)


class DFSimpleImputer(DFWrapped, SimpleImputer):
    def get_fit_features(self, X: DataFrame, X_out: DataFrame = None):
        return X.columns.tolist()


class DFStandardScaler(DFWrapped, StandardScaler):
    def get_fit_features(self, X: DataFrame, X_out: DataFrame = None):
        return X.columns.tolist()


class DFMinMaxScaler(DFWrapped, MinMaxScaler): ...


class DFOrdinalEncoder(DFWrapped, OrdinalEncoder): ...


class DFOneHotEncoder(DFWrapped, OneHotEncoder): ...


class DFVarianceThreshold(DFWrapped, VarianceThreshold): ...


class DFKNNImputer(DFWrapped, KNNImputer): ...


class DFXGBase(Estimator, ABC):
    def __init__(self, *args, **kwargs):
        self.instance = self.get_instance(*args, **kwargs)

    @abstractmethod
    def get_instance(self) -> XGBModel:
        pass

    def fit(self, X: DataFrame, y: Target = None, **fit_params):
        self.instance.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X: DataFrame) -> DataFrame:
        return DataFrame(self.instance.predict_proba(X), index=X.index)

    def predict(self, X: DataFrame) -> Series:
        return Series(self.instance.predict(X), index=X.index)

    def set_params(self, **params):
        self.instance.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.instance.get_params(deep=deep)


class DFSurvivalXGB(Estimator):
    def __init__(self, *args, **kwargs):
        self.booster = None
        self.params = kwargs

    def set_params(self, **params):
        self.params = toolz.merge(self.params, params)
        return self

    def get_params(self, deep=True):
        return toolz.merge(
            {
                "objective": "survival:cox",
                # "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.20,
                "learning_rate": 0.05,
                "max_depth": 2,
                "n_estimators": 100,
                "device": "cuda",
            },
            self.params,
        )

    def fit(self, X: DataFrame, y: Target = None, **fit_params):
        dtrain = get_dmatrix_cox_survival(X, y)
        # y_lower_bound = y["tte"].copy()
        # y_upper_bound = y["tte"].copy()
        # y_upper_bound[y["label"] == 0] = +np.inf
        # dtrain.set_float_info("label_lower_bound", y_lower_bound.to_numpy())
        # dtrain.set_float_info("label_upper_bound", y_upper_bound.to_numpy())
        params = self.get_params()
        self.booster = xgboost.train(
            dissoc(params, "n_estimators"),
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train")],
        )
        return self

    def boost(self, X: DataFrame, y: Target, rounds: int = None):
        if rounds is None:
            rounds = self.get_params()["n_estimators"]

        dtrain = get_dmatrix_cox_survival(X, y)

        for _ in range(rounds):
            self.booster.update(dtrain, self.booster.num_boosted_rounds())
        return self

    def predict(self, X):
        y_pred = self.booster.predict(DMatrix(X, enable_categorical=True))
        # return -y_pred
        return y_pred


def get_dmatrix_cox_survival(X, y):
    y_cox = [
        _tte if _label == 1 else -_tte for _tte, _label in zip(y["tte"], y["label"])
    ]
    dtrain = DMatrix(X, y_cox, enable_categorical=True)
    return dtrain


class DFXGBClassifier(DFXGBase):
    def get_instance(self, *args, **kwargs) -> XGBModel:
        return XGBClassifier(*args, tree_method="gpu_hist", **kwargs)


class DFXGBRegressor(DFXGBase):
    def get_instance(self, *args, **kwargs) -> XGBModel:
        return XGBRegressor(*args, **kwargs)


# noinspection PyUnresolvedReferences
# Estimator.register(DFXGBRegressor)


class DFPipeline(Pipeline, Estimator):
    y_name: Optional[str]

    def __init__(
        self,
        steps,
        *,
        memory=None,
        verbose=False,
        transform_y: Estimator = None,
        skip_optimization: bool = False,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.skip_optimization = skip_optimization
        warnings.filterwarnings("ignore", message="X has feature names")
        self.transform_y = transform_y

    def fit(self, X: DataFrame, y: Target = None, **fit_params):
        if self.transform_y:
            y_ = self.transform_y.transform(y)
        else:
            y_ = y
        self.y_name = y.name

        super().fit(X, y_)

        return self

    def transform_y(self, y: Target) -> Target:
        if self.transform_y:
            return self.transform_y.transform(y)
        else:
            return y

    def get_feature_names(self):
        return self.steps[-1][1].fit_feature_names

    def predict_survival(self, X, *args, **kwargs):
        Xt = self.transform_steps(X)
        return self.steps[-1][1].predict_survival(Xt, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        Xt = self.transform_steps(X)
        return self.steps[-1][1].predict_proba(Xt, *args, **kwargs)

    def predict_target(self, X, **predict_params):
        return self.predict(X, **predict_params)

    def transform_steps(self, Xt):
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return Xt

    def suggest_optuna_(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        if not self.skip_optimization:
            return self.suggest_optuna(trial, X, prefix)
        else:
            return trial, {}

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        prefix_ = (prefix + "_") if prefix else ""
        hyperparamaters = {
            name: step.suggest_optuna_(trial, X, prefix_)[1]
            for (name, step) in self.steps
            if hasattr(step, "suggest_optuna")
        }
        print("PIPELINE hyperparamaters", hyperparamaters)

        return trial, hyperparamaters

    def get_streamlit_configuration(self, config: Dict):
        config = {
            name: step.get_streamlit_configuration(config.get(name, {}))
            for (name, step) in self.steps
            if hasattr(step, "get_streamlit_configuration")
        }
        return config

    def __getattr__(self, item):
        if hasattr(self[-1], item):
            return getattr(self[-1], item)
        else:
            raise AttributeError(f"AttributeError: object has no attribute '{item}'")

    def get_feature_importance(self):
        return self[-1].get_feature_importance()

    def get_final(self):
        return self

    def get_name(self):
        return self[-1].get_name()


class DFLogisticRegression(DFWrapped, LogisticRegression):
    @property
    def coefficients(self):
        return Series(self.coef_[0], index=self.get_feature_names()).sort_values(
            ascending=False
        )

    def get_feature_importance(self):
        return self.coefficients


class DFElasticNet(DFWrapped, ElasticNet):
    @property
    def coefficients(self):
        return Series(self.coef_, index=self.get_feature_names()).sort_values(
            ascending=False
        )

    def get_feature_importance(self):
        return self.coefficients


class ToSurvivalRecord:
    def fit(self, X: DataFrame, y: Target):
        super().fit(X, to_survival_y_records(y))


class DFBinMapper(DFWrapped, _BinMapper): ...


class DFFunctionTransformer(DFWrapped, FunctionTransformer): ...


class DFExtraTreesClassifier(DFWrapped, ExtraTreesClassifier): ...


class DFRandomForestClassifier(DFWrapped, RandomForestClassifier):
    def get_feature_importance(self):
        return Series(super().feature_importances_, index=self.get_feature_names())


class DFRandomForestRegressor(DFWrapped, RandomForestRegressor):
    def get_feature_importance(self):
        return Series(super().feature_importances_, index=self.get_feature_names())


class DFBinarizer(DFWrapped, Binarizer): ...


class DFBaggingClassifier(DFWrapped, BaggingClassifier): ...


class DFGaussianMixture(DFWrapped, GaussianMixture):
    def fit(self, X: DataFrame, y: Target = None, *args, **kwargs):
        self.save_fit_features(X)
        GaussianMixture.fit(self, X, y)
        return self

    def fit_predict(self, X, y, *args, **kwargs):
        X_out = GaussianMixture.fit_predict(self, X, y, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)

        if self.fit_feature_names is not None:
            return Series(X_out, index=X.index)
        else:
            return X_out
