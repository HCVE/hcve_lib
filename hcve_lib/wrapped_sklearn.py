from optuna import Trial
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.tree import SurvivalTree
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
from sklearn.pipeline import Pipeline
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from xgboost import XGBClassifier, XGBRegressor

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
from sksurv.meta import Stacking

from hcve_lib.custom_types import Estimator, Target
from hcve_lib.data import to_survival_y_records
from hcve_lib.functional import dict_subset


class DFWrapped:
    fit_feature_names: List[str]

    def fit(self, X, y=None, *args, **kwargs):
        self.save_fit_features(X)
        super().fit(X, y, *args, **kwargs)
        return self

    def fit_transform(self, X, *args, **kwargs):
        X_out = super().fit_transform(X, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)
        out = use_df_fn(X, X_out, columns=self.fit_feature_names)
        return out

    def fit_predict(self, X, y, *args, **kwargs):
        X_out = super().fit_predict(X, y, *args, **kwargs)
        self.fit_feature_names = self.get_fit_features(X, X_out)
        return use_df_fn(X, X_out, columns=self.fit_feature_names)

    def predict(self, X, *args, **kwargs) -> Series:
        self.save_fit_features(X)
        y_pred = super().predict(X, *args, **kwargs)  # type: ignore
        return y_pred

    def predict_proba(self, X, *args, **kwargs):
        self.save_fit_features(X)
        y_proba = super().predict_proba(X, *args, **kwargs)  # type: ignore
        return DataFrame(y_proba, index=X.index)

    def get_feature_importance(self):
        raise NotImplementedError

    def transform(self, X, *args, **kwargs):
        try:
            X_out = super().transform(X, *args, **kwargs)
            self.fit_feature_names = self.get_fit_features(X, X_out)
            out = use_df_fn(X, X_out, columns=self.fit_feature_names)
            return out
        except AttributeError as e:
            return X

    def save_fit_features(self, X):
        self.fit_feature_names = self.get_fit_features(X)

    def get_params(self, *args, **kwargs):
        try:
            return super().get_params(*args, **kwargs)
        except AttributeError:
            return {
                'random_state': None,
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
                return ( \
                    super().get_feature_names_out(X.columns)  # type: ignore
                )
            except (AttributeError, ValueError) as e:
                try:
                    return super().get_feature_names()  # type: ignore
                except (AttributeError, NotFittedError):
                    if isinstance(self, ColumnTransformer):
                        raise e
                    try:
                        return X.columns
                    except AttributeError:
                        print(f'{X=}')
                        raise Exception('Cannot produce DataFrame with named columns: columns are not defined')


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
        reuse_dtypes=False,
        columns: Optional[List] = None,
) -> DataFrame:
    df_arguments = {}

    if reuse_columns:
        if columns is not None:
            df_arguments['columns'] = columns  # type: ignore
        else:
            df_arguments['columns'] = input_data_frame.columns

    if reuse_index:
        df_arguments['index'] = input_data_frame.index

    if isinstance(output_data, csr_matrix):
        output_data = output_data.toarray()

    dtypes = dict(zip(
        df_arguments['columns'],
        input_data_frame.dtypes,
    ))

    new_data = DataFrame(
        output_data,
        **df_arguments,
    )

    # if reuse_dtypes:
    #     return new_data.astype(dtypes)
    # else:
    return new_data


class DFColumnTransformer(DFWrapped, ColumnTransformer):

    def fit_transform(self, X, *args, **kwargs):
        n_features = 0
        for index, transformer in enumerate(self.transformers):
            transformer_list = list(transformer)
            new_features = [column for column in transformer[2] if column in X.columns]
            n_features += len(new_features)
            transformer_list[2] = new_features
            self.transformers[index] = tuple(transformer_list)

        return super().fit_transform(X, *args, **kwargs)


class DFSimpleImputer(DFWrapped, SimpleImputer):
    ...


class DFStandardScaler(DFWrapped, StandardScaler):
    ...


class DFOrdinalEncoder(DFWrapped, OrdinalEncoder):
    ...


class DFVarianceThreshold(DFWrapped, VarianceThreshold):
    ...


class DFKNNImputer(DFWrapped, KNNImputer):
    ...


class DFXGBClassifier(DFWrapped, XGBClassifier):
    ...


class DFXGBRegressor(DFWrapped, XGBRegressor):
    ...


class DFPipeline(Pipeline, Estimator):
    y_name: Optional[str]

    def fit(self, X: DataFrame, y: Target = None, **fit_params) -> None:
        self.y_name = y.name
        super().fit(X, y)

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

    def suggest_optuna(self, trial: Trial, prefix: str = '') -> Tuple[Trial, Dict]:
        prefix_ = (prefix + '_') if prefix else ''
        return trial, {
            name: step.suggest_optuna(trial, f'{prefix_}{name}_')[1]
            for (name, step) in self.steps
            if hasattr(step, 'suggest_optuna')
        }

    def get_streamlit_configuration(self, config: Dict):
        config = {
            name: step.get_streamlit_configuration(config.get(name, {}))
            for (name, step) in self.steps
            if hasattr(step, 'get_streamlit_configuration')
        }
        return config

    def __getattr__(self, item):
        if hasattr(self[-1], item):
            return getattr(self[-1], item)
        else:
            raise AttributeError(f'AttributeError: object has no attribute \'{item}\'')

    def __repr__(self, **kwargs):
        return 'model'

    def get_final(self):
        return self


class DFLogisticRegression(DFWrapped, LogisticRegression):

    @property
    def coefficients(self):
        return Series(self.coef_[0], index=self.get_feature_names())


class DFElasticNet(DFWrapped, ElasticNet):

    @property
    def coefficients(self):
        return Series(self.coef_[0], index=self.get_feature_names())


class ToSurvivalRecord:

    def fit(self, X: DataFrame, y: Target):
        super().fit(X, to_survival_y_records(y))


class DFCoxnetSurvivalAnalysis(DFWrapped, ToSurvivalRecord, CoxnetSurvivalAnalysis):
    ...


class DFCoxPHSurvivalAnalysis(DFWrapped, CoxPHSurvivalAnalysis):
    ...


class DFSurvivalTree(DFWrapped, SurvivalTree):
    ...


class DFStacking(DFWrapped, Stacking):
    ...


class DFBinMapper(DFWrapped, _BinMapper):
    ...


class DFFunctionTransformer(DFWrapped, FunctionTransformer):
    ...


class DFRandomForestClassifier(DFWrapped, RandomForestClassifier):

    def get_feature_importance(self):
        return Series(super().feature_importances_, index=self.get_feature_names())


class DFRandomForestRegressor(DFWrapped, RandomForestRegressor):

    def get_feature_importance(self):
        return Series(super().feature_importances_, index=self.get_feature_names())
