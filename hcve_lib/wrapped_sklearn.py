import numpy as np
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

from sksurv.linear_model import CoxnetSurvivalAnalysis
from typing import Any, Optional, List

# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
from sksurv.meta import Stacking


class DFWrapped:
    def get_feature_names(self):
        try:
            super().get_feature_names()
        except AttributeError:
            return self.fitted_feature_names

    def get_fitted_feature_names(self, X: DataFrame, X_out: DataFrame = None):
        try:
            return X_out.columns  # type: ignore
        except AttributeError:
            try:
                return super().get_feature_names()  # type: ignore
            except (AttributeError, ValueError) as e:
                try:
                    return ( \
                        super().get_feature_names_out(X.columns)  # type: ignore
                    )
                except (AttributeError, NotFittedError):
                    if isinstance(self, ColumnTransformer):
                        raise e
                    try:
                        return X.columns
                    except AttributeError:
                        raise Exception(
                            'Cannot produce DataFrame with named columns: columns are not defined'
                        )

    def transform(self, X, *args, **kwargs):
        try:
            X_out = super().transform(X, *args, **kwargs)
            self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
            out = use_df_fn(X, X_out, columns=self.fitted_feature_names)
            return out
        except AttributeError:
            return X

    def fit_transform(self, X, *args, **kwargs):
        X_out = super().fit_transform(X, *args, **kwargs)
        self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
        out = use_df_fn(X, X_out, columns=self.fitted_feature_names)

        return out

    def fit_predict(self, X, y, *args, **kwargs):
        X_out = super().fit_predict(X, y, *args, **kwargs)
        self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
        return use_df_fn(X, X_out, columns=self.fitted_feature_names)

    def predict(self, X, *args, **kwargs) -> Series:
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        y_pred = super().predict(X, *args, **kwargs)  # type: ignore
        return y_pred

    def predict_proba(self, X, *args, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        y_proba = super().predict_proba(X, *args, **kwargs)  # type: ignore
        return y_proba

    def fit(self, X, y, *args, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        super().fit(X, y, *args, **kwargs)
        return self

    def get_params(self, *args, **kwargs):
        return super().get_params(*args, **kwargs)


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

    return DataFrame(output_data, **df_arguments)


class DFColumnTransformer(DFWrapped, ColumnTransformer):
    def fit_transform(self, X, *args, **kwargs):
        n_features = 0
        for index, transformer in enumerate(self.transformers):
            transformer_list = list(transformer)
            new_features = [
                column for column in transformer[2] if column in X.columns
            ]
            n_features += len(new_features)
            transformer_list[2] = new_features
            self.transformers[index] = tuple(transformer_list)

        # noinspection PyAttributeOutsideInit
        # print(self.n_features_in_)
        # self.n_features_in_ = n_features

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


class DFPipeline(Pipeline):
    def get_feature_names(self):
        return self.steps[-1][1].fitted_feature_names


class DFLogisticRegression(DFWrapped, LogisticRegression):
    @property
    def coefficients(self):
        return Series(self.coef_[0], index=self.get_feature_names())


class DFCoxnetSurvivalAnalysis(DFWrapped, CoxnetSurvivalAnalysis):
    ...


class DFStacking(DFWrapped, Stacking):
    ...


class DFBinMapper(DFWrapped, _BinMapper):
    ...
