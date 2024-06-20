from copy import deepcopy
from typing import List, Tuple, Iterable, Any
from plotly import express as px
import numpy as np
import pandas
import shap
from joblib import Logger
from pandas import DataFrame
import itertools
from hcve_lib.custom_types import TargetType, Estimator, Result, Prediction
from hcve_lib.progress_reporter import ProgressReporter
from hcve_lib.utils import (
    get_X_split,
    NonDaemonPool,
    get_jobs,
    get_predictions_from_results,
    DummyLogger,
    average_kendall_tau,
)

ShapResult = Tuple[
    DataFrame, List[np.ndarray], Any, List[shap.Explainer], List[Estimator]
]


def get_shap_values(
    results: List[Result],
    X: DataFrame,
    is_test: bool = True,
    logger=DummyLogger(),
    n_jobs=1,
    reporter: ProgressReporter = None,
) -> List[ShapResult]:
    predictions = list(get_predictions_from_results(results))

    if reporter:
        reporter.total = len(predictions)

    args = ((prediction, X, is_test, logger, reporter) for prediction in predictions)
    if n_jobs == 1:
        return list(itertools.starmap(get_shap_values_, args))
    else:
        with NonDaemonPool(get_jobs(n_jobs, len(results))[0]) as pool:
            return pool.starmap(get_shap_values_, args)


def get_average_shap_values(
    results: List[Result],
    X: DataFrame,
    is_test: bool = True,
    logger=DummyLogger(),
    n_jobs=-1,
    reporter: ProgressReporter = None,
) -> ShapResult:
    return average_shap_values(
        get_shap_values(results, X, is_test, logger, n_jobs, reporter=reporter)
    )


def get_shap_values_(
    prediction: Prediction,
    X: DataFrame,
    is_test: bool = True,
    logger: Logger = DummyLogger(),
    reporter: ProgressReporter = None,
) -> ShapResult:
    logger.info(".", end="")
    X_train, X_test = get_X_split(X, prediction, logger=logger)
    X_test_or_train = X_test if is_test else X_train
    Xt = prediction["model"].transform(X_test_or_train)
    explainer = shap.Explainer(prediction["model"]._estimator, Xt)

    try:
        shap_values = explainer.shap_values(Xt, check_additivity=False)
        shap_values2 = explainer(Xt, check_additivity=False)
    except TypeError:
        shap_values = explainer.shap_values(Xt)
        shap_values2 = explainer(Xt)

    shap_values_ = shap_values[1] if isinstance(shap_values, list) else shap_values

    if reporter:
        reporter.finished()

    try:
        shap_values2.values = shap_values2.values[:, :, 1]
        shap_values2.base_values = shap_values2.base_values[:, 1]
    except:
        pass

    return Xt, shap_values_, shap_values2, explainer, prediction["model"]


def average_shap_values(shap_results: List[ShapResult]) -> ShapResult:
    shap_values_per_bootstrap_sample_df = []
    base_values_per_bootstrap_sample_df = []
    Xts_per_bootstrap_sample = []
    for Xt, _, shap_values2, _, _ in shap_results:
        shap_values_per_bootstrap_sample_df.append(
            DataFrame(
                shap_values2.values,
                index=Xt.index,
            )
        )
        base_values_per_bootstrap_sample_df.append(
            DataFrame(
                shap_values2.base_values,
                index=Xt.index,
            )
        )
        Xts_per_bootstrap_sample.append(Xt)

    shap_values_per_bootstrap_sample_df_concat = pandas.concat(
        shap_values_per_bootstrap_sample_df
    )

    base_values_per_bootstrap_sample_df_concat = pandas.concat(
        base_values_per_bootstrap_sample_df
    )

    shap_values_averaged_df = shap_values_per_bootstrap_sample_df_concat.groupby(
        shap_values_per_bootstrap_sample_df_concat.index
    ).mean()

    base_values_averaged_df = shap_values_per_bootstrap_sample_df_concat.groupby(
        shap_values_per_bootstrap_sample_df_concat.index
    ).mean()

    X_final = pandas.concat(Xts_per_bootstrap_sample)
    X_final["index"] = X_final.index
    X_final.reset_index(inplace=True, drop=True)
    X_final.drop_duplicates(subset="index", keep="first", inplace=True)
    X_final.index = X_final["index"]
    X_final.drop(columns=["index"], inplace=True)

    shap_values = deepcopy(shap_results[0][2])
    shap_values.values = shap_values_averaged_df.to_numpy()
    shap_values.base_values = base_values_averaged_df.to_numpy()
    shap_values.data = X_final.loc[shap_values_averaged_df.index]

    return shap_values


def get_shap_ranking(shap_results: List[ShapResult]) -> DataFrame:
    ranking_per_boostrap_sample = []
    for Xt, _, shap_values2, _, _ in shap_results:
        mean_abs_shap_values = np.mean(np.abs(shap_values2.values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        sorted_features = [Xt.columns[i] for i in sorted_indices]
        feature_ranks = np.arange(1, len(sorted_features) + 1)
        ranking_per_boostrap_sample.append(dict(zip(sorted_features, feature_ranks)))

    ranking_per_boostrap_sample_df = DataFrame(ranking_per_boostrap_sample)
    column_means = ranking_per_boostrap_sample_df.mean()
    sorted_columns = column_means.sort_values()
    df_sorted = ranking_per_boostrap_sample_df[sorted_columns.index]
    return df_sorted


def plot_shap_ranking(ranking: DataFrame, limit: int = 20):
    ranking_ = ranking.iloc[:, :limit]
    print("Kandal's τ=", average_kendall_tau(ranking_.values.tolist()))
    fig = px.parallel_coordinates(ranking_ + np.random.normal(0, 0.01, ranking_.shape))
    fig.update_traces(
        unselected=dict(line=dict(opacity=0.5)), selector=dict(type="parcoords")
    )
    fig.update_traces(dimensions=[dict(range=[0, len(ranking.columns)])])
    fig.show()
