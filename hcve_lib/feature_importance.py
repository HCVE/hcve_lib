from itertools import islice
from typing import List, Optional

from pandas import DataFrame
from plotly.graph_objs import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

from hcve_lib.custom_types import Prediction, Target, Result
from hcve_lib.data import Metadata, format_features
from hcve_lib.functional import t
from hcve_lib.utils import get_first_entry, get_models_from_repeats
from typing import List
from pandas import DataFrame, Series
from plotly import express as px
from sklearn.metrics import r2_score

from hcve_lib.custom_types import Result
from hcve_lib.utils import get_X_split, get_y_split, is_numerical


def run_permutation_importance(
    X: DataFrame,
    y: Target,
    results: List[Result],
    random_state: int,
    is_test: bool = True,
    *args,
    **kwargs,
) -> None:
    importance, predictions = get_permutation_importance(
        X,
        y,
        results,
        random_state,
        is_test=is_test,
        *args,
        **kwargs,
    )
    plot_permutation_importance(X, y, importance, predictions)


def get_permutation_importance(
    X, y, results, random_state, is_test: bool = True, n_repeats=10, *args, **kwargs
):
    predictions = [get_first_entry(result) for result in results]
    importances = [
        get_permutation_importance_prediction(
            prediction,
            X,
            y,
            random_state=random_state,
            is_test=is_test,
            n_repeats=n_repeats,
            *args,
            **kwargs,
        )
        for prediction in predictions
    ]
    return importances, predictions


def get_permutation_importance_prediction(
    prediction,
    X,
    y,
    random_state,
    is_test: bool = True,
    n_repeats=10,
    *args,
    **kwargs,
):
    def permutation_score(estimator, X, y_true):
        if is_numerical(y):
            y_pred = estimator.predict(X)
            metric_value = r2_score(y_true, y_pred)
        else:
            y_proba = estimator.predict(X)[1]
            metric_value = average_precision_score(y_true, y_proba)

        return metric_value

    X_train, X_test = get_X_split(X, prediction)
    y_train, y_test = get_y_split(y, prediction)

    permutation = permutation_importance(
        prediction["model"],
        X_test if is_test else X_train,
        y_test if is_test else y_train,
        scoring=permutation_score,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        *args,
        **kwargs,
    )

    return permutation


def plot_permutation_importance(
    X: DataFrame,
    y: Target,
    importances: DataFrame,
    predictions: List[Prediction],
) -> None:
    sorting = None
    columns = predictions[0]["X_columns"]

    sorting = (
        DataFrame(
            {
                num: importance.importances_mean
                for num, importance in enumerate(importances)
            },
            index=columns,
        )
        .mean(axis=1)
        .sort_values()
        .index.tolist()
    )

    images = []

    for importance, prediction in zip(importances, predictions):
        plot_feature_importance_prediction(X, y, prediction, importance, sorting)


def plot_feature_importance_prediction(
    X: DataFrame,
    y: Target,
    prediction: Prediction,
    importance: Series,
    sorting=None,
):
    X_train, X_test = get_X_split(X, prediction)
    y_train, y_test = get_y_split(y, prediction)
    if sorting is None:
        sorting = (
            Series(
                importance.importances_mean,
                index=X_train.columns,
            )
            .sort_values(ascending=True)
            .index
        )
    output_importance = DataFrame(
        importance.importances,
        index=X_train.columns,
    ).loc[sorting]
    fig = px.box(
        output_importance.T,
        orientation="h",
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_tickmode="linear",
        template="simple_white",
        height=get_plot_height(output_importance),
    )
    fig.add_vline(x=0, line_width=2, opacity=0.3, line_color="red")
    fig.update_yaxes(showgrid=True)
    fig.show()


def get_tree_feature_importance(split: Prediction):
    importance = Series(
        split["model"][-1].inner.feature_importances_,
        index=split["X_columns"],
    )
    return importance.sort_values(ascending=False)


def plot_standard_importance(
    importance: DataFrame,
    metadata: Metadata,
) -> Figure:
    importance = importance[::-1]
    importance = importance.T
    importance = format_features(importance, metadata)
    fig = px.bar(importance, orientation="h")
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_tickmode="linear",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True)
    return fig


def plot_standard_importance_(
    importance: DataFrame,
) -> Figure:
    importance["mean"] = importance.mean(axis=1)
    importance.sort_values(by="mean", ascending=True, inplace=True)
    importance = importance.T
    fig = px.strip(importance, orientation="h")
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_tickmode="linear",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True)
    return fig


def get_model_importance(results: List[Result]) -> DataFrame:
    models = get_models_from_repeats(results)
    importances = [model.get_feature_importance() for model in models]
    forest_importances = DataFrame(
        {num: importance for num, importance in enumerate(importances)}
    )
    forest_importance_avg = forest_importances.mean(axis=1).sort_values(ascending=False)
    return forest_importances.loc[forest_importance_avg.index]


def plot_model_importance_results(results: List[Result], limit=None) -> Figure:
    importance = get_model_importance(results)
    if limit is None:
        limit = len(importance)
    return plot_model_importance_results_runs(importance[:limit][::-1])


def plot_model_importance_results_runs(importance: DataFrame) -> Figure:
    importance = importance.loc[importance.mean(axis=1).abs().sort_values().index]
    fig = px.strip(
        importance.T,
        orientation="h",
    )

    fig.update_traces(marker=dict(opacity=0.5))
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_tickmode="linear",
        # template="simple_white",
        height=get_plot_height(importance),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_vline(x=0, line_width=2, opacity=0.3, line_color="red")
    fig.update_yaxes(showgrid=True)
    return fig


def plot_model_importance_results_per_run(
    results: List[Result], limit: Optional[int] = 20
) -> None:
    importance = get_model_importance(results)
    for _, importance_single in islice(importance.iteritems(), limit):
        plot_model_importance_results_per_run_(importance, importance_single)


def plot_model_importance_results_per_run_(importance_single):
    importance_single = importance_single.loc[
        importance_single.abs().sort_values().index
    ]
    fig = px.bar(
        importance_single,
        orientation="h",
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_tickmode="linear",
        template="simple_white",
        height=get_plot_height(importance_single),
    )
    fig.add_vline(x=0, line_width=2, opacity=0.3, line_color="red")
    fig.update_yaxes(showgrid=True)
    fig.show()


def get_plot_height(items):
    return 15 * len(items) + 200
