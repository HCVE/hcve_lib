from functools import partial

from pandas import DataFrame, Series
from plotly import express as px
from plotly.graph_objs import Figure
from sklearn.inspection import permutation_importance

from hcve_lib.custom_types import SplitPrediction, Target
from hcve_lib.data import Metadata, format_features
from hcve_lib.functional import pipe
from hcve_lib.utils import split_data


def run_permutation_importance(
    result: SplitPrediction,
    X: DataFrame,
    y: Target,
    metadata: Metadata,
    show: bool = True,
    previous_plot=None,
    **get_permutation_importance_kwargs,
) -> Figure:
    fig = pipe(
        get_permutation_importance(
            result,
            X,
            y,
            **get_permutation_importance_kwargs,
        ),
        partial(plot_permutation_importance, metadata=metadata),
    )
    if show:
        fig.show()
    else:
        return fig


def get_permutation_importance(
    result_split: SplitPrediction,
    X: DataFrame,
    y: Target,
    train_importance: bool = False,
    limit: int = None,
    **permutation_importance_kwargs,
) -> DataFrame:
    X_train, y_train, X_test, y_test = split_data(X, y, result_split)

    if train_importance:
        X_importance = X_train
        y_importance = y_train
    else:
        X_importance = X_test
        y_importance = y_test

    importance_result = permutation_importance(
        result_split['model'],
        X_importance,
        y_importance,
        **permutation_importance_kwargs,
    )
    sorting = Series(
        importance_result.importances_mean,
        index=X_train.columns,
    ).sort_values(ascending=False).index

    output_importance = DataFrame(
        importance_result.importances,
        index=X_train.columns,
    ).loc[sorting]

    if limit:
        return output_importance.iloc[:limit]
    else:
        return output_importance


def plot_permutation_importance(
    importance: DataFrame,
    metadata: Metadata,
) -> Figure:
    importance = importance[::-1]
    importance = importance.T
    importance = format_features(importance, metadata)
    fig = px.box(importance, orientation='h')
    fig.update_layout(
        xaxis_title="C-index importance",
        yaxis_title='Feature',
        yaxis_tickmode='linear',
    )
    fig.add_vline(x=0)
    fig.update_yaxes(showgrid=True)
    return fig


def get_tree_feature_importance(split: SplitPrediction):
    importance = Series(
        split['model'][-1].inner.feature_importances_,
        index=split['X_columns'],
    )
    return importance.sort_values(ascending=False)


def plot_standard_importance(
    importance: DataFrame,
    metadata: Metadata,
) -> Figure:
    importance = importance[::-1]
    importance = importance.T
    importance = format_features(importance, metadata)
    fig = px.bar(importance, orientation='h')
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title='Feature',
        yaxis_tickmode='linear',
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True)
    return fig
