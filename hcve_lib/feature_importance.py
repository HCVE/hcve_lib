from functools import partial

from hcve_lib.custom_types import Prediction, Target
from hcve_lib.data import Metadata, format_features
from hcve_lib.functional import pipe
from hcve_lib.utils import split_data, loc
from pandas import DataFrame, Series
from plotly import express as px
from plotly.graph_objs import Figure
from sklearn.inspection import permutation_importance

from hcve_lib.data import format_identifier_short


def run_permutation_importance(
    result: Prediction,
    X: DataFrame,
    y: Target,
    metadata: Metadata,
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
    return fig


def get_permutation_importance(
    prediction: Prediction,
    X: DataFrame,
    y: Target,
    train_importance: bool = False,
    limit: int = None,
    **permutation_importance_kwargs,
) -> DataFrame:
    X_train, y_train, X_test, y_test = split_data(X, y, prediction)

    X_train = X_train.dropna()
    y_train = loc(X_train.index, y_train)

    X_test = X_test.dropna()
    y_test = loc(X_test.index, y_test)

    if train_importance:
        X_ = X_train
        y_ = y_train
    else:
        X_ = X_test
        y_ = y_test

    importance_result = permutation_importance(
        prediction['model'],
        X_,
        y_,
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
    *args,
    **kwargs,
) -> Figure:
    importance = importance[::-1]
    importance = importance.T
    importance = format_features(
        importance,
        metadata,
        formatter=format_identifier_short,
    )
    fig = px.box(
        importance,
        orientation='h',
        *args,
        **kwargs,
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title='Feature',
        yaxis_tickmode='linear',
        template='simple_white',
    )
    fig.add_vline(x=0, line_width=2, opacity=0.3, line_color='red')
    fig.update_yaxes(showgrid=True)
    return fig


def get_tree_feature_importance(split: Prediction):
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
