import plotly.graph_objs as go
from pandas import DataFrame
from typing import Optional, Protocol, List, TypedDict, Mapping, Iterable, Callable
from typing_extensions import NotRequired

from hcve_lib.custom_types import Metrics, Target, Results
from hcve_lib.evaluation_functions import compute_metrics
from hcve_lib.feature_importance import get_model_importance
from hcve_lib.utils import transpose_mapping


class FeatureSelectionPoint(TypedDict, total=False):
    metrics: Metrics
    features: List[str]
    results: NotRequired[Results]


FeatureSelectionCurve = Mapping[int, FeatureSelectionPoint]


class CrossValidateCallback(Protocol):
    def __call__(self, X: DataFrame, y: Target) -> Results: ...


def evaluate_all_points(n_features: int) -> Iterable[int]:
    return range(n_features, 0, -1)


def evaluate_stepped_points(
    total_features: int,
    threshold: Optional[int] = None,
    step: int = 10,
    max_features: Optional[int] = None,
) -> Iterable[int]:
    if threshold is None:
        threshold = total_features

    if max_features is not None:
        total_features = min(total_features, max_features)
        threshold = min(threshold, max_features)

    print("step", step)
    stepped_evaluation = range(total_features, threshold, -step)

    full_evaluation = range(threshold, 0, -1)

    return list(stepped_evaluation) + list(full_evaluation)


def get_importance_feature_selection_curve(
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    get_evaluated_points: Callable[[int], Iterable[int]] = evaluate_all_points,
    return_results: bool = False,
    verbose: bool = True,
) -> FeatureSelectionCurve:
    n_features_points = get_evaluated_points(len(X.columns))

    point_all_features = get_feature_selection_point(
        X=X,
        y=y,
        cross_validate_callback=cross_validate_callback,
        return_result=True,
    )
    feature_importance = get_model_importance(point_all_features["results"])

    if not return_results:
        point_all_features.pop("results", None)

    feature_selection_curve = {}
    feature_selection_curve[len(X.columns)] = point_all_features

    for n_features in n_features_points:
        X_selected = select_features_by_importance(
            X=X,
            n_features=n_features,
            feature_importance=feature_importance,
        )

        feature_selection_curve[n_features] = get_feature_selection_point(
            X=X_selected,
            y=y,
            cross_validate_callback=cross_validate_callback,
            return_result=True,
            verbose=verbose,
        )

        feature_importance = get_model_importance(
            feature_selection_curve[n_features]["results"]
        )

        if not return_results:
            feature_selection_curve[n_features].pop("results", None)

    return feature_selection_curve


def get_feature_selection_point(
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    metrics=None,
    return_result=False,
    verbose: bool = False,
) -> FeatureSelectionPoint:
    if verbose:
        print(len(X.columns), end=" ", flush=True)

    results = cross_validate_callback(X=X, y=y)

    metrics_value: Metrics = compute_metrics(
        results,
        y,
        metrics,
    )

    output: FeatureSelectionPoint = FeatureSelectionPoint(
        metrics=metrics_value,
        features=X.columns.tolist(),
    )

    if return_result:
        output["results"] = results

    return output


def select_features_by_importance(
    X: DataFrame,
    n_features: int,
    feature_importance: Optional[DataFrame],
) -> DataFrame:
    if feature_importance is None:
        if n_features != len(X.columns):
            raise Exception("Need feature importance when evaluating a subset")
        return X
    print(feature_importance.index[:n_features])
    return X[feature_importance.index[:n_features]]


def plot_feature_selection_curve(
    feature_selection_curve: FeatureSelectionCurve, metric_name: Optional[str] = None
) -> None:
    metrics: Metrics = transpose_mapping(feature_selection_curve)["metrics"]
    features: Mapping[int, List[str]] = transpose_mapping(feature_selection_curve)[
        "features"
    ]
    per_metric = transpose_mapping(metrics)

    for metric_name, values_per_n in per_metric.items():
        x_values = list(values_per_n.keys())
        y_values = list(map(lambda item: item["mean"], values_per_n.values()))
        lower_band = list(map(lambda item: item["ci"][0], values_per_n.values()))
        upper_band = list(map(lambda item: item["ci"][1], values_per_n.values()))
        fig = go.Figure(
            [
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    line=dict(color="rgb(0,100,80)"),
                    mode="markers+lines",
                    showlegend=False,
                    hovertext=["<br>".join(f) for f in features.values()],
                ),
                go.Scatter(
                    x=x_values + x_values[::-1],
                    y=lower_band + upper_band[::-1],
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
            ]
        )
        fig.update_layout(
            title=metric_name, xaxis_title="n features", yaxis_title=metric_name
        )
        fig.update_traces(mode="markers+lines")
        fig.show()
