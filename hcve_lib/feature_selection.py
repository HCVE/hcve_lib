from typing import Optional, Dict, Protocol, List

import plotly.graph_objs as go
from pandas import DataFrame
from hcve_lib.custom_types import Metrics, Target, Results
from hcve_lib.evaluation_functions import compute_metrics
from hcve_lib.feature_importance import get_model_importance
from hcve_lib.utils import partial
from hcve_lib.utils import transpose_dict


def get_forward_feature_selection_curve(
    X, y, get_results, max_features: Optional[int] = None
):
    selected_features: List = []
    available_features = list(X.columns)
    metrics_per_features = {}
    while len(available_features) != 0:
        potential_features_results = {}
        for potential_feature in available_features:
            results = get_results(X=X[selected_features + [potential_feature]], y=y)
            metrics_value = compute_metrics(
                results,
                y,
            )
            potential_features_results[potential_feature] = {
                "metrics": metrics_value,
                "results": results,
            }

        best_feature = max(
            potential_features_results,
            key=lambda key: potential_features_results.get(key)["metrics"]["roc_auc"][
                "mean"
            ],
        )
        selected_features.append(best_feature)
        available_features.remove(best_feature)
        metrics_per_features[len(selected_features)] = {
            "metrics": potential_features_results[best_feature]["metrics"],
            "results": potential_features_results[best_feature]["results"],
            "features": tuple(selected_features),
        }

    return metrics_per_features


class CrossValidateCallback(Protocol):
    def __call__(self, X: DataFrame, y: Target) -> Results: ...


def get_importance_feature_selection_curve(
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    max_features: Optional[int] = None,
    verbose: bool = True,
):
    feature_selection_curve_all = evaluate_n_features(
        len(X.columns), X, y, cross_validate_callback, return_result=True
    )
    fi = get_model_importance(feature_selection_curve_all["results"])
    max_features_ = max_features or len(X.columns)
    n_feature_range = list(range(1, max_features_ + 1))

    # with NonDaemonPool(5) as pool:
    feature_selection_curve = dict(
        zip(
            n_feature_range,
            map(
                partial(
                    evaluate_n_features,
                    X=X,
                    y=y,
                    cross_validate_callback=cross_validate_callback,
                    fi=fi,
                    verbose=verbose,
                ),
                n_feature_range,
            ),
        )
    )

    feature_selection_curve[len(X.columns)] = feature_selection_curve_all

    return feature_selection_curve


def evaluate_n_features(
    n_features: int,
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    metrics=None,
    fi=None,
    return_result=False,
    verbose: bool = False,
) -> Dict:
    if fi is None:
        if n_features != len(X.columns):
            raise Exception("Need feature importance when evaluating a subset")
        X_selected = X
    else:
        X_selected = X[fi.index[:n_features]]

    if verbose:
        print(len(X_selected.columns), end=" ")

    results = cross_validate_callback(X=X_selected, y=y)

    metrics_value: Metrics = compute_metrics(
        results,
        y,
        metrics,
    )

    output: Dict = dict(metrics=metrics_value, features=X_selected.columns.tolist())

    if return_result:
        output["results"] = results

    return output


def plot_feature_selection_curve(
    feature_selection_curve, metric_name: Optional[str] = None
):
    metrics: Metrics = transpose_dict(feature_selection_curve)["metrics"]
    features = transpose_dict(feature_selection_curve)["features"]
    per_metric = transpose_dict(metrics)

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
