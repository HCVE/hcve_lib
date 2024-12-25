from statistics import mean
from typing import Dict, Protocol, Optional, List

import plotly.graph_objects as go
from pandas import DataFrame

from hcve_lib.custom_types import Results, Target, Metrics, TrainTestSplits
from hcve_lib.evaluation_functions import compute_metrics
from hcve_lib.metrics_types import Metric
from hcve_lib.splitting import get_bootstrap
from hcve_lib.utils import partial


class GetSplitTrainSize(Protocol):
    def __call__(
        self, X: DataFrame, y: Target, random_state: int, train_size: int
    ) -> TrainTestSplits: ...


class GetSplits(Protocol):
    def __call__(self, X: DataFrame, y: Target) -> TrainTestSplits: ...


class CrossValidateCallback(Protocol):
    def __call__(
        self,
        X: DataFrame,
        y: Target,
        random_state: int,
        get_splits: GetSplits,
    ) -> Results: ...


def get_learning_curve_data(
    X: DataFrame,
    y: Target,
    cross_validate_callback: CrossValidateCallback,
    random_state: int,
    get_splits: Optional[GetSplitTrainSize] = None,
    start_samples: float | int = 0.1,
    end_samples: float | int = 1.0,
    n_points: int = 10,
    verbose: bool = True,
) -> Dict[int, Results]:
    if get_splits is None:
        get_splits = get_bootstrap

    if start_samples < 0:
        raise ValueError("start_samples has to be greate or equal to  0")

    if end_samples < 0:
        raise ValueError("end_samples has to be greate or equal to  0")

    if n_points <= 0:
        raise ValueError("n_points to be greate or equal to  1")

    if start_samples >= end_samples:
        raise ValueError("start_sample must be smaller than the end sample")

    if isinstance(start_samples, float):
        if start_samples > 1:
            raise ValueError("Fraction has to be less or equal to 1 ")
        start_samples = round(len(X) * start_samples)

    if isinstance(end_samples, float):
        if end_samples > 1:
            raise ValueError("Fraction has to be less or equal to 1 ")
        end_samples = round(len(X) * end_samples)

    sample_sizes = (
        round(start_samples + i * (end_samples - start_samples) / (n_points - 1))
        for i in range(n_points)
    )

    results_all = {}

    for sample_size in sample_sizes:
        if verbose:
            print(sample_size, end=" ")

        get_splits_with_sizes: GetSplits = partial(
            get_splits,
            train_size=sample_size,
            random_state=random_state * sample_size,
        )

        results = cross_validate_callback(
            X=X,
            y=y,
            random_state=random_state * sample_size,
            get_splits=get_splits_with_sizes,
        )

        training_size = round(get_mean_train_size(results))
        results_all[training_size] = results

    return results_all


def compute_learning_curve_metrics(
    data: Dict[int, Results],
    y: Target,
    metrics: Optional[List[Metric]] = None,
) -> Dict[int, Metrics]:
    metrics_all: Dict[int, Metrics] = {}

    for sample_size, results in data.items():
        metrics_all[sample_size] = compute_metrics(
            results=results, y=y, metrics=metrics
        )

    return metrics_all


def get_mean_train_size(results: Results) -> float:
    train_sizes = []
    for result in results:
        for prediction in result.values():
            train_sizes.append(len(prediction["split"][0]))
    return mean(train_sizes)


def plot_learning_curve(
    learning_curve_metrics: Dict[int, Metrics], metric: str = "roc_auc"
) -> go.Figure:
    samples = sorted(list(learning_curve_metrics.keys()))
    means = [learning_curve_metrics[s][metric]["mean"] for s in samples]
    ci_lower = [learning_curve_metrics[s][metric]["ci"][0] for s in samples]
    ci_upper = [learning_curve_metrics[s][metric]["ci"][1] for s in samples]

    fig = go.Figure()

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=samples,
            y=means,
            mode="lines+markers",
            name="Mean",
            line=dict(color="rgb(31, 119, 180)"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=samples + samples[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    )

    fig.update_layout(
        title=f"Learning Curve - {metric.upper()}",
        xaxis_title="Number of Samples",
        yaxis_title=metric.upper(),
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig
