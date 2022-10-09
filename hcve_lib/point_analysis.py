from typing import Callable, Dict

from pandas import Series, DataFrame
from plotly import express as px
from toolz import valmap

from hcve_lib.custom_types import Result, Prediction, ExceptionValue, Target
from hcve_lib.data import binarize_event
from hcve_lib.tracking import display_run_info, load_run_results
from hcve_lib.utils import split_data
from hcve_lib.visualisation import b, p


def get_result_display_info(run_id: str) -> Result:
    display_run_info(run_id)
    result: Result = load_run_results(run_id)
    return result


def get_point_evaluation(
    result: Result,
    get_proba: Callable[[Prediction], Series],
    X: DataFrame,
    y: Target,
) -> Dict[str, DataFrame]:
    return valmap(
        lambda prediction: get_point_evaluation_split(
            prediction,
            get_proba(prediction),
            X,
            y,
        ),
        result,
    )


def get_point_evaluation_split(
    prediction: Prediction,
    y_proba: Series,
    X: DataFrame,
    y: Target,
    time: int = 5*365,
) -> DataFrame:
    if y_proba is None:
        return ExceptionValue(exception=ValueError('y_proba is None'))

    X_train, y_train, X_test, y_test = split_data(X, y, prediction)
    y_test_binary = 1 - binarize_event(time, y_test['data'])
    y_proba_binary = round(prediction['y_proba'][time])
    y_brier = (prediction['y_proba'][time] - y_test_binary)**2
    return DataFrame({
        'y_test': y_test_binary,
        'y_proba': prediction['y_proba'][time],
        'y_proba_binary': y_proba_binary,
        'y_brier': y_brier,
    })\
        .dropna()\
        .sort_values(['y_test', 'y_proba'])


def plot_point_evaluation(compares: Dict[str, DataFrame]) -> None:
    for name, compare in compares.items():
        b(name)
        if isinstance(compare, ExceptionValue):
            p(f'Exception: {compare.exception}')
        else:
            plot_point_evaluation_split(compare)


def plot_point_evaluation_split(compare: DataFrame) -> None:
    fig = px.imshow(
        compare.to_numpy(),
        aspect="auto",
        color_continuous_scale='RdBu',
        x=compare.columns,
    )
    fig.show()


def display_point_evaluation(
    run_id: str,
    get_proba: Callable[[Prediction], Series],
    X: DataFrame,
    y: Target,
) -> None:
    plot_point_evaluation(
        get_point_evaluation(
            get_result_display_info(run_id),
            get_proba,
            X,
            y,
        ))
