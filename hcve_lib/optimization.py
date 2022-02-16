import operator
from collections import Callable

import mlflow
import numpy as np
import optuna
import toolz
from mlflow import set_tag
from optuna import Study
from optuna.trial import TrialState, FrozenTrial

# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from hcve_lib.custom_types import SplitPrediction
from hcve_lib.evaluation_functions import c_index
from hcve_lib.tracking import log_metrics_ci


class EarlyStoppingCallback(object):
    def __init__(
        self,
        early_stopping_rounds: int,
        direction: str = "minimize",
        stop_callback: Callable[[int], None] = None,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.stop_callback = stop_callback
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: Study, trial: FrozenTrial) -> Study:
        if trial.state.name == "FAIL":
            return study

        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            if self.stop_callback:
                self.stop_callback(self._iter)
            study.stop()

        return study


def optuna_report_mlflow(study, _):
    if len(study.get_trials(states=[TrialState.COMPLETE])) == 0:
        return study

    set_tag(
        'trials',
        len(study.trials),
    )
    set_tag(
        'failed',
        toolz.count(1 for trial in study.trials if trial.state.name == "FAIL"),
    )

    mlflow.log_figure(
        optuna.visualization.plot_optimization_history(study),
        'optimization_history.html',
    )

    mlflow.log_figure(
        optuna.visualization.plot_parallel_coordinate(study),
        'parallel_coordinate_hyperparams.html',
    )
    try:
        if len(study.get_trials(states=[TrialState.COMPLETE])) > 1:
            mlflow.log_figure(
                optuna.visualization.plot_param_importances(study),
                'plot_hyperparam_importances.html',
            )

        log_metrics_ci(study.best_trial.user_attrs['metrics'])
    except RuntimeError:
        pass

    return study


def objective_c_index(estimator, X_test, y_true):
    y_score = estimator.predict(X_test)
    return c_index(
        SplitPrediction(y_true=y_true, y_score=y_score, model=estimator))
