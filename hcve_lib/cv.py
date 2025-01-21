import gc
import logging
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from multiprocessing import cpu_count
from statistics import mean
from time import process_time
from typing import Iterable, Hashable, Mapping, Sequence, TypeVar, Union, Type, Protocol
from typing import List, Dict, Optional, Callable, cast
from typing import Tuple

import pandas
import toolz
import yaml
from mlflow import active_run, get_experiment, start_run, set_tracking_uri, set_tags
from mlflow import set_tag
from optuna import create_study, Trial
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler, BaseSampler
from pandas import DataFrame
from pandas import Series
from pandas.core.groupby import DataFrameGroupBy
from toolz import identity, dissoc
from toolz import valmap
from toolz.curried import valmap, valfilter

from hcve_lib.custom_types import (
    Prediction,
    Estimator,
    Target,
    TrainTestIndex,
    TrainTestSplits,
    Result,
    TrainTestSplitter,
    Results,
    Metrics,
)
from hcve_lib.evaluation_functions import (
    compute_metrics_result,
    log_repeat_metrics,
    compute_metrics,
    compute_metrics_prediction,
)
from hcve_lib.functional import pipe, always
from hcve_lib.metrics import ROC_AUC, get_standard_metrics
from hcve_lib.metrics_types import Metric
from hcve_lib.optimization import optuna_report_mlflow, EarlyStoppingCallback
from hcve_lib.pipelines import PredictionMethod
from hcve_lib.progress_reporter import ProgressReporter
from hcve_lib.splitting import (
    get_k_fold,
    get_k_fold_stratified,
    get_splits_from_str,
)
from hcve_lib.tracking import (
    log_early_stopping,
    get_logger,
    get_standard_repeat_context,
    log_results,
    log_metrics_single,
)
from hcve_lib.tracking import log_metrics, get_active_experiment_id
from hcve_lib.utils import (
    is_noneish,
    empty_dict,
    run_parallel,
    get_first_entry,
    get_mean_importance,
    get_models_from_result,
    get_jobs,
    get_pipeline_name,
    configuration_to_params,
)
from hcve_lib.utils import (
    list_to_dict_by_keys,
    list_to_dict_index,
    cross_validate_apply_mask,
    loc,
    random_seed,
    noop_context_manager,
)
from hcve_lib.utils import partial

CROSS_VALIDATE_KEY = "cross_validate_single_repeat"

GetRepeatContext = Callable[[int, int], AbstractContextManager]
OnRepeatResults = Callable[[Result, int], None]


def cross_validate(
    get_pipeline: Callable,
    X: DataFrame,
    y: Target,
    get_splits: TrainTestSplitter,
    random_state: int,
    n_repeats: int = 1,
    get_repeat_context: GetRepeatContext = None,
    on_repeat_result: List[OnRepeatResults] = None,
    n_jobs: Optional[int] = 1,
    mlflow: Union[bool, str] = False,
    optimize: bool = False,
    tags: Dict = None,
    compute_metrics_fn=compute_metrics,
    on_progress: Callable[[float], None] = None,
    *args,
    **kwargs,
) -> Results:
    get_splits = (
        get_splits_from_str(get_splits) if isinstance(get_splits, str) else get_splits
    )

    if on_repeat_result is None:
        on_repeat_result = []
    elif isinstance(mlflow, str):
        set_tracking_uri(mlflow)

    splits = get_splits(X=X, y=y, random_state=random_state)

    if n_jobs == 1:
        reporter = ProgressReporter(n_repeats * len(splits), on_progress=on_progress)
        reporter.set_message("Training models...")
    else:
        reporter = None

    if not get_repeat_context:
        if mlflow:
            get_repeat_context = get_standard_repeat_context
        else:
            get_repeat_context = noop_context_manager

    if mlflow:
        on_repeat_result = [*on_repeat_result, partial(log_repeat_metrics)]
        run_context = start_run(
            run_name=get_pipeline_name(get_pipeline(X, y, random_state))
        )
    else:
        run_context = noop_context_manager()

    with run_context:
        if mlflow:
            set_tag("optimize", optimize)
            set_tag("y", y.name)
            set_tag("features", len(X.columns))
            set_tag("get_splits", get_splits.__name__)

            if tags is not None:
                set_tags(tags)

        n_jobs, n_jobs_rest = get_jobs(n_jobs, maximum=n_repeats)
        n_jobs_rest = max(1, round(n_jobs_rest / n_repeats))

        data_for_repeats = get_data_for_cv_repeats(
            get_pipeline,
            X,
            y,
            get_splits,
            n_repeats,
            random_state,
            get_repeat_context,
            on_repeat_result,
            n_jobs_rest,
            mlflow,
            optimize,
            reporter,
            *args,
            **kwargs,
        )
        results = pipe(
            run_parallel(
                cross_validate_single_repeat_,
                data=data_for_repeats,
                n_jobs=n_jobs,
            ).values(),
            list,
        )

        if mlflow:
            metrics = compute_metrics_fn(
                results,
                y,
            )
            log_metrics(metrics)
            log_results(results)
            set_tag("root", True)

        return results


def cross_validate_single_repeat_(
    get_repeat_context: Callable[[], AbstractContextManager],
    on_repeat_result: List[Callable[[Result, Target], None]],
    y: Target,
    cv_single_repeat_args,
    cv_single_repeat_kwargs,
):
    with get_repeat_context():
        result = cross_validate_single_repeat(
            *cv_single_repeat_args, y=y, **cv_single_repeat_kwargs
        )
        for fn in on_repeat_result:
            fn(result, y)
    return result


class GetPipeline(Protocol):
    def __call__(self, X: DataFrame, y: Target, random_state: int) -> Results: ...


def get_data_for_cv_repeats(
    get_pipeline: Callable,
    X: DataFrame,
    y: Target,
    get_splits: TrainTestSplitter,
    n_repeats: int,
    random_state: int,
    get_repeat_context: Optional[GetRepeatContext] = None,
    on_repeat_result: Optional[List[OnRepeatResults]] = None,
    n_jobs_rest: Optional[int] = None,
    mlflow: Union[bool, str] = False,
    optimize: bool = False,
    reporter: Optional[ProgressReporter] = None,
    *args,
    **kwargs,
) -> Dict:
    data = {}

    for repeat_index in range(n_repeats):
        run_random_state = random_state + repeat_index * 10000
        get_repeat_context_ = partial(
            get_repeat_context,
            get_pipeline_name(get_pipeline(X=X, y=y, random_state=random_state)),
            repeat_index,
            run_random_state,
        )
        kwargs_ = toolz.merge(
            kwargs,
            {
                "get_pipeline": get_pipeline,
                "random_state": run_random_state,
                "X": X,
                "get_splits": get_splits,
                "n_jobs": n_jobs_rest,
                "mlflow": mlflow,
                "optimize": optimize,
                "reporter": reporter,
            },
        )

        data[repeat_index] = [
            get_repeat_context_,
            on_repeat_result,
            y,
            args,
            kwargs_,
        ]

    return data


def objective_predictive_performance(
    trial: Trial,
    get_pipeline: Callable,
    random_state: int,
    X: DataFrame,
    y: Union[Target, Series],
    get_splits: TrainTestSplitter,
    hyperparameters: Dict,
    logger: logging.Logger = None,
    objective_metric: Metric = None,
    predict_method: str = "predict_proba",
):
    if objective_metric is None:
        objective_metric = get_standard_metrics(y)[0]

    start_time = process_time()

    split_runs = cross_validate_single_repeat(
        get_pipeline,
        X,
        y,
        get_splits=get_splits,
        random_state=random_state,
        predict_method=predict_method,
        hyperparameters=hyperparameters,
        logger=logger,
        n_jobs=1,
        return_models=False,
    )

    metrics = compute_metrics_result(split_runs, y, [objective_metric])

    # TODO: optimizing memory
    # trial.set_user_attr("result_split", split_runs)

    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("hyperparameters", hyperparameters)

    duration = process_time() - start_time
    trial.set_user_attr("duration", duration)

    return [get_first_entry(metrics)["mean"]]


def objective_variance(
    trial: Trial,
    get_pipeline: Callable,
    random_state: int,
    X: DataFrame,
    y: Union[Target, Series],
    get_splits: TrainTestSplitter,
    hyperparameters: Dict,
    logger: logging.Logger = None,
    objective_metric: Metric = ROC_AUC(),
    predict_method: str = "predict_proba",
):
    start_time = process_time()

    result = cross_validate_single_repeat(
        get_pipeline,
        X,
        y,
        get_splits=partial(get_k_fold, n_splits=10),
        random_state=random_state,
        predict_method=predict_method,
        hyperparameters=hyperparameters,
        logger=logger,
        n_jobs=1,
    )

    importance = get_mean_importance(get_models_from_result(result))
    metrics = compute_metrics_result(result, y, [objective_metric])
    trial.set_user_attr("result_split", result)
    trial.set_user_attr("hyperparameters", hyperparameters)
    duration = process_time() - start_time
    trial.set_user_attr("duration", duration)
    return mean(
        [float(-importance["std"].sum()), float(get_first_entry(metrics)["mean"])]
    )


def objective_variance_prediction(
    trial: Trial,
    get_pipeline: Callable,
    random_state: int,
    X: DataFrame,
    y: Union[Target, Series],
    get_splits: TrainTestSplitter,
    hyperparameters: Dict,
    logger: logging.Logger = None,
    objective_metric: Metric = None,
    predict_method: str = "predict_proba",
):
    if objective_metric is None:
        objective_metric = get_standard_metrics(y)[0]

    start_time = process_time()

    split_runs = cross_validate_single_repeat(
        get_pipeline,
        X,
        y,
        get_splits=get_k_fold,
        random_state=random_state,
        predict_method=predict_method,
        hyperparameters=hyperparameters,
        logger=logger,
        n_jobs=1,
    )

    metrics = compute_metrics_result(split_runs, y, [objective_metric])
    trial.set_user_attr("result_split", split_runs)
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("hyperparameters", hyperparameters)

    duration = process_time() - start_time
    trial.set_user_attr("duration", duration)

    return [-get_first_entry(metrics)["std"]]


OptimizeEvaluate = Callable[
    [
        Trial,
        Callable,
        int,
        DataFrame,
        Target,
    ],
    float,
]


@dataclass
class OptimizationParams:
    early_stopping: bool = True
    n_jobs: int = -1
    n_trials: int = 10
    early_stopping_rounds: Optional[int] = 30
    get_splits: TrainTestSplitter = get_k_fold_stratified
    objective: Callable = objective_predictive_performance
    direction: str = "maximize"
    sampler: Type[BaseSampler] = TPESampler


class Optimize:
    fit_best_model: Optional[Estimator]
    direction = None
    study_name = None

    def __init__(
        self,
        get_pipeline: Callable,
        objective_evaluate: Callable,
        random_state: int,
        get_splits=None,
        optimize_params: OptimizationParams = OptimizationParams(),
        mlflow_callback=None,
        optimize_callbacks: List[Callable] = None,
        study_name: str = None,
        catch_exceptions: bool = True,
        logger: logging.Logger = None,
        direction: str = "maximize",
        predict_method: str = "predict_proba",
    ):
        if optimize_callbacks is None:
            optimize_callbacks = []

        if mlflow_callback is True:
            mlflow_callback = MLflowCallback(nest_trials=True)

        self.objective_evaluate = objective_evaluate
        self.optimize_params = optimize_params
        self.fit_best_model = None
        self.mlflow_callback = mlflow_callback
        self.optimize_callbacks = optimize_callbacks
        self.catch_exceptions = catch_exceptions
        self.logger = logger
        self.random_state = random_state
        self.predict_method = predict_method
        self.get_pipeline = get_pipeline

        random_seed(random_state)

        if not study_name:
            if active_run():
                self.study_name = get_experiment(active_run().info.experiment_id).name
            else:
                self.study_name = "study"

        self.get_splits = get_splits

        self.study = create_study(
            direction=direction,
            study_name=self.study_name,
            sampler=optimize_params.sampler(seed=random_state),
        )

    def fit(self, X, y):
        if self.mlflow_callback:
            decorator = self.mlflow_callback.track_in_mlflow()
        else:
            decorator = identity

        catch = (
            (
                (Exception, ArithmeticError, RuntimeError)
                if self.catch_exceptions
                else ()
            ),
        )

        self.study.optimize(
            partial(self.objective, X=X, y=y),
            n_trials=self.optimize_params.n_trials,
            callbacks=self.optimize_callbacks,
            # catch=(),
        )

    def objective(self, trial: Trial, X: DataFrame, y: Target) -> float:
        pipeline = self.get_pipeline(X, y, self.random_state)
        trial, hyperparameters = pipeline.suggest_optuna(trial, X)

        return self.objective_evaluate(
            trial=trial,
            get_pipeline=self.get_pipeline,
            random_state=self.random_state,
            X=X,
            y=y,
            get_splits=self.get_splits,
            logger=self.logger,
            hyperparameters=hyperparameters,
            predict_method=self.predict_method,
        )


def cross_validate_single_repeat(
    get_pipeline: Callable,
    X: DataFrame,
    y: Target,
    get_splits: TrainTestSplitter,
    random_state: int,
    predict_method: Optional[str] = "predict",
    fit_params: Optional[Dict] = None,
    train_test_filter_callback: Optional[Callable] = None,
    optimize: bool = False,
    optimize_params: OptimizationParams = OptimizationParams(),
    optimize_callbacks: Optional[Dict[str, Callable]] = None,
    hyperparameters: Optional[Mapping[str, Dict] | Dict] = None,
    return_models: bool = True,
    logger: Optional[logging.Logger] = None,
    mlflow: bool = False,
    reporter: Optional[ProgressReporter] = None,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Result:
    if fit_params is None:
        fit_params = {}

    if optimize_callbacks is None:
        optimize_callbacks = {}

    if verbose:
        print(".", end="", flush=True)

    random_seed(random_state)
    y_data = y.data if y is Dict else y

    if get_splits is None:
        splits = get_k_fold(X, y_data, random_state=random_state)
    else:
        splits = get_splits(X=X, y=y_data, random_state=random_state)

    if logger is None:
        logger = get_logger()

    n_jobs, n_jobs_rest = get_jobs(n_jobs, maximum=len(splits))
    splits_dict: Dict[Hashable, TrainTestIndex]

    if isinstance(splits, Sequence):
        splits_dict = list_to_dict_index(splits)
    elif isinstance(splits, Dict):
        splits_dict = splits
    else:
        raise TypeError("Incompatible predictions")

    if optimize:
        get_pipeline_ = partial(
            get_nested_optimization,
            get_pipeline=get_pipeline,
            optimize_params=optimize_params,
            logger=logger,
            predict_method=predict_method,
            mlflow=mlflow,
        )
    else:
        get_pipeline_ = get_pipeline

    models = valmap(
        lambda _: get_pipeline_(X=X, y=y, random_state=random_state),
        splits_dict,
    )

    if hyperparameters is not None:
        for split_name, model in models.items():
            model.set_params(**configuration_to_params(hyperparameters))

    column_masks = get_column_mask(splits_dict, X, train_test_filter_callback)

    if logger is not None:
        removed_features = pipe(
            column_masks,
            valmap(valfilter(identity)),
            valmap(lambda k: list(k.keys())),
        )
        logger.debug("\n" + yaml.dump(removed_features))

    models = cross_validate_train(
        X,
        y,
        models,
        splits_dict,
        column_masks,
        n_jobs=n_jobs,
        mlflow=mlflow,
        logger=logger,
        random_state=random_state,
        fit_params=fit_params,
        reporter=reporter,
    )

    runs = list(
        cross_validate_predict(
            X,
            y,
            splits_dict,
            column_masks,
            models,
            predict_method,
            mlflow,
            return_models,
        )
    )

    if return_models is False:
        for _, model in models.values():
            del model
        del models
        gc.collect()

    if isinstance(splits, Dict):
        result = list_to_dict_by_keys(runs, splits.keys())
    else:
        result = list_to_dict_index(runs)

    if mlflow:
        metrics: Metrics = compute_metrics_result(result, y)
        log_metrics(metrics)

    return result


def get_predictions(results: List[Result]) -> Iterable[Prediction]:
    for result in results:
        for prediction in result.values():
            yield prediction


def external_validation(
    X_test: DataFrame,
    y_test: Target,
    results: List[Result],
    on_progress: Callable[[float], None] = None,
) -> List[Result]:
    predictions = list(get_predictions(results))

    if on_progress is not None:
        reporter: Optional[ProgressReporter] = ProgressReporter()
        reporter.total = len(predictions)
        reporter.on_progress = on_progress
    else:
        reporter = None

    new_results = []
    for result in results:
        new_result = {}
        for split_name, prediction in result.items():
            new_prediction = copy(prediction)
            del new_prediction["split"]

            predict_method_callable = getattr(prediction["model"], "predict")
            prediction = Prediction(
                split=([], X_test.index.tolist()),
                X_columns=prediction["X_columns"],
                y_column=prediction["y_column"],
                y_pred=DataFrame(
                    predict_method_callable(X_test),
                    index=X_test.index,
                ),
                model=prediction["model"],
            )

            new_result[split_name] = prediction

            if reporter is not None:
                reporter.finished()

        new_results.append(new_result)

    return new_results


def _objective_instantiate(self, trial, hyperparameters):
    if CROSS_VALIDATE_KEY in hyperparameters:
        cv_hyperparameters = hyperparameters[CROSS_VALIDATE_KEY]
        rest_hyperparameters = dissoc(hyperparameters, CROSS_VALIDATE_KEY)
    else:
        cv_hyperparameters = {}
        rest_hyperparameters = hyperparameters

    trial.set_user_attr("cv_hyperparameters", cv_hyperparameters)
    trial.set_user_attr("hyperparameters", rest_hyperparameters)

    return trial, lambda X, _random_state: self._instantiate_from_hyperparameters(
        rest_hyperparameters,
        X,
        _random_state,
    )


def optimize_per_split(
    get_optimize: Callable[..., Optimize],
    get_splits: Callable[[DataFrame, Target], Dict[Hashable, TrainTestIndex]],
    X: DataFrame,
    y: Target,
    mlflow_track: bool = False,
    n_jobs: int = -1,
) -> Dict[Hashable, Optimize]:
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(get_splits(X, y)))

    splits = get_splits(X, y)

    fold_data = {
        fold_name: (
            fold_name,
            get_optimize(get_splits=always({"train_test": fold})),
            X,
            y,
            mlflow_track,
        )
        for fold_name, fold in splits.items()
    }

    return run_parallel(cross_validate_fit, fold_data, n_jobs)


def optimize_per_group(
    get_optimize: Callable[[], Optimize],
    X: DataFrameGroupBy,
    y: Target,
    mlflow_track: bool = False,
    n_jobs: int = -1,
) -> Dict[Hashable, Optimize]:
    if n_jobs == -1:
        n_jobs = cpu_count()
    fold_data = {
        group_name: (
            group_name,
            get_optimize(),
            group_X,
            loc(group_X.index, y),
            mlflow_track,
        )
        for group_name, group_X in X
    }
    return run_parallel(cross_validate_fit, fold_data, n_jobs)


ExecutePerGroupT = TypeVar("ExecutePerGroupT")


def execute_per_group(
    callback: Callable[[str, DataFrame, Target], ExecutePerGroupT],
    X_group_by: DataFrameGroupBy,
    y: Target,
    n_jobs: int = -1,
) -> Dict[Hashable, ExecutePerGroupT]:
    if n_jobs == -1:
        n_jobs = cpu_count()
    fold_data = {
        group_name: (
            group_X,
            loc(group_X.index, y),
        )
        for group_name, group_X in X_group_by
    }
    return run_parallel(callback, fold_data, n_jobs)


class OptimizeEstimator(Optimize, Estimator):
    def __getstate__(self):
        return self.__dict__

    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)

        for trial in self.trials:
            self.fit_best_model = self.get_pipeline(
                X=X, y=y, random_state=self.random_state
            )
            self.fit_best_model.set_params(
                **configuration_to_params(trial.user_attrs["hyperparameters"])
            )
            self.fit_best_model.fit(X, y)

            return self

        # TODO:
        del self.study

        raise RuntimeError("No trials successful")

    @property
    def trials(self):
        return pipe(
            self.study.trials,
            partial(filter, lambda _trial: not is_noneish(_trial.value)),
            partial(sorted, key=lambda _trial: _trial.value, reverse=True),
        )

    def predict(self, X):
        return self.fit_best_model.predict(X)

    def predict_proba(self, X):
        return self.fit_best_model.predict_proba(X)

    def predict_survival_at_time(self, X, *args, **kwargs):
        return self.fit_best_model.predict_survival_at_time(X, *args, **kwargs)

    def score(self, X, y):
        return self.fit_best_model.score(X, y)

    def transform(self, X: DataFrame):
        return self.fit_best_model.transform(X)

    def __getattr__(self, item):
        if item == "fit_best_model":
            return None

        if hasattr(self.fit_best_model, item):
            return getattr(self.fit_best_model, item)
        else:
            raise AttributeError(f"AttributeError: object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.fit_best_model[item]

    def get_final(self):
        try:
            return self.fit_best_model.get_final()
        except AttributeError:
            return self.fit_best_model

    def __instancecheck__(self, instance):
        return isinstance(self.fit_best_model, instance)

    def get_feature_importance(self):
        return self.fit_best_model.get_feature_importance()


def get_removed_features_from_mask(
    column_masks: Dict[Hashable, Dict[Hashable, bool]],
) -> Dict[Hashable, List[str]]:
    return pipe(
        column_masks,
        valmap(lambda masks: valfilter(identity, masks)),
        valmap(lambda k: list(k.keys())),
    )


def get_column_mask(
    splits: TrainTestSplits,
    X: DataFrame,
    train_test_filter_callback: Callable[[Series, Series], bool] = None,
) -> Dict[Hashable, Dict[Hashable, bool]]:
    if train_test_filter_callback:
        return dict(
            get_column_mask_filter(
                X,
                splits,
                train_test_filter_callback,
            )
        )
    else:
        return get_columns_mask_default(X, splits)


def get_columns_mask_default(
    X: DataFrame,
    splits: TrainTestSplits,
) -> Dict[Hashable, Dict[Hashable, bool]]:
    return valmap(lambda _: {column: False for column in X.columns}, splits)


def get_column_mask_filter(
    X: DataFrame,
    splits: TrainTestSplits,
    _train_test_filter: Callable[[DataFrame, DataFrame], bool],
) -> Iterable[Tuple[Hashable, Dict[Hashable, bool]]]:
    for fold_name, (train, test) in splits.items():
        X_train = X.loc[train]
        X_test = X.loc[test]
        if len(X_test) == 0 or len(X_train) == 0:
            yield fold_name, {}
        else:
            yield (
                fold_name,
                {
                    column_name: _train_test_filter(
                        X_train[column_name],
                        X_test[column_name],
                    )
                    for column_name in X
                },
            )


def get_nested_optimization(
    X: DataFrame,
    y: Target,
    random_state: int,
    get_pipeline: Callable,
    predict_method: str = "predict",
    optimize_params: OptimizationParams = OptimizationParams(),
    mlflow: Union[str, bool] = False,
    logger: Logger = None,
):
    direction = optimize_params.direction

    objective_evaluate: OptimizeEvaluate = cast(
        OptimizeEvaluate,
        partial(
            optimize_params.objective,
        ),
    )

    callbacks = []

    if mlflow:
        set_tag("objective_metric", optimize_params.objective)
        callbacks.append(optuna_report_mlflow)

    if optimize_params.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_rounds=optimize_params.early_stopping_rounds,
                direction=direction,
                stop_callback=partial(log_early_stopping, logger=logger),
            )
        )

    return OptimizeEstimator(
        get_pipeline,
        objective_evaluate=objective_evaluate,
        optimize_params=optimize_params,
        get_splits=partial(optimize_params.get_splits, randon_state=random_state, X=X),
        optimize_callbacks=callbacks,
        logger=logger,
        random_state=random_state,
        direction=direction,
        predict_method=predict_method,
    )


def cross_validate_train(
    X: DataFrame,
    y: Target,
    models: Dict[Hashable, Estimator],
    splits_dict: TrainTestSplits,
    filtered_columns: Dict[Hashable, Dict[Hashable, bool]],
    random_state: int,
    n_jobs: int = -1,
    mlflow: bool = False,
    logger: logging.Logger = None,
    fit_params: Mapping = empty_dict,
    reporter: ProgressReporter = None,
) -> Dict[Hashable, Tuple[Optional[str], Estimator]]:
    n_jobs, _ = get_jobs(n_jobs, maximum=len(models))
    fold_data = {
        fold_name: (
            fold_name,
            models[fold_name],
            cross_validate_preprocess(
                loc(train_split, X, ignore_not_present=True, logger=logger),
                filtered_columns[fold_name],
            ),
            loc(train_split, y, ignore_not_present=True, logger=logger),
            random_state,
            mlflow,
            fit_params,
            reporter,
        )
        for fold_name, (train_split, test_split) in splits_dict.items()
    }

    return run_parallel(cross_validate_fit, fold_data, n_jobs)


def cross_validate_fit(
    split_name: str,
    estimator: Estimator,
    X: DataFrame,
    y: Target,
    random_state: int,
    mlflow: bool = False,
    fit_kwargs: Mapping = None,
    reporter: ProgressReporter = None,
) -> Tuple[Optional[str], Estimator]:
    if fit_kwargs is None:
        fit_kwargs = {}

    random_seed(random_state)

    if mlflow:
        with start_run(
            run_name=str(split_name),
            nested=True,
            experiment_id=get_active_experiment_id(),
        ) as run:
            run_id = run.info.run_id
            estimator.fit(X, y, **fit_kwargs)
    else:
        estimator.fit(X, y, **fit_kwargs)

        run_id = None

    if reporter:
        reporter.finished()

    return run_id, estimator


def cross_validate_preprocess(
    data: DataFrame,
    mask: Dict[Hashable, bool],
):
    return pipe(
        data,
        partial(cross_validate_apply_mask, mask),
    )


def cross_validate_predict(
    X: DataFrame,
    y: Target,
    splits: TrainTestSplits,
    filtered_columns: Dict[Hashable, Dict[Hashable, bool]],
    models: Dict[Hashable, Tuple[Optional[str], PredictionMethod]],
    predict_method: str,
    mlflow: bool = False,
    return_models: bool = True,
) -> Iterable[Prediction]:
    for name, split in splits.items():
        run_id, model = models[name]
        X_test = loc(
            split[1],
            X,
            ignore_not_present=True,
        )
        if len(X_test) == 0:
            y_pred = None
        else:
            predict_method_callable = getattr(model, predict_method)
            y_pred = DataFrame(
                predict_method_callable(X_test),
            )
            y_pred.index = X_test.index

        prediction = Prediction(
            split=split,
            X_columns=X.columns.tolist(),
            y_column=y.name,
            y_pred=y_pred,
            model=model if return_models else None,
        )

        if mlflow:
            with start_run(run_id=run_id, nested=True) as mlflow:
                metrics = compute_metrics_prediction(prediction, y)
                log_metrics_single(metrics)

        yield prediction


def series_to_target(series: Series) -> Target:
    return {"name": series.name, "data": series}


def get_pred_and_X_from_results(X, y, results):
    y_pred_y_pred_merged = get_pred_from_results(results)
    y_pred_complete = pandas.concat(
        [
            y.loc[y_pred_y_pred_merged.index],
            y_pred_y_pred_merged,
            X.loc[y_pred_y_pred_merged.index],
        ],
        axis=1,
    )
    return y_pred_complete


def get_pred_from_results(results):
    y_preds = []
    for repeat_n, result in enumerate(results):
        y_pred = get_pred_from_result(result)
        y_pred.name = f"Repeat {repeat_n}"
        y_preds.append(y_pred)
    y_pred_y_pred_merged = pandas.concat(y_preds)
    return y_pred_y_pred_merged


def get_pred_from_result(result: Result) -> DataFrame:
    y_scores = []
    for split_name, prediction in result.items():
        y_pred = prediction["y_pred"]
        y_scores.append(y_pred)
    return pandas.concat(y_scores)
