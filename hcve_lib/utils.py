import uuid

import argparse
import asyncio
import pickle
from datetime import datetime

from toolz import keyfilter
import enum
import itertools
import multiprocessing
import os
import random
import shelve
from contextlib import contextmanager
from copy import copy
from functools import singledispatch, partial as partial_, update_wrapper
from logging import Logger
from multiprocessing.pool import Pool
from numbers import Real, Number
from pathlib import Path
from pprint import pprint
from typing import (
    Dict,
    Callable,
    Iterator,
    Tuple,
    Any,
    Iterable,
    TypeVar,
    List,
    Optional,
    Sequence,
    Hashable,
    Union,
)

import numpy
import numpy as np
import pandas
import pandas as pd

from filelock import FileLock, UnixFileLock
from flask_socketio import SocketIO
from frozendict import frozendict
from humps import decamelize, camelize

# from imblearn.over_sampling.base import BaseOverSampler
from matplotlib import pyplot
from numpy import ndarray, recarray, isnan
from pandas import Series, DataFrame, Index
from pandas.core.groupby import DataFrameGroupBy
from scipy.stats import t
from toolz import valmap

from hcve_lib.custom_types import (
    SurvivalPairTarget,
    Prediction,
    Target,
    TrainTestIndex,
    Result,
    Estimator,
)
from hcve_lib.functional import pipe, unzip, flatten

empty_dict: Dict = frozendict()


class LockedShelve:
    lock: Optional[UnixFileLock]
    shelve: Optional[shelve.Shelf]

    def __init__(self, path: str, *args, **kwargs):
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.lock = None
        self.shelve = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> "LockedShelve":
        if self.shelve:
            self.shelve.close()

        data_path = Path(self.path)
        lock_folder = data_path.parent / ".lock"
        lock_folder.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(lock_folder / data_path.name))
        self.lock.acquire()
        self.shelve = shelve.open(self.path, *self.args, **self.kwargs)
        return self

    def close(self):
        self.shelve.close()
        self.shelve = None
        self.lock.release()

    def __setitem__(self, key, item):
        self.shelve[key] = item

    def __getitem__(self, key):
        return self.shelve[key]

    def __repr__(self):
        return repr(self.shelve)

    def __len__(self):
        return len(self.shelve)

    def __delitem__(self, key):
        del self.shelve[key]

    def clear(self):
        return self.shelve.clear()

    def copy(self):
        return copy(self.shelve)

    def has_key(self, k):
        return k in self.shelve

    def update(self, *args, **kwargs):
        return self.shelve.update(*args, **kwargs)

    def keys(self):
        return self.shelve.keys()

    def values(self):
        return self.shelve.values()

    def items(self):
        return self.shelve.items()

    def __iter__(self):
        return iter(self.shelve)


@singledispatch
def show(what):
    print(what)


@show.register
def _(what: dict):
    pprint(what)


def get_class_ratios(series: Series) -> Dict:
    class_counts = series.value_counts()
    total = class_counts.sum()
    ratios = class_counts.apply(lambda count: 1 / (count / total))
    return (ratios / ratios.min()).to_dict()


def get_class_ratio(series: Series) -> float:
    return get_class_ratios(series)[1]


def get_fractions(series: Series):
    counted = series.value_counts()
    return counted / counted.sum()


def decamelize_recursive(d):
    if isinstance(d, dict):
        new = {}
        for k, v in d.items():
            new[decamelize(k) if k[0] != "_" else k] = decamelize_recursive(v)
        return new
    elif isinstance(d, list):
        return list(map(decamelize_recursive, d))
    else:
        return d


def camelize_recursive(d):
    try:
        if isinstance(d, dict):
            new = {}
            for k, v in d.items():
                new[camelize_adjusted(k) if k[0] != "_" else k] = camelize_recursive(v)
            return new
        elif isinstance(d, list):
            return list(map(camelize_recursive, d))
        elif isinstance(d, tuple):
            return tuple(map(camelize_recursive, d))
        else:
            return d
    except TypeError as e:
        print("Error during processing", d)
        raise e


def camelize_adjusted(string: str) -> str:
    if len(string) == 1:
        return string
    else:
        return camelize(string)


def decamelize_arguments(function: Callable) -> Callable:
    def decamelize_arguments_(*args, **kwargs):
        return function(
            *[decamelize_recursive(arg) for arg in args],
            **{arg_name: decamelize_recursive(arg) for arg_name, arg in kwargs.items()},
        )

    return decamelize_arguments_


def camelize_return(function: Callable) -> Callable:
    def camelize_return_(*args, **kwargs):
        return camelize_recursive(function(*args, **kwargs))

    return camelize_return_


def to_plain_decorator(function: Callable) -> Callable:
    def to_plain_decorator_(*args, **kwargs):
        return to_plain(function(*args, **kwargs))

    return to_plain_decorator_


def get_event_listener(socketio: SocketIO):
    def event_listener_1(*socketio_args, **socketio_kwargs) -> Callable:
        def event_listener_2(function: Callable):
            return pipe(
                function,
                decamelize_arguments,
                camelize_return,
                to_plain_decorator,
                lambda function_: socketio.on(
                    *socketio_args,
                    **socketio_kwargs,
                )(function_),
            )

        return event_listener_2

    return event_listener_1


def make_event(name: str, payload: Dict, meta=empty_dict) -> Dict:
    return {
        "type": name,
        "payload": payload,
        **meta,
    }


def to_plain(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_plain(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, range):
        return list(obj)
    else:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating) and not numpy.isnan(obj):
            return float(obj)
        else:
            try:
                if numpy.isnan(float(obj)):
                    return None
                else:
                    return obj
            except (ValueError, TypeError):
                return obj


def map_column_names(df: DataFrame, callback: Callable) -> DataFrame:
    return df.rename(callback, axis=1)


def cumulative_count(series: Series) -> Iterator[Tuple[Real, float]]:
    series_no_nan = series.dropna()
    count = 0
    for value in series_no_nan.sort_values():
        count += 1
        yield value, (count / len(series))


def inverse_cumulative_count(series: Series) -> Iterator[Tuple[Real, float]]:
    values, fractions = unzip(cumulative_count(series))
    for value, fraction in zip(values, reversed(fractions)):
        yield value, fraction


def key_value_swap(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


IndexData = TypeVar("IndexData")


def index_data(indexes: Iterable[int], data: IndexData) -> IndexData:
    if isinstance(data, (ndarray, recarray)):
        return data[list(indexes)]
    elif isinstance(data, (DataFrame, Series)):
        return data.iloc[indexes]
    elif isinstance(data, List):
        return [item for index, item in enumerate(data) if index in indexes]  # type: ignore
    elif isinstance(data, Dict) and "name" in data and "data" in data:
        return {**data, "data": index_data(indexes, data["data"])}  # type: ignore

    elif isinstance(data, SurvivalPairTarget):
        return (
            index_data(indexes, data[0]),
            index_data(indexes, data[1]),
        )  # type: ignore
    else:
        raise TypeError("Can't handle this type")


def loc(
        index: Union[Index, List[int], ndarray],
        data: IndexData,
        ignore_not_present: bool = False,
        logger: Logger = None,
) -> IndexData:
    if isinstance(data, (DataFrame, Series)):
        if ignore_not_present:
            actual_index = [index for index in index if index in data.index]
            if logger:
                removed_indexes = len(index) - len(actual_index)
                if removed_indexes > 0:
                    logger.warning(f"Removed samples {removed_indexes}")
        else:
            actual_index = index
            removed_indexes = None
        return data.loc[actual_index]
    elif hasattr(data, "data"):
        return data.update_data(
            loc(
                index,
                data.data,
                ignore_not_present=ignore_not_present,
            )
        )
    else:
        raise Exception(f"Type '{type(data)}' not supported")


ListToDictKey = TypeVar("ListToDictKey")
ListToDictValue = TypeVar("ListToDictValue")


def list_to_dict_by_keys(
        input_list: Iterable[ListToDictValue],
        keys: Iterable[ListToDictKey],
) -> Dict[ListToDictKey, ListToDictValue]:
    return {key: value for key, value in zip(keys, input_list)}


def list_to_dict_index(
        input_list: Sequence[ListToDictValue],
) -> Dict[Hashable, ListToDictValue]:
    return {index: value for index, value in enumerate(input_list)}


SubtractListT = TypeVar("SubtractListT")


def subtract_lists(
        list1: List[SubtractListT],
        list2: List[SubtractListT],
) -> List[SubtractListT]:
    return [value for value in list1 if value not in list2]


def map_groups_iloc(
        groups: DataFrameGroupBy,
        flatten_data: DataFrame,
) -> Iterable[Tuple[Any, List[int]]]:
    current_index = 0
    for key, group in groups:
        group_iloc_subset = group.index.map(
            lambda _key: (
                flatten_data.index.get_loc(_key) if _key in flatten_data.index else -1
            )
        )
        group_iloc_subset = group_iloc_subset[group_iloc_subset != -1]
        yield key, list(group_iloc_subset)
        current_index += len(group)


def map_groups_loc(groups: DataFrameGroupBy) -> Iterable[Tuple[Any, Index]]:
    for name, group in groups:
        yield name, group.index


def remove_column_prefix(X: DataFrame) -> DataFrame:
    return X.rename(
        lambda column_name: pipe(
            column_name,
            partial_(remove_prefix, "categorical__"),
            partial_(remove_prefix, "continuous__"),
            partial_(remove_prefix, "remainder__"),
        ),
        axis=1,
    )


def remove_prefix(prefix: str, input_str: str) -> str:
    if input_str.startswith(prefix):
        return input_str[len(prefix):]
    else:
        return input_str[:]


def get_fraction_missing(series: Series) -> float:
    return len(series[series.isna()]) / len(series)


TransposeDictT1 = TypeVar("TransposeDictT1")
TransposeDictT2 = TypeVar("TransposeDictT2")
TransposeDictValue = TypeVar("TransposeDictValue")

TransposeDictInput = Dict[TransposeDictT1, Dict[TransposeDictT2, TransposeDictValue]]


def transpose_dict(
        dictionary: TransposeDictInput,
) -> Dict[TransposeDictT2, Dict[TransposeDictT1, TransposeDictValue]]:
    outer_keys = dictionary.keys()

    if len(outer_keys) == 0:
        return dictionary

    inner_keys = next(iter(dictionary.values())).keys()

    return {
        inner_key: {
            outer_key: dictionary[outer_key][inner_key] for outer_key in outer_keys
        }
        for inner_key in inner_keys
    }


T1 = TypeVar("T1")


def transpose_list_of_dicts(list_of_dicts):
    if not list_of_dicts:
        return {}

    # Initialize the result dictionary with empty lists for each key
    transposed_dict = {key: [] for key in list_of_dicts[0]}

    # Populate the lists with values from each dictionary
    for d in list_of_dicts:
        for key, value in d.items():
            transposed_dict[key].append(value)

    return transposed_dict


def transpose_list(l: List[List[T1]]) -> List[List[T1]]:
    return list(map(list, itertools.zip_longest(*l, fillvalue=None)))


def partial_args(func, name: str = None, args=tuple(), kwargs=empty_dict):
    partial_func: Callable = partial_(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name:
        func.__name__ = name
    return partial_func


def partial(func, *args, **kwargs):
    return partial_args(func, name=None, args=args, kwargs=kwargs)


def split_data(
        X: DataFrame,
        y: Target,
        prediction: Prediction,
        remove_extended: bool = False,
        logger: Logger = None,
):
    X_train, X_test = get_X_split(X, prediction, logger)

    y_train, y_test = get_y_split(y, prediction, logger)

    if remove_extended:
        X_test_, y_test_ = limit_to_observed(y_train, X_test, y_test)
    else:
        y_test_ = y_test
        X_test_ = X_test

    return X_train, y_train, X_test_, y_test_


def get_X_split(
        X: DataFrame,
        prediction: Prediction,
        logger: Logger = None,
):
    split_train, split_test = filter_split_in_index(prediction["split"], X.index)

    if logger:
        removed_split_train = len(split_train) - len(prediction["split"][0])
        if removed_split_train:
            logger.warning(f"Removed {removed_split_train} from X train set")

        removed_split_test = len(split_test) - len(prediction["split"][0])
        if removed_split_test:
            logger.warning(f"Removed {removed_split_test} from X test set")

    X_ = X[prediction["X_columns"]]

    X_train = loc(split_train, X_)
    X_test = loc(split_test, X_)

    if isinstance(prediction.get("y_pred"), Series):
        X_test = loc(prediction["y_pred"].index, X_test, ignore_not_present=True)

    if logger:
        log_additional_removed(
            X_test,
            prediction["y_pred"],
            logger,
            "from X test set",
        )

    return X_train, X_test


def get_y_split(
        y: Target,
        prediction: Prediction,
        logger: Logger = None,
):
    split_train, split_test = filter_split_in_index(
        prediction["split"],
        y.index,
    )

    if logger is not None:
        removed_split_train = len(split_train) - len(prediction["split"][0])
        if removed_split_train:
            logger.warning(f"Removed {removed_split_train} from y train set")

        removed_split_test = len(split_test) - len(prediction["split"][0])
        if removed_split_test:
            logger.warning(f"Removed {removed_split_test} from y test set")

    y_train = loc(split_train, y)
    y_test = loc(split_test, y)

    # TODO: Causing problems with BoostrapMetric
    # if isinstance(prediction.get('y_score'), Series):
    #     y_test = loc(prediction['y_score'].index, y_test, ignore_not_present=True)

    if logger:
        log_additional_removed(
            y,
            prediction["y_proba"],
            logger,
            "from y test set",
        )
    return y_train, y_test


def get_X_y_split(
        X: DataFrame, y: Target, prediction: Prediction, logger: Logger = None
) -> Tuple[DataFrame, DataFrame, Target, Target]:
    return get_X_split(X, prediction, logger) + get_y_split(y, prediction, logger)


def log_additional_removed(X_test, y_score, logger, message):
    removed_rows = len(X_test) - len(y_score)
    if removed_rows > 0:
        logger.warning(f"Removed additional {removed_rows} {message}")


def limit_to_observed(y_train, X_test, y_test):
    tte_train = get_tte(y_train)
    tte_test = get_tte(y_test)
    mask = tte_test <= max(tte_train)
    y_test_ = {**y_test, "data": y_test["data"][mask]}
    X_test_ = X_test[mask]
    return X_test_, y_test_


def filter_split_in_index(split: TrainTestIndex, index: Index) -> TrainTestIndex:
    return filter_in_index(split[0], index), filter_in_index(split[1], index)


def filter_in_index(iterable: List, index: Index) -> List:
    return [i for i in iterable if i in index]


def get_tte(target: Union[DataFrame, Dict]) -> np.ndarray:
    if isinstance(target, Dict):
        return get_tte(target["data"])
    elif isinstance(target, DataFrame):
        return target["tte"]
    else:
        raise TypeError(f"Unsupported {target.__class__}")


def cross_validate_apply_mask(
        mask: Dict[str, bool],
        data: DataFrame,
) -> DataFrame:
    new_data = data.copy()
    if set(mask.keys()) != set(data.columns):
        raise Exception("Keys do not match")

    for column_name, remove in mask.items():
        if remove:
            new_data.drop(column_name, axis=1, inplace=True)

    return new_data


def cwd_root():
    for folder in itertools.chain([Path.cwd()], Path.cwd().parents):
        if (folder / "Pipfile").exists():
            os.chdir(folder)
            break


def notebook_init():
    from IPython import get_ipython

    cwd_root()
    pandas.set_option("display.max_columns", None)
    pyplot.rcParams["figure.facecolor"] = "white"

    ipython = get_ipython()
    if "autoreload" not in ipython.extension_manager.loaded:
        ipython.magic("load_ext autoreload")
    ipython.magic("autoreload  2")
    ipython.magic("matplotlib inline")
    ipython.magic("config InlineBackend.figure_format = 'retina'")
    from IPython.core.ultratb import VerboseTB

    VerboseTB._tb_highlight = "bg:#622222"


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


def get_key_by_value(d: Dict[KeyT, ValueT], value: ValueT) -> KeyT:
    for _key, _value in d.items():
        if value == _value:
            return _key

    raise Exception("Value not found")


def X_to_pytorch(X):
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import torch

    return torch.from_numpy(X.to_numpy().astype("float32")).to("cuda")


def random_seed(seed: int) -> None:
    numpy.random.seed(seed)
    random.seed(seed)


class NonDaemonProcess(multiprocessing.Process):
    @property  # type: ignore
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        # noinspection PyUnresolvedReferences
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc


@contextmanager
def noop_context_manager(*args, **kwargs):
    yield


def noop(*args, **kwargs):
    pass


GetKeysSubsetT = TypeVar(
    "GetKeysSubsetT",
    bound=Dict[Hashable, Any],
)


def get_keys(
        keys: Iterable[Hashable],
        dictionary: GetKeysSubsetT,
) -> GetKeysSubsetT:
    return {key: dictionary[key] for key in keys}  # type: ignore


def sort_columns_by_order(
        data_frame: DataFrame,
        order: List[str],
) -> DataFrame:
    columns_not_present = [column for column in order if column not in data_frame]

    new_data_frame = data_frame.copy(deep=False)
    new_data_frame[columns_not_present] = np.nan

    return new_data_frame[order]


def sort_index_by_order(
        data_frame: DataFrame,
        order: List[str],
) -> DataFrame:
    index_not_present = [index for index in order if index not in data_frame.index]

    missing_df = DataFrame(index=index_not_present)

    new_data_frame = pandas.concat([missing_df, data_frame])

    return new_data_frame.loc[order]


def is_noneish(what: Any) -> bool:
    if what is None:
        return True
    elif isinstance(what, str):
        return False
    elif isinstance(what, float) and isnan(what):
        return True
    else:
        return False


class SaveEnum(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using SaveEnum")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using SaveEnum")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(SaveEnum, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


# class SurvivalResample(BaseOverSampler):
#
#     def __init__(self, resampler):
#         super().__init__()
#         self.resampler = resampler
#
#     def fit(self, X, y=None):
#         self.resampler.fit(X, y['data']['label'])
#         return self
#
#     def fit_resample(self, X, y):
#         return self._fit_resample(X, y)
#
#     def _fit_resample(self, X, y):
#         Xr, yr = self.resampler.fit_resample(
#             pandas.concat(
#                 [X, Series(X.index, index=X.index, name='index')],
#                 axis=1,
#             ),
#             y['data']['label'],
#         )
#         return loc(Xr['index'], X), loc(Xr['index'], y)


def binarize(s: Series, threshold: float) -> Series:
    return (s >= threshold).map({True: 1, False: 0})


def get_first_entry(something: Dict) -> Any:
    return something[next(iter(something))]


def get_first_item(something: Dict) -> Tuple:
    key = next(iter(something))
    return key, something[key]


def run_parallel(function: Callable, data: Dict, n_jobs: int = None) -> Dict:
    if n_jobs is None:
        n_jobs = min(len(data), multiprocessing.cpu_count())

    data_ = valmap(
        lambda args: args if isinstance(args, list) else args,
        data,
    )
    if n_jobs == 1:
        optimizers = list_to_dict_by_keys(
            itertools.starmap(
                function,
                data_.values(),
            ),
            data_.keys(),
        )
    else:
        with NonDaemonPool(min(len(data_), n_jobs)) as p:  # type: ignore
            optimizers = list_to_dict_by_keys(
                p.starmap(
                    function,
                    data_.values(),
                ),
                data_.keys(),
            )
    return optimizers


def apply_args_and_kwargs(function: Callable, args: List, kwargs: Dict):
    return function(*args, **kwargs)


def put_contents(file: str, content: str) -> None:
    with open(file, "w") as f:
        f.write(content)


def round_significant(value: float, places: int = 3) -> str:
    return "{:g}".format(float(("{:." + str(places) + "g}").format(value)))


def is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_categorical_columns(data: DataFrame) -> List:
    return [column for column, dtype in data.dtypes.items() if dtype == "category"]


def estimate_categorical_columns(data: DataFrame, limit: int = 10) -> List:
    categorical = []
    for name, column in data.items():
        if len(column.unique()) <= limit:
            categorical.append(name)
    return categorical


def estimate_categorical_and_continuous_columns(
        data: DataFrame, limit: int = 10
) -> Tuple:
    categorical = estimate_categorical_columns(data, limit)
    continuous = list(set(data.columns) - set(categorical))
    return categorical, continuous


def auto_convert_columns(data: DataFrame, limit: int = 10) -> DataFrame:
    categorical, continuous = estimate_categorical_and_continuous_columns(data, limit)
    data_new = data.copy()
    for column in continuous:
        data_new[column] = pd.to_numeric(data_new[column], errors="coerce").astype(
            "float"
        )

    for column in categorical:
        data_new[column] = data_new[column].astype("category")

    return data_new


class DictSubSet:
    def __init__(self, items: dict):
        self.items = items

    def __eq__(self, other):
        return self.items == {k: other[k] for k in self.items if k in other}

    def __repr__(self):
        return repr(self.items)


def get_predictions_from_results(results: List[Result]) -> Iterable[Prediction]:
    for result in results:
        for prediction in result.values():
            yield prediction


# TODO: test
def get_models_from_repeats(results: List[Result]) -> List[Estimator]:
    return list(flatten([get_models_from_result(result) for result in results]))


# TODO: test / structure
def get_models_from_result(result: Result) -> List[Estimator]:
    return [prediction["model"] for prediction in result.values()]


def get_mean_importance(models: List[Estimator]) -> DataFrame:
    importances = [forest.get_feature_importance() for forest in models]

    forest_importances_ = pandas.concat(importances, axis=1)

    forest_importance_avg = forest_importances_.mean(axis=1)
    forest_importance_std = forest_importances_.std(axis=1)

    return DataFrame(
        {"mean": forest_importance_avg, "std": forest_importance_std}
    ).sort_values("mean")


def is_numerical(o):
    try:
        return "float" in str(o.dtype) or "int" in str(o.dtype)
    except AttributeError:
        return False


def get_jobs(n_jobs, maximum=None):
    cpu_count_value = multiprocessing.cpu_count()
    if n_jobs == -1:
        if maximum is None:
            jobs_taken = cpu_count_value
        else:
            jobs_taken = min(cpu_count_value, maximum)
    else:
        jobs_taken = min(cpu_count_value, n_jobs, maximum)

    return jobs_taken, max(
        1,
        (
            cpu_count_value - jobs_taken
            if n_jobs == -1
            else min(n_jobs, cpu_count_value - jobs_taken)
        ),
    )


def get_pipeline_name(
        estimator: Any,
):
    try:
        return estimator.get_name()
    except (AttributeError, TypeError):
        try:
            return estimator[-1].get_name()
        except:
            return str(estimator)


def auto_convert_category(data: DataFrame) -> DataFrame:
    data_new = data.copy()
    for column in data_new.columns:
        if len(data_new[column].unique()) < 10:
            data_new.loc[:, column] = data_new[column].astype("category")
        else:
            try:
                data_new.loc[:, column] = data_new[column].astype("float")
            except (TypeError, ValueError):
                data_new.loc[:, column] = data_new[column].astype("category")
    return data_new


def upper_columns(df: DataFrame) -> DataFrame:
    return df.rename(
        columns=lambda column: (
            column.upper()
            if isinstance(column, str)
            else tuple([column_.upper() for column_ in column])
        )
    )


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    dict1_new = dict1.copy()
    for key in dict2.keys():
        if isinstance(dict1_new.get(key), dict) and isinstance(dict2.get(key), dict):
            dict1_new[key] = deep_merge_dicts(dict1_new[key], dict2[key])
        else:
            dict1_new[key] = dict2[key]
    return dict1_new


from typing import Any, Dict, Union


def deep_merge(obj1: Union[Dict, Any], obj2: Union[Dict, Any]) -> Union[Dict, Any]:
    # If one of them is not a dictionary or an instance, return obj2
    if not isinstance(obj1, (dict, object)) or not isinstance(obj2, (dict, object)):
        return obj2

    # If obj1 is a class instance, let's use its dictionary for merging but remember the original
    obj1_dict = obj1.__dict__ if hasattr(obj1, "__dict__") else obj1

    # If obj2 is a class instance, just use its dictionary for merging
    obj2_dict = obj2.__dict__ if hasattr(obj2, "__dict__") else obj2

    obj1_new = obj1_dict.copy()
    for key in obj2_dict.keys():
        if (
                isinstance(obj1_new.get(key), dict) and isinstance(obj2_dict.get(key), dict)
        ) or (
                hasattr(obj1_new.get(key), "__dict__")
                and hasattr(obj2_dict.get(key), "__dict__")
        ):
            obj1_new[key] = deep_merge(obj1_new[key], obj2_dict[key])
        else:
            obj1_new[key] = obj2_dict[key]

    # If obj1 was a class instance, update its attributes directly
    if hasattr(obj1, "__dict__"):
        for key, value in obj1_new.items():
            setattr(obj1, key, value)
        return obj1
    else:
        return obj1_new


import re


def convert_to_snake_case_keys(data: Union[Dict, List]) -> Union[Dict, List]:
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key[0].isupper():
                new_key = key
            else:
                new_key = convert_to_snake_case(key)
            new_dict[new_key] = convert_to_snake_case_keys(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_to_snake_case_keys(item) for item in data]
    else:
        return data


def convert_to_snake_case(name: str) -> str:
    return re.sub("(?!^)([A-Z]+)", r"_\1", name).lower()


def convert_to_camel_case_keys(data: Union[Dict, List]) -> Union[Dict, List]:
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            parts = key.split("_")
            if parts[0][0].islower():
                new_key = parts[0] + "".join(x.capitalize() for x in parts[1:])
            else:
                new_key = key
            new_dict[new_key] = convert_to_camel_case_keys(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_to_camel_case_keys(item) for item in data]
    else:
        return data


def convert_to_camel_case(name: str) -> str:
    return "".join(x.capitalize() or "_" for x in name.split("_"))


DELETE = "__DELETE__"


def update_from_diff(obj: Union[object, Dict], diff: Union[Dict, object]) -> None:
    vars_ = diff.items() if isinstance(diff, dict) else vars(diff).items()
    for key, value in vars_:
        hasattr_ = hasattr if not isinstance(obj, Dict) else lambda d, k: k in d
        get_attr_ = getattr if not isinstance(obj, Dict) else lambda d, k: d[k]
        set_attr_ = (
            setattr if not isinstance(obj, Dict) else lambda d, k, v: d.update(**{k: v})
        )

        try:
            if value == DELETE:
                del obj[key]
                continue
        except ValueError:
            pass

        if hasattr_(obj, key):
            attr = get_attr_(obj, key)
            if isinstance(attr, dict) and isinstance(value, dict):
                update_from_diff(attr, value)
            elif isinstance(attr, list) and isinstance(value, list):
                for i, v in enumerate(value):
                    if i < len(attr):
                        if isinstance(attr[i], dict) and isinstance(v, dict):
                            update_from_diff(attr[i], v)
                        else:
                            attr[i] = v
                    else:
                        attr.append(v)
            else:
                set_attr_(obj, key, value)
        else:
            set_attr_(obj, key, value)


def get_next_key(d: Dict, current_key: any) -> any:
    keys = list(d.keys())
    index = keys.index(current_key)
    if index + 1 >= len(keys):
        return keys[-2] if len(keys) >= 2 else None
    else:
        return keys[index + 1]


def print_structure(obj, indent=0, max_len=10):
    if isinstance(obj, list):
        if len(obj) > max_len:
            print("[...]")
        else:
            print("[")
            for item in obj:
                print_structure(item, indent + 1, max_len)
                print(" " * (indent + 1), end="")
            print("]")
    elif isinstance(obj, dict):
        if len(obj) > max_len:
            print("{...}")
        else:
            print("{")
            for key, value in obj.items():
                print(" " * (indent + 1) + str(key) + ":", end="")
                print_structure(value, indent + 2, max_len)
                print(" " * (indent + 1), end="")
            print("}")
    else:
        print(" " * indent + "<<" + type(obj).__name__ + ">>")


def aggregate_df_with_statistics(df):
    avg = df.mean(axis=1)
    std_dev = df.std(axis=1)
    n = df.shape[1]
    ci = t.interval(alpha=0.95, df=n - 1, loc=avg, scale=std_dev / np.sqrt(n))
    lower_ci, upper_ci = ci
    new_df = pd.DataFrame(
        {"avg": avg, "std_dev": std_dev, "lower_ci": lower_ci, "upper_ci": upper_ci}
    )
    new_df = new_df.sort_values("avg", ascending=False)
    return new_df


def aggregate_dfs_with_statistics(df):
    df = pandas.concat(df, axis=1)
    return aggregate_df_with_statistics(df)


def bootstrap_sample(X, y, random_state: bool):
    n = X.shape[0]
    indices = np.random.choice(n, n, replace=True)
    return X.iloc[indices], y.iloc[indices]


def kendall_tau(rank1: List[Number], rank2: List[Number]) -> float:
    if len(rank1) != len(rank2):
        raise ValueError("Rankings must have the same length.")

    n = len(rank1)
    concordant_pairs = 0
    discordant_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] < rank2[j]) or (
                    rank1[i] > rank1[j] and rank2[i] > rank2[j]
            ):
                concordant_pairs += 1
            elif (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or (
                    rank1[i] > rank1[j] and rank2[i] < rank2[j]
            ):
                discordant_pairs += 1

    tau = (concordant_pairs - discordant_pairs) / (0.5 * n * (n - 1))

    return tau


def average_kendall_tau(rankings: List[List[Number]]) -> float:
    num_rankings = len(rankings)
    total_tau = 0.0
    num_comparisons = 0

    for i in range(num_rankings):
        for j in range(i + 1, num_rankings):
            tau = kendall_tau(rankings[i], rankings[j])
            total_tau += tau
            num_comparisons += 1

    if num_comparisons == 0:
        return 0.0

    return total_tau / num_comparisons


class DummyLogger:
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def standardize_dataframe(df: DataFrame) -> DataFrame:
    df_standardized = (df - df.mean()) / df.std()
    return df_standardized


class ObjectWrapper:
    def __init__(self, obj):
        self._wrapped_obj = obj

    def __getattr__(self, name):
        return getattr(self._wrapped_obj, name)

    def __setattr__(self, name, value):
        if name == "_wrapped_obj":
            # Set attribute directly on the wrapper object
            super().__setattr__(name, value)
        else:
            # Set attribute on the wrapped object
            setattr(self._wrapped_obj, name, value)

    def __delattr__(self, name):
        delattr(self._wrapped_obj, name)

    def __str__(self):
        return str(self._wrapped_obj)

    def __repr__(self):
        return repr(self._wrapped_obj)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def generate_steps(start, n, num_steps):
    step_size = (n - start) / (num_steps - 1)
    steps = [round(start + i * step_size) for i in range(num_steps)]
    steps[-1] = n
    return steps


import importlib.util


def import_module_from_path(module_path):
    print(module_path)
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(module)
    return module


def get_variables_as_dict(module_path):
    imported_module = import_module_from_path(module_path)
    variables_dict = {
        attr: getattr(imported_module, attr)
        for attr in dir(imported_module)
        if not callable(getattr(imported_module, attr)) and not attr.startswith("__")
    }
    return variables_dict


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def omit(blacklist, d):
    return keyfilter(lambda k: k not in blacklist, d)


def retry_async(max_retries=15, retry_delay=1, exception=Exception):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exception as e:  # Catch the specified exception(s)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        raise e

        return wrapper

    return decorator


def find_key(d: Dict, target_key: Any) -> bool:
    if not isinstance(d, dict):
        return False

    if target_key in d:
        return True

    for key, value in d.items():
        if isinstance(value, dict) and find_key(value, target_key):
            return True
    return False


def merge_two_level_dict(d: dict) -> dict:
    merged_dict = {}
    for key1, inner_dict in d.items():
        for key2, value in inner_dict.items():
            merged_key = f"{key1}_{key2}"
            merged_dict[merged_key] = value
    return merged_dict


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


def find_unpicklable_attr(data, path=None) -> Optional[List[str]]:
    if path is None:
        path = []

    if is_picklable(data):
        return None

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + [key]
            result = find_unpicklable_attr(value, new_path)
            if result:
                return result

    elif isinstance(data, (list, tuple)):
        for idx, value in enumerate(data):
            new_path = path + [idx]
            result = find_unpicklable_attr(value, new_path)
            if result:
                return result

    elif isinstance(data, object):
        for attr_name in dir(data):
            if not attr_name.startswith("__"):
                new_path = path + [attr_name]
                attr_value = getattr(data, attr_name)
                if not is_picklable(attr_value):
                    return new_path
                result = find_unpicklable_attr(attr_value, new_path)
                if result:
                    return result
    return None


def split_dict_by_keys(d, keys):
    subset = {k: d[k] for k in keys if k in d}
    remaining = {k: v for k, v in d.items() if k not in keys}
    return subset, remaining


def dump_results(pipeline_name, metrics, results, get_splits, dump_tag=None):
    if dump_tag is None:
        dump_tag = []

    first_metric_key, first_metric_value = get_first_item(metrics)
    dump_name = f"output/{pipeline_name}{(' ' + ' '.join(dump_tag)) if len(dump_tag) > 0 else ''} {get_splits.__name__} {get_date_time()} {first_metric_key}={first_metric_value['mean']:.2f}.pkl"
    with open(dump_name, "wb") as f:
        pickle.dump(results, f)


def get_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def count_lines(input_string: str) -> int:
    return len(input_string.splitlines())


def compute_classification_scores_statistics(
        predictions: Dict[Hashable, DataFrame]
) -> Dict[Hashable, Dict[str, float]]:
    statistics = {}
    for key, prediction in predictions.items():
        statistics[key] = {
            "mean": float(get_1_class_y_score(prediction).mean()),
            "median": float(get_1_class_y_score(prediction).median()),
            "std": float(get_1_class_y_score(prediction).std()),
            "min": float(get_1_class_y_score(prediction).min()),
            "max": float(get_1_class_y_score(prediction).max()),
        }
    return statistics


def average_classification_scores(predictions: Dict[Hashable, DataFrame]) -> DataFrame:
    results = None
    for prediction in predictions.values():
        if results is None:
            results = prediction
        else:
            results += prediction
    return results / len(results)


def get_1_class_y_score(y_score: Union[DataFrame, Series]) -> Series:
    if isinstance(y_score, Series):
        return y_score
    else:
        if len(y_score.columns) == 1:
            return y_score.iloc[:, 0]
        else:
            return y_score.iloc[:, 1]


def flatten_dict(d: Dict, parent_key: str = "", delimiter: str = " ") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{delimiter}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, delimiter=delimiter).items())
        else:
            items.append((new_key, v))
    return dict(items)


def configuration_to_params(dictionary: Dict) -> Dict:
    return_value = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                return_value["%s__%s" % (key, key2)] = value2
        else:
            return_value[key] = value

    return return_value


def camelize_and_capitalize(s):
    camelized = camelize(s)
    return camelized[0].upper() + camelized[1:]


def make_id() -> str:
    return str(uuid.uuid4())
