import itertools
import multiprocessing
import os
import random
import shelve
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from functools import singledispatch, partial, update_wrapper
from logging import Logger
from numbers import Real
from pathlib import Path
from pprint import pprint
from typing import Dict, Callable, Iterator, Tuple, Any, Iterable, TypeVar, List, Optional, Sequence, Hashable

import numpy
import numpy as np
import pandas
from IPython import get_ipython
from filelock import FileLock, UnixFileLock
from flask_socketio import SocketIO
from frozendict import frozendict
from humps import decamelize, camelize
from matplotlib import pyplot
from numpy import ndarray, recarray
from pandas import Series, DataFrame, Index
from pandas.core.groupby import DataFrameGroupBy

from hcve_lib.custom_types import SurvivalPairTarget, SplitPrediction, Target, SplitInput
from hcve_lib.functional import pipe, unzip

empty_dict: Mapping = frozendict()


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

    def open(self) -> 'LockedShelve':
        if self.shelve:
            self.shelve.close()

        data_path = Path(self.path)
        lock_folder = (data_path.parent / '.lock')
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
                new[camelize_adjusted(k)
                    if k[0] != "_" else k] = camelize_recursive(v)
            return new
        elif isinstance(d, list):
            return list(map(camelize_recursive, d))
        elif isinstance(d, tuple):
            return tuple(map(camelize_recursive, d))
        else:
            return d
    except TypeError as e:
        print('Error during processing', d)
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
            **{
                arg_name: decamelize_recursive(arg)
                for arg_name, arg in kwargs.items()
            },
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
        'type': name,
        'payload': payload,
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


IndexData = TypeVar('IndexData')


def index_data(indexes: Iterable[int], data: IndexData) -> IndexData:
    if isinstance(data, (ndarray, recarray)):
        return data[list(indexes)]
    elif isinstance(data, (DataFrame, Series)):
        return data.iloc[indexes]
    elif isinstance(data, List):
        return [item for index, item in enumerate(data)
                if index in indexes]  # type: ignore
    elif isinstance(data, Dict) and 'name' in data and 'data' in data:
        return {
            **data, 'data': index_data(indexes, data['data'])
        }  # type: ignore

    elif isinstance(data, SurvivalPairTarget):
        return (
            index_data(indexes, data[0]),
            index_data(indexes, data[1]),
        )  # type: ignore
    else:
        raise TypeError("Can't handle this type")


def loc(
    indexes: List[Hashable],
    data: IndexData,
    ignore_not_present: bool = False,
    logger: Logger = None,
) -> IndexData:
    if isinstance(data, (DataFrame, Series)):
        if ignore_not_present:
            actual_index = [index for index in indexes if index in data.index]
            if logger:
                removed_indexes = len(indexes) - len(actual_index)
                if removed_indexes > 0:
                    logger.warning(f'Removed samples {removed_indexes}')
        else:
            actual_index = indexes
            removed_indexes = None
        return data.loc[actual_index]
    elif isinstance(data, Dict) and 'data' in data:
        return {
            **data, 'data':
            loc(indexes, data['data'], ignore_not_present=ignore_not_present)
        }
    else:
        raise Exception()


ListToDictKey = TypeVar('ListToDictKey')
ListToDictValue = TypeVar('ListToDictValue')


def list_to_dict_by_keys(
    input_list: Iterable[ListToDictValue],
    keys: Iterable[ListToDictKey],
) -> Dict[ListToDictKey, ListToDictValue]:
    return {key: value for key, value in zip(keys, input_list)}


def list_to_dict_index(
        input_list: Sequence[ListToDictValue] \
) -> Dict[Hashable, ListToDictValue]:
    return {index: value for index, value in enumerate(input_list)}


SubtractListT = TypeVar('SubtractListT')


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
            lambda _key: flatten_data.index.get_loc(_key)
            if _key in flatten_data.index else -1)
        group_iloc_subset = group_iloc_subset[group_iloc_subset != -1]
        yield key, list(group_iloc_subset)
        current_index += len(group)


def map_groups_loc(
        groups: DataFrameGroupBy \
) -> Iterable[Tuple[Any, Index]]:
    for name, group in groups:
        yield name, group.index


def remove_column_prefix(X: DataFrame) -> DataFrame:
    return X.rename(
        lambda column_name: pipe(
            column_name,
            partial(remove_prefix, 'categorical__'),
            partial(remove_prefix, 'continuous__'),
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


TransposeDictT1 = TypeVar('TransposeDictT1')
TransposeDictT2 = TypeVar('TransposeDictT2')
TransposeDictValue = TypeVar('TransposeDictValue')

TransposeDictInput = Dict[
    TransposeDictT1, \
    Dict[TransposeDictT2, TransposeDictValue]
]


def transpose_dict(
    dictionary: TransposeDictInput
) -> Dict[TransposeDictT2, Dict[TransposeDictT1, TransposeDictValue]]:
    outer_keys = dictionary.keys()

    if len(outer_keys) == 0:
        return dictionary

    inner_keys = next(iter(dictionary.values())).keys()

    return {
        inner_key: {
            outer_key: dictionary[outer_key][inner_key]
            for outer_key in outer_keys
        }
        for inner_key in inner_keys
    }


def partial2_args(func, name: str = None, args=tuple(), kwargs=empty_dict):
    partial_func: Callable = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name:
        func.__name__ = name
    return partial_func


def partial2(func, name: str = None, *args, **kwargs):
    return partial2_args(func, name=name, args=args, kwargs=kwargs)


def split_data(
    X: DataFrame,
    y: Target,
    fold: SplitPrediction,
    remove_extended: bool = False,
    logger: Logger = None,
):

    X_train, X_test = get_X_split(X, fold, logger)

    y_train, y_test = get_y_split(y, fold, logger)

    if remove_extended:
        X_test_, y_test_ = limit_to_observed(y_train, X_test, y_test)
    else:
        y_test_ = y_test
        X_test_ = X_test

    return X_train, y_train, X_test_, y_test_


def get_X_split(
    X: DataFrame,
    fold: SplitPrediction,
    logger: Logger = None,
):
    split_train, split_test = filter_split_in_index(fold['split'], X.index)

    if logger:
        if removed_split_train := len(split_train) - len(fold['split'][0]):
            logger.warning(f'Removed {removed_split_train} from X train set')

        if removed_split_test := len(split_test) - len(fold['split'][0]):
            logger.warning(f'Removed {removed_split_test} from X test set')

    X_ = X[fold['X_columns']]

    X_train = loc(split_train, X_)
    X_test = loc(split_test, X_)

    if isinstance(fold.get('y_score'), Series):
        X_test = loc(fold['y_score'].index, X_test, ignore_not_present=True)

    if logger:
        log_additional_removed(
            X_test,
            fold['y_score'],
            logger,
            'from X test set',
        )

    return X_train, X_test


def get_y_split(
    y: Target,
    fold: SplitPrediction,
    logger: Logger = None,
):
    split_train, split_test = filter_split_in_index(
        fold['split'],
        y['data'].index,
    )

    if logger:
        if removed_split_train := len(split_train) - len(fold['split'][0]):
            logger.warning(f'Removed {removed_split_train} from y train set')

        if removed_split_test := len(split_test) - len(fold['split'][0]):
            logger.warning(f'Removed {removed_split_test} from y test set')

    y_train = loc(split_train, y)
    y_test = loc(split_test, y)

    if isinstance(fold.get('y_score'), Series):
        y_test = loc(fold['y_score'].index, y_test, ignore_not_present=True)

    if logger:
        log_additional_removed(
            y['data'],
            fold['y_score'],
            logger,
            'from y test set',
        )

    return y_train, y_test


def log_additional_removed(X_test, y_score, logger, message):
    removed_rows = len(X_test) - len(y_score)
    if removed_rows > 0:
        logger.warning(f'Removed additional {removed_rows} {message}')


def limit_to_observed(y_train, X_test, y_test):
    tte_train = get_tte(y_train)
    tte_test = get_tte(y_test)
    mask = tte_test <= max(tte_train)
    y_test_ = {**y_test, 'data': y_test['data'][mask]}
    X_test_ = X_test[mask]
    return X_test_, y_test_


def filter_split_in_index(split: SplitInput, index: Index) -> SplitInput:
    return filter_in_index(split[0], index), filter_in_index(split[1], index)


def filter_in_index(iterable: List, index: Index) -> List:
    return [i for i in iterable if i in index]


def get_tte(target: Target) -> np.ndarray:
    if isinstance(target, Dict):
        return get_tte(target['data'])
    elif isinstance(target, DataFrame):
        return target['tte']
    else:
        raise TypeError(f'Unsupported {target.__class__}')


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
        if (folder / 'Pipfile').exists():
            os.chdir(folder)
            break


def notebook_init():
    cwd_root()
    pandas.set_option("display.max_columns", None)
    pyplot.rcParams['figure.facecolor'] = 'white'

    ipython = get_ipython()
    if 'autoreload' not in ipython.extension_manager.loaded:
        ipython.magic("load_ext autoreload")
    ipython.magic("autoreload  2")
    ipython.magic("matplotlib inline")
    ipython.magic("config InlineBackend.figure_format = 'retina'")


KeyT = TypeVar('KeyT')
ValueT = TypeVar('ValueT')


def get_key_by_value(dict: Dict[KeyT, ValueT], value: ValueT) -> KeyT:
    for _key, _value in dict.items():
        if value == _value:
            return _key

    raise Exception("Value not found")


def X_to_pytorch(X):
    import torch
    return torch.from_numpy(X.to_numpy().astype('float32')).to('cuda')


MapRecursiveFrom = TypeVar('MapRecursiveFrom')
MapRecursiveTo = TypeVar('MapRecursiveTo')


@singledispatch
def map_recursive(
    obj,
    mapper,
):
    return mapper(obj)


@map_recursive.register(list)
def _(
    obj: List[MapRecursiveFrom],
    mapper: Callable,
) -> List[MapRecursiveTo]:
    return list(map(mapper, obj))


@map_recursive.register(dict)
def _(
    obj: Dict[Hashable, MapRecursiveFrom],
    mapper: Callable,
) -> Dict[Hashable, MapRecursiveTo]:
    return {key: map_recursive(value, mapper) for key, value in obj.items()}


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


class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        # noinspection PyUnresolvedReferences
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NonDaemonProcess
        return proc


@contextmanager
def noop_context_manager(*args, **kwargs):
    yield


def noop(*args, **kwargs):
    pass


GetKeysSubsetT = TypeVar(
    'GetKeysSubsetT',
    bound=Dict[Hashable, Any],
)


def get_keys(
    keys: Iterable[Hashable],
    dictionary: GetKeysSubsetT,
) -> GetKeysSubsetT:
    return {key: dictionary[key] for key in keys}  # type: ignore
