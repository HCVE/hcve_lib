import shelve
from copy import copy

import numpy
from collections.abc import Mapping
from filelock import FileLock, UnixFileLock
from flask_socketio import SocketIO
from frozendict import frozendict
from functools import singledispatch, partial, update_wrapper
from humps import decamelize, camelize
from numbers import Real
from numpy import ndarray, recarray
from pandas import Series, DataFrame, Index
from pandas.core.groupby import DataFrameGroupBy
from pathlib import Path
from pprint import pprint
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Dict, Callable, Iterator, Tuple, Any, Iterable, TypeVar, List, Optional

from hcve_lib.custom_types import SurvivalPairTarget
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


class Debug(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fit_callback: Callable[[DataFrame], Any] = None,
        transform_callback: Callable[[DataFrame], Any] = None,
        set_breakpoint_fit: bool = False,
    ):
        self.set_breakpoint_fit = set_breakpoint_fit
        self.fit_callback = fit_callback
        self.transform_callback = transform_callback
        super()

    def transform(self, X):
        if self.transform_callback:
            self.transform_callback(X)
        else:
            print('transform', X)
        return X

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X)
        else:
            print('fit', X)
        if self.set_breakpoint_fit:
            breakpoint()
        return self


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
    elif isinstance(data, SurvivalPairTarget):
        return (
            index_data(indexes, data[0]),
            index_data(indexes, data[1]),
        )
    else:
        raise TypeError("Can't handle this type")


ListToDictKey = TypeVar('ListToDictKey')
ListToDictValue = TypeVar('ListToDictValue')


def list_to_dict_by_keys(
    input_list: List[ListToDictValue],
    keys: Iterable[ListToDictKey],
) -> Dict[ListToDictKey, ListToDictValue]:
    return {key: value for key, value in zip(keys, input_list)}


def list_to_dict_index(
        input_list: List[ListToDictValue] \
) -> Dict[int, ListToDictValue]:
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
            lambda _key: flatten_data.index.get_loc(_key)).tolist()
        yield key, group_iloc_subset
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


def percent_missing(series: Series) -> float:
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


def partial2(func, name: str = None, args=tuple(), kwargs=empty_dict):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name:
        partial_func.__name__ = name
    return partial_func