import inspect
import pprint
from copy import copy
from functools import reduce, partial, wraps
from itertools import starmap as starmap_itertools, tee
from typing import Callable, TypeVar, Iterable, Tuple, Dict, List, Any, Union, Sequence

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series, isna
from toolz import curry, valfilter, valmap

from hcve_lib.custom_types import IndexAccess

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def t(arg, callback=None):
    if callback:
        callback(arg)
    else:
        print(arg)
    return arg


def or_fn(*fns: Callable[..., bool]) -> Callable[..., bool]:
    return lambda *args: reduce(
        lambda current_value, fn: current_value or fn(*args), fns, False)


def star_args(function: Callable[..., T1]) -> Callable[[Iterable], T1]:
    def unpacked(args):
        return function(*args)

    return unpacked


def dict_from_items(items: Iterable[Tuple[T1, T2]]) -> Dict[T1, T2]:
    out_dict = {}
    for key, value in items:
        out_dict[key] = value
    return out_dict


def list_from_items(items: Iterable[Tuple[int, T1]]) -> List[T1]:
    out_list: List[T1] = []
    for key, value in items:
        out_list.insert(key, value)
    return out_list


@curry
def mapl(func, iterable):
    return list(map(func, iterable))


def mapi(func, iterable):
    return map(star_args(func), enumerate(iterable))


def map_tuples(callback: Callable[..., T2],
               iterable: Iterable[T1]) -> Iterable[T2]:
    return map(star_args(callback), iterable)  # type: ignore


def dict_subset(list_keys: List[str], dictionary: dict) -> dict:
    return {k: dictionary[k] for k in list_keys}


def dict_subset_list(list_keys: List[str], dictionary: dict) -> List:
    return [dictionary[k] for k in list_keys]


def flatten(iterable_outer: Iterable[Union[Iterable[T1], T1]]) -> Iterable[T1]:
    for iterable_inner in iterable_outer:
        if isinstance(iterable_inner,
                      Iterable) and not isinstance(iterable_inner, str):
            for item in iterable_inner:
                yield item
        else:
            yield iterable_inner  # type: ignore


# TODO: moore efficient implementation
def flatten_recursive(
        list_of_lists: Union[Sequence, ndarray]) -> Union[Sequence, ndarray]:
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], Sequence) or isinstance(
            list_of_lists[0], np.ndarray):
        return [
            *flatten_recursive(list_of_lists[0]),
            *flatten_recursive(list_of_lists[1:])
        ]
    return [*list_of_lists[:1], *flatten_recursive(list_of_lists[1:])]


def if_(condition: Any, then: Callable, els: Callable = None):
    if condition:
        return then()
    elif els:
        return els()


def find(callback: Callable[[T1], bool], list_to_search: Iterable[T1]) -> T1:
    return next(filter(callback, list_to_search))


def find_index(
    callback: Callable[[float], bool],
    list_to_search: Union[List[float], str],
    reverse=False,
) -> int:
    if reverse:
        iterable = add_index_reversed(list_to_search)
    else:
        iterable = add_index(list_to_search)
    return next(filter(lambda item: callback(item[1]), iterable))[0]


def add_index(iterable: Iterable) -> Iterable:
    for index, item in enumerate(iterable):
        yield index, item


def add_index_reversed(iterable: Union[List, str]) -> Iterable:
    for index in reversed(range(len(iterable))):
        yield index, iterable[index]


def do_nothing():
    pass


def pass_value() -> Callable[[T1], T1]:
    def pass_value_callback(value):
        return value

    return pass_value_callback


def in_ci(string: str, sequence: Union[List, str]) -> bool:
    normalized_sequence = [item.upper() for item in sequence] if isinstance(
        sequence, List) else sequence.upper()
    return string.upper() in normalized_sequence


def partial_method(method: Callable, *args, **kwargs) -> Callable:
    def partial_callback(self: object):
        method(self, *args, **kwargs)

    return partial_callback


def pipe(*args: Any, log=False) -> Any:
    current_value = args[0]
    for function in args[1:]:
        current_value = function(current_value)
        if log:
            print(
                f'\'{function.__name__}\' with input \n {pprint.pformat(current_value)}'
            )
    return current_value


def piped(*args: Any) -> Any:
    current_value = args[0]
    for function in args[1:]:
        breakpoint()
        current_value = function(current_value)
    return current_value


def lambda_with_consts(define, to):
    return lambda: to(*define)


def pass_args(define, to):
    return to(*define)


def define_consts(to: Callable[..., T1], **variables) -> T1:
    return to(**variables)


def statements(*args: Any) -> Any:
    return args[-1]


def filter_keys(keys: Iterable[Any], dictionary: Dict) -> Dict:
    return {key: dictionary[key] for key in keys}


def unzip(iterable: Iterable) -> Iterable:
    return zip(*iterable)


def try_except(
    try_clause: Callable,
    except_clauses: Union[Dict, Callable],
) -> Any:
    # noinspection PyBroadException
    try:
        return try_clause()
    # noinspection PyBroadException
    except Exception as e:
        if callable(except_clauses):
            except_clauses(e)
        else:
            for ExceptionClass, except_clause in except_clauses.items():
                if isinstance(e, ExceptionClass):
                    return except_clause()
            raise e


def merge_by(callback: Callable, sequence: Iterable) -> Any:
    sequence_iterable = iter(sequence)
    last_item = next(sequence_iterable)
    for item in sequence_iterable:
        last_item = callback(last_item, item)
    return last_item


TIndexAccess = TypeVar('TIndexAccess', bound=IndexAccess)


def assign_index(something: TIndexAccess, index: Any,
                 value: Any) -> TIndexAccess:
    something_copied = copy(something)
    something_copied[index] = value
    return something_copied


def skip_first(input_iterable: Iterable) -> Iterable:
    is_first = True
    for item in input_iterable:
        if is_first:
            is_first = False
            continue
        yield item


def iter_is_last(input_iterable: Iterable[T1]) -> Iterable[Tuple[bool, T1]]:
    iterator = iter(input_iterable)
    previous_value = next(iterator)
    for value in iterator:
        yield False, previous_value
        previous_value = value
    yield True, previous_value


@curry
def find_index_right(function: Callable, list_: List) -> Any:
    for index, value in reversed(list(enumerate(list_))):
        if function(index, value):
            return index, value


def does_objects_equal(obj1: Any, obj2: Any) -> bool:
    try:
        obj1dict = obj1.__dict__
        obj2dict = obj2.__dict__
    except AttributeError:
        return obj1 == obj2
    else:
        return obj2dict == obj1dict


def raise_exception(exception: Any) -> None:
    raise exception


def compact(iterable: Iterable) -> Iterable:
    return filter(lambda i: i is not None, iterable)


def tap(callback: Callable[[T1], None]) -> Callable[[T1], T1]:
    def tap_callback(arg: T1) -> T1:
        callback(arg)
        return arg

    return tap_callback


def map_columns(
    callback: Callable[[str, Series], Series],
    data: DataFrame,
) -> DataFrame:
    data_new = data.copy()
    for column_name in data.columns:
        data_new[column_name] = callback(
            column_name,
            data[column_name],
        )
    return data_new


def map_columns_(
    data: DataFrame,
    callback: Callable[[str, Series], Series],
) -> DataFrame:
    return map_columns(callback, data)


def reject_none_values(dictionary: Dict) -> Dict:
    return valfilter(lambda o: o is not None and not isna(o), dictionary)


def reject_none(sequence: Sequence) -> Iterable:
    return iter(filter(lambda o: o is not None and not isna(o), sequence))


ReturnValueT = TypeVar('ReturnValueT')


def return_value(*args, **kwargs) -> ReturnValueT:
    return kwargs['val']


def always(val: Any) -> Callable:
    return partial(return_value, val=val)


starmap = curry(starmap_itertools)


def accept_extra_parameters(function: Callable):
    function_signature = inspect.signature(function)
    parameters = list(function_signature.parameters)

    @wraps(function)
    def accept_extra_parameters_(*args, **kwargs):
        parameters_call = copy(parameters)
        kwargs_selected = {}
        for arg, value in kwargs.items():
            if arg in parameters_call:
                kwargs_selected[arg] = value
                parameters_call.remove(arg)

        return function(
            *args[:min(len(parameters_call), len(args))],
            **kwargs_selected,
        )

    return accept_extra_parameters_


def valmap_(first, second):
    return valmap(second, first)


def starmap_(first, second):
    return starmap(second, first)


def lagged(iterable: Iterable) -> Iterable:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def subtract(a: Iterable, b: Iterable) -> Iterable:
    b_ = list(b)
    return (x for x in a if x not in b_)
