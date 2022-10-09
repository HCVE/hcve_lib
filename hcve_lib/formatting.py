from typing import Union

from pandas.core.dtypes.inference import is_integer

from hcve_lib.functional import map_deep
from hcve_lib.utils import is_numeric
import yaml


def format_number(i: Union[float, int]) -> str:
    if is_integer(i):
        return f'{i:,}'
    else:
        return f'{i:,.2f}'


def format_percents(
    fraction: float,
    decimals: int = 1,
    sign: bool = True,
) -> str:
    return str(round(fraction * 100, decimals)) + ('%' if sign else '')


def format_value(value):
    if is_numeric(value):
        return float(value)
    else:
        return repr(value)


def format_recursive(structure):
    return map_deep(structure, lambda value, _: format_value(value))


def pp(value):
    formatted = format_recursive(value)
    print(yaml.dump(formatted, allow_unicode=True, default_flow_style=False))
