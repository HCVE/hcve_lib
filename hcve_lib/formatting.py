from functools import partial

import textwrap

from numbers import Integral, Rational
from pandas import DataFrame
from typing import Union, Dict, Callable


def format_number(i: Union[Integral, Rational, int]) -> str:
    if type(i) == int:
        return f'{i:,}'
    else:
        return f'{i:,.2f}'


def format_percents(
    fraction: float,
    decimals: int = 1,
    sign: bool = True,
) -> str:
    return str(round(fraction * 100, decimals)) + ('%' if sign else '')
