from functools import partial
from math import ceil
from numbers import Rational
from typing import Any

import yaml
from IPython.core.display import HTML
from IPython.display import display
from matplotlib import pyplot
from numpy import arange
from pandas import DataFrame, Series
from scipy.stats import gaussian_kde
from toolz import merge

from hcve_lib.data import Metadata, format_features_and_values
from hcve_lib.formatting import format_number
from hcve_lib.functional import flatten, pipe


def display_number(i: Rational) -> None:
    display_html(format_number(i))


def display_html(html: str) -> None:
    # noinspection PyTypeChecker
    display(HTML(html))


def h1(text: str) -> None:
    display_html(f'<h1>{text}</h1>')


def h2(text: str) -> None:
    display_html(f'<h2>{text}</h2>')


def h3(text: str) -> None:
    display_html(f'<h3>{text}</h3>')


def h4(text: str) -> None:
    display_html(f'<h4>{text}</h4>')


def b(text: str) -> None:
    display_html(f'<b>{text}</b>')


def p(text: str) -> None:
    display_html(f'<p>{text}</b>')


def make_subplots(n_items: int, columns: int = 3, width=None, **subplot_args):
    rows = ceil(n_items / columns)
    # noinspection PyTypeChecker
    if not width:
        width = 5 * columns
    fig, axes = pyplot.subplots(
        nrows=rows,
        ncols=columns,
        **{
            **dict(figsize=(width, width * (rows / columns))),
            **subplot_args,
        },
    )
    axes = list(flatten(axes))

    for ax in axes[n_items - (rows * columns):]:
        ax.axis('off')

    return axes, fig


def savefig(*args, **kwargs) -> None:
    pyplot.savefig(
        *args,
        **merge(
            dict(bbox_inches='tight',
                 pad_inches=0.1,
                 dpi=300,
                 facecolor='white'),
            kwargs,
        ))


def grid(ax=None) -> None:
    if ax is None:
        ax = pyplot.gca()

    ax.grid(linestyle='dashed', alpha=0.8, linewidth=0.5)


def histogram(series: Series, ax=None, bins='auto') -> None:
    if ax is None:
        ax = pyplot.gca()

    ax.hist(series, density=True, bins=bins)
    density = gaussian_kde(series)
    density_x = list(arange(series.min(), series.max()))
    ax.plot(density_x, density(density_x))
    grid(ax=ax)
    ax.set_xlabel(series.name)


def show_dtale(data: DataFrame, metadata: Metadata) -> None:
    import dtale
    table = pipe(
        data,
        partial(format_features_and_values, metadata=metadata),
        dtale.show,
    )
    return table


def print_formatted(something: Any) -> None:
    print(yaml.dump(something))
