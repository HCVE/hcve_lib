import json
from functools import partial
from math import ceil
from numbers import Rational
from typing import Any
from typing import Tuple, List

import numpy as np
import yaml
from IPython.core.display import HTML
from IPython.display import display
from matplotlib import pyplot
from numpy import arange
from pandas import DataFrame, Series
from plotly.graph_objs import Figure
from prettytable import PrettyTable, PLAIN_COLUMNS
from ray.dashboard.modules.metrics.dashboards.common import Target
from scipy.stats import gaussian_kde
from toolz import merge, valmap

from hcve_lib.custom_types import Result, TrainTestSplits
from hcve_lib.data import Metadata, format_features_and_values
from hcve_lib.formatting import format_number
from hcve_lib.functional import flatten, pipe, itemmap_recursive, itemmap_recursive_
from hcve_lib.utils import is_noneish

TRANSPARENT = "rgba(0,0,0,0)"


def display_number(i: Rational) -> None:
    display_html(format_number(i))


def display_html(html: str) -> None:
    # noinspection PyTypeChecker
    display(HTML(html))


def h1(text: str) -> None:
    display_html(f"<h1>{text}</h1>")


def h2(text: str) -> None:
    display_html(f"<h2>{text}</h2>")


def h3(text: str) -> None:
    display_html(f"<h3>{text}</h3>")


def h4(text: str) -> None:
    display_html(f"<h4>{text}</h4>")


def b(text: str) -> None:
    display_html(f"<b>{text}</b>")


def p(text: str) -> None:
    display_html(f"<p>{text}</b>")


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

    for ax in axes[n_items - (rows * columns) :]:
        ax.axis("off")

    return axes, fig


def savefig(*args, **kwargs) -> None:
    pyplot.savefig(
        *args,
        **merge(
            dict(bbox_inches="tight", pad_inches=0.1, dpi=300, facecolor="white"),
            kwargs,
        ),
    )


def grid(ax=None) -> None:
    if ax is None:
        ax = pyplot.gca()

    ax.grid(linestyle="dashed", alpha=0.8, linewidth=0.5)


def histogram(series: Series, ax=None, bins="auto") -> None:
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


def get_formatted(something: Any) -> None:
    return yaml.dump(json.loads(json.dumps(something, default=str)))


def print_formatted(something: Any) -> None:
    print(get_formatted(something))


def display_tree(what: Any, levels: int = 5) -> None:
    def get_indent(level: int) -> str:
        return "  " * level

    def itemmap_recursive_print_item(key, value, level):
        print(f"{get_indent(level)}{key}: {type(value).__name__}")
        if isinstance(value, (List, Tuple)) and len(value) > 3:
            print(f"{get_indent(level+1)}0")
            print(f"{get_indent(level+1)}.")
            print(f"{get_indent(level+1)}.")
            print(f"{get_indent(level+1)}.")
            print(f"{get_indent(level+1)}{len(value)}")
            itemmap_recursive_(
                value[0], itemmap_recursive_print_item, level, levels=5 - level
            )
            return key, None
        else:
            return key, value

    itemmap_recursive(
        what,
        itemmap_recursive_print_item,
        levels=levels,
    )


def setup_plotly_style(fig: Figure) -> None:
    fig.update_layout(
        template="simple_white",
        font=dict(family="Calibri", size=25),
        bargroupgap=0.1,
        xaxis=dict(
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=True,
        ),
        legend={
            "title": {"font": {"color": "rgba(0,0,0,0)"}},
        },
    )


def plot_splits_results(X: DataFrame, y: Target, result: Result):
    X_all_index = X.index
    mesh = get_mesh((len(result.keys()), len(X_all_index)))

    run_names = []
    for number, (name, prediction) in enumerate(result.items()):
        row = X_all_index.to_numpy().copy()
        place_row_split_colors(
            mesh[number], row, prediction["split"][0], prediction["split"][1]
        )
        run_names.append(name)

    pyplot.imshow(mesh, aspect="auto", interpolation="none")
    pyplot.xlabel("Individuals")
    pyplot.yticks(ticks=list(range(len(run_names))), labels=run_names)
    pyplot.ylabel("Iteration")
    pyplot.show()


def plot_splits(X: DataFrame, splits: TrainTestSplits):
    X_all_index = X.index
    mesh = get_mesh((len(splits.keys()), len(X_all_index)))

    run_names = []
    for number, (name, split) in enumerate(splits.items()):
        row = X_all_index.to_numpy().copy()
        place_row_split_colors(mesh[number], row, split[0], split[1])
        run_names.append(name)

    pyplot.imshow(mesh, aspect="auto", interpolation="none")
    pyplot.xlabel("Individuals")
    pyplot.yticks(ticks=list(range(len(run_names))), labels=run_names)
    pyplot.ylabel("Iteration")
    pyplot.show()


def get_mesh(shape):
    return np.ones((*shape, 3), dtype="float")


def place_row_split_colors(mesh, all_index, train_index, test_index):
    mesh[np.isin(all_index, train_index)] = (0, 1, 0)
    mesh[np.isin(all_index, test_index)] = (1, 0, 0)


def format_value_with_statistics(value):
    output = f'{value["mean"]:.3f}'
    if not is_noneish(value["ci"][0]):
        output += f' ({value["ci"][0]:.3f}-{value["ci"][1]:.3f})'
    return output


def format_metrics(metrics):
    return valmap(format_value_with_statistics, metrics)


def print_metrics(metrics):
    formatted = format_metrics(metrics)
    for name, value in formatted.items():
        print(name, value)


def display_metrics(metrics):
    formatted = format_metrics(metrics)
    for name, value in formatted.items():
        p(f"{name}: {value}")


def get_metrics_table(metrics):
    metric_table = PrettyTable()
    first_field = list(metrics.keys())[0]

    if isinstance(first_field, (list, tuple)):
        n_field = len(first_field)
    else:
        n_field = 1

    name_fields = [" " * num for num in range(n_field)]

    metric_table.set_style(PLAIN_COLUMNS)
    metric_table.align = "l"
    has_some_ci = False

    metric_table.field_names = [*name_fields, "μ", "σ", "95% CI"]

    for metric_name, metric_value in metrics.items():
        has_ci = not is_noneish(metric_value["ci"][0])

        if has_ci:
            has_some_ci = True

        metric_table.add_row(
            [
                *(metric_name if n_field > 1 else [metric_name]),
                f'{metric_value["mean"]:.3f}',
                *(
                    [
                        f'{metric_value["std"]:.3f}',
                        f'{metric_value["ci"][0]:.3f}-{metric_value["ci"][1]:.3f}',
                    ]
                    if has_ci
                    else ["-", "-"]
                ),
            ]
        )

    return metric_table


def print_metrics_table(metrics):
    print(get_metrics_table(metrics))


def display_metrics_table(metrics):
    display_html(get_metrics_table(metrics).get_html_string())
