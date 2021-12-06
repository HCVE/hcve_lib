from functools import partial

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from missingno import missingno
from pandas import DataFrame
from pandas import Series

from hcve_lib.data import Metadata, get_identifiers, get_available_identifiers_per_category
from hcve_lib.data import format_identifier
from hcve_lib.functional import unzip, pipe
from hcve_lib.utils import inverse_cumulative_count
from hcve_lib.utils import map_column_names


def missing_values(data: DataFrame, metadata: Metadata, **plot_kwargs) -> None:
    # noinspection PyUnresolvedReferences
    colors = matplotlib.cm.get_cmap('tab10').colors

    available_identifiers = list(
        get_available_identifiers_per_category(
            metadata,
            data,
        ))

    width_ratios = [
        len(identifiers) for item, identifiers in available_identifiers
    ]
    n_subsets = len(available_identifiers)

    fig, ax = pyplot.subplots(
        1,
        n_subsets,
        figsize=(20, 5),
        gridspec_kw={'width_ratios': width_ratios},
    )
    # pyplot.subplots_adjust(wspace=0.007, hspace=0)
    pyplot.suptitle(plot_kwargs.get('title'), y=1.6, fontsize=20)

    for num, (category, identifiers) in enumerate(available_identifiers):

        subset = data[identifiers]
        subset_features_formatted = map_column_names(
            subset,
            partial(format_identifier, metadata=metadata),
        )

        ax[num].set_title(
            category.get('identifier'),
            y=1.65,
            fontdict={'fontsize': 11},
        )

        axis = missingno.matrix(
            subset_features_formatted,
            sparkline=False,
            labels=True,
            color=colors[num],
            ax=ax[num],
            fontsize=9,
        )

        axis.get_yaxis().set_visible(num == 0)


def follow_ups_vs_threshold(series: Series, **plot_kwargs) -> None:

    pyplot.title('Follow-ups vs threshold_missing')
    pyplot.plot(*unzip(inverse_cumulative_count(series)), **plot_kwargs)
    pyplot.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda tick, _: f'{round(tick*100)}%'))
    pyplot.xlabel('Follow up threshold_missing [days]')
    pyplot.ylabel('% Follow ups after threshold_missing')
