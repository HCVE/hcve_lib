import os
from itertools import chain
from logging import Logger
from typing import Iterable, Tuple, Optional

import numpy as np
from pandas import DataFrame, Series, isna
from pandas.core.groupby import DataFrameGroupBy
from toolz import keyfilter

from hcve_lib.data import Metadata, MetadataItem, categorize_features
from hcve_lib.formatting import format_number
from hcve_lib.functional import pipe, valmap_, starmap_


def get_table(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
):
    rows = os.linesep.join(get_rows(metadata, X, X_grouped))
    return f'<table>' + \
           f'{rows}' + \
           f'</table>'


def make_header(X_grouped: DataFrameGroupBy) -> Iterable[str]:
    yield '<th></th>'
    for group_name, X_group in X_grouped:
        yield f'<th>{group_name} (n={len(X_group)})</th><th>Missing&nbsp;%</th>'


def get_rows(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
) -> Iterable[str]:
    names = make_description_cells(get_description_column(metadata, X))
    values_per_cohort = []
    for name, X_cohort in X_grouped:
        values_per_cohort.append(
            pipe(
                get_value_column(metadata, X_cohort),
                make_value_cells,
                list,
            ))
        values_per_cohort.append(
            pipe(
                get_missing_column(metadata, X_cohort),
                make_value_cells,
                list,
            ))

    rows = [''.join(make_header(X_grouped))]
    rows = chain(rows, zip(names, *values_per_cohort))
    rows = map(''.join, rows)
    rows = list(rows)

    return (f'<tr>{row}</tr>' for row in rows)


def iterate_over_items(
    items: Metadata,
    X: DataFrame,
    level: int = 0,
) -> Iterable[Tuple[MetadataItem, int]]:
    for item in items:
        if (item['identifier'] not in X.columns
                and 'children' not in item) or item.get('type') == 'outcome':
            continue
        yield item, level
        if 'children' in item:
            yield from iterate_over_items(item['children'], X, level + 1)


def get_description_column(
    items: Metadata,
    X: DataFrame,
) -> Iterable[Tuple[int, str, str]]:
    categorical_features, continuous_features = categorize_features(X)

    for item, level in iterate_over_items(items, X):
        if level == 0:
            yield level, 'category', item.get("meaning", item["identifier"])
        elif level >= 1:
            label = item.get("meaning", item["identifier"])
            if 'unit' in item:
                label += f', {item["unit"]}'
            if (item['identifier'] in categorical_features) \
                    and (mapping := item.get('mapping')):
                if not (set(mapping.values()) == {'Yes', 'No'}):
                    label += ', ' + (' / '.join(map(str, mapping.values())))
                label += ", % (n)"
            yield level, 'item', label


def make_description_cells(
        descriptions: Iterable[Tuple[int, str, str]]) -> Iterable[str]:
    for level, item_type, text in descriptions:
        if level == 0:
            text_formatted = f'<b>{text}</b>'
        else:
            text_formatted = text
        yield f'<td><span style="padding-left: {20 * level}px">{text_formatted}</td>'


def get_value_column(
    metadata: Metadata,
    X: DataFrame,
    logger: Logger = None,
) -> Iterable[str]:
    categorical_features, continuous_features = categorize_features(X)
    for item, level in iterate_over_items(metadata, X):
        if level == 0:
            yield ''
        elif level >= 1:
            if item['identifier'] in continuous_features:
                yield get_continuous_statistic(X[item['identifier']])
            else:
                yield get_categorical_statistic(X, item, logger)


def get_missing_column(
    metadata: Metadata,
    X: DataFrame,
) -> Iterable[str]:
    for item, level in iterate_over_items(metadata, X):
        if level == 0:
            yield ''
        elif level >= 1:
            missing_fraction = (len(X[X[item['identifier']].isna()]) /
                                len(X)) * 100
            if missing_fraction == 0:
                yield ''
            else:
                yield f'{missing_fraction:.1f}'


def get_categorical_statistic(
    X: DataFrame,
    item: MetadataItem,
    logger: Optional[Logger],
) -> str:
    value_counts_data = X[item['identifier']].value_counts()
    item_mapping = item.get('mapping')
    if not item_mapping:
        raise Exception('Missing mapping of categorical feature in metadata')

    try:
        value_counts = {
            item_mapping[key]: value_counts_data[key]
            for key in item_mapping.keys()
        }
    except IndexError as e:
        if logger:
            logger.error(item['identifier'], e)
        raise e

    yes_no_feature = item_mapping and set(
        item_mapping.values()) == {'Yes', 'No'}

    if not yes_no_feature:
        value_filtered = value_counts
    else:
        value_filtered = keyfilter(lambda key: key == 'Yes', value_counts)

    len_X_non_missing = len(X[item['identifier']].dropna())
    value_fraction = valmap_(
        value_filtered,
        lambda value_count: (value_count / len_X_non_missing) * 100
        if len_X_non_missing != 0 else np.nan,
    )

    output = starmap_(
        zip(value_fraction.values(), value_filtered.values()),
        lambda fraction, count: '—' if isna(fraction) else
        (f'{fraction:.0f}' + f' ({format_number(count)})'),
    )

    return ' / '.join(output)


def get_continuous_statistic(feature_values: Series) -> str:
    mean_value = float(feature_values.mean())
    spread_statistic = f' ({round(feature_values.quantile(0.1), 2)}' + \
                       f'-{round(feature_values.quantile(0.9), 2)})'
    if isna(mean_value):
        return '—'
    else:
        return str(f'{mean_value:.1f}') + spread_statistic


def make_value_cells(column: Iterable[str]) -> Iterable[str]:
    for value in column:
        yield f'<td>{value}</td>'
