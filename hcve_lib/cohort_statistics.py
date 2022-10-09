import os
from itertools import chain
from logging import Logger

from numpy import std
from typing import Iterable, Tuple, Optional

import numpy as np
from pandas import DataFrame, Series, isna
from pandas.core.groupby import DataFrameGroupBy
from toolz import keyfilter

from hcve_lib.data import Metadata, MetadataItem, categorize_features, get_targets
from hcve_lib.formatting import format_number, format_percents
from hcve_lib.functional import pipe, valmap_, starmap_
from hcve_lib.utils import round_significant
from scipy.stats import iqr


def get_events_table(
    metadata: Metadata,
    data_grouped: DataFrame,
) -> str:
    return make_table_from_rows(get_events_rows(metadata, data_grouped))


def get_events_rows(
    metadata: Metadata,
    data_grouped: DataFrame,
) -> Iterable[str]:
    targets = list(get_targets(metadata))
    columns = []

    for column_number, (name, data) in enumerate(data_grouped):
        description_column: Iterable[str] = iter(())
        column_for_group: Iterable[str] = iter(())
        for target in targets:
            description_column = chain(
                description_column,
                [
                    F'<b>{target["meaning"]}</b>',
                    offset('Events per 1000 py'),
                    offset('Incidence'),
                    offset('Events'),
                    offset('Median follow-up (years)'),
                    offset('Missing'),
                ],
            )
            events = get_events(data, target)

            median_fu = get_median_follow_up(data, target) / 365
            fu_iqr = iqr(data[target["identifier_tte"]]) / 365
            if events != 0:
                column_for_group = chain(
                    column_for_group,
                    [
                        '',
                        format_number(get_events_per_person_years(data, target)),
                        format_percents(get_incidence(data, target)),
                        str(events),
                        f'{median_fu:.1f} ({median_fu-(fu_iqr/2):.1f}-{median_fu+(fu_iqr/2):.1f})',
                        format_percents(get_missing_fraction(data, target)),
                    ],
                )
            else:
                column_for_group = chain(
                    column_for_group,
                    [' '] * 4 + [
                        format_percents(get_missing_fraction(data, target)),
                    ],
                )

        if column_number == 0:
            columns.append(pipe(
                description_column,
                make_value_cells,
                list,
            ))
        columns.append(pipe(
            column_for_group,
            make_value_cells,
            list,
        ))
    rows = [''.join(make_events_header(data_grouped))]
    rows = chain(rows, zip(*columns))
    rows = map(''.join, rows)
    rows = list(rows)
    return (f'<tr>{row}</tr>' for row in rows)


def get_events_per_person_years(
    data: DataFrame,
    feature: MetadataItem,
    desired_person_years: int = 1000,
) -> float:
    events = get_events(data, feature)
    total_person_years = data[f'{feature["identifier_tte"]}'].sum() / 365
    events_per_person_year = (events / total_person_years)
    return events_per_person_year * desired_person_years


def get_incidence(
    data: DataFrame,
    feature: MetadataItem,
) -> float:
    return get_events(data, feature) / get_non_missing(data, feature)


def get_events(
    data: DataFrame,
    feature: MetadataItem,
) -> float:
    return data[feature['identifier']].value_counts()[1]


def get_non_missing(
    data: DataFrame,
    feature: MetadataItem,
) -> float:
    return data[feature['identifier']].value_counts().sum()


def get_missing(
    data: DataFrame,
    feature: MetadataItem,
) -> float:
    return data[feature['identifier']].isna().sum()


def get_median_follow_up(data, feature):
    return data[feature['identifier_tte']].median()


def get_missing_fraction(
    data: DataFrame,
    feature: MetadataItem,
) -> float:
    return get_missing(data, feature) / len(data)


def make_events_header(X_grouped: DataFrameGroupBy) -> Iterable[str]:
    yield '<th></th>'
    for group_name, X_group in X_grouped:
        yield f'<th>{group_name} (n={len(X_group)})</th>'


def get_characteristics_table(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
) -> str:
    return make_table_from_rows(get_rows_characteristics(metadata, X, X_grouped))


def get_missing_table(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
) -> str:
    return make_table_from_rows(get_rows_missing(metadata, X, X_grouped))


def make_table_from_rows(rows: Iterable[str]):
    rows = os.linesep.join(rows)
    return f'<table>' + \
           f'{rows}' + \
           f'</table>'


def make_header_characteristics(X_grouped: DataFrameGroupBy) -> Iterable[str]:
    yield '<th></th>'
    for group_name, X_group in X_grouped:
        yield f'<th>{group_name} (n={len(X_group)})</th>'


def make_header_missing(X_grouped: DataFrameGroupBy) -> Iterable[str]:
    yield '<th></th>'
    for group_name, X_group in X_grouped:
        yield f'<th>{group_name} (n={len(X_group)})</th>'


def get_rows_characteristics(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
) -> Iterable[str]:
    names = make_description_cells(get_description_column(metadata, X))
    value_columns = []
    for name, X_cohort in X_grouped:
        value_columns.append(pipe(
            get_value_column(metadata, X_cohort),
            make_value_cells,
            list,
        ))

    value_rows = zip(names, *value_columns)
    rows = [''.join(make_header_characteristics(X_grouped))]
    rows = chain(rows, value_rows)
    rows = map(''.join, rows)
    rows = list(rows)

    return (f'<tr>{row}</tr>' for row in rows)


def get_rows_missing(
    metadata: Metadata,
    X: DataFrame,
    X_grouped: DataFrameGroupBy,
) -> Iterable[str]:
    names = make_description_cells(get_description_column(metadata, X))
    value_columns = []
    for name, X_cohort in X_grouped:

        value_columns.append(pipe(
            get_missing_column(metadata, X_cohort),
            make_value_cells,
            list,
        ))
    value_rows = zip(names, *value_columns)

    rows = [''.join(make_header_missing(X_grouped))]
    rows = chain(rows, value_rows)
    rows = map(''.join, rows)
    rows = list(rows)

    return (f'<tr>{row}</tr>' for row in rows)


def iterate_over_items(
    items: Metadata,
    X: DataFrame,
    level: int = 0,
) -> Iterable[Tuple[MetadataItem, int]]:
    for item in items:
        if (item['identifier'] not in X.columns and 'children' not in item) or item.get('type') == 'outcome':
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
                label += ", n (%)"
            yield level, 'item', label


def make_description_cells(descriptions: Iterable[Tuple[int, str, str]]) -> Iterable[str]:
    for level, item_type, text in descriptions:
        if level == 0:
            text_formatted = f'<b>{text}</b>'
        else:
            text_formatted = text
        yield f'<td>{offset(text_formatted, level)}</td>'


def offset(what: str, level: int = 2) -> str:
    return f'<span style="padding-left: {20 * level}px">{what}</span>'


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
            missing_fraction = (len(X[X[item['identifier']].isna()]) / len(X)) * 100
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
        value_counts = {item_mapping[key]: value_counts_data[key] for key in item_mapping.keys()}
    except IndexError as e:
        if logger:
            logger.error(item['identifier'], e)
        raise e

    yes_no_feature = item_mapping and set(item_mapping.values()) == {'Yes', 'No'}

    if not yes_no_feature:
        value_filtered = value_counts
    else:
        value_filtered = keyfilter(lambda key: key == 'Yes', value_counts)

    len_X_non_missing = len(X[item['identifier']].dropna())
    value_fraction = valmap_(
        value_filtered,
        lambda value_count: (value_count / len_X_non_missing) * 100 if len_X_non_missing != 0 else np.nan,
    )

    output = starmap_(
        zip(value_fraction.values(), value_filtered.values()),
        lambda fraction,
        count: '—' if isna(fraction) else (f'{format_number(count)}' + f' ({fraction:.0f})'),
    )

    return ' / '.join(output)


def get_continuous_statistic(feature_values: Series) -> str:
    mean_value = float(feature_values.mean())
    spread_statistic = f' ± {round_significant(std(feature_values))}'
    if isna(mean_value):
        return '—'
    else:
        mean_value_rounded = round_significant(mean_value)
        return str(f'{mean_value_rounded}') + spread_statistic


def make_value_cells(column: Iterable[str]) -> Iterable[str]:
    for value in column:
        yield f'<td>{value}</td>'
