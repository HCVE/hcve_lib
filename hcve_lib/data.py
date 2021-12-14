from enum import Enum
from functools import partial as p, partial
from typing import TypedDict, List, Iterator, Optional, Iterable, Tuple, Any, Dict, Callable

import numpy as np
from pandas import DataFrame, Series

from hcve_lib.custom_types import SurvivalPairTarget, Target, TargetData
from hcve_lib.functional import pipe, map_columns_, statements
from hcve_lib.utils import key_value_swap

Metadata = List['MetadataItem']  # type: ignore


class MetadataItem(TypedDict):
    identifier: str
    identifier_tte: Optional[str]
    meaning: str
    children: Optional['Metadata']  # type: ignore
    type: Optional['MetadataItemType']
    mapping: Optional[Dict]
    unit: Optional[str]


class MetadataItemType(Enum):
    SURVIVAL_TARGET = 'survival_target'
    BINARY_TARGET = 'binary_target'


def flatten_metadata(metadata: List[MetadataItem]) -> Iterator[MetadataItem]:
    for item in metadata:
        yield item
        if has_children(item):
            yield from flatten_metadata(item['children'])  # type: ignore


def has_children(item: MetadataItem) -> bool:
    if 'children' not in item or item['children'] is None:
        return False
    else:
        return True


def find_item(
    identifier: str,
    metadata: Metadata,
) -> Optional[MetadataItem]:

    for item in flatten_metadata(metadata):
        if item.get('identifier') == identifier:
            return item
    return None


def format_features_and_values(
    data: DataFrame,
    metadata: Metadata,
    feature_axis: int = 1,
) -> DataFrame:
    return pipe(
        data,
        p(format_values, metadata=metadata),
        p(format_features, metadata=metadata, axis=feature_axis),
    )


def format_features(
    data: DataFrame,
    metadata: Metadata,
    axis: int = 1,
    formatter: Callable[[str, Metadata], str] = None,
) -> DataFrame:
    if formatter is None:
        formatter = format_identifier
    return data.rename(
        lambda identifier: formatter(identifier, metadata=metadata),
        axis=axis,
    )


def format_value(value: Any, metadata_item: Optional[MetadataItem]) -> Any:
    if not metadata_item:
        return value

    mapping = metadata_item.get('mapping')
    if mapping:
        return mapping.get(value, value)
    else:
        return value


def inverse_format_value(
    value: Any,
    metadata_item: Optional[MetadataItem],
) -> Any:

    if metadata_item is None or 'mapping' not in metadata_item or metadata_item[
            'mapping'] is None:
        return value

    mapping = key_value_swap(metadata_item['mapping'])

    if mapping:
        return mapping.get(value, value)
    else:
        return value


def format_values(data: DataFrame, metadata: Metadata) -> DataFrame:
    return map_columns_(data, partial(format_series, metadata=metadata))


def format_identifier(
    identifier: str,
    metadata: List[MetadataItem],
) -> str:
    meaning = format_identifier_raw(identifier, metadata)
    return meaning if meaning is not None else identifier


def format_identifier_long(
    identifier: str,
    metadata: List[MetadataItem],
) -> str:
    return f'[{identifier}] {format_identifier(identifier, metadata)}'


def format_identifier_raw(
    identifier: str,
    metadata: List[MetadataItem],
):
    item = find_item(identifier, metadata)
    if item:
        return item.get('meaning')
    else:
        return None


def get_feature_subset(df: DataFrame, feature_names: List[str]) -> DataFrame:
    return df[feature_names]


def get_identifiers(metadata: Iterable[MetadataItem]) -> Iterator[str]:
    for item in metadata:
        if 'identifier' in item:
            yield item['identifier']

        if 'identifier_tte' in item and item['identifier_tte']:
            yield item['identifier_tte']


def sanitize_data_inplace(data: DataFrame) -> DataFrame:
    data.columns = [column.upper() for column in data.columns]
    data['VISIT'] = data['VISIT'].str.upper()


def is_target(item: MetadataItem) -> bool:
    return item.get('type') in (
        MetadataItemType.SURVIVAL_TARGET.value,
        MetadataItemType.BINARY_TARGET.value,
    )


def get_variable_identifier(metadata: Metadata) -> Iterator[str]:
    return pipe(
        metadata,
        get_variables,
        get_identifiers,
    )


def get_variables(metadata: List[MetadataItem]) -> Iterator[MetadataItem]:
    return pipe(
        metadata,
        flatten_metadata,
        partial(filter, is_variable),
    )


def is_variable(item: MetadataItem) -> bool:
    return item.get('children') is None


def get_survival_y(
    data: DataFrame,
    target_feature: str,
    metadata: Metadata,
) -> Target:

    metadata_item: Optional[MetadataItem] = find_item(target_feature, metadata)

    if metadata_item:
        return Target(
            name=target_feature,
            data=data[[
                metadata_item['identifier'],
                metadata_item['identifier_tte'],
            ]].copy().rename(
                {
                    metadata_item['identifier']: 'label',
                    metadata_item['identifier_tte']: 'tte',
                },
                axis=1,
            ),
        )
    else:
        raise KeyError()


def to_survival_y_records(survival_y: Target) -> np.recarray:
    return survival_y['data'].to_records(
        index=False,
        column_dtypes={
            'label': np.bool_,
            'tte': np.int32
        },
    )


def to_survival_y_pair(survival_y: DataFrame) -> SurvivalPairTarget:
    return SurvivalPairTarget(
        survival_y['tte'].to_numpy(),
        survival_y['label'].to_numpy(),
    )


def binarize_survival(tte: int, survival_y: DataFrame) -> Series:
    y_binary = Series(index=survival_y.index.copy())
    y_binary[(survival_y['tte'] > tte)] = 0
    y_binary[(survival_y['tte'] <= tte) & (survival_y['label'] == 1)] = 1
    return y_binary


def get_X(
    data: DataFrame,
    metadata: Metadata,
) -> DataFrame:
    features = [
        item.get('identifier') for item in flatten_metadata(metadata)
        if not is_target(item) and item.get('identifier') in data.columns
    ]

    return data[features]


def remove_nan_target(X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    y_defined = y.dropna()
    X_defined = X.loc[y_defined.index]
    return X_defined, y_defined


def get_default_mapping(column_series: Series) -> Dict:
    return {column: column for column in column_series.unique()}


def format_series(name: str, series: Series, metadata: Metadata) -> Series:
    item = find_item(
        name,
        metadata,
    )
    if not item or 'mapping' not in item:
        return series

    mapping_with_defaults = (
        get_default_mapping(series) | item['mapping']  # type: ignore
    )

    return series.map(mapping_with_defaults)


def get_available_identifiers_per_category(
    metadata: Metadata,
    data: DataFrame,
) -> Iterator[Tuple[MetadataItem, List[str]]]:
    for num, item in enumerate(metadata):
        identifiers = pipe(
            item['children'],
            get_identifiers,
            partial(filter, lambda feature_name: feature_name in data.columns),
            list,
        )
        if len(identifiers) > 0:
            yield item, identifiers


def categorize_features(X: DataFrame) -> Tuple[List[str], List[str]]:
    categorical_features = [
        column_name for column_name in X.columns
        if X[column_name].dtype.name == 'object'
        or X[column_name].dtype.name == 'category'
    ]
    continuous_features = [
        column_name for column_name in X.columns
        if column_name not in categorical_features
    ]
    return categorical_features, continuous_features
