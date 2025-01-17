from enum import Enum
from functools import partial as p, partial
from typing import (
    List,
    Iterator,
    Optional,
    Iterable,
    Tuple,
    Any,
    Dict,
    Callable,
    Union,
    Mapping,
)
from typing_extensions import TypedDict

import numpy as np
from pandas import DataFrame, Series

from hcve_lib.custom_types import SurvivalPairTarget, Target, TargetObject
from hcve_lib.functional import pipe, map_columns_
from hcve_lib.utils import key_value_swap

Metadata = List["MetadataItem"]  # type: ignore


class MetadataItem(TypedDict, total=False):
    identifier: str
    identifier_tte: str
    name: str
    name_short: str
    children: "Metadata"
    type: "MetadataItemType"
    value_name_map: Dict
    unit: str


class MetadataItemType(Enum):
    SURVIVAL_TARGET = "survival_target"
    BINARY_TARGET = "binary_target"


def flatten_metadata(metadata: Metadata) -> Iterator[MetadataItem]:
    for item in metadata:
        yield item
        if has_children(item):
            yield from flatten_metadata(item["children"])  # type: ignore


def has_children(item: MetadataItem) -> bool:
    if "children" not in item or item["children"] is None:
        return False
    else:
        return True


def find_item(
    identifier: str,
    metadata: Metadata,
) -> Optional[MetadataItem]:
    for item in flatten_metadata(metadata):
        if item.get("identifier") == identifier:
            return item
    return None


def format_features_and_values(
    data: DataFrame,
    metadata: Metadata,
    feature_axis: int = 1,
) -> DataFrame:
    return pipe(
        data,
        p(format_feature_values, metadata=metadata),
        p(format_features, metadata=metadata, axis=feature_axis),
    )


def format_features(
    data: DataFrame,
    metadata: Metadata,
    axis: int = 1,
    formatter: Optional[Callable] = None,
) -> DataFrame:
    if formatter is None:
        formatter = format_identifier
    return data.rename(
        lambda identifier: formatter(identifier, metadata=metadata),
        axis=axis,
    )


def format_feature_value(value: Any, metadata_item: Optional[MetadataItem]) -> Any:
    if not metadata_item:
        return value

    mapping = metadata_item.get("value_name_map")
    if isinstance(mapping, Mapping):
        return mapping.get(value, value)
    else:
        return value


def inverse_format_feature_value(
    value: Any,
    metadata_item: Optional[MetadataItem],
) -> Any:
    if (
        metadata_item is None
        or "value_name_map" not in metadata_item
        or metadata_item["value_name_map"] is None
    ):
        return value

    mapping = key_value_swap(metadata_item["value_name_map"])

    if mapping:
        return mapping.get(value, value)
    else:
        return value


def format_feature_values(data: DataFrame, metadata: Metadata) -> DataFrame:
    return map_columns_(data, partial(format_series, metadata=metadata))


def format_identifier(
    identifier: str,
    metadata: Metadata,
) -> str:
    name = format_identifier_raw(identifier, metadata)
    return name if name is not None else identifier


def format_identifier_long(
    identifier: str,
    metadata: Metadata,
) -> str:
    return f"[{identifier}] {format_identifier(identifier, metadata)}"


def format_identifier_raw(
    identifier: str,
    metadata: Metadata,
):
    item = find_item(identifier, metadata)
    if item:
        return item.get("name")
    else:
        return None


def format_identifier_short(
    identifier: str,
    metadata: Metadata,
) -> Optional[str]:
    item = find_item(identifier, metadata)
    if isinstance(item, Mapping):
        return item.get("name_short", item.get("name", identifier))
    else:
        return None


def get_feature_subset(df: DataFrame, feature_names: List[str]) -> DataFrame:
    return df[feature_names]


def get_identifiers(metadata: Iterable[MetadataItem]) -> Iterator[str]:
    for item in metadata:
        if "identifier" in item:
            yield item["identifier"]

        if "identifier_tte" in item and item["identifier_tte"]:
            yield item["identifier_tte"]


def get_targets(metadata: Metadata) -> Iterator[MetadataItem]:
    return pipe(
        metadata,
        flatten_metadata,
        partial(filter, is_target),
        iter,
    )


def is_target(item: MetadataItem) -> bool:
    return item.get("type") in (
        MetadataItemType.SURVIVAL_TARGET.value,
        MetadataItemType.BINARY_TARGET.value,
    )


def get_variables(metadata: Metadata) -> Iterator[MetadataItem]:
    return pipe(
        metadata,
        flatten_metadata,
        partial(filter, is_variable),
    )


def get_variable_identifier(metadata: Metadata) -> Iterator[str]:
    return pipe(
        metadata,
        get_variables,
        get_identifiers,
    )


def is_variable(item: MetadataItem) -> bool:
    return item.get("children") is None


def get_survival_y(
    data: DataFrame,
    target_feature: str,
    metadata: Metadata,
) -> Target:
    metadata_item: Optional[MetadataItem] = find_item(target_feature, metadata)

    if metadata_item:
        return TargetObject(
            name=target_feature,
            data=data[
                [
                    metadata_item["identifier"],
                    metadata_item["identifier_tte"],
                ]
            ]
            .copy()
            .rename(
                {
                    metadata_item["identifier"]: "label",
                    metadata_item["identifier_tte"]: "tte",
                },
                axis=1,
            ),
        )
    else:
        raise KeyError()


def to_survival_y_records(survival_y: Target) -> np.recarray:
    return survival_y.to_records(
        index=False,
        column_dtypes={"label": np.bool_, "tte": np.int32},
    )


def to_survival_y_pair(survival_y: DataFrame) -> SurvivalPairTarget:
    return SurvivalPairTarget(
        survival_y["tte"].to_numpy(),
        survival_y["label"].to_numpy(),
    )


def binarize_event(
    tte: int,
    survival_y: DataFrame,
    drop_censored: bool = True,
) -> Series:
    y_binary = Series(index=survival_y.index.copy(), dtype="float64")
    y_binary[(survival_y["tte"] > tte)] = 0
    y_binary[(survival_y["tte"] <= tte) & (survival_y["label"] == 1)] = 1
    y_binary.name = survival_y.name + " " + str(tte / 365) + " years"
    if drop_censored:
        return y_binary.dropna().astype("category")
    else:
        return y_binary


def get_X(
    data: DataFrame,
    metadata: Metadata,
) -> DataFrame:
    features = [
        item.get("identifier")
        for item in flatten_metadata(metadata)
        if not is_target(item) and item.get("identifier") in data.columns
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
    if not item or "value_name_map" not in item:
        return series

    mapping_with_defaults = (
        get_default_mapping(series) | item["value_name_map"]  # type: ignore
    )

    return series.map(mapping_with_defaults)


def get_available_identifiers_per_category(
    metadata: Metadata,
    data: DataFrame,
) -> Iterator[Tuple[MetadataItem, List[str]]]:
    for num, item in enumerate(metadata):
        identifiers = pipe(
            item["children"],
            get_identifiers,
            partial(filter, lambda feature_name: feature_name in data.columns),
            list,
        )
        if len(identifiers) > 0:
            yield item, identifiers


def categorize_features(X: DataFrame) -> Tuple[List[str], List[str]]:
    categorical_features = [
        column_name
        for column_name in X.columns
        if X[column_name].dtype.name == "object"
        or X[column_name].dtype.name == "category"
    ]
    continuous_features = [
        column_name
        for column_name in X.columns
        if column_name not in categorical_features
    ]
    return categorical_features, continuous_features


def get_age_range(X: DataFrame, age_range: Union[List, Tuple]) -> DataFrame:
    return X[(X["AGE"] >= age_range[0]) & (X["AGE"] <= age_range[1])]
