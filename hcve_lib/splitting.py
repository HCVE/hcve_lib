import itertools
from collections import defaultdict
from functools import partial, reduce
from typing import Callable, Sequence, List, Tuple, Dict, Any, Hashable, cast

from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, GroupBy
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
    LeaveOneOut,
)
from toolz import identity, merge, valmap
from toolz.curried import valfilter, map

from hcve_lib.custom_types import (
    Target,
    TrainTestSplits,
    Prediction,
    ExceptionValue,
    TrainTestSplitter,
    Index,
)
from hcve_lib.data import get_survival_y
from hcve_lib.functional import pipe, mapl, accept_extra_parameters, flatten, valmap_
from hcve_lib.utils import (
    subtract_lists,
    map_groups_iloc,
    list_to_dict_index,
    get_fraction_missing,
    partial,
    loc,
    empty_dict,
    generate_steps,
    transpose_dict,
    merge_two_level_dict,
    flatten_dict,
)


@accept_extra_parameters
def get_lco(X: DataFrame, data: DataFrame, column: str = "STUDY") -> TrainTestSplits:
    return get_lo_splits(X, data, column)


@accept_extra_parameters
def get_lo_splits(
    X: DataFrame,
    data: DataFrame,
    group_by_column: str,
) -> TrainTestSplits:
    data_subset = data.loc[X.index]
    all_indexes = data_subset.index
    groups = data_subset.groupby(group_by_column)
    return pipe(
        {
            key: (
                subtract_lists(list(all_indexes), subset.index.tolist()),
                subset.index.tolist(),
            )
            for key, subset in groups
        },
        valfilter(lambda split: len(split[0]) > 0 and len(split[1]) > 0),
    )


@accept_extra_parameters
def get_1_to_1_splits(
    X: DataFrame,
    data: DataFrame,
    group_by_column: str,
) -> TrainTestSplits:
    data_subset = data.loc[X.index]
    groups = data_subset.groupby(group_by_column)
    permutations = itertools.permutations(
        groups,
        2,
    )
    return {
        (key1, key2): (subset1.index.tolist(), subset2.index.tolist())
        for ((key1, subset1), (key2, subset2)) in permutations
    }


@accept_extra_parameters
def get_group_splits(
    X: DataFrame,
    group_by: Any,
) -> TrainTestSplits:
    all_indexes = X.index
    return pipe(
        {
            key: (
                subtract_lists(list(all_indexes), subset.index.tolist()),
                loc(subset.index, X, ignore_not_present=True).index.tolist(),
            )
            for key, subset in group_by
        },
        valfilter(lambda split: len(split[0]) > 0 and len(split[1]) > 0),
    )


@accept_extra_parameters
def get_splits_per_group(
    X: DataFrame,
    y: Target,
    data: DataFrame,
    random_state: int,
    get_splits: Callable = None,
    group_by_feature: str = "STUDY",
):
    if get_splits is None:
        get_splits = partial(get_k_fold_stratified, random_state=random_state)

    groups = X.groupby(data[group_by_feature])

    return pipe(
        (
            [
                ((name, name_inner), split)
                for name_inner, split in get_splits(
                    X=group_df, y=loc(group_df.index, y)
                ).items()
            ]
            for name, group_df in groups
        ),
        flatten,
        dict,
    )


@accept_extra_parameters
def get_lm(
    X: DataFrame,
    data: DataFrame,
    random_state: int,
    bootstrap: bool = True,
    include_local_in_test: bool = True,
) -> TrainTestSplits:
    # if bootstrap:
    # _X = X.sample(frac=0.8, replace=False, random_state=random_state)
    # else:
    _X = X

    lco_splits: TrainTestSplits = get_lco(_X, data)
    lm_splits = {}
    for key, fold_input in lco_splits.items():
        lm_split = list(reversed(fold_input))
        if include_local_in_test:
            lm_train_series = Series(lm_split[0])
            local_in_test = lm_train_series.sample(frac=0.2, random_state=random_state)

            lm_split[1].extend(list(local_in_test))
            lm_split[0] = lm_train_series.drop(index=local_in_test.index).tolist()

        lm_splits[key] = lm_split

    return lm_splits


@accept_extra_parameters
def get_reproduce_split(data: DataFrame) -> TrainTestSplits:
    return train_test_filter(
        data,
        train_filter=lambda _data: _data["STUDY"].isin(
            [
                "HEALTHABC",
                "PREDICTOR",
                "PROSPER",
            ]
        ),
        test_filter=lambda _data: _data["STUDY"] == "ASCOT",
    )


@accept_extra_parameters
def get_healthabc_ascot_split(data: DataFrame) -> TrainTestSplits:
    return train_test_filter(
        data,
        train_filter=lambda _data: _data["STUDY"].isin(
            [
                "HEALTHABC",
            ]
        ),
        test_filter=lambda _data: _data["STUDY"] == "ASCOT",
    )


@accept_extra_parameters
def get_loo_splits(
    X: DataFrame,
) -> TrainTestSplits:
    return pipe(
        LeaveOneOut().split(X),
        list,
        map(mapl(partial(iloc_to_loc, X))),
        list_to_dict_index,
    )


@accept_extra_parameters
def get_full_train(
    X: DataFrame,
) -> TrainTestSplits:
    train_indices = X.index.tolist()
    test_indices = []

    train_loc_indices = X.loc[train_indices].index.tolist()
    test_loc_indices = X.loc[test_indices].index.tolist()

    return {
        "full_train": (
            train_loc_indices,
            test_loc_indices,
        )
    }


@accept_extra_parameters
def get_k_fold(
    X: DataFrame,
    random_state: int,
    n_splits: int = 5,
    *args,
    **kwargs,
) -> TrainTestSplits:
    splits = pipe(
        KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ).split(X),
        list,
        map(mapl(partial(iloc_to_loc, X))),
        list_to_dict_index,
    )
    return splits


@accept_extra_parameters
def get_per_subset_split(
    X: DataFrame,
    y: Target,
    group_by: GroupBy,
    get_splits: TrainTestSplitter,
    random_state: int,
) -> TrainTestSplits:
    splits_subsets: Dict[Any, Dict[Hashable, Tuple[list, list]]] = {}

    for name, train_idx in group_by.groups.items():
        X_subset = loc(train_idx, X, ignore_not_present=True)
        splits_subsets[name] = get_splits(X=X_subset, y=y, random_state=random_state)

    output_splits = defaultdict(lambda: ([], []))

    for key, value in transpose_dict(splits_subsets).items():
        for train, test in value.values():
            output_splits[key][0].extend(train)
            output_splits[key][1].extend(test)

    return dict(output_splits)


@accept_extra_parameters
def get_k_fold_stratified(
    X: DataFrame,
    y: Target,
    random_state: int,
    n_splits: int = 5,
) -> TrainTestSplits:
    try:
        y_ = y["label"]
    except KeyError:
        y_ = y

    if str(y_.dtype).startswith("float"):
        return get_k_fold(X, random_state, n_splits)

    return pipe(
        StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        ).split(X, y_),
        list,
        map(mapl(partial(iloc_to_loc, X))),
        list_to_dict_index,
    )


@accept_extra_parameters
def get_train_test(
    X: DataFrame,
    y: Target,
    random_state: int,
    test_size=0.1,
    train_size=None,
    shuffle=True,
    stratify=None,
    *args,
    **kwargs,
) -> TrainTestSplits:
    data_train, data_test = train_test_split(
        X,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify.loc[X.index] if stratify is not None else None,
        *args,
        **kwargs,
    )
    data_train_index = data_train.index.tolist()
    data_test_index = data_test.index.tolist()
    return {"train_test": (data_train_index, data_test_index)}


from typing import Dict, Tuple, Hashable
import numpy as np
from pandas import DataFrame, Index
from typing import Union, Optional
from functools import wraps

Target = Union[DataFrame, np.ndarray, None]


def accept_extra_parameters(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function's valid parameters
        return func(*args, **kwargs)

    return wrapper


@accept_extra_parameters
def get_bootstrap(
    X: DataFrame,
    y: Target,
    random_state: int,
    n_samples=None,
    *args,
    **kwargs,
) -> TrainTestSplits:
    """
    Generate a bootstrap sample from the input data.

    Parameters:
    -----------
    X : DataFrame
        Input features
    y : Target
        Target variable (not used but kept for API consistency)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    Dict[Hashable, Tuple[Index, Index]]
        Dictionary containing the train (bootstrap) and test (out-of-bag) indices
    """
    if n_samples is None:
        n_samples = len(X)
        
    rng = np.random.RandomState(random_state)

    # Generate bootstrap sample indices (sampling with replacement)
    bootstrap_indices = rng.randint(0, n_samples, size=n_samples)

    # Get unique indices for the bootstrap sample (train set)
    train_indices = X.index[bootstrap_indices].tolist()

    # Get out-of-bag indices (test set)
    # Convert to sets for efficient operation
    all_indices_set = set(X.index)
    bootstrap_indices_set = set(train_indices)
    test_indices = list(all_indices_set - bootstrap_indices_set)

    return {"bootstrap": (train_indices, test_indices)}


@accept_extra_parameters
def get_learning_curve_splits(
    X: DataFrame,
    y: Target,
    random_state: int,
    test_size=0.1,
    shuffle=True,
    n_step: int = 100,
    min_samples: int = 500,
    *args,
    **kwargs,
) -> TrainTestSplits:
    test_size_n = int(test_size * len(X))
    data_train, data_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        *args,
        **kwargs,
    )
    data_train_index = data_train.index.tolist()
    data_test_index = data_test.index.tolist()

    steps = list(range(min_samples, len(data_train_index), n_step))
    return {step: (data_train_index[:step], data_test_index) for step in steps}


@accept_extra_parameters
def train_test_filter(
    data: DataFrame,
    train_filter: Callable,
    test_filter: Callable = None,
) -> TrainTestSplits:
    train_mask = train_filter(data)
    train_data = data[train_mask]

    if not test_filter:
        test_data = data[~train_mask]
    else:
        test_data = data[test_filter(data)]

    return {
        "train_test_filter": (
            train_data.index.to_list(),
            test_data.index.to_list(),
        )
    }


def ilocs_from_index(subset_index: Index, full_index: Index) -> List[int]:
    return [full_index.get_loc(index) for index in subset_index]


def iloc_to_loc(data: DataFrame, ilocs: Sequence[int]):
    return data.index[ilocs].tolist()


def filter_missing_features(
    x_train: Series,
    x_test: Series,
    threshold: float = 1,
) -> bool:
    return (
        get_fraction_missing(x_train) >= threshold
        or get_fraction_missing(x_test) >= threshold
    )


def get_splitter(splitter_name: str) -> Callable:
    if splitter_name == "lco":
        return get_lco
    elif splitter_name == "reproduce":
        return get_reproduce_split
    elif splitter_name == "healtahc_ascot":
        return get_healthabc_ascot_split
    elif splitter_name == "lm":
        return partial(get_1_to_1_splits, group_by_column="STUDY")
    elif splitter_name == "cohort_10_fold":
        return get_splits_per_group
    elif splitter_name == "10_fold":
        return partial(get_k_fold_stratified, n_splits=10)
    elif splitter_name == "5_fold":
        return partial(get_k_fold_stratified, n_splits=5)
    else:
        raise Exception("Splitting not know")


def train_test_fold(data, fold: Prediction, metadata) -> Tuple[DataFrame, Target]:
    return data[fold["X_columns"]], get_survival_y(data, fold["y_column"], metadata)


def get_group_indexes(
    data: DataFrame,
    feature_name: str,
) -> Dict[str, Index]:
    groups = data.groupby(feature_name)
    return {name: group.index for name, group in groups}


def resample_prediction_test(
    index: Index,
    prediction: Prediction,
) -> Prediction:
    y_pred = loc(index, prediction["y_pred"].copy(), ignore_not_present=True)
    return merge(
        prediction,
        dict(
            split=(prediction["split"][0], list(y_pred.index)),
            y_pred=y_pred,
        ),
    )
