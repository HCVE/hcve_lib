import itertools
from functools import partial
from typing import Callable, Sequence, List, Tuple, Dict

from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from pandas.core.groupby import DataFrameGroupBy
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from toolz import identity
from toolz.curried import valfilter, map

from hcve_lib.custom_types import Target, Splits, SplitPrediction
from hcve_lib.data import get_survival_y
from hcve_lib.functional import pipe, mapl, t, accept_extra_parameters, flatten
from hcve_lib.utils import subtract_lists, map_groups_iloc, list_to_dict_index, get_fraction_missing, partial2


@accept_extra_parameters
def get_lco_splits(X: DataFrame, data: DataFrame) -> Splits:
    return get_lo_splits(X, data, 'STUDY')


@accept_extra_parameters
def get_lo_splits(
    X: DataFrame,
    data: DataFrame,
    group_by_column: str,
) -> Splits:
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
) -> Splits:
    data_subset = data.loc[X.index]
    groups = data_subset.groupby(group_by_column)
    permutations = itertools.permutations(
        groups,
        2,
    )
    return {(key1, key2): (subset1.index.tolist(), subset2.index.tolist())
            for ((key1, subset1), (key2, subset2)) in permutations}


@accept_extra_parameters
def get_group_splits(
    X: DataFrame,
    data: DataFrameGroupBy,
) -> Splits:
    flatten_data = data.apply(identity).loc[X.index]
    all_indexes = range(0, len(flatten_data))
    groups = map_groups_iloc(data, flatten_data)
    return {
        key: (subtract_lists(list(all_indexes), subset), subset)
        for key, subset in groups
    }


@accept_extra_parameters
def get_splitting_per_group(
    X: DataFrame,
    data: DataFrame,
    get_splits: Callable = None,
    group_by_feature: str = "STUDY",
):
    if get_splits is None:
        get_splits = get_kfold_splits
    groups = X.groupby(data[group_by_feature])
    return pipe(
        ([((name, name_inner), split)
          for name_inner, split in get_splits(group_df).items()]
         for name, group_df in groups),
        flatten,
        dict,
    )


@accept_extra_parameters
def get_lm_splits(
    X: DataFrame,
    data: DataFrameGroupBy,
) -> Splits:
    lco_splits: Splits = get_lco_splits(X, data)
    return {
        key: pipe(
            fold_input,
            reversed,
            list,
        )
        for key, fold_input in lco_splits.items()
    }


@accept_extra_parameters
def get_reproduce_split(data: DataFrame) -> Splits:
    return train_test_filter(
        data,
        train_filter=lambda _data: _data['STUDY'].isin([
            'HEALTHABC',
            'PREDICTOR',
            'PROSPER',
        ]),
        test_filter=lambda _data: _data['STUDY'] == 'ASCOT',
    )


@accept_extra_parameters
def get_healthabc_ascot_split(data: DataFrame) -> Splits:
    return train_test_filter(
        data,
        train_filter=lambda _data: _data['STUDY'].isin([
            'HEALTHABC',
        ]),
        test_filter=lambda _data: _data['STUDY'] == 'ASCOT',
    )


@accept_extra_parameters
def get_kfold_splits(
    X: DataFrame,
    n_splits: int = 5,
    random_state: int = None,
) -> Splits:
    return pipe(
        KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ).split(X),
        list,
        map(mapl(partial(iloc_to_loc, X))),
        list_to_dict_index,
    )


@accept_extra_parameters
def get_kfold_stratified_splits(
    X: DataFrame,
    y: Target,
    n_splits: int = 5,
) -> Splits:
    return pipe(
        StratifiedKFold(n_splits=n_splits,
                        shuffle=True).split(X, y['data']['label']),
        list,
        map(mapl(partial(iloc_to_loc, X))),
        list_to_dict_index,
    )


@accept_extra_parameters
def get_train_test(
    X: DataFrame,
    y: Target,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
) -> Splits:
    data_train, data_test = train_test_split(
        X,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y['data']['label'],
    )
    data_train_index = data_train.index.tolist()
    data_test_index = data_test.index.tolist()

    return {'train_test': (data_train_index, data_test_index)}


@accept_extra_parameters
def train_test_filter(
    data: DataFrame,
    train_filter: Callable,
    test_filter: Callable = None,
) -> Splits:

    train_mask = train_filter(data)
    train_data = data[train_mask]

    if not test_filter:
        test_data = data[~train_mask]
    else:
        test_data = data[test_filter(data)]

    return {
        'train_test_filter': (
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
    return \
        get_fraction_missing(x_train) >= threshold \
        or get_fraction_missing(x_test) >= threshold


def get_splitter(splitter_name: str) -> Callable:
    if splitter_name == "lco":
        return get_lco_splits
    elif splitter_name == 'reproduce':
        return get_reproduce_split
    elif splitter_name == 'healtahc_ascot':
        return get_healthabc_ascot_split
    elif splitter_name == 'lm':
        return partial2(get_1_to_1_splits, group_by_column='STUDY')
    elif splitter_name == 'cohort_10_fold':
        return get_splitting_per_group
    elif splitter_name == "10_fold":
        return partial2(get_kfold_stratified_splits, n_splits=10)
    elif splitter_name == "5_fold":
        return partial2(get_kfold_stratified_splits, n_splits=5)
    else:
        raise Exception('Splitting not know')


def train_test_fold(data, fold: SplitPrediction,
                    metadata) -> Tuple[DataFrame, Target]:
    return data[fold['X_columns']], get_survival_y(data, fold['y_column'],
                                                   metadata)


def get_group_indexes(
    data: DataFrame,
    feature_name: str,
) -> Dict[str, Index]:
    groups = data.groupby(feature_name)
    return {name: group.index for name, group in groups}
