import sys
from typing import List, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest

from hcve_lib.functional import itemmap_recursive, map_recursive
from hcve_lib.utils import (
    get_class_ratios,
    decamelize_arguments,
    camelize_return,
    map_column_names,
    cumulative_count,
    inverse_cumulative_count,
    key_value_swap,
    index_data,
    list_to_dict_by_keys,
    subtract_lists,
    map_groups_iloc,
    remove_prefix,
    remove_column_prefix,
    transpose_mapping,
    map_groups_loc,
    loc,
    split_data,
    get_fraction_missing,
    get_keys,
    sort_columns_by_order,
    is_noneish,
    sort_index_by_order,
    # SurvivalResample,
    transpose_list,
    binarize,
    get_fractions,
    run_parallel,
    is_numeric,
    get_models_from_repeats,
    get_jobs,
    deep_merge_dicts,
    convert_to_snake_case,
    convert_to_camel_case,
    convert_to_camel_case_keys,
    update_from_diff,
    auto_convert_columns,
    average_kendall_tau,
    kendall_tau,
    get_predictions_from_results,
    is_iterable,
    generate_steps,
    deep_merge,
    get_categorical_columns,
    retry_async,
    find_key,
    merge_two_level_dict,
    find_unpicklable_attr,
    split_dict_by_keys,
    count_lines,
    compute_classification_scores_statistics,
    average_classification_scores,
    flatten_dict,
    camelize_and_capitalize,
    diff_dict_values,
)
from numpy.testing import assert_array_equal
from pandas import Series, DataFrame, Index
from pandas.testing import assert_frame_equal, assert_series_equal


def test_get_class_ratio():
    assert get_class_ratios(Series([1, 1, 1, 1, 0, 0])) == {0: 2.0, 1: 1.0}
    assert get_class_ratios(Series([1, 1, 1, 2, 0, 0])) == {0: 1.5, 1: 1.0, 2: 3.0}


def test_decamelize_arguments():
    @decamelize_arguments
    def test_function(arg1: Dict, arg2: List):
        return arg1, arg2

    assert test_function(
        {"oneVariable": 1},
        [{"secondVariable": 2}],
    ) == (
        {"one_variable": 1},
        [{"second_variable": 2}],
    )


def test_camelize_return():
    @camelize_return
    def test_function(arg1: Dict, arg2: List):
        return arg1, arg2

    assert test_function(
        {"one_variable": 1},
        [{"second_variable": 2}],
    ) == (
        {"oneVariable": 1},
        [{"secondVariable": 2}],
    )


def test_map_columns():
    assert_frame_equal(
        map_column_names(
            DataFrame({"a": [1], "b": [2]}),
            lambda k: k + "x",
        ),
        DataFrame({"ax": [1], "bx": [2]}),
    )


def test_cumulative_count():
    assert list(
        cumulative_count(
            Series(
                [
                    0,
                    3,
                    5,
                    8,
                ]
            )
        )
    ) == [
        (0, 0.25),
        (3, 0.5),
        (5, 0.75),
        (8, 1.0),
    ]

    assert list(
        cumulative_count(
            Series(
                [
                    np.nan,
                    3,
                    5,
                    8,
                ]
            )
        )
    ) == [
        (3, 0.25),
        (5, 0.5),
        (8, 0.75),
    ]


def test_inverse_cumulative_count():
    assert list(
        inverse_cumulative_count(
            Series(
                [
                    0,
                    3,
                    5,
                    8,
                ]
            )
        )
    ) == [
        (0, 1.0),
        (3, 0.75),
        (5, 0.5),
        (8, 0.25),
    ]


def test_key_value_swap():
    assert key_value_swap({"a": 1, "b": 2}) == {1: "a", 2: "b"}


def test_index_data():
    assert_frame_equal(
        index_data(
            [1, 2],
            DataFrame(
                {"a": [5, 6, 7]},
                index=[1, 2, 3],
            ),
        ),
        DataFrame(
            {"a": [6, 7]},
            index=[2, 3],
        ),
    )

    assert_series_equal(
        index_data(
            [1, 2],
            Series(
                [5, 6, 7],
                index=[1, 2, 3],
            ),
        ),
        Series(
            [6, 7],
            index=[2, 3],
        ),
    )

    assert_array_equal(
        index_data(
            [1, 2],
            np.array(
                [(1.0, 2), (2.0, 3), (3.0, 4)],
                dtype=[("x", "<f8"), ("y", "<i8")],
            ),
        ),
        np.array(
            [(2.0, 3), (3.0, 4)],
            dtype=[("x", "<f8"), ("y", "<i8")],
        ),
    )


def test_list_to_dict():
    assert list_to_dict_by_keys([1, 2], ["a", "b"]) == {"a": 1, "b": 2}


def test_subtract_lists():
    assert subtract_lists([1, 2, 3], [2]) == [1, 3]


def test_map_groups_iloc():
    data = DataFrame(
        {"a": [1, 1, 1, 2, 2, 3]},
        index=[10, 1, 2, 3, 4, 5],
    )
    assert list(map_groups_iloc(data.groupby("a"), data)) == [
        (1, [0, 1, 2]),
        (2, [3, 4]),
        (3, [5]),
    ]


def test_remove_prefix():
    assert remove_prefix("prefix_", "prefix_x_prefix") == "x_prefix"
    assert remove_prefix("prefix_", "refix_x_prefix") == "refix_x_prefix"


def test_remove_column_prefix():
    assert_frame_equal(
        remove_column_prefix(
            DataFrame({"a": [1], "categorical__b": ["a"], "continuous__c": [3]})
        ),
        DataFrame({"a": [1], "b": ["a"], "c": [3]}),
    )


def test_percent_missing():
    assert get_fraction_missing(Series([np.nan, np.nan])) == 1
    assert get_fraction_missing(Series([np.nan, 5])) == 0.5
    assert get_fraction_missing(Series([1, 5])) == 0


def test_transpose_dict():
    assert transpose_mapping(
        {
            0: {"a": "x", "b": 1},
            1: {"a": "y", "b": 2},
            2: {"a": "z", "b": 3},
        }
    ) == {
        "a": {0: "x", 1: "y", 2: "z"},
        "b": {0: 1, 1: 2, 2: 3},
    }


def test_transpose_list_of_dicts():
    # Test case 1: Typical case
    data = [
        {"name": "Alice", "age": 25, "city": "New York"},
        {"name": "Bob", "age": 30, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
    ]
    expected = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "Los Angeles", "Chicago"],
    }
    assert transpose_list_of_dicts(data) == expected


def test_transpose_list():
    assert transpose_list([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


def test_map_groups_loc():
    results = list(
        map_groups_loc(
            DataFrame(
                {"a": [0, 0, 1, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ).groupby("a")
        )
    )

    assert results[0][0] == 0
    assert_array_equal(results[0][1], Index([10, 20], dtype="int64"))

    assert results[1][0] == 1
    assert_array_equal(results[1][1], Index([30, 40, 50], dtype="int64"))


def test_loc():
    assert_frame_equal(
        loc(
            [10, 30],
            DataFrame(
                {"a": [0, 0, 1, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        ),
        DataFrame(
            {"a": [0, 1]},
            index=[10, 30],
        ),
    )

    with pytest.raises(Exception):
        loc(
            [10, 30, -100],
            DataFrame(
                {"a": [0, 0, 1, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        )

    loc(
        [10, 30, -100],
        DataFrame(
            {"a": [0, 0, 1, 1, 1]},
            index=[10, 20, 30, 40, 50],
        ),
        ignore_not_present=True,
    )


def test_split_data():
    X_train, y_train, X_test, y_test = split_data(
        X=DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [0, 1, 2, 3, 4],
            },
            index=[10, 20, 30, 40, 50],
        ),
        y={
            "data": DataFrame(
                {"tte": [0, 10, 200, 30, 40], "label": [0, 0, 0, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        },
        prediction={
            "split": ([10, 50], [20, 30]),
            # Skipping index 20 in y_score
            "y_score": Series([1, 3, 4, 5], index=[10, 30, 40, 50]),
            "X_columns": ["a"],
        },
    )

    assert_frame_equal(
        X_train,
        DataFrame(
            {
                "a": [0, 4],
            },
            index=[10, 50],
        ),
    )
    assert_frame_equal(
        y_train["data"],
        DataFrame(
            {"tte": [0, 40], "label": [0, 1]},
            index=[10, 50],
        ),
    )

    assert_frame_equal(
        X_test,
        DataFrame(
            {
                "a": [2],
            },
            index=[30],
        ),
    )
    assert_frame_equal(
        y_test["data"],
        DataFrame(
            {"tte": [200], "label": [0]},
            index=[30],
        ),
    )


def test_split_data_remove_extended():
    X_train, y_train, X_test, y_test = split_data(
        DataFrame(
            {
                "a": [0, 1, 2, 3, 4],
                "b": [0, 1, 2, 3, 4],
            },
            index=[10, 20, 30, 40, 50],
        ),
        {
            "data": DataFrame(
                {"tte": [0, 10, 200, 30, 40], "label": [0, 0, 0, 1, 1]},
                index=[10, 20, 30, 40, 50],
            ),
        },
        {"split": ([10, 50], [20, 30, 40]), "X_columns": ["a"]},
        remove_extended=True,
    )
    assert_frame_equal(
        X_train,
        DataFrame(
            {
                "a": [0, 4],
            },
            index=[10, 50],
        ),
    )
    assert_frame_equal(
        y_train["data"],
        DataFrame(
            {"tte": [0, 40], "label": [0, 1]},
            index=[10, 50],
        ),
    )

    assert_frame_equal(
        X_test,
        DataFrame(
            {
                "a": [1, 3],
            },
            index=[20, 40],
        ),
    )
    assert_frame_equal(
        y_test["data"],
        DataFrame(
            {"tte": [10, 30], "label": [0, 1]},
            index=[20, 40],
        ),
    )


def test_fraction_missing():
    assert get_fraction_missing(Series([0, 1, 2, np.nan])) == 0.25


def test_map_recursive():
    assert map_recursive(
        {
            "a": {
                "b": [2, 3],
                "c": 4,
            },
        },
        lambda num, _: num + 1 if isinstance(num, int) else num,
    ) == {
        "a": {
            "b": [3, 4],
            "c": 5,
        }
    }


def test_get_keys():
    assert get_keys(["x"], {"x": 1, "y": 2}) == {"x": 1}


def test_itemmap_recursive():
    with pytest.raises(TypeError):
        itemmap_recursive("x", lambda x: x + "b")

    assert itemmap_recursive(
        {"x": 1, "y": 2},
        lambda k, v, l: (k + "b", v + 1),
    ) == {"xb": 2, "yb": 3}

    assert itemmap_recursive(
        (1, 2, 3),
        lambda k, v, l: (None, v + 1),  # Key ignored
    ) == (2, 3, 4)

    assert itemmap_recursive(
        [1, 2, 3],
        lambda k, v, l: (None, v + 1),  # Key ignored
    ) == [2, 3, 4]

    assert itemmap_recursive(
        {
            "x": {"y": 1},
        },
        lambda k, v, l: (k + "b", v + 1 if isinstance(v, int) else v),
    ) == {
        "xb": {"yb": 2},
    }

    assert itemmap_recursive(
        {"x": [1, 2, 3]},
        lambda k, v, l: (str(k) + "b", v + l if isinstance(v, int) else v),
    ) == {
        "xb": [2, 3, 4],
    }


def test_sort_columns_by_order():
    assert_frame_equal(
        sort_columns_by_order(
            DataFrame({"a": [1], "b": [2], "c": [3]}),
            ["x", "a", "c"],
        ),
        DataFrame(
            {
                "x": [np.nan],
                "a": [1],
                "c": [3],
            }
        ),
    )


def test_sort_index_by_order():
    assert_frame_equal(
        sort_index_by_order(
            DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"]),
            ["w", "z", "y"],
        ),
        DataFrame({"a": [np.nan, 3, 2]}, index=["w", "z", "y"]),
    )


def test_is_noneish():
    assert is_noneish(None) is True
    assert is_noneish(np.nan) is True
    assert is_noneish(False) is False
    assert is_noneish(0) is False
    assert is_noneish("0") is False
    assert is_noneish(5) is False


def subprocess(something):
    print("3")


def test_capture_output():
    # test_std = StringIO()
    # sys.stdout = test_std
    # sys.stderr = test_std

    from hcve_lib.log_output import capture_output

    with capture_output() as get_output:
        print("1")
        print("2", file=sys.stderr)
        run_parallel(subprocess, {0: ["a"]}, n_jobs=2)

    assert get_output() == "1\n2\n3\n"


#
# def test_SurvivalResample():
#     X = DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
#     y = {
#         "data": DataFrame(
#             {
#                 "tte": [10, 10, 20],
#                 "label": [1, 1, 0],
#             },
#             index=[10, 20, 30],
#         )
#     }
#     Xr, yr = SurvivalResample(RandomUnderSampler()).fit_resample(X, y)
#     assert (
#         yr["data"]["label"].value_counts()[1] == yr["data"]["label"].value_counts()[0]
#     )


def test_binarize():
    assert_series_equal(
        binarize(Series([0.1, 0.2, 0.3]), threshold=0.2),
        Series([0, 1, 1]),
    )


def test_get_fractions():
    assert_series_equal(
        get_fractions(Series(["a", "a", "b", "c"])),
        Series([0.5, 0.25, 0.25], index=["a", "b", "c"]),
    )


def test_is_numeric():
    assert is_numeric("5") is True
    assert is_numeric("-5") is True
    assert is_numeric("5.5") is True
    assert is_numeric(100) is True
    assert is_numeric("a5") is False


def test_get_categorical_columns():
    categories = get_categorical_columns(
        DataFrame({"x": [1], "y": ["cat"]}, dtypes={"x": int, "y": "category"})
    )


def test_get_models_from_repeats():
    model1 = Mock()
    model1.feature_importances_ = [0.1, 0.5]

    forests = get_models_from_repeats([{"split1": {"model": model1}}])
    print(forests)


@patch("multiprocessing.cpu_count", Mock(return_value=10))
def test_get_jobs():
    granted, residual = get_jobs(4, 6)
    assert granted == 4
    assert residual == 4

    granted, residual = get_jobs(7, 8)
    assert granted == 7
    assert residual == 3

    granted, residual = get_jobs(-1, 8)
    assert granted == 8
    assert residual == 2

    granted, residual = get_jobs(12, 8)
    assert granted == 8
    assert residual == 2

    granted, residual = get_jobs(-1)
    assert granted == 10
    assert residual == 1

    granted, residual = get_jobs(1, 8)
    assert granted == 1
    assert residual == 1


def test_deep_merge_dicts():
    assert {"x": {"y": 5, "a": 1}, "z": 8, "z2": 9} == deep_merge_dicts(
        {"x": {"y": 5}, "z": 7}, {"x": {"a": 1}, "z": 8, "z2": 9}
    )


class A:
    def __init__(self):
        self.x = 10
        self.y = {"a": 20, "b": 30}


class B:
    def __init__(self):
        self.y = {"b": 40, "c": 50}
        self.z = 60


def test_deep_merge_with_dicts():
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    merged = deep_merge(dict1, dict2)
    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_deep_merge_with_dfs():
    dict1 = {"a": 1, "b": {"c": DataFrame({"x": [1]})}}
    dict2 = {"a": 1, "b": {"c": DataFrame({"x": [2]})}}
    merged = deep_merge(dict1, dict2)
    assert merged["b"]["c"].equals(DataFrame({"x": [2]}))


def test_deep_merge_instance_and_dict():
    a = A()
    dict_to_merge = {"y": {"b": 40, "c": 50}, "z": 70}
    merged = deep_merge(a, dict_to_merge)
    assert merged.x == 10
    assert merged.y == {"a": 20, "b": 40, "c": 50}
    assert merged.z == 70


def test_deep_merge_two_instances():
    a = A()
    b = B()
    merged = deep_merge(a, b)
    assert merged.x == 10
    assert merged.y == {"a": 20, "b": 40, "c": 50}
    assert merged.z == 60


def test_deep_merge_with_non_overlapping_keys():
    dict1 = {"a": 1}
    dict2 = {"b": 2}
    merged = deep_merge(dict1, dict2)
    assert merged == {"a": 1, "b": 2}


def test_convert_to_snake_case_keys():
    assert {
        "my_case": 5,
        "some_case_no": {"OhNo": 6, "oh_no": 7},
    } == test_convert_to_snake_case_keys(
        {"myCase": 5, "someCaseNo": {"OhNo": 6, "ohNo": 7}}
    )


def test_convert_to_snake_case():
    assert convert_to_snake_case("myCase") == "my_case"


def test_convert_to_camel_case_keys():
    assert {
        "myCase": 5,
        "someCaseNo": {"OhNo": 6, "ohNo": 7},
    } == convert_to_camel_case_keys(
        {"my_case": 5, "some_case_no": {"OhNo": 6, "oh_no": 7}}
    )


def test_convert_to_camel_case():
    assert convert_to_camel_case("my_case") == "myCase"


def test_update_from_diff():
    class A:
        pass

    a = A()
    a.x = 1
    a.y = {"a": 1, "b": 2}
    a.z = [1, 2, {"c": 3, "d": 4}]

    diff = {"x": 2, "y": {"a": 3}, "z": [2, 3, {"c": 4}]}
    update_from_diff(a, diff)

    assert a.x == 2
    assert a.y == {"a": 3, "b": 2}
    assert a.z == [2, 3, {"c": 4, "d": 4}]


def test_auto_convert_columns():
    dtypes = auto_convert_columns(
        DataFrame({"x": [1, 2, 3], "y": [1, 1, 3], "z": [1, 2, "x"]}), limit=2
    ).dtypes
    assert_series_equal(
        dtypes,
        Series(
            {
                "x": "float64",
                "y": "category",
                "z": "float64",
            }
        ),
    )


def test_kendall_tau():
    test_cases = [
        ([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], 0.6),
        ([3, 1, 2, 5, 4], [1, 2, 3, 4, 5], 0.4),
    ]

    for rank1, rank2, expected_tau in test_cases:
        tau = kendall_tau(rank1, rank2)
        assert tau == pytest.approx(expected_tau)


def test_average_kendall_tau():
    rankings = [
        [1, 2, 3, 4, 5],
        [2, 1, 4, 3, 5],
    ]
    avg_tau = average_kendall_tau(rankings)
    assert avg_tau == pytest.approx(0.6)


def test_get_predictions_from_results():
    predictions = [{}, {}, {}, {}]
    results = [
        {"split1": predictions[0], "split2": predictions[1]},
        {"split1": predictions[2], "split2": predictions[3]},
    ]

    predictions_results = list(get_predictions_from_results(results))

    assert predictions_results[0] is predictions[0]
    assert predictions_results[1] is predictions[1]
    assert predictions_results[2] is predictions[2]
    assert predictions_results[3] is predictions[3]
    assert predictions_results[2] is not predictions[3]


def test_is_iterable():
    assert is_iterable([1, 2, 3]) == True
    assert is_iterable("Hello") == True
    assert is_iterable(123) == False
    assert is_iterable({"key": "value"}) == True
    assert is_iterable(()) == True


def test_generate_steps():
    assert list(generate_steps(1, 5, 5)) == [1, 2, 3, 4, 5]

    assert list(generate_steps(1, 10, 4)) == [1, 4, 7, 10]

    assert list(generate_steps(1, 10, 5)) == [1, 3, 6, 8, 10]


@pytest.mark.asyncio
async def test_retry_async():
    # 1. Test that it retries the specified number of times
    call_count = 0

    @retry_async(max_retries=5, retry_delay=0.1, exception=ValueError)
    async def fail_until_last():
        nonlocal call_count
        call_count += 1
        if call_count < 5:
            raise ValueError()
        return True

    result = await fail_until_last()
    assert call_count == 5
    assert result is True

    # 2. Ensure it raises the exception if retries are exceeded
    call_count = 0

    @retry_async(max_retries=3, retry_delay=0.1, exception=ValueError)
    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise ValueError()

    with pytest.raises(ValueError):
        await always_fail()
    assert call_count == 3

    # 3. Ensure that delay is applied (not precise due to system scheduling, but for demonstration purposes)
    import time

    @retry_async(max_retries=3, retry_delay=1, exception=ValueError)
    async def fail_three_times():
        raise ValueError()

    start_time = time.time()

    with pytest.raises(ValueError):
        await fail_three_times()

    elapsed_time = time.time() - start_time
    assert elapsed_time >= 2

    @retry_async(max_retries=5, retry_delay=0.1, exception=ValueError)
    async def always_succeed():
        return "success"

    assert await always_succeed() == "success"


def test_find_key():
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

    assert find_key(d, "a") is True
    assert find_key(d, "e") is True
    assert find_key(d, "f") is False
    assert find_key(d, "b") is True
    assert find_key(d, "c") is True
    assert find_key({}, "a") is False


def test_merge_two_level_dict():
    input_dict = {
        "group1": {"a": 1, "b": 2},
        "group2": {"c": 3, "d": 4},
    }
    expected_output = {"group1_a": 1, "group1_b": 2, "group2_c": 3, "group2_d": 4}
    assert merge_two_level_dict(input_dict) == expected_output

    input_dict = {}
    expected_output = {}
    assert merge_two_level_dict(input_dict) == expected_output

    input_dict = {"group1": {"a": 1}}
    expected_output = {"group1_a": 1}
    assert merge_two_level_dict(input_dict) == expected_output

    input_dict = {"group1": {"a": 1}, "group2": "not a dict"}

    with pytest.raises(AttributeError):
        merge_two_level_dict(input_dict)


def test_find_unpicklable_attr():
    class TestClass:
        def __init__(self):
            self.a = 1
            self.b = lambda x: x

    d = {"key1": 1, "key2": {"nested_key": TestClass()}}

    assert find_unpicklable_attr(d) == ["key2", "nested_key", "b"]


def test_split_dict_by_keys():
    d = {"a": 1, "b": 2, "c": 3, "d": 4}
    keys = {"a", "c"}

    subset, remaining = split_dict_by_keys(d, keys)

    assert subset == {"a": 1, "c": 3}
    assert remaining == {"b": 2, "d": 4}

    # Testing with keys not present in dict
    keys = {"a", "c", "e"}
    subset, remaining = split_dict_by_keys(d, keys)

    assert subset == {"a": 1, "c": 3}  # 'e' should not be in the subset
    assert remaining == {"b": 2, "d": 4}


def test_count_lines():
    return count_lines("my super\n lines\n") == 3


def test_compute_classification_scores_statistics():
    predictions = {
        0: DataFrame({0: [0.9, 0.7], 1: [0.1, 0.3]}),
        1: DataFrame({0: [0.9, 0.7], 1: [0.1, 0.3]}),
    }
    assert compute_classification_scores_statistics(predictions) == {
        0: {
            "mean": 0.2,
            "median": 0.2,
            "std": 0.1414213562373095,
            "min": 0.1,
            "max": 0.3,
        },
        1: {
            "mean": 0.2,
            "median": 0.2,
            "std": 0.1414213562373095,
            "min": 0.1,
            "max": 0.3,
        },
    }


def test_average_classification_scores():
    predictions = DataFrame({"class_0_proba": [0.9, 0.7], "class_1_proba": [0.1, 0.3]})
    averaged_df = average_classification_scores(predictions)
    assert "average_class_1" in averaged_df.columns
    np.testing.assert_almost_equal(averaged_df["average_class_1"].iloc[0], 0.2)


def test_flatten_dict():
    nested_dict = {"key1": {"subkey1": "value1", "subkey2": "value2"}, "key2": "value3"}

    expected_flat_dict = {
        "key1 subkey1": "value1",
        "key1 subkey2": "value2",
        "key2": "value3",
    }

    flat_dict = flatten_dict(nested_dict)

    assert flat_dict == expected_flat_dict


def test_camelize_and_capitalize():
    assert camelize_and_capitalize("my_function_name") == "MyFunctionName"
    assert camelize_and_capitalize("example_test_case") == "ExampleTestCase"
    assert camelize_and_capitalize("singleword") == "Singleword"
    assert (
        camelize_and_capitalize("multiple_words_separated_by_underscore")
        == "MultipleWordsSeparatedByUnderscore"
    )
    assert camelize_and_capitalize("") == ""


def test_diff_dict_values():
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"a": 1, "b": 3, "d": 5}

    expected = {"b": 3, "c": None, "d": 5}
    result = diff_dict_values(dict1, dict2)
    assert result == expected, f"Expected {expected}, but got {result}"

    dict3 = {"x": 10, "y": 20}
    dict4 = {"x": 10, "y": 30, "z": 40}

    expected = {"y": 30, "z": 40}
    result = diff_dict_values(dict3, dict4)
    assert result == expected, f"Expected {expected}, but got {result}"

    dict5 = {"a": 1, "b": 2}
    dict6 = {"a": 1, "b": 2}

    expected = {}
    result = diff_dict_values(dict5, dict6)
    assert result == expected, f"Expected {expected}, but got {result}"
