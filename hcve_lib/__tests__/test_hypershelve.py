import pytest
import tempfile
import shutil
from pathlib import Path
import fsspec
from typing import Generator

from hcve_lib.hypershelve import HyperShelve

# Test Data
SAMPLE_DATA = {
    "string": "hello world",
    "int": 42,
    "float": 3.14,
    "list": [1, 2, 3],
    "dict": {"a": 1, "b": 2},
    "nested": {"x": {"y": {"z": "deep"}}},
    "none": None,
    "bool": True,
}


class SampleClass:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, SampleClass):
            return False
        return self.value == other.value


@pytest.fixture
def complex_object() -> SampleClass:
    return SampleClass("test_value")


@pytest.fixture
def temp_path() -> Generator[Path, None, None]:
    """Provide a temporary directory path."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def memory_fs() -> fsspec.AbstractFileSystem:
    """Provide an in-memory filesystem."""
    return fsspec.filesystem("memory")


@pytest.fixture
def file_fs() -> fsspec.AbstractFileSystem:
    """Provide an in-memory filesystem."""
    return fsspec.filesystem("file")


@pytest.fixture
def file_storage(temp_path, file_fs) -> HyperShelve:
    """Create local filesystem-based storage."""
    return HyperShelve(temp_path, file_fs)


@pytest.fixture
def mem_storage(memory_fs) -> HyperShelve:
    """Create memory-based storage."""
    return HyperShelve("test_store", filesystem=memory_fs)


def test_store_and_retrieve_string(file_storage):
    """Test storing and retrieving a string."""
    file_storage["test"] = "hello"
    assert file_storage["test"] == "hello"


def test_store_and_retrieve_number(file_storage):
    """Test storing and retrieving numbers."""
    file_storage["int"] = 42
    file_storage["float"] = 3.14
    assert file_storage["int"] == 42
    assert file_storage["float"] == 3.14


def test_store_and_retrieve_dict(file_storage):
    """Test storing and retrieving a dictionary."""
    data = {"a": 1, "b": {"c": 2}}
    file_storage["dict"] = data
    assert file_storage["dict"] == data


def test_store_and_retrieve_list(file_storage):
    """Test storing and retrieving a list."""
    data = [1, 2, {"a": 3}]
    file_storage["list"] = data
    assert file_storage["list"] == data


def test_store_and_retrieve_complex_object(file_storage, complex_object):
    """Test storing and retrieving a custom class instance."""
    file_storage["complex"] = complex_object
    retrieved = file_storage["complex"]
    assert isinstance(retrieved, SampleClass)
    assert retrieved == complex_object


def test_nested_key_storage(file_storage):
    """Test storing with nested keys."""
    file_storage[["a", "b", "c"]] = "nested_value"
    assert file_storage[["a", "b", "c"]] == "nested_value"
    assert ["a", "b", "c"] in file_storage


def test_nested_key_overwrite(file_storage):
    """Test overwriting nested keys."""
    keys = ["a", "b", "c"]
    file_storage[keys] = "original"
    file_storage[keys] = "updated"
    assert file_storage[keys] == "updated"


def test_nested_key_delete(file_storage):
    """Test deleting nested keys."""
    file_storage[["a", "b", "c"]] = "value"
    file_storage[["a", "b", "d"]] = "other"

    del file_storage[["a", "b", "c"]]
    assert ["a", "b", "c"] not in file_storage
    assert ["a", "b", "d"] in file_storage


def test_key_listing(file_storage):
    """Test listing keys."""
    file_storage["a"] = 1
    file_storage[["a", "b"]] = 2
    file_storage[["a", "b", "c"]] = 3
    file_storage[["x", "y"]] = 4

    all_keys = file_storage.list_keys()
    assert len(all_keys) == 4
    assert ["a"] in all_keys
    assert ["a", "b"] in all_keys
    assert ["a", "b", "c"] in all_keys
    assert ["x", "y"] in all_keys


def test_prefix_listing(file_storage):
    """Test listing keys with prefix."""
    file_storage[["a", "b"]] = 1
    file_storage[["a", "c"]] = 2
    file_storage[["x", "y"]] = 3

    a_keys = file_storage.list_keys("a")
    assert len(a_keys) == 2
    assert ["a", "b"] in a_keys
    assert ["a", "c"] in a_keys


def test_clear_storage(file_storage):
    """Test clearing storage."""
    file_storage["a"] = 1
    file_storage[["b", "c"]] = 2

    file_storage.clear()
    assert len(file_storage.list_keys()) == 0
    assert "a" not in file_storage
    assert ["b", "c"] not in file_storage


def test_invalid_dict_key(file_storage):
    """Test using invalid dictionary key."""
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        file_storage[{"dict": "key"}] = "value"


def test_concurrent_access(temp_path):
    """Test concurrent access from multiple instances."""
    store1 = HyperShelve(temp_path)
    store2 = HyperShelve(temp_path)

    store1["key"] = "value"
    assert store2["key"] == "value"

    store2["key"] = "new_value"
    assert store1["key"] == "new_value"


def test_persistence(temp_path):
    """Test data persistence across instance recreation."""
    store1 = HyperShelve(temp_path)
    store1["key"] = "value"
    store1[["nested", "key"]] = "nested_value"

    store2 = HyperShelve(temp_path)
    assert store2["key"] == "value"
    assert store2[["nested", "key"]] == "nested_value"


def test_large_data(file_storage):
    """Test handling large data chunks."""
    large_data = b"x" * (1024 * 1024)  # 1MB
    file_storage["large"] = large_data
    assert file_storage["large"] == large_data


def test_missing_key(file_storage):
    """Test accessing non-existent keys."""
    with pytest.raises(KeyError):
        _ = file_storage["nonexistent"]
    with pytest.raises(KeyError):
        _ = file_storage[["nested", "nonexistent"]]


def test_special_characters(file_storage):
    """Test special characters in keys."""
    special_keys = [
        ["test space", "key"],
        ["test.dot", "key"],
        ["test-dash", "key"],
        ["test_underscore", "key"],
        ["test@symbol", "key"],
    ]

    for key in special_keys:
        file_storage[key] = "value"
        assert file_storage[key] == "value"
        assert key in file_storage


def test_memory_filesystem(mem_storage):
    """Test using memory filesystem."""
    mem_storage["test"] = "memory_value"
    assert mem_storage["test"] == "memory_value"

    mem_storage[["nested", "key"]] = "nested_value"
    assert mem_storage[["nested", "key"]] == "nested_value"


def test_get_key_structure_empty(file_storage):
    """Test get_key_structure with empty storage."""
    structure = file_storage.get_key_structure()
    assert structure == {}


def test_get_key_structure_flat(file_storage):
    """Test get_key_structure with flat keys."""
    file_storage["a"] = 1
    file_storage["b"] = 2
    file_storage["c"] = 3

    structure = file_storage.get_key_structure()
    assert structure == {"a": {}, "b": {}, "c": {}}


def test_get_key_structure_nested(file_storage):
    """Test get_key_structure with nested keys."""
    file_storage[["dataset1", "model1", "params1"]] = "value1"
    file_storage[["dataset1", "model2", "params1"]] = "value2"
    file_storage[["dataset2", "model1", "params2"]] = "value3"

    expected = {
        "dataset1": {"model1": {"params1": {}}, "model2": {"params1": {}}},
        "dataset2": {"model1": {"params2": {}}},
    }

    structure = file_storage.get_key_structure()
    assert structure == expected


def test_get_key_structure_mixed(file_storage):
    """Test get_key_structure with mixed flat and nested keys."""
    file_storage["flat"] = "value"
    file_storage[["a", "b"]] = "nested_value"
    file_storage[["a", "c", "d"]] = "deep_nested_value"

    expected = {"flat": {}, "a": {"b": {}, "c": {"d": {}}}}

    structure = file_storage.get_key_structure()
    assert structure == expected


def test_get_key_structure_after_delete(file_storage):
    """Test get_key_structure after deleting some keys."""
    file_storage[["a", "b", "c"]] = "value1"
    file_storage[["a", "b", "d"]] = "value2"
    file_storage[["a", "e"]] = "value3"

    del file_storage[["a", "b", "c"]]

    expected = {"a": {"b": {"d": {}}, "e": {}}}

    structure = file_storage.get_key_structure()
    assert structure == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
