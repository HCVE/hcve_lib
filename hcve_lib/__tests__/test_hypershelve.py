import pytest
from fsspec.implementations.memory import MemoryFileSystem

from hcve_lib.hypershelve import PathDict, DictPathDict, FSPathDict


class MockPathDict(PathDict):
    """Mock implementation for testing abstract base class"""

    def __init__(self):
        super().__init__()
        self._store = {}
        self.get_calls = []
        self.set_calls = []

    def _set_nested_value(self, path, value):
        self.set_calls.append((path, value))
        current = self._store
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _get_nested_value(self, path):
        self.get_calls.append(path)
        current = self._store
        if not path:
            return current
        for key in path:
            current = current[key]
        return current


def test_abstract_methods():
    with pytest.raises(TypeError):
        PathDict()


def test_path_tracking():
    d = MockPathDict()
    path = d["x"]["y"]["z"]
    assert path._path == ["x", "y", "z"]
    assert path._parent == d


def test_direct_assignment():
    d = MockPathDict()
    d["a"] = 1
    assert d.set_calls == [(["a"], 1)]


def test_string_representation():
    d = MockPathDict()
    d["a"]["b"] = 1
    path = d["a"]["b"]
    assert str(path) == "1"
    assert str(d["x"]["y"]) == "<MockPathDict: unassigned path ['x', 'y']>"


# Integration tests with concrete implementation
def test_dict_implementation():
    d = DictPathDict()
    d["a"]["b"]["c"] = 1
    d["x"] = 2
    assert str(d) == "{'a': {'b': {'c': 1}}, 'x': 2}"


def test_dict_nested_access():
    d = DictPathDict()
    d["a"]["b"] = {"c": 1}
    assert str(d["a"]) == "{'b': {'c': 1}}"


def test_dict_with_initial_data():
    d = DictPathDict({"a": {"b": 1}})
    d["a"]["c"] = 2
    assert str(d) == "{'a': {'b': 1, 'c': 2}}"


def test_error_accessing_value_as_dict():
    d = DictPathDict()
    d["a"] = 1
    with pytest.raises(TypeError):
        d["a"]["b"] = 2


def test_parallel_paths():
    d = DictPathDict()
    path1 = d["a"]["b"]
    path2 = d["x"]["y"]
    assert path1._path == ["a", "b"]
    assert path2._path == ["x", "y"]

    d["a"]["b"] = 1
    d["x"]["y"] = 2
    assert str(path1) == "1"
    assert str(path2) == "2"


def test_basic_operations():
    fs = MemoryFileSystem()
    d = FSPathDict(fs)

    d["a"] = 1
    assert d["a"] == 1

    d["b"] = 2
    assert d["b"] == 2
