import pickle
from typing import Any, Union, List, Optional
from pathlib import Path
import fsspec
import posixpath
from typing import Iterator, Tuple
from hcve_lib.visualisation import h1, h2, h3, h4, h5, h6

import pytest
from unittest.mock import patch


class HyperShelve:
    def __init__(
        self,
        base_path: Union[str, Path],
        filesystem: Optional[fsspec.AbstractFileSystem] = None,
    ):
        """
        Initialize the storage with a base directory path and optional filesystem.

        Args:
            base_path: Base path for storage
            filesystem: fsspec filesystem object. If None, local filesystem is used.
        """
        self.base_path = str(base_path)
        self.fs = filesystem or fsspec.filesystem("file")

        # Ensure base path exists
        if not self.fs.exists(self.base_path):
            self.fs.makedirs(self.base_path)

    def _normalize_path(self, *parts: str) -> str:
        """Convert path parts to a normalized path string."""
        return posixpath.join(self.base_path, *parts)

    def _get_path(self, keys: Union[str, List[str], tuple]) -> str:
        """Convert a key or sequence of keys into a filesystem path."""
        if isinstance(keys, (str, int, float)):
            keys = [str(keys)]
        elif not isinstance(keys, (list, tuple)):
            raise TypeError("Keys must be a string or sequence of strings")

        # Convert all keys to strings and sanitize
        keys = [str(k) for k in keys]
        if len(keys) > 1:
            dir_path = self._normalize_path(*keys[:-1])
        else:
            dir_path = self.base_path

        return self._normalize_path(*keys[:-1], f"{keys[-1]}.pickle")

    def __setitem__(self, keys: Union[str, List[str]], value: Any) -> None:
        """Save a value to storage using a key or sequence of keys."""
        path = self._get_path(keys)
        dir_path = posixpath.dirname(path)

        if not self.fs.exists(dir_path):
            self.fs.makedirs(dir_path)

        with self.fs.open(path, "wb") as f:
            pickle.dump(value, f)

    def __getitem__(self, keys: Union[str, List[str]]) -> Any:
        """Retrieve a value from storage using a key or sequence of keys."""
        path = self._get_path(keys)
        if not self.fs.exists(path):
            raise KeyError(f"No value found for keys: {keys}")

        with self.fs.open(path, "rb") as f:
            return pickle.load(f)

    def __delitem__(self, keys: Union[str, List[str]]) -> None:
        """Delete a value from storage using a key or sequence of keys."""
        path = self._get_path(keys)
        if not self.fs.exists(path):
            raise KeyError(f"No value found for keys: {keys}")

        self.fs.delete(path)

        # Clean up empty directories
        current_dir = posixpath.dirname(path)
        while current_dir != self.base_path:
            try:
                if not self.fs.ls(current_dir):  # Directory is empty
                    self.fs.delete(current_dir)
                    current_dir = posixpath.dirname(current_dir)
                else:
                    break
            except Exception:  # Directory not empty or other error
                break

    def __contains__(self, keys: Union[str, List[str]]) -> bool:
        """Check if a key or sequence of keys exists in storage."""
        return self.fs.exists(self._get_path(keys))

    def list_keys(self, prefix: Union[str, List[str], None] = None) -> List[List[str]]:
        """List all keys under a given prefix. If no prefix is provided, list all keys."""
        if prefix is None:
            start_path = self.base_path
        else:
            if isinstance(prefix, (str, int, float)):
                prefix = [str(prefix)]
            start_path = self._normalize_path(*[str(k) for k in prefix])

        if not self.fs.exists(start_path):
            return []

        result = []
        for file_info in self.fs.find(start_path, maxdepth=None):
            if not file_info.endswith(".pickle"):
                continue

            rel_path = posixpath.relpath(file_info, self.base_path)
            if rel_path == ".":
                continue

            # Split path and remove .pickle extension from last component
            path_parts = rel_path.split("/")
            path_parts[-1] = path_parts[-1][:-7]  # Remove .pickle extension
            result.append(path_parts)

        return result

    def get_key_structure(self) -> dict:
        """
        Returns a nested dictionary representing the structure of keys in storage,
        without loading the actual pickle objects.

        Returns:
            dict: Nested dictionary where each level represents the key hierarchy
        """
        structure = {}

        for key_list in self.list_keys():
            current = structure
            # Navigate through each part of the key list
            for part in key_list:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return structure

    def clear(self) -> None:
        """Remove all items from storage."""
        if self.fs.exists(self.base_path):
            for path in self.fs.find(self.base_path):
                try:
                    self.fs.delete(path)
                except Exception:
                    pass  # Skip if deletion fails

            # Clean up empty directories
            for path in sorted(
                self.fs.find(self.base_path, maxdepth=None), key=len, reverse=True
            ):
                if self.fs.isdir(path) and path != self.base_path:
                    try:
                        self.fs.delete(path)
                    except Exception:
                        pass  # Skip if deletion fails

    def items(self) -> Iterator[Tuple[List[str], Any]]:
        """
        Implements items method for HyperShelve.
        Yields all complete key paths and their values in the storage structure.

        Returns:
            Iterator yielding tuples of (key_path, value) where:
            - key_path: List[str] representing the full path to the value
            - value: The deserialized object stored at that path

        Example:
            for key_path, value in store.items():
                print(f"At {key_path}: {value}")
        """
        for key_path in self.list_keys():
            yield key_path, self[key_path]

    def __iter__(self):
        """
        Implements iterator protocol for HyperShelve.
        Yields all complete key paths in the storage structure.
        """
        for key_path in self.list_keys():
            yield key_path

    def display_headers(self) -> Iterator[Tuple[List[str], Any]]:
        """
        Generate hierarchical headers for each level in the structure.
        Only displays a header when it's first encountered at that level and path.

        Example:
            If structure is:
            dataset1/
                model1/
                    params1
                model2/
                    params1

            Will display:
            h1("dataset1")  # First and only time at level 0
            h2("model1")    # First time at level 1
            h3("params1")   # First time at level 2 under model1
            h2("model2")    # New occurrence at level 1
            h3("params1")   # First time at level 2 under model2

        Returns:
            Iterator yielding tuples of (key_path, value)
        """
        header_functions = [h1, h2, h3, h4, h5, h6]
        displayed_paths = set()  # Track which paths we've displayed headers for

        def _get_header_function(level: int):
            return header_functions[min(level, len(header_functions) - 1)]

        for key_path in self:
            # Check and display headers for each level of the path
            for i in range(len(key_path)):
                current_path = tuple(key_path[: i + 1])
                if current_path not in displayed_paths:
                    header_function = _get_header_function(i)
                    header_function(key_path[i])
                    displayed_paths.add(current_path)

            yield key_path, self[key_path]


@pytest.fixture
def mock_heading_functions():
    """Mock all heading functions h1-h6."""
    with (
        patch("hcve_lib.h1") as mock_h1,
        patch("hcve_lib.h2") as mock_h2,
        patch("hcve_lib.h3") as mock_h3,
        patch("hcve_lib.h4") as mock_h4,
        patch("hcve_lib.h5") as mock_h5,
        patch("hcve_lib.h6") as mock_h6,
    ):
        # Setup return values to be unique for each heading level
        mock_h1.side_effect = lambda x: f"<h1>{x}</h1>"
        mock_h2.side_effect = lambda x: f"<h2>{x}</h2>"
        mock_h3.side_effect = lambda x: f"<h3>{x}</h3>"
        mock_h4.side_effect = lambda x: f"<h4>{x}</h4>"
        mock_h5.side_effect = lambda x: f"<h5>{x}</h5>"
        mock_h6.side_effect = lambda x: f"<h6>{x}</h6>"

        yield {
            "h1": mock_h1,
            "h2": mock_h2,
            "h3": mock_h3,
            "h4": mock_h4,
            "h5": mock_h5,
            "h6": mock_h6,
        }


def test_generate_headers_empty(file_storage, mock_heading_functions):
    """Test header generation with empty storage."""
    headers = list(file_storage.display_headers())
    assert headers == []
    for mock_fn in mock_heading_functions.values():
        mock_fn.assert_not_called()


def test_generate_headers_single_level(file_storage, mock_heading_functions):
    """Test header generation with single-level keys."""
    file_storage["key1"] = "value1"
    file_storage["key2"] = "value2"

    headers = list(file_storage.display_headers())

    assert len(headers) == 2
    assert headers == [(["key1"], "<h1>key1</h1>"), (["key2"], "<h1>key2</h1>")]

    mock_heading_functions["h1"].assert_any_call("key1")
    mock_heading_functions["h1"].assert_any_call("key2")
    assert mock_heading_functions["h1"].call_count == 2

    # Other heading functions should not be called
    for name, mock_fn in mock_heading_functions.items():
        if name != "h1":
            mock_fn.assert_not_called()


def test_generate_headers_nested(file_storage, mock_heading_functions):
    """Test header generation with nested structure."""
    # Create nested structure
    file_storage[["dataset1", "model1", "params1"]] = "value1"
    file_storage[["dataset1", "model2", "params1"]] = "value2"

    headers = list(file_storage.display_headers())

    # Expected calls in order
    expected_calls = [
        (["dataset1"], "<h1>dataset1</h1>"),
        (["dataset1", "model1"], "<h2>model1</h2>"),
        (["dataset1", "model1", "params1"], "<h3>params1</h3>"),
        (["dataset1", "model2"], "<h2>model2</h2>"),
        (["dataset1", "model2", "params1"], "<h3>params1</h3>"),
    ]

    assert headers == expected_calls

    # Verify correct heading functions were called with correct arguments
    mock_heading_functions["h1"].assert_called_once_with("dataset1")
    assert mock_heading_functions["h2"].call_count == 2
    mock_heading_functions["h2"].assert_any_call("model1")
    mock_heading_functions["h2"].assert_any_call("model2")
    assert mock_heading_functions["h3"].call_count == 2
    mock_heading_functions["h3"].assert_any_call("params1")


def test_generate_headers_deep_structure(file_storage, mock_heading_functions):
    """Test header generation with structure deeper than 6 levels."""
    deep_key = [
        "level1",
        "level2",
        "level3",
        "level4",
        "level5",
        "level6",
    ]
    file_storage[deep_key] = "deep_value"

    headers = list(file_storage.display_headers())

    # Check that levels beyond 6 use h6
    for i, (path, header) in enumerate(headers):
        level = i + 1
        if level <= 6:
            mock_heading_functions[f"h{level}"].assert_any_call(f"level{level}")
        else:
            mock_heading_functions["h6"].assert_any_call(f"level{level}")


def test_generate_headers_call_order(file_storage, mock_heading_functions):
    """Test that headers are generated in the correct order."""
    file_storage[["a", "b", "c"]] = "value1"
    file_storage[["a", "b", "d"]] = "value2"

    # Collect all calls to track order
    calls = []
    for _, header in file_storage.display_headers():
        calls.append(header)

    # Verify the order
    expected_order = [
        "<h1>a</h1>",
        "<h2>b</h2>",
        "<h3>c</h3>",
        "<h2>b</h2>",  # b appears again because it's part of the second path
        "<h3>d</h3>",
    ]

    assert calls == expected_order
