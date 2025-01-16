import pickle

from pathlib import PurePosixPath

from abc import ABC, abstractmethod
from fsspec import AbstractFileSystem


class PathDict(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._path = []
        self._parent = None

    def __getitem__(self, key):
        if self._parent is None:
            # Root node - start new path
            new_path_dict = self.__class__(*self.args, **self.kwargs)
            new_path_dict._path = [key]
            new_path_dict._parent = self
            return new_path_dict
        else:
            # Continue building path
            new_path_dict = self.__class__()
            new_path_dict._path = self._path + [key]
            new_path_dict._parent = self._parent
            return new_path_dict

    def __setitem__(self, key, value):
        if self._parent is None:
            # Direct assignment to root
            self._set_nested_value([key], value)
        else:
            # Get the complete path and assign to root
            path = self._path + [key]
            self._parent._set_nested_value(path, value)

    @abstractmethod
    def _set_nested_value(self, path, value):
        """Set value at the specified path"""
        pass

    @abstractmethod
    def _get_nested_value(self, path):
        """Get value at the specified path"""
        pass

    def __str__(self):
        if self._parent is None:
            return str(self._get_nested_value([]))
        try:
            return str(self._parent._get_nested_value(self._path))
        except KeyError:
            return f"<{self.__class__.__name__}: unassigned path {self._path}>"


class DictPathDict(PathDict):
    def __init__(self, data=None):
        super().__init__()
        self._data = data if data is not None else {}

    def _set_nested_value(self, path, value):
        data = self._data
        for key in path[:-1]:
            if key not in data:
                data[key] = {}
            data = data[key]
        data[path[-1]] = value

    def _get_nested_value(self, path):
        if not path:
            return self._data
        data = self._data
        for key in path:
            data = data[key]
        return data


class FSPathDict(PathDict):
    def __init__(self, fs: AbstractFileSystem, root_dir: str = "data"):
        super().__init__(fs, root_dir)
        self.fs = fs
        self.root_dir = PurePosixPath(root_dir)
        if not self.fs.exists(str(self.root_dir)):
            self.fs.makedirs(str(self.root_dir))

    def _set_nested_value(self, path, value):
        str_path = [str(p) for p in path]
        dir_path = str(self.root_dir.joinpath(*str_path[:-1]))
        file_path = str(self.root_dir.joinpath(*str_path[:-1], f"{str_path[-1]}.pkl"))

        # Create directories if they don't exist
        if not self.fs.exists(dir_path):
            self.fs.makedirs(dir_path)

        # Pickle and save the value
        with self.fs.open(file_path, "wb") as f:
            pickle.dump(value, f)

    def _get_nested_value(self, path):
        print("xxxxx")
        print(path)
        pass


def _get_nested_value(self, path):
    if not path:
        return self._data
    data = self._data
    for key in path:
        data = data[key]
    return data
