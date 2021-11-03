from abc import abstractmethod

from typing import TypedDict, Optional, Any


class MethodInfo(TypedDict):
    parallel: bool
    iterations: Optional[int]


class Method:
    @staticmethod
    def get_info() -> MethodInfo:
        return MethodInfo(parallel=True, iterations=None)

    @staticmethod
    @abstractmethod
    def get_optuna_hyperparameters() -> Any:
        ...

    @staticmethod
    @abstractmethod
    def get_pipeline() -> Any:
        ...
