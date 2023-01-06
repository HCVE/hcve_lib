from abc import ABC, abstractmethod
from enum import auto
from typing import Union, List

from hcve_lib.custom_types import Prediction, Target, ExceptionValue, TargetData, ValueWithCI, StrEnum
from hcve_lib.utils import get_y_split


class OptimizationDirection(StrEnum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


class Metric(ABC):
    def __init__(self, is_test: bool = True):
        self.is_test = is_test

    def get_y(
            self,
            y: Target,
            prediction: Prediction,
    ) -> TargetData:
        y_train, y_test = get_y_split(y, prediction)
        return y_test if self.is_test else y_train

    @abstractmethod
    def get_names(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[str]:
        ...

    @abstractmethod
    def get_values(
            self,
            prediction: Prediction,
            y: Target,
    ) -> List[Union[ExceptionValue, float, ValueWithCI]]:
        ...

    @abstractmethod
    def get_direction(self) -> OptimizationDirection:
        ...
