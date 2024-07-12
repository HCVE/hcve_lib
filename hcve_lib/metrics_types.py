from abc import ABC, abstractmethod
from enum import auto
from typing import Union, List

from pandas import DataFrame

from hcve_lib.custom_types import (
    Prediction,
    Target,
    ExceptionValue,
    TargetData,
    ValueWithCI,
    StrEnum,
)
from hcve_lib.utils import get_y_split, loc


class OptimizationDirection(StrEnum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


class Metric(ABC):
    is_test = True

    def get_y(
        self,
        y: Target,
        prediction: Prediction,
    ) -> TargetData:
        y_train, y_test = get_y_split(y, prediction)
        return y_test if self.is_test else y_train

    # @abstractmethod
    # def compute(
    #     self, y_true: Target, y_pred: DataFrame
    # ) -> Union[List[Union[ExceptionValue, float]], Union[ExceptionValue, float]]: ...

    @abstractmethod
    def get_names(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[str]: ...

    def get_values(
        self,
        prediction: Prediction,
        y: Target,
    ) -> List[Union[ExceptionValue, float]]:
        _y = self.get_y(y, prediction)
        y_pred = loc(_y.index, prediction["y_pred"])
        return self.compute(_y, y_pred)

    @abstractmethod
    def get_direction(self) -> OptimizationDirection: ...
