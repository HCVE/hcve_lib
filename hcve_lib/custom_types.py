import traceback
from abc import abstractmethod, ABC
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from enum import auto, Enum
from typing import TypedDict, Optional, Tuple, Generic, TypeVar, Any, Union, List, Dict, Hashable, Callable, Type

import numpy as np
from optuna import Trial
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator


class IndexAccess(ABC):

    @abstractmethod
    def __getitem__(self, key):
        ...

    @abstractmethod
    def __setitem__(self, key, value):
        ...


class DictAccess:

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class Printable:

    def __str__(self):
        return '\n'.join([
            f'{key}: {value}' for key, value in self.__dict__.items()
            if not key.startswith('_')
        ])


class ClassMapping(Mapping):

    def __getitem__(self, item):
        try:
            return self.__dict__[item]
        except AttributeError:
            return self[item]

    def __iter__(self):
        return (k for k in self.__dict__.keys())

    def __len__(self):
        return len(self.__dict__.keys())


class DataStructure(DictAccess, ClassMapping, Printable):
    ...


class TargetTransformer(BaseEstimator):

    @abstractmethod
    def fit(self, y):
        ...

    @abstractmethod
    def transform(self, y):
        ...

    @abstractmethod
    def inverse_transform(self, y):
        ...


class Estimator(BaseEstimator):

    @abstractmethod
    def fit(self, X, y, **kwargs):
        ...

    @abstractmethod
    def predict(self, X, **kwargs):
        ...

    @abstractmethod
    def predict_proba(self, X, **kwargs):
        ...

    @abstractmethod
    def predict_survival_function(self, X, **kwargs):
        ...

    @abstractmethod
    def predict_survival(self, X, *args, **kwargs):
        ...

    @abstractmethod
    def score(self, X, y):
        ...


class ClassificationMetrics(TypedDict):
    recall: float
    precision: float
    f1: float
    tnr: float
    fpr: float
    fnr: float
    accuracy: float
    roc_auc: float
    average_precision: float
    balanced_accuracy: float
    brier_score: float
    npv: float


class ValueWithCI(TypedDict):
    mean: float
    ci: Tuple[float, float]


class ValueWithStatistics(TypedDict):
    mean: float
    ci: Optional[Tuple[float, float]]
    std: float


class ClassificationMetricsWithStatistics(TypedDict):
    recall: ValueWithStatistics
    precision: ValueWithStatistics
    f1: ValueWithStatistics
    tnr: ValueWithStatistics
    fpr: ValueWithStatistics
    fnr: ValueWithStatistics
    npv: ValueWithStatistics
    accuracy: ValueWithStatistics
    roc_auc: ValueWithStatistics
    average_precision: ValueWithStatistics
    balanced_accuracy: ValueWithStatistics


T1 = TypeVar('T1')


@dataclass
class GenericConfusionMatrix(DataStructure, Generic[T1]):
    fn: T1
    tn: T1
    tp: T1
    fp: T1


@dataclass
class ConfusionMatrix(DataStructure):
    fn: float
    tn: float
    tp: float
    fp: float


class ConfusionMetrics(DataStructure):
    recall: float
    precision: float
    f1: float
    fpr: float
    tnr: float
    fnr: float
    npv: float

    def __init__(self, recall, precision, fpr, tnr, fnr, npv):
        self.fnr = fnr
        self.tnr = tnr
        self.recall = recall
        self.precision = precision
        self.fpr = fpr
        self.npv = npv
        try:
            self.f1 = 2 / ((1 / precision) + (1 / recall))
        except ZeroDivisionError:
            self.f1 = 0


SurvivalPairTarget = namedtuple('SurvivalPairTarget', ('tte', 'label'))

TargetData = Union[DataFrame, Series, np.recarray]

Index = List[int]

TrainTestIndex = Tuple[Index, Index]

TrainTestSplits = Dict[Hashable, TrainTestIndex]

Splits = Dict[Hashable, Index]

TrainTestSplitter = Callable[..., TrainTestSplits]


class Target(TypedDict):
    name: str
    data: TargetData


class Prediction(TypedDict, total=False):
    y_score: Any
    y_proba: Dict[Any, Any]
    y_column: str
    X_columns: List[str]
    model: Estimator
    split: TrainTestIndex
    random_state: int
    method: Type['Method']


class Method(ABC):

    @staticmethod
    @abstractmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: Dict,
        verbose=0,
    ):
        ...

    @staticmethod
    @abstractmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        ...

    @staticmethod
    @abstractmethod
    def predict(
        X: DataFrame,
        y: Target,
        split: TrainTestIndex,
        model: Estimator,
        method: Type['Method'],
        random_state: int,
    ) -> Prediction:
        ...


Result = Dict[Hashable, Prediction]


class ExceptionValue:
    traceback: str
    value: Any

    def __init__(
        self,
        value: Any = None,
        exception: Exception = None,
    ):
        self.traceback = traceback.format_exc()
        self.value = value
        self.exception = exception

    def __repr__(self):
        return f'Value:\n {self.value}\n\n Exception:\n{self.exception}\n\n {self.traceback}'


class StrEnum(Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self._name_

    def __eq__(self, other):
        if type(self).__qualname__ != type(other).__qualname__:
            return False

        return self.name == other.name and self.value == other.value


class OptimizationDirection(StrEnum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


class Metric(ABC):

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


class Minimize:

    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE


class Maximize:

    def get_direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE
