import traceback
from abc import abstractmethod, ABC
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Optional,
    Tuple,
    Generic,
    TypeVar,
    Any,
    Union,
    List,
    Dict,
    Hashable,
    Callable,
    Type,
)
from typing_extensions import TypedDict

import numpy as np
from optuna import Trial
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator

SurvivalPairTarget = namedtuple("SurvivalPairTarget", ("tte", "label"))

TargetData = Union[DataFrame, Series, np.recarray]

Index = List[Any]

TrainTestIndex = Tuple[Index, Index]

TrainTestSplits = Dict[Hashable, TrainTestIndex]

Splits = Dict[Hashable, Index]

TrainTestSplitter = Callable[..., TrainTestSplits]


class StrEnum(Enum):
    # noinspection PyMethodParameters
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self._name_

    def __eq__(self, other):
        if type(self).__qualname__ != type(other).__qualname__:
            return False

        return self.name == other.name and self.value == other.value


class TargetType(StrEnum):
    REGRESSION = auto()
    CLASSIFICATION = auto()
    TIME_TO_EVENT = auto()
    NA = auto()


@dataclass
class TargetObject:
    _inner: TargetData
    _name: Optional[str]

    def __init__(self, data: TargetData, name: Optional[str] = None):
        self._inner = data
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return self._inner.name

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._inner

    def update_data(self, data):
        cloned = TargetObject(data=data, name=self.name)
        return cloned

    def __getattr__(self, item):
        if hasattr(self._inner, item):
            return getattr(self._inner, item)
        else:
            raise AttributeError(f"AttributeError: object has no attribute '{item}'")

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def __getitem__(self, item):
        return self._inner[item]


Target = Union[TargetObject, Series, DataFrame]


class IndexAccess(ABC):
    @abstractmethod
    def __getitem__(self, key): ...

    @abstractmethod
    def __setitem__(self, key, value): ...


class DictAccess:
    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class Printable:
    def __str__(self):
        return "\n".join(
            [
                f"{key}: {value}"
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            ]
        )


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


class DataStructure(DictAccess, ClassMapping, Printable): ...


class TargetTransformer(BaseEstimator):
    @abstractmethod
    def fit(self, y): ...

    @abstractmethod
    def transform(self, y): ...

    @abstractmethod
    def inverse_transform(self, y): ...


class Estimator:
    def fit(
        self,
        X: DataFrame,
        y: Target,
        X_validate: DataFrame = None,
        y_validate: Target = None,
    ): ...

    def predict(self, X: DataFrame): ...

    def predict_proba(self, X: DataFrame): ...

    def predict_survival_at_time(self, X: DataFrame, time: int, *args, **kwargs): ...

    def suggest_optuna(
        self, trial: Trial, X: DataFrame, prefix: str = ""
    ) -> Tuple[Trial, Dict]:
        return trial, {}

    def transform(self, X: DataFrame):
        return X

    def get_feature_importance(self):
        raise NotImplementedError(f"{self} do not implement feature importance")

    @classmethod
    def get_name(cls):
        return cls.__name__


class Pipeline:
    steps: List[Tuple[str, Any]]
    optimize: bool

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        optimize: bool = False,
    ):
        self.steps = steps
        self.optimize = optimize

    def fit(self): ...


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


Metrics = Dict[Hashable, ValueWithStatistics]


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


T1 = TypeVar("T1")


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


class Prediction(TypedDict, total=False):
    y_pred: Target
    y_column: str
    X_columns: List[str]
    model: Any
    split: TrainTestIndex


Result = Dict[Hashable, Prediction]

Results = List[Result]


class Method(ABC):
    @staticmethod
    @abstractmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: Dict,
        verbose=0,
    ): ...

    @staticmethod
    @abstractmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]: ...

    @staticmethod
    @abstractmethod
    def predict(
        X: DataFrame,
        y: Target,
        split: TrainTestIndex,
        model: Estimator,
        method: Type["Method"],
        random_state: int,
    ) -> Prediction: ...


class ExceptionValue:
    traceback: str
    value: Any

    def __init__(
        self,
        value: Any = None,
        exception: Optional[Exception] = None,
    ):
        self.traceback = traceback.format_exc()
        self.value = value
        self.exception = exception

    def __getstate__(self):
        return {
            "value": self.value,
        }

    def __repr__(self):
        return f"Value:\n {self.value}\n\n Exception:\n{self.exception}\n\n {self.traceback}"
