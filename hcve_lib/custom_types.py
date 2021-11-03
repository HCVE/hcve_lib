from abc import abstractmethod, ABC
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, Optional, Tuple, Generic, TypeVar, Any, Union, List

import numpy as np
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


class Estimator(BaseEstimator):
    @abstractmethod
    def predict(self, X):
        ...


class EstimatorProba(Estimator):
    @abstractmethod
    def predict_proba(self, X):
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
    std: float
    ci: Optional[Tuple[float, float]]


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


class FoldPrediction(TypedDict):
    X_train: DataFrame
    X_test: DataFrame
    y_train: DataFrame
    y_true: Series
    y_score: Any
    model: Optional[Estimator]


SurvivalPairTarget = namedtuple('SurvivalPairTarget', ('tte', 'label'))
Target = Union[Series, np.recarray]
FoldInput = Tuple[List[int], List[int]]
