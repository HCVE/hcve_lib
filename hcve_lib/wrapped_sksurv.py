from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.meta import Stacking

from hcve_lib.wrapped_sklearn import DFWrapped, ToSurvivalRecord


class DFCoxnetSurvivalAnalysis(DFWrapped, ToSurvivalRecord, CoxnetSurvivalAnalysis):
    ...


class DFCoxPHSurvivalAnalysis(DFWrapped, CoxPHSurvivalAnalysis):
    ...


class DFStacking(DFWrapped, Stacking):
    ...
