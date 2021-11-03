from logging import Logger
from typing import Callable, TypedDict, Optional, Iterable

from pandas import DataFrame

from hcve_lib.formatting import format_number
from logging import Logger
from typing import Iterable
from pandas import DataFrame
from typing import Callable
from hcve_lib.data import inverse_format_value, Metadata, find_item, format_value


class Step(TypedDict, total=False):
    action: Callable[[DataFrame], DataFrame]
    log: Callable[[Logger, DataFrame, Optional[DataFrame]], None]


def perform(steps: Iterable[Step], logger: Logger = None) -> DataFrame:
    current_data = None
    previous_data = None

    for step in steps:
        if step.get('action'):
            current_data = step['action'](previous_data)

        if logger and step.get('log'):
            step['log'](logger, current_data, previous_data)
        previous_data = current_data

    return current_data


def log_step(description: str, metadata: Metadata) -> Callable:
    def log_step_(
        logger: Logger,
        current: DataFrame,
        previous: DataFrame,
    ) -> None:
        current_study_nrs = set(current["STUDY_NUM"].unique())
        if previous is None:
            previous_study_nrs = None
        else:
            previous_study_nrs = set(previous["STUDY_NUM"].unique())

        cohort_nr_changed = previous is not None and len(
            current_study_nrs) != len(previous_study_nrs)

        logger.info(f'{description}\n' + ('' if previous is None else ((
            ' ' if not cohort_nr_changed else
            (f'\tn cohorts removed={len(previous_study_nrs)-len(current_study_nrs)}: '
             +
             f'{", ".join(get_cohorts_identifiers(previous_study_nrs-current_study_nrs, metadata))}\n'
             )
        ) + f'\tn individuals removed={format_number(len(previous)-len(current))}\n\n'
                                                                       )) +
                    ((' ' if not (cohort_nr_changed or previous is None) else
                      f'\tn cohorts={len(current["STUDY_NUM"].unique())}\n') +
                     f'\tn individuals={format_number(len(current))}\n'))

    return log_step_


def remove_cohorts(
    data: DataFrame,
    cohorts: Iterable[str],
    metadata: Metadata,
) -> DataFrame:
    return remove_cohort_nrs(data, get_cohort_nrs(cohorts, metadata))


def remove_cohort_nrs(data: DataFrame, cohort_nrs: Iterable[str]) -> DataFrame:
    return data[~data['STUDY_NUM'].isin(cohort_nrs)]


def get_cohort_nrs(
    cohort_names: Iterable[str],
    metadata: Metadata,
) -> Iterable[int]:
    return (inverse_format_value(cohort_name, find_item('STUDY_NUM', metadata))
            for cohort_name in cohort_names)


def get_cohorts_identifiers(
    cohort_nrs: Iterable[int],
    metadata: Metadata,
) -> Iterable[str]:
    return [
        format_value(cohort_nr, find_item('STUDY_NUM', metadata))
        for cohort_nr in cohort_nrs
    ]
