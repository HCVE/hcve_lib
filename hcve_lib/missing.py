from collections import defaultdict
from typing import Dict, List, Tuple

from pandas import DataFrame

from hcve_lib.utils import get_fraction_missing


def get_missing_per_cohort(
        X: DataFrame,
        data: DataFrame,
        threshold_missing: float = 0.05
) -> Dict[str, List[Tuple[str, float]]]:
    missing_in_cohorts = defaultdict(list)
    for study_name, group in X.groupby(data['STUDY']):
        for feature in group.columns:
            fraction_missing = get_fraction_missing(group[feature])
            if fraction_missing >= threshold_missing:
                missing_in_cohorts[feature]\
                    .append((study_name, f'{fraction_missing:.2f}'))
    return dict(missing_in_cohorts)
