from pandas import Series, DataFrame

from hcve_lib.custom_types import SplitPrediction
from hcve_lib.evaluation_functions import compute_metric_groups, get_2_level_groups


def test_compute_metric_groups():
    def dummy_metric(fold: SplitPrediction):
        return (fold['y_score']).tolist()

    fold = SplitPrediction(
        y_score=Series(
            [10, 20, 30],
            index=['a', 'b', 'c'],
        ),
        y_column='y_column',
        X_columns=['a', 'b'],
        split=[([0], [1])],
        model=None,
    )
    expected_result = {0: [30], 1: [10, 20]}

    data = DataFrame(
        {'agg': [0, 1, 1, 0]},
        index=['x', 'a', 'b', 'c'],
    )
    assert compute_metric_groups(
        dummy_metric,
        get_2_level_groups(
            {
                'a': fold,
                'b': fold
            },
            data.groupby('agg'),
            data,
        )) == {
            'a': expected_result,
            'b': expected_result
        }
