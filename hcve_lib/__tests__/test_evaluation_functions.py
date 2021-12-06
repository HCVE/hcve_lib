from pandas import Series, DataFrame

from hcve_lib.custom_types import SplitPrediction
from hcve_lib.evaluation_functions import compute_metric_groups, get_2_level_groups


def test_compute_metric_groups():
    def dummy_metric(fold: SplitPrediction):
        return (fold['y_true'] - fold['y_score']).tolist()

    fold = SplitPrediction(
        y_score=Series(
            [10, 20, 30],
            index=['a', 'b', 'c'],
        ),
        y_true=[100, 200, 300],
        model=None,
    )
    expected_result = {0: [270], 1: [90, 180]}

    assert compute_metric_groups(
        dummy_metric,
        get_2_level_groups(
            {
                'a': fold,
                'b': fold
            },
            DataFrame(
                {
                    'agg': [0, 1, 1, 0]
                },
                index=['x', 'a', 'b', 'c'],
            ).groupby('agg'),
        )) == {
            'a': expected_result,
            'b': expected_result
        }
