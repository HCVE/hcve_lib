from unittest.mock import call

from hcve_lib.custom_types import ValueWithCI
from hcve_lib.tracking import log_metrics, encode_run_name


def test_log_metrics_ci(mocker):
    log_metric = mocker.patch('hcve_lib.tracking.log_metric', return_value=None)
    log_metrics({'metric': ValueWithCI(mean=20, ci=(10, 30))})
    log_metric.assert_has_calls([call('metric_l', 10), call('metric', 20), call('metric_r', 30)])


def test_encode_run_name():
    assert encode_run_name('xxx') == 'xxx'
    assert encode_run_name(('a', 'b')) == 'a,b'
