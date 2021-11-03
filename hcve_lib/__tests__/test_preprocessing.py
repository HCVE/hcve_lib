from pandas import DataFrame
from pandas.testing import assert_frame_equal

from hcve_lib.preprocessing import perform, Step


def test_perform(mocker):
    class Logger:
        info = mocker.stub(name='on_something_stub')

    logger = Logger()

    assert_frame_equal(
        perform(
            [
                Step(action=lambda _: DataFrame({'x': [1]})),
                Step(
                    action=lambda data: data + 1,
                    log=lambda _logger, current, previous: _logger.info(
                        current.iloc[0, 0],
                        previous.iloc[0, 0],
                    ),
                ),
            ],
            logger=logger,
        ),
        DataFrame({'x': [2]}),
    )

    logger.info.assert_called_once_with(
        2,
        1,
    )
