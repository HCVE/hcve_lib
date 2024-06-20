from multiprocessing import Value, Lock
import pytest

from hcve_lib.progress_reporter import ProgressReporter


def test_default_callback():
    counter = Value("i", 0)
    reporter = ProgressReporter(finished_counter=counter)

    # Setting the total items
    reporter.total = 5

    # No items finished yet
    assert reporter.default_callback() == "0/5 items finished."

    counter.value = 3
    assert reporter.default_callback() == "3/5 items finished."


def test_default_callback_without_total():
    reporter = ProgressReporter()

    # Attempting to call the callback without setting total items should raise a ValueError
    with pytest.raises(ValueError, match="total_items is not set."):
        reporter.default_callback()


def test_finished():
    counter = Value("i", 0)
    lock = Lock()
    reporter = ProgressReporter(5, finished_counter=counter, lock=lock)

    # Mimic one item finished
    result = reporter.finished()
    assert result == "1/5 items finished."
    assert counter.value == 1

    # Mimic another item finished
    result = reporter.finished()
    assert result == "2/5 items finished."
    assert counter.value == 2


def test_custom_callback():
    def new_callback():
        return "New Callback!"

    reporter = ProgressReporter(5)
    reporter.on_progress = new_callback

    assert reporter.on_progress() == "New Callback!"
