from multiprocessing import Value, Lock
from typing import Callable, Optional


class ProgressReporter:
    message: Optional[str] = None

    def __init__(
        self,
        total_items=None,
        finished_counter=None,
        lock=None,
        # on_progress: Callable[[float, ...], None] = None,
        on_progress: Callable = None,
    ):
        self.finished_counter = finished_counter or Value("i", 0)
        self._total_items = total_items
        self.lock = lock or Lock()
        self.on_progress = on_progress or self.default_callback

    @property
    def fraction(self):
        return min(self.finished_counter.value / self.total, 1.0)

    @property
    def total(self):
        with self.lock:
            return self._total_items

    @total.setter
    def total(self, value):
        with self.lock:
            if value is not None and value <= 0:
                raise ValueError("Total items must be a positive integer.")
            self._total_items = value

        self.on_progress(self.fraction)

    def default_callback(self, fraction: float = None, message: str = None):
        if self.total is None:
            raise ValueError("total_items is not set.")
        return f"{self.finished_counter.value}/{self.total} ({fraction * 100:.1f}%) items finished."

    def set_message(self, message: str):
        self.message = message
        return self.on_progress(self.fraction, self.message)

    def finished(self):
        with self.lock:
            self.finished_counter.value += 1

        return self.on_progress(self.fraction, self.message)
