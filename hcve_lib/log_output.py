import sys
from contextlib import contextmanager
from io import StringIO
from multiprocessing import Manager
from typing import Iterator

from ansi2html import Ansi2HTMLConverter
from mlflow import log_text


class CopyOutput(object):
    def __init__(self, output, file):
        self.terminal = output
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        # self.file.write(message)
        self.file.append(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        # self.file.flush()


# output_buffer = StringIO()
manager = Manager()
output_buffer = manager.list()
original_stdout = sys.stdout
stdout = CopyOutput(original_stdout, output_buffer)
sys.stdout = stdout  # type: ignore
original_stderr = sys.stderr
stderr = CopyOutput(original_stderr, output_buffer)
sys.stderr = stderr  # type: ignore


@contextmanager
def capture_output() -> Iterator:
    # output_buffer.truncate(0)
    # output_buffer.seek(0)
    output_buffer[:] = []
    yield lambda: "".join(list(output_buffer))
    # output_buffer.flush()
    # this_buffer.write(output_buffer.getvalue())
    # this_buffer.flush()


def log_output(output, path: str = 'log.html') -> None:
    html = Ansi2HTMLConverter().convert(output)
    log_text(html, path)
