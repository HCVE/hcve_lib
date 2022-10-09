import sys

import tempfile
import traceback
from contextlib import contextmanager
from io import StringIO


class CopyOutput(object):
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def get_file(self) -> str:
        self.file.flush()
        return self.file.name


@contextmanager
def redirect_std():
    file = tempfile.NamedTemporaryFile('a')

    original_stdout = sys.stdout
    stdout = CopyOutput(original_stdout, file)
    sys.stdout = stdout  # type: ignore

    original_stderr = sys.stderr
    stderr = CopyOutput(original_stderr, file)
    sys.stderr = stderr  # type: ignore

    @contextmanager
    def read_log_path():
        file.flush()
        try:
            yield file.name
        finally:
            file.close()

    try:
        yield read_log_path
    except Exception:
        stderr.write(traceback.format_exc())
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


output_buffer = StringIO()
original_stdout = sys.stdout
stdout = CopyOutput(original_stdout, output_buffer)
sys.stdout = stdout  # type: ignore
original_stderr = sys.stderr
stderr = CopyOutput(original_stderr, output_buffer)
sys.stderr = stderr  # type: ignore


def get_output_path():
    output_buffer.flush()
    return output_buffer.name


def log_output():
    print(output_buffer)
