import threading
from typing import Dict

from hcve_lib.utils import empty_dict

global_context = threading.local()


class Context:
    current_context: Dict

    def __init__(self, **context):
        global global_context
        global_context = context

    def __enter__(self, **context):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_context
        global_context = {}


def get_context() -> Dict:
    return global_context
