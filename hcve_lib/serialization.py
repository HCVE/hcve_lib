import codecs
import numpy
import pickle
from typing import Any


def serialize_base64(what: Any) -> str:
    return codecs.encode(pickle.dumps(what), "base64").decode()


def deserialize_base64(what: str) -> Any:
    return pickle.loads(codecs.decode(what.encode(), "base64"))


def to_json_serializable(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, range):
        return list(obj)
    else:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating) and not numpy.isnan(obj):
            return float(obj)
        else:
            try:
                if numpy.isnan(float(obj)):
                    return None
                else:
                    return obj
            except (ValueError, TypeError):
                return obj
