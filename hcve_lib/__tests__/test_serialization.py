import numpy
from numpy import array
from sklearn.linear_model import LogisticRegression

from hcve_lib.serialization import serialize_base64, deserialize_base64, to_json_serializable


def test_serialize_base64():
    obj = {'test': 45}
    assert obj == deserialize_base64(serialize_base64(obj))

    pipeline = LogisticRegression()
    pipeline.fit([[1], [2], [3]], [4, 5, 6])
    deserialized_pipeline = deserialize_base64(serialize_base64(pipeline))
    numpy.testing.assert_equal(deserialized_pipeline.coef_, pipeline.coef_)


def test_to_json_serializable():
    assert to_json_serializable({'a': array([1, 2, 3])}) == {'a': [1, 2, 3]}
