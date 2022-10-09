from hcve_lib.context import Context, get_context


def test_get_context():
    with Context(var=10):
        assert get_context()['var'] == 10

    assert 'var' not in get_context()
