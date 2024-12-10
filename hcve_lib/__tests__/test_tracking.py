from hcve_lib.tracking import encode_run_name


def test_encode_run_name():
    assert encode_run_name("xxx") == "xxx"
    assert encode_run_name(("a", "b")) == "a,b"
