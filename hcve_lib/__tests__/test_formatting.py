from hcve_lib.formatting import format_number, format_percents


def test_format_number():
    assert format_number(123132) == '123,132'
    assert format_number(123.132) == '123.13'


def test_format_percents():
    assert format_percents(0.5) == '50.0%'
