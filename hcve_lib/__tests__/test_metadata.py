from hcve_lib.data import (
    has_children,
    flatten_metadata,
    find_item,
    format_identifier,
    Metadata,
)


def test_has_children() -> None:
    assert has_children({}) is False
    assert has_children({"children": []}) is True


def test_flatten_metadata() -> None:
    assert list(
        flatten_metadata(
            [
                {"identifier": "1", "children": [{"identifier": "2"}]},
                {"identifier": "3"},
            ]
        )
    ) == [
        {"identifier": "1", "children": [{"identifier": "2"}]},
        {"identifier": "2"},
        {"identifier": "3"},
    ]


def test_find_item() -> None:
    assert find_item(
        "3",
        [
            {"identifier": "1", "children": [{"identifier": "2"}]},
            {"identifier": "3"},
        ],
    ) == {"identifier": "3"}


def test_format_identifier() -> None:
    metadata: Metadata = [
        {
            "identifier": "1",
            "name": "a",
        },
        {"identifier": "3"},
    ]
    assert (
        format_identifier(
            "1",
            metadata,
        )
        == "a"
    )
