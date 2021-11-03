from hcve_lib.data import has_children, flatten_metadata, find_item, format_identifier


def test_has_children():
    assert has_children({}) is False
    assert has_children({'children': None}) is False
    assert has_children({'children': []}) is True


def test_flatten_metadata():
    assert list(
        flatten_metadata([
            {
                'identifier': '1',
                'children': [{
                    'identifier': '2'
                }]
            },
            {
                'identifier': '3'
            },
        ])) == [
            {
                'identifier': '1',
                'children': [{
                    'identifier': '2'
                }]
            },
            {
                'identifier': '2'
            },
            {
                'identifier': '3'
            },
        ]


def test_find_item():
    assert find_item('3', [
        {
            'identifier': '1',
            'children': [{
                'identifier': '2'
            }]
        },
        {
            'identifier': '3'
        },
    ]) == {
        'identifier': '3'
    }


def test_get_meaning():
    assert format_identifier(
        '1',
        [
            {
                'identifier': '1',
                'meaning': 'a',
            },
            {
                'identifier': '3'
            },
        ],
    ) == 'a'
