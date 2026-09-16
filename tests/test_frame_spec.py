import pytest

from wavefront_refocus.core.frame_spec import parse_frame_spec


def test_simple_list():
    assert parse_frame_spec("1,38,39,100") == [1, 38, 39, 100]


def test_blank_is_empty():
    assert parse_frame_spec("") == []
    assert parse_frame_spec("   ") == []
    assert parse_frame_spec(None) == []


def test_ranges_and_mixed():
    assert parse_frame_spec("0-3, 7, 10-12") == [0, 1, 2, 3, 7, 10, 11, 12]


def test_alternate_range_syntax():
    assert parse_frame_spec("2..4") == [2, 3, 4]
    assert parse_frame_spec("2:4") == [2, 3, 4]


def test_spaces_around_range_operator_rejected():
    # Whitespace separates entries, so a spaced range cannot be read as one.
    with pytest.raises(ValueError, match="cannot read"):
        parse_frame_spec("2 - 4")


def test_whitespace_and_semicolon_separators():
    assert parse_frame_spec("1 2;3") == [1, 2, 3]


def test_sorted_and_deduplicated():
    assert parse_frame_spec("5,1,5,3,1") == [1, 3, 5]


def test_reversed_range_normalised():
    assert parse_frame_spec("6-4") == [4, 5, 6]


def test_validates_against_known_frames():
    assert parse_frame_spec("1,2", valid=range(5)) == [1, 2]
    with pytest.raises(ValueError, match="outside the loaded sequence"):
        parse_frame_spec("1,99", valid=range(5))


def test_garbage_raises():
    with pytest.raises(ValueError, match="cannot read"):
        parse_frame_spec("1,abc")
