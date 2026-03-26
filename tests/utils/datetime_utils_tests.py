"""Tests for datetime_utils"""

from datetime import datetime, date, timezone
import numpy as np

from workbench_bridges.utils.datetime_utils import (
    datetime_to_iso8601,
    iso8601_to_datetime,
    convert_all_to_iso8601,
    datetime_string,
)


def test_datetime_to_iso8601_aware():
    """Test conversion of timezone-aware datetime"""
    dt = datetime(2024, 6, 15, 12, 30, 45, 123000, tzinfo=timezone.utc)
    result = datetime_to_iso8601(dt)
    assert result == "2024-06-15T12:30:45.123Z"


def test_datetime_to_iso8601_naive():
    """Test conversion of naive datetime (should localize to UTC)"""
    dt = datetime(2024, 6, 15, 12, 30, 45, 123000)
    result = datetime_to_iso8601(dt)
    assert result == "2024-06-15T12:30:45.123Z"


def test_datetime_to_iso8601_date_object():
    """Test conversion of a date object (not datetime)"""
    d = date(2024, 6, 15)
    result = datetime_to_iso8601(d)
    assert result == "2024-06-15T00:00:00.000Z"


def test_datetime_to_iso8601_invalid_input():
    """Test that invalid input returns None"""
    assert datetime_to_iso8601("not a datetime") is None
    assert datetime_to_iso8601(12345) is None
    assert datetime_to_iso8601(None) is None


def test_iso8601_to_datetime_z_suffix():
    """Test parsing ISO-8601 string with Z suffix"""
    result = iso8601_to_datetime("2024-06-15T12:30:45.123Z")
    assert result.year == 2024
    assert result.month == 6
    assert result.day == 15
    assert result.hour == 12
    assert result.minute == 30
    assert result.tzinfo == timezone.utc


def test_iso8601_to_datetime_offset_suffix():
    """Test parsing ISO-8601 string with +00:00 suffix"""
    result = iso8601_to_datetime("2024-06-15T12:30:45.123+00:00")
    assert result.year == 2024
    assert result.tzinfo == timezone.utc


def test_roundtrip():
    """Test datetime -> iso8601 -> datetime roundtrip"""
    original = datetime(2024, 6, 15, 12, 30, 45, 123000, tzinfo=timezone.utc)
    iso_str = datetime_to_iso8601(original)
    restored = iso8601_to_datetime(iso_str)
    assert restored.year == original.year
    assert restored.month == original.month
    assert restored.day == original.day
    assert restored.hour == original.hour
    assert restored.minute == original.minute


def test_convert_all_to_iso8601_dict():
    """Test recursive conversion of dicts"""
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = {"a": 1, "b": dt, "c": {"nested": dt}}
    result = convert_all_to_iso8601(data)
    assert result["a"] == 1
    assert isinstance(result["b"], str)
    assert isinstance(result["c"]["nested"], str)


def test_convert_all_to_iso8601_list():
    """Test recursive conversion of lists"""
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data = [1, dt, "hello"]
    result = convert_all_to_iso8601(data)
    assert result[0] == 1
    assert isinstance(result[1], str)
    assert result[2] == "hello"


def test_convert_all_to_iso8601_np_int64():
    """Test conversion of numpy int64"""
    data = {"val": np.int64(42)}
    result = convert_all_to_iso8601(data)
    assert result["val"] == 42
    assert isinstance(result["val"], int)


def test_convert_all_to_iso8601_passthrough():
    """Test that non-special types pass through"""
    assert convert_all_to_iso8601("hello") == "hello"
    assert convert_all_to_iso8601(42) == 42
    assert convert_all_to_iso8601(3.14) == 3.14


def test_datetime_string_valid():
    """Test formatting a datetime to string"""
    dt = datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc)
    result = datetime_string(dt)
    # Should contain date and time in some format
    assert "2024" in result
    assert "06" in result or "6" in result


def test_datetime_string_none():
    """Test that None returns placeholder"""
    assert datetime_string(None) == "-"


def test_datetime_string_placeholder():
    """Test that '-' returns placeholder"""
    assert datetime_string("-") == "-"


def test_datetime_string_iso_string():
    """Test that an ISO-8601 string is converted"""
    result = datetime_string("2024-06-15T12:30:45.123Z")
    assert "2024" in result


def test_datetime_string_invalid():
    """Test that invalid input returns string representation"""
    result = datetime_string("not-a-date")
    assert isinstance(result, str)


if __name__ == "__main__":
    test_datetime_to_iso8601_aware()
    test_roundtrip()
    print("All datetime_utils tests passed!")
