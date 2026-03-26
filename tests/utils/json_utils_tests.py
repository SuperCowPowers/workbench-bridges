"""Tests for json_utils"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timezone

from workbench_bridges.utils.json_utils import CustomEncoder, custom_decoder


def test_encode_numpy_int():
    """Test encoding numpy integers"""
    data = {"val": np.int64(42)}
    result = json.dumps(data, cls=CustomEncoder)
    assert '"val": 42' in result


def test_encode_numpy_float():
    """Test encoding numpy floats"""
    data = {"val": np.float64(3.14)}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert abs(decoded["val"] - 3.14) < 0.001


def test_encode_numpy_bool():
    """Test encoding numpy booleans"""
    data = {"val": np.bool_(True)}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["val"] is True


def test_encode_numpy_array():
    """Test encoding numpy arrays"""
    data = {"arr": np.array([1, 2, 3])}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["arr"] == [1, 2, 3]


def test_encode_datetime():
    """Test encoding datetime objects"""
    dt = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    data = {"dt": dt}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["dt"]["__datetime__"] is True
    assert "2024-06-15" in decoded["dt"]["datetime"]


def test_encode_date():
    """Test encoding date objects"""
    d = date(2024, 6, 15)
    data = {"d": d}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["d"]["__datetime__"] is True


def test_encode_dataframe():
    """Test encoding pandas DataFrames"""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    data = {"df": df}
    result = json.dumps(data, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["df"]["__dataframe__"] is True
    assert "df" in decoded["df"]
    assert "index" in decoded["df"]


def test_encode_dataframe_named_index():
    """Test encoding DataFrame with named index"""
    df = pd.DataFrame({"A": [1, 2]})
    df.index.name = "my_index"
    result = json.dumps({"df": df}, cls=CustomEncoder)
    decoded = json.loads(result)
    assert decoded["df"]["index_name"] == "my_index"


def test_precision_reduction():
    """Test encoding with precision reduction"""
    data = {"pi": 3.141592653589793, "e": 2.718281828459045}
    result = json.dumps(data, cls=CustomEncoder, precision=3)
    decoded = json.loads(result)
    assert decoded["pi"] == 3.142
    assert decoded["e"] == 2.718


def test_precision_nested():
    """Test precision reduction with nested structures"""
    data = {"outer": {"inner": 3.141592653589793}, "list": [1.23456, 2.34567]}
    result = json.dumps(data, cls=CustomEncoder, precision=2)
    decoded = json.loads(result)
    assert decoded["outer"]["inner"] == 3.14
    assert decoded["list"] == [1.23, 2.35]


def test_precision_tuple():
    """Test precision reduction preserves tuple->list"""
    data = {"t": (1.23456, 2.34567)}
    result = json.dumps(data, cls=CustomEncoder, precision=2)
    decoded = json.loads(result)
    assert decoded["t"] == [1.23, 2.35]


def test_decode_datetime():
    """Test decoding datetime objects"""
    encoded = '{"__datetime__": true, "datetime": "2024-06-15T12:00:00.000Z"}'
    result = json.loads(encoded, object_hook=custom_decoder)
    assert isinstance(result, datetime)
    assert result.year == 2024


def test_decode_dataframe():
    """Test decoding DataFrame objects"""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    encoded = json.dumps({"df": df}, cls=CustomEncoder)
    decoded = json.loads(encoded, object_hook=custom_decoder)
    assert isinstance(decoded["df"], pd.DataFrame)
    assert list(decoded["df"].columns) == ["A", "B"]


def test_decode_passthrough():
    """Test that normal dicts pass through decoder"""
    data = {"a": 1, "b": "hello"}
    encoded = json.dumps(data)
    decoded = json.loads(encoded, object_hook=custom_decoder)
    assert decoded == data


def test_roundtrip_complex():
    """Test full round-trip with mixed types"""
    original = {
        "int": 42,
        "float": 3.14,
        "np_int": np.int64(99),
        "np_float": np.float64(2.71),
        "np_bool": np.bool_(False),
        "dt": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    encoded = json.dumps(original, cls=CustomEncoder)
    decoded = json.loads(encoded, object_hook=custom_decoder)
    assert decoded["int"] == 42
    assert abs(decoded["float"] - 3.14) < 0.001
    assert decoded["np_int"] == 99
    assert isinstance(decoded["dt"], datetime)


if __name__ == "__main__":
    test_encode_numpy_int()
    test_roundtrip_complex()
    print("All json_utils tests passed!")
