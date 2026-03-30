"""Tests for glue_utils"""

from workbench_bridges.utils.glue_utils import get_resolved_options


def test_basic_parsing():
    """Test basic key-value argument parsing"""
    args = ["script.py", "--name", "value"]
    result = get_resolved_options(args)
    assert result["name"] == "value"


def test_multiple_args():
    """Test parsing multiple arguments"""
    args = ["script.py", "--key1", "val1", "--key2", "val2"]
    result = get_resolved_options(args)
    assert result["key1"] == "val1"
    assert result["key2"] == "val2"


def test_auto_discover_options():
    """Test that options are auto-discovered when None"""
    args = ["script.py", "--s3path", "s3://bucket/key", "--JOB_NAME", "my_job"]
    result = get_resolved_options(args)
    assert result["s3path"] == "s3://bucket/key"
    assert result["JOB_NAME"] == "my_job"


def test_specified_options_filter():
    """Test that specifying options filters results"""
    args = ["script.py", "--keep", "yes", "--drop", "no"]
    result = get_resolved_options(args, options=["keep"])
    assert result["keep"] == "yes"
    assert "drop" not in result


def test_flag_without_value():
    """Test flags that have no value (followed by another flag)"""
    args = ["script.py", "--flag", "--other", "val"]
    result = get_resolved_options(args)
    assert result["flag"] == ""
    assert result["other"] == "val"


def test_positional_args_skipped():
    """Test that positional (non-flag) arguments are skipped"""
    args = ["/tmp/script.py", "true", "--key", "val", "extra"]
    result = get_resolved_options(args)
    assert result["key"] == "val"
    assert "true" not in result


def test_glue_style_args():
    """Test with realistic Glue job arguments"""
    args = [
        "/tmp/dispatch_test.py",
        "true",
        "--s3path",
        "s3://blah/foo.csv",
        "--job-bookmark-option",
        "job-bookmark-disable",
        "--JOB_ID",
        "j_a123",
        "true",
        "--JOB_RUN_ID",
        "jr_z456",
        "--JOB_NAME",
        "dispatch_test",
    ]
    result = get_resolved_options(args)
    assert result["s3path"] == "s3://blah/foo.csv"
    assert result["JOB_ID"] == "j_a123"
    assert result["JOB_NAME"] == "dispatch_test"


def test_empty_argv():
    """Test with empty arguments"""
    result = get_resolved_options([])
    assert result == {}


def test_only_positional():
    """Test with no flags at all"""
    result = get_resolved_options(["script.py", "arg1", "arg2"])
    assert result == {}


if __name__ == "__main__":
    test_glue_style_args()
    print("All glue_utils tests passed!")
