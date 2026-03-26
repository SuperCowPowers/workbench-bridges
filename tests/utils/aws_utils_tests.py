"""Tests for aws_utils"""

from botocore.exceptions import ClientError

from workbench_bridges.utils.aws_utils import not_found_returns_none


def test_not_found_returns_none_basic():
    """Test decorator returns None for ResourceNotFoundException"""

    @not_found_returns_none
    def raises_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    assert raises_not_found() is None


def test_not_found_returns_none_with_name():
    """Test decorator with resource_name argument"""

    @not_found_returns_none(resource_name="my_resource")
    def raises_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    assert raises_not_found() is None


def test_entity_not_found():
    """Test decorator catches EntityNotFoundException"""

    @not_found_returns_none
    def raises_entity_not_found():
        raise ClientError({"Error": {"Code": "EntityNotFoundException"}}, "test")

    assert raises_entity_not_found() is None


def test_validation_exception():
    """Test decorator catches ValidationException"""

    @not_found_returns_none
    def raises_validation():
        raise ClientError({"Error": {"Code": "ValidationException"}}, "test")

    assert raises_validation() is None


def test_no_such_bucket():
    """Test decorator catches NoSuchBucket"""

    @not_found_returns_none
    def raises_no_bucket():
        raise ClientError({"Error": {"Code": "NoSuchBucket"}}, "test")

    assert raises_no_bucket() is None


def test_other_error_reraises():
    """Test decorator re-raises non-not-found errors"""

    @not_found_returns_none
    def raises_other():
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "test")

    try:
        raises_other()
        assert False, "Should have raised"
    except ClientError:
        pass  # Expected


def test_no_files_found():
    """Test decorator catches awswrangler NoFilesFound"""
    import awswrangler as wr

    @not_found_returns_none
    def raises_no_files():
        raise wr.exceptions.NoFilesFound("No files")

    assert raises_no_files() is None


def test_success_passthrough():
    """Test decorator passes through successful results"""

    @not_found_returns_none
    def returns_value():
        return "hello"

    assert returns_value() == "hello"


def test_success_passthrough_with_name():
    """Test decorator passes through successful results with resource_name"""

    @not_found_returns_none(resource_name="test")
    def returns_value():
        return 42

    assert returns_value() == 42


if __name__ == "__main__":
    test_not_found_returns_none_basic()
    test_other_error_reraises()
    print("All aws_utils tests passed!")
