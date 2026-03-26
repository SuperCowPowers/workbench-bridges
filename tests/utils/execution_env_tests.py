"""Tests for execution_environment"""

import os

from workbench_bridges.utils.execution_environment import (
    running_on_glue,
    running_on_lambda,
    running_on_ecs,
    running_on_docker,
    running_as_service,
    glue_job_name,
)


def test_running_on_glue_false():
    """Test that glue detection returns False when not on Glue"""
    # Make sure env vars are not set
    os.environ.pop("GLUE_VERSION", None)
    os.environ.pop("GLUE_PYTHON_VERSION", None)
    assert running_on_glue() is False


def test_running_on_glue_true():
    """Test glue detection with GLUE_VERSION set"""
    os.environ["GLUE_VERSION"] = "4.0"
    assert running_on_glue() is True
    del os.environ["GLUE_VERSION"]


def test_running_on_glue_python_version():
    """Test glue detection with GLUE_PYTHON_VERSION set"""
    os.environ["GLUE_PYTHON_VERSION"] = "3"
    assert running_on_glue() is True
    del os.environ["GLUE_PYTHON_VERSION"]


def test_running_on_lambda_false():
    """Test that lambda detection returns False locally"""
    os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
    assert running_on_lambda() is False


def test_running_on_lambda_true():
    """Test lambda detection with env var set"""
    os.environ["AWS_LAMBDA_FUNCTION_NAME"] = "my_func"
    assert running_on_lambda() is True
    del os.environ["AWS_LAMBDA_FUNCTION_NAME"]


def test_running_on_ecs_false():
    """Test that ECS detection returns False locally"""
    for key in ["ECS_SERVICE_NAME", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_EXECUTION_ENV"]:
        os.environ.pop(key, None)
    assert running_on_ecs() is False


def test_running_on_ecs_true():
    """Test ECS detection with indicator set"""
    os.environ["ECS_CONTAINER_METADATA_URI"] = "http://169.254.170.2/v3"
    assert running_on_ecs() is True
    del os.environ["ECS_CONTAINER_METADATA_URI"]


def test_running_on_docker_false():
    """Test that docker detection returns False on dev machine"""
    # Clean up ECS env vars that would trigger docker detection
    for key in ["ECS_SERVICE_NAME", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_EXECUTION_ENV"]:
        os.environ.pop(key, None)
    assert running_on_docker() is False


def test_running_as_service_false():
    """Test that service detection returns False locally"""
    for key in ["GLUE_VERSION", "GLUE_PYTHON_VERSION", "AWS_LAMBDA_FUNCTION_NAME",
                "ECS_SERVICE_NAME", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_EXECUTION_ENV"]:
        os.environ.pop(key, None)
    assert running_as_service() is False


def test_running_as_service_glue():
    """Test that service detection returns True when on Glue"""
    os.environ["GLUE_VERSION"] = "4.0"
    assert running_as_service() is True
    del os.environ["GLUE_VERSION"]


def test_running_as_service_lambda():
    """Test that service detection returns True when on Lambda"""
    os.environ["AWS_LAMBDA_FUNCTION_NAME"] = "my_func"
    assert running_as_service() is True
    del os.environ["AWS_LAMBDA_FUNCTION_NAME"]


def test_glue_job_name_default():
    """Test default glue job name when not running on Glue"""
    name = glue_job_name()
    assert isinstance(name, str)


if __name__ == "__main__":
    test_running_on_glue_false()
    test_running_as_service_false()
    print("All execution_environment tests passed!")
