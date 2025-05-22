import botocore
import logging
import sys
import time
import functools
import json
import base64
import re
import os
from typing import Union, List, Callable, Optional
import pandas as pd
import awswrangler as wr
from awswrangler.exceptions import NoFilesFound
from pathlib import Path
import posixpath
from botocore.exceptions import ClientError
from sagemaker.session import Session as SageSession
from collections.abc import Mapping, Iterable


# Workbench-Bridges Logger
log = logging.getLogger("workbench-bridges")


def not_found_returns_none(func: Optional[Callable] = None, *, resource_name: str = "AWS resource") -> Callable:
    """Decorator to handle AWS resource not found (returns None) and re-raising otherwise.

    Args:
        func (Callable, optional): The function being decorated.
        resource_name (str): Name of the AWS resource being accessed. Used for clearer error messages.
    """
    not_found_errors = {
        "ResourceNotFound",
        "ResourceNotFoundException",
        "EntityNotFoundException",
        "ValidationException",
        "NoSuchBucket",
    }

    def decorator(inner_func: Callable) -> Callable:
        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs):
            try:
                return inner_func(*args, **kwargs)
            except ClientError as error:
                error_code = error.response["Error"]["Code"]
                if error_code in not_found_errors:
                    log.warning(f"{resource_name} not found: {error_code}, returning None...")
                    return None
                else:
                    log.critical(f"Critical error in AWS call: {error_code}")
                    raise
            except wr.exceptions.NoFilesFound:
                log.info(f"Resource {resource_name} not found returning None...")
                return None

        return wrapper

    # If func is None, the decorator was called with arguments
    if func is None:
        return decorator
    else:
        # If func is not None, the decorator was used without arguments
        return decorator(func)


if __name__ == "__main__":
    """Exercise the AWS Utils"""

    @not_found_returns_none
    def test_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    test_not_found()

    @not_found_returns_none(resource_name="my_not_found_resource")
    def test_not_found():
        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "test")

    test_not_found()

    try:

        @not_found_returns_none
        def test_other_error():
            # Raise a different error to test the error handler
            raise ClientError({"Error": {"Code": "SomeOtherError"}}, "test")

        test_other_error()
    except ClientError:
        print("AOK Expected Error :)")
