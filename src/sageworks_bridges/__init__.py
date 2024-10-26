# Copyright (c) 2021-2024 SuperCowPowers LLC

"""
SageWorks Bridges
- TBD
  - TBD1
  - TBD2
"""
from importlib.metadata import version

try:
    __version__ = version("sageworks_bridges")
except Exception:
    __version__ = "unknown"

# SageWorks Bridges Logging
from sageworks_bridges.utils.logger import logging_setup

logging_setup()
