# Copyright (c) 2021-2024 SuperCowPowers LLC

"""
Workbench Bridges
- TBD
  - TBD1
  - TBD2
"""
from importlib.metadata import version

try:
    __version__ = version("workbench_bridges")
except Exception:
    __version__ = "unknown"

# Workbench Bridges Logging
from workbench_bridges.utils.logger import logging_setup

logging_setup()
