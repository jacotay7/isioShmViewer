"""
isioshmviewer - A Python library for interacting with ImageStreamIO shared memory streams

This package provides tools for connecting to, monitoring, and processing data
from shared memory streams used in optical systems and other real-time applications.
"""

from .shm_stream import SHMStream
from .isio_wrapper import ImageStreamIO

__version__ = '0.1.0'