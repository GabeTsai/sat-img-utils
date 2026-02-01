"""Configuration constants for satellite image processing."""
from enum import Enum

from .constants import (
    MAX_WINDOW_SIZE_GB,
    CHUNK_HEIGHT,
)

from .ds_constants import *

__all__ = [
    "BUILDINGS_THRESHOLD",
    "MAX_WINDOW_SIZE_GB",
    "CHUNK_HEIGHT",
]
