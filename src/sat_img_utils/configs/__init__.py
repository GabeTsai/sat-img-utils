"""Configuration constants for satellite image processing."""
from enum import Enum

from .constants import (
    BUILDINGS_THRESHOLD,
    MAX_WINDOW_SIZE_GB,
    CHUNK_HEIGHT,
)

__all__ = [
    "BUILDINGS_THRESHOLD",
    "MAX_WINDOW_SIZE_GB",
    "CHUNK_HEIGHT",
]
