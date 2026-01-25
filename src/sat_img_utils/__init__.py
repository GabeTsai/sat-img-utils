"""Satellite image utilities for processing and analysis."""

__version__ = "0.1.0"

# Import submodules to make them available at package level
from . import configs
from . import core
from . import datasets
from . import geo
from . import pipelines

__all__ = [
    "configs",
    "core",
    "datasets",
    "geo",
    "pipelines",
]
