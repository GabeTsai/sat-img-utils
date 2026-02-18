"""Core utility functions for monitoring and diagnostics."""
import numpy as np
import psutil
import os
from pathlib import Path

def get_sat_tile_memory(height, width, dtype: np.dtype, n_channels, pow = 3) -> float:
    """
    Estimate the memory size of a numpy array.
    
    Args:
        height: Height of the satellite tile in pixels
        width: Width of the satellite tile in pixels
        dtype: Data type of the array (e.g., np.uint16)
        n_channels: Number of channels in the satellite tile
        pow: Power of 1024 to convert bytes to desired unit (default: 3 for GB)
    Returns:
        float: Estimated size in bytes
    """
    return (height * width * np.dtype(dtype).itemsize * n_channels) / (1024 ** pow)

def get_memory_mb() -> float:
    """
    Get the current memory usage of the process in megabytes.
    
    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def make_dirs_if_not_exists(dir_path: str) -> None:
    """
    Create directories if they do not already exist.
    
    Args:
        dir_path: Path to the directory to create
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)