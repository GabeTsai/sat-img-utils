"""Core utility functions for monitoring and diagnostics."""

import psutil
import os


def get_memory_mb() -> float:
    """
    Get the current memory usage of the process in megabytes.
    
    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
