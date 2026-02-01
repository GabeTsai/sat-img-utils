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

def make_dirs_if_not_exists(dir_path: str) -> None:
    """
    Create directories if they do not already exist.
    
    Args:
        dir_path: Path to the directory to create
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)