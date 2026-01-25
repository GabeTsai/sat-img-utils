"""Raster data reading and reprojection utilities."""

import logging
import gc
from typing import Tuple
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
import numpy as np

from sat_img_utils.configs import constants
from sat_img_utils.core import get_memory_mb

def estimate_window_size_gb(window: Window, dtype_bytes: int = 2) -> float:
    """
    Estimate the memory size of a raster window in gigabytes.
    
    Args:
        window: Rasterio window object
        dtype_bytes: Number of bytes per pixel (default: 2 for uint16)
    
    Returns:
        float: Estimated size in GB
    """
    return (window.width * window.height * dtype_bytes) / (1024 * 1024 * 1024)


def read_raster_window_chunked(
    raster: rasterio.DatasetReader,
    window: Window,
    band: int = 1,
    max_size_gb: float = None,
    chunk_height: int = None
) -> np.ndarray:
    """
    Read a raster window in chunks to avoid memory issues with large windows.
    
    Args:
        raster: Open rasterio dataset reader
        window: Window to read from the raster
        band: Band number to read (default: 1)
        max_size_gb: Maximum window size in GB before chunking (default: from constants)
        chunk_height: Height of each chunk in rows (default: from constants)
    
    Returns:
        np.ndarray: Array containing the windowed raster data
    """
    if max_size_gb is None:
        max_size_gb = constants.MAX_WINDOW_SIZE_GB
    if chunk_height is None:
        chunk_height = constants.CHUNK_HEIGHT
    
    window_size_gb = estimate_window_size_gb(window)
    logging.info(f"Estimated window size: {window_size_gb:.2f}GB")
    
    if window_size_gb > max_size_gb:
        logging.info("Large window detected - reading in chunks")
        chunks = []
        
        for row_start in range(0, window.height, chunk_height):
            chunk_window = Window(
                window.col_off, 
                window.row_off + row_start,
                window.width,
                min(chunk_height, window.height - row_start)
            )
            
            logging.info(
                f"Reading chunk at row {row_start}/{window.height} - "
                f"Memory: {get_memory_mb():.0f}MB"
            )
            chunk = raster.read(band, window=chunk_window)
            chunks.append(chunk)
        
        result = np.vstack(chunks)
        del chunks
        gc.collect()
        return result
    else:
        return raster.read(band, window=window)

def reproject_raster_to_match(
    source: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    dtype: np.dtype = np.uint16,
    resampling: Resampling = Resampling.nearest
) -> np.ndarray:
    """
    Reproject a raster array to match the geometry of another raster.
    Crash course on reprojection: 
    Rasterio creates a new grid in the new CRS. 
    For every pixel in the new grid, will calculate:
    1. Where the center of that pixel is on Earth
    2. Where that location maps to in the source CRS
    3. Which pixels cover that location
    4. How to interpolate those pixels to get a value for the new pixel
    
    Args:
        source: Source array to reproject
        src_transform: Affine transform of the source array
        src_crs: CRS of the source array
        dst_shape: Shape (height, width) of the destination array
        dst_transform: Affine transform of the destination array
        dst_crs: CRS of the destination array
        dtype: Data type for the destination array
        resampling: Resampling method to use
    
    Returns:
        np.ndarray: Reprojected array matching destination geometry
    """
    destination = np.zeros(dst_shape, dtype=dtype)
    
    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )
    
    logging.info(f"After reprojection - Memory: {get_memory_mb():.0f}MB")
    return destination

