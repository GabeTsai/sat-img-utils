import gc
import logging
from typing import Tuple
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling, transform_bounds

from sat_img_utils.configs import ds_constants
from sat_img_utils.core import get_memory_mb
from sat_img_utils.core.filters import calculate_threshold_fraction
from sat_img_utils.geo import read_raster_window_chunked, reproject_raster_to_match

def detect_buildings(
    ghsl: rasterio.DatasetReader,
    sar: rasterio.DatasetReader,
    filter_value: float = None
) -> float:
    """
    Detect buildings in a SAR image using GHSL (Global Human Settlement Layer) data.
    Reads GHSL building density data, reprojects it to match the SAR
    image geometry, and calculates the fraction of pixels indicating buildings.
    
    Args:
        ghsl: Open GHSL raster dataset
        sar: Open SAR raster dataset to match geometry
        filter_value: Building detection threshold (default: from datasets.GHSL_BUILDINGS_THRESHOLD)
    
    Returns:
        float: Fraction of pixels indicating buildings (0.0 to 1.0)
    """
    if filter_value is None:
        filter_value = ds_constants.GHSL_BUILDINGS_THRESHOLD
    
    wgs84_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)
    logging.info(f"Starting GHSL processing - Memory: {get_memory_mb():.0f}MB")
    
    window = from_bounds(*wgs84_bounds, ghsl.transform)
    logging.info(f"After GHSL read - Memory: {get_memory_mb():.0f}MB")
    
    ghsl_subset = read_raster_window_chunked(ghsl, window)
    
    # Reproject GHSL to match SAR geometry
    ghsl_resampled = reproject_raster_to_match(
        source=ghsl_subset,
        src_transform=ghsl.window_transform(window),
        src_crs=ghsl.crs,
        dst_shape=(sar.height, sar.width),
        dst_transform=sar.transform,
        dst_crs=sar.crs,
        dtype=np.uint16,
        resampling=Resampling.nearest
    )
    
    del ghsl_subset
    gc.collect()
    
    building_fraction = calculate_threshold_fraction(
        data=ghsl_resampled,
        filter_value=filter_value,
        nodata=ghsl.nodata,
        greater=True,
        strict=True
    )
    
    del ghsl_resampled
    gc.collect()
    
    return building_fraction

def iter_windows(width: int, height: int, block: int):
    for row_off in range(0, height, block):
        h = min(block, height - row_off)
        for col_off in range(0, width, block):
            w = min(block, width - col_off)
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=w,
                height=h
            )

def detect_buildings_chunked(
    ghsl: rasterio.DatasetReader,
    sar: rasterio.DatasetReader,
    filter_value: float = None,
    block: int = 4096,
) -> float:
    """
    Memory-safe GHSL building fraction computation:
    - read GHSL subset covering SAR bounds (chunked)
    - reproject into SAR grid in small windows
    - accumulate thresholded counts, never allocate full (H,W)
    """
    if filter_value is None:
        filter_value = ds_constants.GHSL_BUILDINGS_THRESHOLD

    # Bounds of SAR in GHSL CRS
    ghsl_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)
    logging.info(f"Starting GHSL processing - Memory: {get_memory_mb():.0f}MB")

    # Window in GHSL that covers SAR footprint
    ghsl_window = from_bounds(*ghsl_bounds, transform=ghsl.transform)

    # Read GHSL subset (your chunked reader)
    ghsl_subset = read_raster_window_chunked(ghsl, ghsl_window)
    src_transform = ghsl.window_transform(ghsl_window)

    logging.info(
        f"GHSL subset shape={ghsl_subset.shape}, dtype={ghsl_subset.dtype}, "
        f"Memory: {get_memory_mb():.0f}MB"
    )

    # Accumulators
    total = 0
    hits = 0

    # Reproject in SAR windows
    for win in iter_windows(sar.width, sar.height, block):
        dst = np.zeros((int(win.height), int(win.width)), dtype=np.uint16)

        reproject(
            source=ghsl_subset,
            destination=dst,
            src_transform=src_transform,
            src_crs=ghsl.crs,
            dst_transform=window_transform(win, sar.transform),
            dst_crs=sar.crs,
            resampling=Resampling.nearest,
        )

        # Valid mask: if GHSL has nodata defined, exclude it
        if ghsl.nodata is not None:
            valid = (dst != ghsl.nodata)
            total += int(valid.sum())
            hits += int(((dst > filter_value) & valid).sum())
        else:
            total += dst.size
            hits += int((dst > filter_value).sum())

        # free window buffer promptly
        del dst
        
    del ghsl_subset
    gc.collect()

    return 0.0 if total == 0 else hits / total