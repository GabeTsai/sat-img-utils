import gc
import logging
from typing import Tuple
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
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