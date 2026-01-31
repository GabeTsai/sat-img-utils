"""Raster data reading and reprojection utilities."""

import logging
import gc
from typing import Tuple
from pyproj import Transformer
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds, transform
import numpy as np
from shapely.geometry import box

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

def window_center_longlat(
    ds: rasterio.io.DatasetReader,
    window: Window,
    out_epsg: int = 4326,
) -> Tuple[float, float]:
    """
    Returns (long, lat) of the window center. Uses a cached Transformer for performance.
    """
    center_row = window.row_off + window.height / 2.0
    center_col = window.col_off + window.width / 2.0

    x_center, y_center = rasterio.transform.xy(ds.transform, center_row, center_col, offset="center")

    if ds.crs is None or ds.crs.to_epsg() == out_epsg:
        return float(x_center), float(y_center)

    transformer = Transformer.from_crs(ds.crs, f"EPSG:{out_epsg}", always_xy=True)
    long, lat = transformer.transform(x_center, y_center)
    return float(long), float(lat)

def drop_null_empty_invalid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:   
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.is_valid]
    return gdf

def clean_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply zero-width buffer to clean invalid geometries in a GeoDataFrame.
    """
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return drop_null_empty_invalid(gdf)

def rasterize_gdf_to_mask(gdf, sat_tile):
    """
    Rasterize binary polygon gdf for specific satellite image tile

    Args:
      gdf: GeoDataFrame of land polygons
      sat_tile: rasterio dataset of satellite tile
    Returns:
      mask: 2D numpy uint8 array mask where land = 1, water = 0
    """

    # Reproject land polygons to satellite CRS
    gdf_in_sat_crs = gdf.to_crs(sat_tile.crs)

    # Drop null/empty/invalid after reprojection
    gdf_in_sat_crs = drop_null_empty_invalid(gdf_in_sat_crs)
    # Clip to satellite image tile bounds (in satellite image CRS)
    b = sat_tile.bounds
    tile_geom = box(b.left, b.bottom, b.right, b.top)
    gdf_clip = gdf_in_sat_crs[gdf_in_sat_crs.intersects(tile_geom)]
    
    # No land intersecting tile
    if gdf_clip.empty:
        return np.zeros((sat_tile.height, sat_tile.width), dtype=np.uint8)

    gdf_clip = gdf_clip.intersection(tile_geom)

    gdf_clip = drop_null_empty_invalid(gdf_clip)

    land_mask = rasterize(
        ((geom, 1) for geom in gdf_clip.geometry),
        out_shape=(sat_tile.height, sat_tile.width),
        transform=sat_tile.transform,
        fill=0,
        dtype="uint8"
    )

    return land_mask
