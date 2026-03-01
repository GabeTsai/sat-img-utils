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
from shapely.ops import unary_union

from sat_img_utils.configs import constants
from sat_img_utils.core import get_memory_mb
from typing import Union, Sequence

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

def choose_overview_level(ds: rasterio.DatasetReader, target_w: int) -> int:
    """
    Return the index into ds.overviews(1) whose width is <= target_w,
    or None if the full-resolution image is already within target_w.
    """
    base_width = ds.width
    if base_width <= target_w:
        return None
    overviews = ds.overviews(1)
    if not overviews:
        return None
    for i, factor in enumerate(overviews):
        if base_width // factor <= target_w:
            return i
    return len(overviews) - 1

def get_overview(
    ds: rasterio.DatasetReader,
    bands: Union[int, Sequence[int]],
    overview_level: int = None,
) -> np.ndarray:
    """
    Read a raster at the given overview level.
    
    If overview_level is None, returns the full-resolution data.
    """
    if overview_level is None:
        return ds.read(bands)
    ovr_factors = ds.overviews(1)
    factor = ovr_factors[overview_level]
    out_h = max(1, ds.height // factor)
    out_w = max(1, ds.width // factor)
    if isinstance(bands, int):
        return ds.read(bands, out_shape=(out_h, out_w))
    return ds.read(list(bands), out_shape=(len(bands), out_h, out_w))

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

def get_gdf(gdf_path: str) -> gpd.GeoDataFrame:
    """
    Read a GeoDataFrame from a file and clean it by dropping null, empty, and invalid geometries.
    
    Args:
        gdf_path: Path to the GeoDataFrame file (e.g., shapefile, GeoJSON)
    """
    gdf = gpd.read_file(gdf_path)
    gdf = drop_null_empty_invalid(gdf)
    gdf = clean_gdf(gdf)
    return gdf

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

    # Keep this conservative: we only use sindex to get candidate rows by bbox,
    # then still run your exact intersects() filter afterward.
    gdf_prefilter = gdf_in_sat_crs
    try:
        sidx = gdf_in_sat_crs.sindex  # may build lazily
        cand_idx = list(sidx.intersection(tile_geom.bounds))
        if len(cand_idx) == 0:
            # No candidates intersect bounding boxes => definitely no geometry intersects tile
            return np.zeros((sat_tile.height, sat_tile.width), dtype=np.uint8)
        gdf_prefilter = gdf_in_sat_crs.iloc[cand_idx]
    except Exception:
        # If spatial index is unavailable for any reason, fall back to original behavior.
        gdf_prefilter = gdf_in_sat_crs

    gdf_clip = gdf_prefilter[gdf_prefilter.intersects(tile_geom)]

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

def get_aoi_from_bboxes(
    bboxes: list,
    reproj_crs: int = 4326,
    buffer_m: float = 2000,
) -> gpd.GeoSeries:
    """
    Given a list of bounding boxes (shapely geometries in reproj_crs),
    return a unified AOI GeoSeries, buffered in meters safely.
    """
    aoi_geom = unary_union(bboxes)
    if hasattr(aoi_geom, "buffer"):
        aoi_geom = aoi_geom.buffer(0)

    if aoi_geom.is_empty or (hasattr(aoi_geom, "is_valid") and not aoi_geom.is_valid):
        raise ValueError("AOI geometry is empty or invalid after union. Check input bounding boxes.")

    aoi = gpd.GeoSeries([aoi_geom], crs=reproj_crs)

    if buffer_m > 0:
        meter_crs = "EPSG:3857"
        aoi_meter = aoi.to_crs(meter_crs)
        aoi_meter = aoi_meter.buffer(buffer_m)
        aoi_meter = aoi_meter.buffer(0)
        if aoi_meter.is_empty.any():
            raise ValueError("AOI geometry became empty after buffering in EPSG:3857.")
        aoi = aoi_meter.to_crs(reproj_crs)

    return aoi

def convert_bbox_crs(bbox: box, src_crs: int, dst_crs: int):
    """
    Convert a bounding box from source CRS to destination CRS.

    Args:
      bbox: shapely box geometry in source CRS
      src_crs: EPSG code of source CRS
      dst_crs: EPSG code of destination CRS
    Returns:
      converted_bbox: shapely box geometry in destination CRS
    """
    bbox_wgs84 = gpd.GeoSeries([bbox], crs=src_crs).to_crs(dst_crs).iloc[0]
    return bbox_wgs84
