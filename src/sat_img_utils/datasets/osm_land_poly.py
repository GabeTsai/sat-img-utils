import numpy as np
import geopandas as gpd
import rasterio
from sat_img_utils.geo.raster import drop_null_empty_invalid, clean_gdf, rasterize_gdf_to_mask

def osm_rasterize_sat_land_mask(land_global: gpd.GeoDataFrame, sat_tile: rasterio.io.DatasetReader) -> np.ndarray:
    land_global = drop_null_empty_invalid(land_global)
    land_global = clean_gdf(land_global)
    land_mask = rasterize_gdf_to_mask(land_global, sat_tile=sat_tile)
    return land_mask
