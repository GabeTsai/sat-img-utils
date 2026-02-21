import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from sat_img_utils.geo.raster import drop_null_empty_invalid, clean_gdf, rasterize_gdf_to_mask

class LandMaskVRT:
    """
    VRT file containing a single band of the land mask.
    Allows for efficient rasterization of the land mask to a satellite tile.
    """
    def __init__(self, vrt_path: str):
        self.ds = rasterio.open(vrt_path)

    def close(self):
        self.ds.close()

    def get_mask_for_tile(self, sat_tile: rasterio.io.DatasetReader) -> np.ndarray:
        """
        Returns (H, W) uint8 mask aligned to sat_tile:
          land = 1, water/unknown = 0
        """
        dst = np.zeros((sat_tile.height, sat_tile.width), dtype=np.uint8)

        reproject(
            source=rasterio.band(self.ds, 1),
            destination=dst,
            src_transform=self.ds.transform,
            src_crs=self.ds.crs,
            dst_transform=sat_tile.transform,
            dst_crs=sat_tile.crs,
            dst_nodata=255,
            resampling=Resampling.nearest,
        )
        return dst

def osm_rasterize_sat_land_mask(land_global: gpd.GeoDataFrame, sat_tile: rasterio.io.DatasetReader) -> np.ndarray:
    land_global = drop_null_empty_invalid(land_global)
    land_global = clean_gdf(land_global)
    land_mask = rasterize_gdf_to_mask(land_global, sat_tile=sat_tile)
    return land_mask
