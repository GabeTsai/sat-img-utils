import logging
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
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
          land = 1, water = 0, uncovered = 255
        Rasters in the VRT must use nodata=255 (not 0) so that water (0)
        is not misinterpreted as missing data by rasterio's reproject.
        """
        MASK_NODATA = 255
        dst = np.full((sat_tile.height, sat_tile.width), MASK_NODATA, dtype=np.uint8)

        reproject(
            source=rasterio.band(self.ds, 1),
            destination=dst,
            src_transform=self.ds.transform,
            src_crs=self.ds.crs,
            dst_transform=sat_tile.transform,
            dst_crs=sat_tile.crs,
            dst_nodata=MASK_NODATA,
            resampling=Resampling.nearest,
        )

        if (dst == MASK_NODATA).all():
            # VRT has no coverage. Read the window directly in the VRT's own CRS
            # to distinguish a CRS/transform bug from genuinely missing tile data.
            sat_in_vrt_crs = transform_bounds(sat_tile.crs, self.ds.crs, *sat_tile.bounds)
            vrt_win = window_from_bounds(*sat_in_vrt_crs, transform=self.ds.transform)
            try:
                direct = self.ds.read(1, window=vrt_win, out_shape=(64, 64),
                                      boundless=True, fill_value=MASK_NODATA)
                logging.warning(
                    f"VRT has no reprojected coverage for {sat_tile.name}. "
                    f"Direct VRT read at SAR bounds (in VRT CRS {self.ds.crs} = {sat_in_vrt_crs}) "
                    f"gives unique values: {np.unique(direct).tolist()}. "
                )
            except Exception as e:
                logging.warning(f"VRT direct read also failed for {sat_tile.name}: {e}")
        return dst

def osm_rasterize_sat_land_mask(land_global: gpd.GeoDataFrame, sat_tile: rasterio.io.DatasetReader) -> np.ndarray:
    land_global = drop_null_empty_invalid(land_global)
    land_global = clean_gdf(land_global)
    land_mask = rasterize_gdf_to_mask(land_global, sat_tile=sat_tile)
    return land_mask
