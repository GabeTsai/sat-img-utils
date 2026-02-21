import os
import geopandas as gpd
from sat_img_utils.core.utils import make_dirs_if_not_exists
import logging
from pathlib import Path
from sat_img_utils.geo.raster import clean_gdf
from sat_img_utils.configs.constants import CRS

def split_aoi(aoi: gpd.GeoDataFrame, 
              out_dir: str, 
              out_crs: CRS = CRS.WEB_MERCATOR,
              prefix: str = "aoi"):
    """
    Split an AOI into parts and save each part as a separate GeoJSON file.
    Generally used for AOIs that are MultiPolygon geometries.
    """
    make_dirs_if_not_exists(out_dir)
    crs = int(out_crs)
    aoi_crs = aoi.to_crs(crs)
    for i in range(len(aoi_crs)):
        out_path = os.path.join(out_dir, f"{prefix}_{i:02d}_{crs}.geojson")
        gpd.GeoDataFrame({"id":[i]}, geometry=[aoi_crs.geometry.iloc[i]], crs=crs).to_file(
            out_path, driver="GeoJSON"
        )
    logging.info(f"Wrote {len(aoi_crs)} AOI parts to {out_dir}/")

def explode_aoi_to_files(
    in_geojson: str,
    out_dir: str,
    out_crs: CRS = CRS.WEB_MERCATOR,
    prefix="aoi",
):
    """
    Explode an AOI into parts and save each part as a separate GeoJSON file.
    Generally used for AOIs that are MultiPolygon geometries.

    Example usage: 
    explode_aoi_to_files(
        in_geojson="aoi_geojsons/osm_aoi_capella_def.geojson",
        out_dir="aoi_parts_3857_capella_def",
        out_crs=CRS.WEB_MERCATOR,
        prefix="aoi",
    )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(in_geojson)
    
    crs_int = int(out_crs)
    if out_crs is not None:
        gdf = gdf.to_crs(crs_int)

    parts = gdf.explode(index_parts=False, ignore_index=True)

    parts = clean_gdf(parts)

    for i, row in parts.iterrows():
        one = gpd.GeoDataFrame([row], crs=parts.crs)
        out_path = os.path.join(out_dir, f"{prefix}_{i:02d}_{crs_int}.geojson")
        one.to_file(out_path, driver="GeoJSON")

    logging.info(f"Wrote {len(parts)} AOI parts -> {out_dir}")