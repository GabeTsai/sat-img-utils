from typing import Any, Callable, Dict, Optional, Sequence
import rasterio
import geopandas as gpd
from pathlib import Path
from sat_img_utils.geo.raster import get_gdf
from sat_img_utils.datasets.osm_land_poly import osm_rasterize_sat_land_mask
from sat_img_utils.datasets.capella import process_capella_sar_tile
# Usage example:
# cfg = PatchIterPipelineConfig(
    #     patch_size=512,
    #     patch_dtype=ds.dtypes[0],
    #     out_dir=out_dir,
    #     img_name=Path(img_path).stem,
    #     nodata=0,
    #     pad_value=0,
    #     gc_every=1000,
    #     bands=1,  # single band SAR
    # )
    
    # extra_ctx = {
    #     "threshold_fraction_filter_eq": { 
    #         "filter_value": 0.0,
    #         "nodata": None, 
    #     "upper": True,
    #         "fraction_value": 0.75
    #     }, 
    #     "min_land_fraction_filter_random": {
    #         "land_mask": land_mask,
    #         "min_land_threshold": 0.1,
    #         "discard_prob": 0.9,
    #     } 
    # }

    # cut_patches(
    #     ds,
    #     img_name=cfg.img_name,
    #     out_dir=out_dir,
    #     cfg=cfg,
    #     transform=capella_sar_transform_to_uint8,
    #     filters_before_transform=[
    #         min_land_fraction_filter_random,
    #         threshold_fraction_filter_eq,
    #     ],
    #     metadata_fn = default_metadata_fn,
    #     writer_fn=save_capella_patch,
    #     extra_ctx=extra_ctx,
    # )

def test_patches():
    img_path = "../../test_data/CAPELLA_C03_SP_GEO_HH_20211231164052_20211231164115.tif"
    osm_land_polygon_path = "../../test_data/land_polygons.shp"
    out_dir = "./test_output/sar_patches_3"
    ds = rasterio.open(img_path)
    land_global = get_gdf(osm_land_polygon_path)
    land_mask = osm_rasterize_sat_land_mask(land_global, ds)

    metadata = process_capella_sar_tile(
        ds=ds,
        out_dir=out_dir,
        land_mask=land_mask,
        patch_size=512,
        nodata=0,
    )

if __name__ == "__main__":
    test_patches()