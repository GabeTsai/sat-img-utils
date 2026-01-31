from typing import Any, Callable, Dict, Optional, Sequence
import rasterio
import geopandas as gpd
from pathlib import Path

from sat_img_utils.pipelines.context import Context
from sat_img_utils.pipelines.config import PatchIterPipelineConfig
from sat_img_utils.pipelines.patches import cut_patches
from sat_img_utils.core.filters import (
    min_land_fraction_filter_random,
    threshold_fraction_filter_eq,
)
from sat_img_utils.datasets.osm_land_poly import get_global_osm_land_mask
from sat_img_utils.datasets.capella import (
    init_capella_patch_config,
    process_capella_sar_tile, 
    save_capella_patch, 
    capella_sar_transform_to_uint8, 
    sar_up_contrast_convert_to_uint8
)
from sat_img_utils.geo.raster import drop_null_empty_invalid, clean_gdf, rasterize_gdf_to_mask
from sat_img_utils.geo.metadata import default_metadata_fn
# Usage example:
# cfg = PatchIterConfig(patch_size=416, nodata=0, pad_value=0, gc_every=1000)
# meta = cut_patches_general(
#     ds,
#     img_name="capella_tile_xyz",
#     out_dir="/path/to/sar_patches",
#     cfg=cfg,
#     patch_transform=up_contrast_convert_to_uint8,  # your function
#     filters_before_transform=[
#         make_max_nodata_ratio_filter(max_ratio=0.95, nodata_value=0),
#         # your land/water filter can live here too (it can consult extra_ctx["land_mask"])
#     ],
#     filters_after_transform=[
#         make_min_dynamic_range_filter(min_range=0.5, nodata_value=0),
#     ],
#     writer_fn=writer_geotiff_uint8_singleband,
#     metadata_fn=default_metadata_fn,
#     extra_ctx={"rng": rng, "land_mask": land_mask},
# )

def test_patches():
    img_path = "../../test_data/CAPELLA_C03_SP_GEO_HH_20211231164052_20211231164115.tif"
    osm_land_polygon_path = "../../test_data/land_polygons.shp"
    out_dir = "./test_output/sar_patches_3"
    ds = rasterio.open(img_path)
    print(ds.shape)
    land_mask = get_global_osm_land_mask(osm_land_polygon_path, ds)

    metadata = process_capella_sar_tile(
        ds=ds,
        out_dir=out_dir,
        land_mask=land_mask,
        patch_size=512,
        nodata=0,
    )
    # # nodata is 0 for SAR because that's physically impossible return value
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

if __name__ == "__main__":
    test_patches()