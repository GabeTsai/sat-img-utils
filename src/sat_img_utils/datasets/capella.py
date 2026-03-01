import rasterio
import numpy as np
from sat_img_utils.configs.ds_constants import (
    CAPELLA_BANDS, 
    CAPELLA_EXTRA_CTX, 
    CAPELLA_OVERVIEW_TARGET_WIDTH, 
    CapellaPercentValue, 
    CapellaPolarization, 
)
from sat_img_utils.core.filters import min_land_fraction_filter_random, threshold_fraction_filter_eq
from sat_img_utils.pipelines.config import PatchIterPipelineConfig
from sat_img_utils.pipelines.context import Context
from sat_img_utils.core.transforms import sar_up_contrast_convert_uint8_pval_ctx, sar_log10

from sat_img_utils.pipelines.patches import cut_patches, init_patch_config
from sat_img_utils.geo.metadata import default_metadata_fn
from sat_img_utils.geo.raster import choose_overview_level, get_overview
from sat_img_utils.core.masks import get_valid_mask
from sat_img_utils.configs.constants import LOG_EPS

from pathlib import Path
import logging
import json
from typing import Tuple
import time

def read_scale_factor_from_capella_metadata(path_to_metadata: str) -> float:
    with open(path_to_metadata, 'r') as f:
        metadata = json.load(f)
    return metadata['collect']['image']['scale_factor']    

def get_capella_percentiles(img_name: str) -> Tuple[float, float]:
    if CapellaPolarization.VV.value in img_name or CapellaPolarization.HH.value in img_name:
        return CapellaPercentValue.LO_HH_VV.value, CapellaPercentValue.HI_HH_VV.value
    elif CapellaPolarization.VH.value in img_name or CapellaPolarization.HV.value in img_name:
        return CapellaPercentValue.LO_HV_VH.value, CapellaPercentValue.HI_HV_VH.value
    else:
        raise ValueError(f"Unknown Capella polarization in image name: {img_name}")

def save_capella_patch(patch, context: Context):
    """
    Save a SAR patch as a GeoTIFF file with appropriate georeferencing. Assumes patch is read
    from a window of a Capella SAR tile. All required parameters are provided via Context.
    Required context attributes: patch, sar_tile_window_transform, sar_tile_crs, sar_tile_name, i, j, out_dir, nodata (optional)
    """
    try:
        patch_name = f"{context.img_name}_patch_{context.patch.i}_{context.patch.j}.tif"
        out_path = f"{context.out_dir}/{patch_name}"
        profile = {
            'driver': 'GTiff',
            'height': context.patch.height,
            'width': context.patch.width,
            'count': 1,
            'dtype': 'uint8',
            'crs': context.crs,
            'transform': context.patch.transform,
            'nodata': context.nodata 
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(patch, 1)
        return 1
    except Exception as e:
        logging.error(f"Error saving patch {context.img_name}_patch_{context.patch.i}_{context.patch.j}.tif: {e}")
        return 0

def init_capella_patch_config(
    patch_size: int,
    out_dir: str,
    img_name: str,
    nodata: int = 0,
) -> PatchIterPipelineConfig:
    return init_patch_config(
        patch_size=patch_size,
        patch_dtype=np.uint8,
        out_dir=out_dir,
        img_name=img_name,
        nodata=nodata,
        pad_value=nodata,
        gc_every=1000,
        bands=CAPELLA_BANDS  # single band SAR
    )

def gen_capella_tile_patches(
    ds: rasterio.io.DatasetReader,
    out_dir: str,
    extended_metadata_path: str,
    land_mask: np.ndarray = None,
    patch_size: int = 512,
    nodata: int = 0,
) -> list[dict]:
    
    img_name = Path(ds.name).stem
    scale_factor = read_scale_factor_from_capella_metadata(extended_metadata_path)
    cfg = init_capella_patch_config(
        patch_size=patch_size,
        out_dir=out_dir,
        img_name=img_name,
        nodata=nodata,
    )
    start = time.time()
    overview_level = choose_overview_level(ds, CAPELLA_OVERVIEW_TARGET_WIDTH)
    overview = get_overview(ds, CAPELLA_BANDS, overview_level)
    valid = get_valid_mask(overview, nodata=nodata) & (overview > 0) & (overview > LOG_EPS)
    overview_db = sar_log10(overview[valid], scale_factor)
    low_percentile, high_percentile = get_capella_percentiles(img_name)
    low_percentile_val, high_percentile_val = np.percentile(overview_db, (low_percentile, high_percentile))
    end = time.time()
    logging.info(f"Time taken to get overview: {end - start:.2f} seconds")
    
    extra_ctx = CAPELLA_EXTRA_CTX.copy()
    extra_ctx["min_land_fraction_filter_random"]["land_mask"] = land_mask
    extra_ctx["sar_up_contrast_convert_uint8_pval_ctx"]["scale_factor"] = scale_factor
    extra_ctx["sar_up_contrast_convert_uint8_pval_ctx"]["low_percentile_val"] = low_percentile_val
    extra_ctx["sar_up_contrast_convert_uint8_pval_ctx"]["high_percentile_val"] = high_percentile_val

    metadata = cut_patches(
        ds=ds,
        img_name=img_name,
        out_dir=out_dir,
        cfg=cfg,
        transform=sar_up_contrast_convert_uint8_pval_ctx,
        filters_before_transform=[
            min_land_fraction_filter_random,
            threshold_fraction_filter_eq,
        ],
        writer_fn=save_capella_patch,
        metadata_fn=default_metadata_fn,
        extra_ctx=extra_ctx,
    )
    return metadata    

