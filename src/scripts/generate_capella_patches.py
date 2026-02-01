import os
import logging

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd

from sat_img_utils.core.utils import get_memory_mb, make_dirs_if_not_exists, make_dirs_if_not_exists
from sat_img_utils.geo.metadata import list_dict_to_parquet
from sat_img_utils.datasets.ghsl import detect_buildings
from sat_img_utils.datasets.osm_land_poly import osm_rasterize_sat_land_mask
from sat_img_utils.datasets.capella import process_capella_sar_tile
from sat_img_utils.geo.raster import get_gdf
from sat_img_utils.configs import ds_constants

from pathlib import Path
import glob
import gc
import logging

def process_sar_single_image(sar_path, ghsl, land_global, out_dir,
                             patch_size, metadata_out_dir, crs):
    logging.info(f"\nProcessing {sar_path}")
    logging.info(f"Initial memory: {get_memory_mb():.0f}MB")
    num_patches = 0
    with rasterio.open(sar_path) as sar:
        if max(sar.res[0], sar.res[1]) >= ds_constants.CAPELLA_RES_THRESHOLD_M:
            logging.info(f"Skipping {sar_path} due to low resolution: {sar.res}")
        else:
            building_ratio = detect_buildings(ghsl, sar, filter_value = ds_constants.GHSL_BUILDINGS_THRESHOLD)
            logging.info(f"Total building coverage for {Path(sar_path).stem}: {building_ratio}")
            
            if building_ratio >= ds_constants.GHSL_MIN_BUILDING_COVG:
                land_mask = osm_rasterize_sat_land_mask(land_global, sar)

                metadata = process_capella_sar_tile(
                    ds=sar,
                    out_dir=out_dir,
                    land_mask=land_mask,
                    patch_size=patch_size,
                    nodata=0,
                )
                metadata_path = f"{metadata_out_dir}/patch_metadata_{Path(sar_path).stem}.parquet"
                list_dict_to_parquet(
                    metadata_list=metadata,
                    out_path=metadata_path,
                    crs=crs
                )
                num_patches = len(metadata)
        logging.info(f"Total patches saved for {Path(sar_path).stem}: {num_patches}")
            
    sar.close()
    gc.collect()
    logging.info(f"Final memory: {get_memory_mb():.0f}MB")
    return num_patches

def process_sar(capella_dir, 
                target_dir, 
                ghsl_path, 
                osm_land_poly_path, 
                patch_size, 
                crs=4326, 
                flat=False):
    """
    Process SAR images, supporting two directory structures:
    1. Year-based: capella_dir/YEAR/DIR_NAME/DIR_NAME.tif
    2. Flat: capella_dir/DIR_NAME/DIR_NAME.tif
    Set flat=True for the second structure.
    """

    make_dirs_if_not_exists(target_dir)
    new_target_dir = f'{target_dir}/sar_patches'
    make_dirs_if_not_exists(new_target_dir)
    patch_metadata_path = f'{target_dir}/patch_metadata'
    make_dirs_if_not_exists(patch_metadata_path)

    land_global = get_gdf(osm_land_poly_path)

    total_num_patches = 0
    logging.info(f'Processing SAR images in {capella_dir}')
    with rasterio.open(ghsl_path) as ghsl:
        if flat:
            dir_names = os.listdir(capella_dir)
            for dir_name in dir_names:
                if 'geo' in dir_name.lower() and 'capella' in dir_name.lower():
                    sar_path = f'{capella_dir}/{dir_name}/{dir_name}.tif'
                    logging.info(f'Processing SAR image: {sar_path}')
                    total_num_patches += process_sar_single_image(
                        sar_path, ghsl, land_global, new_target_dir, patch_size, patch_metadata_path, crs=crs
                    )
        else:
            for year in ds_constants.CAPELLA_YEARS:
                year_dir = f'{capella_dir}/{year}'
                logging.info(f'Processing year directory: {year_dir}')
                if not os.path.exists(year_dir):
                    continue
                dir_names = os.listdir(year_dir)
                for dir_name in dir_names:
                    if 'geo' in dir_name.lower() and 'capella' in dir_name.lower():
                        sar_path = f'{year_dir}/{dir_name}/{dir_name}.tif'
                        total_num_patches += process_sar_single_image(
                            sar_path, ghsl, land_global, new_target_dir, patch_size, patch_metadata_path, crs=crs
                        )

    logging.info(f'Total patches saved: {total_num_patches}')

    files = glob.glob(f"{patch_metadata_path}/patch_metadata_*.parquet")
    dfs = [gpd.read_parquet(f) for f in files]
    if len(dfs) == 0:
        logging.info("No patch metadata files found to merge.")
    else:
        merged = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=f"EPSG:{crs}")
        merged.to_parquet(f"{patch_metadata_path}/patch_metadata_all.parquet", index=False)
        logging.info(f'Merged metadata saved to {patch_metadata_path}/patch_metadata_all.parquet')

if __name__ == "__main__":
    """
    Sample usage: 
    python src/scripts/generate_capella_patches.py \
        --capella_dir /path/to/capella_sar_root \
        --target_dir /path/to/output_patches_and_metadata \
        --ghsl_path /path/to/ghsl_raster.tif \
        --osm_land_poly_path /path/to/osm_land_polygons.shp \
        --flat
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate Capella SAR patches.")
    parser.add_argument('--capella_dir', required=True, help='Path to Capella SAR root directory')
    parser.add_argument('--target_dir', required=True, help='Output directory for patches and metadata')
    parser.add_argument('--ghsl_path', required=True, help='Path to GHSL raster file')
    parser.add_argument('--osm_land_poly_path', required=True, help='Path to OSM land polygons file')
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches to generate')
    parser.add_argument('--crs', type=int, default=ds_constants.CAPELLA_DEFAULT_OUT_CRS, help='Output CRS EPSG code for metadata')
    parser.add_argument('--flat', action='store_true', help='Set if capella_dir is flat (no year subfolders)')
    parser.add_argument('--log', action='store_true', help='Enable logging output')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.INFO)

    process_sar(
        capella_dir=args.capella_dir,
        target_dir=args.target_dir,
        ghsl_path=args.ghsl_path,
        osm_land_poly_path=args.osm_land_poly_path,
        patch_size=args.patch_size,
        crs=args.crs,
        flat=args.flat
    )

