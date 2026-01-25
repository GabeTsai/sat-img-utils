import rasterio
import numpy as np
from sat_img_utils.datasets.ghsl import detect_buildings

def test_detect_buildings(ghsl_dataset: rasterio.DatasetReader, sar_dataset: rasterio.DatasetReader):
    """
    Test the detect_buildings function using GHSL and SAR datasets.
    
    Args:
        ghsl_dataset: Open GHSL raster dataset
        sar_dataset: Open SAR raster dataset to match geometry 
    """
    building_fraction = detect_buildings(ghsl_dataset, sar_dataset)
    print(f"Detected building fraction: {building_fraction:.4f}")
    assert 0.0 <= building_fraction <= 1.0, "Building fraction should be between 0.0 and 1.0"

if __name__ == "__main__":
    ghsl_path = "../../test_data/GHS_BUILT_S_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif"
    sar_path = "../../test_data/CAPELLA_C05_SM_GEO_HH_20211226193545_20211226193549_patch_11264_8192.tif"
    
    with rasterio.open(ghsl_path) as ghsl_dataset, rasterio.open(sar_path) as sar_dataset:
        test_detect_buildings(ghsl_dataset, sar_dataset)