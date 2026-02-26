import rasterio
import numpy as np
from types import SimpleNamespace
from sat_img_utils.core.transforms import sar_up_contrast_convert_uint8_pval_ctx, sar_log10
from sat_img_utils.pipelines.context import Context
from sat_img_utils.configs.ds_constants import CAPELLA_EXTRA_CTX, CAPELLA_OVERVIEW_TARGET_WIDTH, CAPELLA_BANDS
from sat_img_utils.geo.raster import choose_overview_level, get_overview
from sat_img_utils.core.masks import get_valid_mask
from sat_img_utils.datasets.capella import get_capella_percentiles, read_scale_factor_from_capella_metadata
import matplotlib.pyplot as plt
from sat_img_utils.configs.constants import LOG_EPS
from pathlib import Path
import json

def test_sar_up_contrast_convert_to_uint8(path_to_img: str):
    with rasterio.open(path_to_img) as ds:
        img = ds.read(1)  # single-band HH or VV
        nodata = ds.nodata  # may be None for Capella (nodata=0 by convention)
        overview_level = choose_overview_level(ds, CAPELLA_OVERVIEW_TARGET_WIDTH)
        print(f"Overview level: {overview_level}")
        overview = get_overview(ds, CAPELLA_BANDS, overview_level)

    ctx = SimpleNamespace(
        nodata=nodata if nodata is not None else 0,
        sar_up_contrast_convert_uint8_pval_ctx=SimpleNamespace(
            scale_factor=1.0,
            low_percentile_val=0.0,
            high_percentile_val=255.0,
        )
    )
    scale_factor = read_scale_factor_from_capella_metadata(path_to_metadata)
    print(f"Scale factor: {scale_factor}")
    low_percentile, high_percentile = get_capella_percentiles(Path(path_to_img).stem)
    valid = get_valid_mask(overview, nodata=nodata) & (overview > 0) & (overview * scale_factor> LOG_EPS)
    overview_db = sar_log10(overview[valid], scale_factor)
    low_percentile_val, high_percentile_val = np.percentile(overview_db, (low_percentile, high_percentile))
    print(f"Low percentile value: {low_percentile_val}, High percentile value: {high_percentile_val}")
    ctx.sar_up_contrast_convert_uint8_pval_ctx.scale_factor = scale_factor
    ctx.sar_up_contrast_convert_uint8_pval_ctx.low_percentile_val = low_percentile_val
    ctx.sar_up_contrast_convert_uint8_pval_ctx.high_percentile_val = high_percentile_val
    img_u8 = sar_up_contrast_convert_uint8_pval_ctx(img, ctx=ctx)[11000:13000, 11000:13000]

    plt.figure(figsize=(10, 10))
    # plt.hist(img_u8.flatten(), bins=256, range=(0, 256))
    # plt.show()
    plt.imshow(img_u8, cmap="gray")
    plt.colorbar(label="Intensity (uint8, dB-stretched)")
    plt.title("SAR Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path_to_img = "../../test_data/test_capella_data/2025/CAPELLA_C15_SS_GEO_HH_20250112055128_20250112055142/CAPELLA_C15_SS_GEO_HH_20250112055128_20250112055142.tif"
    path_to_metadata = "../../test_data/test_capella_data/2025/CAPELLA_C15_SS_GEO_HH_20250112055128_20250112055142/CAPELLA_C15_SS_GEO_HH_20250112055128_20250112055142_extended.json"
    test_sar_up_contrast_convert_to_uint8(path_to_img)