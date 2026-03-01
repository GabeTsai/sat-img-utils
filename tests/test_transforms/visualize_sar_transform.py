"""
Visualize SAR transform and land mask for a Capella GEO tile.

Run from the repo root:
    python tests/test_transforms/visualize_sar_transform.py

Or pass paths directly:
    python tests/test_transforms/visualize_sar_transform.py \
        --img     /path/to/tile.tif \
        --meta    /path/to/tile_extended.json \
        --vrt     /path/to/land_mask.vrt \
        --row     11000 --col 11000 --size 2000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from types import SimpleNamespace

import rasterio

from sat_img_utils.core.transforms import sar_up_contrast_convert_uint8_pval_ctx, sar_log10
from sat_img_utils.core.masks import get_valid_mask
from sat_img_utils.configs.constants import LOG_EPS
from sat_img_utils.configs.ds_constants import CAPELLA_BANDS, CAPELLA_OVERVIEW_TARGET_WIDTH
from sat_img_utils.datasets.capella import get_capella_percentiles, read_scale_factor_from_capella_metadata
from sat_img_utils.datasets.osm_land_poly import LandMaskVRT
from sat_img_utils.geo.raster import choose_overview_level, get_overview

from typing import Optional

_LAND_COLOUR   = np.array([0.20, 0.75, 0.20, 0.35])   # semi-transparent green
_WATER_COLOUR  = np.array([0.10, 0.45, 0.85, 0.45])   # semi-transparent blue
_NODATA_COLOUR = np.array([0.80, 0.80, 0.80, 0.25])   # light grey


def _make_overlay(land_patch: np.ndarray) -> np.ndarray:
    """Convert a (H,W) land mask (0=water, 1=land, 255=nodata) to an RGBA image."""
    h, w = land_patch.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[land_patch == 1]   = _LAND_COLOUR
    rgba[land_patch == 0]   = _WATER_COLOUR
    rgba[land_patch == 255] = _NODATA_COLOUR
    return rgba


def visualize(
    path_to_img: str,
    path_to_metadata: str,
    vrt_path: str | None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    size: Optional[int] = None,
) -> None:
    img_stem = Path(path_to_img).stem
    
    with rasterio.open(path_to_img) as ds:
        if row is None:
            row = 0
        if col is None:
            col = 0
        if size is None:
            size = ds.height

        print("Percent of image that is black: ", np.mean(ds.read(1) == 0))
        nodata = ds.nodata if ds.nodata is not None else 0
        overview_level = choose_overview_level(ds, CAPELLA_OVERVIEW_TARGET_WIDTH)
        overview = get_overview(ds, CAPELLA_BANDS, overview_level)

        scale_factor = read_scale_factor_from_capella_metadata(path_to_metadata)
        low_pct, high_pct = get_capella_percentiles(img_stem)

        valid = (
            get_valid_mask(overview, nodata=nodata)
            & (overview > 0)
            & (overview * scale_factor > LOG_EPS)
        )
        overview_db = sar_log10(overview[valid], scale_factor)
        low_val, high_val = np.percentile(overview_db, (low_pct, high_pct))
        print(f"Scale factor : {scale_factor}")
        print(f"Percentiles  : [{low_pct}, {high_pct}]  →  [{low_val:.2f}, {high_val:.2f}] dB")

        ctx = SimpleNamespace(
            nodata=nodata,
            sar_up_contrast_convert_uint8_pval_ctx=SimpleNamespace(
                scale_factor=scale_factor,
                low_percentile_val=low_val,
                high_percentile_val=high_val,
            ),
        )

        img_raw = ds.read(1)
        img_u8  = sar_up_contrast_convert_uint8_pval_ctx(img_raw, ctx=ctx)

        land_mask = None
        if vrt_path is not None:
            lmv = LandMaskVRT(vrt_path)
            land_mask = lmv.get_mask_for_tile(ds)
            lmv.close()
            total = land_mask.size
            n_land  = int((land_mask == 1).sum())
            n_water = int((land_mask == 0).sum())
            n_nodata= int((land_mask == 255).sum())
            print(
                f"Land mask    : land={n_land/total*100:.1f}%  "
                f"water={n_water/total*100:.1f}%  "
                f"nodata={n_nodata/total*100:.1f}%"
            )

    r0, r1 = row, row + size
    c0, c1 = col, col + size

    crop_raw  = img_raw[r0:r1, c0:c1]
    crop_u8   = img_u8 [r0:r1, c0:c1]
    crop_mask = land_mask[r0:r1, c0:c1] if land_mask is not None else None

    ncols = 4 if crop_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    fig.suptitle(img_stem, fontsize=9)

    raw_log = np.log1p(crop_raw.astype(np.float32))
    axes[0].imshow(raw_log, cmap="gray")
    axes[0].set_title("Raw SAR (log scale)")
    axes[0].axis("off")

    axes[1].imshow(crop_u8, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("SAR uint8 (dB stretched)")
    axes[1].axis("off")

    if crop_mask is not None:
        cmap_mask = ListedColormap(
            [_WATER_COLOUR[:3], _LAND_COLOUR[:3], _NODATA_COLOUR[:3]]
        )
        im = axes[2].imshow(
            np.where(crop_mask == 255, 2, crop_mask),  # remap nodata→2 for 3-class cmap
            cmap=cmap_mask, vmin=0, vmax=2, interpolation="nearest",
        )
        legend_patches = [
            mpatches.Patch(color=_WATER_COLOUR[:3], label="Water (0)"),
            mpatches.Patch(color=_LAND_COLOUR[:3],  label="Land (1)"),
            mpatches.Patch(color=_NODATA_COLOUR[:3], label="Nodata (255)"),
        ]
        axes[2].legend(handles=legend_patches, loc="lower right", fontsize=7)
        axes[2].set_title("Land mask")
        axes[2].axis("off")

        # Overlay: SAR + land mask
        axes[3].imshow(crop_u8, cmap="gray", vmin=0, vmax=255)
        axes[3].imshow(_make_overlay(crop_mask), interpolation="nearest")
        axes[3].legend(handles=legend_patches, loc="lower right", fontsize=7)
        axes[3].set_title("SAR + land mask overlay")
        axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAR transform + land mask.")
    parser.add_argument("--img",  default="test_data/test_capella_data/2024/CAPELLA_C09_SP_GEO_HH_20240512190416_20240512190441/CAPELLA_C09_SP_GEO_HH_20240512190416_20240512190441.tif")
    parser.add_argument("--meta", default="test_data/test_capella_data/2024/CAPELLA_C09_SP_GEO_HH_20240512190416_20240512190441/CAPELLA_C09_SP_GEO_HH_20240512190416_20240512190441_extended.json")
    parser.add_argument("--vrt",  default=None, help="Path to land mask VRT (optional)")
    parser.add_argument("--row",  type=int, help="Top-left row of crop window")
    parser.add_argument("--col",  type=int, help="Top-left col of crop window")
    parser.add_argument("--size", type=int,  help="Crop window side length in pixels")
    args = parser.parse_args()

    visualize(
        path_to_img=args.img,
        path_to_metadata=args.meta,
        vrt_path=args.vrt,
        row=args.row,
        col=args.col,
        size=args.size,
    )
