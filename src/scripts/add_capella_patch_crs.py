import logging

import rasterio
from pathlib import Path
from sat_img_utils.datasets.capella import iter_capella_sar_paths

import argparse
from tqdm import tqdm


def build_tile_path_map(capella_dir: str, flat: bool = False) -> dict[str, Path]:
    """Map tile stem -> tile path for all Capella SAR tiles found in capella_dir."""
    return {Path(sar_path).stem: Path(sar_path) for sar_path, _ in iter_capella_sar_paths(capella_dir, flat=flat)}


def add_patch_crs(patch_dir: str, capella_dir: str, flat: bool = False):
    """
    Add the source CRS of each SAR tile to its corresponding patches.
    Patches whose source tile is not found in capella_dir are skipped cleanly
    since patches may have been generated from multiple source directories.
    """
    tile_path_map = build_tile_path_map(capella_dir, flat)
    for patch_path in tqdm(list(Path(patch_dir).glob("*.tif")), desc="Adding source crs to patches"):
        patch_sar_tile = patch_path.stem.split("_patch_")[0]
        sar_tile_path = tile_path_map.get(patch_sar_tile)
        if sar_tile_path is None:
            logging.debug(f"Skipping {patch_path.name}: source tile '{patch_sar_tile}' not in {capella_dir}")
            continue
        with rasterio.open(sar_tile_path) as sar_tile_ds:
            sar_tile_crs = sar_tile_ds.crs
        with rasterio.open(patch_path, "r+") as patch_ds:
            patch_ds.crs = sar_tile_crs


def verify_patch_has_crs(patch_dir: str):
    for patch_path in tqdm(Path(patch_dir).glob("*.tif"), desc="Verifying patches have crs"):
        with rasterio.open(patch_path) as patch_ds:
            if patch_ds.crs is None:
                logging.warning(f"Patch {patch_path} does not have a crs")
                continue
            patch_ds.close()
    return True

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Add source crs of sar tile to corresponding patches")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to patch directory")
    parser.add_argument("--capella_dir", type=str, required=True, help="Path to capella directory")
    parser.add_argument("--flat", action="store_true", help="Set if capella_dir is flat (no year subfolders)")
    args = parser.parse_args()
    add_patch_crs(args.patch_dir, args.capella_dir, args.flat)
    assert(verify_patch_has_crs(args.patch_dir))

if __name__ == "__main__":
    main()

