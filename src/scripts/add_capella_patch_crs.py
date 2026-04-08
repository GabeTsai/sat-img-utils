import logging

import rasterio
from pathlib import Path
from sat_img_utils.datasets.capella import iter_capella_sar_paths

import argparse
from tqdm import tqdm
def build_tile_path_map(capella_dirs: list[tuple[str, bool]]) -> dict[str, Path]:
    """
    Map tile stem to tile path across all provided Capella source directories.
    Each entry is a (capella_dir_path, flat) tuple indicating the layout of that directory.
    """
    tile_path_map = {}
    for capella_dir, flat in capella_dirs:
        tile_path_map.update(
            {Path(sar_path).stem: Path(sar_path) for sar_path, _ in iter_capella_sar_paths(capella_dir, flat=flat)}
        )
    return tile_path_map


def add_patch_crs(patch_dir: str, capella_dirs: list[tuple[str, bool]], corrupt_log_path: str | None = None):
    """
    Add the source CRS of each SAR tile to its corresponding patches.
    Patches whose source tile is not found across any of capella_dirs are skipped cleanly.
    Corrupt patches (e.g. truncated TIFFs missing StripOffsets) are skipped and their paths
    written to corrupt_log_path so they can be regenerated.
    """
    tile_path_map = build_tile_path_map(capella_dirs)
    sar_tile_crs_cache: dict[Path, rasterio.crs.CRS] = {}
    corrupt_patches: list[Path] = []
    logging.info(f"Built tile path map with {len(tile_path_map)} tiles from {len(capella_dirs)} source directories")
    for patch_path in tqdm(list(Path(patch_dir).glob("*.tif")), desc="Adding source crs to patches"):
        patch_sar_tile = patch_path.stem.split("_patch_")[0]
        sar_tile_path = tile_path_map.get(patch_sar_tile)
        if sar_tile_path is None:
            logging.debug(f"Skipping {patch_path.name}: source tile '{patch_sar_tile}' not found in any source directory")
            continue
        if sar_tile_path not in sar_tile_crs_cache:
            with rasterio.open(sar_tile_path) as sar_tile_ds:
                sar_tile_crs_cache[sar_tile_path] = sar_tile_ds.crs
        try:
            with rasterio.open(patch_path, "r+") as patch_ds:
                patch_ds.crs = sar_tile_crs_cache[sar_tile_path]
        except Exception as e:
            logging.warning(f"Corrupt patch skipped ({patch_path.name}): {e}")
            corrupt_patches.append(patch_path)

    if corrupt_patches:
        out_path = Path(corrupt_log_path) if corrupt_log_path else Path(patch_dir) / "corrupt_patches.txt"
        out_path.write_text("\n".join(str(p) for p in corrupt_patches))
        logging.warning(f"{len(corrupt_patches)} corrupt patches written to {out_path} — these must be regenerated")

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Add source crs of sar tile to corresponding patches")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to patch directory")
    parser.add_argument("--capella_dirs", type=str, nargs="*", default=[], help="Year-nested Capella source directories")
    parser.add_argument("--flat_capella_dirs", type=str, nargs="*", default=[], help="Flat Capella source directories (no year subfolders)")
    parser.add_argument("--corrupt_log", type=str, default=None, help="Path to write corrupt patch filenames (default: <patch_dir>/corrupt_patches.txt)")
    args = parser.parse_args()
    capella_dirs = [(d, False) for d in args.capella_dirs] + [(d, True) for d in args.flat_capella_dirs]
    if not capella_dirs:
        parser.error("Provide at least one directory via --capella_dirs or --flat_capella_dirs")
    add_patch_crs(args.patch_dir, capella_dirs, corrupt_log_path=args.corrupt_log)

if __name__ == "__main__":
    main()

