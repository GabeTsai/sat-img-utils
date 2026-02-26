from __future__ import annotations
import gc 
import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

from sat_img_utils.pipelines.config import PatchIterPipelineConfig
from sat_img_utils.pipelines.context import Context
from sat_img_utils.core.transforms import pad_to_square
from sat_img_utils.geo.raster import window_center_longlat   

# return true to keep patch, false to skip it
PatchFilter = Callable[[np.ndarray, Context], bool]

PatchTransform = Callable[[np.ndarray, Context], Optional[np.ndarray]]

# produce metadata dict for a patch
PatchMetadata = Callable[[Context], Dict[str, Any]]

WriterFn = Callable[[np.ndarray, Context], None]

def cut_patches(
    ds: rasterio.io.DatasetReader,
    *,
    img_name: str,
    out_dir: Union[str, Path],
    cfg: PatchIterPipelineConfig,
    transform: Optional[PatchTransform] = None,
    filters_before_transform: Optional[Sequence[PatchFilter]] = None,
    filters_after_transform: Optional[Sequence[PatchFilter]] = None,
    metadata_fn: Optional[PatchMetadata] = None,
    writer_fn: Optional[WriterFn] = None,
    extra_ctx: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
    """
    General pattern: 
    patch through the image
    pad patch as necessary
    apply patch-level filters to decide whether to keep patch
    log any metadata about the patch with whatever custom function (default is just save patch name and longitude/latitude center)
    return the patch metadata as a list of dicts

    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = cfg.step if cfg.step is not None else cfg.patch_size
    H, W = ds.height, ds.width
    kept = 0
    metadata: List[Dict[str, Any]] = []
    filters_before_transform = list(filters_before_transform or [])
    filters_after_transform = list(filters_after_transform or [])
    
    logging.info(f"Starting patches for {img_name}")

    for i in range(0, H, step):
        for j in range(0, W, step):
            window_h, window_w = min(cfg.patch_size, H - i), min(cfg.patch_size, W - j)
            window = Window(j, i, window_w, window_h)

            patch_name = f"{img_name}_patch_{i}_{j}.npy"
            out_path = out_dir / patch_name

            if cfg.bands is None:
                patch = ds.read(window=window)
            else:
                patch = ds.read(cfg.bands, window=window)
            patch = pad_to_square(
                patch,
                patch_size=cfg.patch_size,
                pad_value=cfg.pad_value,
            ) 
            long_center, lat_center = window_center_longlat(ds, window)
            patch_extra_ctx = dict(extra_ctx or {})
            # General per-patch context
            patch_extra_ctx["patch"] = {
                "patch_name": patch_name,
                "i": i,
                "j": j,
                "height": cfg.patch_size,
                "width": cfg.patch_size,
                "window": window,
                "transform": ds.window_transform(window),
                "long_center": long_center,
                "lat_center": lat_center,
                "crs": cfg.crs,
            }

            context = Context(cfg=cfg, extra=patch_extra_ctx)

            # Filters before transform
            skip = False
            for f in filters_before_transform:
                if not f(patch, context):
                    skip = True
                    break
            if skip:
                continue

            # Transform
            if transform is not None:
                patch = transform(patch, context)
                if patch is None:
                    continue

            # Filters after transform
            skip = False
            for f in filters_after_transform:
                if not f(patch, context):
                    skip = True
                    break
            if skip:
                continue
            
            writer_fn(patch, context)
            metadata.append(metadata_fn(context))

            kept += 1
            if cfg.gc_every and kept % cfg.gc_every == 0:
                gc.collect()
    logging.info(f"Finished {img_name}: kept {kept} patches")
    return metadata

def init_patch_config(
    patch_size: int,
    patch_dtype: np.dtype,
    out_dir: Union[str, Path],
    img_name: str,
    crs: Optional[Union[str, int]] = None,
    nodata: Optional[Union[int, float]] = None,
    pad_value: Optional[Union[int, float]] = 0,
    step: Optional[int] = None,
    bands: Optional[Union[Sequence[int], int]] = None,
    gc_every: int = 1000,
) -> PatchIterPipelineConfig:
    """
    Helper function to initialize PatchIterPipelineConfig
    """
    return PatchIterPipelineConfig(
        patch_size=patch_size,
        patch_dtype=patch_dtype,
        out_dir=out_dir,
        img_name=img_name,
        crs=crs,
        nodata=nodata,
        pad_value=pad_value,
        step=step,
        bands=bands,
        gc_every=gc_every,
)
