from typing import Union
import numpy as np
from sat_img_utils.pipelines.context import Context
from sat_img_utils.core.masks import get_valid_mask

def sar_up_contrast_convert_to_uint8(
    img_uint16: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    *,
    ctx: Context,
) -> np.ndarray:
    img = img_uint16.astype(np.float32)

    valid = get_valid_mask(img, nodata=ctx.nodata) & (img > 0)
    out = np.zeros(img.shape, dtype=np.uint8)
    if not np.any(valid):
        return out

    img_db = 10.0 * np.log10(np.maximum(img, 1.0))
    # pcts = np.percentile(img_db[valid], [0, 1, 5, 50, 95, 99, 100])
    # print("db percentiles:", pcts)
    # print("clip fraction at max:", np.mean(img_db[valid] >= db_max))
    # print("clip fraction at min:", np.mean(img_db[valid] <= db_min))
    # print("db min/max:", img_db[valid].min(), img_db[valid].max())

    # db = np.clip(img_db, db_min, db_max)
    # norm = (db - db_min) / max(db_max - db_min, 1e-6)
    # norm = np.clip(norm, 0.0, 1.0)

    # out[valid] = (norm[valid] * 255.0 + 0.5).astype(np.uint8)
    # return out

    vmin, vmax = np.percentile(img_db[valid], (low_percentile, high_percentile))
    img_db = np.clip(img_db, vmin, vmax)

    img_norm = np.zeros_like(img_db)
    img_norm[valid] = (img_db[valid] - vmin) / (vmax - vmin) * 255
    return img_norm.astype(np.uint8)
    
def pad_to_square(
    patch: np.ndarray,
    patch_size: int,
    pad_value: Union[int, float],
) -> np.ndarray:
    """
    Pad (H,W) or (C,H,W) to (patch_size,patch_size) (or (C,patch_size,patch_size)).
    """
    if patch.ndim == 2:
        h, w = patch.shape
    elif patch.ndim == 3:   
        _, h, w = patch.shape
    else:
        raise ValueError(f"Unsupported patch ndim={patch.ndim}")
    
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if patch.ndim == 2:
        h, w = patch.shape
        if pad_h == 0 and pad_w == 0:
            return patch
        return np.pad(
            patch,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    if patch.ndim == 3:
        c, h, w = patch.shape
        if pad_h == 0 and pad_w == 0:
            return patch
        return np.pad(
            patch,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    raise ValueError(f"Unsupported patch ndim={patch.ndim}")
