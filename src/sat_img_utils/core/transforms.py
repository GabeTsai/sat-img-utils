from typing import Union
import numpy as np
from sat_img_utils.pipelines.context import Context
from sat_img_utils.core.masks import get_valid_mask
from sat_img_utils.configs.constants import LOG_EPS

def log10_eps(img: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(img, LOG_EPS))

def sar_log10(img: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    return 20.0 * log10_eps(img * scale_factor)

def clip_percentile(img: np.ndarray, low_percentile_val: float, high_percentile_val: float) -> np.ndarray:
    return np.clip(img, low_percentile_val, high_percentile_val)

def normalize_percentile(img: np.ndarray, low_percentile_val: float, high_percentile_val: float) -> np.ndarray:
    return (img - low_percentile_val) / (high_percentile_val - low_percentile_val) * 255

def sar_up_contrast_convert_to_uint8_pval(
    img_uint16: np.ndarray,
    low_percentile_val: float,
    high_percentile_val: float,
    scale_factor: float = 1.0,
    nodata: float = 0.0,
) -> np.ndarray:
    img = img_uint16.astype(np.float32)
    valid = get_valid_mask(img, nodata=nodata) & (img > 0) & (img * scale_factor > LOG_EPS)

    out = np.zeros(img.shape, dtype=np.uint8)
    if not np.any(valid):
        return out

    img_db = clip_percentile(
        sar_log10(img[valid], scale_factor),
        low_percentile_val,
        high_percentile_val,
    )
    out[valid] = normalize_percentile(img_db, low_percentile_val, high_percentile_val)
    return out

def sar_up_contrast_convert_to_uint8_p(
    img_uint16: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    scale_factor: float = 1.0,
    nodata: float = 0.0,
) -> np.ndarray:
    img = img_uint16.astype(np.float32)

    valid = get_valid_mask(img, nodata=nodata) & (img > 0)
    out = np.zeros(img.shape, dtype=np.uint8)
    if not np.any(valid):
        return out

    img_db = sar_log10(img, scale_factor)

    vmin, vmax = np.percentile(img_db[valid], (low_percentile, high_percentile))
    img_db = clip_percentile(img_db, vmin, vmax)

    img_db[valid] = normalize_percentile(img_db[valid], vmin, vmax)
    return img_db.astype(np.uint8)

def sar_up_contrast_convert_uint8_p_ctx(
    img_uint16: np.ndarray,
    ctx: Context,
) -> np.ndarray:
    p = ctx.sar_up_contrast_convert_uint8_p
    return sar_up_contrast_convert_to_uint8_p(
        img_uint16,
        low_percentile=p.low_percentile,
        high_percentile=p.high_percentile,
        scale_factor=p.scale_factor,
        nodata=ctx.nodata,
    )

def sar_up_contrast_convert_uint8_pval_ctx(
    img_uint16: np.ndarray,
    ctx: Context,
) -> np.ndarray:
    p = ctx.sar_up_contrast_convert_uint8_pval_ctx
    return sar_up_contrast_convert_to_uint8_pval(
        img_uint16,
        low_percentile_val=p.low_percentile_val,
        high_percentile_val=p.high_percentile_val,
        scale_factor=p.scale_factor,
        nodata=ctx.nodata,
    )

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
