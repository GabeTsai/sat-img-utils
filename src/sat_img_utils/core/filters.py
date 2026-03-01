from typing import Optional, Callable
import rasterio
import numpy as np

from sat_img_utils.pipelines.context import Context
from sat_img_utils.core.masks import get_binary_mask_fraction, calculate_threshold_fraction, get_land_fraction, patch_value_fraction

def _land_fraction_check(patch: np.ndarray, ctx: Context, params) -> float:
    return get_land_fraction(
        land_mask=params.land_mask,
        i=ctx.patch.i,
        j=ctx.patch.j,
        patch=patch,
        patch_size=ctx.patch_size,
    )

def min_land_fraction_filter(patch: np.ndarray, ctx: Context) -> bool:
    p = ctx.min_land_fraction_filter
    lf = _land_fraction_check(patch, ctx, p)
    return lf >= p.min_land_threshold

def min_land_fraction_filter_random(patch: np.ndarray, ctx: Context) -> bool:
    p = ctx.min_land_fraction_filter_random
    lf = _land_fraction_check(patch, ctx, p)
    if lf >= p.min_land_threshold:
        return True
    else:   # discard below threshold images with probability discard_prob
        return np.random.rand() > p.discard_prob
        
def threshold_fraction_filter(
    patch: np.ndarray,
    ctx: Context,
) -> bool:
    """
    Callback filter to check if fraction of pixels in the patch satisfying
    data >= threshold (or <= threshold) meets minimum fraction.
    """
    p = ctx.threshold_fraction_filter
    frac = calculate_threshold_fraction(
        data=patch,
        filter_value=p.filter_value,
        nodata=p.nodata,
        greater = p.greater
    )
    if p.max_fraction: # if the fraction we passed is an upper bound
        return frac <= p.fraction_value
    else: # else it's a lower bound
        return frac >= p.fraction_value

def threshold_fraction_filter_eq(
    patch: np.ndarray,
    ctx: Context,
) -> bool:
    """
    Callback filter to check if fraction of pixels in the patch satisfying
    data == target_value meets minimum fraction.
    """
    p = ctx.threshold_fraction_filter_eq
    frac = patch_value_fraction(
        patch=patch,
        filter_value=p.filter_value,
        nodata=p.nodata
    )

    if p.upper: # if the fraction we passed is an upper bound
        return frac <= p.fraction_value
    else: # else it's a lower bound
        return frac >= p.fraction_value

