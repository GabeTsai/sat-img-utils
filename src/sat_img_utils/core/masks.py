from typing import Optional
import rasterio
import numpy as np

def get_valid_mask(
    arr: np.ndarray,
    nodata: Optional[float | int] = None,
) -> np.ndarray:
    """
    Return boolean mask of valid pixels in `patch` (i.e., pixels not equal to `nodata`).
    If `nodata` is None, all pixels are considered valid.
    """
    valid = np.isfinite(arr)
    if nodata is not None:
        return valid & (arr != nodata)
    return valid

def patch_value_fraction(
    patch: np.ndarray,
    *,
    filter_value: float | int,
    nodata: Optional[float | int] = None,
    valid_pixels: Optional[np.ndarray] = None,
) -> float:
    """
    Return fraction of pixels in `patch` equal to `value`, among valid pixels.
    """
    if valid_pixels is None:
        if nodata is None:
            valid_pixels = np.ones(patch.shape, dtype=bool)
        else:
            valid_pixels = patch != nodata
    valid_count = int(valid_pixels.sum())
    if valid_count == 0:
        return 0.0
    return float(((patch == filter_value) & valid_pixels).sum() / valid_count)

def get_binary_mask_fraction(
    binary_mask: np.ndarray, 
    i: int, 
    j: int, 
    patch: np.ndarray, 
    patch_size: int, 
    nodata: float = 0,
    target_value: int = 1, 
    background_value: int = 0,
    default_fraction: float = 1.0,
    valid_pixels: Optional[np.ndarray] = None) -> float:
    """
    Calculate the fraction of target pixels in a binary mask patch.
    
    Args:
        binary_mask: Binary mask array where target_value indicates target pixels 
                     and background_value indicates background pixels (e.g., land=1, water=0)
        i: Row index for patch extraction
        j: Column index for patch extraction
        patch: Reference patch used to determine valid pixels (non-nodata pixels are valid)
        patch_size: Size of the patch to extract
        nodata: Value representing nodata pixels to exclude from calculation (default: 0)
        target_value: Value in the mask representing the target class (default: 1)
        background_value: Value in the mask representing the background class (default: 0)
        default_fraction: Fraction to return when binary_mask is None (default: 1.0),
        valid_pixels: Optional[np.ndarray] = None) -> float: Optional array indicating valid pixels; if None, determined from patch and nodata

    Returns:
        float: Fraction of valid pixels that are target_value (0.0 to 1.0)
    """
    if binary_mask is None:
        return default_fraction
    
    mask_patch = binary_mask[i:i+patch_size, j:j+patch_size]
    if mask_patch.shape != (patch_size, patch_size):
        mask_patch = np.pad(
            mask_patch,
            ((0, patch_size - mask_patch.shape[0]),
             (0, patch_size - mask_patch.shape[1])),
            mode="constant",
            constant_values=background_value,
        )

    if valid_pixels is None:
        if patch is None:
            raise ValueError("Provide either `valid_pixels` or `patch` to derive valid pixels.")
        valid_pixels = (patch != nodata)

    valid_count = int(valid_pixels.sum())
    if valid_count == 0:
        return 0.0

    target_pixels = (mask_patch == target_value) & valid_pixels
    return float(target_pixels.sum() / valid_count)
    
def calculate_threshold_fraction(
    data: np.ndarray,
    filter_value: float,
    nodata: Optional[float],
    *,
    greater: bool = True,
    strict: bool = False,
) -> float:
    """
    Generic: fraction of valid pixels satisfying data >= filter_value (or <= filter_value).
    """
    valid = np.ones(data.shape, dtype=bool) if nodata is None else (data != nodata)
    if greater:
        if strict:
            satisfied = data > filter_value
        else:
            satisfied = data >= filter_value   # NOTE: >= not >
    else:
        if strict:
            satisfied = data < filter_value
        else:
            satisfied = data <= filter_value

    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0

    return float((satisfied & valid).sum() / valid_count)

def get_land_fraction(
        land_mask: np.ndarray, 
        i: int, 
        j: int, 
        patch: np.ndarray, 
        patch_size: int, 
        nodata: float = 255) -> float:
    """
    Calculate the fraction of land pixels in a patch using land mask from OSM. 
    
    Wrapper around get_binary_mask_fraction for land/water masks.
    
    Args:
        land_mask: Binary mask where 1=land, 0=water
        i: Row index for patch extraction
        j: Column index for patch extraction
        patch: Reference patch used to determine valid pixels
        patch_size: Size of the patch to extract
        nodata: Value representing nodata pixels to exclude from calculation 
    
    Returns:
        float: Fraction of valid pixels that are land (0.0 to 1.0) 
    """
    land_fraction = 1.0
    if land_mask is not None:
        land_patch = land_mask[i:i+patch_size, j:j+patch_size]

        if land_patch.shape != (patch_size, patch_size):
            land_patch = np.pad(
                land_patch,
                ((0, patch_size - land_patch.shape[0]),
                (0, patch_size - land_patch.shape[1])),
                mode="constant",
                constant_values=255,
            )
        
        valid_land = (patch != nodata)
        valid_count = valid_land.sum()
        if valid_count == 0:
            land_fraction = 0.0
        else:
            land_fraction = ((land_patch == 1) & valid_land).sum() / valid_count
    return land_fraction