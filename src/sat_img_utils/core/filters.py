from typing import Optional
import rasterio
import numpy as np

def fraction_from_mask(
    binary_mask: np.ndarray, 
    valid_pixels: np.ndarray,
    target_value: int = 1
) -> float:
    """
    Calculate the fraction of target pixels in a binary mask.
    
    Args:
        binary_mask: Binary mask array where target_value indicates target pixels 
                     and background_value indicates background pixels (e.g., land=1, water=0)
        valid_pixels: Array indicating valid pixels; True for valid, False for nodata
        target_value: Value in the mask representing the target class (default: 1)
    """
    valid_count = int(valid_pixels.sum())
    if valid_count == 0:
        return 0.0
    return float(((binary_mask == target_value) & valid_pixels).sum() / valid_count)

def calculate_threshold_fraction(
    data: np.ndarray,
    threshold: float,
    nodata: Optional[float],
    *,
    greater: bool = True,
) -> float:
    """
    Generic: fraction of valid pixels satisfying data > threshold (or >= / < / etc.)
    """
    
    valid = np.ones(data.shape, dtype=bool) if nodata is None else data != nodata
    if greater:
        binary = (data > threshold) & valid
    else:
        binary = (data < threshold) & valid

    return fraction_from_mask(binary, valid, target_value=True)

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

def get_land_fraction(land_mask: np.ndarray, i: int, j: int, patch: np.ndarray, patch_size: int, nodata: float = 0) -> float:
    """
    Calculate the fraction of land pixels in a patch using land mask from OSM. 
    
    Wrapper around get_binary_mask_fraction for land/water masks.
    
    Args:
        land_mask: Binary mask where 1=land, 0=water
        i: Row index for patch extraction
        j: Column index for patch extraction
        patch: Reference patch used to determine valid pixels
        patch_size: Size of the patch to extract
        nodata: Value representing nodata pixels to exclude from calculation (default: 0)
    
    Returns:
        float: Fraction of valid pixels that are land (0.0 to 1.0)
    """
    return get_binary_mask_fraction(
        binary_mask=land_mask,
        i=i,
        j=j,
        patch=patch,
        patch_size=patch_size,
        nodata=nodata,
        target_value=1,
        background_value=0,
        default_fraction=1.0
    )

