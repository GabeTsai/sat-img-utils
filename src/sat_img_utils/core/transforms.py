import numpy as np

def up_contrast_convert_to_uint8(img_uint16: np.ndarray) -> np.ndarray:
    img = img_uint16.astype(np.float32)

    mask = img > 0  # ignore zero values
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.uint8)

    img = np.maximum(img, 1.0)
    img_db = 10 * np.log10(img)

    vmin, vmax = np.percentile(img_db[mask], (1, 99))
    img_db = np.clip(img_db, vmin, vmax)

    img_norm = np.zeros_like(img_db)
    img_norm[mask] = (img_db[mask] - vmin) / (vmax - vmin) * 255

    return img_norm.astype(np.uint8)