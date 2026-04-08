from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any, Tuple, Union, Sequence
from pathlib import Path
import numpy as np

# Keys reserved for the pipeline namespace. Cannot be used for pipeline component names.
RESERVED = {"patch"} 

# note - metadata_crs is the CRS of the metadata, not the patch CRS
# we always use the input tile's CRS for the patch to prevent reprojection errors
@dataclass(frozen=True)
class PatchIterPipelineConfig:
    patch_size: int
    patch_dtype: np.dtype
    out_dir: Union[str, Path]
    img_name: str
    metadata_crs: Optional[Union[str, int]] = None  # e.g. "EPSG:4326" or 4326
    nodata: Optional[Union[int, float]] = None
    pad_value: Optional[Union[int, float]] = 0
    step: Optional[int] = None  # default = patch_size
    bands: Optional[Union[Sequence[int], int]] = None  # 1-based band indices; default = all
    gc_every: int = 1000  # call gc every N kept patches (not visited windows)
