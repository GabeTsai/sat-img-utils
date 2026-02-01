# sat-img-utils
Geospatial preprocessing utilities for SAR and optical imagery for reuse in ML pipelines and datasets.

## sat_img_utils/
### core/
Utils mainly for operating on arrays - 

### geo/
Geospatial utilites - CRS, transforms, reprojection, etc. Anything that involves rasterio or other geospatial libraries probably belongs here. 

### datasets/
Satellite image dataset (FMOW, Capella, NAIP) -specific utils. 

### pipelines/
Image processing/generation pipelines. Pipelines 
come with default configs that contain general parameters that will always exist
in that type of pipeline that should be set by the user.  \
\
Pipelines may also accept `extra_ctx`, a dict that stores dict configs for pipeline components such as filters, transformers, metadata logging methods, and more defined by the user. Certain component names are reserved, such as `"patches"`, and cannot be set by the user. 

-----
## configs/
Configs that define constants for modules/pipelines. 