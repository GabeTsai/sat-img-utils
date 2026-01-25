# sat-img-utils
Geospatial preprocessing utilities for SAR and optical imagery for reuse in ML pipelines and datasets.

## sat_img_utils/
### core/
Utils mainly for operating on arrays. 

### geo/
Geospatial utilites - CRS, transforms, reprojection, etc. Anything that involves rasterio or other geospatial libraries probably belongs here. 

### datasets/
Satellite image dataset (FMOW, Capella, NAIP) -specific utils. 

### pipelines/
Image processing/generation pipelines. Configs 

-----
## configs/
Configs that define knobs for general pipeline recipes.