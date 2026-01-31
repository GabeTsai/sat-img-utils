import rasterio
from shapely.geometry import Point
from sat_img_utils.pipelines.context import Context


def default_metadata_fn(ctx: Context):
    """
    Default metadata function to record patch name and center longitude/latitude.
    """
    return {
        "img_name": ctx.img_name,
        "patch_name": ctx.patch.patch_name,
        "center": Point(ctx.patch.long_center, ctx.patch.lat_center), 
    } 