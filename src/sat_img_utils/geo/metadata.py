import rasterio
from shapely.geometry import Point
import geopandas as gpd
from sat_img_utils.pipelines.context import Context


def default_metadata_fn(ctx: Context):
    """
    Default metadata function to record patch name and center longitude/latitude.
    """
    return {
        "img_name": ctx.img_name,
        "patch_name": ctx.patch.patch_name,
        "geometry": Point(ctx.patch.long_center, ctx.patch.lat_center), 
        "crs": f"EPSG:{ctx.crs}",
    } 

def list_dict_to_parquet(
    metadata_list: list[dict],
    out_path: str,
    crs: int = 4326,
):
    """
    Save a list of metadata dictionaries to a Parquet file.
    """
    gdf = gpd.GeoDataFrame(metadata_list, crs=f'EPSG:{crs}')
    gdf.to_parquet(out_path)