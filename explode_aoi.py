from pathlib import Path
import geopandas as gpd

def explode_aoi_to_files(
    in_geojson: str,
    out_dir: str,
    out_crs="EPSG:3857",
    prefix="aoi",
    pad=2,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(in_geojson)

    if out_crs is not None:
        gdf = gdf.to_crs(out_crs)

    parts = gdf.explode(index_parts=False, ignore_index=True)

    # Optional cleanup (can help avoid GEOS issues downstream)
    parts["geometry"] = parts["geometry"].buffer(0)

    # Drop empties just in case
    parts = parts[~parts.geometry.is_empty].reset_index(drop=True)

    for i, row in parts.iterrows():
        one = gpd.GeoDataFrame([row], crs=parts.crs)
        out_path = out_dir / f"{prefix}_{i:0{pad}d}_3857.geojson"
        one.to_file(out_path, driver="GeoJSON")

    print(f"Wrote {len(parts)} AOI parts -> {out_dir}")

explode_aoi_to_files(
    in_geojson="aoi_geojsons/osm_aoi_capella_def.geojson",
    out_dir="aoi_parts_3857_capella_def",
    out_crs="EPSG:3857",
    prefix="aoi",
    pad=2,
)
