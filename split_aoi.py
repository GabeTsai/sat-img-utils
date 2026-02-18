import os
import geopandas as gpd

IN_AOI = "aoi_parts.geojson"
OUT_DIR = "aoi_parts_3857"

os.makedirs(OUT_DIR, exist_ok=True)

aoi = gpd.read_file(IN_AOI)
aoi_3857 = aoi.to_crs(3857)

for i in range(len(aoi_3857)):
    out_path = os.path.join(OUT_DIR, f"aoi_{i:02d}_3857.geojson")
    gpd.GeoDataFrame({"id":[i]}, geometry=[aoi_3857.geometry.iloc[i]], crs=3857).to_file(
        out_path, driver="GeoJSON"
    )

print(f"Wrote {len(aoi_3857)} AOI parts to {OUT_DIR}/")
