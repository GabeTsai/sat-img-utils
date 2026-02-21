#!/usr/bin/env bash
set -euo pipefail

# Rasterize landmask per AOI using GDAL/OGR. Run from repo root so paths resolve.

# -------------------------
# Parse arguments
# -------------------------
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <resolution_meters>"
  echo "Example: $0 5"
  exit 1
fi

RES="$1"

if ! [[ "$RES" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: resolution must be a positive number (meters)"
  exit 1
fi

echo "Rasterizing landmask at ${RES} m resolution"

# -------------------------
# Output directory
# -------------------------
OUT_DIR="landmask_tiles_${RES}m"
mkdir -p $OUT_DIR

# -------------------------
# Main loop
# -------------------------
for aoi in aoi_parts_3857/aoi_*_3857.geojson; do
  base=$(basename "$aoi" .geojson)     # aoi_00_3857
  idx=${base%%_3857}                   # aoi_00

  echo "=== $idx ==="

  # Clip land polygons to AOI
  ogr2ogr -overwrite -nlt PROMOTE_TO_MULTI \
    -clipsrc "$aoi" \
    "${OUT_DIR}/${idx}_land_clip.gpkg" \
    osm_land_3857.gpkg

  # Robust extent parse (keeps negatives)
  ext=$(ogrinfo -al -so "$aoi" \
    | sed -n 's/^Extent: //p' \
    | grep -Eo '[-]?[0-9]+([.][0-9]+)?' \
    | head -n 4 \
    | tr '\n' ' ')

  if [ -z "$ext" ]; then
    echo "ERROR: could not parse Extent for $aoi"
    exit 1
  fi

  read xmin ymin xmax ymax <<< "$ext"

  echo "  extent: $xmin $ymin $xmax $ymax"

  out_tif="${OUT_DIR}/${idx}_landmask_${RES}m_3857.tif"

  gdal_rasterize \
    -burn 1 -a_nodata 0 -ot Byte \
    -tr "$RES" "$RES" -tap -at \
    -te "$xmin" "$ymin" "$xmax" "$ymax" \
    -a_srs EPSG:3857 \
    -co TILED=YES -co COMPRESS=ZSTD -co BIGTIFF=YES \
    "${OUT_DIR}/${idx}_land_clip.gpkg" \
    "$out_tif"

  # Overviews
  gdaladdo -r nearest "$out_tif" 2 4 8 16 32 64
done
