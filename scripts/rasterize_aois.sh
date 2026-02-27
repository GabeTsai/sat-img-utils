#!/usr/bin/env bash
set -euo pipefail

# Rasterize landmask per AOI using GDAL/OGR. Run from repo root so paths resolve.

# -------------------------
# Parse arguments
# -------------------------
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <resolution_meters> <input_dir> <output_dir> <osm_land_gpkg>"
  echo "Example: $0 5 aoi_parts_3857 landmask_tiles_5m osm_land_3857.gpkg"
  exit 1
fi

RES="$1"
IN_DIR="$2"
OUT_DIR="$3"
OSM_LAND="$4"

if ! [[ "$RES" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: resolution must be a positive number (meters)"
  exit 1
fi

if [ ! -d "$IN_DIR" ]; then
  echo "ERROR: input directory '$IN_DIR' does not exist"
  exit 1
fi

if [ ! -f "$OSM_LAND" ]; then
  echo "ERROR: OSM land file '$OSM_LAND' does not exist"
  exit 1
fi

echo "Rasterizing landmask at ${RES} m resolution"
echo "  input dir:  $IN_DIR"
echo "  output dir: $OUT_DIR"
echo "  OSM land:   $OSM_LAND"

# -------------------------
# Output directory
# -------------------------
mkdir -p "$OUT_DIR"

# -------------------------
# Main loop
# -------------------------
for aoi in "${IN_DIR}"/aoi_*_3857.geojson; do
  base=$(basename "$aoi" .geojson)     # aoi_00_3857
  idx=${base%%_3857}                   # aoi_00

  echo "=== $idx ==="

  # Clip land polygons to AOI
  ogr2ogr -overwrite -nlt PROMOTE_TO_MULTI \
    -clipsrc "$aoi" \
    "${OUT_DIR}/${idx}_land_clip.gpkg" \
    "$OSM_LAND"

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
    -burn 1 -init 0 -a_nodata 255 -ot Byte \
    -tr "$RES" "$RES" -tap -at \
    -te "$xmin" "$ymin" "$xmax" "$ymax" \
    -a_srs EPSG:3857 \
    -co TILED=YES -co COMPRESS=ZSTD -co BIGTIFF=YES \
    "${OUT_DIR}/${idx}_land_clip.gpkg" \
    "$out_tif"

  # Overviews
  gdaladdo -r nearest "$out_tif" 2 4 8 16 32 64
done
