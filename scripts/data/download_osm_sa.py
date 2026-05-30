"""Download Geofabrik's Texas OSM extract and clip roads + buildings to
the San Antonio model AOI.

Mirror of download_osm_minneapolis.py for the SA city. The Texas free.shp.zip
is ~1.34 GB — larger than MN's 485 MB so first run takes longer.

Outputs (EPSG:5070, matching `data/sa/flood/land_use_2021_sa.tif`):
  data/sa/roads_sa.geojson      (Option B class filter — drop sub-pixel surfaces)
  data/sa/buildings_sa.geojson
"""

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pyproj
import requests
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from download_osm_minneapolis import ROADS_DROP_CLASSES   # reuse Option B filter

OSM_URL  = "https://download.geofabrik.de/north-america/us/texas-latest-free.shp.zip"
OSM_DIR  = Path("data/osm")
ZIP_PATH = OSM_DIR / "texas.shp.zip"
EXTRACT_DIR = OSM_DIR / "texas"

TEMPLATE  = Path("data/sa/flood/land_use_2021_sa.tif")

ROADS_LAYER     = "gis_osm_roads_free_1.shp"
BUILDINGS_LAYER = "gis_osm_buildings_a_free_1.shp"

OUT_DIR        = Path("data/sa")
ROADS_OUT      = OUT_DIR / "roads_sa.geojson"
BUILDINGS_OUT  = OUT_DIR / "buildings_sa.geojson"


def download_zip():
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000_000_000:
        print(f"  zip already present at {ZIP_PATH} "
              f"({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB), skipping download")
        return
    print(f"Downloading Texas OSM shapefiles from Geofabrik...")
    print(f"  URL: {OSM_URL}")
    print(f"  (Texas state extract is ~1.34 GB — expect 5–15 min)")
    with requests.get(OSM_URL, stream=True, timeout=1200) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written, next_pct = 0, 10
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = int(100 * written / total)
                    if pct >= next_pct:
                        print(f"    {pct:>3}% ({written / 1024 / 1024:.0f} / "
                              f"{total / 1024 / 1024:.0f} MB)")
                        next_pct = pct + 10
    print(f"  wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB)")


def extract_zip():
    needed = [EXTRACT_DIR / ROADS_LAYER, EXTRACT_DIR / BUILDINGS_LAYER]
    if all(p.exists() for p in needed):
        print(f"  shapefiles already extracted under {EXTRACT_DIR}, skipping")
        return
    print(f"Extracting {ZIP_PATH} → {EXTRACT_DIR} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print(f"  extracted {len(list(EXTRACT_DIR.iterdir()))} files")


def aoi_bbox():
    """Return (4326 bbox with buffer, target_crs from template)."""
    with rasterio.open(TEMPLATE) as tpl:
        l, b, r, t = tpl.bounds
        crs = tpl.crs
    minx, miny, maxx, maxy = transform_bounds(crs, "EPSG:4326", l, b, r, t)
    return (minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01), crs


def clip_layer(layer_path, label, out_path, target_crs, drop_classes=None):
    bbox, _ = aoi_bbox()
    print(f"\n{label}: reading {layer_path.name} with bbox filter {bbox} ...")
    gdf = gpd.read_file(layer_path, bbox=bbox, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    print(f"  bbox-filtered features: {len(gdf):,} (in {gdf.crs})")

    gdf = gdf.to_crs(target_crs)
    with rasterio.open(TEMPLATE) as tpl:
        rect = box(*tpl.bounds)
    clipped = gdf[gdf.intersects(rect)].copy()
    print(f"  intersects template AOI rectangle: {len(clipped):,}")

    if drop_classes and "fclass" in clipped.columns:
        before = len(clipped)
        clipped = clipped[~clipped["fclass"].isin(drop_classes)].copy()
        print(f"  after dropping {len(drop_classes)} narrow classes: {len(clipped):,} "
              f"(removed {before - len(clipped):,})")

    if "fclass" in clipped.columns:
        print(f"  fclass counts: {dict(clipped['fclass'].value_counts())}")
    geom_types = clipped.geom_type.value_counts().to_dict()
    print(f"  geometry types: {geom_types}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    clipped.to_file(out_path, driver="GeoJSON")
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{len(clipped):,} features)")


def main():
    if not TEMPLATE.exists():
        print(f"ERROR: template raster {TEMPLATE} missing — run download_sa_data.py first")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_zip()
    extract_zip()

    bbox, target_crs = aoi_bbox()
    print(f"\nTemplate CRS: {target_crs}")
    print(f"AOI bbox in 4326 (with 0.01° buffer): {bbox}")

    clip_layer(EXTRACT_DIR / ROADS_LAYER, "ROADS", ROADS_OUT, target_crs,
               drop_classes=ROADS_DROP_CLASSES)
    clip_layer(EXTRACT_DIR / BUILDINGS_LAYER, "BUILDINGS", BUILDINGS_OUT, target_crs)


if __name__ == "__main__":
    sys.exit(main() or 0)
