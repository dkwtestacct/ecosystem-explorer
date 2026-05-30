"""Download Geofabrik's Minnesota OSM extract and clip roads + buildings to
the Minneapolis model AOI.

Idempotent: re-running skips the download/unzip if the artifacts already
exist. The Geofabrik state file is ~508 MB compressed; first run takes a
few minutes on a typical connection.

Outputs (EPSG:26915, matching `data/cooling/land_use_2021.tif`):
  data/osm/minneapolis_roads.geojson
  data/osm/minneapolis_buildings.geojson
"""

import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pyproj
import requests
from shapely.geometry import box

OSM_URL  = "https://download.geofabrik.de/north-america/us/minnesota-latest-free.shp.zip"
OSM_DIR  = Path("data/osm")
ZIP_PATH = OSM_DIR / "minnesota.shp.zip"
EXTRACT_DIR = OSM_DIR / "minnesota"

# Minneapolis model AOI in EPSG:26915 (UTM zone 15N, metres). Matches the
# bounds of `data/cooling/land_use_2021.tif`.
AOI_CRS    = "EPSG:26915"
AOI_BOUNDS = (478_738.8, 4_969_314.3, 489_538.8, 4_979_994.3)  # (minx, miny, maxx, maxy)

# Geofabrik layer filenames (constant across state extracts)
ROADS_LAYER     = "gis_osm_roads_free_1.shp"
BUILDINGS_LAYER = "gis_osm_buildings_a_free_1.shp"

# Outputs
ROADS_OUT     = OSM_DIR / "minneapolis_roads.geojson"
BUILDINGS_OUT = OSM_DIR / "minneapolis_buildings.geojson"


def download_zip():
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print(f"  zip already present at {ZIP_PATH} "
              f"({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB), skipping download")
        return
    print(f"Downloading Minnesota OSM shapefiles from Geofabrik...")
    print(f"  URL: {OSM_URL}")
    with requests.get(OSM_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        next_pct = 10
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
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


def aoi_bbox_4326():
    """Project the EPSG:26915 AOI bounds into EPSG:4326 with a small buffer
    so the GDAL bbox-filter reads cover the full rectangle even after the
    coordinate axes rotate slightly."""
    transformer = pyproj.Transformer.from_crs(AOI_CRS, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(AOI_BOUNDS[0], AOI_BOUNDS[1])
    maxx, maxy = transformer.transform(AOI_BOUNDS[2], AOI_BOUNDS[3])
    # 0.01° ≈ 1.1 km buffer; safe given UTM-vs-lonlat axis tilt
    return (minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01)


# Road classes excluded from the rasterized non-convertible mask. These are
# physical surfaces narrower than one 30 m NLCD pixel, so flagging the whole
# pixel as non-convertible because (e.g.) a 1.5 m sidewalk crosses it would
# substantially overstate the unconvertible fraction. We also drop the
# unclassified + track* classes (low-quality data, often actually convertible
# rural lanes).
ROADS_DROP_CLASSES = (
    "footway", "cycleway", "steps", "service", "path", "pedestrian",
    "unclassified", "track", "track_grade1", "track_grade2",
    "track_grade3", "track_grade4", "track_grade5",
)


def clip_layer(layer_path, label, out_path, drop_classes=None):
    bbox = aoi_bbox_4326()
    print(f"\n{label}: reading {layer_path.name} with bbox filter {bbox} ...")
    gdf = gpd.read_file(layer_path, bbox=bbox, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    print(f"  bbox-filtered features: {len(gdf):,} (in {gdf.crs})")
    gdf = gdf.to_crs(AOI_CRS)
    aoi_rect = box(*AOI_BOUNDS)
    clipped = gdf[gdf.intersects(aoi_rect)].copy()
    print(f"  intersects AOI rectangle: {len(clipped):,}")
    if drop_classes and "fclass" in clipped.columns:
        before = len(clipped)
        clipped = clipped[~clipped["fclass"].isin(drop_classes)].copy()
        print(f"  after dropping classes {drop_classes}: {len(clipped):,} "
              f"(removed {before - len(clipped):,})")
    geom_types = clipped.geom_type.value_counts().to_dict()
    print(f"  geometry types: {geom_types}")
    if "fclass" in clipped.columns:
        print(f"  fclass counts: {dict(clipped['fclass'].value_counts())}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    clipped.to_file(out_path, driver="GeoJSON")
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{len(clipped):,} features)")
    return clipped


def main():
    download_zip()
    extract_zip()
    clip_layer(EXTRACT_DIR / ROADS_LAYER,     "ROADS",     ROADS_OUT,
               drop_classes=ROADS_DROP_CLASSES)
    clip_layer(EXTRACT_DIR / BUILDINGS_LAYER, "BUILDINGS", BUILDINGS_OUT)


if __name__ == "__main__":
    sys.exit(main())
