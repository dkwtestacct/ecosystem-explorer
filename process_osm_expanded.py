"""Re-clip the Geofabrik Minnesota OSM extract to the expanded Minneapolis
NLCD bounds (EPSG:5070). Mirrors `download_osm_minneapolis.py` but uses
the expanded raster's bounds and writes to `data/minneapolis_expanded/`.

Roads: same Option B class filter as the downtown view (drop sub-pixel
surfaces — footway/cycleway/steps/service/path/pedestrian/track/unclassified).

Output:
  data/minneapolis_expanded/roads_mpls_full.geojson
  data/minneapolis_expanded/buildings_mpls_full.geojson
"""

import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from download_osm_minneapolis import ROADS_DROP_CLASSES   # reuse Option B list

EXTRACT_DIR  = Path("data/osm/minnesota")
ROADS_LAYER     = "gis_osm_roads_free_1.shp"
BUILDINGS_LAYER = "gis_osm_buildings_a_free_1.shp"

TEMPLATE = Path("data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif")
OUT_DIR  = Path("data/minneapolis_expanded")
ROADS_OUT     = OUT_DIR / "roads_mpls_full.geojson"
BUILDINGS_OUT = OUT_DIR / "buildings_mpls_full.geojson"


def aoi_bbox_4326():
    with rasterio.open(TEMPLATE) as tpl:
        l, b, r, t = tpl.bounds
        crs = tpl.crs
    minx, miny, maxx, maxy = transform_bounds(crs, "EPSG:4326", l, b, r, t)
    return (minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01), crs


def clip_layer(layer_path, label, out_path, target_crs, drop_classes=None):
    bbox, _ = aoi_bbox_4326()
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
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB, {len(clipped):,} features)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox, target_crs = aoi_bbox_4326()
    print(f"Template CRS: {target_crs}")
    print(f"AOI bbox in 4326: {bbox}")

    if not (EXTRACT_DIR / ROADS_LAYER).exists():
        print(f"ERROR: {EXTRACT_DIR / ROADS_LAYER} missing — run download_osm_minneapolis.py first")
        return 1

    clip_layer(EXTRACT_DIR / ROADS_LAYER, "ROADS", ROADS_OUT, target_crs,
               drop_classes=ROADS_DROP_CLASSES)
    clip_layer(EXTRACT_DIR / BUILDINGS_LAYER, "BUILDINGS", BUILDINGS_OUT, target_crs)


if __name__ == "__main__":
    sys.exit(main() or 0)
