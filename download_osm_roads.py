"""
download_osm_roads.py — one-shot offline build of a Minneapolis road-network
shapefile from OpenStreetMap, used by app.py to exclude road pixels from the
convertible-land pool alongside building footprints.

Usage:
    python3 download_osm_roads.py

Output:
    data/osm/minneapolis_roads.geojson — a single buffered MultiPolygon in
    EPSG:26915. Stored as one feature rather than tens of thousands of
    individual road segments so the GeoJSON stays small enough to commit
    (under 10 MB) — the rasterization just needs the union, not per-segment
    attributes.

Choices:
    - `network_type='drive'` keeps motor-vehicle roads (highways, residential
      streets) and excludes pedestrian paths, bike trails, and service-only
      ways. The convertible-pixel exclusion is about not building on streets;
      foot paths through parks aren't blocking conversions.
    - 12 m buffer = half a 30 m NLCD pixel, so rasterization reliably catches
      each road centerline regardless of which pixel a thin OSM linestring
      lands in.
    - `buffer(..., resolution=4)` keeps the polygon vertex count down — a
      4-segment quarter-circle is plenty for raster snapping at 30 m.
    - `unary_union` collapses the per-segment polygons into one MultiPolygon
      before write, which dominates the size reduction.

The download hits the Overpass API once (a few seconds for Minneapolis), so
this is meant to run rarely. The GeoJSON is committed to the repo so the
running app stays offline-deterministic.
"""
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.ops import unary_union

OUT_DIR  = Path("data/osm")
OUT_PATH = OUT_DIR / "minneapolis_roads.geojson"
BUFFER_M = 12


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying Overpass API for Minneapolis roads (network_type='drive')...")
    G = ox.graph_from_place("Minneapolis, Minnesota, USA", network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    print(f"  Pulled {len(edges):,} edges (raw OSM linestrings)")

    print(f"Reprojecting to EPSG:26915 and buffering by {BUFFER_M} m...")
    edges_proj = edges.to_crs("EPSG:26915")
    buffered = edges_proj.geometry.buffer(BUFFER_M, resolution=4)

    print("Unioning per-segment polygons into a single MultiPolygon...")
    merged = unary_union(buffered.tolist())

    out_gdf = gpd.GeoDataFrame(geometry=[merged], crs="EPSG:26915")
    out_gdf.to_file(OUT_PATH, driver="GeoJSON")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(
        f"Saved {OUT_PATH} "
        f"({len(out_gdf):,} feature, {size_kb:,.0f} KB; "
        f"covers {len(edges):,} OSM road segments after dissolve)"
    )


if __name__ == "__main__":
    main()
