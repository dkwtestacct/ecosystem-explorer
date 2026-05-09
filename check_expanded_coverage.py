"""Two coverage checks for the expanded Minneapolis NLCD raster:

1. Does the new raster (374 × 607 px, EPSG:5070) cover the legal Minneapolis
   city boundary? Pulls TIGER 2021 place polygons, filters to PLACEFP=43000,
   reprojects to EPSG:5070, computes overlap.

2. Does the existing soil-group raster (data/flood/soil_group_MN.tif) cover
   the new raster extent? Flood CN calculations need a soil-group value at
   every developed pixel, so any uncovered area would silently produce NaN.
"""

import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds
from shapely.geometry import box

NEW_RASTER  = Path("data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif")
SOIL_RASTER = Path("data/flood/soil_group_MN.tif")
TIGER_URL   = "https://www2.census.gov/geo/tiger/TIGER2021/PLACE/tl_2021_27_place.zip"
TIGER_DIR   = Path("data/population/tiger")  # already gitignored
TIGER_ZIP   = TIGER_DIR / "tl_2021_27_place.zip"
TIGER_SHP   = TIGER_DIR / "tl_2021_27_place.shp"
MPLS_PLACEFP = "43000"


def download_tiger():
    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    if TIGER_SHP.exists():
        print(f"  using cached {TIGER_SHP}")
        return
    print(f"Downloading TIGER 2021 places for MN ({TIGER_URL})...")
    r = requests.get(TIGER_URL, timeout=120)
    r.raise_for_status()
    TIGER_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(TIGER_ZIP) as z:
        z.extractall(TIGER_DIR)
    print(f"  extracted to {TIGER_DIR}")


def check_city_coverage():
    print("\n=== Check 1: Minneapolis city boundary coverage ===")
    download_tiger()
    places = gpd.read_file(TIGER_SHP)
    mpls = places[places["PLACEFP"] == MPLS_PLACEFP]
    if len(mpls) == 0:
        print(f"  ERROR: no place with PLACEFP={MPLS_PLACEFP}")
        return
    mpls = mpls.to_crs("EPSG:5070")
    mpls_geom = mpls.geometry.unary_union
    mpls_area_km2 = mpls_geom.area / 1_000_000

    with rasterio.open(NEW_RASTER) as src:
        rb = src.bounds
        raster_box = box(rb.left, rb.bottom, rb.right, rb.top)
    raster_area_km2 = raster_box.area / 1_000_000

    inter = mpls_geom.intersection(raster_box)
    inter_km2 = inter.area / 1_000_000
    pct_city_covered = 100 * inter_km2 / mpls_area_km2
    pct_raster_in_city = 100 * inter_km2 / raster_area_km2

    print(f"  Minneapolis city area:        {mpls_area_km2:7.1f} km²")
    print(f"  Expanded raster bbox area:    {raster_area_km2:7.1f} km²")
    print(f"  Intersection area:            {inter_km2:7.1f} km²")
    print(f"  % of CITY covered by raster:  {pct_city_covered:5.1f} %")
    print(f"  % of RASTER inside city:      {pct_raster_in_city:5.1f} %")
    if pct_city_covered < 95:
        missing = mpls_area_km2 - inter_km2
        print(f"  ⚠ {missing:.1f} km² ({100 - pct_city_covered:.1f}%) of city is OUTSIDE raster.")
    else:
        print(f"  ✓ Raster covers ≥95% of legal city boundary.")
    if pct_raster_in_city < 95:
        outside = raster_area_km2 - inter_km2
        print(f"  ⚠ {outside:.1f} km² of raster is OUTSIDE city limits "
              f"(suburbs / lakes / outer suburbs included)")


def check_soil_coverage():
    print("\n=== Check 2: Soil-group raster coverage ===")
    if not SOIL_RASTER.exists():
        print(f"  ERROR: {SOIL_RASTER} not found")
        return
    with rasterio.open(NEW_RASTER) as new, rasterio.open(SOIL_RASTER) as soil:
        # Reproject the soil raster's bounds into the new raster's CRS
        soil_in_new = transform_bounds(soil.crs, new.crs, *soil.bounds)
        nb = new.bounds
        sb = soil_in_new
        new_box  = box(nb.left, nb.bottom, nb.right, nb.top)
        soil_box = box(sb[0], sb[1], sb[2], sb[3])

        new_area_km2  = new_box.area / 1_000_000
        soil_area_km2 = soil_box.area / 1_000_000
        inter = new_box.intersection(soil_box)
        inter_km2 = inter.area / 1_000_000
        pct_new_covered = 100 * inter_km2 / new_area_km2

        print(f"  Soil raster ({SOIL_RASTER}):")
        print(f"    CRS:      {soil.crs}")
        print(f"    Pixels:   {soil.width} × {soil.height}")
        print(f"    Bounds (in soil CRS):     {soil.bounds}")
        print(f"    Bounds (reproj to 5070):  {tuple(round(v) for v in soil_in_new)}")
        print(f"    Area:                     {soil_area_km2:.1f} km²")
        print()
        print(f"  Expanded NLCD raster:")
        print(f"    CRS:      {new.crs}")
        print(f"    Pixels:   {new.width} × {new.height}")
        print(f"    Area:     {new_area_km2:.1f} km²")
        print()
        print(f"  Intersection: {inter_km2:.1f} km²")
        print(f"  % of EXPANDED raster covered by soil: {pct_new_covered:5.1f} %")
        if pct_new_covered >= 99:
            print(f"  ✓ Soil raster fully covers the expanded extent.")
        elif pct_new_covered >= 80:
            uncovered = new_area_km2 - inter_km2
            print(f"  ⚠ {uncovered:.1f} km² ({100 - pct_new_covered:.1f}%) of expanded "
                  f"raster has no soil data — flood CN would be NaN there.")
        else:
            print(f"  ❌ Soil raster covers only {pct_new_covered:.0f}% of expanded "
                  f"extent — would need a new SSURGO download for full coverage.")


def main():
    check_city_coverage()
    check_soil_coverage()


if __name__ == "__main__":
    sys.exit(main())
