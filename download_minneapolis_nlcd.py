"""Download an expanded Minneapolis NLCD extent from MRLC's WCS endpoint.

The current `data/cooling/land_use_2021.tif` covers only ~10.8 × 10.7 km of
downtown / inner-ring Minneapolis. This script pulls the full city + a
buffer (~151 km², the legal city limits) for use in any future analysis
that needs city-wide coverage.

Notes on the MRLC API (corrected vs the snippet in the task):
- Single GeoServer WCS endpoint at `/geoserver/mrlc_download/wcs`
  (not `/.../layer/ows`).
- Coverage ID is `mrlc_download__NLCD_2021_Land_Cover_L48` (double underscore).
- Native CRS is EPSG:5070 (NAD83 / Conus Albers, metres). Axes are X/Y in
  metres, not Long/Lat in degrees — we transform the bbox before subsetting.
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds

# Generous Minneapolis bbox covering the legal city limits + a small ring.
MINNEAPOLIS_BBOX = {
    "west":  -93.329,
    "east":  -93.193,
    "south":  44.890,
    "north":  45.051,
}

WCS_URL      = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
COVERAGE_ID  = "mrlc_download__NLCD_2021_Land_Cover_L48"
NATIVE_CRS   = "EPSG:5070"

OUT_DIR  = Path("data/minneapolis_expanded")
OUT_PATH = OUT_DIR / "lulc_nlcd_2021_mpls_full.tif"

EXISTING_PATH = Path("data/cooling/land_use_2021.tif")

DEVELOPED_CODES = {21, 22, 23, 24}
SQM_PER_KM2     = 1_000_000
NLCD_NAMES = {
    11: "Open Water",                      21: "Developed, Open Space",
    22: "Developed, Low Intensity",        23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",       31: "Barren Land",
    41: "Deciduous Forest",                42: "Evergreen Forest",
    43: "Mixed Forest",                    52: "Shrub/Scrub",
    71: "Herbaceous",                      81: "Hay/Pasture",
    82: "Cultivated Crops",                90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 100_000:
        print(f"  raster already at {OUT_PATH} "
              f"({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB), skipping download")
        return

    x_min, y_min, x_max, y_max = transform_bounds(
        "EPSG:4326", NATIVE_CRS,
        MINNEAPOLIS_BBOX["west"],  MINNEAPOLIS_BBOX["south"],
        MINNEAPOLIS_BBOX["east"],  MINNEAPOLIS_BBOX["north"],
    )
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": COVERAGE_ID,
        "subset": [f"X({x_min},{x_max})", f"Y({y_min},{y_max})"],
        "format": "image/tiff",
    }
    print("Requesting NLCD 2021 from MRLC WCS...")
    print(f"  WGS84 bbox: {MINNEAPOLIS_BBOX}")
    print(f"  EPSG:5070 subset: X({x_min:.0f},{x_max:.0f})  Y({y_min:.0f},{y_max:.0f})")
    r = requests.get(WCS_URL, params=params, stream=True, timeout=300)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "tiff" not in ctype.lower() and "image" not in ctype.lower():
        body = r.content[:2000].decode("utf-8", errors="replace")
        raise RuntimeError(f"Unexpected content-type {ctype}:\n{body}")
    with open(OUT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def inspect():
    with rasterio.open(OUT_PATH) as src:
        print("\n--- Raster metadata ---")
        print(f"CRS:        {src.crs}")
        print(f"Bounds:     {src.bounds}")
        print(f"Size:       {src.width} x {src.height} ({src.width * src.height:,} px)")
        print(f"Resolution: {src.res}")
        print(f"Dtype:      {src.dtypes[0]}")
        arr = src.read(1)
        nodata = src.nodata
        valid = arr[arr != nodata] if nodata is not None else arr.ravel()

        # Pixel area in m² and km²
        if src.crs and src.crs.is_projected:
            px_area_m2 = abs(src.res[0] * src.res[1])
        else:
            l, b, r_, t = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
            px_area_m2 = abs((r_ - l) * (t - b)) / (src.width * src.height)

        codes, counts = np.unique(valid, return_counts=True)
        print("\n--- NLCD class counts ---")
        for c, n in zip(codes.tolist(), counts.tolist()):
            pct = 100.0 * n / valid.size
            label = NLCD_NAMES.get(int(c), "?")
            print(f"  code {int(c):>3} ({label:<30s}) {int(n):>10,} px  ({pct:5.2f}%)")

        total_dev = int(sum(int(n) for c, n in zip(codes, counts) if int(c) in DEVELOPED_CODES))
        total_valid = int(valid.size)
        total_area_km2 = total_valid * px_area_m2 / SQM_PER_KM2
        dev_area_km2   = total_dev   * px_area_m2 / SQM_PER_KM2

        print("\n--- Coverage summary ---")
        print(f"Pixel area:              {px_area_m2:,.0f} m²")
        print(f"Total valid pixels:      {total_valid:,}  ≈ {total_area_km2:.1f} km²")
        print(f"Developed pixels (21–24): {total_dev:,}  ≈ {dev_area_km2:.1f} km² "
              f"({100*total_dev/total_valid:.1f}% of AOI)")

    return src


def compare_to_existing():
    print("\n--- Extent comparison ---")
    if not EXISTING_PATH.exists():
        print(f"  (existing raster {EXISTING_PATH} not found)")
        return
    with rasterio.open(OUT_PATH) as new, rasterio.open(EXISTING_PATH) as old:
        # Reproject old bounds into the new CRS for apples-to-apples comparison
        old_in_new_crs = transform_bounds(old.crs, new.crs, *old.bounds)
        nl, nb, nr, nt = new.bounds
        old_w_m = old_in_new_crs[2] - old_in_new_crs[0]
        old_h_m = old_in_new_crs[3] - old_in_new_crs[1]
        new_w_m = nr - nl
        new_h_m = nt - nb
        print(f"  EXISTING ({EXISTING_PATH}):")
        print(f"    CRS:     {old.crs}")
        print(f"    Pixels:  {old.width} x {old.height}  ({old.width * old.height:,})")
        print(f"    Extent:  ~{old_w_m / 1000:.1f} km × {old_h_m / 1000:.1f} km "
              f"(~{old_w_m * old_h_m / SQM_PER_KM2:.1f} km²)")
        print(f"  NEW ({OUT_PATH}):")
        print(f"    CRS:     {new.crs}")
        print(f"    Pixels:  {new.width} x {new.height}  ({new.width * new.height:,})")
        print(f"    Extent:  ~{new_w_m / 1000:.1f} km × {new_h_m / 1000:.1f} km "
              f"(~{new_w_m * new_h_m / SQM_PER_KM2:.1f} km²)")
        ratio_area = (new_w_m * new_h_m) / (old_w_m * old_h_m)
        ratio_px   = (new.width * new.height) / (old.width * old.height)
        print(f"  AREA RATIO (new/old): {ratio_area:.2f}×")
        print(f"  PIXEL-COUNT RATIO:    {ratio_px:.2f}×")


def population_coverage_note():
    print("\n--- Population coverage ---")
    print("  The current `data/population/minneapolis_pop_2020.tif` was rasterized")
    print("  from Census 2020 block totals for Hennepin County to match the existing")
    print("  360 × 356 cooling LULC grid (EPSG:26915). Using this expanded NLCD as")
    print("  the model AOI would require re-rasterizing population to the new grid")
    print("  via an updated `download_census_pop.py` (Census API call is unchanged;")
    print("  only the target raster's CRS / transform / shape changes).")


def main():
    download()
    inspect()
    compare_to_existing()
    population_coverage_note()


if __name__ == "__main__":
    sys.exit(main())
