"""Download NLCD 2021 for San Antonio and compute baseline constants.

Fetches a clipped NLCD 2021 land-cover GeoTIFF from MRLC's WCS endpoint,
inspects it, and computes the preliminary baseline_cn, baseline_hm,
total developed pixel count, and pixel area (acres) needed for the
CITIES['San Antonio, TX'] block in app.py.

CN baseline uses CN_B as the default soil group until SSURGO is wired in.
HM baseline uses a CC = 0.6*shade + 0.2*albedo + 0.2*kc proxy until
the reference-ET raster is available.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds

SA_BBOX = {
    "west": -98.80,
    "east": -98.20,
    "south": 29.20,
    "north": 29.65,
}

# MRLC publishes NLCD via a single GeoServer WCS endpoint. The native CRS is
# EPSG:5070 (Albers CONUS) with axes X,Y in meters, so we transform the
# WGS84 bbox before subsetting.
WCS_URL = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
COVERAGE_ID = "mrlc_download__NLCD_2021_Land_Cover_L48"
NATIVE_CRS = "EPSG:5070"

OUT_DIR = "data/sa/flood"
RAW_PATH = os.path.join(OUT_DIR, "lulc_nlcd_2021_sa.tif")
CLIP_PATH = os.path.join(OUT_DIR, "land_use_2021_sa.tif")

CN_TABLE = "data/sa/flood/UFR_biophysical_table_SA.csv"
COOLING_TABLE = "data/sa/cooling/biophysical_table_urban_cooling_SA.csv"

DEVELOPED_CODES = {21, 22, 23, 24}
SQM_PER_ACRE = 4046.8564224


def download_nlcd():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Transform WGS84 bbox -> EPSG:5070 metres for WCS subsetting
    x_min, y_min, x_max, y_max = transform_bounds(
        "EPSG:4326",
        NATIVE_CRS,
        SA_BBOX["west"], SA_BBOX["south"],
        SA_BBOX["east"], SA_BBOX["north"],
    )
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": COVERAGE_ID,
        "subset": [
            f"X({x_min},{x_max})",
            f"Y({y_min},{y_max})",
        ],
        "format": "image/tiff",
    }
    print("Requesting NLCD 2021 from MRLC WCS...")
    print(f"  WGS84 bbox: {SA_BBOX}")
    print(f"  EPSG:5070 subset: X({x_min:.0f},{x_max:.0f})  Y({y_min:.0f},{y_max:.0f})")
    r = requests.get(WCS_URL, params=params, stream=True, timeout=300)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "tiff" not in ctype.lower() and "image" not in ctype.lower():
        body = r.content[:2000].decode("utf-8", errors="replace")
        raise RuntimeError(f"Unexpected content-type {ctype}:\n{body}")
    with open(RAW_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(RAW_PATH) / 1024 / 1024
    print(f"  wrote {RAW_PATH} ({size_mb:.1f} MB)")


def inspect_and_compute():
    with rasterio.open(RAW_PATH) as src:
        print("\n--- Raster metadata ---")
        print(f"CRS:        {src.crs}")
        print(f"Bounds:     {src.bounds}")
        print(f"Size:       {src.width} x {src.height} ({src.width * src.height:,} px)")
        print(f"Resolution: {src.res}")
        print(f"Dtype:      {src.dtypes[0]}")

        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            valid = arr[arr != nodata]
        else:
            valid = arr.ravel()

        # Pixel area in acres (project pixel center to a metric CRS for area)
        if src.crs and src.crs.is_projected:
            px_area_m2 = abs(src.res[0] * src.res[1])
        else:
            # Geographic CRS: estimate via bounds reprojected to EPSG:3857
            l, b, r_, t = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
            px_area_m2 = abs((r_ - l) * (t - b)) / (src.width * src.height)
        pixel_area_acres = px_area_m2 / SQM_PER_ACRE

        # Class counts
        codes, counts = np.unique(valid, return_counts=True)
        print("\n--- NLCD class counts ---")
        for c, n in zip(codes.tolist(), counts.tolist()):
            pct = 100.0 * n / valid.size
            print(f"  code {int(c):>3}: {int(n):>10,} px  ({pct:5.2f}%)")

        class_counts = dict(zip(codes.tolist(), counts.tolist()))
        total_valid = int(valid.size)
        total_developed = int(sum(class_counts.get(c, 0) for c in DEVELOPED_CODES))

    # Baseline CN (CN_B as default soil group)
    cn_df = pd.read_csv(CN_TABLE)
    cn_map = dict(zip(cn_df["lucode"].astype(int), cn_df["CN_B"].astype(float)))
    cn_sum, cn_n = 0.0, 0
    missing_cn = []
    for code, n in class_counts.items():
        c = int(code)
        if c in cn_map:
            cn_sum += cn_map[c] * n
            cn_n += n
        else:
            missing_cn.append(c)
    baseline_cn = cn_sum / cn_n if cn_n else float("nan")
    if missing_cn:
        print(f"\n  CN: no row for codes {missing_cn} (excluded from mean)")

    # Baseline HM (CC proxy: 0.6*shade + 0.2*albedo + 0.2*kc)
    cool_df = pd.read_csv(COOLING_TABLE)
    cool_df["cc"] = 0.6 * cool_df["shade"] + 0.2 * cool_df["albedo"] + 0.2 * cool_df["kc"]
    hm_map = dict(zip(cool_df["lucode"].astype(int), cool_df["cc"].astype(float)))
    hm_sum, hm_n = 0.0, 0
    missing_hm = []
    for code, n in class_counts.items():
        c = int(code)
        if c in hm_map:
            hm_sum += hm_map[c] * n
            hm_n += n
        else:
            missing_hm.append(c)
    baseline_hm = hm_sum / hm_n if hm_n else float("nan")
    if missing_hm:
        print(f"  HM: no row for codes {missing_hm} (excluded from mean)")

    return {
        "SA_BASELINE_CN": baseline_cn,
        "SA_BASELINE_HM": baseline_hm,
        "SA_TOTAL_DEVELOPED_PIXELS": total_developed,
        "SA_TOTAL_VALID_PIXELS": total_valid,
        "SA_PIXEL_AREA_ACRES": pixel_area_acres,
    }


def save_clipped():
    # WCS already returned the bbox-clipped coverage; copy as the canonical name.
    with rasterio.open(RAW_PATH) as src:
        profile = src.profile
        data = src.read()
    with rasterio.open(CLIP_PATH, "w", **profile) as dst:
        dst.write(data)
    print(f"\nWrote canonical clipped raster: {CLIP_PATH}")


def main():
    if not os.path.exists(RAW_PATH):
        download_nlcd()
    else:
        print(f"Using existing {RAW_PATH}")

    constants = inspect_and_compute()
    save_clipped()

    print("\n--- Baseline constants for CITIES['San Antonio, TX'] ---")
    print(f"SA_BASELINE_CN            = {constants['SA_BASELINE_CN']:.4f}")
    print(f"SA_BASELINE_HM            = {constants['SA_BASELINE_HM']:.4f}")
    print(f"SA_TOTAL_DEVELOPED_PIXELS = {constants['SA_TOTAL_DEVELOPED_PIXELS']:,}")
    print(f"SA_TOTAL_VALID_PIXELS     = {constants['SA_TOTAL_VALID_PIXELS']:,}")
    print(f"SA_PIXEL_AREA_ACRES       = {constants['SA_PIXEL_AREA_ACRES']:.6f}")
    print(
        "\nNote: CN uses CN_B (default soil group) until SSURGO is rasterized; "
        "HM is a shade/albedo/kc proxy until reference-ET is available."
    )


if __name__ == "__main__":
    sys.exit(main())
