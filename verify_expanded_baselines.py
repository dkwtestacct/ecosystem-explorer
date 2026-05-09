"""Compute baseline constants for the 'Minneapolis Full, MN' city *without*
running the Streamlit app — needed because that city has `available=False`,
so the live BASELINE_CN / BASELINE_HM overrides at module load only run for
the currently-selected (active) city.

Mirrors the same lookups used in app.py:
  - BASELINE_CN: cn_table[lulc_idx, soil] mean over valid pixels
  - BASELINE_HM: Gaussian-smoothed CC = 0.6·shade + 0.2·albedo + 0.2·ETI mean
  - BASELINE_NDVI: per-NLCD synthetic proxy via app.py:compute_mean_ndvi
  - Population sanity check + AOI coverage report

These numbers are what the user wants to see before flipping `available=True`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

DATA_DIR = Path("data/minneapolis_expanded")
LULC_PATH = DATA_DIR / "lulc_nlcd_2021_mpls_full.tif"
SOIL_PATH = DATA_DIR / "soil_group_mpls_full.tif"
POP_PATH  = DATA_DIR / "pop_mpls_full.tif"
ROADS_PATH     = DATA_DIR / "roads_mpls_full.geojson"
BUILDINGS_PATH = DATA_DIR / "buildings_mpls_full.geojson"

CN_TABLE_PATH = Path("data/flood/UFR_biophysical_table_MN.csv")
COOLING_TABLE_PATH = Path("data/cooling/biophysical_table_urban_cooling_MN.csv")
ET_PATH = Path("data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/"
               "reference_evapotranspiration_annual.tif")

# Same constants as app.py
DEVELOPED_CODES = [21, 22, 23, 24]
CC_SIGMA_PX = 15  # 450 m / 30 m pixels
UHI_MAX_C = 2.05


def main():
    # --- Load LULC ---
    with rasterio.open(LULC_PATH) as src:
        lulc = src.read(1)
        lulc_crs = src.crs
        lulc_shape = lulc.shape
        lulc_total_px = lulc.size
    print(f"LULC: {LULC_PATH.name}  {lulc_shape}, CRS {lulc_crs}, {lulc_total_px:,} px")

    # --- Load soil ---
    with rasterio.open(SOIL_PATH) as src:
        soil = src.read(1)
    print(f"Soil: {SOIL_PATH.name}  {soil.shape}")

    # --- BASELINE_CN ---
    bio_cn = pd.read_csv(CN_TABLE_PATH)
    cn_by_soil = {
        int(row["lucode"]): {1: row["CN_A"], 2: row["CN_B"], 3: row["CN_C"], 4: row["CN_D"]}
        for _, row in bio_cn.iterrows()
    }
    valid_lucodes = sorted(cn_by_soil.keys())
    max_lucode = max(valid_lucodes + [int(lulc.max())])
    cn_table = np.zeros((max_lucode + 1, 5), dtype=np.float32)
    lucode_idx_arr = np.zeros(max_lucode + 1, dtype=np.int32)
    for i, lc in enumerate(valid_lucodes):
        lucode_idx_arr[lc] = i
    cn_table = np.zeros((len(valid_lucodes), 5), dtype=np.float32)
    for i, lc in enumerate(valid_lucodes):
        for sg in (1, 2, 3, 4):
            cn_table[i, sg] = cn_by_soil[lc][sg]

    soil_clamped = np.clip(soil, 1, 4)
    lulc_safe = np.clip(lulc, 0, max_lucode)
    lulc_idx = lucode_idx_arr[lulc_safe]
    cn_grid = cn_table[lulc_idx, soil_clamped]
    valid_cn = cn_grid[cn_grid > 0]
    baseline_cn = float(valid_cn.mean().round(2)) if valid_cn.size else float("nan")

    # --- BASELINE_HM (smoothed CC) ---
    bio_cool = pd.read_csv(COOLING_TABLE_PATH)
    shade_a  = np.full(max_lucode + 2, np.nan, dtype=np.float32)
    kc_a     = np.full(max_lucode + 2, np.nan, dtype=np.float32)
    albedo_a = np.full(max_lucode + 2, np.nan, dtype=np.float32)
    for _, r in bio_cool.iterrows():
        c = int(r["lucode"])
        if c <= max_lucode:
            shade_a[c]  = r["shade"]
            kc_a[c]     = r["kc"]
            albedo_a[c] = r["albedo"]

    # ET handling (mirrors app.py post-fix: nodata-aware mask before resize)
    with rasterio.open(ET_PATH) as et_src:
        et_raw = et_src.read(1).astype(float)
        et_nodata = et_src.nodata
    if et_nodata is not None:
        et_raw[et_raw == et_nodata] = np.nan
    et_raw[et_raw > 10_000] = np.nan
    et_raw[et_raw < 0]      = np.nan
    et_resized = resize(et_raw, lulc_shape, order=1, preserve_range=True)
    finite = np.isfinite(et_resized)
    et_resized = np.where(finite, et_resized, np.nanmedian(et_resized[finite]))
    max_et = float(et_resized.max()) if et_resized.max() > 0 else 1.0

    safe2 = np.clip(lulc, 0, max_lucode + 1)
    cc_raw = (
        0.6 * shade_a[safe2]
        + 0.2 * albedo_a[safe2]
        + 0.2 * (kc_a[safe2] * et_resized / max_et)
    )
    nan_mask = ~np.isfinite(cc_raw)
    fill = float(cc_raw[~nan_mask].mean()) if (~nan_mask).any() else 0.0
    cc_filled = np.where(nan_mask, fill, cc_raw)
    cc_smoothed = gaussian_filter(cc_filled.astype(np.float32),
                                   sigma=CC_SIGMA_PX, mode="nearest")
    cc_smoothed[nan_mask] = np.nan
    valid_cc = cc_smoothed[~np.isnan(cc_smoothed)]
    baseline_hm = float(valid_cc.mean().round(4)) if valid_cc.size else float("nan")

    # --- BASELINE_NDVI (synthetic proxy from app.py:compute_mean_ndvi) ---
    NDVI_BY_CODE = {  # transcribed from app.py
        90: 0.70, 41: 0.75, 24: 0.10, 23: 0.15, 22: 0.20, 21: 0.30,
    }
    DEFAULT_DEV = 0.25
    DEFAULT_NAT = 0.60
    ndvi_arr = np.where(np.isin(lulc, DEVELOPED_CODES), DEFAULT_DEV, DEFAULT_NAT).astype(np.float32)
    for code, val in NDVI_BY_CODE.items():
        ndvi_arr = np.where(lulc == code, val, ndvi_arr)
    baseline_ndvi = float(ndvi_arr.mean().round(4))

    # --- Population coverage ---
    with rasterio.open(POP_PATH) as src:
        pop = src.read(1)
    total_pop = float(pop[pop > 0].sum())
    pop_pixels = int((pop > 0).sum())

    # --- Developed pixel summary ---
    developed_count = int(np.isin(lulc, DEVELOPED_CODES).sum())
    developed_pct   = 100 * developed_count / lulc_total_px

    # --- Print ---
    print()
    print("=" * 70)
    print(f"  BASELINE CONSTANTS for 'Minneapolis Full, MN'")
    print("=" * 70)
    print(f"  AOI:              {lulc_shape[1]} × {lulc_shape[0]} = {lulc_total_px:,} pixels")
    print(f"  AOI area:         {lulc_total_px * 30 * 30 / 1e6:.1f} km² (compare 122.8 km² downtown)")
    print(f"  Developed pixels: {developed_count:,} ({developed_pct:.1f}% of AOI)")
    print()
    print(f"  BASELINE_CN:      {baseline_cn:.2f}     (downtown: 75.67)")
    print(f"  BASELINE_HM:      {baseline_hm:.4f}   (downtown: 0.1859)")
    print(f"  BASELINE_NDVI:    {baseline_ndvi:.4f}   (downtown: 0.2326)")
    print()
    print(f"  Population total: {total_pop:>10,.0f}   (downtown ~154K, expected full city ~425K)")
    print(f"  Population pxls:  {pop_pixels:>10,}")
    print(f"  Roads features:   {len(__import__('geopandas').read_file(ROADS_PATH)):,} segments")
    bldgs = __import__('geopandas').read_file(BUILDINGS_PATH)
    print(f"  Building features: {len(bldgs):,} polygons")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
