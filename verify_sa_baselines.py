"""Compute baseline constants for the 'San Antonio, TX' city WITHOUT
running the Streamlit app. Mirrors verify_expanded_baselines.py for SA,
using the SA inputs:
  - data/sa/flood/land_use_2021_sa.tif       (LULC, EPSG:5070, 1984×1713)
  - data/sa/flood/soil_group_sa.tif          (SSURGO Bexar, uint8 1–4)
  - data/sa/cooling/et_annual_sa.tif         (CGIAR v3.1 mm/yr)
  - data/sa/population/sa_pop_2020.tif       (Census 2020 Bexar)
  - data/sa/flood/UFR_biophysical_table_SA.csv (CN per soil group)
  - data/sa/cooling/biophysical_table_urban_cooling_SA.csv (shade/Kc/albedo)

Reports BASELINE_CN, BASELINE_HM, BASELINE_NDVI, total population, and
developed pixel count, matching the metrics the user asked for.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

DATA_DIR  = Path("data/sa")
LULC_PATH = DATA_DIR / "flood" / "land_use_2021_sa.tif"
SOIL_PATH = DATA_DIR / "flood" / "soil_group_sa.tif"
ET_PATH   = DATA_DIR / "cooling" / "et_annual_sa.tif"
POP_PATH  = DATA_DIR / "population" / "sa_pop_2020.tif"

CN_TABLE_PATH      = DATA_DIR / "flood"   / "UFR_biophysical_table_SA.csv"
COOLING_TABLE_PATH = DATA_DIR / "cooling" / "biophysical_table_urban_cooling_SA.csv"

DEVELOPED_CODES = [21, 22, 23, 24]
CC_SIGMA_PX = 15        # 450 m / 30 m px (matches MN UCM convolution)
UHI_MAX_C   = 2.05      # MN value used as fallback until an SA-specific is sourced


def main():
    with rasterio.open(LULC_PATH) as src:
        lulc = src.read(1)
        lulc_crs = src.crs
        lulc_shape = lulc.shape
    print(f"LULC: {LULC_PATH.name}  {lulc_shape}, CRS {lulc_crs}, "
          f"{lulc.size:,} px")

    with rasterio.open(SOIL_PATH) as src:
        soil = src.read(1)
    print(f"Soil: {SOIL_PATH.name}  {soil.shape}")

    # ── BASELINE_CN ────────────────────────────────────────────────────────
    bio_cn = pd.read_csv(CN_TABLE_PATH)
    cn_by_soil = {
        int(row["lucode"]): {1: row["CN_A"], 2: row["CN_B"], 3: row["CN_C"], 4: row["CN_D"]}
        for _, row in bio_cn.iterrows()
    }
    valid_lucodes = sorted(cn_by_soil.keys())
    max_lucode = max(valid_lucodes + [int(lulc.max())])
    lucode_idx_arr = np.zeros(max_lucode + 1, dtype=np.int32)
    for i, lc in enumerate(valid_lucodes):
        lucode_idx_arr[lc] = i
    cn_table = np.zeros((len(valid_lucodes), 5), dtype=np.float32)
    for i, lc in enumerate(valid_lucodes):
        for sg in (1, 2, 3, 4):
            cn_table[i, sg] = cn_by_soil[lc][sg]

    soil_clamped = np.clip(soil, 1, 4)
    lulc_safe = np.clip(lulc, 0, max_lucode)
    cn_grid = cn_table[lucode_idx_arr[lulc_safe], soil_clamped]
    valid_cn = cn_grid[cn_grid > 0]
    baseline_cn = float(valid_cn.mean().round(2)) if valid_cn.size else float("nan")

    # ── BASELINE_HM (smoothed CC, full InVEST UCM) ─────────────────────────
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

    with rasterio.open(ET_PATH) as et_src:
        et_raw = et_src.read(1).astype(float)
        et_nodata = et_src.nodata
    if et_nodata is not None:
        et_raw[et_raw == et_nodata] = np.nan
    et_raw[et_raw > 1e5] = np.nan
    et_raw[et_raw < 0]   = np.nan
    finite = np.isfinite(et_raw)
    if et_raw.shape != lulc_shape:
        # Already pre-warped to NLCD grid by download_et_sa.py — but resample
        # defensively in case shapes differ.
        et_resized = resize(et_raw, lulc_shape, order=1, preserve_range=True)
        finite_r = np.isfinite(et_resized)
        et_resized = np.where(finite_r, et_resized, np.nanmedian(et_resized[finite_r]))
    else:
        et_resized = np.where(finite, et_raw, np.nanmedian(et_raw[finite]))
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

    # ── BASELINE_NDVI (synthetic NDVI proxy) ───────────────────────────────
    NDVI_BY_CODE = {  # mirrors app.py
        90: 0.70, 41: 0.75, 24: 0.10, 23: 0.15, 22: 0.20, 21: 0.30,
    }
    DEFAULT_DEV = 0.25
    DEFAULT_NAT = 0.60
    ndvi_arr = np.where(np.isin(lulc, DEVELOPED_CODES), DEFAULT_DEV, DEFAULT_NAT).astype(np.float32)
    for code, val in NDVI_BY_CODE.items():
        ndvi_arr = np.where(lulc == code, val, ndvi_arr)
    baseline_ndvi = float(ndvi_arr.mean().round(4))

    # ── Population coverage ────────────────────────────────────────────────
    with rasterio.open(POP_PATH) as src:
        pop = src.read(1)
    total_pop = float(pop[pop > 0].sum())
    pop_pixels = int((pop > 0).sum())

    # ── Developed pixel summary ────────────────────────────────────────────
    developed_count = int(np.isin(lulc, DEVELOPED_CODES).sum())
    developed_pct   = 100 * developed_count / lulc.size

    # ── Print ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  BASELINE CONSTANTS for 'San Antonio, TX'")
    print("=" * 70)
    print(f"  AOI:              {lulc_shape[1]} × {lulc_shape[0]} = {lulc.size:,} pixels")
    print(f"  AOI area:         {lulc.size * 30 * 30 / 1e6:.1f} km²")
    print(f"  Developed pixels: {developed_count:,} ({developed_pct:.1f}% of AOI)")
    print()
    print(f"  BASELINE_CN:      {baseline_cn:.2f}")
    print(f"  BASELINE_HM:      {baseline_hm:.4f}")
    print(f"  BASELINE_NDVI:    {baseline_ndvi:.4f}")
    print()
    print(f"  Population total: {total_pop:>11,.0f}")
    print(f"  Population pxls:  {pop_pixels:>11,}")
    print()
    print("  Comparison cities:")
    print(f"    Mpls downtown:  CN 75.67 | HM 0.1859 | NDVI 0.2326 | pop ~154K")
    print(f"    Mpls Full:      CN 77.68 | HM 0.1600 | NDVI 0.2072 | pop ~464K")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
