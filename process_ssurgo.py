"""Rasterize the Hennepin County SSURGO shapefile to the expanded
Minneapolis NLCD grid.

Input:  data/minneapolis_expanded/ssurgo_hennepin_hsg.shp  (32,442 polygons, EPSG:4326)
        data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif  (374 × 607 grid template, EPSG:5070)

Output: data/minneapolis_expanded/soil_group_mpls_full.tif  (uint8, 1=A, 2=B, 3=C, 4=D)

The 9.3 % of polygons with `hsg_code = 0` (no SSURGO data — typically
Udorthents / engineered urban fill / open water) get reassigned to C (3)
per NRCS convention for unknown urban/disturbed soils. Storing as the
canonical 1–4 hsg integer matches the lookup `evaluate_scenario` already
uses (`cn_table[lulc_idx, soil]` with `soil ∈ {1,2,3,4}`).
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

OUT_DIR     = Path("data/minneapolis_expanded")
SSURGO_SHP  = OUT_DIR / "ssurgo_hennepin_hsg.shp"
TEMPLATE    = OUT_DIR / "lulc_nlcd_2021_mpls_full.tif"
OUT_TIF     = OUT_DIR / "soil_group_mpls_full.tif"

DEFAULT_HSG_CODE = 3  # NRCS convention: assume C-class for missing/urban-disturbed soils


def main():
    print(f"Reading SSURGO shapefile: {SSURGO_SHP}")
    g = gpd.read_file(SSURGO_SHP)
    print(f"  features: {len(g):,}, CRS: {g.crs}")

    # Fill no-data hsg_code = 0 → 3 (C-class) per NRCS convention
    n_filled = int((g["hsg_code"] == 0).sum())
    g.loc[g["hsg_code"] == 0, "hsg_code"] = DEFAULT_HSG_CODE
    print(f"  filled {n_filled:,} no-data polygons (hsg_code 0 → {DEFAULT_HSG_CODE} C-class)")

    print("\n--- Pre-rasterization hsg_code distribution ---")
    print(g["hsg_code"].value_counts().sort_index().to_string())

    # Open the NLCD template to get target grid params
    print(f"\nReading template raster: {TEMPLATE}")
    with rasterio.open(TEMPLATE) as tpl:
        dst_crs    = tpl.crs
        dst_xform  = tpl.transform
        dst_height = tpl.height
        dst_width  = tpl.width
        dst_bounds = tpl.bounds
        print(f"  template CRS:    {dst_crs}")
        print(f"  template size:   {dst_width} × {dst_height}")
        print(f"  template bounds: {dst_bounds}")

    # Reproject polygons into the template's CRS, then crop to template extent
    # (cuts down rasterization work — Hennepin extends 3-4× beyond the AOI bbox)
    print(f"\nReprojecting SSURGO polygons {g.crs} → {dst_crs} ...")
    g = g.to_crs(dst_crs)
    g = g.cx[dst_bounds.left : dst_bounds.right, dst_bounds.bottom : dst_bounds.top]
    print(f"  features intersecting AOI: {len(g):,}")

    print("\nRasterizing hsg_code field...")
    shapes = ((geom, int(code)) for geom, code in zip(g.geometry, g["hsg_code"]))
    soil = rasterize(
        shapes,
        out_shape=(dst_height, dst_width),
        transform=dst_xform,
        fill=DEFAULT_HSG_CODE,   # background pixels with no SSURGO polygon → C
        dtype="uint8",
        all_touched=False,
    )

    profile = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "count":     1,
        "width":     dst_width,
        "height":    dst_height,
        "crs":       dst_crs,
        "transform": dst_xform,
        "nodata":    0,            # 0 = nodata sentinel (we never write 0 — DEFAULT_HSG_CODE fills gaps)
        "compress":  "deflate",
        "tiled":     True,
    }
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(soil, 1)

    # Coverage stats
    total = soil.size
    print(f"\nWrote: {OUT_TIF} ({OUT_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  shape: {soil.shape}, dtype: {soil.dtype}")
    print(f"  pixel-level hsg distribution:")
    for code, group in [(1, "A"), (2, "B"), (3, "C"), (4, "D")]:
        n = int((soil == code).sum())
        print(f"    code {code} ({group}): {n:>10,} px ({100 * n / total:5.1f}%)")
    n_zero = int((soil == 0).sum())
    if n_zero > 0:
        print(f"    code 0 (none):  {n_zero:>10,} px (should be 0)")


if __name__ == "__main__":
    sys.exit(main())
