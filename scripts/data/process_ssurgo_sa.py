"""Rasterize the Bexar County SSURGO shapefile to the San Antonio NLCD
grid. Adaptation of process_ssurgo.py for SA.

Input:  data/sa/flood/ssurgo_bexar_hsg.shp  (6,090 polygons, EPSG:4326)
        data/sa/flood/land_use_2021_sa.tif  (template grid, EPSG:5070)

Output: data/sa/flood/soil_group_sa.tif  (uint8, 1=A, 2=B, 3=C, 4=D)

Bexar has zero missing-hydgrp polygons (verified by download_ssurgo_sa.py),
so the C-class default fill is unused in practice — but the same code path
preserves robustness against future updates.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

OUT_DIR     = Path("data/sa/flood")
SSURGO_SHP  = OUT_DIR / "ssurgo_bexar_hsg.shp"
TEMPLATE    = OUT_DIR / "land_use_2021_sa.tif"
OUT_TIF     = OUT_DIR / "soil_group_sa.tif"

DEFAULT_HSG_CODE = 3   # C-class fill for any hsg_code=0 / background pixels


def main():
    print(f"Reading SSURGO shapefile: {SSURGO_SHP}")
    g = gpd.read_file(SSURGO_SHP)
    print(f"  features: {len(g):,}, CRS: {g.crs}")

    n_filled = int((g["hsg_code"] == 0).sum())
    if n_filled:
        g.loc[g["hsg_code"] == 0, "hsg_code"] = DEFAULT_HSG_CODE
        print(f"  filled {n_filled:,} no-data polygons (hsg_code 0 → {DEFAULT_HSG_CODE} C-class)")
    else:
        print("  no missing hsg_code rows in this dataset")

    print("\n--- Pre-rasterization hsg_code distribution ---")
    print(g["hsg_code"].value_counts().sort_index().to_string())

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
        fill=DEFAULT_HSG_CODE,
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
        "nodata":    0,
        "compress":  "deflate",
        "tiled":     True,
    }
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(soil, 1)

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
