"""
Reproject and clip the USA WorldPop 2020 raster to match the Minneapolis NLCD raster
exactly — same CRS, extent, resolution, and pixel alignment.

Inputs:
  data/population/usa_pop_2020.tif  (source, EPSG:4326)
  data/cooling/land_use_2021.tif    (template; CRS/extent/resolution/transform)

Output:
  data/population/minneapolis_pop_2020.tif

Units: WorldPop "ppp" (people-per-pixel) layers are per-pixel **counts**, not
density. The reprojection below uses bilinear resampling and rescales by the
source/destination pixel-area ratio, which preserves total population (counts
in → counts out). If you ever swap in a density layer (people/km²) instead,
multiply by `(PIXEL_SIZE_M ** 2) / 1_000_000` after reprojection to convert to
counts per 30 m pixel before saving.

Sanity check: Minneapolis proper population ~425,000 in 2020.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parent
SRC_POP  = ROOT / "data" / "population" / "usa_pop_2020.tif"
TEMPLATE = ROOT / "data" / "cooling" / "land_use_2021.tif"
DST_POP  = ROOT / "data" / "population" / "minneapolis_pop_2020.tif"


def main():
    with rasterio.open(TEMPLATE) as tpl:
        dst_crs       = tpl.crs
        dst_transform = tpl.transform
        dst_width     = tpl.width
        dst_height    = tpl.height
        print(f"Template: {TEMPLATE.name}")
        print(f"  CRS:        {dst_crs}")
        print(f"  Size:       {dst_width} x {dst_height}")
        print(f"  Resolution: {dst_transform.a:.3f}, {-dst_transform.e:.3f}")
        print(f"  Bounds:     {tpl.bounds}")

    # WorldPop uses NoData near -3.4e38; we read it raw and let reproject handle it.
    with rasterio.open(SRC_POP) as src:
        src_nodata = src.nodata
        print(f"\nSource: {SRC_POP.name}")
        print(f"  CRS:    {src.crs}")
        print(f"  Size:   {src.width} x {src.height}")
        print(f"  NoData: {src_nodata}")

        # Population counts must be summed, not averaged. WorldPop is already a
        # per-pixel count, so we reproject with bilinear resampling for now and
        # rescale by the area ratio to preserve total population. For a strict
        # population-conserving warp, sum-based resampling would be ideal but
        # rasterio's warp does not expose a sum kernel — bilinear with area
        # rescaling is the standard workaround for WorldPop downscaling.
        dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # Rescale so that total population is preserved across the projection change.
    # Source pixel area in m^2 (assume EPSG:4326 → approximate via cosine of mean lat).
    with rasterio.open(SRC_POP) as src:
        src_pixel_deg = abs(src.transform.a)  # ~0.000833° (3 arc-sec)
        # Mean latitude of destination bounds, transformed to WGS84 for cosine factor.
        from rasterio.warp import transform_bounds
        l, b, r, t = transform_bounds(dst_crs, "EPSG:4326",
                                       *rasterio.open(TEMPLATE).bounds)
        mean_lat = np.deg2rad((t + b) / 2.0)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(mean_lat)
        src_pixel_area_m2 = (src_pixel_deg * m_per_deg_lat) * (src_pixel_deg * m_per_deg_lon)

    dst_pixel_area_m2 = abs(dst_transform.a * dst_transform.e)
    area_ratio = dst_pixel_area_m2 / src_pixel_area_m2
    dst_arr = dst_arr * area_ratio

    # Replace NaNs with 0 for clean integer-like population output.
    dst_arr = np.where(np.isnan(dst_arr), 0.0, dst_arr).astype(np.float32)

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     1,
        "width":     dst_width,
        "height":    dst_height,
        "crs":       dst_crs,
        "transform": dst_transform,
        "nodata":    -9999.0,
        "compress":  "deflate",
        "tiled":     True,
    }
    DST_POP.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(DST_POP, "w", **profile) as dst:
        dst.write(dst_arr, 1)

    total_pop = float(dst_arr.sum())
    print(f"\nWrote: {DST_POP}")
    print(f"  Pixel area ratio (dst/src): {area_ratio:.4f}")
    print(f"  Total population (sanity check): {total_pop:,.0f}")
    print(f"  Expected Minneapolis proper ≈ 425,000")


if __name__ == "__main__":
    main()
