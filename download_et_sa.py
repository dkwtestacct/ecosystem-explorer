"""Download CGIAR Global-AI/ET0 v3.1 annual reference evapotranspiration,
clip to the San Antonio AOI, and reproject to the SA NLCD grid.

CGIAR v3.1 is the current canonical reference dataset behind the InVEST
UCM sample ET raster (the InVEST docs cite v3, but v3.1 — released later —
is the same WorldClim 1970-2000 base, recomputed). Native resolution is
30 arc-seconds (≈ 1 km at the equator).

The brief's URL (https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/
wc2.1_30s_pet.zip) is invalid: WorldClim v2.1 doesn't host PET in its
standard distribution; PET must come from CGIAR's separate Aridity DB.

Output: data/sa/cooling/et_annual_sa.tif (uint16 mm/yr, EPSG:5070, matched
to the SA NLCD 1984×1713 grid)
"""

import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds as window_from_bounds

# CGIAR v3.1 annual ET0 — figshare download URL (figshare API confirmed).
# Sized 645 MB compressed; expect 5–10 min on a typical connection.
ZIP_URL = "https://ndownloader.figshare.com/files/56300327"
ZIP_NAME = "Global-AI_ET0__annual_v3_1.zip"

OUT_DIR    = Path("data/sa/cooling")
WORK_DIR   = OUT_DIR / "cgiar_et0"
ZIP_PATH   = WORK_DIR / ZIP_NAME
DST_TIF    = OUT_DIR / "et_annual_sa.tif"

TEMPLATE   = Path("data/sa/flood/land_use_2021_sa.tif")

# SA bbox in WGS84 (matches download_sa_data.py — generous around the AOI)
SA_BBOX_WGS84 = (-98.85, 29.15, -98.15, 29.70)  # west, south, east, north


def download_zip():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 600_000_000:
        print(f"  zip already present at {ZIP_PATH} "
              f"({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB), skipping")
        return
    print(f"Downloading CGIAR Global-AI/ET0 v3.1 annual ({ZIP_URL})...")
    print("  (~645 MB — expect 5–10 min)")
    with requests.get(ZIP_URL, stream=True, timeout=900) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written, next_pct = 0, 10
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = int(100 * written / total)
                    if pct >= next_pct:
                        print(f"    {pct:>3}% ({written / 1024 / 1024:.0f} / "
                              f"{total / 1024 / 1024:.0f} MB)")
                        next_pct = pct + 10
    print(f"  wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB)")


def extract_annual_tiff():
    """Find an annual ET0 raster inside the CGIAR zip and extract it."""
    print(f"\nInspecting zip contents...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = z.namelist()
        # CGIAR v3.1 annual zip should contain one .tif (et0_v3_yr.tif or similar)
        tif_members = [m for m in members if m.lower().endswith(".tif")]
        print(f"  {len(members)} files in zip, {len(tif_members)} TIFFs:")
        for m in tif_members[:5]:
            print(f"    {m}")
        if not tif_members:
            raise RuntimeError("No .tif found in CGIAR zip")
        target = tif_members[0]
        out_path = WORK_DIR / Path(target).name
        if out_path.exists() and out_path.stat().st_size > 1_000_000:
            print(f"  already extracted: {out_path}")
            return out_path
        print(f"  extracting {target} → {out_path}")
        with z.open(target) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def clip_and_warp(et_global_path):
    """Clip the global ET raster to the SA bbox (in WGS84), then warp to
    the SA NLCD grid (EPSG:5070, 30 m, 1984×1713)."""
    print(f"\nReading template: {TEMPLATE}")
    with rasterio.open(TEMPLATE) as tpl:
        dst_crs       = tpl.crs
        dst_transform = tpl.transform
        dst_height    = tpl.height
        dst_width     = tpl.width
        dst_bounds    = tpl.bounds
    print(f"  CRS: {dst_crs}, size: {dst_width} × {dst_height}, bounds: {dst_bounds}")

    print(f"\nReading global ET source: {et_global_path}")
    with rasterio.open(et_global_path) as src:
        print(f"  CRS: {src.crs}, native size: {src.width} × {src.height}, dtype: {src.dtypes[0]}")
        print(f"  global bounds: {src.bounds}")
        nodata = src.nodata
        print(f"  nodata: {nodata}")

        # Window read in source CRS — assume EPSG:4326 (CGIAR v3.1 standard).
        w, s, e, n = SA_BBOX_WGS84
        window = window_from_bounds(w, s, e, n, transform=src.transform)
        et_clip = src.read(1, window=window).astype("float32")
        clip_transform = src.window_transform(window)
        print(f"  clipped shape: {et_clip.shape}, "
              f"raw range: {et_clip.min():.0f} .. {et_clip.max():.0f}")

        if nodata is not None:
            et_clip[et_clip == nodata] = np.nan
        # CGIAR v3 packs ET0 as int / scale; values look like real mm/yr already
        # at this point. Clamp obvious nodata sentinels.
        et_clip[et_clip > 1e5] = np.nan
        et_clip[et_clip < 0]   = np.nan

        finite = np.isfinite(et_clip)
        if finite.any():
            print(f"  valid range (mm/yr): {et_clip[finite].min():.0f} .. {et_clip[finite].max():.0f}")
            print(f"  mean: {et_clip[finite].mean():.0f} mm/yr")

        # Warp to SA NLCD grid
        dst = np.full((dst_height, dst_width), np.nan, dtype="float32")
        reproject(
            source=et_clip,
            destination=dst,
            src_transform=clip_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    # Fill any leftover NaN with the field median so the InVEST UCM ETI
    # term doesn't propagate NaN.
    finite = np.isfinite(dst)
    if (~finite).any():
        fill = float(np.nanmedian(dst[finite])) if finite.any() else 0.0
        dst = np.where(finite, dst, fill)

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
    with rasterio.open(DST_TIF, "w", **profile) as out:
        out.write(dst.astype("float32"), 1)

    print(f"\nWrote: {DST_TIF} ({DST_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  shape: {dst.shape}, dtype: float32")
    print(f"  warped ET range (mm/yr): {dst.min():.0f} .. {dst.max():.0f}, "
          f"mean: {dst.mean():.0f}")
    print(f"  Reference: SA annual PET ~1,200–1,600 mm; Minneapolis ~600–800 mm.")


def main():
    if not TEMPLATE.exists():
        print(f"ERROR: template raster {TEMPLATE} not found. Run download_sa_data.py first.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_zip()
    et_global = extract_annual_tiff()
    clip_and_warp(et_global)


if __name__ == "__main__":
    sys.exit(main() or 0)
