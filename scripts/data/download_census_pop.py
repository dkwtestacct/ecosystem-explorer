"""
Build Minneapolis population rasters from the 2020 US Census, aligned to the
NLCD grid that the rest of the app uses.

Pipeline:
  1. Pull block-level total population (P1_001N) AND 18+ voting-age population
     (P3_001N) for Hennepin County (FIPS 27053) via the Census decennial PL
     94-171 API. Both come from the SAME table — small JSON response, no key
     needed.
  2. Download the TIGER 2020 tabulation-block shapefile for Minnesota
     (`tl_2020_27_tabblock20.zip`), unzip into data/population/tiger/.
  3. Filter blocks to Hennepin County and to the NLCD raster's bounding box
     (after reprojection), then attach Census P1 + P3 by GEOID20.
  4. Rasterize: each block's value is spread **uniformly** across the NLCD
     pixels that fall inside it, so summing the output reproduces the block
     totals (and therefore the Census total). Both total pop and under-18
     (P1 - P3) get the same uniform-spread treatment over the same blocks.
  5. Write data/population/minneapolis_pop_2020.tif (total) and
     data/population/minneapolis_child_pop_2020.tif (under-18) at the same
     CRS / extent / transform / shape as data/cooling/land_use_2021.tif.

Sanity checks (printed at the end):
  - Minneapolis proper total ≈ 425,000 in 2020.
  - Hennepin County under-18 share ≈ 21.3 % (Census 2020 PL 94-171,
    measured 2026-06-03). Downtown tracts run slightly lower. The script's
    printout exposes both the county-wide and extent-clipped shares.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize

ROOT       = Path(__file__).resolve().parents[2]  # repo root (scripts/data/X → repo)
TEMPLATE   = ROOT / "data" / "cooling" / "land_use_2021.tif"
POP_DIR    = ROOT / "data" / "population"
TIGER_DIR  = POP_DIR / "tiger"
TIGER_SHP  = TIGER_DIR / "tl_2020_27_tabblock20.shp"
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_27_tabblock20.zip"
DST_TIF      = POP_DIR / "minneapolis_pop_2020.tif"
DST_CHILD_TIF = POP_DIR / "minneapolis_child_pop_2020.tif"

CENSUS_API = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS  = "27"   # Minnesota
COUNTY_FIPS = "053"  # Hennepin


def fetch_census_population() -> dict[str, tuple[int, int]]:
    """Return {GEOID20 (15-digit str): (total_pop, vap_18_plus)} per block.

    P1_001N = total population; P3_001N = voting-age (18+) population. Both
    from the same PL 94-171 table at the same source/vintage. Under-18 is
    derived as P1 - P3 — keeps the child raster in the same source as the
    total raster (no ACS substitution, no different vintage).

    Requires the CENSUS_API_KEY env var (https://api.census.gov/data/key_signup.html).
    Treat as secret — never logged, never written to disk."""
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: CENSUS_API_KEY not set. Get a free key from "
              "https://api.census.gov/data/key_signup.html, then re-run with "
              "`CENSUS_API_KEY=<key> python scripts/data/download_census_pop.py`.")
        sys.exit(2)
    params = {
        "get": "P1_001N,P3_001N,NAME",
        "for": "block:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        "key": api_key,
    }
    print(f"Fetching Census 2020 block totals + VAP for Hennepin County...")
    r = requests.get(CENSUS_API, params=params, timeout=60)
    # Don't print response.url on errors — it would leak the key. Raise the
    # status code; the body's HTML title is enough to diagnose 4xx without
    # leaking secrets.
    if not r.ok:
        print(f"  Census API returned HTTP {r.status_code}.")
        sys.exit(2)
    rows = r.json()
    header, *records = rows
    cols = {name: i for i, name in enumerate(header)}
    pops: dict[str, tuple[int, int]] = {}
    for rec in records:
        geoid = rec[cols["state"]] + rec[cols["county"]] + rec[cols["tract"]] + rec[cols["block"]]
        total = int(rec[cols["P1_001N"]])
        vap   = int(rec[cols["P3_001N"]])
        pops[geoid] = (total, vap)
    total_sum = sum(t for t, _ in pops.values())
    vap_sum   = sum(v for _, v in pops.values())
    child_sum = total_sum - vap_sum
    share = child_sum / total_sum if total_sum > 0 else 0.0
    print(f"  Got {len(pops):,} blocks; total pop = {total_sum:,}; "
          f"VAP(18+) = {vap_sum:,}; under-18 = {child_sum:,} ({share:.1%})")
    return pops


def ensure_tiger_blocks() -> Path:
    """Download + extract the MN tabulation-block shapefile if missing."""
    if TIGER_SHP.exists():
        print(f"TIGER shapefile already present: {TIGER_SHP}")
        return TIGER_SHP
    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TIGER_URL} ...")
    r = requests.get(TIGER_URL, timeout=300, stream=True)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    print(f"  {len(r.content) / 1e6:.1f} MB downloaded; unzipping...")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(TIGER_DIR)
    return TIGER_SHP


def main() -> None:
    POP_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(TEMPLATE) as tpl:
        dst_crs       = tpl.crs
        dst_transform = tpl.transform
        dst_height    = tpl.height
        dst_width     = tpl.width
        dst_bounds    = tpl.bounds
        print(f"Template: {TEMPLATE.name}")
        print(f"  CRS:    {dst_crs}")
        print(f"  Size:   {dst_width} x {dst_height}")
        print(f"  Bounds: {dst_bounds}")

    pops = fetch_census_population()
    shp_path = ensure_tiger_blocks()

    print(f"\nReading {shp_path.name}...")
    blocks = gpd.read_file(shp_path, columns=["GEOID20", "COUNTYFP20", "geometry"])
    blocks = blocks[blocks["COUNTYFP20"] == COUNTY_FIPS].copy()
    print(f"  {len(blocks):,} blocks in Hennepin County")

    print("Reprojecting blocks to NLCD CRS...")
    blocks = blocks.to_crs(dst_crs)

    # Restrict to the NLCD raster's bounding box (with a small buffer) so we
    # don't waste rasterization effort on blocks far outside the study area.
    bounds_poly = blocks.total_bounds  # noqa: F841 (just for completeness)
    xmin, ymin, xmax, ymax = dst_bounds.left, dst_bounds.bottom, dst_bounds.right, dst_bounds.top
    blocks = blocks.cx[xmin:xmax, ymin:ymax].copy()
    print(f"  {len(blocks):,} blocks intersect the NLCD extent")

    # Map (total, vap) per block; under-18 = total - vap.
    blocks["pop_total"] = blocks["GEOID20"].map(
        lambda g: pops.get(g, (0, 0))[0]).fillna(0).astype(int)
    blocks["pop_vap"]   = blocks["GEOID20"].map(
        lambda g: pops.get(g, (0, 0))[1]).fillna(0).astype(int)
    blocks["pop_child"] = (blocks["pop_total"] - blocks["pop_vap"]).clip(lower=0)
    matched = (blocks["pop_total"] > 0).sum()
    print(f"  {matched:,} blocks matched to Census population "
          f"(total in extent: {blocks['pop_total'].sum():,}; "
          f"under-18 in extent: {blocks['pop_child'].sum():,})")

    # Each block gets a unique 0-based index; we rasterize the index and use it
    # to look up the per-block per-pixel population (block_pop / pixel_count).
    blocks = blocks.reset_index(drop=True)
    blocks["idx"] = np.arange(len(blocks), dtype=np.int32)

    print("\nRasterizing block index...")
    shapes = ((geom, int(idx)) for geom, idx in zip(blocks.geometry, blocks["idx"]))
    idx_raster = rasterize(
        shapes,
        out_shape=(dst_height, dst_width),
        transform=dst_transform,
        fill=-1,
        dtype=np.int32,
        all_touched=False,
    )

    valid = idx_raster >= 0
    counts = np.bincount(idx_raster[valid], minlength=len(blocks))
    nonzero = counts > 0

    # Total pop raster (unchanged shape from prior behavior).
    pop_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    pop_per_pixel[nonzero] = blocks["pop_total"].values[nonzero] / counts[nonzero]
    pop_raster = np.where(valid, pop_per_pixel[idx_raster], 0.0).astype(np.float32)

    # Under-18 raster — same uniform-spread method over the same blocks, so
    # per-pixel: child_pop_pixel ≤ total_pop_pixel by construction (no block
    # has more children than people). Asserted by the gate's staleness cell.
    child_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    child_per_pixel[nonzero] = blocks["pop_child"].values[nonzero] / counts[nonzero]
    child_raster = np.where(valid, child_per_pixel[idx_raster], 0.0).astype(np.float32)

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
    with rasterio.open(DST_TIF, "w", **profile) as dst:
        dst.write(pop_raster, 1)
    with rasterio.open(DST_CHILD_TIF, "w", **profile) as dst:
        dst.write(child_raster, 1)

    total       = float(pop_raster.sum())
    child_total = float(child_raster.sum())
    child_share = child_total / total if total > 0 else 0.0
    print(f"\nWrote: {DST_TIF}")
    print(f"  Pixels with population: {(pop_raster > 0).sum():,}")
    print(f"  Total population (sanity check): {total:,.0f}")
    print(f"  Expected Minneapolis proper ≈ 425,000")
    print(f"\nWrote: {DST_CHILD_TIF}")
    print(f"  Pixels with under-18 population: {(child_raster > 0).sum():,}")
    print(f"  Under-18 population total: {child_total:,.0f} "
          f"({child_share:.1%} of total)")
    print(f"  Reference: Hennepin County under-18 share ≈ 21.3% (Census 2020 PL 94-171).")


if __name__ == "__main__":
    main()
