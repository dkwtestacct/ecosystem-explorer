"""
Build a Minneapolis population raster from the 2020 US Census, aligned to the
NLCD grid that the rest of the app uses.

Pipeline:
  1. Pull block-level population (P1_001N) for Hennepin County (FIPS 27053)
     via the Census decennial PL API — small JSON response, no key needed.
  2. Download the TIGER 2020 tabulation-block shapefile for Minnesota
     (`tl_2020_27_tabblock20.zip`), unzip into data/population/tiger/.
  3. Filter blocks to Hennepin County and to the NLCD raster's bounding box
     (after reprojection), then attach the Census population by GEOID20.
  4. Rasterize: each block's total population is spread **uniformly** across
     the NLCD pixels that fall inside it, so summing the output reproduces the
     block totals (and therefore the Census total).
  5. Write data/population/minneapolis_pop_2020.tif at the same CRS / extent /
     transform / shape as data/cooling/land_use_2021.tif.

Sanity check: Minneapolis proper ≈ 425,000 in 2020. If the NLCD extent covers
more of Hennepin County, the printed total will be higher.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize

ROOT       = Path(__file__).resolve().parent
TEMPLATE   = ROOT / "data" / "cooling" / "land_use_2021.tif"
POP_DIR    = ROOT / "data" / "population"
TIGER_DIR  = POP_DIR / "tiger"
TIGER_SHP  = TIGER_DIR / "tl_2020_27_tabblock20.shp"
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_27_tabblock20.zip"
DST_TIF    = POP_DIR / "minneapolis_pop_2020.tif"

CENSUS_API = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS  = "27"   # Minnesota
COUNTY_FIPS = "053"  # Hennepin


def fetch_census_population() -> dict[str, int]:
    """Return {GEOID20 (15-digit str): population (int)} for Hennepin blocks."""
    params = {
        "get": "P1_001N,NAME",
        "for": "block:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    print(f"Fetching Census 2020 block populations for Hennepin County...")
    r = requests.get(CENSUS_API, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    header, *records = rows
    cols = {name: i for i, name in enumerate(header)}
    pops: dict[str, int] = {}
    for rec in records:
        geoid = rec[cols["state"]] + rec[cols["county"]] + rec[cols["tract"]] + rec[cols["block"]]
        pops[geoid] = int(rec[cols["P1_001N"]])
    print(f"  Got {len(pops):,} blocks; total pop = {sum(pops.values()):,}")
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

    blocks["pop"] = blocks["GEOID20"].map(pops).fillna(0).astype(int)
    matched = (blocks["pop"] > 0).sum()
    print(f"  {matched:,} blocks matched to Census population "
          f"(total in extent: {blocks['pop'].sum():,})")

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
    pop_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    nonzero = counts > 0
    pop_per_pixel[nonzero] = blocks["pop"].values[nonzero] / counts[nonzero]

    pop_raster = np.where(valid, pop_per_pixel[idx_raster], 0.0).astype(np.float32)

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

    total = float(pop_raster.sum())
    print(f"\nWrote: {DST_TIF}")
    print(f"  Pixels with population: {(pop_raster > 0).sum():,}")
    print(f"  Total population (sanity check): {total:,.0f}")
    print(f"  Expected Minneapolis proper ≈ 425,000")


if __name__ == "__main__":
    main()
