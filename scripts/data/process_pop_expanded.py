"""Re-rasterize Census 2020 block-level population to the expanded
Minneapolis NLCD grid.

Mirrors `download_census_pop.py` but with a different target grid
(EPSG:5070, 374 × 607 covering ~204 km² vs the existing 360 × 356 ~123 km²
EPSG:26915 grid). Census API call is unchanged — Hennepin County FIPS
27053 block-level totals from `dec/pl P1_001N`. The bbox extends slightly
beyond Hennepin into Ramsey, but Hennepin contains all of Minneapolis
proper plus most of the buffer; population from outside Hennepin will be
0 in the output (acceptable since the legal city is fully covered).

Output: data/minneapolis_expanded/pop_mpls_full.tif
"""

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize

ROOT       = Path(__file__).resolve().parent
TEMPLATE   = ROOT / "data" / "minneapolis_expanded" / "lulc_nlcd_2021_mpls_full.tif"
POP_DIR    = ROOT / "data" / "minneapolis_expanded"
TIGER_DIR  = ROOT / "data" / "population" / "tiger"
TIGER_SHP  = TIGER_DIR / "tl_2020_27_tabblock20.shp"
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_27_tabblock20.zip"
DST_TIF    = POP_DIR / "pop_mpls_full.tif"

CENSUS_API = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS  = "27"   # Minnesota
COUNTY_FIPS = "053"  # Hennepin (where Minneapolis sits)


def fetch_census_population():
    params = {
        "get": "P1_001N,NAME",
        "for": "block:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    print("Fetching Census 2020 block populations for Hennepin County...")
    r = requests.get(CENSUS_API, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    header, *records = rows
    cols = {name: i for i, name in enumerate(header)}
    pops = {}
    for rec in records:
        geoid = rec[cols["state"]] + rec[cols["county"]] + rec[cols["tract"]] + rec[cols["block"]]
        pops[geoid] = int(rec[cols["P1_001N"]])
    print(f"  Got {len(pops):,} blocks; total pop = {sum(pops.values()):,}")
    return pops


def ensure_tiger_blocks():
    if TIGER_SHP.exists():
        print(f"TIGER shapefile already present: {TIGER_SHP}")
        return TIGER_SHP
    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TIGER_URL} ...")
    r = requests.get(TIGER_URL, timeout=300, stream=True)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(TIGER_DIR)
    return TIGER_SHP


def main():
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

    print(f"Reprojecting blocks to template CRS ({dst_crs})...")
    blocks = blocks.to_crs(dst_crs)
    xmin, ymin, xmax, ymax = dst_bounds.left, dst_bounds.bottom, dst_bounds.right, dst_bounds.top
    blocks = blocks.cx[xmin:xmax, ymin:ymax].copy()
    print(f"  {len(blocks):,} blocks intersect the template extent")

    blocks["pop"] = blocks["GEOID20"].map(pops).fillna(0).astype(int)
    matched = (blocks["pop"] > 0).sum()
    print(f"  {matched:,} blocks matched to Census population "
          f"(total in extent: {blocks['pop'].sum():,})")

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
    print(f"\nWrote: {DST_TIF} ({DST_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  Pixels with population: {(pop_raster > 0).sum():,}")
    print(f"  Total population: {total:,.0f}")
    print(f"  Expected Minneapolis legal city ≈ 425,000 (2020 Census)")


if __name__ == "__main__":
    sys.exit(main())
