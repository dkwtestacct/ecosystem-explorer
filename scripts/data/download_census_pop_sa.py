"""Download Census 2020 population for Bexar County, Texas, rasterized to
the San Antonio NLCD grid. Adaptation of download_census_pop.py for SA.

Output: data/sa/population/sa_pop_2020.tif

Note on path: the brief's `data/sa/lulc_nlcd_2021_sa.tif` doesn't exist —
the canonical SA LULC raster lives at `data/sa/flood/land_use_2021_sa.tif`
(matches `CITIES['San Antonio, TX']['lulc_file']`). Using that as the
target template.

Population sanity check: legal San Antonio is ~1.4 M; full Bexar County
is ~2.0 M (2020 Census). Number landing in the LULC raster's bbox depends
on how the bbox was drawn.
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
TEMPLATE   = ROOT / "data" / "sa" / "flood" / "land_use_2021_sa.tif"
POP_DIR    = ROOT / "data" / "sa" / "population"
TIGER_DIR  = ROOT / "data" / "population" / "tiger"
TIGER_SHP  = TIGER_DIR / "tl_2020_48_tabblock20.shp"
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_48_tabblock20.zip"
DST_TIF    = POP_DIR / "sa_pop_2020.tif"

CENSUS_API = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS  = "48"   # Texas
COUNTY_FIPS = "029"  # Bexar (San Antonio)


def fetch_census_population():
    params = {
        "get": "P1_001N,NAME",
        "for": "block:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    print("Fetching Census 2020 block populations for Bexar County, TX...")
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
    print(f"  (Texas state-wide TABBLOCK file is ~600 MB compressed — slower than MN)")
    r = requests.get(TIGER_URL, timeout=900, stream=True)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    print(f"  {len(r.content) / 1e6:.1f} MB downloaded; unzipping...")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(TIGER_DIR)
    return TIGER_SHP


def main():
    POP_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE.exists():
        print(f"ERROR: template raster {TEMPLATE} not found. Run download_sa_data.py first.")
        return 1

    with rasterio.open(TEMPLATE) as tpl:
        dst_crs       = tpl.crs
        dst_transform = tpl.transform
        dst_height    = tpl.height
        dst_width     = tpl.width
        dst_bounds    = tpl.bounds
        print(f"Template: {TEMPLATE}")
        print(f"  CRS:    {dst_crs}")
        print(f"  Size:   {dst_width} x {dst_height}")
        print(f"  Bounds: {dst_bounds}")

    pops = fetch_census_population()
    shp_path = ensure_tiger_blocks()

    print(f"\nReading {shp_path.name}...")
    blocks = gpd.read_file(shp_path, columns=["GEOID20", "COUNTYFP20", "geometry"])
    blocks = blocks[blocks["COUNTYFP20"] == COUNTY_FIPS].copy()
    print(f"  {len(blocks):,} blocks in Bexar County")

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
        shapes, out_shape=(dst_height, dst_width), transform=dst_transform,
        fill=-1, dtype=np.int32, all_touched=False,
    )

    valid = idx_raster >= 0
    counts = np.bincount(idx_raster[valid], minlength=len(blocks))
    pop_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    nonzero = counts > 0
    pop_per_pixel[nonzero] = blocks["pop"].values[nonzero] / counts[nonzero]
    pop_raster = np.where(valid, pop_per_pixel[idx_raster], 0.0).astype(np.float32)

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": dst_width, "height": dst_height,
        "crs": dst_crs, "transform": dst_transform,
        "nodata": -9999.0, "compress": "deflate", "tiled": True,
    }
    with rasterio.open(DST_TIF, "w", **profile) as dst:
        dst.write(pop_raster, 1)

    total = float(pop_raster.sum())
    print(f"\nWrote: {DST_TIF} ({DST_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  Pixels with population: {(pop_raster > 0).sum():,}")
    print(f"  Total population in raster: {total:,.0f}")
    print(f"  Reference: legal SA ~1,434,625 (2020); full Bexar Co ~2,009,324.")


if __name__ == "__main__":
    sys.exit(main() or 0)
