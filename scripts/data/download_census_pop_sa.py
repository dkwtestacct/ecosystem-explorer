"""Download Census 2020 population for Bexar County, Texas, rasterized to
the San Antonio NLCD grid. Adaptation of download_census_pop.py for SA.

Outputs:
  data/sa/population/sa_pop_2020.tif        — total population
  data/sa/population/sa_child_pop_2020.tif  — under-18 (P1 - P3, same source)

Note on path: the brief's `data/sa/lulc_nlcd_2021_sa.tif` doesn't exist —
the canonical SA LULC raster lives at `data/sa/flood/land_use_2021_sa.tif`
(matches `CITIES['San Antonio, TX']['lulc_file']`). Using that as the
target template.

Population sanity checks:
  - Legal San Antonio is ~1.4 M; full Bexar County is ~2.0 M (2020 Census).
  - Bexar County under-18 share ≈ 27 % (Census QuickFacts) — SA skews
    young vs. the national 22 % share. The modeled extent for SA is
    essentially the full county, so the raster share should land close
    to this published figure.
"""

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
TEMPLATE   = ROOT / "data" / "sa" / "flood" / "land_use_2021_sa.tif"
POP_DIR    = ROOT / "data" / "sa" / "population"
TIGER_DIR  = ROOT / "data" / "population" / "tiger"
TIGER_SHP  = TIGER_DIR / "tl_2020_48_tabblock20.shp"
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_48_tabblock20.zip"
DST_TIF       = POP_DIR / "sa_pop_2020.tif"
DST_CHILD_TIF = POP_DIR / "sa_child_pop_2020.tif"

CENSUS_API = "https://api.census.gov/data/2020/dec/pl"
STATE_FIPS  = "48"   # Texas
COUNTY_FIPS = "029"  # Bexar (San Antonio)


def fetch_census_population():
    """Return {GEOID20: (total_pop, vap_18_plus)} per block.

    P1_001N = total population; P3_001N = voting-age (18+) population.
    Under-18 is derived as P1 - P3 — same PL 94-171 source/vintage as
    the total, no ACS substitution. See download_census_pop.py for the
    rationale (children's nature access RELAY).

    Requires the CENSUS_API_KEY env var. Treat as secret — never logged,
    never written to disk."""
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: CENSUS_API_KEY not set. Get a free key from "
              "https://api.census.gov/data/key_signup.html, then re-run with "
              "`CENSUS_API_KEY=<key> python scripts/data/download_census_pop_sa.py`.")
        sys.exit(2)
    params = {
        "get": "P1_001N,P3_001N,NAME",
        "for": "block:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        "key": api_key,
    }
    print("Fetching Census 2020 block totals + VAP for Bexar County, TX...")
    r = requests.get(CENSUS_API, params=params, timeout=60)
    if not r.ok:
        print(f"  Census API returned HTTP {r.status_code}.")
        sys.exit(2)
    rows = r.json()
    header, *records = rows
    cols = {name: i for i, name in enumerate(header)}
    pops = {}
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
    nonzero = counts > 0

    # Total pop raster (unchanged shape from prior behavior).
    pop_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    pop_per_pixel[nonzero] = blocks["pop_total"].values[nonzero] / counts[nonzero]
    pop_raster = np.where(valid, pop_per_pixel[idx_raster], 0.0).astype(np.float32)

    # Under-18 raster — same uniform-spread method over the same blocks.
    child_per_pixel = np.zeros(len(blocks), dtype=np.float32)
    child_per_pixel[nonzero] = blocks["pop_child"].values[nonzero] / counts[nonzero]
    child_raster = np.where(valid, child_per_pixel[idx_raster], 0.0).astype(np.float32)

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": dst_width, "height": dst_height,
        "crs": dst_crs, "transform": dst_transform,
        "nodata": -9999.0, "compress": "deflate", "tiled": True,
    }
    with rasterio.open(DST_TIF, "w", **profile) as dst:
        dst.write(pop_raster, 1)
    with rasterio.open(DST_CHILD_TIF, "w", **profile) as dst:
        dst.write(child_raster, 1)

    total       = float(pop_raster.sum())
    child_total = float(child_raster.sum())
    child_share = child_total / total if total > 0 else 0.0
    print(f"\nWrote: {DST_TIF} ({DST_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  Pixels with population: {(pop_raster > 0).sum():,}")
    print(f"  Total population in raster: {total:,.0f}")
    print(f"  Reference: legal SA ~1,434,625 (2020); full Bexar Co ~2,009,324.")
    print(f"\nWrote: {DST_CHILD_TIF} ({DST_CHILD_TIF.stat().st_size / 1024:.0f} KB)")
    print(f"  Pixels with under-18 population: {(child_raster > 0).sum():,}")
    print(f"  Under-18 total in raster: {child_total:,.0f} "
          f"({child_share:.1%} of total)")
    print(f"  Reference: Bexar County under-18 share ≈ 27% (Census QuickFacts).")


if __name__ == "__main__":
    sys.exit(main() or 0)
