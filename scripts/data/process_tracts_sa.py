"""Download TIGER 2020 census tracts for Bexar County and write the
filtered shapefile used by 'San Antonio, TX' for per-tract reporting.

Adaptation of process_tracts_expanded.py for SA. Filter: COUNTYFP = '029'
(Bexar). Reproject to EPSG:5070 to match the SA NLCD grid.

Output: data/sa/tracts_bexar.shp
"""

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

ROOT       = Path(__file__).resolve().parent
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_48_tract.zip"
WORK_DIR   = ROOT / "data" / "sa" / "tracts_bexar"
SOURCE_SHP = WORK_DIR / "tl_2020_48_tract.shp"
DST_SHP    = ROOT / "data" / "sa" / "tracts_bexar.shp"

COUNTY_FIPS = "029"  # Bexar
DST_CRS     = "EPSG:5070"


def ensure_source():
    if SOURCE_SHP.exists():
        print(f"  source shapefile already present: {SOURCE_SHP}")
        return
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TIGER_URL} ...")
    r = requests.get(TIGER_URL, timeout=120)
    r.raise_for_status()
    print(f"  {len(r.content) / 1024:.0f} KB; unzipping...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(WORK_DIR)


def main():
    DST_SHP.parent.mkdir(parents=True, exist_ok=True)
    ensure_source()

    print(f"\nReading {SOURCE_SHP.name}...")
    g = gpd.read_file(SOURCE_SHP)
    print(f"  total Texas tracts: {len(g):,}")

    g_b = g[g["COUNTYFP"] == COUNTY_FIPS].copy()
    print(f"  Bexar (COUNTYFP={COUNTY_FIPS}): {len(g_b):,}")

    print(f"  reprojecting {g_b.crs} → {DST_CRS}")
    g_b = g_b.to_crs(DST_CRS)

    if DST_SHP.exists():
        for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
            p = DST_SHP.with_suffix(ext)
            if p.exists():
                p.unlink()

    g_b.to_file(DST_SHP)
    print(f"\nWrote {DST_SHP} ({DST_SHP.stat().st_size / 1024:.0f} KB)")
    print(f"  CRS:    {g_b.crs}")
    print(f"  Bounds: {tuple(round(v) for v in g_b.total_bounds)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
