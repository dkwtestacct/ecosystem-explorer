"""Download TIGER 2020 census tracts for Hennepin County and write the
filtered shapefile used by 'Minneapolis Full, MN' for per-tract reporting.

Source: https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_27_tract.zip
Filter: COUNTYFP == '053' (Hennepin)
Output: data/minneapolis_expanded/tracts_hennepin.shp

Idempotent: skips download/extract if the source files already exist.
"""

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

ROOT       = Path(__file__).resolve().parent
TIGER_URL  = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_27_tract.zip"
WORK_DIR   = ROOT / "data" / "minneapolis_expanded" / "tracts_hennepin"
SOURCE_SHP = WORK_DIR / "tl_2020_27_tract.shp"
DST_SHP    = ROOT / "data" / "minneapolis_expanded" / "tracts_hennepin.shp"

COUNTY_FIPS = "053"  # Hennepin


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
    print(f"  total Minnesota tracts: {len(g):,}")

    g_h = g[g["COUNTYFP"] == COUNTY_FIPS].copy()
    print(f"  Hennepin (COUNTYFP={COUNTY_FIPS}): {len(g_h):,}")

    if DST_SHP.exists():
        # GeoPandas to_file refuses to overwrite some shapefile sidecars;
        # remove the bundle first.
        for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
            p = DST_SHP.with_suffix(ext)
            if p.exists():
                p.unlink()

    g_h.to_file(DST_SHP)
    print(f"\nWrote {DST_SHP} ({DST_SHP.stat().st_size / 1024:.0f} KB)")
    print(f"  CRS:    {g_h.crs}")
    print(f"  Bounds: {tuple(round(v, 4) for v in g_h.total_bounds)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
