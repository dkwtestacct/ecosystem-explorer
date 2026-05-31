"""Download City of San Antonio council districts and reproject to the
prototype's canonical SA CRS (EPSG:5070).

Source: City of San Antonio Open Data Portal — "Redistricted Council Districts
2022" (boundaries effective for the May 2023 municipal election).
Dataset page: https://data.sanantonio.gov/dataset/redistricted-council-districts-2022

License: not explicitly stated on the dataset page. Operated under the City of
San Antonio Open Data Portal; attribution cited in `data/sa/README.md` and the
DATA_INVENTORY catalog entry.

**Portal CRS quirk handled here.** The GeoJSON response declares EPSG:4326
in its metadata but the geometry values are actually EPSG:3857 (Web Mercator,
meters). We force-override the declared CRS before reprojecting; without this,
geopandas would treat Web-Mercator-meter values as degrees and the reprojection
would produce infinities.

Output: `data/sa/sa_council_districts.gpkg` (EPSG:5070, ~1.4 MB), 10 polygons
with attributes `OBJECTID`, `District` ('1' through '10'), `geometry`.

Rebuild command: `python scripts/data/download_sa_council_districts.py`
"""

import sys
from pathlib import Path

import geopandas as gpd
import requests

GEOJSON_URL = (
    "https://opendata-cosagis.opendata.arcgis.com/api/download/v1/items/"
    "b25026ba7f55479b88b6d93552a4237c/geojson?layers=0"
)

# The prototype's canonical SA CRS — equal-area for area-based metrics.
TARGET_CRS = "EPSG:5070"

# What the portal's GeoJSON metadata declares (incorrect — see module docstring).
DECLARED_CRS = "EPSG:4326"

# What the geometry values actually are (Web Mercator meters).
ACTUAL_SOURCE_CRS = "EPSG:3857"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "data" / "sa" / "sa_council_districts.gpkg"


def main() -> None:
    print(f"Fetching {GEOJSON_URL}")
    response = requests.get(GEOJSON_URL, timeout=60)
    response.raise_for_status()

    tmp_path = OUTPUT_PATH.with_suffix(".source.geojson")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(response.content)

    g = gpd.read_file(tmp_path)

    if str(g.crs) != DECLARED_CRS:
        print(
            f"Note: portal-declared CRS is {g.crs} (expected {DECLARED_CRS}); "
            "the CRS-override fix may need re-verification."
        )

    # Force-override the declared CRS to match the actual geometry units, then
    # reproject to the prototype's canonical SA CRS.
    g = g.set_crs(ACTUAL_SOURCE_CRS, allow_override=True)
    g_target = g.to_crs(TARGET_CRS)

    # Sanity checks.
    assert len(g_target) == 10, f"expected 10 districts, got {len(g_target)}"
    districts = sorted(g_target["District"].astype(str).tolist(), key=int)
    assert districts == [str(i) for i in range(1, 11)], (
        f"district IDs should be 1..10, got {districts}"
    )

    g_target.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  10 districts, EPSG:5070, bounds {g_target.total_bounds.tolist()}")

    # Clean up the raw download — the canonical artifact is the reprojected
    # GeoPackage. Re-run this script to refresh.
    tmp_path.unlink()


if __name__ == "__main__":
    sys.exit(main())
