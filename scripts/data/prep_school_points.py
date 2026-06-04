"""prep_school_points.py — per-city K-12 school points from NCES.

Sources (one-time download, not committed):
  EDGE Geocode Public Schools 2021-22  → lat/lon for ~102k public + charter schools
  EDGE Geocode Private Schools 2021-22 → lat/lon for ~22k private K-12 schools (PSS)
  CCD Directory 2022-23                → LEVEL + CHARTER_TEXT for public schools
                                         (vintage offset is ~1 year, schools don't churn that fast)

Outputs (committed, per-city GeoJSONs, ~10-100 KB each):
  data/sa/schools_sa.geojson          → schools inside the SA modelable extent
  data/population/schools_mpls.geojson → schools inside the MN downtown extent

Schema per output feature:
  NAME        — school name
  sector      — 'public' | 'charter' | 'private'
  source      — 'NCES CCD 2022-23 + EDGE 2021-22' or 'NCES PSS 2021-22 + EDGE 2021-22'
  level       — CCD LEVEL string ('Elementary' / 'Middle' / 'High' / 'Other' for public/charter; 'PSS' for private)
  county_fips — 5-digit FIPS (e.g. '48029' = Bexar)
  geometry    — Point in EPSG:4269 (the EDGE source CRS; reprojected per-city when consumed)

Filter rules:
- K-12 only. Public LEVEL must not be 'Postsecondary'. Private (PSS) assumed K-12;
  any 'Postsecondary Title IV' rows we encounter are dropped explicitly.
- Geographically clipped to each city's modelable extent. SA uses Bexar bbox
  (~3,060 km² county); MN uses Hennepin downtown extent (a strict subset of the
  county). Filter is point-in-bbox of the city's LULC raster after CRS reprojection.
- Include private. The On-the-Radar decision was 'private included; documented'.
  Caveat surfaced in REFERENCE.md / DESIGN_NOTES.md / on the metric card.

Run once per data vintage update. Do not invoke from app.py / Streamlit; the
inputs aren't in the repo (gitignored) and only the per-city outputs are."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
EDGE_PUB = ROOT / "data/schools/edge_2122/EDGE_GEOCODE_PUBLICSCH_2122/Shapefiles_SCH/EDGE_GEOCODE_PUBLICSCH_2122.shp"
EDGE_PRV = ROOT / "data/schools/edge_2122/EDGE_GEOCODE_PRIVATESCH_2122/EDGE_GEOCODE_PRIVATESCH_2122.shp"
CCD_DIR  = ROOT / "data/schools/ccd_sch_029_2223_w_1a_083023.csv"

# Per-city: (county_fips, output_path, lulc_template_path).
# lulc_template defines the city's modelable bbox; schools whose lat/lon fall
# outside this bbox after CRS reprojection are dropped.
CITIES = [
    ("48029",
     ROOT / "data/sa/schools_sa.geojson",
     ROOT / "data/sa/flood/land_use_2021_sa.tif"),
    ("27053",
     ROOT / "data/population/schools_mpls.geojson",
     ROOT / "data/cooling/land_use_2021.tif"),
]


def load_ccd_levels() -> pd.DataFrame:
    """CCD 2022-23 directory: NCESSCH → LEVEL + CHARTER_TEXT.

    Columns of interest (per CCD docs):
      NCESSCH      — 12-char school ID (matches EDGE)
      LEVEL        — 'Elementary' / 'Middle' / 'High' / 'Other' / 'Ungraded' / 'Adult' / 'Postsecondary'
      CHARTER_TEXT — 'Yes' / 'No' / 'Not Applicable' / 'Missing/Not Reported'
    """
    print(f"Loading CCD: {CCD_DIR.name}")
    df = pd.read_csv(
        CCD_DIR, usecols=["NCESSCH", "LEVEL", "CHARTER_TEXT"],
        dtype={"NCESSCH": str, "LEVEL": str, "CHARTER_TEXT": str},
        low_memory=False,
    )
    df["NCESSCH"] = df["NCESSCH"].str.zfill(12)
    print(f"  CCD rows: {len(df):,}")
    print(f"  LEVEL value counts:")
    for lvl, n in df["LEVEL"].value_counts().items():
        print(f"    {lvl}: {n:,}")
    return df


def load_public_with_levels() -> gpd.GeoDataFrame:
    """EDGE public schools joined with CCD level + charter status.
    Drops postsecondary, restricted, and unjoined rows."""
    print(f"\nLoading EDGE public: {EDGE_PUB.name}")
    pub = gpd.read_file(EDGE_PUB, columns=["NCESSCH", "NAME", "STATE", "CNTY"])
    pub["NCESSCH"] = pub["NCESSCH"].astype(str).str.zfill(12)
    print(f"  EDGE public rows: {len(pub):,}")
    ccd = load_ccd_levels()
    merged = pub.merge(ccd, on="NCESSCH", how="left")
    print(f"  merged: {len(merged):,}; unmatched: "
          f"{int(merged['LEVEL'].isna().sum()):,}")
    # Drop postsecondary, adult, prekindergarten-only, and unreported.
    # Keep all K-12 levels including 'Secondary' (combined middle+high
    # schools) and 'Other' (e.g. K-12 spanning schools).
    keep = merged["LEVEL"].isin(["Elementary", "Middle", "High",
                                  "Secondary", "Other", "Ungraded"])
    merged = merged[keep].copy()
    print(f"  after K-12 filter: {len(merged):,}")
    # Sector
    merged["sector"] = merged["CHARTER_TEXT"].where(
        merged["CHARTER_TEXT"] == "Yes", "public"
    ).replace({"Yes": "charter"})
    merged["source"] = "NCES CCD 2022-23 + EDGE 2021-22"
    return merged[["NAME", "sector", "source", "LEVEL", "CNTY", "geometry"]] \
        .rename(columns={"LEVEL": "level", "CNTY": "county_fips"})


def load_private() -> gpd.GeoDataFrame:
    """EDGE private (PSS) schools. PSS covers K-12 + sometimes Pre-K only;
    the EDGE geocode file doesn't expose a level field, so we keep all rows
    and surface 'level=PSS' as the placeholder. Caveat documented."""
    print(f"\nLoading EDGE private: {EDGE_PRV.name}")
    prv = gpd.read_file(EDGE_PRV, columns=["NAME", "STATE", "CNTY"])
    print(f"  EDGE private rows: {len(prv):,}")
    prv["sector"] = "private"
    prv["source"] = "NCES PSS 2021-22 + EDGE 2021-22"
    prv["level"]  = "PSS"
    return prv[["NAME", "sector", "source", "level", "CNTY", "geometry"]] \
        .rename(columns={"CNTY": "county_fips"})


def clip_to_city_extent(gdf: gpd.GeoDataFrame, county_fips: str,
                        lulc_template: Path) -> gpd.GeoDataFrame:
    """Filter to county_fips, then to the LULC raster's bbox after CRS reproject."""
    sub = gdf[gdf["county_fips"] == county_fips].copy()
    print(f"  county {county_fips}: {len(sub):,} schools")
    if not sub.shape[0]:
        return sub
    with rasterio.open(lulc_template) as src:
        dst_crs = src.crs.to_string()
        bnd = src.bounds
        bbox_geom = box(bnd.left, bnd.bottom, bnd.right, bnd.top)
    sub = sub.to_crs(dst_crs)
    in_bbox = sub.geometry.within(bbox_geom)
    sub = sub[in_bbox].copy()
    # Reproject back to EPSG:4269 (the EDGE source CRS) for output consistency.
    sub = sub.to_crs("EPSG:4269")
    print(f"  in modelable extent: {len(sub):,}")
    return sub


def main() -> int:
    if not EDGE_PUB.exists() or not EDGE_PRV.exists() or not CCD_DIR.exists():
        print("ERROR: source files missing. Run the bash one-liner in this "
              "script's docstring to download EDGE 2021-22 + CCD 2022-23 "
              "into data/schools/ first.")
        return 2
    pub = load_public_with_levels()
    prv = load_private()
    all_schools = pd.concat([pub, prv], ignore_index=True)
    all_schools = gpd.GeoDataFrame(all_schools, geometry="geometry",
                                   crs="EPSG:4269")
    print(f"\nTotal K-12 schools (public + charter + private, US-wide): "
          f"{len(all_schools):,}")
    print(f"  public:  {(all_schools['sector'] == 'public').sum():,}")
    print(f"  charter: {(all_schools['sector'] == 'charter').sum():,}")
    print(f"  private: {(all_schools['sector'] == 'private').sum():,}")

    for county_fips, out_path, lulc_template in CITIES:
        print(f"\n=== {county_fips} → {out_path.name} ===")
        sub = clip_to_city_extent(all_schools, county_fips, lulc_template)
        if sub.shape[0] == 0:
            print("  WARN: no schools to write")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_file(out_path, driver="GeoJSON")
        sz_kb = out_path.stat().st_size / 1024
        print(f"  wrote {sub.shape[0]:,} schools, {sz_kb:.1f} KB")
        print(f"  sector breakdown:")
        for sec, n in sub["sector"].value_counts().items():
            print(f"    {sec}: {n:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
