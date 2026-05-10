"""Download SSURGO hydrologic-soil-group data for Bexar County, Texas
programmatically via the USDA Soil Data Access (SDA) REST API.

Adaptation of download_ssurgo.py for San Antonio. Outputs:
  data/sa/flood/ssurgo_hydgrp_bexar.csv
  data/sa/flood/ssurgo_bexar_hsg.shp

Same encoding (A=1, B=2, C=3, D=4; A/D→A, B/D→B, C/D→C; missing→0). South
Texas is known for clay-rich expansive soils so we expect a higher D-class
share than Hennepin (which was ~67 % C/C-D).
"""

import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt as shapely_wkt

SDA_URL    = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
AREASYMBOL = "TX029"  # Bexar County, Texas

OUT_DIR    = Path("data/sa/flood")
CSV_PATH   = OUT_DIR / "ssurgo_hydgrp_bexar.csv"
SHP_PATH   = OUT_DIR / "ssurgo_bexar_hsg.shp"

HSG_TO_CODE = {
    "A": 1, "A/D": 1,
    "B": 2, "B/D": 2,
    "C": 3, "C/D": 3,
    "D": 4,
}


def sda_query(query, *, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(SDA_URL,
                              data={"query": query, "format": "json+columnname"},
                              timeout=180)
            r.raise_for_status()
            d = r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(2)
            continue
        if "Table" not in d or not d["Table"]:
            return pd.DataFrame()
        cols = d["Table"][0]
        rows = d["Table"][1:]
        return pd.DataFrame(rows, columns=cols)


def fetch_hydgrp_table():
    print(f"\n=== 1. Tabular: hydgrp by mukey for areasymbol = {AREASYMBOL} ===")
    q = f"""
    SELECT mu.mukey, mu.musym, mu.muname, c.hydgrp, c.comppct_r AS comp_pct
    FROM mapunit mu
    INNER JOIN legend l ON mu.lkey = l.lkey
    INNER JOIN component c ON mu.mukey = c.mukey
    WHERE l.areasymbol = '{AREASYMBOL}' AND c.majcompflag = 'Yes'
    ORDER BY mu.mukey, c.comppct_r DESC
    """
    df = sda_query(q)
    print(f"  raw rows (one per major component): {len(df):,}")
    df["mukey"] = df["mukey"].astype(str)
    df["comp_pct"] = pd.to_numeric(df["comp_pct"], errors="coerce")
    dedup = df.sort_values("comp_pct", ascending=False).drop_duplicates("mukey", keep="first")
    print(f"  one row per map unit (largest major component): {len(dedup):,}")
    print(f"  hydgrp distribution:")
    print(dedup["hydgrp"].value_counts(dropna=False).to_string())
    return dedup[["mukey", "musym", "muname", "hydgrp"]]


def fetch_polygons(mukeys):
    print(f"\n=== 2. Spatial: mupolygon WKT for {len(mukeys):,} map units ===")
    chunk_size = 25
    rows = []
    for i in range(0, len(mukeys), chunk_size):
        chunk = mukeys[i : i + chunk_size]
        in_list = ",".join(f"'{m}'" for m in chunk)
        q = f"""
        SELECT mukey, mupolygonkey, mupolygongeo.STAsText() AS wkt
        FROM mupolygon
        WHERE areasymbol = '{AREASYMBOL}' AND mukey IN ({in_list})
        """
        df = sda_query(q)
        rows.append(df)
        n_chunks = (len(mukeys) + chunk_size - 1) // chunk_size
        print(f"  chunk {i // chunk_size + 1}/{n_chunks}: +{len(df):,} polygons "
              f"(cumulative: {sum(len(r) for r in rows):,})")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hg = fetch_hydgrp_table()
    hg.to_csv(CSV_PATH, index=False)
    print(f"  wrote {CSV_PATH}")

    polys = fetch_polygons(hg["mukey"].tolist())
    polys["mukey"] = polys["mukey"].astype(str)
    print(f"\n  total polygons fetched: {len(polys):,}")

    print("  parsing WKT geometries...")
    geom = polys["wkt"].apply(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(
        polys[["mukey", "mupolygonkey"]],
        geometry=geom.values,
        crs="EPSG:4326",
    )

    gdf = gdf.merge(hg[["mukey", "hydgrp", "musym", "muname"]], on="mukey", how="left")
    gdf["hsg_code"] = gdf["hydgrp"].map(HSG_TO_CODE).fillna(0).astype("int8")

    print(f"\n=== 3. Coverage summary ===")
    print(f"  total polygons:           {len(gdf):,}")
    print(f"  polygons w/ hydgrp:       {gdf['hydgrp'].notna().sum():,}")
    print(f"  polygons missing hydgrp:  {gdf['hydgrp'].isna().sum():,}")
    print(f"  total area (km², proj):   {gdf.to_crs('EPSG:5070').geometry.area.sum() / 1e6:.1f}")
    print(f"  hydgrp distribution:")
    print(gdf["hydgrp"].value_counts(dropna=False).to_string())
    print(f"  hsg_code distribution:")
    print(gdf["hsg_code"].value_counts().sort_index().to_string())

    gdf_out = gdf.rename(columns={"mupolygonkey": "polykey", "muname": "muname"})
    print(f"\n=== 4. Writing shapefile ===")
    gdf_out.to_file(SHP_PATH, driver="ESRI Shapefile")
    print(f"  wrote {SHP_PATH} (+ .dbf .shx .prj)")
    print(f"  CRS: {gdf_out.crs}")
    print(f"  bounds: {tuple(round(v, 4) for v in gdf_out.total_bounds)}")


if __name__ == "__main__":
    sys.exit(main())
