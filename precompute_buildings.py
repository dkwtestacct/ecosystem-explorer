"""
precompute_buildings.py — one-shot offline precompute of the Phase 8 buildings
rasterize products, so cold start can read from disk instead of re-rasterizing
~691k SA building polygons (~32 s of the ~60 s SA cold start after Lever 1's
Fast→dense CSV switch).

Mirrors `_load_city_runtime_state` Phase 8 exactly — same source file, same CRS
transform, same fill/dtype/typing logic — and writes three artifacts to the
paths declared in `config.CITIES[<city>]`:

  buildings_precomputed_file       — uint8 binary mask (1 = building pixel)
  buildings_type_precomputed_file  — int32 typed-code raster (fill -1; codes 0-3)
  buildings_precomputed_meta_file  — JSON sidecar:
                                       total_potential_damage_usd: float
                                       buildings_have_types: bool
                                       buildings_type_coverage: float
                                       source_files: {logical → path}
                                       source_sha256: {logical → hex digest}
                                       generated_at: ISO timestamp
                                       schema_version: SCENARIO_SCHEMA_VERSION
                                       grid_shape: [rows, cols]
                                       crs: <EPSG string>

Run after the source buildings layer changes:

    PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
    GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
    .venv/bin/python precompute_buildings.py --city 'San Antonio, TX'

verify_baselines.py runs a fresh rasterize and asserts byte-identity vs the
on-disk artifacts; if the buildings source file changes without a re-precompute,
the gate fails on the staleness cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, ~MB-range memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True,
                    help="CITIES key (e.g. 'San Antonio, TX').")
    args = ap.parse_args()
    city_key = args.city

    # Streamlit stub — same pattern as verify_baselines / precompute_scenarios.
    sys.path.insert(0, ".")
    import verify_baselines as vb
    vb._DESIRED_CITY = city_key
    vb._SessionStateStub._store["entry_city"] = city_key
    sys.modules["streamlit"] = vb._StubSt()

    print(f"Importing app (Streamlit stub, entry={city_key})...")
    t0 = time.time()
    import app  # noqa: E402
    print(f"  app import: {time.time() - t0:.1f}s")

    cfg = app.CITIES[city_key]
    out_bin  = cfg.get("buildings_precomputed_file")
    out_type = cfg.get("buildings_type_precomputed_file")
    out_meta = cfg.get("buildings_precomputed_meta_file")
    if not (out_bin and out_type and out_meta):
        print(f"ERROR: {city_key} has no buildings_precomputed_* config keys. "
              "Add them to CITIES before running this script.")
        return 2
    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)

    # Drive a city load so we have the CityState ref_transform + cooling_lulc
    # shape for the rasterize call. After this, we re-run the Phase 8 logic
    # in isolation so the artifacts match what Phase 8 writes into CityState
    # exactly — no dependence on the internal struct layout.
    state = vb._rebind_city(app, city_key)

    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    import geopandas as gpd  # noqa: E402
    import rasterio  # noqa: E402
    from rasterio.features import rasterize  # noqa: E402

    src_path = cfg["buildings_file"]
    print(f"\nLoading source: {src_path}")
    t1 = time.time()
    gdf = gpd.read_file(src_path)
    if gdf.crs is None or str(gdf.crs) != cfg["crs"]:
        gdf = gdf.to_crs(cfg["crs"])
    print(f"  {len(gdf):,} polygons in {time.time() - t1:.1f}s")

    # Mirror Phase 8's typing logic exactly. Numeric type column → InVEST
    # codes 0–3 directly; OSM string tag column → app._osm_to_invest_type.
    # `buildings_have_types` flips on path (a); stays False on path (b) until
    # the rasterize coverage check below fills it in.
    invest_types = None
    if "type" in gdf.columns:
        numeric = pd.to_numeric(gdf["type"], errors="coerce")
        numeric_clean = numeric.dropna()
        if len(numeric_clean) > 0 and numeric_clean.between(0, 3).all():
            invest_types = numeric.fillna(-1).astype("int32")
            buildings_have_types = True
        else:
            invest_types = (
                gdf["type"].map(app._osm_to_invest_type).fillna(-1).astype("int32")
            )
            buildings_have_types = False  # filled in from pixel coverage below
    else:
        buildings_have_types = False

    # Damage table → potential damage dollars. Only computed when both types
    # are present AND the damage table is configured (mirrors Phase 8 exactly).
    damage_table_file = cfg.get("damage_table_file")
    if buildings_have_types and damage_table_file:
        damage_table = pd.read_csv(damage_table_file)
        type_to_damage = dict(zip(damage_table["Type"], damage_table["Damage"]))
        gdf["damage_rate_usd_m2"] = (
            gdf["type"].map(type_to_damage).fillna(0)
        )
        gdf["area_m2"] = gdf.geometry.area
        gdf["potential_damage_usd"] = (
            gdf["area_m2"] * gdf["damage_rate_usd_m2"]
        )
        total_potential_damage_usd = float(gdf["potential_damage_usd"].sum())
    else:
        total_potential_damage_usd = 0.0

    ref_shape = state.cooling_lulc.shape
    ref_transform = state.ref_transform
    crs_wkt = rasterio.crs.CRS.from_user_input(cfg["crs"]).to_wkt()

    print(f"\nRasterizing binary mask ({ref_shape[0]:,}×{ref_shape[1]:,}, uint8)...")
    t2 = time.time()
    bin_raster = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=ref_shape, transform=ref_transform,
        fill=0, dtype="uint8",
    )
    print(f"  {time.time() - t2:.1f}s — {int(np.sum(bin_raster > 0)):,} "
          "building pixels")

    if invest_types is not None:
        print(f"Rasterizing typed codes ({ref_shape[0]:,}×{ref_shape[1]:,}, "
              f"int32, fill=-1)...")
        t3 = time.time()
        type_raster = rasterize(
            ((geom, int(t)) for geom, t in zip(gdf.geometry, invest_types)),
            out_shape=ref_shape, transform=ref_transform,
            fill=-1, dtype="int32",
        )
        print(f"  {time.time() - t3:.1f}s")
    else:
        type_raster = np.full(ref_shape, -1, dtype="int32")

    # Recompute the pixel-level type-coverage stat the same way Phase 8 does.
    total_building_pixels = int(np.sum(bin_raster > 0))
    typed_pixels = int(np.sum(type_raster > 0))
    if total_building_pixels > 0:
        buildings_type_coverage = typed_pixels / total_building_pixels
    else:
        buildings_type_coverage = 0.0
    if not buildings_have_types:
        # Path (b) — flip only if OSM mapping actually produced typed pixels.
        buildings_have_types = typed_pixels > 0

    def _wr(path, arr, dtype, nodata):
        with rasterio.open(
            path, "w", driver="GTiff",
            height=arr.shape[0], width=arr.shape[1], count=1,
            dtype=dtype, crs=crs_wkt, transform=ref_transform,
            nodata=nodata, compress="deflate",
        ) as dst:
            dst.write(arr.astype(dtype), 1)

    print(f"\nWriting {out_bin}")
    _wr(out_bin, bin_raster, "uint8", None)
    print(f"Writing {out_type}")
    _wr(out_type, type_raster, "int32", -1)

    sources = {"buildings_file": src_path}
    if damage_table_file:
        sources["damage_table_file"] = damage_table_file
    meta = {
        "city": city_key,
        "schema_version": int(app.SCENARIO_SCHEMA_VERSION),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "grid_shape": [int(ref_shape[0]), int(ref_shape[1])],
        "crs": cfg["crs"],
        "total_potential_damage_usd": total_potential_damage_usd,
        "buildings_have_types": buildings_have_types,
        "buildings_type_coverage": buildings_type_coverage,
        "source_files": sources,
        "source_sha256": {k: _sha256(v) for k, v in sources.items()},
    }
    print(f"Writing {out_meta}")
    Path(out_meta).write_text(json.dumps(meta, indent=2))

    _bin_sz = Path(out_bin).stat().st_size  / 1024 / 1024
    _typ_sz = Path(out_type).stat().st_size / 1024 / 1024
    print(f"\nDone — {city_key} buildings precomputed in {time.time() - t0:.1f}s "
          f"total. On-disk: {_bin_sz:.1f} MB binary + {_typ_sz:.1f} MB typed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
