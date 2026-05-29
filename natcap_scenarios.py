"""natcap_scenarios.py — load NatCap's San Antonio fixed-scenario LULC rasters as
first-class scenario inputs, plus the scenario-provenance taxonomy.

Streamlit-agnostic (mirrors `surrogate.py`). This module provides the loader,
the scenario metadata, the provenance taxonomy, and a pure flood helper. Wiring
into the dashboard (sidebar selector, per-card display, validation badges) is a
later brief (B2); this is the standalone scaffolding those briefs build on.

ENCODING NOTE (Brief B1, 2026-05-29 — investigate-first finding):
NatCap shipped the SA scenario LULCs only in the *flood* encoding — NLCD ×
tree-canopy 3-tier codes (e.g. 211/212/213 = NLCD 21 × low/med/high canopy;
998 = food forest, 999 = garden) — at 10 m in a WGS84-datum Albers CRS. That
matches the prototype's SA flood CN table (`biophys_floodmitig_sa.csv`)
directly, so the FLOOD metric computes correctly on these rasters.

It does NOT match the compound NLCD×NLUD×tree `lucode` (0–1983) encoding the
Carbon / UCM (temperature) / UNA tables are keyed on. The integers collide with
unrelated compound classes (e.g. scenario code 901 = woody wetland, but compound
lucode 901 = Perennial Ice/Snow), so feeding these rasters into those tables
yields silent garbage. Carbon/temperature reproduction is therefore GATED pending
NatCap's compound scenario inputs (requested separately). A local content-
signature hunt (2026-05-29) found only baseline compound rasters on disk — no
scenario variants. Full record: DESIGN_NOTES.md "Brief B1".
"""
from __future__ import annotations
import os
from functools import lru_cache

import numpy as np

# ── Provenance taxonomy ──────────────────────────────────────────────────────
# Every active scenario carries exactly one provenance value. B2's per-metric
# validation badges key off (provenance, scenario_id); D1's generator block
# reads provenance to decide which scenarios are user-editable.
PROVENANCE_BASELINE     = "baseline"
PROVENANCE_NATCAP_FIXED = "natcap_fixed_scenario"
PROVENANCE_EXPLORER     = "explorer_generated"
PROVENANCE_OPTIMIZER    = "optimizer_suggested"

ALL_PROVENANCES = (
    PROVENANCE_BASELINE,
    PROVENANCE_NATCAP_FIXED,
    PROVENANCE_EXPLORER,
    PROVENANCE_OPTIMIZER,
)

# ── NatCap SA fixed scenarios ────────────────────────────────────────────────
# Keys MUST match the scenario_id values in
# data/sa/natcap_reference_outputs.csv (baseline, FF_20ac, FF_40ac, FF_MAX,
# UA_20ac, UA_40ac, UA_MAX) so B2 can join the prototype's output for a loaded
# scenario to NatCap's published reference value.
#
# `source_filename` is the flood-encoded 10 m raster (NLCD×tree). `baseline` has
# no dedicated raster — the prototype's existing unmodified LULC IS the baseline
# — so load_natcap_fixed_scenario("baseline") raises by design.
SA_NATCAP_FIXED_SCENARIOS = {
    "baseline": dict(
        label="SA Baseline (NatCap)", conversion_type=None, acres=0,
        source_filename=None, provenance=PROVENANCE_BASELINE,
    ),
    "FF_20ac": dict(
        label="Food forest — 20 acres (NatCap)", conversion_type="food_forest", acres=20,
        source_filename="sa_lc_w_20ac_foodfor_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
    "FF_40ac": dict(
        label="Food forest — 40 acres (NatCap)", conversion_type="food_forest", acres=40,
        source_filename="sa_lc_w_40ac_foodfor_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
    "FF_MAX": dict(
        label="Food forest — full (NatCap)", conversion_type="food_forest", acres=None,
        source_filename="sa_lc_w_full_foodfor_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
    "UA_20ac": dict(
        label="Urban agriculture — 20 acres (NatCap)", conversion_type="garden", acres=20,
        source_filename="sa_lc_w_20ac_garden_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
    "UA_40ac": dict(
        label="Urban agriculture — 40 acres (NatCap)", conversion_type="garden", acres=40,
        source_filename="sa_lc_w_40ac_garden_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
    "UA_MAX": dict(
        label="Urban agriculture — full (NatCap)", conversion_type="garden", acres=None,
        source_filename="sa_lc_w_full_garden_10m.tif", provenance=PROVENANCE_NATCAP_FIXED,
    ),
}

# Candidate directories holding the flood-encoded scenario rasters. These live
# OUTSIDE the repo (the NatCap drive pull). The resolver returns the first that
# exists so a machine without the drive pull raises a clear error rather than
# silently mis-resolving. When B2 wires this into the deployed app, the chosen
# scenarios should be vendored + pre-reprojected into the repo for deploy-safety
# (Streamlit Cloud has no ~/Desktop) — flagged in DESIGN_NOTES "Brief B1".
_SA_SCENARIO_SOURCE_DIRS = (
    "data/sa/natcap_scenarios",  # in-repo vendored location (preferred once it exists)
    os.path.expanduser("~/Desktop/natcap_drive_pull/drive_download_misc"),
    os.path.expanduser("~/Desktop/natcap_drive_pull/floodmitig_10m/FloodMitig_10m"),
)


def _resolve_source_path(filename: str) -> str:
    for d in _SA_SCENARIO_SOURCE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"NatCap scenario raster {filename!r} not found in any known source dir: "
        f"{_SA_SCENARIO_SOURCE_DIRS}"
    )


@lru_cache(maxsize=8)
def load_natcap_fixed_scenario(scenario_id: str, reference_grid_path: str):
    """Load a NatCap SA fixed-scenario LULC raster, reprojected + majority-
    resampled onto the prototype's SA grid (the reference grid — 30 m EPSG:5070).

    Args:
        scenario_id: one of SA_NATCAP_FIXED_SCENARIOS (not "baseline").
        reference_grid_path: a raster defining the target grid (CRS, transform,
            shape) — pass the prototype's SA LULC, e.g.
            "data/sa/flood/land_use_compound_sa.tif".

    Returns (lulc_nlcd_tree, metadata):
        lulc_nlcd_tree: int16 array matching the reference grid shape, in the
            NLCD×tree-canopy 3-tier encoding. Nodata → 0 (excluded downstream by
            the CN>0 filter, matching evaluate_scenario).
        metadata: dict (scenario_id, provenance, conversion_type, acres, label,
            source_path, source_crs, source_resolution_m, encoding,
            n_valid_pixels).

    NOTE: the returned array is NLCD×tree, NOT compound — it must NOT be fed to
    the Carbon / UCM / UNA compound-keyed tables (see module docstring). Cached on
    (scenario_id, reference_grid_path); callers must not mutate the returned
    array or dict.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    if scenario_id not in SA_NATCAP_FIXED_SCENARIOS:
        raise KeyError(
            f"unknown scenario_id {scenario_id!r}; known: {list(SA_NATCAP_FIXED_SCENARIOS)}"
        )
    spec = SA_NATCAP_FIXED_SCENARIOS[scenario_id]
    if spec["source_filename"] is None:
        raise ValueError(
            f"scenario_id {scenario_id!r} has no dedicated raster — the prototype's "
            "unmodified LULC IS the baseline scenario; load that instead."
        )
    src_path = _resolve_source_path(spec["source_filename"])

    with rasterio.open(reference_grid_path) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs
    dst = np.zeros(dst_shape, dtype=np.int16)
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=src.nodata, dst_nodata=0,
            resampling=Resampling.mode,  # majority rule — categorical LULC
        )
    metadata = dict(
        scenario_id=scenario_id,
        provenance=spec["provenance"],
        conversion_type=spec["conversion_type"],
        acres=spec["acres"],
        label=spec["label"],
        source_path=src_path,
        source_crs=str(src_crs),
        source_resolution_m=10,
        encoding="nlcd_tree_3tier",
        n_valid_pixels=int((dst > 0).sum()),
    )
    return dst, metadata


def flood_reduction_from_nlcd_tree(lulc_nlcd_tree, soil_clamped, cn_table, lucode_idx_arr):
    """Compute (mean_cn, flood_reduction) for an NLCD×tree LULC raster, mirroring
    evaluate_scenario's SA flood path (app.py:1758-1772, 1843).

    Dependency-injected so this module stays Streamlit-agnostic: pass app.py's
    module-level `cn_table` and `lucode_idx_arr`, and a soil raster clamped to
    [1, 4]. The lookup is `cn_table[lucode_idx_arr[lulc], soil]`; `mean_cn` is the
    mean over pixels with CN > 0 (nodata maps to CN 0 and is excluded).

    Runoff (acre-feet) is intentionally NOT computed here — it depends on the
    city's developed-acre count and the design storm. Call app.py's
    `cn_to_runoff_acre_feet(mean_cn, total_developed_acres)` for that, so the
    SCS-CN runoff formula lives in exactly one place.
    """
    lulc_safe = np.clip(lulc_nlcd_tree, 0, len(lucode_idx_arr) - 1)
    lulc_idx = lucode_idx_arr[lulc_safe]
    cn = cn_table[lulc_idx, soil_clamped]
    valid = cn > 0
    mean_cn = float(cn[valid].mean().round(2)) if valid.any() else 0.0
    flood_reduction = round(100.0 - mean_cn, 2)
    return mean_cn, flood_reduction


if __name__ == "__main__":
    # Standalone smoke test: reproject each NatCap scenario onto the SA grid and
    # compute the flood metric, proving the loader + flood path end-to-end.
    # Carbon/UCM/UNA are intentionally NOT exercised (encoding-gated).
    # Run with the venv's bundled PROJ/GDAL data:
    #   PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
    #   GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
    #   .venv/bin/python3 natcap_scenarios.py
    import rasterio
    import pandas as pd

    REF_GRID = "data/sa/flood/land_use_compound_sa.tif"
    CN_CSV   = "data/sa/flood/biophys_floodmitig_sa.csv"
    SOIL_TIF = "data/sa/flood/soil_group_sa.tif"

    # Build cn_table + lucode_idx_arr the same way load_data does (app.py:795-810).
    # This standalone build mirrors the loader for a sanity proof; the authoritative
    # wiring is via app.py's module-level cn_table when B2 lands.
    bio = pd.read_csv(CN_CSV)
    all_lucodes = sorted(int(x) for x in bio["lucode"])
    lucode_to_idx = {lc: i + 1 for i, lc in enumerate(all_lucodes)}
    cn_table = np.zeros((len(all_lucodes) + 1, 5), dtype=np.float32)
    soil_cols = {1: "CN_A", 2: "CN_B", 3: "CN_C", 4: "CN_D"}
    for _, row in bio.iterrows():
        lc = int(row["lucode"])
        for sg, col in soil_cols.items():
            cn_table[lucode_to_idx[lc], sg] = float(row[col])
    max_lucode = max(all_lucodes)
    lucode_idx_arr = np.zeros(max_lucode + 1, dtype=np.int32)
    for lc, idx in lucode_to_idx.items():
        lucode_idx_arr[lc] = idx

    with rasterio.open(SOIL_TIF) as s:
        soil_clamped = np.clip(s.read(1).astype(int), 1, 4)

    print(f"{'scenario':9s} {'mean_cn':>8s} {'flood_red':>10s} {'valid_px':>10s}  source")
    print("-" * 70)
    for sid, spec in SA_NATCAP_FIXED_SCENARIOS.items():
        if spec["source_filename"] is None:
            print(f"{sid:9s}  [baseline — no dedicated raster]")
            continue
        try:
            lulc, meta = load_natcap_fixed_scenario(sid, REF_GRID)
        except FileNotFoundError as e:
            print(f"{sid:9s}  [skip: {e}]")
            continue
        mean_cn, flood_red = flood_reduction_from_nlcd_tree(
            lulc, soil_clamped, cn_table, lucode_idx_arr
        )
        print(f"{sid:9s} {mean_cn:8.2f} {flood_red:10.2f} {meta['n_valid_pixels']:10d}  "
              f"{os.path.basename(meta['source_path'])}")
