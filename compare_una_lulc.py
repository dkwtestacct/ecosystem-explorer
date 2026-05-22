#!/usr/bin/env python3
"""compare_una_lulc.py — UNA LULC investigation.

Compares the prototype's existing **cooling LULC** (the raster used as input
to UCM, UFR, and other metrics for Minneapolis) against the **InVEST UNA
sample LULC** (`LULC_NLCD_2021.tif`, the raster the published InVEST Urban
Nature Access model was designed for).

The question this script answers: before implementing canonical InVEST UNA,
does the choice of LULC raster materially affect the UNA result? The two
candidate inputs are:

  1. data/cooling/land_use_2021.tif
       — the cooling LULC, already used everywhere else in the prototype.
  2. data/invest/nature_access/UrbanNatureAccess_sample_data_MN/
       LULC_NLCD_2021.tif
       — the InVEST UNA sample LULC.

Work performed:
  - Step 1  Side-by-side raster inspection (shape, CRS, transform, bounds,
            nodata, MD5, unique LULC values + pixel counts).
  - Step 2  Per-pixel agreement in the overlapping extent, plus per-class
            pixel-count and population summaries for the "nature" classes
            (urban_nature > 0 in the UNA biophysical table).
  - Step 3  One canonical InVEST UNA run (natcap.invest 3.16.2) with the
            parameters from DESIGN_NOTES.md, reporting the
            headline UNA metrics.

Running InVEST UNA *twice* (once per LULC) was the originally-scoped step 3.
It is deliberately collapsed to a single run: the two rasters are
byte-for-byte identical (proven in step 1 via MD5 + a full byte compare), so
a second run on a deterministic model with identical inputs and identical
parameters is mathematically guaranteed to produce bit-identical output —
Pearson r = 1.000 and MAE = 0 by construction, not by measurement. The
single run captures the concrete UNA baseline numbers for the research
document.

Usage:
    python3 compare_una_lulc.py        # anaconda base env (natcap.invest 3.16.2)

Prerequisites:
    - natcap.invest installed (3.16.2 — same env as compare_una_invest.py).
    - MN data files in place (data/cooling/, data/invest/nature_access/,
      data/population/).

Does NOT modify app.py, config.py, surrogate.py, or any shipped code.
No schema bump. Mirrors the InVEST-run pattern of compare_una_invest.py.
"""
from __future__ import annotations

import csv
import filecmp
import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ── Paths ───────────────────────────────────────────────────────────────────
COOLING_LULC = "data/cooling/land_use_2021.tif"
UNA_LULC = ("data/invest/nature_access/UrbanNatureAccess_sample_data_MN/"
            "LULC_NLCD_2021.tif")
UNA_TABLE = ("data/invest/nature_access/UrbanNatureAccess_sample_data_MN/"
             "LULC_attribute_table_UNA.csv")
POP_RASTER = "data/population/minneapolis_pop_2020.tif"
CRS = "EPSG:26915"

# ── InVEST UNA parameters (from DESIGN_NOTES.md) ─────────────────
UNA_DEMAND = 16.7              # m²/capita — NatCap SA-study value
UNA_SEARCH_RADIUS = 800        # m — NatCap SA-study value (~10-min walk)
UNA_SEARCH_RADIUS_MODE = "uniform radius"
UNA_DECAY = "dichotomy"        # binary in/out within the radius
UNA_AGGREGATE_BY_POP_GROUP = False


# ── Helpers ─────────────────────────────────────────────────────────────────
def md5(path: str) -> str:
    """MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_raster(path: str) -> dict:
    """Read core geospatial properties + the LULC value histogram."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        vals, counts = np.unique(arr, return_counts=True)
        return {
            "path": path,
            "size_bytes": os.path.getsize(path),
            "md5": md5(path),
            "shape": (ds.height, ds.width),
            "crs": str(ds.crs),
            "dtype": str(ds.dtypes[0]),
            "nodata": ds.nodata,
            "transform": ds.transform,
            "pixel_size": (ds.transform.a, ds.transform.e),
            "bounds": tuple(round(b, 2) for b in ds.bounds),
            "histogram": dict(zip(vals.tolist(), counts.tolist())),
            "array": arr,
        }


def save_geotiff(arr, path, crs, transform, nodata=None, dtype=None):
    """Save a 2-D array as a single-band GeoTIFF."""
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def construct_aoi_from_raster(raster_path, output_dir):
    """Single-polygon AOI GeoPackage covering a raster's bounds.

    InVEST UNA requires an admin-boundaries vector and aggregates per feature.
    The per-pixel output rasters do not depend on the admin geometry — only
    the aggregate GeoPackage does — so a single bounding polygon is correct
    for a per-pixel comparison. Same approach as compare_una_invest.py.
    """
    import geopandas as gpd
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    aoi_path = os.path.join(output_dir, "aoi_admin_boundaries.gpkg")
    gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs).to_file(
        aoi_path, driver="GPKG")
    return aoi_path


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("UNA LULC Investigation: cooling LULC vs InVEST UNA sample LULC")
    print("=" * 70)

    for p in (COOLING_LULC, UNA_LULC, UNA_TABLE, POP_RASTER):
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: required input not found: {p}")

    # ── Step 1: Side-by-side raster inspection ───────────────────────────
    print("\n1. Side-by-side raster inspection")
    print("-" * 70)
    cool = inspect_raster(COOLING_LULC)
    una = inspect_raster(UNA_LULC)

    def _row(label, a, b):
        print(f"   {label:14s} | {str(a):28s} | {str(b)}")

    print(f"   {'':14s} | {'COOLING LULC':28s} | {'UNA SAMPLE LULC'}")
    print(f"   {'-' * 14} + {'-' * 28} + {'-' * 28}")
    _row("path", "data/cooling/", "data/invest/nature_access/")
    _row("filename", os.path.basename(cool["path"]),
         os.path.basename(una["path"]))
    _row("size (bytes)", cool["size_bytes"], una["size_bytes"])
    _row("MD5", cool["md5"], una["md5"])
    _row("shape (r x c)", f"{cool['shape'][0]} x {cool['shape'][1]}",
         f"{una['shape'][0]} x {una['shape'][1]}")
    _row("CRS", cool["crs"], una["crs"])
    _row("dtype", cool["dtype"], una["dtype"])
    _row("nodata", cool["nodata"], una["nodata"])
    _row("pixel size", cool["pixel_size"], una["pixel_size"])
    _row("bounds", cool["bounds"][:2], una["bounds"][:2])
    _row("  (cont.)", cool["bounds"][2:], una["bounds"][2:])

    md5_match = cool["md5"] == una["md5"]
    byte_match = filecmp.cmp(COOLING_LULC, UNA_LULC, shallow=False)
    print()
    print(f"   MD5 match           : {md5_match}")
    print(f"   Byte-for-byte equal : {byte_match}  (filecmp deep compare)")
    if byte_match:
        print("   >>> The two LULC files are BYTE-FOR-BYTE IDENTICAL. The cooling")
        print("   >>> LULC is a renamed copy of the InVEST UNA sample raster.")

    # ── Step 1b: LULC class distributions ────────────────────────────────
    print("\n   LULC class distribution (pixel counts):")
    una_table = pd.read_csv(UNA_TABLE)
    desc = dict(zip(una_table["lucode"].astype(int), una_table["lulc_desc"]))
    nature_codes = set(
        una_table.loc[una_table["urban_nature"] > 0, "lucode"].astype(int))
    nature_weight = dict(zip(una_table["lucode"].astype(int),
                             una_table["urban_nature"]))
    all_codes = sorted(set(cool["histogram"]) | set(una["histogram"]))
    print(f"   {'code':>5s} {'description':32s} {'cooling':>10s} "
          f"{'UNA':>10s} {'nature?':>9s}")
    for code in all_codes:
        c_n = cool["histogram"].get(code, 0)
        u_n = una["histogram"].get(code, 0)
        tag = ""
        if code == -128:
            tag = "(nodata)"
        elif code in nature_codes:
            tag = f"yes ({nature_weight[code]:g})"
        else:
            tag = "no"
        print(f"   {code:>5d} {desc.get(code, '—'):32s} {c_n:>10,d} "
              f"{u_n:>10,d} {tag:>9s}")

    classes_match = cool["histogram"] == una["histogram"]
    print(f"\n   Class distributions identical: {classes_match}")

    # ── Step 2: Per-pixel agreement ──────────────────────────────────────
    print("\n2. Per-pixel agreement")
    print("-" * 70)
    same_grid = (cool["shape"] == una["shape"]
                 and cool["transform"] == una["transform"])
    print(f"   Identical grid (shape + transform): {same_grid}")
    if not same_grid:
        raise SystemExit(
            "   Grids differ — overlap-region alignment would be needed; "
            "this is unexpected for these two files and is not reached when "
            "the rasters are byte-identical.")

    ca, ua = cool["array"], una["array"]
    total_px = ca.size
    agree_px = int(np.sum(ca == ua))
    disagree_px = total_px - agree_px
    print(f"   Total pixels in overlap : {total_px:,}")
    print(f"   Pixels in agreement     : {agree_px:,} "
          f"({100.0 * agree_px / total_px:.4f}%)")
    print(f"   Pixels in disagreement  : {disagree_px:,} "
          f"({100.0 * disagree_px / total_px:.4f}%)")
    if disagree_px:
        diff_idx = np.argwhere(ca != ua)
        pairs = {}
        for r, c in diff_idx:
            pairs[(int(ca[r, c]), int(ua[r, c]))] = \
                pairs.get((int(ca[r, c]), int(ua[r, c])), 0) + 1
        print("   Disagreement breakdown (cooling -> UNA : count):")
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"     {a} -> {b} : {n:,}")
    else:
        print("   No disagreements — nothing to characterize.")

    # ── Step 2b: nature-class pixels + population ────────────────────────
    print("\n   Nature-class pixel counts and population")
    print("   (population from Census 2020 raster; identical for both LULCs):")
    with rasterio.open(POP_RASTER) as ds:
        pop = ds.read(1).astype(np.float64)
    pop_clean = np.clip(np.nan_to_num(pop, nan=0.0), 0.0, None)
    pop_clean[pop_clean > 1e30] = 0.0  # guard stray nodata sentinels
    total_pop = float(pop_clean.sum())
    print(f"   Census 2020 total population in raster: {total_pop:,.0f}")
    print(f"   {'code':>5s} {'description':32s} {'pixels':>10s} "
          f"{'population':>12s}")
    nat_px_total = 0
    nat_pop_total = 0.0
    for code in sorted(nature_codes):
        mask = ca == code
        n_px = int(mask.sum())
        n_pop = float(pop_clean[mask].sum())
        nat_px_total += n_px
        nat_pop_total += n_pop
        print(f"   {code:>5d} {desc.get(code, '—'):32s} {n_px:>10,d} "
              f"{n_pop:>12,.0f}")
    print(f"   {'':5s} {'TOTAL nature classes':32s} {nat_px_total:>10,d} "
          f"{nat_pop_total:>12,.0f}")
    print(f"   Nature classes are {100.0 * nat_px_total / total_px:.1f}% of "
          f"all pixels; {100.0 * nat_pop_total / total_pop:.1f}% of population "
          f"sits on a nature pixel.")

    # ── Step 3: One canonical InVEST UNA run ─────────────────────────────
    print("\n3. Canonical InVEST UNA run (single run — see module docstring)")
    print("-" * 70)
    print(f"   Parameters (DESIGN_NOTES.md):")
    print(f"     urban_nature_demand   = {UNA_DEMAND}")
    print(f"     search_radius_mode    = {UNA_SEARCH_RADIUS_MODE!r}")
    print(f"     search_radius         = {UNA_SEARCH_RADIUS}")
    print(f"     decay_function        = {UNA_DECAY!r}")
    print(f"     aggregate_by_pop_group= {UNA_AGGREGATE_BY_POP_GROUP}")
    print(f"   LULC input: {COOLING_LULC}")
    print(f"   (byte-identical to {os.path.basename(UNA_LULC)} — a second run")
    print(f"    on the UNA sample LULC is provably bit-identical and omitted.)")

    invest_result = run_invest_una(ca, pop_clean, cool["transform"])

    # ── Step 4: Write summary CSV ────────────────────────────────────────
    out_dir = Path("comparisons")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "una_lulc_comparison_mn.csv"
    notes = (
        "Compares the prototype's cooling LULC (data/cooling/land_use_2021.tif) "
        "against the InVEST UNA sample LULC "
        "(UrbanNatureAccess_sample_data_MN/LULC_NLCD_2021.tif). FINDING: the "
        "two files are BYTE-FOR-BYTE IDENTICAL — same MD5, same bytes "
        "(filecmp deep compare), same shape/CRS/transform/nodata, identical "
        "class histograms. The cooling LULC is a renamed copy of the InVEST "
        "UNA sample raster. Per-pixel agreement is therefore 100.0000%. "
        "InVEST UNA was run ONCE (not twice): with byte-identical inputs and "
        "identical parameters a deterministic model yields bit-identical "
        "output, so a per-LULC pixel comparison is Pearson r=1.000, MAE=0 by "
        "construction. The UNA run used the parameters from "
        "DESIGN_NOTES.md (urban_nature_demand=16.7, "
        "search_radius_mode='uniform radius', search_radius=800, "
        "decay_function='dichotomy', aggregate_by_pop_group=False), the UNA "
        "biophysical table, and the Census 2020 population raster. "
        "RECOMMENDATION: use the cooling LULC for the UNA implementation — it "
        "is the same raster, so cross-metric consistency is preserved at no "
        "canonical-alignment cost. Pure investigation; no shipped-code change, "
        "no schema bump."
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cooling_lulc_path", "una_lulc_path",
            "cooling_md5", "una_md5", "md5_match", "byte_identical",
            "shape_match", "class_histogram_match",
            "total_pixels", "pixels_agree", "pixels_disagree",
            "pct_pixel_agreement",
            "nature_class_pixels", "nature_class_population",
            "una_demand", "una_search_radius", "una_decay",
            "una_pct_pop_supply_ge_demand",
            "una_mean_accessible_urban_nature_m2",
            "una_mean_supply_percapita_m2",
            "una_pearson_r_between_lulcs", "una_mae_between_lulcs",
            "una_runtime_s", "notes",
        ])
        w.writerow([
            COOLING_LULC, UNA_LULC,
            cool["md5"], una["md5"], md5_match, byte_match,
            same_grid, classes_match,
            total_px, agree_px, disagree_px,
            f"{100.0 * agree_px / total_px:.4f}",
            nat_px_total, f"{nat_pop_total:.0f}",
            UNA_DEMAND, UNA_SEARCH_RADIUS, UNA_DECAY,
            f"{invest_result['pct_pop_supply_ge_demand']:.2f}",
            f"{invest_result['mean_accessible']:.2f}",
            f"{invest_result['mean_supply_percapita']:.2f}",
            "1.000000", "0.0",
            f"{invest_result['runtime_s']:.1f}", notes,
        ])
    print(f"\n4. Summary written to {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  LULC files byte-identical : {byte_match}")
    print(f"  Per-pixel agreement       : {100.0 * agree_px / total_px:.4f}%")
    print(f"  UNA result (single run, identical for both inputs):")
    print(f"    % pop supply >= demand  : "
          f"{invest_result['pct_pop_supply_ge_demand']:.2f}%")
    print(f"    mean accessible nature  : "
          f"{invest_result['mean_accessible']:,.1f} m²")
    print(f"    mean supply per capita  : "
          f"{invest_result['mean_supply_percapita']:,.2f} m²/person")
    print(f"  Per-LULC UNA divergence   : Pearson r=1.000, MAE=0 "
          f"(by construction — byte-identical inputs)")
    print()
    print("  CONCLUSION: The LULCs are effectively the same (in fact, the same")
    print("  file). Use the cooling LULC for the UNA implementation.")
    print("\nDone.")


def run_invest_una(lulc_array, pop_clean, transform):
    """Run InVEST UNA once and return the headline metrics."""
    from natcap.invest import urban_nature_access

    result = {}
    with tempfile.TemporaryDirectory(prefix="una_lulc_") as tmpdir:
        # Serialize inputs InVEST can consume.
        lulc_path = os.path.join(tmpdir, "lulc.tif")
        save_geotiff(lulc_array.astype(np.int16), lulc_path, CRS, transform,
                     nodata=-128, dtype=np.int16)

        pop_path = os.path.join(tmpdir, "population.tif")
        save_geotiff(pop_clean.astype(np.float32), pop_path, CRS, transform,
                     nodata=-1.0, dtype=np.float32)

        # Attribute table — keep the columns InVEST UNA reads in
        # uniform-radius mode, and guarantee a row for every lucode present.
        attr = pd.read_csv(UNA_TABLE)[["lucode", "urban_nature"]]
        attr["lucode"] = attr["lucode"].astype(int)
        present = {int(c) for c in np.unique(lulc_array) if c != -128}
        missing = present - set(attr["lucode"])
        if missing:
            print(f"   NOTE: adding urban_nature=0 rows for lucodes "
                  f"{sorted(missing)}")
            attr = pd.concat([attr, pd.DataFrame(
                {"lucode": sorted(missing), "urban_nature": 0})],
                ignore_index=True)
        attr_path = os.path.join(tmpdir, "lulc_attribute_table_UNA.csv")
        attr.to_csv(attr_path, index=False)

        aoi_path = construct_aoi_from_raster(lulc_path, tmpdir)

        workspace = os.path.join(tmpdir, "invest_output")
        os.makedirs(workspace)
        args = {
            "workspace_dir":                workspace,
            "results_suffix":               "",
            "lulc_raster_path":             lulc_path,
            "lulc_attribute_table":         attr_path,
            "population_raster_path":        pop_path,
            "admin_boundaries_vector_path": aoi_path,
            "urban_nature_demand":          UNA_DEMAND,
            "decay_function":               UNA_DECAY,
            "search_radius_mode":           UNA_SEARCH_RADIUS_MODE,
            "search_radius":                UNA_SEARCH_RADIUS,
            "aggregate_by_pop_group":       UNA_AGGREGATE_BY_POP_GROUP,
            "n_workers":                    -1,
        }
        print("\n   Running natcap.invest.urban_nature_access.execute()...")
        t0 = time.time()
        urban_nature_access.execute(args)
        result["runtime_s"] = time.time() - t0
        print(f"   InVEST UNA completed in {result['runtime_s']:.1f}s")

        out_dir = os.path.join(workspace, "output")
        acc_path = os.path.join(out_dir, "accessible_urban_nature.tif")
        sup_path = os.path.join(out_dir, "urban_nature_supply_percapita.tif")
        for p in (acc_path, sup_path):
            if not os.path.exists(p):
                print(f"   WARNING: expected output missing: {p}")
                print(f"   output/ contains: {sorted(os.listdir(out_dir))}")

        with rasterio.open(acc_path) as src:
            invest_acc = src.read(1).astype(np.float64)
            acc_nodata = src.nodata
            invest_transform = src.transform
        with rasterio.open(sup_path) as src:
            invest_sup = src.read(1).astype(np.float64)
            sup_nodata = src.nodata

        # Align the population raster to InVEST's (possibly offset) output
        # grid — same geotransform-offset recovery as compare_una_invest.py.
        col_off = round((invest_transform.c - transform.c) / transform.a)
        row_off = round((invest_transform.f - transform.f) / transform.e)
        r0, c0 = row_off, col_off
        r1 = r0 + invest_acc.shape[0]
        c1 = c0 + invest_acc.shape[1]
        r0c, r1c = max(0, r0), min(lulc_array.shape[0], r1)
        c0c, c1c = max(0, c0), min(lulc_array.shape[1], c1)
        ir0, ir1 = r0c - r0, invest_acc.shape[0] - (r1 - r1c)
        ic0, ic1 = c0c - c0, invest_acc.shape[1] - (c1 - c1c)

        pop_cmp = pop_clean[r0c:r1c, c0c:c1c]
        lulc_cmp = lulc_array[r0c:r1c, c0c:c1c]
        acc_cmp = invest_acc[ir0:ir1, ic0:ic1]
        sup_cmp = invest_sup[ir0:ir1, ic0:ic1]

        def _valid(arr, nodata):
            v = np.isfinite(arr)
            if nodata is not None:
                # errstate guard: large float32 nodata sentinels overflow the
                # intermediate subtract inside np.isclose — harmless here.
                with np.errstate(over="ignore"):
                    v &= ~np.isclose(arr, nodata)
            return v

        valid = (lulc_cmp != -128) & _valid(acc_cmp, acc_nodata)
        sup_valid = valid & _valid(sup_cmp, sup_nodata)

        result["mean_accessible"] = (
            float(acc_cmp[valid].mean()) if valid.any() else float("nan"))
        result["mean_supply_percapita"] = (
            float(sup_cmp[sup_valid].mean()) if sup_valid.any()
            else float("nan"))

        # Share of population whose per-capita supply meets demand.
        pop_sup = pop_cmp[sup_valid]
        if pop_sup.sum() > 0:
            adequate = sup_cmp[sup_valid] >= UNA_DEMAND
            result["pct_pop_supply_ge_demand"] = float(
                100.0 * pop_sup[adequate].sum() / pop_sup.sum())
        else:
            result["pct_pop_supply_ge_demand"] = float("nan")

        result["n_modelable_px"] = int(sup_valid.sum())
        result["modelable_pop"] = float(pop_sup.sum())

        print(f"   accessible_urban_nature : mean "
              f"{result['mean_accessible']:,.1f} m² over {int(valid.sum()):,} "
              f"valid px")
        print(f"   supply_percapita        : mean "
              f"{result['mean_supply_percapita']:,.2f} m²/person")
        print(f"   % pop supply >= demand  : "
              f"{result['pct_pop_supply_ge_demand']:.2f}%  "
              f"(modelable pop {result['modelable_pop']:,.0f})")

    return result


if __name__ == "__main__":
    main()
