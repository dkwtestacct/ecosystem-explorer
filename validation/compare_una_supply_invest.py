#!/usr/bin/env python3
"""compare_una_supply_invest.py — UNA per-pixel `supply_percapita` parity:
the app's numpy `calculate_nature_access` (`_una_supply_percapita`) vs canonical
`natcap.invest.urban_nature_access` 3.19.0.

This is the Relay-69/71-grade UNA reproducer the prior ungrounded per-pixel claim lacked.
It puts the app's numpy 2SFCA port UNDER TEST against an INDEPENDENT InVEST
computation on MATCHED inputs — NOT InVEST-vs-itself (the flaw in the retired
`una_lulc_comparison_mn.csv`, which compared InVEST's output to a run on its own
byte-identical sample LULC and so trivially scored r=1.0).

Two-environment design (mirrors compare_umh_invest.py — InVEST 3.19.0 needs
py>=3.10, which the app's py3.9 .venv can't host):

  1. EXPORT  (app .venv):
         PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
         GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
         .venv/bin/python validation/compare_una_supply_invest.py export
     Imports `app`, rebinds MN, and writes to tests/una_fixtures/minneapolis_mn/:
       app_supply.tif       — the app's per-pixel supply_percapita
                              (`_una_supply_percapita`, the shipped numpy port)
       app_supply_pert.tif  — the SAME port fed a +2% population input
                              (the non-vacuous guard: a perturbed input must
                              break parity vs InVEST)
       lulc.tif, pop.tif    — the inputs (also fed to InVEST — MATCHED)
       una_attr.csv         — the LULC→urban_nature attribute table InVEST reads
       params.json          — crs, search_radius, demand, decay

  2. COMPARE (isolated natcap_umh_validation conda env, InVEST 3.19.0):
         conda run -n natcap_umh_validation python \
             validation/compare_una_supply_invest.py compare
     Feeds the SAME lulc/pop/attr/params into InVEST UNA execute(), reads its
     INDEPENDENT `urban_nature_supply_percapita.tif`, aligns it to the app grid,
     and compares per-pixel vs app_supply. Writes comparisons/una_supply_parity_mn.csv
     with CLEAN verdict + non-vacuous guard.

Run EXPORT first, then COMPARE.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

FIX = Path("tests/una_fixtures")
CITY = "Minneapolis, MN"
SLUG = "minneapolis_mn"

# Parity pass criterion (composed AND). supply_percapita is a large-magnitude
# continuous field (~1e5 m²/person), so the per-pixel error bar is RELATIVE
# (MAE / mean), not absolute — an absolute m²/person threshold would be
# meaningless at this scale. Thresholds sit orders looser than float32
# round-trip noise so a real port regression (wrong kernel, ratio rule, radius)
# trips at least one, while storage/edge-alignment residual passes.
_PARITY_MAX_REL_MAE  = 1.0e-4   # relative per-pixel MAE (MAE / mean supply)
_PARITY_MIN_R        = 0.999
_PARITY_MAX_REL_TOTAL = 0.005   # 0.5% relative total divergence
_GUARD_POP_SCALE     = 1.02     # +2% population for the perturbed-input guard


def _slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_")


# ────────────────────────────────────────────────────────────────────────────
# EXPORT mode (app .venv)
# ────────────────────────────────────────────────────────────────────────────
def run_export() -> int:
    import pandas as pd
    import rasterio
    import verify_baselines as vb  # reuse its Streamlit stub + _rebind_city

    sys.modules["streamlit"] = vb._StubSt()
    print("Importing app (Streamlit stub)...")
    import app  # noqa: E402

    state = vb._rebind_city(app, CITY)
    cfg = app.CITIES[CITY]

    # Baseline UNA-view LULC (MN: plain NLCD cooling_lulc — no compound view).
    lulc = np.asarray(state.cooling_lulc)
    pop = np.asarray(state.pop_count_raster, dtype="float64")
    una_arr = np.asarray(state.urban_nature_arr)

    # The app's shipped per-pixel supply_percapita (numpy 2SFCA port).
    app_supply, valid = app._una_supply_percapita_pure(lulc, pop, una_arr)
    # Guard variant: SAME port, +2% population input → supply must diverge.
    app_supply_pert, _ = app._una_supply_percapita_pure(
        lulc, pop * _GUARD_POP_SCALE, una_arr)

    # Mask supply to the modelable extent (valid LULC); off-extent → nodata.
    NOD = -1.0
    app_supply = np.where(valid, app_supply, NOD).astype("float32")
    app_supply_pert = np.where(valid, app_supply_pert, NOD).astype("float32")

    transform = state.ref_transform
    crs_wkt = rasterio.crs.CRS.from_user_input(cfg["crs"]).to_wkt()
    h, w = lulc.shape
    outdir = FIX / SLUG
    outdir.mkdir(parents=True, exist_ok=True)

    def wr(name, arr, dtype, nodata=None):
        with rasterio.open(
            outdir / name, "w", driver="GTiff", height=h, width=w, count=1,
            dtype=dtype, crs=crs_wkt, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(arr.astype(dtype), 1)

    wr("app_supply.tif", app_supply, "float32", nodata=NOD)
    wr("app_supply_pert.tif", app_supply_pert, "float32", nodata=NOD)
    wr("lulc.tif", lulc.astype("int16"), "int16", nodata=-128)
    wr("pop.tif", np.clip(np.nan_to_num(pop, nan=0.0), 0.0, None).astype("float32"),
       "float32", nodata=-1.0)

    # LULC→urban_nature attribute table InVEST reads (uniform-radius mode needs
    # lucode + urban_nature). Add urban_nature=0 rows for any lucode present in
    # the raster but absent from the app's table (the app treats those as 0).
    una_src = pd.read_csv(cfg["una_table_file"])[["lucode", "urban_nature"]]
    una_src["lucode"] = una_src["lucode"].astype(int)
    present = {int(c) for c in np.unique(lulc) if c != -128}
    missing = present - set(una_src["lucode"])
    if missing:
        print(f"  NOTE: adding urban_nature=0 rows for lucodes {sorted(missing)}")
        una_src = pd.concat([una_src, pd.DataFrame(
            {"lucode": sorted(missing), "urban_nature": 0})], ignore_index=True)
    una_src.to_csv(outdir / "una_attr.csv", index=False)

    params = {
        "city": CITY,
        "crs": cfg["crs"],
        "search_radius_m": float(app.UNA_SEARCH_RADIUS_M),
        "urban_nature_demand": float(app.UNA_DEMAND_M2_PER_CAPITA),
        "decay_function": str(app.UNA_DECAY_FUNCTION),
        "guard_pop_scale": _GUARD_POP_SCALE,
        "app_supply_total": float(app_supply[app_supply != NOD].sum()),
    }
    (outdir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"  wrote fixtures to {outdir}  (grid {h}x{w}, app supply total "
          f"{params['app_supply_total']:,.1f} m²/person)")
    print("EXPORT done.")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# COMPARE mode (isolated natcap_umh_validation env: GDAL + natcap.invest 3.19.0)
# ────────────────────────────────────────────────────────────────────────────
def _gdal_read(path):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray().astype("float64")
    nod = b.GetNoDataValue()
    gt = ds.GetGeoTransform()
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return arr, nod, gt, shape


def _metrics(invest, app_, valid):
    diff = np.abs(invest - app_)
    mae = float(diff[valid].mean()) if valid.any() else float("nan")
    mean_mag = float(np.abs(app_[valid]).mean()) if valid.any() else float("nan")
    rel_mae = mae / mean_mag if mean_mag > 0 else float("nan")
    if valid.sum() > 2 and np.std(invest[valid]) > 0 and np.std(app_[valid]) > 0:
        r = float(np.corrcoef(invest[valid], app_[valid])[0, 1])
    else:
        r = float("nan")
    it, at = float(invest[valid].sum()), float(app_[valid].sum())
    rel = abs(at - it) / abs(it) if it != 0 else float("nan")
    return mae, rel_mae, r, it, at, rel


def _parity(rel_mae, r, rel):
    fails = []
    if not (rel_mae < _PARITY_MAX_REL_MAE):
        fails.append(f"rel_MAE={rel_mae:.3g} ≥ {_PARITY_MAX_REL_MAE:.0e}")
    if not (r > _PARITY_MIN_R):
        fails.append(f"r={r:.6f} ≤ {_PARITY_MIN_R}")
    if not (rel < _PARITY_MAX_REL_TOTAL):
        fails.append(f"|Δtotal|={rel:.3%} ≥ {_PARITY_MAX_REL_TOTAL:.1%}")
    return len(fails) == 0, fails


def _run_invest(tmp, lulc_p, pop_p, attr_p, crs, gt, h, w, demand, decay, radius):
    import geopandas as gpd
    from shapely.geometry import box
    from natcap.invest import urban_nature_access as una
    minx, maxy = gt[0], gt[3]
    maxx, miny = minx + gt[1] * w, maxy + gt[5] * h
    aoi_p = os.path.join(tmp, "aoi.gpkg")
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(minx, miny, maxx, maxy)],
                     crs=crs).to_file(aoi_p, driver="GPKG")
    ws = os.path.join(tmp, "ws")
    os.makedirs(ws, exist_ok=True)
    una.execute({
        "workspace_dir": ws, "results_suffix": "", "n_workers": "-1",
        "lulc_raster_path": str(lulc_p), "lulc_attribute_table": str(attr_p),
        "population_raster_path": str(pop_p), "admin_boundaries_vector_path": aoi_p,
        "urban_nature_demand": demand, "decay_function": decay,
        "search_radius_mode": "uniform radius", "search_radius": radius,
        "aggregate_by_pop_group": False,
    })
    return os.path.join(ws, "output", "urban_nature_supply_percapita.tif")


def run_compare() -> int:
    import csv
    import tempfile
    import natcap.invest as _ni

    d = FIX / SLUG
    if not (d / "params.json").exists():
        print(f"[{CITY}] no fixtures at {d} — run EXPORT first.")
        return 1
    p = json.loads((d / "params.json").read_text())
    inv_ver = getattr(_ni, "__version__", "unknown")

    app_sup, app_nod, gt, (h, w) = _gdal_read(d / "app_supply.tif")
    app_pert, pert_nod, _, _ = _gdal_read(d / "app_supply_pert.tif")
    lulc, lulc_nod, _, _ = _gdal_read(d / "lulc.tif")

    print("=" * 70)
    print(f"UNA supply_percapita parity: app numpy 2SFCA vs canonical "
          f"natcap.invest {inv_ver}")
    print("=" * 70)
    print(f"=== {CITY}  (grid {h}x{w}, radius {p['search_radius_m']:.0f} m, "
          f"demand {p['urban_nature_demand']:.0f}, decay {p['decay_function']}) ===")

    with tempfile.TemporaryDirectory() as tmp:
        sup_path = _run_invest(
            tmp, d / "lulc.tif", d / "pop.tif", d / "una_attr.csv", p["crs"], gt,
            h, w, p["urban_nature_demand"], p["decay_function"], p["search_radius_m"])
        inv, inv_nod, igt, ishape = _gdal_read(sup_path)
        # Align InVEST output (padded/clipped to AOI) back to the app grid.
        dx = int(round((gt[0] - igt[0]) / gt[1]))
        dy = int(round((gt[3] - igt[3]) / gt[5]))
        inv = inv[dy:dy + h, dx:dx + w]
        if inv.shape != (h, w):
            print(f"  WARN aligned InVEST {inv.shape} != app {(h, w)} "
                  f"(dx={dx}, dy={dy}) — abort")
            return 1
        # Compare on the common modelable extent: valid LULC, both supplies
        # finite + non-nodata.
        def _ok(a, nod):
            v = np.isfinite(a)
            if nod is not None:
                v &= ~np.isclose(a, nod)
            return v
        valid = (lulc != -128) & _ok(app_sup, app_nod) & _ok(inv, inv_nod)
        if inv_nod is not None:
            inv = np.where(np.isclose(inv, inv_nod), 0.0, inv)
        inv = np.nan_to_num(inv, nan=0.0)
        app_clean = np.where(_ok(app_sup, app_nod), app_sup, 0.0)
        pert_clean = np.where(_ok(app_pert, pert_nod), app_pert, 0.0)

        mae, rel_mae, r, itot, atot, rel = _metrics(inv, app_clean, valid)
        ok, fails = _parity(rel_mae, r, rel)
        mean_mag = atot / int(valid.sum()) if valid.any() else float("nan")
        print(f"  n_valid={int(valid.sum()):,}  (mean supply ≈ {mean_mag:,.0f} m²/person)")
        print(f"  app total = {atot:,.2f} | InVEST total = {itot:,.2f} m²/person")
        print(f"  per-pixel MAE = {mae:.6g} ({rel_mae:.3g} relative)  | "
              f"Pearson r = {r:.6f}  | |Δtotal| = {rel:.6%}")
        print(f"  parity: {'OK' if ok else 'FAIL — ' + '; '.join(fails)}")

        # Non-vacuous guard: InVEST vs the +2%-population app variant MUST fail.
        gmae, grel_mae, gr, _, _, grel = _metrics(inv, pert_clean, valid)
        g_ok, _ = _parity(grel_mae, gr, grel)
        guard_ok = not g_ok
        print(f"  guard (InVEST vs +{(_GUARD_POP_SCALE - 1) * 100:.0f}% pop app "
              f"variant): rel_MAE={grel_mae:.3g}, r={gr:.6f} → "
              f"{'tripped (good)' if guard_ok else 'DID NOT TRIP (vacuous!)'}")

        clean = ok and guard_ok
        art = Path("comparisons/una_supply_parity_mn.csv")
        art.parent.mkdir(exist_ok=True)
        row = {
            "comparison": "MN UNA per-pixel supply_percapita (baseline)",
            "city": CITY, "invest_version": inv_ver,
            "per_pixel_mae": f"{mae:.6e}", "per_pixel_rel_mae": f"{rel_mae:.6e}",
            "pearson_r": f"{r:.10f}",
            "aoi_sum_pct_diff": f"{rel * 100.0:.6f}",
            "app_total_m2_percapita": f"{atot:.4f}",
            "invest_total_m2_percapita": f"{itot:.4f}",
            "guard_mae": f"{gmae:.6e}", "guard_ok": guard_ok,
            "n_pixels_compared": int(valid.sum()),
            "search_radius_m": f"{p['search_radius_m']:.0f}",
            "urban_nature_demand": f"{p['urban_nature_demand']:.0f}",
            "decay_function": p["decay_function"], "clean": clean,
            "notes": (
                "MN UNA per-pixel supply_percapita parity: app numpy 2SFCA "
                "(_una_supply_percapita) vs canonical natcap.invest "
                "urban_nature_access urban_nature_supply_percapita, fed the "
                "IDENTICAL LULC + population + urban_nature attribute table + "
                "params (uniform 1000 m radius, exponential decay, demand 250). "
                "MATCHED-BUT-INDEPENDENT: InVEST computes its own supply via its "
                "taskgraph 2SFCA; this puts the app's numpy port under test, NOT "
                "InVEST-vs-itself (unlike the retired una_lulc_comparison_mn.csv). "
                "Non-vacuous guard: feeding the app port a +2% population input "
                "diverges it from InVEST (guard parity trips)."
            ),
        }
        with open(art, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
            wr.writeheader()
            wr.writerow(row)
        print(f"\nWrote → {art}  (clean={clean}, guard_ok={guard_ok})")
        return 0 if (clean) else 1


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "export":
        return run_export()
    elif mode == "compare":
        return run_compare()
    print(__doc__)
    print("ERROR: specify 'export' (app .venv) or 'compare' (3.19.0 env).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
