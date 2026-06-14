#!/usr/bin/env python3
"""compare_ucm_invest.py — UCM per-pixel HMI parity: the app's smoothed-CC /
Heat Mitigation Index (`_compute_hmi_raster`, the shipped `baseline_hm_raster`)
vs canonical `natcap.invest.urban_cooling_model` 3.19.0's `hm.tif`.

Two-environment design (mirrors compare_umh/una — InVEST 3.19.0 needs py>=3.10,
which the app's py3.9 .venv can't host; the conda 3.16.2 base would pin the run
to a stale InVEST). Pinning to 3.19.0 keeps the validated set on one canonical
version with no carve-out.

  1. EXPORT  (app .venv):
         PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
         GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
         .venv/bin/python validation/compare_ucm_invest.py export
     Imports `app`, rebinds MN, writes to tests/ucm_fixtures/minneapolis_mn/:
       app_hm.tif        — the app's HMI = max(CC_local, CC_park) (baseline_hm_raster)
       lulc.tif, et.tif  — the UCM inputs (also fed to InVEST — MATCHED)
       params.json       — biophysical-table path + UCM args (the published MN config)

  2. COMPARE (isolated natcap_umh_validation conda env, InVEST 3.19.0):
         conda run -n natcap_umh_validation python \
             validation/compare_ucm_invest.py compare
     Feeds the SAME lulc/et/biophysical-table/args into InVEST UCM execute(),
     reads its INDEPENDENT hm.tif, aligns to the app grid, and compares per-pixel
     vs app_hm. Writes comparisons/ucm_baseline_mn.csv with CLEAN + non-vacuous guard.

Run EXPORT first, then COMPARE.

Scope: the HMI / Temperature-Change reading (the validated card). The separate
Cooling Energy Savings divergence (per-pixel vs per-building T_air sampling) is
NOT this metric and is documented in REFERENCE §8.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

FIX = Path("tests/ucm_fixtures")
CITY = "Minneapolis, MN"
SLUG = "minneapolis_mn"

# Parity pass criterion (composed AND). HMI ∈ [0,1]; absolute thresholds are
# appropriate at this scale. Loose of float32 round-trip noise so a real kernel
# regression (wrong decay, park threshold, CC weights) trips at least one.
_PARITY_MAX_MAE       = 1.0e-4
_PARITY_MIN_R         = 0.9999
_PARITY_MAX_REL_TOTAL = 0.005
_GUARD_SCALE          = 1.005   # +0.5% on app HMI for the perturbed guard


# ────────────────────────────────────────────────────────────────────────────
# EXPORT mode (app .venv)
# ────────────────────────────────────────────────────────────────────────────
def run_export() -> int:
    import rasterio
    import verify_baselines as vb

    sys.modules["streamlit"] = vb._StubSt()
    print("Importing app (Streamlit stub)...")
    import app  # noqa: E402

    state = vb._rebind_city(app, CITY)
    cfg = app.CITIES[CITY]

    app_hm = np.asarray(state.baseline_hm_raster, dtype="float32")
    lulc = np.asarray(state.cooling_lulc).astype("int16")
    et = np.asarray(state.et_resized, dtype="float32")
    transform = state.ref_transform
    crs_wkt = rasterio.crs.CRS.from_user_input(cfg["crs"]).to_wkt()
    h, w = app_hm.shape
    outdir = FIX / SLUG
    outdir.mkdir(parents=True, exist_ok=True)

    def wr(name, arr, dtype, nodata=None):
        with rasterio.open(
            outdir / name, "w", driver="GTiff", height=h, width=w, count=1,
            dtype=dtype, crs=crs_wkt, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(arr.astype(dtype), 1)

    NOD = -9999.0
    wr("app_hm.tif", np.where(np.isfinite(app_hm), app_hm, NOD), "float32", nodata=NOD)
    wr("lulc.tif", lulc, "int16", nodata=-128)
    wr("et.tif", et, "float32", nodata=-1.0)

    # Biophysical table: copy into the fixture dir so COMPARE is self-contained.
    bio_src = os.path.join(cfg["data_dir_cooling"], cfg["cooling_table_file"])
    shutil.copy(bio_src, outdir / "biophysical_table.csv")

    params = {
        "city": CITY,
        "crs": cfg["crs"],
        # Published MN UCM config (invest_urban_cooling_model_args_MN.json) —
        # the same parameters the app's cooling engine uses.
        "args": {
            "green_area_cooling_distance": 450,
            "t_air_average_radius": 600,
            "t_ref": 23.2,
            "uhi_max": float(app.UHI_MAX_C),
            "cc_method": "factors",
            "do_energy_valuation": False,
            "do_productivity_valuation": False,
        },
        "app_hm_mean": float(app_hm[np.isfinite(app_hm)].mean()),
    }
    (outdir / "params.json").write_text(json.dumps(params, indent=2))
    print(f"  wrote fixtures to {outdir}  (grid {h}x{w}, app HMI mean "
          f"{params['app_hm_mean']:.4f})")
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
    if valid.sum() > 2 and np.std(invest[valid]) > 0 and np.std(app_[valid]) > 0:
        r = float(np.corrcoef(invest[valid], app_[valid])[0, 1])
    else:
        r = float("nan")
    it, at = float(invest[valid].sum()), float(app_[valid].sum())
    rel = abs(at - it) / abs(it) if it != 0 else float("nan")
    return mae, r, it, at, rel


def _parity(mae, r, rel):
    fails = []
    if not (mae < _PARITY_MAX_MAE):
        fails.append(f"MAE={mae:.3g} ≥ {_PARITY_MAX_MAE:.0e}")
    if not (r > _PARITY_MIN_R):
        fails.append(f"r={r:.6f} ≤ {_PARITY_MIN_R}")
    if not (rel < _PARITY_MAX_REL_TOTAL):
        fails.append(f"|Δtotal|={rel:.3%} ≥ {_PARITY_MAX_REL_TOTAL:.1%}")
    return len(fails) == 0, fails


def run_compare() -> int:
    import csv
    import tempfile
    import natcap.invest as _ni
    import geopandas as gpd
    from shapely.geometry import box
    from natcap.invest import urban_cooling_model as ucm

    d = FIX / SLUG
    if not (d / "params.json").exists():
        print(f"[{CITY}] no fixtures at {d} — run EXPORT first.")
        return 1
    p = json.loads((d / "params.json").read_text())
    inv_ver = getattr(_ni, "__version__", "unknown")
    a = p["args"]

    app_hm, app_nod, gt, (h, w) = _gdal_read(d / "app_hm.tif")

    print("=" * 70)
    print(f"UCM HMI parity: app smoothed-CC/HMI vs canonical natcap.invest {inv_ver}")
    print("=" * 70)
    print(f"=== {CITY}  (grid {h}x{w}) ===")

    with tempfile.TemporaryDirectory() as tmp:
        # AOI bbox from the app geotransform (covers the full app extent).
        minx, maxy = gt[0], gt[3]
        maxx, miny = minx + gt[1] * w, maxy + gt[5] * h
        aoi_p = os.path.join(tmp, "aoi.gpkg")
        gpd.GeoDataFrame({"id": [1]}, geometry=[box(minx, miny, maxx, maxy)],
                         crs=p["crs"]).to_file(aoi_p, driver="GPKG")
        ws = os.path.join(tmp, "ws")
        os.makedirs(ws, exist_ok=True)
        ucm.execute({
            "workspace_dir": ws, "n_workers": "-1",
            "lulc_raster_path": str(d / "lulc.tif"),
            "ref_eto_raster_path": str(d / "et.tif"),
            "aoi_vector_path": aoi_p,
            "biophysical_table_path": str(d / "biophysical_table.csv"),
            "green_area_cooling_distance": a["green_area_cooling_distance"],
            "t_air_average_radius": a["t_air_average_radius"],
            "t_ref": a["t_ref"], "uhi_max": a["uhi_max"],
            "cc_method": a["cc_method"],
            "do_energy_valuation": a["do_energy_valuation"],
            "do_productivity_valuation": a["do_productivity_valuation"],
        })
        inv, inv_nod, igt, ishape = _gdal_read(os.path.join(ws, "hm.tif"))
        dx = int(round((gt[0] - igt[0]) / gt[1]))
        dy = int(round((gt[3] - igt[3]) / gt[5]))
        inv = inv[dy:dy + h, dx:dx + w]
        if inv.shape != (h, w):
            print(f"  WARN aligned InVEST {inv.shape} != app {(h, w)} "
                  f"(dx={dx}, dy={dy}) — abort")
            return 1

        def _ok(arr, nod):
            v = np.isfinite(arr)
            if nod is not None:
                v &= ~np.isclose(arr, nod)
            return v
        valid = _ok(app_hm, app_nod) & _ok(inv, inv_nod)
        app_clean = np.where(_ok(app_hm, app_nod), app_hm, 0.0)
        inv_clean = np.where(_ok(inv, inv_nod), inv, 0.0)

        mae, r, itot, atot, rel = _metrics(inv_clean, app_clean, valid)
        ok, fails = _parity(mae, r, rel)
        print(f"  n_valid={int(valid.sum()):,}")
        print(f"  app HMI mean = {atot / max(1, int(valid.sum())):.4f} | "
              f"InVEST HMI mean = {itot / max(1, int(valid.sum())):.4f}")
        print(f"  per-pixel MAE = {mae:.6g}  | Pearson r = {r:.6f}  | "
              f"|Δtotal| = {rel:.6%}")
        print(f"  parity: {'OK' if ok else 'FAIL — ' + '; '.join(fails)}")

        # Non-vacuous guard: InVEST vs a +0.5%-scaled app HMI MUST fail parity.
        g_mae, g_r, _, _, g_rel = _metrics(inv_clean, app_clean * _GUARD_SCALE, valid)
        g_ok, _ = _parity(g_mae, g_r, g_rel)
        guard_ok = not g_ok
        print(f"  guard (InVEST vs +{(_GUARD_SCALE - 1) * 100:.1f}% app HMI): "
              f"MAE={g_mae:.3g} → {'tripped (good)' if guard_ok else 'VACUOUS!'}")

        clean = ok and guard_ok
        art = Path("comparisons/ucm_baseline_mn.csv")
        art.parent.mkdir(exist_ok=True)
        n_large = int((np.abs(inv_clean - app_clean)[valid] > 0.1).sum())
        row = {
            "comparison": "MN UCM Heat Mitigation Index (baseline)",
            "city": CITY, "invest_version": inv_ver,
            "per_pixel_mae": f"{mae:.6e}", "pearson_r": f"{r:.10f}",
            "aoi_sum_pct_diff": f"{rel * 100.0:.6f}",
            "hmi_large_divergences_gt_0.1": n_large,
            "app_hm_mean": f"{atot / max(1, int(valid.sum())):.6f}",
            "invest_hm_mean": f"{itot / max(1, int(valid.sum())):.6f}",
            "guard_mae": f"{g_mae:.6e}", "guard_ok": guard_ok,
            "n_valid_pixels": int(valid.sum()), "clean": clean,
            "notes": (
                "MN UCM per-pixel Heat Mitigation Index parity: the app's HMI = "
                "max(CC_local, CC_park) with 2-ha park threshold + exponential "
                "decay over d_cool=450 m (_compute_hmi_raster, baseline_hm_raster) "
                "vs canonical natcap.invest.urban_cooling_model hm.tif, fed the "
                "IDENTICAL LULC + ET + biophysical table + args (t_ref=23.2, "
                "uhi_max, t_air_radius=600, green_dist=450, cc_method=factors). "
                "MATCHED-BUT-INDEPENDENT: InVEST computes its own HMI. Non-vacuous "
                "guard: +0.5% scaling of the app HMI trips parity. Scope: HMI / "
                "Temperature Change only (the separate Cooling Energy Savings "
                "per-building-sampling divergence is documented in REFERENCE §8). "
                "natcap.invest 3.19.0."
            ),
        }
        with open(art, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
            wr.writeheader()
            wr.writerow(row)
        print(f"\nWrote → {art}  (clean={clean}, guard_ok={guard_ok})")
        return 0 if clean else 1


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
