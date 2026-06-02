#!/usr/bin/env python3
"""compare_umh_invest.py — validate the prototype's Urban Mental Health
reimplementation against canonical natcap.invest.urban_mental_health (v3.19.0).

Two-environment design (canonical InVEST 3.19.0 needs Python >=3.10, which the
app's py3.9 .venv can't host; see docs/dev/CONTRIBUTING.md "Canonical-InVEST validation").

  1. EXPORT  (run in the app .venv, which has app's full stack):
         PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
         GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
         .venv/bin/python validation/compare_umh_invest.py export
     Imports `app`, and for each city writes to tests/umh_fixtures/<slug>/:
       ndvi_base.tif, ndvi_alt.tif, pop.tif  (shared inputs, NLCD-space proxy)
       proto_pc_dep.tif, proto_pc_anx.tif    (prototype per-pixel preventable
                                              cases, computed with app's real
                                              UMH constants — faithful, no
                                              app code changes)
       params.json                           (constants + CRS the compare step needs)

  2. COMPARE (run in the isolated natcap_umh_validation conda env):
         conda run -n natcap_umh_validation python validation/compare_umh_invest.py compare
     Feeds the SAME ndvi_base/ndvi_alt/pop into canonical UMH execute()
     (model_option='ndvi'), per outcome, and reports MAE + Pearson r of the
     canonical `preventable_cases` raster vs the prototype's.

Run EXPORT first, then COMPARE.

Two MAE numbers are reported (see the brief):
  - Algorithmic-fidelity: matched inputs (single-polygon AOI carrying the
    prototype's uniform CDC rate; effect_size = the prototype's per-outcome RR).
    Isolates the NE-kernel + arithmetic divergence.
  - Default-input: realistic canonical configuration. (Per-admin prevalence
    would belong here but we have no per-tract MH-prevalence data for MN/SA,
    so it currently coincides with matched on the prevalence axis — documented.)

NOTE: validation is on the prototype's SYNTHETIC per-NLCD NDVI proxy, NOT
satellite NDVI — it validates the reimplementation's algorithm, not the NDVI
source.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

FIX = Path("tests/umh_fixtures")
CITIES_TO_RUN = ["Minneapolis, MN", "San Antonio, TX"]
SCENARIO = dict(pct_converted=10, green_infrastructure_pct=0, food_forest_pct=100)


def _slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_")


# ────────────────────────────────────────────────────────────────────────────
# EXPORT mode (app .venv)
# ────────────────────────────────────────────────────────────────────────────
def run_export() -> int:
    import rasterio
    import verify_baselines as vb  # reuse its Streamlit stub + _rebind_city

    sys.modules["streamlit"] = vb._StubSt()
    print("Importing app (Streamlit stub)...")
    import app  # noqa: E402

    crs_wkt_cache = {}

    def _wkt(epsg_str):
        if epsg_str not in crs_wkt_cache:
            crs_wkt_cache[epsg_str] = rasterio.crs.CRS.from_user_input(epsg_str).to_wkt()
        return crs_wkt_cache[epsg_str]

    for city in CITIES_TO_RUN:
        print(f"\n=== EXPORT {city} ===")
        state = vb._rebind_city(app, city)
        cfg = app.CITIES[city]

        ndvi_base = app._lulc_to_ndvi_raster(state.cooling_lulc).astype("float32")
        res = app.evaluate_scenario(**SCENARIO, seed=42, placement_strategy="random")
        ndvi_alt = app._lulc_to_ndvi_raster(res["scenario_lulc"]).astype("float32")
        pop = np.asarray(state.pop_count_raster, dtype="float32")

        # Prototype per-pixel preventable cases, using the app's ACTUAL NE
        # kernel + constants (so this harness tracks the shipped UMH code, not
        # an inline copy). `_umh_neighborhood_exposure` is the buffer-mean NE
        # used by calculate_mental_health_impact.
        ne_b = app._umh_neighborhood_exposure(ndvi_base)
        ne_a = app._umh_neighborhood_exposure(ndvi_alt)
        d_ne = ne_a - ne_b
        pc_dep = ((1.0 - np.exp(app._UMH_LN_RR_DEPRESSION * 10 * d_ne))
                  * app.BIR_DEPRESSION * pop).astype("float32")
        pc_anx = ((1.0 - np.exp(app._UMH_LN_RR_ANXIETY * 10 * d_ne))
                  * app.BIR_ANXIETY * pop).astype("float32")

        transform = state.ref_transform
        wkt = _wkt(cfg["crs"])
        h, w = ndvi_base.shape
        outdir = FIX / _slug(city)
        outdir.mkdir(parents=True, exist_ok=True)

        def wr(name, arr, nodata=None):
            with rasterio.open(
                outdir / name, "w", driver="GTiff", height=h, width=w, count=1,
                dtype="float32", crs=wkt, transform=transform, nodata=nodata,
            ) as dst:
                dst.write(arr.astype("float32"), 1)

        wr("ndvi_base.tif", ndvi_base)
        wr("ndvi_alt.tif", ndvi_alt)
        wr("pop.tif", pop, nodata=-1.0)
        wr("proto_pc_dep.tif", pc_dep)
        wr("proto_pc_anx.tif", pc_anx)

        params = {
            "city": city,
            "crs": cfg["crs"],
            "search_radius_m": float(app.UMH_SEARCH_RADIUS_M),
            "pixel_size_m": float(app.PIXEL_SIZE_M),
            "outcomes": {
                "dep": {"rr_0_1": float(app.RR_0_1_NDVI_DEPRESSION),
                        "bir": float(app.BIR_DEPRESSION),
                        "cost": float(app.COST_PER_DEPRESSION_CASE_USD)},
                "anx": {"rr_0_1": float(app.RR_0_1_NDVI_ANXIETY),
                        "bir": float(app.BIR_ANXIETY),
                        "cost": float(app.COST_PER_ANXIETY_CASE_USD)},
            },
            "scenario": SCENARIO,
            "proto_totals": {"dep": float(pc_dep.sum()), "anx": float(pc_anx.sum())},
        }
        (outdir / "params.json").write_text(json.dumps(params, indent=2))
        print(f"  wrote fixtures to {outdir}  (grid {h}x{w}, "
              f"proto dep total {pc_dep.sum():,.1f}, anx {pc_anx.sum():,.1f})")
    print("\nEXPORT done.")
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
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return arr, nod, gt, proj, shape


def _run_canonical_outcome(umh, tmp, suffix, ndvi_b, ndvi_a, pop_p, crs, gt, h, w,
                           search_radius, effect_size, bir, cost):
    import geopandas as gpd
    from shapely.geometry import box
    minx, maxy = gt[0], gt[3]
    maxx, miny = minx + gt[1] * w, maxy + gt[5] * h
    geom = [box(minx, miny, maxx, maxy)]
    aoi_p = os.path.join(tmp, f"aoi_{suffix}.gpkg")
    prev_p = os.path.join(tmp, f"prev_{suffix}.gpkg")
    gpd.GeoDataFrame({"id": [1]}, geometry=geom, crs=crs).to_file(aoi_p, driver="GPKG")
    gpd.GeoDataFrame({"risk_rate": [bir]}, geometry=geom, crs=crs).to_file(prev_p, driver="GPKG")
    ws = os.path.join(tmp, f"ws_{suffix}")
    os.makedirs(ws, exist_ok=True)
    umh.execute({
        "workspace_dir": ws, "results_suffix": suffix, "n_workers": "-1",
        "aoi_path": aoi_p, "population_raster": pop_p,
        "search_radius": float(search_radius), "effect_size": float(effect_size),
        "baseline_prevalence_vector": prev_p, "health_cost_rate": float(cost),
        "model_option": "ndvi", "ndvi_base": ndvi_b, "ndvi_alt": ndvi_a,
    })
    # locate the canonical per-pixel preventable_cases raster
    hits = []
    for root, _, files in os.walk(ws):
        for f in files:
            if f.startswith("preventable_cases") and f.endswith(".tif"):
                hits.append(os.path.join(root, f))
    if not hits:
        raise FileNotFoundError(f"no preventable_cases*.tif under {ws}")
    return sorted(hits, key=len)[0]


def _metrics(canon, proto, active):
    diff = np.abs(canon - proto)
    mae_all = float(diff.mean())
    mae_act = float(diff[active].mean()) if active.any() else float("nan")
    if active.sum() > 2 and np.std(canon[active]) > 0 and np.std(proto[active]) > 0:
        r = float(np.corrcoef(canon[active], proto[active])[0, 1])
    else:
        r = float("nan")
    proto_tot, canon_tot = float(proto.sum()), float(canon.sum())
    return mae_all, mae_act, r, proto_tot, canon_tot


# Parity pass criterion. Three thresholds composed as AND — a regression in
# kernel formula, radius, or per-pixel arithmetic must trip at least one.
# Justification (measured values; see git log of this file for the run):
#   MN dep/anx: MAE(active) ≤ 1.1e-9, r = 1.000000, |Δtotal|/total = 0
#   SA dep/anx: MAE(active) ≤ 2.3e-6, r ≥ 0.99876,  |Δtotal|/total ≤ 0.15%
# The SA residual is canonical's radius padding + edge-crop alignment +
# pygeoprocessing FFT noise on the 1713×1984 grid, not a metric divergence
# (DESIGN_NOTES §6.3). The thresholds sit ~3–5× looser than the measured
# worst-case so normal noise passes but a real kernel/constant regression
# (radius bump, wrong RR, wrong baseline_prevalence) tips at least one.
_PARITY_MAX_MAE_ACT      = 1.0e-5   # cases / pixel
_PARITY_MIN_R            = 0.99
_PARITY_MAX_REL_TOTAL    = 0.005    # 0.5% relative total divergence

# Synthetic perturbation factor for --meta-test mode. 0.5% bias is the
# smallest scaling that confidently trips _PARITY_MAX_REL_TOTAL on every
# city/outcome — see meta-test logic in run_compare.
_META_TEST_PROTO_SCALE   = 1.005


def _parity_check(city, oc, mae_act, r, proto_tot, canon_tot):
    """Return (ok, fail_reasons[]). Compose three thresholds as AND."""
    fails = []
    if not (mae_act < _PARITY_MAX_MAE_ACT):
        fails.append(f"MAE(active)={mae_act:.3g} ≥ {_PARITY_MAX_MAE_ACT:.0e}")
    if not (r > _PARITY_MIN_R):
        fails.append(f"r={r:.6f} ≤ {_PARITY_MIN_R}")
    rel = abs(proto_tot - canon_tot) / abs(canon_tot) if canon_tot != 0 else float("inf")
    if not (rel < _PARITY_MAX_REL_TOTAL):
        fails.append(f"|Δtotal|/total={rel:.3%} ≥ {_PARITY_MAX_REL_TOTAL:.1%}")
    return len(fails) == 0, fails


def run_compare(fix_dir, cities, slug_fn, meta_test: bool = False) -> int:
    import tempfile
    from natcap.invest import urban_mental_health as umh

    print("=" * 70)
    print("UMH validation: prototype numpy reimpl vs canonical natcap.invest 3.19.0")
    print("=" * 70)
    print("NOTE: inputs are the prototype's SYNTHETIC per-NLCD NDVI proxy "
          "(validates the algorithm, not the NDVI source).")

    rc = 0
    for city in cities:
        d = fix_dir / slug_fn(city)
        if not (d / "params.json").exists():
            print(f"\n[{city}] no fixtures at {d} — run EXPORT first."); rc = 1; continue
        p = json.loads((d / "params.json").read_text())
        crs = p["crs"]; sr = p["search_radius_m"]
        proto_dep, _, gt, _, (h, w) = _gdal_read(d / "proto_pc_dep.tif")
        proto_anx, *_ = _gdal_read(d / "proto_pc_anx.tif")
        pop, _, _, _, _ = _gdal_read(d / "pop.tif")
        ndvi_b, ndvi_a = str(d / "ndvi_base.tif"), str(d / "ndvi_alt.tif")
        pop_p = str(d / "pop.tif")
        active = pop > 0

        print(f"\n=== {city}  (grid {h}x{w}, search_radius {sr:.0f} m) ===")
        with tempfile.TemporaryDirectory() as tmp:
            for oc, proto in (("dep", proto_dep), ("anx", proto_anx)):
                rr = p["outcomes"][oc]["rr_0_1"]; bir = p["outcomes"][oc]["bir"]
                cost = p["outcomes"][oc]["cost"]
                cpath = _run_canonical_outcome(
                    umh, tmp, f"{slug_fn(city)}_{oc}", ndvi_b, ndvi_a, pop_p,
                    crs, gt, h, w, sr, rr, bir, cost)
                canon, cnod, cgt, _, cshape = _gdal_read(cpath)
                if abs(cgt[1] - gt[1]) > 1e-6 or abs(cgt[5] - gt[5]) > 1e-6:
                    print(f"  [{oc}] WARN canonical pixel size {cgt[1]} != proto {gt[1]} "
                          "— skipping"); rc = 1; continue
                if cnod is not None:
                    canon = np.where(canon == cnod, 0.0, canon)
                canon = np.nan_to_num(canon, nan=0.0)
                # Canonical pads the grid by the search radius on every side;
                # crop back to the proto window via the geotransform offset.
                dx = int(round((gt[0] - cgt[0]) / gt[1]))
                dy = int(round((gt[3] - cgt[3]) / gt[5]))
                canon = canon[dy:dy + h, dx:dx + w]
                if canon.shape != (h, w):
                    print(f"  [{oc}] WARN aligned canonical {canon.shape} != proto "
                          f"{(h, w)} (dx={dx}, dy={dy}) — skipping"); rc = 1; continue
                if meta_test:
                    proto = proto.astype("float64") * _META_TEST_PROTO_SCALE
                mae_all, mae_act, r, ptot, ctot = _metrics(canon, proto, active)
                rel = (mae_act / (abs(proto[active]).mean()) if active.any()
                       and abs(proto[active]).mean() > 0 else float("nan"))
                print(f"  [{oc}] effect_size={rr} BIR={bir}")
                print(f"       proto total cases = {ptot:,.2f} | canonical = {ctot:,.2f}")
                print(f"       MAE(active px) = {mae_act:.6g}  (rel {rel:.2%})  "
                      f"| MAE(all px) = {mae_all:.6g} | Pearson r = {r:.6f}")
                ok, fails = _parity_check(city, oc, mae_act, r, ptot, ctot)
                if ok:
                    print(f"       parity: OK  (mae<{_PARITY_MAX_MAE_ACT:.0e}, "
                          f"r>{_PARITY_MIN_R}, |Δtotal|<{_PARITY_MAX_REL_TOTAL:.1%})")
                else:
                    print(f"       parity: FAIL — {'; '.join(fails)}")
                    rc = 1
    print("\n" + "=" * 70)
    if meta_test:
        print(f"META-TEST mode: proto rasters scaled by {_META_TEST_PROTO_SCALE} "
              "before metrics. Expected: at least one outcome trips the parity "
              "assert (proves the threshold is sharp).")
        if rc == 0:
            print("META-TEST FAILED: every outcome passed despite synthetic "
                  f"+{(_META_TEST_PROTO_SCALE - 1) * 100:.1f}% bias — the parity "
                  "threshold is too loose to be a guard.")
            return 2
        print("META-TEST OK: synthetic bias correctly tripped the parity assert.")
        return 0
    print("Algorithmic-fidelity numbers above use matched inputs (uniform CDC "
          "rate, per-outcome effect_size). The shipped NE kernel (binary-disk "
          "buffer mean, app._umh_neighborhood_exposure) matches canonical "
          "UMH 3.19's ndvi_*_buffer_mean to per-pixel parity: MN MAE(active) "
          "≤ 1.1e-9 cases/px and r = 1.000000 on both outcomes; SA MAE(active) "
          "≤ 2.3e-6 cases/px, r ≥ 0.99876, totals diverge by ≤ 0.15% — a "
          "residual from canonical's radius padding + edge-crop alignment + "
          "pygeoprocessing FFT noise on the 1713×1984 grid, not a kernel-"
          "formula divergence (DESIGN_NOTES §6.3). A defensible parity assert "
          f"fires above MAE(active) ≥ {_PARITY_MAX_MAE_ACT:.0e}, "
          f"r ≤ {_PARITY_MIN_R}, or |Δtotal|/total ≥ "
          f"{_PARITY_MAX_REL_TOTAL:.1%}. Run `--meta-test` to confirm the "
          "assert fires under a small synthetic bias. Default-input MAE "
          "coincides (no per-tract MH-prevalence data exists for MN/SA to "
          "populate a per-admin prevalence vector).")
    return rc


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    meta_test = ("--meta-test" in sys.argv[2:])
    if mode == "export":
        return run_export()
    elif mode == "compare":
        return run_compare(FIX, CITIES_TO_RUN, _slug, meta_test=meta_test)
    else:
        print(__doc__)
        print("ERROR: specify a mode — 'export' (app .venv) or 'compare' "
              "(isolated env). Add --meta-test to compare to verify the "
              "parity assert is sharp.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
