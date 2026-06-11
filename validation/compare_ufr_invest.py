#!/usr/bin/env python3
"""compare_ufr_invest.py — SA UFR Runoff Retention per-pixel parity (Relay 71).

Grounds the prototype's Runoff Retention reading (`rnf_rt_idx = mean(1 − Q/P)`,
`app.cn_array_to_retention_index`) against canonical natcap.invest Urban Flood Risk
Mitigation 3.19.0, the way carbon/UCM/UNA/UMH are grounded: a COMMITTED harness +
committed result artifact (`comparisons/ufr_sa_retention_parity.csv`).

What it tests — the formula/Ia-convention crux (Relay 70). UFRM 3.19.0 computes
  S_max = 25400/CN − 254 (mm);  Q = (P − λ·S_max)² / (P + (1−λ)·S_max), λ=0.2,
          0 where P ≤ λ·S_max;  runoff_retention_index = 1 − Q/P.
The evaluator computes (inches)
  S = 1000/CN − 10;  Ia = 0.2·S;  Q = (P − Ia)²/(P − Ia + S), 0 where P ≤ Ia;
  retention = 1 − Q/P.
These are algebraically identical (Ia = λS; the mm-vs-inch scale cancels in the
dimensionless Q/P). This harness proves it empirically on SA's real per-pixel CN.

Drift-free by construction: rather than re-derive CN from lulc×soil (SA's CN table
is keyed by NLCD×tree-canopy via `reduce_compound_to_nlcd_tree`), the app phase dumps
the evaluator's ACTUAL baseline per-pixel CN raster, and the InVEST phase feeds UFRM a
synthetic LULC + a CN table that map each pixel back to that exact CN value — so UFRM's
internally-built `cn_raster` equals the evaluator's CN to the bit (verified), and the
only thing under test is UFRM's S_max→Q→retention chain vs the evaluator's formula.

Two phases (3.19.0 lacks rasterio; the app needs it — no deps forced into either env):
  • --phase app    : run under the project .venv. Imports app, reconstructs the
                     evaluator's baseline per-pixel CN, writes cn_eval.tif + meta.json.
  • --phase invest : run under natcap_umh_validation (3.19.0). Runs UFRM, compares.
  • no flag        : orchestrator — runs the app phase via .venv/bin/python, then the
                     invest phase in-process. Invoke exactly like the carbon harness:
        conda run -n natcap_umh_validation python validation/compare_ufr_invest.py

NOT in the 40/40 verify_baselines gate (env-isolated, manual). Reads shipped data
read-only; modifies no shipped code. Only Runoff Retention maps to a per-pixel UFRM
output; Flood Index (100−mean_CN) and Runoff Volume (lumped mean-CN) are lumped proxies
and stay aligned-method regardless.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CITY = "San Antonio, TX"


# ── Phase APP (run under .venv: rasterio + app available) ────────────────────
def phase_app(workdir: Path):
    import rasterio  # noqa: F401  (proves we're in the right env)
    sys.path.insert(0, str(REPO))
    import app

    state = app._load_city_runtime_state(CITY)
    cfg = app.CITIES[CITY]
    P_inches = float(cfg["design_storm_inches"])
    crs = cfg["crs"]

    # Reproduce evaluate_scenario's BASELINE per-pixel CN exactly (app.py ~2609).
    soil_clamped = np.clip(state.soil_resized, 1, 4)
    cn_lookup = app.reduce_compound_to_nlcd_tree(
        state.cooling_lulc_compound, state.compound_to_nlcd_tree)
    lulc_safe = np.clip(cn_lookup, 0, len(state.lucode_idx_arr) - 1)
    cn = state.cn_table[state.lucode_idx_arr[lulc_safe], soil_clamped].astype(np.float32)

    NOD = np.float32(-9999.0)
    cn_out = np.where(cn > 0, cn, NOD).astype(np.float32)  # evaluator's cn>0 mask

    import rasterio
    out = workdir / "cn_eval.tif"
    with rasterio.open(
        out, "w", driver="GTiff", height=cn_out.shape[0], width=cn_out.shape[1],
        count=1, dtype="float32", crs=crs, transform=state.ref_transform, nodata=float(NOD),
    ) as dst:
        dst.write(cn_out, 1)

    (workdir / "meta.json").write_text(json.dumps({
        "design_storm_inches": P_inches,
        "n_valid": int((cn > 0).sum()),
    }))
    print(f"[app] wrote {out} (valid CN>0 pixels: {int((cn > 0).sum()):,}), "
          f"P={P_inches} in")


# ── Phase INVEST (run under natcap_umh_validation: natcap.invest + gdal) ──────
def _read(path):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray()
    nod = b.GetNoDataValue()
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return arr, nod, gt, proj


def _write(path, arr, gt, proj, gdtype, nodata):
    from osgeo import gdal
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, gdtype)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    band.FlushCache()
    ds = None


def _make_aoi(path, gt, proj, shape):
    """Single rectangle polygon over the raster extent (UFRM requires an AOI;
    only serv aggregation uses it — per-pixel retention does not)."""
    from osgeo import ogr, osr
    h, w = shape
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + w * gt[1], gt[3] + h * gt[5]
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(path):
        drv.DeleteDataSource(str(path))
    ds = drv.CreateDataSource(str(path))
    lyr = ds.CreateLayer("aoi", srs, ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]:
        ring.AddPoint(x, y)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feat = ogr.Feature(lyr.GetLayerDefn())
    feat.SetGeometry(poly)
    lyr.CreateFeature(feat)
    feat = None
    ds = None


def _run_ufrm(workdir, lulc_tif, soil_tif, cn_csv, aoi_path, P_mm, tag):
    import natcap.invest.urban_flood_risk_mitigation as ufr
    ws = workdir / f"ufrm_{tag}"
    ws.mkdir(exist_ok=True)
    ufr.execute({
        "workspace_dir": str(ws),
        "aoi_watersheds_path": str(aoi_path),
        "rainfall_depth": P_mm,
        "lulc_path": str(lulc_tif),
        "soils_hydrological_group_raster_path": str(soil_tif),
        "curve_number_table_path": str(cn_csv),
    })
    ret, rnod, _, _ = _read(ws / "Runoff_retention_index.tif")
    cn_path = next(p for p in ws.rglob("*.tif") if "cn" in p.name.lower())
    cnr, cnnod, _, _ = _read(cn_path)
    return ret.astype(np.float64), rnod, cnr.astype(np.float64), cnnod


def _evaluator_retention(cn, P_inches):
    """app.cn_array_to_retention_index per-pixel, vectorized (inches)."""
    S = 1000.0 / cn - 10.0
    Ia = 0.2 * S
    Q = np.where(P_inches <= Ia, 0.0, (P_inches - Ia) ** 2 / (P_inches - Ia + S))
    return 1.0 - Q / P_inches


def _build_synthetic_inputs(workdir, cn_eval, nod, gt, proj):
    """LULC + CN table that reclassify back to the exact evaluator CN."""
    valid = (cn_eval != nod) & (cn_eval > 0)
    uniq = np.unique(cn_eval[valid])
    cn_to_code = {float(v): i + 1 for i, v in enumerate(uniq)}  # 1..K
    lulc = np.full(cn_eval.shape, -1, dtype=np.int32)
    for v, code in cn_to_code.items():
        lulc[valid & (cn_eval == np.float32(v))] = code
    soil = np.where(valid, 1, 0).astype(np.int32)  # all HSG A on valid pixels

    lulc_tif = workdir / "lulc_synth.tif"
    soil_tif = workdir / "soil_synth.tif"
    _write(lulc_tif, lulc, gt, proj, __import__("osgeo").gdal.GDT_Int32, -1)
    _write(soil_tif, soil, gt, proj, __import__("osgeo").gdal.GDT_Int32, 0)
    return lulc_tif, soil_tif, cn_to_code, valid


def _write_cn_table(path, cn_to_code, perturb_code=None, delta=0.0):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lucode", "CN_A", "CN_B", "CN_C", "CN_D"])
        for v, code in cn_to_code.items():
            cn = v + (delta if code == perturb_code else 0.0)
            cn = min(cn, 100.0)
            w.writerow([code, f"{cn:.6f}", f"{cn:.6f}", f"{cn:.6f}", f"{cn:.6f}"])


def phase_invest(workdir: Path):
    from osgeo import gdal
    gdal.UseExceptions()
    meta = json.loads((workdir / "meta.json").read_text())
    P_inches = meta["design_storm_inches"]
    P_mm = P_inches * 25.4

    cn_eval, nod, gt, proj = _read(workdir / "cn_eval.tif")
    cn_eval = cn_eval.astype(np.float64)
    lulc_tif, soil_tif, cn_to_code, valid = _build_synthetic_inputs(
        workdir, cn_eval, nod, gt, proj)
    aoi = workdir / "aoi.gpkg"
    _make_aoi(aoi, gt, proj, cn_eval.shape)
    print(f"[invest] {len(cn_to_code)} distinct CN values; P={P_inches} in "
          f"({P_mm:.1f} mm); valid pixels {int(valid.sum()):,}")

    cn_csv = workdir / "cn_table.csv"
    _write_cn_table(cn_csv, cn_to_code)
    print("[invest] running canonical UFRM 3.19.0...")
    ufrm_ret, rnod, ufrm_cn, cnnod = _run_ufrm(
        workdir, lulc_tif, soil_tif, cn_csv, aoi, P_mm, "main")

    # CN identity: UFRM's reclassified CN must equal the evaluator's CN.
    cn_ok_mask = valid & (ufrm_cn != cnnod if cnnod is not None else valid)
    cn_identity_max = float(np.max(np.abs(ufrm_cn[cn_ok_mask] - cn_eval[cn_ok_mask])))

    eval_ret = _evaluator_retention(cn_eval, P_inches)
    mask = valid & (ufrm_ret != rnod if rnod is not None else valid) & np.isfinite(ufrm_ret)

    a, b = ufrm_ret[mask], eval_ret[mask]
    mae = float(np.mean(np.abs(a - b)))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    aoi_u, aoi_e = float(a.sum()), float(b.sum())
    pct = (aoi_u - aoi_e) / aoi_e * 100 if aoi_e else float("inf")

    print("\n=== Runoff Retention parity (per-pixel 1−Q/P, unitless) ===")
    print(f"  CN identity max|UFRM−eval|: {cn_identity_max:.6e}  (drift check)")
    print(f"  per-pixel MAE  : {mae:.6e}")
    print(f"  Pearson r      : {r:.10f}")
    print(f"  AOI-sum % diff : {pct:.6f} %")
    print(f"  pixels compared: {int(mask.sum()):,}")

    # Non-vacuous guard: perturb the CN value covering the most pixels.
    code_px = {code: int(np.sum(valid & (cn_eval == np.float32(v))))
               for v, code in cn_to_code.items()}
    bump_code = max(code_px, key=code_px.get)
    cn_csv_p = workdir / "cn_table_pert.csv"
    _write_cn_table(cn_csv_p, cn_to_code, perturb_code=bump_code, delta=-15.0)
    ufrm_ret_p, rnod_p, _, _ = _run_ufrm(
        workdir, lulc_tif, soil_tif, cn_csv_p, aoi, P_mm, "pert")
    mask_p = valid & (ufrm_ret_p != rnod_p if rnod_p is not None else valid) & np.isfinite(ufrm_ret_p)
    guard_mae = float(np.mean(np.abs(ufrm_ret_p[mask_p] - eval_ret[mask_p])))
    guard_ok = guard_mae > 1e-4
    print(f"  guard MAE (CN −15 on busiest class): {guard_mae:.4f} -> "
          f"{'PASS' if guard_ok else 'FAIL'}")

    inv_ver = __import__("natcap.invest", fromlist=["__version__"]).__version__
    CLEAN = (cn_identity_max < 1e-3) and (mae < 1e-3) and (r > 0.999999) and guard_ok
    print(f"\nnatcap.invest {inv_ver}  CLEAN={CLEAN}")

    out_dir = REPO / "comparisons"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ufr_sa_retention_parity.csv"
    notes = (
        "SA UFR Runoff Retention per-pixel parity: evaluator (1−Q/P, "
        "cn_array_to_retention_index) vs canonical natcap.invest UFRM "
        "runoff_retention_index. Drift-free: UFRM fed a synthetic LULC + CN table "
        "that reproduce the evaluator's ACTUAL baseline per-pixel CN exactly (CN "
        "identity verified). Both unitless ratios — no units artifact. UFRM λ=0.2, "
        "S_max=25400/CN−254 mm; evaluator Ia=0.2S, S=1000/CN−10 in — algebraically "
        f"identical, Q/P scale-invariant. P={P_inches} in. natcap.invest {inv_ver}."
    )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "comparison", "city", "invest_version", "per_pixel_mae", "pearson_r",
            "aoi_sum_pct_diff", "cn_identity_max_abs", "guard_mae", "guard_ok",
            "n_pixels_compared", "design_storm_inches", "clean", "notes",
        ])
        w.writerow([
            "SA UFR runoff retention index (baseline)", CITY, inv_ver,
            f"{mae:.6e}", f"{r:.10f}", f"{pct:.6f}", f"{cn_identity_max:.6e}",
            f"{guard_mae:.4f}", guard_ok, int(mask.sum()), P_inches, CLEAN, notes,
        ])
    print(f"[invest] result artifact -> {out_path}")
    return 0 if CLEAN else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["app", "invest"])
    ap.add_argument("--workdir")
    args = ap.parse_args()

    if args.phase == "app":
        phase_app(Path(args.workdir))
        return 0
    if args.phase == "invest":
        return phase_invest(Path(args.workdir))

    # Orchestrator (run under 3.19.0): app dump via .venv, then invest in-process.
    with tempfile.TemporaryDirectory(prefix="ufr_parity_") as tmp:
        tmp = Path(tmp)
        venv_py = REPO / ".venv" / "bin" / "python"
        print(f"[orchestrator] app dump via {venv_py} ...")
        subprocess.run(
            [str(venv_py), str(Path(__file__).resolve()),
             "--phase", "app", "--workdir", str(tmp)],
            check=True, cwd=str(REPO))
        return phase_invest(tmp)


if __name__ == "__main__":
    sys.exit(main())
