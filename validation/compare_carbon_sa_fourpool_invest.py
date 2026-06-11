#!/usr/bin/env python3
"""compare_carbon_sa_fourpool_invest.py — SA four-pool carbon per-pixel parity.

Authored Relay 69. GROUNDS the carbon validation claim that Relay 68 withdrew:
the prior per-pixel r = 1.0 / MAE 121 figure came from an uncommitted /tmp script
(since deleted) and was unreproducible. This harness is COMMITTED and writes a
result artifact, so the number is re-verifiable — no recorded carbon number
without a committed reproducer.

DISTINCT from validation/compare_carbon_invest.py, which is the MN single-pool
annual-SEQUESTRATION proxy vs InVEST four-pool stock compared as an AOI-sum scalar
(its own header says the two measure different things — NOT a parity test). This
harness instead feeds canonical natcap.invest.carbon.execute() the IDENTICAL SA
compound LULC + the IDENTICAL four-pool C-density table the evaluator uses, then
compares the per-pixel total carbon STORAGE against the evaluator's per-pixel
four-pool stock — per-pixel MAE, Pearson r, AOI-sum % diff — in matched units.

Why a single baseline snapshot is sufficient: the evaluator's carbon metric is a
stock DELTA (scenario − baseline) summed over pixels; per-pixel storage is the
atomic building block and the delta is linear in it. If per-pixel storage matches
canonical InVEST, the delta matches by construction.

Faithfulness without importing app: app._load_city_runtime_state reads the compound
raster as `src.read(1).astype(int16)` with NO post-read mutation (app.py ~L1397),
so reading data/sa/flood/land_use_compound_sa.tif here yields the identical array.
The evaluator's per-pixel storage is reproduced literally from
_compute_carbon_four_pool_pure: pool_sum = (c_above+c_below+c_soil+c_dead) keyed by
the compound lucode (zero-filled for codes absent from the table; clipped to the
array bound), masked to code >= 0, times PIXEL_AREA_HA. The four-pool table read
here is the SAME file config points the evaluator at. (No app import is also why
this runs in the rasterio-free 3.19.0 env — raster IO is via osgeo.gdal.)

Units (verified against natcap.invest 3.19.0 source, not assumed):
  - InVEST c_storage_bas.tif is metric tons / HECTARE — per-pixel DENSITY
    (carbon/carbon.py declares units = metric_ton / hectare; carbon/reporter.py
    converts to total tons via raster_sum * pixel_area_m2 / 10000).
  - The evaluator reports per-pixel TOTAL Mg C = pool_density * PIXEL_AREA_HA (0.09).
  - Both are normalized to per-pixel-total Mg C before comparison:
        invest_total = c_storage_bas * pixel_area_ha(raster)
        eval_total   = pool_sum[code] * PIXEL_AREA_HA(0.09)
    The raw, un-normalized density-vs-total gap is also reported, to show it is a
    units artifact (≈ mean density × 0.91) and not a methodological disagreement.

Run in the isolated 3.19.0 env (NOT base conda's 3.16.2):
    conda run -n natcap_umh_validation python validation/compare_carbon_sa_fourpool_invest.py

NOT added to the 40/40 verify_baselines gate (env-isolated, manual — like the
UCM/UNA/UMH harnesses). Reads shipped data read-only; modifies no shipped code.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

REPO = Path(__file__).resolve().parent.parent
COMPOUND_TIF = REPO / "data/sa/flood/land_use_compound_sa.tif"
CARBON_CSV = REPO / "data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv"
EXPECTED_CRS_EPSG = 5070
PIXEL_AREA_HA = 0.09  # app.PIXEL_AREA_HA — the evaluator's constant (NLCD 30 m)


def _load_pool_arrays(csv_path):
    """Mirror app's loader: zero-filled arrays sized max(lucode)+1, set from rows."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    max_lc = int(df["lucode"].max())
    n = max_lc + 1
    c_above = np.zeros(n, dtype=np.float64)
    c_below = np.zeros(n, dtype=np.float64)
    c_soil = np.zeros(n, dtype=np.float64)
    c_dead = np.zeros(n, dtype=np.float64)
    for _, row in df.iterrows():
        lc = int(row["lucode"])
        c_above[lc] = float(row["c_above"])
        c_below[lc] = float(row["c_below"])
        c_soil[lc] = float(row["c_soil"])
        c_dead[lc] = float(row["c_dead"])
    return c_above, c_below, c_soil, c_dead


def _read_raster(path):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nod = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, nod, gt, proj


def _write_int16_raster(path, arr, gt, proj, nodata=-1):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Int16)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr.astype(np.int16))
    band.FlushCache()
    ds = None


def _build_pools_csv(path, codes, c_above, c_below, c_soil, c_dead):
    """Write an InVEST carbon-pools CSV from the EXACT evaluator pool arrays.

    Clipping to the array bound mirrors the evaluator's np.clip(code, 0, n-1), and
    codes absent from the table carry their zero-fill — so the table InVEST sees is
    identical to the one the evaluator indexes, and InVEST has every raster code.
    """
    n = len(c_above)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lucode", "c_above", "c_below", "c_soil", "c_dead"])
        for code in codes:
            i = min(int(code), n - 1)
            w.writerow([int(code),
                        f"{c_above[i]:.6f}", f"{c_below[i]:.6f}",
                        f"{c_soil[i]:.6f}", f"{c_dead[i]:.6f}"])


def _run_invest_storage(lulc_tif, pools_csv, workspace):
    """Run natcap.invest.carbon (3.19.0); return (density_raster t/ha, pixel_area_ha)."""
    import natcap.invest.carbon

    args = {
        "workspace_dir": workspace,
        "lulc_bas_path": str(lulc_tif),
        "calc_sequestration": False,   # baseline storage snapshot only
        "carbon_pools_path": str(pools_csv),
        "do_valuation": False,
    }
    t0 = time.time()
    natcap.invest.carbon.execute(args)
    print(f"   InVEST carbon.execute() done in {time.time() - t0:.1f}s")

    arr, nod, gt, _ = _read_raster(os.path.join(workspace, "c_storage_bas.tif"))
    density = arr.astype(np.float64)
    if nod is not None:
        density = np.where(density == nod, np.nan, density)
    pa_ha = abs(gt[1] * gt[5]) / 10_000.0
    return density, pa_ha


def _metrics(invest_total, eval_total, mask):
    a = invest_total[mask]
    b = eval_total[mask]
    mae = float(np.mean(np.abs(a - b)))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    aoi_inv, aoi_eval = float(a.sum()), float(b.sum())
    pct = (aoi_inv - aoi_eval) / aoi_eval * 100 if aoi_eval else float("inf")
    return mae, r, aoi_inv, aoi_eval, pct


def main():
    import natcap.invest
    inv_ver = natcap.invest.__version__
    print("=" * 66)
    print("SA four-pool Carbon per-pixel parity: evaluator vs canonical InVEST")
    print(f"natcap.invest {inv_ver}  (must be 3.19.x, NOT base 3.16.2)")
    print("=" * 66)

    print("\n1. Loading matched inputs (same files the evaluator reads)...")
    base, src_nod, gt, proj = _read_raster(COMPOUND_TIF)
    base = base.astype(np.int32)
    # CRS sanity (defends against an accidentally-3857 raster, like app's assertion).
    assert f"{EXPECTED_CRS_EPSG}" in proj or "5070" in proj, f"unexpected CRS:\n{proj[:200]}"
    c_above, c_below, c_soil, c_dead = _load_pool_arrays(CARBON_CSV)
    n = len(c_above)
    pool_sum = c_above + c_below + c_soil + c_dead

    valid = base >= 0  # evaluator's validity test (independent of declared nodata)
    safe = np.clip(base, 0, n - 1)
    eval_density = np.where(valid, pool_sum[safe], np.nan)      # t C/ha
    eval_total = eval_density * PIXEL_AREA_HA                   # Mg C / pixel
    print(f"   Compound LULC {base.shape}, valid pixels {int(valid.sum()):,}, "
          f"max table lucode {n - 1}, src nodata {src_nod}")
    if base[valid].max() > n - 1:
        print(f"   NOTE: raster has codes > max table lucode; clipped (evaluator parity).")

    with tempfile.TemporaryDirectory(prefix="carbon_sa_") as tmp:
        lulc_tif = os.path.join(tmp, "lulc_compound_sa.tif")
        # Re-emit with nodata=-1 so InVEST skips exactly the base<0 pixels.
        base16 = np.where(valid, base, -1).astype(np.int16)
        _write_int16_raster(lulc_tif, base16, gt, proj, nodata=-1)

        codes = sorted(int(c) for c in np.unique(base[valid]))
        pools_csv = os.path.join(tmp, "carbon_pools.csv")
        _build_pools_csv(pools_csv, codes, c_above, c_below, c_soil, c_dead)
        print(f"\n2. Matched inputs written: {len(codes)} lucodes in pools table.")

        print("\n3. Running canonical InVEST carbon (3.19.0) on baseline LULC...")
        ws = os.path.join(tmp, "invest_out")
        os.makedirs(ws)
        inv_density, pa_ha = _run_invest_storage(lulc_tif, pools_csv, ws)
        inv_total = inv_density * pa_ha
        print(f"   InVEST raster pixel_area_ha={pa_ha:.8f} (evaluator constant {PIXEL_AREA_HA})")

        mask = valid & np.isfinite(inv_density)
        print(f"   Compared pixels (valid in both): {int(mask.sum()):,}")
        mae, r, aoi_inv, aoi_eval, pct = _metrics(inv_total, eval_total, mask)
        raw_mae = float(np.mean(np.abs(inv_density[mask] - eval_total[mask])))

        print("\n4. MATCHED-UNITS parity (per-pixel total Mg C):")
        print(f"   per-pixel MAE      : {mae:.6e} Mg C")
        print(f"   Pearson r          : {r:.10f}")
        print(f"   AOI sum (InVEST)   : {aoi_inv:,.2f} Mg C")
        print(f"   AOI sum (evaluator): {aoi_eval:,.2f} Mg C")
        print(f"   AOI-sum % diff     : {pct:.6f} %")
        print(f"   [context] raw density-vs-total MAE (un-normalized): {raw_mae:.2f} Mg C")

        # ── Non-vacuous perturbation guard ──────────────────────────────────
        print("\n5. Non-vacuous guard: perturb one pool density, expect nonzero diff...")
        vals, counts = np.unique(base[valid], return_counts=True)
        bump_code = int(vals[int(np.argmax(counts))])
        c_above_pert = c_above.copy()
        c_above_pert[min(bump_code, n - 1)] += 100.0
        pools_pert = os.path.join(tmp, "carbon_pools_pert.csv")
        _build_pools_csv(pools_pert, codes, c_above_pert, c_below, c_soil, c_dead)
        ws2 = os.path.join(tmp, "invest_out_pert")
        os.makedirs(ws2)
        inv_density_p, pa_ha_p = _run_invest_storage(lulc_tif, pools_pert, ws2)
        inv_total_p = inv_density_p * pa_ha_p
        mask_p = valid & np.isfinite(inv_density_p)
        guard_mae = float(np.mean(np.abs(inv_total_p[mask_p] - eval_total[mask_p])))
        n_bump = int((base[mask_p] == bump_code).sum())
        guard_ok = guard_mae > 1e-3
        print(f"   Perturbed code {bump_code} (+100 t/ha c_above), {n_bump:,} pixels affected")
        print(f"   Guard MAE (perturbed InVEST vs evaluator): {guard_mae:.4f} Mg C "
              f"-> {'PASS' if guard_ok else 'FAIL'} (MAE≈0 is not vacuous)")

    CLEAN = (mae < 1e-2) and (r > 0.999999) and guard_ok
    print("\n6. Verdict:")
    print(f"   CLEAN (matched-units per-pixel parity on {inv_ver}): {CLEAN}")

    out_dir = REPO / "comparisons"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "carbon_sa_fourpool_parity.csv"
    notes = (
        "SA four-pool carbon STORAGE per-pixel parity: evaluator per-pixel four-pool "
        "stock vs canonical natcap.invest.carbon c_storage_bas, fed the IDENTICAL "
        "compound LULC (data/sa/flood/land_use_compound_sa.tif) + IDENTICAL four-pool "
        "C-density table (data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv). InVEST "
        "c_storage_bas is metric tons/HECTARE (density); evaluator reports per-pixel "
        "TOTAL Mg C (density*PIXEL_AREA_HA 0.09); both normalized to per-pixel-total "
        "Mg C. calc_sequestration=False (baseline snapshot; the metric's stock delta "
        "is linear in per-pixel storage). Non-vacuous guard: perturbing one pool "
        f"density moves the InVEST result. Isolated natcap.invest {inv_ver} env."
    )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "comparison", "city", "invest_version", "per_pixel_mae_mgC", "pearson_r",
            "aoi_sum_pct_diff", "aoi_sum_invest_mgC", "aoi_sum_evaluator_mgC",
            "raw_density_vs_total_mae_mgC", "guard_mae_mgC", "guard_ok",
            "n_pixels_compared", "pixel_area_ha", "clean", "notes",
        ])
        w.writerow([
            "SA four-pool carbon storage (baseline snapshot)", "San Antonio, TX", inv_ver,
            f"{mae:.6e}", f"{r:.10f}", f"{pct:.6f}", f"{aoi_inv:.2f}", f"{aoi_eval:.2f}",
            f"{raw_mae:.2f}", f"{guard_mae:.4f}", guard_ok,
            int(mask.sum()), f"{pa_ha:.8f}", CLEAN, notes,
        ])
    print(f"\n7. Result artifact -> {out_path}")
    print("Done.")
    return 0 if CLEAN else 2


if __name__ == "__main__":
    sys.exit(main())
