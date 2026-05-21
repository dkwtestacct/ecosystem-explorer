#!/usr/bin/env python3
"""compare_una_invest.py — Phase 1 UNA comparison: app proxy vs canonical InVEST.

Runs the MN baseline through both the app's per-pixel
`max(urban_nature × in_range)` Nature Access calculation and canonical
`natcap.invest.urban_nature_access.execute()`, then writes a single-row
comparison CSV with summary statistics.

Phase 1 scope: one scenario (baseline), one model (UNA), one city (MN).
Targets the documented "Proxy" parity gap (REFERENCE.md "Official InVEST
alignment — UNA") — the largest documented methodology divergence remaining
after the UCM HMI gap was closed.

The prototype reports a per-pixel quality-weighted *reachability* score
(0 / 0.5 / 1.0 — "is meaningful nature reachable from here?"). Canonical
InVEST UNA runs two-step floating catchment area (2SFCA): a per-capita
supply/demand ratio that accounts for population competition for nature.
These represent different things mathematically; this script quantifies how
different they are spatially and on aggregate.

Usage:
    python compare_una_invest.py

Prerequisites:
    - natcap.invest installed (conda env, 3.16.2 — same as UCM/Carbon)
    - All MN data files in place (same as verify_baselines.py)

Does NOT modify app.py, config.py, surrogate.py, or any shipped code.
Mirrors the Phase 1 UCM comparison pattern (compare_ucm_invest.py).
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
import rasterio

_DESIRED_CITY = "Minneapolis, MN"


# ── Streamlit stub (copied from compare_ucm_invest.py) ──────────────────────
# Must be installed before `import app`.

class _SessionStateStub:
    _store = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def pop(self, key, *args):
        return self._store.pop(key, *args) if args else self._store.pop(key, None)

    def setdefault(self, key, default=None):
        return self._store.setdefault(key, default)

    def __getattr__(self, name):
        if name == "_store":
            return object.__getattribute__(self, "_store")
        return self._store.get(name)

    def __getitem__(self, key):
        return self._store.get(key)

    def __setitem__(self, key, value):
        self._store[key] = value

    def __setattr__(self, name, value):
        if name == "_store":
            object.__setattr__(self, name, value)
        else:
            self._store[name] = value

    def __contains__(self, key):
        return key in self._store


class _StubSt:
    def __getattr__(self, name):
        if name in ("cache_data", "cache_resource"):
            return self._cache
        if name == "columns":
            return self._columns
        if name == "tabs":
            return self._tabs
        if name == "selectbox":
            def _sb(label, options, **kw):
                if not options:
                    return None
                if "City" in str(label) and _DESIRED_CITY:
                    for o in options:
                        if o == _DESIRED_CITY:
                            return o
                return options[0]
            return _sb
        if name == "radio":
            return lambda label, options, **kw: options[0] if options else None
        if name == "multiselect":
            return lambda label, options=(), default=None, **kw: list(default or [])
        if name == "slider":
            return lambda *a, **kw: kw.get("value", a[3] if len(a) >= 4 else 0)
        if name == "number_input":
            return lambda *a, **kw: kw.get("value", a[3] if len(a) >= 4 else 0)
        if name == "text_input":
            return lambda *a, **kw: kw.get("value", "")
        if name == "text_area":
            return lambda *a, **kw: kw.get("value", "")
        if name in ("toggle", "checkbox", "button"):
            return lambda *a, **kw: False
        if name == "session_state":
            return _SessionStateStub()
        return self

    def _cache(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda f: f

    def _columns(self, spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_StubSt() for _ in range(n))

    def _tabs(self, labels, *args, **kwargs):
        return tuple(_StubSt() for _ in labels)

    def __call__(self, *args, **kwargs):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def __getitem__(self, key):
        return self
    def __setitem__(self, key, value):
        pass
    def __setattr__(self, name, value):
        pass
    def __contains__(self, key):
        return False
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return True


# ── GeoTIFF / vector helpers ────────────────────────────────────────────────

def save_geotiff(arr, path, crs, transform, nodata=None, dtype=None):
    """Save a 2-D array as a single-band GeoTIFF."""
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def construct_aoi_from_raster(raster_path, output_dir):
    """Single-polygon AOI shapefile covering a raster's bounds.

    InVEST UNA requires `admin_boundaries_vector_path` and aggregates per
    feature. The published MN config ships a census-tract vector, but it is
    geometrically tied to InVEST's own sample LULC extent — not the app's
    `cooling_lulc` grid that this comparison runs InVEST on. The per-pixel
    output rasters (accessible_urban_nature, supply_percapita, ...) do not
    depend on the admin geometry — only the aggregate GeoPackage does — so a
    single bounding polygon is correct for a per-pixel comparison.
    """
    import geopandas as gpd
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    aoi_path = os.path.join(output_dir, "aoi_admin_boundaries.gpkg")
    gpd.GeoDataFrame({'id': [1]}, geometry=[geom], crs=crs).to_file(
        aoi_path, driver='GPKG')
    return aoi_path


def _norm01(arr, valid):
    """Min-max normalize the valid values of `arr` to [0, 1]."""
    v = arr[valid]
    lo, hi = float(v.min()), float(v.max())
    out = np.zeros_like(arr, dtype=np.float64)
    if hi > lo:
        out[valid] = (arr[valid].astype(np.float64) - lo) / (hi - lo)
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Phase 1 UNA Comparison: App proxy vs InVEST 2SFCA")
    print("=" * 64)

    # ── Step 1: Load app ──────────────────────────────────────────────
    print("\n1. Loading app (Streamlit stub)...")
    sys.modules["streamlit"] = _StubSt()
    t0 = time.time()
    import app
    print(f"   app.py import: {time.time() - t0:.1f}s")

    city_cfg = app.CITIES[_DESIRED_CITY]
    state = app._CURRENT_CITY_STATE
    lulc = np.asarray(state.cooling_lulc)
    pop = np.asarray(state.pop_count_raster, dtype=np.float32)
    transform = state.ref_transform
    crs = city_cfg['crs']
    print(f"   LULC: shape={lulc.shape}, dtype={lulc.dtype}")
    print(f"   Population: sum={pop.sum():,.0f}, max={pop.max():.1f}")

    # ── Step 2: Prototype Nature Access (baseline scenario) ───────────
    print("\n2. Computing the prototype's Nature Access outputs...")
    proto_score = np.asarray(app._compute_access_score_raster(lulc), dtype=np.float32)
    base = app.evaluate_scenario(
        pct_converted=0, green_infrastructure_pct=0, food_forest_pct=0,
        seed=42, placement_strategy='random',
    )
    proto_access_pct = float(base['nature_access_pct'])
    proto_quality_score = float(base['nature_quality_score'])
    print(f"   Prototype access score raster: range "
          f"[{proto_score.min():.2f}, {proto_score.max():.2f}], "
          f"distinct={sorted(np.unique(proto_score).tolist())}")
    print(f"   Nature Access %     : {proto_access_pct}")
    print(f"   Nature Quality Score: {proto_quality_score}")

    with tempfile.TemporaryDirectory(prefix="una_compare_") as tmpdir:
        # ── Step 3: Serialize the app's inputs for InVEST ─────────────
        print(f"\n3. Preparing InVEST UNA inputs in {tmpdir}...")
        lulc_path = os.path.join(tmpdir, "lulc_baseline.tif")
        save_geotiff(lulc.astype(np.int16), lulc_path, crs, transform,
                     nodata=-128, dtype=np.int16)

        pop_path = os.path.join(tmpdir, "population.tif")
        pop_clean = np.clip(np.nan_to_num(pop, nan=0.0), 0.0, None).astype(np.float32)
        save_geotiff(pop_clean, pop_path, crs, transform,
                     nodata=-1.0, dtype=np.float32)

        # LULC attribute table — keep only the columns InVEST UNA reads in
        # uniform-radius mode (lucode, urban_nature). Defensive: ensure every
        # lucode present in the raster has a row (absent in the app's table ==
        # urban_nature 0, which is how the app treats it).
        una_src = pd.read_csv(city_cfg['una_table_file'])[['lucode', 'urban_nature']]
        una_src['lucode'] = una_src['lucode'].astype(int)
        present = {int(c) for c in np.unique(lulc) if c != -128}
        missing = present - set(una_src['lucode'])
        if missing:
            print(f"   NOTE: adding urban_nature=0 rows for lucodes {sorted(missing)}")
            una_src = pd.concat([una_src, pd.DataFrame(
                {'lucode': sorted(missing), 'urban_nature': 0})], ignore_index=True)
        attr_path = os.path.join(tmpdir, "lulc_attribute_table_UNA.csv")
        una_src.to_csv(attr_path, index=False)

        aoi_path = construct_aoi_from_raster(lulc_path, tmpdir)
        print(f"   LULC + population rasters, attribute table, AOI polygon written.")

        # ── Step 4: Run InVEST UNA ────────────────────────────────────
        # Args from the published MN config
        # (invest_urban_nature_access_args_MN.json): uniform 1000 m radius,
        # exponential decay, urban_nature_demand = 250 m²/capita.
        workspace = os.path.join(tmpdir, "invest_output")
        os.makedirs(workspace)
        args = {
            'workspace_dir':                workspace,
            'results_suffix':               '',
            'lulc_raster_path':             lulc_path,
            'lulc_attribute_table':         attr_path,
            'population_raster_path':        pop_path,
            'admin_boundaries_vector_path': aoi_path,
            'urban_nature_demand':          250,
            'decay_function':               'exponential',
            'search_radius_mode':           'uniform radius',
            'search_radius':                1000,
            'aggregate_by_pop_group':       False,
            'n_workers':                    -1,
        }
        print("\n4. Running natcap.invest.urban_nature_access.execute()...")
        from natcap.invest import urban_nature_access
        t0 = time.time()
        urban_nature_access.execute(args)
        invest_runtime_s = time.time() - t0
        print(f"   InVEST UNA completed in {invest_runtime_s:.1f}s")

        # ── Step 5: Read InVEST outputs ───────────────────────────────
        out_dir = os.path.join(workspace, "output")
        acc_path = os.path.join(out_dir, "accessible_urban_nature.tif")
        sup_path = os.path.join(out_dir, "urban_nature_supply_percapita.tif")
        bal_path = os.path.join(out_dir, "urban_nature_balance_totalpop.tif")
        for p in (acc_path, sup_path, bal_path):
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
        with rasterio.open(bal_path) as src:
            invest_bal = src.read(1).astype(np.float64)
            bal_nodata = src.nodata
        print(f"   accessible_urban_nature.tif: shape={invest_acc.shape}")

        # ── Step 6: Spatial alignment ─────────────────────────────────
        # InVEST aligns/clips to the AOI; recover its pixel offset within the
        # app grid via the geotransforms (same pattern as compare_ucm_invest).
        print("\n5. Aligning rasters spatially...")
        col_off = round((invest_transform.c - transform.c) / transform.a)
        row_off = round((invest_transform.f - transform.f) / transform.e)
        r0, c0 = row_off, col_off
        r1, c1 = r0 + invest_acc.shape[0], c0 + invest_acc.shape[1]
        r0c, r1c = max(0, r0), min(lulc.shape[0], r1)
        c0c, c1c = max(0, c0), min(lulc.shape[1], c1)
        ir0, ir1 = r0c - r0, invest_acc.shape[0] - (r1 - r1c)
        ic0, ic1 = c0c - c0, invest_acc.shape[1] - (c1 - c1c)
        print(f"   InVEST origin offset in app grid: row={row_off}, col={col_off}")

        proto_cmp = proto_score[r0c:r1c, c0c:c1c].astype(np.float64)
        pop_cmp   = pop_clean[r0c:r1c, c0c:c1c].astype(np.float64)
        lulc_cmp  = lulc[r0c:r1c, c0c:c1c]
        acc_cmp   = invest_acc[ir0:ir1, ic0:ic1]
        sup_cmp   = invest_sup[ir0:ir1, ic0:ic1]
        bal_cmp   = invest_bal[ir0:ir1, ic0:ic1]

        def _valid(arr, nodata):
            v = np.isfinite(arr)
            if nodata is not None:
                v &= ~np.isclose(arr, nodata)
            return v

        valid = (lulc_cmp != -128) & _valid(acc_cmp, acc_nodata)
        n_valid = int(valid.sum())
        print(f"   Aligned region: {proto_cmp.shape}, valid pixels: {n_valid:,}")

        # ── Step 7: Comparison statistics ─────────────────────────────
        print("\n6. Computing comparison statistics...")
        from scipy.stats import spearmanr

        p = proto_cmp[valid]
        a = acc_cmp[valid]
        proto_mean = float(p.mean())
        invest_acc_mean = float(a.mean())

        # Raw-value MAE is omitted on purpose: the prototype score is 0–1 and
        # InVEST accessible_urban_nature is in m² — different units. Pearson /
        # Spearman r (do the two rank pixels alike?) and a min-max-normalized
        # MAE are the comparable divergence statistics.
        if p.std() > 0 and a.std() > 0:
            pearson_r = float(np.corrcoef(p, a)[0, 1])
            spearman_r = float(spearmanr(p, a).statistic)
        else:
            pearson_r = spearman_r = float('nan')

        norm_p = _norm01(proto_cmp, valid)
        norm_a = _norm01(acc_cmp, valid)
        normalized_mae = float(np.mean(np.abs(norm_p[valid] - norm_a[valid])))

        # Do the two metrics agree on *where* nature is reachable at all?
        proto_zero = p <= 0.0
        invest_zero = a <= 1e-9
        zero_nonzero_agreement = float(100.0 * np.mean(proto_zero == invest_zero))

        # Population-weighted divergence (Nature Quality Score is pop-weighted).
        pw = pop_cmp[valid]
        pw_tot = pw.sum()
        if pw_tot > 0:
            proto_popw = float((norm_p[valid] * pw).sum() / pw_tot)
            invest_popw = float((norm_a[valid] * pw).sum() / pw_tot)
        else:
            proto_popw = invest_popw = float('nan')

        # InVEST aggregate metrics.
        sup_valid = valid & _valid(sup_cmp, sup_nodata)
        bal_valid = valid & _valid(bal_cmp, bal_nodata)
        invest_sup_mean = float(sup_cmp[sup_valid].mean()) if sup_valid.any() else float('nan')
        invest_bal_mean = float(bal_cmp[bal_valid].mean()) if bal_valid.any() else float('nan')
        # InVEST analogue of "% population with adequate access": share of
        # population in pixels whose per-capita supply meets the 250 m² demand.
        demand = 250.0
        pop_sup = pop_cmp[sup_valid]
        if pop_sup.sum() > 0:
            invest_pct_pop_adequate = float(
                100.0 * pop_sup[sup_cmp[sup_valid] >= demand].sum() / pop_sup.sum())
        else:
            invest_pct_pop_adequate = float('nan')

        print(f"   Prototype access score mean : {proto_mean:.4f}")
        print(f"   InVEST accessible nature mean: {invest_acc_mean:,.2f} m²")
        print(f"   InVEST supply/capita mean   : {invest_sup_mean:,.2f} m²/person")
        print(f"   Pearson r                   : {pearson_r:.4f}")
        print(f"   Spearman r                  : {spearman_r:.4f}")
        print(f"   Normalized MAE              : {normalized_mae:.4f}")
        print(f"   Zero/non-zero agreement     : {zero_nonzero_agreement:.1f}%")
        print(f"   Proto vs InVEST (pop-wtd, normalized): "
              f"{proto_popw:.4f} vs {invest_popw:.4f}")

        # ── Step 8: Diff raster (normalized) ──────────────────────────
        out_path = Path("comparisons")
        out_path.mkdir(exist_ok=True)
        diff = np.full(proto_cmp.shape, np.nan, dtype=np.float32)
        diff[valid] = (norm_p[valid] - norm_a[valid]).astype(np.float32)
        diff_path = out_path / "una_diff_baseline_mn.tif"
        save_geotiff(diff, str(diff_path), crs, invest_transform, dtype=np.float32)
        print(f"\n   Diff raster (normalized proto − InVEST) saved: {diff_path}")

    # ── Step 9: Write comparison CSV ──────────────────────────────────
    notes = (
        "Compares the app's per-pixel Nature Access proxy against canonical "
        "InVEST UNA (natcap.invest 3.16.2). THE TWO METRICS MEASURE DIFFERENT "
        "THINGS: the app computes a per-pixel quality-weighted REACHABILITY "
        "score (0/0.5/1.0 = 'is meaningful nature reachable from this pixel') "
        "via max(urban_nature x in_range); InVEST UNA runs two-step floating "
        "catchment area (2SFCA) — accessible_urban_nature.tif is the "
        "decay-weighted m2 of urban nature reachable per pixel, and "
        "urban_nature_supply_percapita.tif divides that by distance-weighted "
        "population (supply ADEQUACY for the people competing for it). The "
        "unweighted invest_supply_percapita mean is inflated by low-population "
        "pixels (tiny denominator) and is not population-grounded — "
        "invest_pct_pop_supply_ge_demand (share of residents whose per-capita "
        "supply meets the 250 m2 demand) is the meaningful adequacy figure. "
        "The primary pixel comparison is the app score vs "
        "accessible_urban_nature (both reachability, no per-pixel population "
        "term); raw-value MAE is omitted because units differ (0-1 vs m2) — "
        "Pearson/Spearman r and a min-max-normalized MAE are the comparable "
        "stats. The zero/non-zero agreement is near-trivial in this dense "
        "downtown AOI — essentially every pixel has SOME nature reachable "
        "under both methods, so the metrics diverge on magnitude and "
        "per-capita adequacy, not on binary reachability. Method differences "
        "stacked into this gap: (a) proxy vs 2SFCA; (b) the app uses per-class "
        "search radii capped at 1000 m, InVEST here uses a uniform 1000 m "
        "radius (NatCap's published MN config); (c) the app uses a hard "
        "distance cutoff (~dichotomy decay), InVEST here uses exponential "
        "decay. Args from the published invest_urban_nature_access_args_MN.json "
        "(search_radius_mode='uniform radius', search_radius=1000, "
        "urban_nature_demand=250, decay_function='exponential', "
        "aggregate_by_pop_group=false). Admin boundary: a single polygon built "
        "from the LULC bounds — the published census-tract vector is tied to "
        "InVEST's own sample LULC extent, and per-pixel outputs do not depend "
        "on admin geometry. Caveat: MN downtown is dense and fairly uniform, "
        "which under-exercises 2SFCA's supply/demand contrast vs a "
        "mixed-density city — the SA gap could read larger. Phase 1: one "
        "scenario (baseline), one city."
    )
    csv_path = Path("comparisons") / "una_baseline_mn.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'model', 'scenario', 'city',
            'proto_access_pct', 'proto_quality_score', 'proto_access_score_mean',
            'invest_accessible_nature_mean_m2', 'invest_supply_percapita_mean_m2',
            'invest_balance_totalpop_mean', 'invest_pct_pop_supply_ge_demand',
            'pearson_r', 'spearman_r', 'normalized_mae',
            'zero_nonzero_agreement_pct',
            'proto_popwtd_norm', 'invest_popwtd_norm',
            'n_valid_pixels', 'invest_runtime_s', 'args_source', 'notes',
        ])
        w.writerow([
            'UNA', 'baseline', 'Minneapolis, MN',
            f'{proto_access_pct:.1f}', f'{proto_quality_score:.3f}',
            f'{proto_mean:.4f}',
            f'{invest_acc_mean:.2f}', f'{invest_sup_mean:.2f}',
            f'{invest_bal_mean:.2f}', f'{invest_pct_pop_adequate:.1f}',
            f'{pearson_r:.4f}', f'{spearman_r:.4f}', f'{normalized_mae:.4f}',
            f'{zero_nonzero_agreement:.1f}',
            f'{proto_popw:.4f}', f'{invest_popw:.4f}',
            n_valid, f'{invest_runtime_s:.1f}',
            'published invest_urban_nature_access_args_MN.json',
            notes,
        ])
    print(f"\n7. Results written to {csv_path}")
    print("\n" + "=" * 64)
    print("Summary — UNA proxy vs InVEST 2SFCA (MN baseline)")
    print("=" * 64)
    print(f"  Prototype : access {proto_access_pct}% | quality {proto_quality_score} "
          f"| score mean {proto_mean:.4f}")
    print(f"  InVEST    : accessible {invest_acc_mean:,.0f} m² | "
          f"supply/capita {invest_sup_mean:,.0f} m²/person | "
          f"{invest_pct_pop_adequate:.1f}% pop ≥ demand")
    print(f"  Agreement : Pearson r={pearson_r:.4f}  Spearman r={spearman_r:.4f}  "
          f"normalized MAE={normalized_mae:.4f}")
    print(f"  Zero/non-zero reachability agreement: {zero_nonzero_agreement:.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
