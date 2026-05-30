#!/usr/bin/env python3
"""compare_ucm_invest.py — Phase 1 UCM comparison: app vs canonical InVEST.

Runs the MN baseline through both the app's mean-smoothed-CC calculation
and natcap.invest.urban_cooling_model.execute(), then writes a single-row
comparison CSV with summary statistics.

Phase 1 scope: one scenario (baseline), one model (UCM), one city (MN).
Targets the documented HMI park-proximity divergence (REFERENCE.md
"Official InVEST alignment — UCM"): the app skips the canonical
max(CC_local, CC_park) step with exponential decay near green areas >= 2 ha,
substituting a Gaussian smoothing at sigma=15 px (450 m).

Usage:
    python validation/compare_ucm_invest.py

Prerequisites:
    - natcap.invest installed (conda install -c conda-forge natcap.invest)
    - All MN data files in place (same as verify_baselines.py)

Does NOT modify app.py, config.py, surrogate.py, or any shipped code.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

# ── Streamlit stub (copied from verify_baselines.py) ────────────────────────
# Must be installed before `import app`.

_DESIRED_CITY = "Minneapolis, MN"


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


def _rebind_city(app_mod, city_name):
    """Load a city's runtime state and rebind module-level aliases."""
    state = app_mod._load_city_runtime_state(city_name)
    city_cfg = app_mod.CITIES[city_name]

    app_mod.lulc                = state.lulc
    app_mod.soil_resized        = state.soil_resized
    app_mod.cooling_lulc        = state.cooling_lulc
    app_mod.developed_pixels    = state.developed_pixels
    app_mod.cn_table            = state.cn_table
    app_mod.lucode_idx_arr      = state.lucode_idx_arr
    app_mod.hm_arr              = state.hm_arr
    app_mod.max_raster_lucode   = state.max_raster_lucode
    app_mod.max_hm_lucode       = state.max_hm_lucode
    app_mod.equity_weights      = state.equity_weights
    app_mod.shade_arr           = state.shade_arr
    app_mod.kc_arr              = state.kc_arr
    app_mod.albedo_arr          = state.albedo_arr
    app_mod.pop_count_raster    = state.pop_count_raster
    app_mod.POPULATION_DATA_AVAILABLE = state.population_data_available
    app_mod.ET_RESIZED          = state.et_resized
    app_mod.MAX_ET_REF          = state.max_et_ref
    app_mod.ET_DATA_AVAILABLE   = state.et_data_available
    app_mod.ENERGY_BY_TYPE           = state.energy_by_type
    app_mod.ENERGY_TABLE_AVAILABLE   = state.energy_table_available
    app_mod.UNA_ACTIVE               = state.una_active
    app_mod.PRECOMPUTED_NATURE_DISTANCES = state.precomputed_nature_distances
    app_mod._REF_SHAPE          = state.ref_shape
    app_mod._REF_TRANSFORM      = state.ref_transform
    app_mod.BUILDINGS_RASTER         = state.buildings_raster
    app_mod.BUILDINGS_TYPE_RASTER    = state.buildings_type_raster
    app_mod.BUILDINGS_DATA_AVAILABLE = state.buildings_data_available
    app_mod.BUILDINGS_HAVE_TYPES     = state.buildings_have_types
    app_mod.BUILDINGS_TYPE_COVERAGE  = state.buildings_type_coverage
    app_mod.TOTAL_POTENTIAL_DAMAGE_USD = state.total_potential_damage_usd
    app_mod.ROADS_RASTER        = state.roads_raster
    app_mod.OSM_ROADS_AVAILABLE = state.osm_roads_available
    app_mod.CONSUMPTION_RATE_PER_PIXEL = state.consumption_rate_per_pixel
    app_mod.CONVERTIBLE_PIXELS  = state.convertible_pixels
    app_mod.TRACTS              = state.tracts
    app_mod.TRACT_ID_RASTER     = state.tract_id_raster
    app_mod.TRACTS_DATA_AVAILABLE = state.tracts_data_available
    app_mod._BASELINE_ACCESS_SCORE_RASTER = state.baseline_access_score_raster
    app_mod._BASELINE_HM_RASTER = state.baseline_hm_raster
    app_mod._BASELINE_NE_RASTER = state.baseline_ne_raster
    app_mod._CURRENT_CITY_STATE = state

    app_mod.PIXEL_AREA_ACRES     = city_cfg['pixel_area_acres']
    app_mod.FOOD_FOREST_LBS_ACRE = city_cfg['food_forest_lbs_acre']
    app_mod.UHI_MAX_C            = city_cfg['uhi_max_c']
    app_mod.HM_TO_FAHRENHEIT     = city_cfg['uhi_max_c'] * 1.8

    app_mod.BASELINE_NATURE_ACCESS_PCT, app_mod.BASELINE_NATURE_QUALITY_SCORE, _ = (
        app_mod.calculate_nature_access(state.cooling_lulc, state.pop_count_raster)
    )
    app_mod.BASELINE_RUNOFF_ACRE_FEET = app_mod.cn_to_runoff_acre_feet(
        state.baseline_cn, len(state.developed_pixels) * city_cfg['pixel_area_acres']
    )
    app_mod.BASELINE_NDVI = app_mod.compute_mean_ndvi(state.cooling_lulc)

    return state


# ── GeoTIFF helpers ──────────────────────────────────────────────────────────

def save_geotiff(arr, path, crs, transform, nodata=None, dtype=None):
    """Save a 2-D array as a single-band GeoTIFF."""
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1 UCM Comparison: App smoothed CC vs InVEST HMI")
    print("=" * 60)

    # ── Step 1: Load app and compute baseline CC raster ───────────────
    print("\n1. Loading app (Streamlit stub)...")
    sys.modules["streamlit"] = _StubSt()
    t0 = time.time()
    import app
    print(f"   app.py import: {time.time() - t0:.1f}s")

    print("\n2. Rebinding to Minneapolis, MN...")
    state = _rebind_city(app, "Minneapolis, MN")
    city_cfg = app.CITIES["Minneapolis, MN"]

    # The app's baseline smoothed CC raster is already precomputed
    app_cc = state.baseline_hm_raster  # = _BASELINE_HM_RASTER
    valid_mask_app = np.isfinite(app_cc)
    app_mean_cc = float(np.nanmean(app_cc))
    print(f"   App baseline mean smoothed CC: {app_mean_cc:.4f}")
    print(f"   App CC raster shape: {app_cc.shape}, dtype: {app_cc.dtype}")
    print(f"   Valid pixels: {valid_mask_app.sum():,} / {app_cc.size:,}")

    # ── Step 2: Prepare InVEST UCM inputs ─────────────────────────────
    # The args JSON tells us the canonical configuration.
    # All paths are relative to the UrbanCooling sample data dir.
    invest_data_dir = "data/invest/cooling/UrbanCooling_sample_data/UrbanCooling"

    with tempfile.TemporaryDirectory(prefix="ucm_compare_") as tmpdir:
        print(f"\n3. Preparing InVEST UCM inputs in {tmpdir}...")

        # Save baseline LULC as GeoTIFF (the app loads it as a numpy array;
        # InVEST needs a raster file with CRS and transform).
        lulc_path = os.path.join(tmpdir, "lulc_baseline.tif")
        lulc_arr = state.cooling_lulc.astype(np.int16)
        save_geotiff(lulc_arr, lulc_path, city_cfg['crs'],
                     state.ref_transform, nodata=-128, dtype=np.int16)
        print(f"   LULC raster saved: {lulc_arr.shape}")

        # Save the app's ET raster — InVEST needs a GeoTIFF for ref_eto.
        # The app's ET_RESIZED is already resampled to the LULC grid.
        et_path = os.path.join(tmpdir, "et_ref.tif")
        et_arr = state.et_resized.astype(np.float32)
        save_geotiff(et_arr, et_path, city_cfg['crs'],
                     state.ref_transform, nodata=-1.0, dtype=np.float32)
        print(f"   ET raster saved: {et_arr.shape}")

        # Biophysical table: use the app's MN table directly
        bio_table_path = os.path.join(
            city_cfg['data_dir_cooling'],
            city_cfg['cooling_table_file'],
        )
        print(f"   Biophysical table: {bio_table_path}")

        # AOI vector: use the InVEST sample AOI polygon
        aoi_path = os.path.join(invest_data_dir, "AOI_polygon.shp")
        if not os.path.exists(aoi_path):
            print(f"   WARNING: AOI polygon not found at {aoi_path}")
            print("   Will construct from LULC bounds...")
            aoi_path = _construct_aoi_from_raster(lulc_path, tmpdir)
        else:
            print(f"   AOI vector: {aoi_path}")

        # ── Step 3: Run InVEST UCM ────────────────────────────────────
        workspace = os.path.join(tmpdir, "invest_output")
        os.makedirs(workspace)

        args = {
            'workspace_dir': workspace,
            'lulc_raster_path': lulc_path,
            'ref_eto_raster_path': et_path,
            'aoi_vector_path': aoi_path,
            'biophysical_table_path': bio_table_path,
            'green_area_cooling_distance': 450,
            't_air_average_radius': 600,
            't_ref': 23.2,         # from args JSON
            'uhi_max': 2.05,       # from args JSON
            'cc_method': 'factors',
            'do_energy_valuation': False,
            'do_productivity_valuation': False,
            'n_workers': -1,
        }

        print("\n4. Running natcap.invest.urban_cooling_model.execute()...")
        from natcap.invest import urban_cooling_model
        t0 = time.time()
        urban_cooling_model.execute(args)
        elapsed = time.time() - t0
        print(f"   InVEST UCM completed in {elapsed:.1f}s")

        # ── Step 4: Read InVEST outputs and spatially align ────────────
        hm_path = os.path.join(workspace, "hm.tif")
        cc_path = os.path.join(workspace, "intermediate", "cc.tif")

        print("\n5. Reading InVEST output rasters...")

        with rasterio.open(hm_path) as src:
            invest_hm = src.read(1)
            invest_hm_nodata = src.nodata
            invest_transform = src.transform
            print(f"   HMI raster: shape={invest_hm.shape}, "
                  f"nodata={invest_hm_nodata}, dtype={invest_hm.dtype}")

        with rasterio.open(cc_path) as src:
            invest_cc = src.read(1)
            invest_cc_nodata = src.nodata
            print(f"   CC raster:  shape={invest_cc.shape}, "
                  f"nodata={invest_cc_nodata}, dtype={invest_cc.dtype}")

        # ── Step 5: Spatial alignment ─────────────────────────────────
        # InVEST clips its output to the AOI polygon, so the output raster
        # has a different origin and extent than the app's full LULC raster.
        # We use the geotransforms to find the pixel offset of the InVEST
        # raster within the app's coordinate space.
        print("\n6. Aligning rasters spatially...")
        app_transform = state.ref_transform

        # InVEST raster origin in app pixel coordinates
        inv_col_off = round((invest_transform.c - app_transform.c) / app_transform.a)
        inv_row_off = round((invest_transform.f - app_transform.f) / app_transform.e)
        print(f"   InVEST origin offset in app grid: row={inv_row_off}, col={inv_col_off}")
        print(f"   InVEST extent: [{inv_row_off}:{inv_row_off+invest_hm.shape[0]}, "
              f"{inv_col_off}:{inv_col_off+invest_hm.shape[1]}]")

        # Extract the matching sub-region from the app rasters
        r0, r1 = inv_row_off, inv_row_off + invest_hm.shape[0]
        c0, c1 = inv_col_off, inv_col_off + invest_hm.shape[1]

        # Clamp to valid range
        r0c, r1c = max(0, r0), min(app_cc.shape[0], r1)
        c0c, c1c = max(0, c0), min(app_cc.shape[1], c1)
        # Corresponding InVEST slice
        ir0, ir1 = r0c - r0, invest_hm.shape[0] - (r1 - r1c)
        ic0, ic1 = c0c - c0, invest_hm.shape[1] - (c1 - c1c)

        app_cc_cmp = app_cc[r0c:r1c, c0c:c1c].copy()
        invest_hm_cmp = invest_hm[ir0:ir1, ic0:ic1]
        invest_cc_cmp = invest_cc[ir0:ir1, ic0:ic1]
        lulc_cmp = state.cooling_lulc[r0c:r1c, c0c:c1c]
        print(f"   Aligned region shape: {app_cc_cmp.shape}")

        # Build valid masks
        invest_hm_valid = np.isfinite(invest_hm_cmp)
        if invest_hm_nodata is not None:
            invest_hm_valid &= (invest_hm_cmp != invest_hm_nodata)

        invest_cc_valid = np.isfinite(invest_cc_cmp)
        if invest_cc_nodata is not None:
            invest_cc_valid &= (invest_cc_cmp != invest_cc_nodata)

        app_valid = np.isfinite(app_cc_cmp)

        # ── Comparison 1: Raw CC (before smoothing/HMI) ──────────────
        print("\n   --- Raw CC comparison (before smoothing/HMI) ---")
        # Compute the app's raw CC (no Gaussian) for the aligned region
        safe = np.clip(lulc_cmp, 0, len(state.shade_arr) - 1)
        shade_px = state.shade_arr[safe]
        albedo_px = state.albedo_arr[safe]
        kc_px = state.kc_arr[safe]
        et_region = state.et_resized[r0c:r1c, c0c:c1c]
        eti_px = (kc_px * et_region) / state.max_et_ref
        app_raw_cmp = (0.6 * shade_px + 0.2 * albedo_px + 0.2 * eti_px).astype(np.float32)
        raw_nan = (lulc_cmp < 0) | (lulc_cmp >= len(state.shade_arr))
        app_raw_cmp[raw_nan] = np.nan

        both_valid_raw = np.isfinite(app_raw_cmp) & invest_cc_valid
        n_valid_raw = int(both_valid_raw.sum())

        if n_valid_raw > 0:
            raw_diff = invest_cc_cmp[both_valid_raw] - app_raw_cmp[both_valid_raw]
            raw_mae = float(np.mean(np.abs(raw_diff)))
            raw_corr = float(np.corrcoef(
                invest_cc_cmp[both_valid_raw].ravel(),
                app_raw_cmp[both_valid_raw].ravel()
            )[0, 1])
            raw_mean_app = float(np.mean(app_raw_cmp[both_valid_raw]))
            raw_mean_invest = float(np.mean(invest_cc_cmp[both_valid_raw]))
            print(f"   Valid pixels:     {n_valid_raw:,}")
            print(f"   App raw CC mean:  {raw_mean_app:.4f}")
            print(f"   InVEST CC mean:   {raw_mean_invest:.4f}")
            print(f"   MAE:              {raw_mae:.6f}")
            print(f"   Pearson r:        {raw_corr:.6f}")
            n_raw_large = int(np.sum(np.abs(raw_diff) > 0.01))
            print(f"   |diff| > 0.01:   {n_raw_large:,} pixels")
        else:
            raw_mae = raw_corr = raw_mean_app = raw_mean_invest = float('nan')
            n_raw_large = 0
            print("   No overlapping valid pixels for raw CC comparison!")

        # ── Comparison 2: InVEST HMI vs app smoothed CC ──────────────
        print("\n   --- HMI vs app smoothed CC comparison ---")
        both_valid_hm = app_valid & invest_hm_valid
        n_valid_hm = int(both_valid_hm.sum())

        if n_valid_hm > 0:
            hm_diff = invest_hm_cmp[both_valid_hm] - app_cc_cmp[both_valid_hm]
            hm_mae = float(np.mean(np.abs(hm_diff)))
            hm_corr = float(np.corrcoef(
                invest_hm_cmp[both_valid_hm].ravel(),
                app_cc_cmp[both_valid_hm].ravel()
            )[0, 1])
            hm_mean_invest = float(np.mean(invest_hm_cmp[both_valid_hm]))
            hm_mean_app = float(np.mean(app_cc_cmp[both_valid_hm]))
            n_hm_large = int(np.sum(np.abs(hm_diff) > 0.1))
            print(f"   Valid pixels:       {n_valid_hm:,}")
            print(f"   App smoothed CC:    {hm_mean_app:.4f}")
            print(f"   InVEST HMI mean:    {hm_mean_invest:.4f}")
            print(f"   MAE:                {hm_mae:.4f}")
            print(f"   Pearson r:          {hm_corr:.4f}")
            print(f"   |diff| > 0.1:      {n_hm_large:,} pixels")

            # Spatial analysis: where are the largest divergences?
            large_mask = both_valid_hm & (np.abs(invest_hm_cmp - app_cc_cmp) > 0.1)
            if large_mask.any():
                lulc_at_large = lulc_cmp[large_mask]
                unique_codes, counts = np.unique(lulc_at_large, return_counts=True)
                print(f"   Large-divergence LULC breakdown:")
                for code, count in sorted(zip(unique_codes, counts), key=lambda x: -x[1])[:10]:
                    print(f"     NLCD {code}: {count:,} pixels ({count/n_hm_large*100:.1f}%)")
        else:
            hm_mae = hm_corr = hm_mean_invest = hm_mean_app = float('nan')
            n_hm_large = 0
            print("   No overlapping valid pixels for HMI comparison!")

        # ── Step 6: Save diff raster ──────────────────────────────────
        out_dir = Path("comparisons")
        out_dir.mkdir(exist_ok=True)

        if n_valid_hm > 0:
            diff_raster = invest_hm_cmp.astype(np.float32) - app_cc_cmp.astype(np.float32)
            diff_raster[~both_valid_hm] = np.nan
            diff_path = out_dir / "ucm_diff_baseline_mn.tif"
            # Use the InVEST output's transform for the diff raster
            save_geotiff(diff_raster, str(diff_path), city_cfg['crs'],
                         invest_transform, dtype=np.float32)
            print(f"\n   Diff raster saved: {diff_path}")

    # ── Step 7: Write comparison CSV ──────────────────────────────────
    csv_path = out_dir / "ucm_baseline_mn.csv"

    notes = (
        "Comparison of app's Gaussian-smoothed CC raster (sigma=15 px / 450 m) "
        "against InVEST UCM's canonical HMI = max(CC_local, CC_park) with "
        "exponential decay near green areas >= 2 ha. Both use the same biophysical "
        "table (biophysical_table_urban_cooling_MN.csv) and the same LULC raster "
        "(MN baseline, no conversions). InVEST args from "
        "invest_urban_cooling_model_args_MN.json: t_ref=23.2, uhi_max=2.05, "
        "t_air_average_radius=600, green_area_cooling_distance=450, "
        "cc_method=factors. Raw CC comparison isolates biophysical-table "
        "differences (should be near-zero); HMI-vs-smoothed-CC comparison "
        "isolates the park-proximity divergence (expected to be the main gap). "
        "natcap.invest version used: see conda env. Phase 1: one scenario, "
        "one city, baseline only."
    )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'scenario', 'city',
            'app_smoothed_cc_mean', 'invest_hmi_mean',
            'hmi_vs_cc_mae', 'hmi_vs_cc_pearson_r', 'hmi_large_divergences_gt_0.1',
            'raw_cc_app_mean', 'raw_cc_invest_mean',
            'raw_cc_mae', 'raw_cc_pearson_r', 'raw_cc_large_divergences_gt_0.01',
            'n_valid_pixels_hm', 'n_valid_pixels_raw',
            'notes',
        ])
        writer.writerow([
            'UCM', 'baseline', 'Minneapolis, MN',
            f'{hm_mean_app:.4f}', f'{hm_mean_invest:.4f}',
            f'{hm_mae:.6f}', f'{hm_corr:.6f}', n_hm_large,
            f'{raw_mean_app:.4f}', f'{raw_mean_invest:.4f}',
            f'{raw_mae:.6f}', f'{raw_corr:.6f}', n_raw_large,
            n_valid_hm, n_valid_raw,
            notes,
        ])

    print(f"\n7. Results written to {csv_path}")
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Raw CC:      App mean={raw_mean_app:.4f}  InVEST mean={raw_mean_invest:.4f}  "
          f"MAE={raw_mae:.6f}  r={raw_corr:.6f}")
    print(f"  HMI vs CC:   App mean={hm_mean_app:.4f}  InVEST mean={hm_mean_invest:.4f}  "
          f"MAE={hm_mae:.4f}  r={hm_corr:.4f}")
    print(f"  Large divergences (|HMI-CC| > 0.1): {n_hm_large:,} pixels")
    print("\nDone.")


def _construct_aoi_from_raster(raster_path, output_dir):
    """Construct an AOI polygon shapefile from a raster's bounds."""
    import fiona
    from fiona.crs import from_epsg
    from shapely.geometry import box, mapping

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    aoi_path = os.path.join(output_dir, "aoi_from_raster.shp")

    schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}
    with fiona.open(aoi_path, 'w', driver='ESRI Shapefile',
                    crs=crs.to_dict(), schema=schema) as dst:
        dst.write({'geometry': mapping(geom), 'properties': {'id': 1}})

    print(f"   Constructed AOI from raster bounds: {aoi_path}")
    return aoi_path


if __name__ == "__main__":
    main()
