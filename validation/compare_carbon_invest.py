#!/usr/bin/env python3
"""compare_carbon_invest.py — Phase 1 Carbon comparison: app vs canonical InVEST.

Runs one scenario (MN Food Forest, pct_converted=10, ff_pct=100) through
both the app's evaluate_scenario and natcap.invest.carbon.execute, then
writes a single-row comparison CSV.

Usage:
    python validation/compare_carbon_invest.py

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
from rasterio.transform import from_bounds

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


# ── InVEST carbon pool table construction ────────────────────────────────────
#
# METHODOLOGY JUDGMENT CALL — read before interpreting results.
#
# The app uses single-pool annual SEQUESTRATION rates:
#   Food Forest (NLCD 41): 3.5 tons CO2e/acre/yr
#   Green Infrastructure (NLCD 90): 2.0 tons CO2e/acre/yr
#
# InVEST Carbon uses four-pool STOCK densities (metric tons C/ha):
#   c_above, c_below, c_soil, c_dead
#
# These measure fundamentally different things:
#   - App: "how much CO2e does this land cover remove from the atmosphere per year?"
#   - InVEST: "how much carbon is stored in this land cover right now?"
#
# To make them somewhat comparable, we convert the app's annual rate into
# a one-year stock increment and assign it entirely to c_above with zeros
# in other pools. This is an approximation that:
#   (a) Treats one year of sequestration as the "stock" for the alternate LULC
#   (b) Puts all carbon in one pool (real systems distribute across all four)
#   (c) Assumes baseline developed land stores zero carbon (conservative)
#
# The conversion: tons CO2e/acre/yr ÷ 3.667 (CO2e→C) × 2.471 (acres→ha)
# = metric tons C/ha for a one-year increment.
#
# This makes the comparison measure: "do both methods agree on the TOTAL
# carbon gain from converting N pixels to food forest for one year?"
# The answer is informative but bounded by the single-pool approximation.

CO2E_TO_C = 1.0 / 3.667  # tons CO2e -> tons elemental C
ACRES_TO_HA = 2.471       # acres -> hectares

# App rates in tons CO2e/acre/yr (from app.py CARBON_SEQ_RATES)
APP_RATE_FF = 3.5   # Food Forest (NLCD 41)
APP_RATE_GI = 2.0   # Green Infrastructure (NLCD 90)
APP_RATE_HD = 0.0   # High Density (NLCD 24)

# Convert to InVEST units: metric tons C/ha (one-year stock increment)
INVEST_C_FF = APP_RATE_FF * CO2E_TO_C * ACRES_TO_HA
INVEST_C_GI = APP_RATE_GI * CO2E_TO_C * ACRES_TO_HA
INVEST_C_HD = APP_RATE_HD * CO2E_TO_C * ACRES_TO_HA


def build_carbon_pools_csv(csv_path: str, unique_lucodes: set[int]):
    """Write an InVEST-format carbon pools CSV.

    All lucodes get c_above from the app's converted rates; all other
    pools are zero. Developed codes (21-24) and other non-converted
    codes get zero across all pools (baseline stores zero carbon —
    conservative assumption that lets the stock difference equal the
    sequestration from conversions only).
    """
    # Map converted codes to their one-year C stock
    code_to_c_above = {
        41: INVEST_C_FF,   # Food Forest
        90: INVEST_C_GI,   # Green Infrastructure
        24: INVEST_C_HD,   # High Density
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lucode', 'c_above', 'c_below', 'c_soil', 'c_dead'])
        for code in sorted(unique_lucodes):
            c_above = code_to_c_above.get(code, 0.0)
            writer.writerow([code, f'{c_above:.6f}', '0.0', '0.0', '0.0'])


def save_lulc_geotiff(arr: np.ndarray, path: str, crs: str,
                       ref_transform, ref_shape):
    """Save an LULC array as a GeoTIFF with correct CRS and transform."""
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs,
        transform=ref_transform,
        nodata=-128,  # app's NODATA sentinel — InVEST skips these pixels
    ) as dst:
        dst.write(arr, 1)


def run_invest_carbon(baseline_tif: str, scenario_tif: str,
                      pools_csv: str, workspace: str) -> float:
    """Run natcap.invest.carbon and return total sequestration in metric tons C."""
    import natcap.invest.carbon

    args = {
        'workspace_dir': workspace,
        'lulc_bas_path': baseline_tif,
        'calc_sequestration': True,
        'lulc_alt_path': scenario_tif,
        'carbon_pools_path': pools_csv,
        'do_valuation': False,
    }

    print("  Running natcap.invest.carbon.execute()...")
    t0 = time.time()
    natcap.invest.carbon.execute(args)
    print(f"  InVEST Carbon completed in {time.time() - t0:.1f}s")

    # Read the change raster and sum to get total sequestration
    change_path = os.path.join(workspace, 'c_change_bas_alt.tif')
    with rasterio.open(change_path) as src:
        change_data = src.read(1)
        nodata_val = src.nodata  # InVEST uses -1.0 as nodata for output
        # Units: metric tons C/ha per pixel
        # Need to multiply by pixel area in hectares
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        pixel_area_ha = pixel_area_m2 / 10_000

    # Mask out nodata pixels before summing
    if nodata_val is not None:
        valid_mask = change_data != nodata_val
        change_valid = change_data[valid_mask]
    else:
        change_valid = change_data

    # Sum: total metric tons C across all valid pixels
    # (positive = sequestration, negative = loss)
    total_c_tons = float(np.sum(change_valid * pixel_area_ha))
    return total_c_tons


def main():
    print("=" * 60)
    print("Phase 1 Carbon Comparison: App vs Canonical InVEST")
    print("=" * 60)

    # ── Step 1: Load app and run scenario ────────────────────────────────
    print("\n1. Loading app (Streamlit stub)...")
    sys.modules["streamlit"] = _StubSt()
    t0 = time.time()
    import app
    print(f"   app.py import: {time.time() - t0:.1f}s")

    print("\n2. Rebinding to Minneapolis, MN...")
    state = _rebind_city(app, "Minneapolis, MN")
    city_cfg = app.CITIES["Minneapolis, MN"]

    print("\n3. Running Food Forest scenario (pct=10, ff=100)...")
    results = app.evaluate_scenario(
        pct_converted=10,
        green_infrastructure_pct=0,
        food_forest_pct=100,
    )

    app_carbon = results['carbon_tons_co2']
    scenario_lulc = results['scenario_lulc']
    print(f"   App carbon_tons_co2 = {app_carbon}")

    # Count converted pixels for sanity check
    baseline_lulc = state.cooling_lulc
    n_changed = int(np.sum(scenario_lulc != baseline_lulc))
    n_to_ff = int(np.sum((scenario_lulc == 41) & (baseline_lulc != 41)))
    print(f"   Pixels changed: {n_changed}, of which {n_to_ff} to Food Forest (41)")
    print(f"   Expected carbon: {n_to_ff} px * {city_cfg['pixel_area_acres']} ac/px * {APP_RATE_FF} t CO2e/ac/yr = {n_to_ff * city_cfg['pixel_area_acres'] * APP_RATE_FF:.1f}")

    # ── Step 2: Save LULC rasters as GeoTIFFs ───────────────────────────
    with tempfile.TemporaryDirectory(prefix="carbon_compare_") as tmpdir:
        print(f"\n4. Saving LULC GeoTIFFs to {tmpdir}...")

        baseline_tif = os.path.join(tmpdir, "lulc_baseline.tif")
        scenario_tif = os.path.join(tmpdir, "lulc_scenario.tif")
        pools_csv = os.path.join(tmpdir, "carbon_pools.csv")

        # Ensure int dtype for LULC
        bas_arr = baseline_lulc.astype(np.int16)
        scn_arr = scenario_lulc.astype(np.int16)

        save_lulc_geotiff(bas_arr, baseline_tif, city_cfg['crs'],
                          state.ref_transform, state.ref_shape)
        save_lulc_geotiff(scn_arr, scenario_tif, city_cfg['crs'],
                          state.ref_transform, state.ref_shape)

        # ── Step 3: Build carbon pools table ─────────────────────────────
        print("\n5. Building carbon pools CSV...")
        all_codes = set(np.unique(bas_arr)) | set(np.unique(scn_arr))
        # Filter out nodata
        all_codes.discard(-128)
        build_carbon_pools_csv(pools_csv, all_codes)

        # Print table for verification
        print("   Carbon pools table (one-year stock increments):")
        print(f"   {'lucode':>8} {'c_above (t C/ha)':>18}")
        for code in sorted(all_codes):
            c = {41: INVEST_C_FF, 90: INVEST_C_GI, 24: INVEST_C_HD}.get(code, 0.0)
            marker = " <-- converted" if code in (41, 90, 24) else ""
            print(f"   {code:>8} {c:>18.6f}{marker}")

        # ── Step 4: Run InVEST Carbon ────────────────────────────────────
        print("\n6. Running InVEST Carbon...")
        invest_workspace = os.path.join(tmpdir, "invest_output")
        os.makedirs(invest_workspace)

        invest_total_c = run_invest_carbon(
            baseline_tif, scenario_tif, pools_csv, invest_workspace
        )

        # Convert InVEST output (metric tons C) to tons CO2e for comparison
        invest_co2e = invest_total_c * 3.667
        print(f"   InVEST total sequestration: {invest_total_c:.2f} metric tons C")
        print(f"   InVEST in CO2e terms:       {invest_co2e:.2f} tons CO2e")

        # ── Step 5: Compare ──────────────────────────────────────────────
        print("\n7. Comparison:")
        print(f"   App carbon_tons_co2:     {app_carbon:.1f} tons CO2e/yr")
        print(f"   InVEST (CO2e equivalent):   {invest_co2e:.1f} tons CO2e")
        abs_diff = abs(app_carbon - invest_co2e)
        pct_diff = abs_diff / app_carbon * 100 if app_carbon else float('inf')
        print(f"   Absolute difference:        {abs_diff:.2f} tons CO2e")
        print(f"   Percent difference:         {pct_diff:.2f}%")

    # ── Step 6: Write comparison CSV ─────────────────────────────────────
    out_dir = Path("comparisons")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "carbon_food_forest_mn.csv"

    notes = (
        "App uses single-pool annual sequestration rate (3.5 t CO2e/acre/yr for "
        "Food Forest); InVEST Carbon uses four-pool stock-difference between two "
        "LULC snapshots (metric tons C/ha). To bridge: app rate was converted to "
        "a one-year stock increment (t CO2e/acre/yr / 3.667 * 2.471 = t C/ha) and "
        "assigned entirely to c_above with c_below=c_soil=c_dead=0. Baseline "
        "developed-land carbon is assumed zero in both methods. The comparison "
        "measures total carbon gain from converting N pixels to food forest over "
        "one year. Numbers should agree closely because the same rate feeds both "
        "calculations — any difference comes from pixel-area rounding, nodata "
        "handling, or rasterio vs InVEST pixel-area computation. natcap.invest "
        f"version: 3.16.2. Scenario: MN Food Forest pct_converted=10 ff_pct=100. "
        f"Pixels converted to FF: {n_to_ff}. Pixel area: {city_cfg['pixel_area_acres']} acres."
    )

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario', 'city', 'app_carbon_tons_co2e_yr', 'invest_carbon_tons_co2e',
            'absolute_diff_tons_co2e', 'percent_diff', 'invest_raw_tons_c',
            'n_pixels_converted', 'pixel_area_acres', 'notes'
        ])
        writer.writerow([
            'Food Forest (pct=10, ff=100)', 'Minneapolis, MN',
            f'{app_carbon:.1f}', f'{invest_co2e:.2f}',
            f'{abs_diff:.2f}', f'{pct_diff:.2f}%',
            f'{invest_total_c:.2f}',
            n_to_ff, city_cfg['pixel_area_acres'],
            notes
        ])

    print(f"\n8. Results written to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
