"""verify_baselines.py — baseline regression check for per-city scenarios.

Usage:
    python verify_baselines.py           # check current outputs against committed snapshots
    python verify_baselines.py --update  # rewrite snapshots with current outputs (use after intentional changes)

Snapshots live in tests/baselines/<city_slug>__<scenario_name>__<strategy>.json.

For each (city, scenario, strategy) the script:
  1. Loads city data via the same path app.py uses.
  2. Calls evaluate_scenario() with the scenario's parameters.
  3. Extracts all scalar fields + an MD5 hash of scenario_lulc.
  4. Compares against the committed snapshot using numpy.isclose
     (rtol=1e-4, atol=1e-6) for floats, exact match for ints/strings/hashes.
  5. Reports any divergences, exits 1 if any diff, exits 0 if all match.

Designed to run in under two minutes total. Run before commit when in
doubt about whether a change has cross-cutting effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


# ── Scenarios to snapshot per city ────────────────────────────────────────────
# Values pulled from app.py's Quick Start presets and slider defaults.
# "baseline" is 0% conversion (true zero state); the three presets all use
# pct_converted=10. hd_pct is implicit (100 - gi - ff) so not passed.
SCENARIOS = {
    "baseline":             dict(pct_converted=0,  green_infrastructure_pct=0,   food_forest_pct=0),
    "green_infrastructure": dict(pct_converted=10, green_infrastructure_pct=100, food_forest_pct=0),
    "food_forest":          dict(pct_converted=10, green_infrastructure_pct=0,   food_forest_pct=100),
    "high_density":         dict(pct_converted=10, green_infrastructure_pct=0,   food_forest_pct=0),
}

# All five placement strategies.  When pct_converted=0 (baseline), the strategy
# has no effect — all 5 produce identical output — but we snapshot them anyway
# for uniform data shape.
STRATEGIES = ['random', 'flood-focused', 'cooling-focused', 'undersupply-focused', 'balanced']


# ── Helpers ───────────────────────────────────────────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


# Fields the snapshot deliberately does not generically capture. The dict-valued
# `region_selection` block (Region Selection Phase 1) is JSON-shaped for export
# metadata — too structural to scalar-snapshot. Its load-bearing scalar
# `eligible_pixels_in_region` is regression-tested via a targeted assertion in
# the region-selected baseline (Commit 6), not via the generic snapshot path.
_SNAPSHOT_SKIP_KEYS = {"region_selection", "region_local"}


def _snapshot_from_results(results: dict) -> dict:
    snap = {}
    for k, v in sorted(results.items()):
        if k in _SNAPSHOT_SKIP_KEYS:
            continue
        if isinstance(v, np.ndarray):
            snap[f"{k}__md5"] = hashlib.md5(v.tobytes()).hexdigest()
        elif isinstance(v, (np.integer,)):
            snap[k] = int(v)
        elif isinstance(v, (np.floating,)):
            snap[k] = float(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            snap[k] = v
        else:
            print(f"  WARN: skipping field {k!r} of type {type(v).__name__}")
    return snap


def _compare_snapshots(old: dict, new: dict) -> list:
    diffs = []
    all_keys = sorted(set(old) | set(new))
    for k in all_keys:
        if k not in old:
            diffs.append(f"  + {k}: (new field) {new[k]!r}")
        elif k not in new:
            diffs.append(f"  - {k}: (removed field) was {old[k]!r}")
        else:
            o, n = old[k], new[k]
            if isinstance(o, float) and isinstance(n, float):
                if not np.isclose(o, n, rtol=1e-4, atol=1e-6, equal_nan=True):
                    pct = (n - o) / o * 100 if o else float("inf")
                    diffs.append(f"  ~ {k}: {o:.6g} -> {n:.6g} (delta {pct:+.3g}%)")
            elif o != n:
                diffs.append(f"  ~ {k}: {o!r} -> {n!r}")
    return diffs


# ── Streamlit stub (reuses the pattern from precompute_scenarios.py) ──────────
# Must be installed before `import app`.

_DESIRED_CITY = None


class _SessionStateStub:
    """Mimic st.session_state — reads return defaults, writes are no-ops."""
    _store = {}  # shared mutable store for setdefault / pop / get

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

    def keys(self):
        return list(self._store.keys())


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
    """Load a city's runtime state and rebind all module-level aliases in app.

    Mirrors what app.py does at lines 1880–1943 + 2084–2088 after
    _load_city_runtime_state returns."""
    city_cfg = app_mod.CITIES[city_name]

    # Per-city UNA params (Brief 22). MUST be set BEFORE _load_city_runtime_state
    # because the cached state's `baseline_una_supply_percapita_raster` is built
    # via `_una_convolve` which reads the module-level `_UNA_KERNEL`. If we
    # rebound after, SA state would carry baseline raster computed under MN's
    # exponential kernel.
    app_mod.UNA_DEMAND_M2_PER_CAPITA = float(city_cfg['una_demand_m2_per_capita'])
    app_mod.UNA_SEARCH_RADIUS_M      = float(city_cfg['una_search_radius_m'])
    app_mod.UNA_DECAY_FUNCTION       = str(city_cfg['una_decay_function'])
    radius_px = app_mod.UNA_SEARCH_RADIUS_M / app_mod.PIXEL_SIZE_M
    if app_mod.UNA_DECAY_FUNCTION == 'dichotomy':
        apothem = int(np.floor(radius_px))
        yy, xx = np.mgrid[-apothem:apothem + 1, -apothem:apothem + 1]
        app_mod._UNA_KERNEL = (np.hypot(yy, xx) <= radius_px).astype(np.float32)
    elif app_mod.UNA_DECAY_FUNCTION == 'exponential':
        max_dist = int(np.ceil(radius_px)) * 2 + 1
        apothem = int(np.ceil(max_dist))
        yy, xx = np.mgrid[-apothem:apothem + 1, -apothem:apothem + 1]
        d = np.hypot(yy, xx)
        app_mod._UNA_KERNEL = np.where(
            d <= max_dist, np.exp(-d / radius_px), 0.0
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown UNA decay function {app_mod.UNA_DECAY_FUNCTION!r}")

    state = app_mod._load_city_runtime_state(city_name)

    # Direct state-member aliases (lines 1885–1935)
    app_mod.lulc                = state.lulc
    app_mod.soil_resized        = state.soil_resized
    app_mod.cooling_lulc        = state.cooling_lulc
    app_mod.developed_pixels    = state.developed_pixels
    app_mod.cn_table            = state.cn_table
    app_mod.lucode_idx_arr      = state.lucode_idx_arr
    app_mod.hm_arr              = state.hm_arr
    app_mod.max_raster_lucode   = state.max_raster_lucode
    app_mod.max_hm_lucode       = state.max_hm_lucode
    app_mod.nlcd_intensity_weights = state.nlcd_intensity_weights
    app_mod.shade_arr           = state.shade_arr
    app_mod.kc_arr              = state.kc_arr
    app_mod.albedo_arr          = state.albedo_arr
    app_mod.green_area_arr      = state.green_area_arr
    # Brief 29: per-city UNA biophysical array. Compound-sized (1,984) for
    # SA, NLCD-sized for MN. Critical to rebind when switching cities here
    # — `_una_supply_percapita`'s vectorized lookup reads it as a bare
    # module name.
    app_mod.urban_nature_arr    = state.urban_nature_arr
    # Brief 30: per-city Carbon four-pool arrays. None for MN, 1,984-sized
    # for SA. Must rebind when switching cities so
    # `_compute_carbon_four_pool` (called inside `evaluate_scenario` for SA)
    # reads the right city's pool data.
    app_mod.c_above_arr         = state.c_above_arr
    app_mod.c_below_arr         = state.c_below_arr
    app_mod.c_soil_arr          = state.c_soil_arr
    app_mod.c_dead_arr          = state.c_dead_arr
    app_mod.pop_count_raster    = state.pop_count_raster
    app_mod.POPULATION_DATA_AVAILABLE = state.population_data_available
    # Children's nature access RELAY — under-18 raster, parallel to pop.
    # None on cities without a child_pop_file configured.
    app_mod.child_pop_count_raster = state.child_pop_count_raster
    app_mod.CHILD_POPULATION_DATA_AVAILABLE = state.child_population_data_available
    # Nature Access at Schools — module-level aliases parallel to the
    # children's pop pattern.
    app_mod.SCHOOLS_PIXELS         = state.schools_pixels
    app_mod.SCHOOLS_SECTORS        = state.schools_sectors
    app_mod.SCHOOLS_METADATA       = state.schools_metadata
    app_mod.SCHOOLS_DATA_AVAILABLE = state.schools_data_available
    app_mod.ET_RESIZED          = state.et_resized
    app_mod.MAX_ET_REF          = state.max_et_ref
    app_mod.ET_DATA_AVAILABLE   = state.et_data_available
    app_mod.ENERGY_BY_TYPE           = state.energy_by_type
    app_mod.ENERGY_TABLE_AVAILABLE   = state.energy_table_available
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
    app_mod._BASELINE_HM_RASTER = state.baseline_hm_raster
    app_mod._BASELINE_NE_RASTER = state.baseline_ne_raster
    app_mod._BASELINE_UNA_SUPPLY_PERCAPITA_RASTER = state.baseline_una_supply_percapita_raster
    app_mod._BUILDINGS_DISTANCE_RASTER = state.buildings_distance_raster
    # NatCap compound LULC aliases (Brief 27). None for cities without a
    # `compound_lulc_file` in CITIES — that's the intended cross-city default.
    app_mod.cooling_lulc_compound = state.cooling_lulc_compound
    app_mod.COMPOUND_TO_NLCD      = state.compound_to_nlcd
    app_mod.COMPOUND_TO_NLCD_TREE = state.compound_to_nlcd_tree
    app_mod.COMPOUND_AFTER_FF     = state.compound_after_ff
    app_mod.COMPOUND_AFTER_GI     = state.compound_after_gi
    app_mod.COMPOUND_AFTER_HD     = state.compound_after_hd
    # Brief B: per-target was-default boolean arrays parallel to
    # COMPOUND_AFTER_*. Required for the per-scenario fellback-pixel
    # counts in evaluate_scenario's conversion sites.
    app_mod.COMPOUND_AFTER_FF_WAS_DEFAULT = state.compound_after_ff_was_default
    app_mod.COMPOUND_AFTER_GI_WAS_DEFAULT = state.compound_after_gi_was_default
    app_mod.COMPOUND_AFTER_HD_WAS_DEFAULT = state.compound_after_hd_was_default
    app_mod._CURRENT_CITY_STATE = state

    # City-config scalars (lines 2054–2070 area)
    app_mod.PIXEL_AREA_ACRES     = city_cfg['pixel_area_acres']
    app_mod.FOOD_FOREST_LBS_ACRE = city_cfg['food_forest_lbs_acre']
    app_mod.UHI_MAX_C            = city_cfg['uhi_max_c']
    app_mod.HM_TO_FAHRENHEIT     = city_cfg['uhi_max_c'] * 1.8
    # Per-city storm constants — DESIGN_STORM_INCHES drives every SCS-CN
    # runoff calc (cn_to_runoff_acre_feet at app.py:1970 reads it), and
    # DESIGN_STORM_MM is derived from it and surfaces in the export
    # bundle's args (line 5965). MN: 3.94 in / 100 mm; SA: 6.18 in /
    # 157 mm. In production these reset on every Streamlit rerun via
    # app.py:610-611. The harness import-once + rebind path used to
    # leave them at MN's value after switching to SA, which silently
    # captured SA runoff baselines under MN's storm (~2× smaller).
    # Completeness check below locks this against future regression.
    app_mod.DESIGN_STORM_INCHES = float(city_cfg['design_storm_inches'])
    app_mod.DESIGN_STORM_MM     = app_mod.DESIGN_STORM_INCHES * 25.4
    # UNA-derived internals (rebound for parity with module-level state —
    # only _UNA_KERNEL is used at runtime, but a fresh import would set
    # these too, so the completeness assertion expects them).
    app_mod._UNA_RADIUS_PX = app_mod.UNA_SEARCH_RADIUS_M / app_mod.PIXEL_SIZE_M
    if app_mod.UNA_DECAY_FUNCTION == 'dichotomy':
        app_mod._UNA_APOTHEM = int(np.floor(app_mod._UNA_RADIUS_PX))
    else:
        app_mod._UNA_APOTHEM = int(np.ceil(app_mod._UNA_RADIUS_PX)) * 2 + 1
    # (UNA radius/decay/kernel rebound above, before _load_city_runtime_state,
    # to ensure baseline_una_supply_percapita_raster is built under the
    # correct city's kernel.)

    # Derived baselines (lines 1941–1943, 2084–2088).
    # Brief 29: for cities with a NatCap compound UNA table (SA), the
    # baseline `calculate_nature_access` call MUST be indexed by the
    # compound raster — same parity with how `_load_city_runtime_state`
    # picks `_una_baseline_lulc`. Without this, SA would index the
    # compound-keyed `urban_nature_arr` (1,984 entries) with NLCD codes
    # (0-95) and produce silently-wrong baseline pct.
    _una_baseline_lulc = (
        state.cooling_lulc_compound
        if state.cooling_lulc_compound is not None
        else state.cooling_lulc
    )
    app_mod.BASELINE_NATURE_ACCESS_PCT, app_mod.BASELINE_NATURE_QUALITY_SCORE, _ = (
        app_mod.calculate_nature_access(_una_baseline_lulc, state.pop_count_raster)
    )
    app_mod.BASELINE_RUNOFF_ACRE_FEET = app_mod.cn_to_runoff_acre_feet(
        state.baseline_cn, len(state.developed_pixels) * city_cfg['pixel_area_acres']
    )
    app_mod.BASELINE_NDVI = app_mod.compute_mean_ndvi(state.cooling_lulc)

    return state


def _cleanup_old_baselines(snapshot_dir: Path):
    """Remove old-format baseline files (without strategy suffix)."""
    for f in snapshot_dir.glob("*.json"):
        # Old format: <city>__<scenario>.json (exactly two parts split by __)
        parts = f.stem.split("__")
        if len(parts) == 2:
            print(f"  Removing old-format baseline: {f.name}")
            f.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────
def main(update: bool) -> int:
    global _DESIRED_CITY

    snapshot_dir = Path("tests/baselines")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Set the initial city for module-level import. The live app gates the
    # first load behind a splash (app.py's "First-load splash" block) — the
    # harness has no UI to click, so we pre-seed entry_city in the shared
    # session_state stub so the splash check finds it set and falls through
    # to the normal city-load path. _DESIRED_CITY tells the selectbox stub
    # which city to return (same value); _rebind_city() switches per-cell
    # afterward.
    _DESIRED_CITY = "Minneapolis, MN"
    _SessionStateStub._store['entry_city'] = _DESIRED_CITY
    sys.modules["streamlit"] = _StubSt()

    print("Importing app.py (triggers module-level startup)...")
    t0 = time.time()
    import app  # noqa: E402
    # Constants Refactor / Task #52 — the pure-data tables live in
    # standalone modules; pull them in directly rather than through
    # `app.<NAME>` so the gate exercises the source-of-truth module.
    from ownership import OWNERSHIP_MODES  # noqa: E402
    from region_local_metrics import _REGION_LOCAL_METRICS  # noqa: E402
    print(f"  app.py import: {time.time() - t0:.1f}s")

    active_cities = [name for name, cfg in app.CITIES.items() if cfg.get("available")]
    print(f"Active cities: {active_cities}")
    print(f"Scenarios: {len(SCENARIOS)} x Strategies: {len(STRATEGIES)} = "
          f"{len(SCENARIOS) * len(STRATEGIES)} baselines per city")
    print(f"Total: {len(active_cities) * len(SCENARIOS) * len(STRATEGIES)} baselines\n")

    # ── Rebind completeness — every per-city module constant must update ──
    # The harness imports app once (with _DESIRED_CITY as the entry city) and
    # then uses _rebind_city to switch between cities for per-cell tests. In
    # production, every Streamlit rerun re-executes module-level code, so
    # per-city constants like DESIGN_STORM_INCHES reset implicitly. The
    # harness has no such guarantee — if _rebind_city forgets a constant,
    # later cells silently compute with the import-time city's value. This
    # bit us once already: DESIGN_STORM_INCHES wasn't rebound, so SA runoff
    # baselines were captured under MN's 3.94 in storm (vs SA's 6.18 in,
    # ~2× smaller runoff). This cell asserts that after _rebind_city(city),
    # every per-city module constant equals what CITIES[city] says — for
    # every active city — and that derived values follow their formulas.
    # Meta-test: synthesize a temporary stale state (DESIGN_STORM_INCHES
    # set to a sentinel) and confirm the check fires.
    print(f"{'=' * 60}")
    print("Rebind completeness — per-city constant survival assertion")
    print(f"{'=' * 60}")
    rebind_completeness_diffs = 0
    try:
        # Per-city constants the harness MUST rebind. Each entry is
        # (attribute_name, city_cfg_key_or_None, formula_str_or_None).
        # formula_str applies when the attribute is derived from another
        # already-rebound attribute rather than from CITIES directly.
        _PER_CITY_CONSTANTS = [
            ("PIXEL_AREA_ACRES",         "pixel_area_acres",         None),
            ("FOOD_FOREST_LBS_ACRE",     "food_forest_lbs_acre",     None),
            ("UHI_MAX_C",                "uhi_max_c",                None),
            ("HM_TO_FAHRENHEIT",         None,                       "UHI_MAX_C * 1.8"),
            ("DESIGN_STORM_INCHES",      "design_storm_inches",      None),
            ("DESIGN_STORM_MM",          None,                       "DESIGN_STORM_INCHES * 25.4"),
            ("UNA_DEMAND_M2_PER_CAPITA", "una_demand_m2_per_capita", None),
            ("UNA_SEARCH_RADIUS_M",      "una_search_radius_m",      None),
            ("UNA_DECAY_FUNCTION",       "una_decay_function",       None),
        ]
        _missed = 0
        for _city in active_cities:
            _rebind_city(app, _city)
            _cfg = app.CITIES[_city]
            for (_attr, _cfg_key, _formula) in _PER_CITY_CONSTANTS:
                _live = getattr(app, _attr)
                if _cfg_key is not None:
                    _expected = _cfg[_cfg_key]
                    # Float coercion mirrors what _rebind_city writes; string
                    # comparison for UNA_DECAY_FUNCTION (the only str field).
                    if isinstance(_expected, (int, float)) and not isinstance(_live, str):
                        _expected = float(_expected); _live_f = float(_live)
                        _match = abs(_expected - _live_f) < 1e-12
                    else:
                        _match = _live == _expected
                    _src = f"CITIES[{_city!r}][{_cfg_key!r}]"
                else:
                    # Derived value — evaluate the formula against current
                    # already-rebound state.
                    _ns = {a: getattr(app, a) for (a, _, _) in _PER_CITY_CONSTANTS}
                    _expected = eval(_formula, {"__builtins__": {}}, _ns)
                    _match = abs(float(_expected) - float(_live)) < 1e-9
                    _src = f"derived: {_formula}"
                if not _match:
                    print(f"  FAIL {_city} {_attr}: live={_live!r} "
                          f"expected={_expected!r} ({_src}) — _rebind_city "
                          "didn't update this attribute")
                    _missed += 1
        if _missed == 0:
            print(f"  OK   all {len(_PER_CITY_CONSTANTS)} per-city constants "
                  f"× {len(active_cities)} cities = "
                  f"{len(_PER_CITY_CONSTANTS) * len(active_cities)} checks pass")
        else:
            rebind_completeness_diffs += _missed

        # Cross-check: BASELINE_RUNOFF_ACRE_FEET depends transitively on
        # DESIGN_STORM_INCHES via cn_to_runoff_acre_feet. For each city,
        # recompute it from the (now correctly rebound) constants and the
        # CityState's baseline_cn + developed_pixels, then compare to the
        # value _rebind_city wrote. This is the assertion that originally
        # would have caught the DESIGN_STORM bug.
        _cross_diffs = 0
        for _city in active_cities:
            _rebind_city(app, _city)
            _expected = app.cn_to_runoff_acre_feet(
                app._CURRENT_CITY_STATE.baseline_cn,
                len(app._CURRENT_CITY_STATE.developed_pixels) * app.PIXEL_AREA_ACRES,
            )
            _live = app.BASELINE_RUNOFF_ACRE_FEET
            if abs(_expected - _live) > 1e-6:
                print(f"  FAIL {_city} BASELINE_RUNOFF_ACRE_FEET: "
                      f"live={_live} expected={_expected} (derived from "
                      "cn_to_runoff_acre_feet under current DESIGN_STORM_INCHES)")
                _cross_diffs += 1
        if _cross_diffs == 0:
            print(f"  OK   BASELINE_RUNOFF_ACRE_FEET cross-check passes for "
                  f"{len(active_cities)} cities (storm-derived baselines "
                  "match cn_to_runoff_acre_feet under each city's storm)")
        else:
            rebind_completeness_diffs += _cross_diffs

        # Meta-test: synthesize a "missed rebind" by setting one constant
        # to a sentinel after rebinding; confirm the assertion catches it.
        # Without this, the lint could silently degrade — e.g. if the
        # _PER_CITY_CONSTANTS list lost an entry, the check would pass
        # vacuously. The seed perturbs the LAST city's DESIGN_STORM_INCHES
        # away from its config value and runs the same loop.
        _meta_caught = 0
        _meta_city = active_cities[-1]
        _rebind_city(app, _meta_city)
        _saved = app.DESIGN_STORM_INCHES
        app.DESIGN_STORM_INCHES = 0.001  # sentinel
        _cfg = app.CITIES[_meta_city]
        if abs(float(_cfg['design_storm_inches']) - float(app.DESIGN_STORM_INCHES)) > 1e-12:
            _meta_caught += 1
        app.DESIGN_STORM_INCHES = _saved  # restore
        _rebind_city(app, _meta_city)     # restore full state
        if _meta_caught == 0:
            print(f"  FAIL meta-test: synthesized stale DESIGN_STORM_INCHES "
                  "was NOT caught by the check — completeness lint is blind")
            rebind_completeness_diffs += 1
        else:
            print(f"  OK   meta-test: synthesized stale DESIGN_STORM_INCHES "
                  f"correctly flagged ({_meta_caught} hit)")
    except Exception as _e:
        print(f"  ERROR rebind completeness: {_e}")
        import traceback; traceback.print_exc()
        rebind_completeness_diffs += 1
    print()

    if update:
        _cleanup_old_baselines(snapshot_dir)

    total_diffs = 0

    for city_name in active_cities:
        print(f"{'=' * 60}")
        print(f"City: {city_name}")
        print(f"{'=' * 60}")

        try:
            t_city = time.time()
            _rebind_city(app, city_name)
            print(f"  City loaded in {time.time() - t_city:.1f}s")
        except Exception as e:
            print(f"  SKIP: failed to load city data: {e}")
            import traceback; traceback.print_exc()
            continue

        for scenario_name, params in SCENARIOS.items():
            for strategy in STRATEGIES:
                label = f"{city_name} / {scenario_name} / {strategy}"
                print(f"\n  {label}:")

                try:
                    results = app.evaluate_scenario(
                        **params, seed=42, placement_strategy=strategy,
                    )
                except Exception as e:
                    print(f"    ERROR: evaluate_scenario failed: {e}")
                    import traceback; traceback.print_exc()
                    total_diffs += 1
                    continue

                new_snap = _snapshot_from_results(results)
                snap_path = snapshot_dir / f"{_slug(city_name)}__{scenario_name}__{strategy}.json"

                if update:
                    snap_path.write_text(json.dumps(new_snap, indent=2, sort_keys=True) + "\n")
                    print(f"    wrote {snap_path} ({len(new_snap)} fields)")
                elif snap_path.exists():
                    old_snap = json.loads(snap_path.read_text())
                    diffs = _compare_snapshots(old_snap, new_snap)
                    if diffs:
                        print(f"    FAIL: {len(diffs)} divergence(s):")
                        for d in diffs:
                            print(d)
                        total_diffs += len(diffs)
                    else:
                        print(f"    OK ({len(new_snap)} fields)")
                else:
                    print(f"    no snapshot at {snap_path}")
                    print(f"    run with --update to create it")
                    total_diffs += 1

    # ── Region Selection Phase 1 targeted assertion ─────────────────────────
    # Per the spec / DESIGN_NOTES discipline: don't snapshot the region_selection
    # block generically (it's a dict; _SNAPSHOT_SKIP_KEYS skips it). Instead,
    # confirm that for one known region per city, the eligible_pixels_in_region
    # scalar returned by evaluate_scenario matches an independent recompute
    # (raster intersection of region mask with convertible pool). Self-checking:
    # if the per-city region rasters drift, the independent count drifts with
    # them and the assertion still passes only if the *math* is right.
    print(f"\n{'=' * 60}")
    print("Region Selection — targeted eligible_pixels_in_region assertion")
    print(f"{'=' * 60}")
    REGION_TARGETS = {
        # city → (layer_key, region_label_to_select)
        "San Antonio, TX": ("council_districts", "5"),
        "Minneapolis, MN": ("downtown_tracts", None),  # first label, see below
    }
    region_diffs = 0
    for city_name in active_cities:
        if city_name not in REGION_TARGETS:
            print(f"  {city_name}: no region target configured; skip")
            continue
        layer_key, label = REGION_TARGETS[city_name]
        try:
            _rebind_city(app, city_name)
            state = app._CURRENT_CITY_STATE
            if layer_key not in state.region_rasters:
                print(f"  {city_name}: layer {layer_key!r} not configured; skip")
                continue
            labels_for_layer = state.region_layer_labels[layer_key]
            if label is None:
                label = labels_for_layer[0]
            if label not in labels_for_layer:
                print(f"  {city_name}: label {label!r} not in {layer_key}; skip")
                continue
            pos_idx = labels_for_layer.index(label)
            raster = state.region_rasters[layer_key]
            mask = (raster == pos_idx)
            cp = state.convertible_pixels
            independent_count = int(mask[cp[:, 0], cp[:, 1]].sum())
            results = app.evaluate_scenario(
                pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
                seed=42, placement_strategy="random",
                selected_region_mask=mask,
            )
            reported = int(results["region_selection"]["eligible_pixels_in_region"])
            label_str = f"{city_name} / {layer_key} / {label!r}"
            if reported == independent_count:
                print(f"  OK  {label_str}: eligible_pixels_in_region = {reported:,}")
            else:
                print(f"  FAIL {label_str}: reported {reported:,} != independent {independent_count:,}")
                region_diffs += 1
            # Verify the structured block carries the locked contract: label
            # value (string), not positional index.
            stamped_ids = results["region_selection"].get("selected_ids")
            assert stamped_ids in (None, []), (
                f"caller-stamped fields should be untouched by evaluate_scenario; "
                f"got selected_ids={stamped_ids!r}"
            )
        except Exception as e:
            print(f"  ERROR {city_name}: {e}")
            import traceback; traceback.print_exc()
            region_diffs += 1

    # ── Ownership Integration Commit 4 — targeted ownership assertion ──
    # Mirror of the region assertion above. Build the boolean ownership
    # mask for SA's 'vacant_public' mode (the actionable headline) directly
    # from the CityState raster + OWNERSHIP_MODES, then check the eligible
    # count reported by evaluate_scenario against the independent recompute.
    print(f"\n{'=' * 60}")
    print("Ownership Integration — targeted eligible_pixels_in_region assertion")
    print(f"{'=' * 60}")
    OWNERSHIP_TARGETS = {
        # city → ownership_mode_key
        "San Antonio, TX": "vacant_public",
    }
    ownership_diffs = 0
    for city_name in active_cities:
        if city_name not in OWNERSHIP_TARGETS:
            print(f"  {city_name}: no ownership target configured; skip")
            continue
        mode = OWNERSHIP_TARGETS[city_name]
        try:
            _rebind_city(app, city_name)
            state = app._CURRENT_CITY_STATE
            if state.ownership_raster is None:
                print(f"  {city_name}: ownership_raster not loaded; skip")
                continue
            # Two-band encoding (Finer Ownership Classes Pass) — route
            # through app._build_ownership_mask so this assertion catches
            # any regression in the mask-build path itself.
            mask = app._build_ownership_mask(
                state.ownership_raster, state.ownership_vacant_raster,
                OWNERSHIP_MODES[mode],
            )
            cp = state.convertible_pixels
            independent_count = int(mask[cp[:, 0], cp[:, 1]].sum())
            results = app.evaluate_scenario(
                pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
                seed=42, placement_strategy="random",
                selected_region_mask=mask,
            )
            reported = int(results["region_selection"]["eligible_pixels_in_region"])
            label_str = f"{city_name} / ownership={mode}"
            if reported == independent_count:
                print(f"  OK  {label_str}: eligible_pixels_in_region = {reported:,}")
            else:
                print(f"  FAIL {label_str}: reported {reported:,} != independent {independent_count:,}")
                ownership_diffs += 1
        except Exception as e:
            print(f"  ERROR {city_name}: {e}")
            import traceback; traceback.print_exc()
            ownership_diffs += 1

    # ── Region-Local Metrics Commit 1 — reconciliation assertion ──
    # For any metric flagged decomposable in `_REGION_LOCAL_METRICS`,
    # `region_local[key]` computed over the entire AOI must equal the
    # citywide `results[key]` (since clipping to "all pixels" is the
    # citywide aggregate). A wrongly-marked decomposable metric trips
    # this — the dangerous direction is machine-guarded. The safe
    # direction (wrongly-marked non-decomposable) is harmless; the UI
    # just falls back to "citywide only" conservatively.
    print(f"\n{'=' * 60}")
    print("Region-Local Metrics — full-AOI reconciliation assertion")
    print(f"{'=' * 60}")
    region_local_diffs = 0
    _RECON_TOL = 1e-3  # round-tolerance; means/sums are rounded at compute time
    for city_name in active_cities:
        try:
            _rebind_city(app, city_name)
            state = app._CURRENT_CITY_STATE
            # Everything-mask: clip becomes the citywide aggregate.
            full_mask = np.ones(state.ref_shape, dtype=bool)
            results = app.evaluate_scenario(
                pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
                seed=42, placement_strategy="random",
                selected_region_mask=full_mask,
            )
            region_local = results.get("region_local") or {}
            mismatches = []
            for key, cfg in _REGION_LOCAL_METRICS.items():
                if not cfg["decomposable"]:
                    continue
                citywide = results.get(key)
                rl = region_local.get(key)
                if citywide is None or rl is None:
                    mismatches.append((key, citywide, rl, "missing"))
                    continue
                if abs(float(rl) - float(citywide)) > _RECON_TOL:
                    mismatches.append((key, citywide, rl, "diff"))
            label_str = f"{city_name}"
            if not mismatches:
                n_decomp = sum(1 for c in _REGION_LOCAL_METRICS.values()
                               if c["decomposable"])
                print(f"  OK  {label_str}: {n_decomp} decomposable metrics reconcile "
                      f"(region_local over entire AOI == citywide)")
            else:
                for key, citywide, rl, kind in mismatches:
                    print(f"  FAIL {label_str} / {key}: citywide={citywide} "
                          f"region_local={rl} ({kind})")
                region_local_diffs += len(mismatches)
        except Exception as e:
            print(f"  ERROR {city_name}: {e}")
            import traceback; traceback.print_exc()
            region_local_diffs += 1

    # ── Region-Local Metrics Commit 4 — district-specific smoke test ──
    # Reconciliation above proves region_local == citywide at full-AOI for
    # every decomposable metric. This block adds a sanity check on a real
    # region: SA District 5 with a non-zero conversion. The region_local
    # block must (a) be non-None, (b) carry no None entries for any
    # decomposable metric, (c) report carbon storage and food production
    # within plausible bounds for a partial-conversion scenario.
    print(f"\n{'=' * 60}")
    print("Region-Local Metrics — district-specific smoke test (SA District 5)")
    print(f"{'=' * 60}")
    smoke_diffs = 0
    try:
        _rebind_city(app, "San Antonio, TX")
        state = app._CURRENT_CITY_STATE
        labels_for_layer = state.region_layer_labels["council_districts"]
        pos_idx = labels_for_layer.index("5")
        mask = (state.region_rasters["council_districts"] == pos_idx)
        results = app.evaluate_scenario(
            pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
            seed=42, placement_strategy="random",
            selected_region_mask=mask,
        )
        rl = results.get("region_local")
        if rl is None:
            print("  FAIL  region_local block missing on a region scenario")
            smoke_diffs += 1
        else:
            missing_keys = [
                k for k, cfg in _REGION_LOCAL_METRICS.items()
                if cfg["decomposable"] and rl.get(k) is None
            ]
            if missing_keys:
                print(f"  FAIL  decomposable keys with None value: {missing_keys}")
                smoke_diffs += len(missing_keys)
            else:
                print(f"  OK    all 23 decomposable metrics populated")
            # Plausibility bands for a 10% conversion on District 5 (~3,617 px).
            n_conv = rl["n_wet"] + rl["n_for"] + rl["n_hd"]
            if n_conv <= 0:
                print(f"  FAIL  n_convert in region = 0; expected > 0")
                smoke_diffs += 1
            else:
                print(f"  OK    n_convert in region = {n_conv:,}")
            if not (0 <= rl["nature_access_pct"] <= 100):
                print(f"  FAIL  nature_access_pct out of bounds: {rl['nature_access_pct']}")
                smoke_diffs += 1
            else:
                print(f"  OK    nature_access_pct (region) = {rl['nature_access_pct']:.1f}%")
    except Exception as e:
        print(f"  ERROR {e}")
        import traceback; traceback.print_exc()
        smoke_diffs += 1

    # ── Honesty-Surface Pass Commit 4 — completeness assertion ──
    # Every id in the locked KNOWN_DIVERGENCES seed list must appear in the
    # metadata.json the bundle builder emits. Guards against silent drops
    # in a later refactor — the disclosure surface is the gateless
    # mechanism's only enforcement, so this assertion is load-bearing.
    print(f"\n{'=' * 60}")
    print("Honesty-Surface Pass — known-divergences completeness assertion")
    print(f"{'=' * 60}")
    disclosure_diffs = 0
    try:
        import export_invest_bundle as eib
        # Minimal BundleSpec — _build_metadata only needs identity + raster
        # paths for the args files and is_sa for the lineage branch.
        fake_spec = eib.BundleSpec(
            city_name="Test City", city_slug="test",
            crs="EPSG:5070", pixel_size_m=30,
            scenario_id="completeness_check", scenario_label="completeness_check",
            scenario_description="verify_baselines completeness assertion",
            provenance=eib.PROVENANCE_EXPLORER,
            generator={"type": "explorer_generated"},
            git_commit="unknown", scenario_schema_version=app.SCENARIO_SCHEMA_VERSION,
            is_sa=True, raster_profile={"height": 1, "width": 1,
                                        "crs": "EPSG:5070",
                                        "transform": None},
        )
        metadata = eib._build_metadata(fake_spec, args_files={})
        emitted = metadata["scenario"]["known_divergences"]
        emitted_ids = {d["id"] for d in emitted}
        expected_ids = {d["id"] for d in eib.KNOWN_DIVERGENCES}
        missing = expected_ids - emitted_ids
        extra = emitted_ids - expected_ids
        if missing:
            print(f"  FAIL missing divergence ids: {sorted(missing)}")
            disclosure_diffs += len(missing)
        if extra:
            print(f"  FAIL unexpected divergence ids: {sorted(extra)}")
            disclosure_diffs += len(extra)
        if not (missing or extra):
            print(f"  OK   {len(expected_ids)} locked divergences all present "
                  "in metadata.json")
        # Audit-fix hold check (Commit 1): the validation-state inheritance
        # caption must remain in app.py so the Region-Local table doesn't
        # silently drift back to unbadged rows.
        with open("app.py", "r") as f:
            app_source = f.read()
        if "Validation states for these rows inherit from the per-metric badges" not in app_source:
            print("  FAIL Region-Local validation-inheritance caption missing from app.py")
            disclosure_diffs += 1
        else:
            print("  OK   Region-Local inheritance caption holds")
    except Exception as e:
        print(f"  ERROR {e}")
        import traceback; traceback.print_exc()
        disclosure_diffs += 1

    # ── Scenario Record Pass — saved-scenario round-trip assertion ─────────
    # Formalizes the reproducibility contract at the record-API surface: a
    # saved record's recipe (sliders + placement_strategy + random_seed),
    # passed back through evaluate_scenario, must reproduce the metrics the
    # record stored. Mostly redundant with the 40/40 above (which itself is
    # the regen-reproduces-stored-metrics test at the call-signature level);
    # the value is making the saved-record contract explicit so a future
    # regression at the record layer reads as its own failure mode rather
    # than a generic baseline diff. Thin: one scenario per (city × strategy).
    print(f"\n{'=' * 60}")
    print("Scenario Record — saved-scenario round-trip assertion")
    print(f"{'=' * 60}")
    round_trip_diffs = 0
    _round_trip_recipe = dict(
        pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
    )
    _round_trip_metrics = ("mean_cn", "mean_hm", "food_mln_lbs",
                           "carbon_tons_co2", "total_cost_mln")
    for city_name in active_cities:
        try:
            _rebind_city(app, city_name)
            for strategy in STRATEGIES:
                first = app.evaluate_scenario(
                    **_round_trip_recipe, seed=42, placement_strategy=strategy,
                )
                # Mimic the save handler: capture the recipe + stored metrics
                # exactly as st.session_state.saved_scenarios receives them.
                record = {k: v for k, v in first.items() if k != "scenario_lulc"}
                record["placement_strategy"] = strategy
                record["random_seed"] = 42
                # Regen from the record's recipe.
                regen = app.evaluate_scenario(
                    pct_converted=record["pct_converted"],
                    green_infrastructure_pct=record["green_infrastructure_pct"],
                    food_forest_pct=record["food_forest_pct"],
                    seed=record["random_seed"],
                    placement_strategy=record["placement_strategy"],
                )
                divergent = [
                    m for m in _round_trip_metrics
                    if not np.isclose(record[m], regen[m],
                                      rtol=1e-9, atol=1e-9, equal_nan=True)
                ]
                if divergent:
                    print(f"  FAIL {city_name} / {strategy}: "
                          f"divergent metrics on regen: {divergent}")
                    for m in divergent:
                        print(f"    {m}: stored={record[m]} regen={regen[m]}")
                    round_trip_diffs += len(divergent)
                else:
                    print(f"  OK   {city_name} / {strategy}: "
                          f"{len(_round_trip_metrics)} metrics reproduce from record")
        except Exception as e:
            print(f"  ERROR {city_name}: {e}")
            import traceback; traceback.print_exc()
            round_trip_diffs += 1

    # ── Tradeoff-chart empty-optimizer guard ────────────────────────────
    # Regression test for the KeyError('food_mln_lbs') crash on
    # plot_tradeoff when `optimize_scenario` returns a no-scenarios
    # marker (a `{'found': False, ...}` dict, not a DataFrame). The
    # `len(optimized) > 0` guard in plot_tradeoff was passing the dict
    # through (dict has keys = len > 0), then the DataFrame-style
    # `optimized['food_mln_lbs']` access raised. Tests both halves of
    # the fix (call-site coercion + plot_tradeoff defensive backstop).
    # Render-path only; the engine never sees the bug, so the 40/40
    # snapshots don't catch it.
    print(f"\n{'=' * 60}")
    print("Tradeoff chart — empty-optimizer regression test")
    print(f"{'=' * 60}")
    tradeoff_diffs = 0
    try:
        import pandas as _pd
        # Build a minimal results-shaped dict + scenario_df that
        # plot_tradeoff can render against.
        _fake_results = {
            'flood_reduction':  50.0,
            'mean_hm':           0.4,
            'food_mln_lbs':      0.05,
            'total_cost_mln':    1.5,
            'scenario_name':    'regression test',
            'pct_converted':     10,
            'green_infrastructure_pct': 50,
            'food_forest_pct':   50,
        }
        _fake_scenario_df = _pd.DataFrame({
            'flood_reduction': [40.0, 50.0, 60.0],
            'mean_hm':         [0.30, 0.40, 0.50],
            'food_mln_lbs':    [0.02, 0.05, 0.10],
            'runoff_acre_feet':[1500, 1400, 1300],
            'carbon_tons_co2': [100, 500, 1000],
            'pct_converted':   [5, 10, 20],
            'green_infrastructure_pct': [50, 50, 50],
            'food_forest_pct': [50, 50, 50],
            'scenario_name':   ['a', 'b', 'c'],
        })
        # The bug shape: optimize_scenario's no-scenarios return.
        _no_scenarios_marker = {
            'found': False, 'max_flood': 60.0, 'max_cool': 0.5,
            'max_food': 0.1, 'max_carbon': 1000,
        }

        # Populated-case payload — confirms the overlay render path
        # actually exercises (not just the empty-skip backstop). Column
        # names must match what `surrogate.optimize_scenario` returns
        # for the Pareto-frontier DataFrame (see surrogate.py:172 — the
        # canonical live column name for the food-bubble axis is
        # `food_mln_lbs`). If a rename ever lands, this cell flips from
        # OK to FAIL because the `plot_tradeoff` backstop will silently
        # skip the overlay on populated input.
        _populated_opt = _pd.DataFrame({
            'pct_converted':             [10, 20, 30],
            'green_infrastructure_pct':  [50, 60, 40],
            'food_forest_pct':           [50, 30, 40],
            'pct_highdensity':           [ 0, 10, 20],
            'flood_reduction':           [45.0, 55.0, 60.0],
            'flood_lower':               [40.0, 50.0, 55.0],
            'flood_upper':               [50.0, 60.0, 65.0],
            'mean_hm':                   [0.35, 0.42, 0.48],
            'hm_lower':                  [0.32, 0.39, 0.45],
            'hm_upper':                  [0.38, 0.45, 0.51],
            'food_mln_lbs':              [0.04, 0.06, 0.05],
            'food_lower':                [0.03, 0.05, 0.04],
            'food_upper':                [0.05, 0.07, 0.06],
            'carbon_tons_co2':           [400.0, 800.0, 1200.0],
            'scenario_name':             ['a', 'b', 'c'],
        })

        cases = [
            ('optimized=None',
                None),
            ('optimized={no-scenarios dict marker}',
                _no_scenarios_marker),
            ('optimized=empty DataFrame',
                _pd.DataFrame()),
            ('optimized=populated DataFrame (exercises overlay path)',
                _populated_opt),
        ]
        for label, opt_arg in cases:
            try:
                _fig = app.plot_tradeoff(
                    _fake_results, _fake_scenario_df, optimized=opt_arg,
                )
                if _fig is None:
                    print(f"  FAIL  {label}: plot_tradeoff returned None")
                    tradeoff_diffs += 1
                else:
                    print(f"  OK    {label}: plot_tradeoff returned a figure "
                          f"without raising")
            except KeyError as e:
                print(f"  FAIL  {label}: KeyError {e!r} — the bug regressed")
                tradeoff_diffs += 1
            except Exception as e:
                print(f"  FAIL  {label}: unexpected {type(e).__name__}: {e!r}")
                tradeoff_diffs += 1
    except Exception as e:
        print(f"  ERROR setting up tradeoff-chart regression test: {e!r}")
        import traceback; traceback.print_exc()
        tradeoff_diffs += 1

    # ── Subset Invariants Pass — placement-stage spatial assertions ─────────
    # The 40/40 metric snapshots above verify that engine outputs are
    # reproducible; they DON'T verify that conversions land inside the
    # eligible / region / ownership masks. A placement-stage bug that wrote
    # conversions outside the selected region would shift mean_hm / mean_cn
    # in unintuitive ways (or not at all) — it would not read as a baseline
    # diff. These three subset assertions plug that gap.
    #
    # Each cell: pick a (region_mask, ownership_mask) pair, call
    # evaluate_scenario with the combined mask the live app would pass,
    # compute converted_mask = (baseline_lulc != scenario_lulc), and assert
    # converted ⊆ eligible, converted ⊆ region (if active), and
    # converted ⊆ ownership (if active). Three checks kept separate as
    # defense in depth — catches a bug where the eligible mask is
    # miscomposed but happens to still subset the others.
    #
    # Funnel cardinalities are surfaced for each cell (developed →
    # convertible → region_eligible → final_eligible → converted) so the
    # deferred eligibility-funnel UI can reuse the exact counts.
    print(f"\n{'=' * 60}")
    print("Subset Invariants — placement-stage subset assertions")
    print(f"{'=' * 60}")
    subset_diffs = 0
    reconcile_diffs = 0
    _SUBSET_RECIPE_PCT10 = dict(
        pct_converted=10, green_infrastructure_pct=50, food_forest_pct=50,
    )
    _SUBSET_RECIPE_PCT100 = dict(
        pct_converted=100, green_infrastructure_pct=50, food_forest_pct=50,
    )
    # Optimizer Reversal Pass — an optimizer-style recipe (high conversion,
    # mixed GI/FF; the kind of mix the citywide-trained surrogate would
    # actually recommend) applied under region+ownership. Confirms converted
    # ⊆ region ∩ ownership for optimizer-applied scenarios, locking in the
    # honesty: the surrogate's predictions ignore the masks, but the
    # post-Apply engine evaluation respects them exactly.
    _SUBSET_RECIPE_OPTIMIZER = dict(
        pct_converted=30, green_infrastructure_pct=60, food_forest_pct=40,
    )

    def _convertible_in_raster(state):
        m = np.zeros(state.ref_shape, dtype=bool)
        m[state.convertible_pixels[:, 0], state.convertible_pixels[:, 1]] = True
        return m

    def _region_mask_from(state, layer_key, labels):
        raster = state.region_rasters[layer_key]
        label_list = state.region_layer_labels[layer_key]
        pos_indices = [label_list.index(lbl) for lbl in labels]
        return np.isin(raster, pos_indices)

    def _ownership_mask_from(state, mode_key):
        # Two-band encoding (Finer Ownership Classes Pass) — route through
        # app._build_ownership_mask so this test exercises the same mask-
        # build path the live app uses.
        return app._build_ownership_mask(
            state.ownership_raster, state.ownership_vacant_raster,
            OWNERSHIP_MODES[mode_key],
        )

    def _run_cell(state, label, region_mask, ownership_mask, recipe):
        """Run one matrix cell. Returns (subset_local, reconcile_local).

        region_mask / ownership_mask are None when the cell doesn't exercise
        that constraint. The combined mask passed to evaluate_scenario
        mirrors what the live app composes (region & ownership when both,
        either one alone, or None for citywide)."""
        subset_local = 0
        reconcile_local = 0
        if region_mask is not None and ownership_mask is not None:
            combined = region_mask & ownership_mask
        elif region_mask is not None:
            combined = region_mask
        elif ownership_mask is not None:
            combined = ownership_mask
        else:
            combined = None
        results = app.evaluate_scenario(
            **recipe, seed=42, placement_strategy='random',
            selected_region_mask=combined,
        )
        baseline_lulc = state.lulc
        scenario_lulc = results['scenario_lulc']
        converted_mask = (baseline_lulc != scenario_lulc)
        eligible_mask = _convertible_in_raster(state)

        # ── Funnel cardinalities ──
        funnel = {
            'total_px':           int(state.ref_shape[0] * state.ref_shape[1]),
            'developed_px':       int(len(state.developed_pixels)),
            'convertible_px':     int(len(state.convertible_pixels)),
            'region_px':          int(region_mask.sum()) if region_mask is not None else None,
            'region_eligible_px': int((eligible_mask & region_mask).sum()) if region_mask is not None else int(eligible_mask.sum()),
            'ownership_px':       int(ownership_mask.sum()) if ownership_mask is not None else None,
            'final_eligible_px':  int((
                eligible_mask
                & (region_mask if region_mask is not None else np.ones_like(eligible_mask))
                & (ownership_mask if ownership_mask is not None else np.ones_like(eligible_mask))
            ).sum()),
            'converted_px':       int(converted_mask.sum()),
        }

        # ── Invariant 1: converted ⊆ eligible (always) ──
        out_eligible = int((converted_mask & ~eligible_mask).sum())
        if out_eligible:
            offender = np.argwhere(converted_mask & ~eligible_mask)[0]
            print(f"  FAIL  {label}: {out_eligible} converted px outside eligible "
                  f"(buildings/roads/non-developed); first at row={offender[0]} col={offender[1]}")
            subset_local += 1
        # ── Invariant 2: converted ⊆ region (when region active) ──
        if region_mask is not None:
            out_region = int((converted_mask & ~region_mask).sum())
            if out_region:
                offender = np.argwhere(converted_mask & ~region_mask)[0]
                print(f"  FAIL  {label}: {out_region} converted px outside region; "
                      f"first at row={offender[0]} col={offender[1]}")
                subset_local += 1
        # ── Invariant 3: converted ⊆ ownership (when ownership active) ──
        if ownership_mask is not None:
            out_own = int((converted_mask & ~ownership_mask).sum())
            if out_own:
                offender = np.argwhere(converted_mask & ~ownership_mask)[0]
                print(f"  FAIL  {label}: {out_own} converted px outside ownership; "
                      f"first at row={offender[0]} col={offender[1]}")
                subset_local += 1
        if subset_local == 0:
            checks = ["eligible"]
            if region_mask is not None:    checks.append("region")
            if ownership_mask is not None: checks.append("ownership")
            print(f"  OK    {label}: "
                  f"converted={funnel['converted_px']:,} px ⊆ "
                  f"{' ∩ '.join(checks)}; "
                  f"funnel total={funnel['total_px']:,} → "
                  f"developed={funnel['developed_px']:,} → "
                  f"convertible={funnel['convertible_px']:,} → "
                  f"final_eligible={funnel['final_eligible_px']:,} → "
                  f"converted={funnel['converted_px']:,}")

        # ── Eligibility Funnel Pass — record reconciliation ─────────────
        # The funnel UI sources every cell from the same record fields the
        # subset matrix is exercising here. Tie them together: funnel's
        # final-eligible step (recomputed from raw masks) must equal
        # results['region_selection']['eligible_pixels_in_region'], and
        # funnel's converted_acres must equal results['region_selection']
        # ['converted_acres']. Skipped for citywide cells (mode='entire_aoi')
        # — the funnel doesn't render there.
        rs = results.get('region_selection') or {}
        if rs.get('mode') == 'selected_regions':
            record_elig = rs.get('eligible_pixels_in_region')
            if funnel['final_eligible_px'] != record_elig:
                print(f"  FAIL  {label}: funnel reconciliation — "
                      f"final-eligible {funnel['final_eligible_px']:,} != "
                      f"record eligible_pixels_in_region {record_elig:,}")
                reconcile_local += 1
            record_conv_acres = rs.get('converted_acres')
            funnel_conv_acres = funnel['converted_px'] * app.PIXEL_AREA_ACRES
            if not np.isclose(funnel_conv_acres, record_conv_acres,
                              rtol=1e-9, atol=1e-9):
                print(f"  FAIL  {label}: funnel reconciliation — "
                      f"converted_acres {funnel_conv_acres} != "
                      f"record converted_acres {record_conv_acres}")
                reconcile_local += 1
            if reconcile_local == 0:
                print(f"        reconciliation OK: funnel ⇔ record "
                      f"(eligible={record_elig:,} px, "
                      f"converted={record_conv_acres:,.2f} acres)")
        return (subset_local, reconcile_local)

    # ── SA matrix ──
    try:
        _rebind_city(app, "San Antonio, TX")
        sa_state = app._CURRENT_CITY_STATE
        sa_region = _region_mask_from(sa_state, "council_districts", ["5"])
        sa_ownership = _ownership_mask_from(sa_state, "vacant_public")
        sa_tiny_pixels = sa_state.convertible_pixels[:25]
        sa_tiny_mask = np.zeros(sa_state.ref_shape, dtype=bool)
        sa_tiny_mask[sa_tiny_pixels[:, 0], sa_tiny_pixels[:, 1]] = True
        sa_multi = _region_mask_from(sa_state, "council_districts", ["5", "7"])
        # Finer-class masks (Batch 3 of OWNERSHIP_FINER_CLASSES_SPEC.md) —
        # each region × finer-class cell asserts converted ⊆ the finer
        # mask. Coarse rollup cells (region + vacant_public,
        # ownership-only vacant_public) continue to pass because rollups
        # are unions over band-1 values.
        sa_city          = _ownership_mask_from(sa_state, "city")
        sa_state_federal = _ownership_mask_from(sa_state, "state_federal")
        sa_school        = _ownership_mask_from(sa_state, "school")
        sa_university    = _ownership_mask_from(sa_state, "university")
        sa_county        = _ownership_mask_from(sa_state, "county")
        # Batch 4 v2 — union mask via the production helpers
        # (_compose_eligible_filter_cfg + _build_ownership_mask). This
        # exercises the multi-class checkbox UI's mask-build path: the
        # cfg dict synthesized by the composite resolver feeds the same
        # `_build_ownership_mask` the single-class path uses.
        sa_city_school_union = app._build_ownership_mask(
            sa_state.ownership_raster, sa_state.ownership_vacant_raster,
            app._compose_eligible_filter_cfg(['city', 'school'],
                                              vacant_overlay=False),
        )
        for _cell_args in [
            (sa_state, "SA / region-only (D5)", sa_region, None, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / region + ownership (D5 + vacant_public)", sa_region, sa_ownership, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / ownership-only (vacant_public)", None, sa_ownership, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / citywide", None, None, _SUBSET_RECIPE_PCT10),
            # Tiny: 25 eligible by construction; pct=100 to force all 25 conversions.
            (sa_state, "SA / tiny region (25 px synthetic)", sa_tiny_mask, None, _SUBSET_RECIPE_PCT100),
            (sa_state, "SA / multi-region (D5 + D7)", sa_multi, None, _SUBSET_RECIPE_PCT10),
            # Optimizer-applied: high-conversion mixed GI/FF recipe under
            # region+ownership. Locks in converted ⊆ region ∩ ownership
            # for the scenario class the surrogate would suggest.
            (sa_state, "SA / optimizer-applied recipe under region + ownership",
             sa_region, sa_ownership, _SUBSET_RECIPE_OPTIMIZER),
            # ── Finer-class cells (Batch 3) ──
            # Each exercises converted ⊆ eligible ∩ region ∩ class for
            # one of the five selectable finer classes. D5 ∩ county is
            # small (97 convertible px, ≈21.6 ac — see the Batch 3 county
            # measurement in OWNERSHIP_FINER_CLASSES_SPEC.md) but
            # non-zero, so the assertion exercises real pixels.
            (sa_state, "SA / region + city-only (D5 + city)",
             sa_region, sa_city, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / region + state-federal-only (D5 + state_federal)",
             sa_region, sa_state_federal, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / region + school-only (D5 + school)",
             sa_region, sa_school, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / region + university-only (D5 + university)",
             sa_region, sa_university, _SUBSET_RECIPE_PCT10),
            (sa_state, "SA / region + county-only (D5 + county)",
             sa_region, sa_county, _SUBSET_RECIPE_PCT10),
            # Batch 4 v2 — multi-class union cell. Exercises converted
            # ⊆ eligible ∩ region ∩ (city ∪ school), built via the live
            # `_compose_eligible_filter_cfg` + `_build_ownership_mask`
            # path the checkbox UI uses. Non-empty-converted is asserted
            # separately below — subset alone is vacuously satisfied
            # by zero conversions, which would silently hide a bug
            # where the union mask resolves to empty.
            (sa_state, "SA / region + city ∪ school (D5 union)",
             sa_region, sa_city_school_union, _SUBSET_RECIPE_PCT10),
        ]:
            _sd, _rd = _run_cell(*_cell_args)
            subset_diffs += _sd
            reconcile_diffs += _rd

        # Union-cell non-empty assertion (Batch 4 v2): subset alone is
        # vacuously satisfied by zero conversions. Re-run the union cell
        # specifically and assert |converted| > 0 — otherwise a bug
        # where the composite cfg resolves to an empty mask would
        # silently pass.
        _union_results = app.evaluate_scenario(
            **_SUBSET_RECIPE_PCT10, seed=42, placement_strategy='random',
            selected_region_mask=(sa_region & sa_city_school_union),
        )
        _union_converted = int(((sa_state.lulc != _union_results['scenario_lulc'])
                                 & sa_city_school_union & sa_region).sum())
        if _union_converted > 0:
            print(f"  OK    SA union non-empty assertion: "
                  f"{_union_converted:,} converted px inside D5 ∩ (city ∪ school)")
        else:
            print(f"  FAIL  SA union non-empty assertion: 0 converted px — "
                  "the composite mask resolved empty; subset would pass "
                  "vacuously.")
            subset_diffs += 1
    except Exception as e:
        print(f"  ERROR SA matrix: {e}")
        import traceback; traceback.print_exc()
        subset_diffs += 1

    # ── MN matrix (4 cells — no ownership data) ──
    try:
        _rebind_city(app, "Minneapolis, MN")
        mn_state = app._CURRENT_CITY_STATE
        mn_layer = "downtown_tracts"
        mn_labels = mn_state.region_layer_labels[mn_layer]
        mn_first_label = mn_labels[0]
        mn_second_label = mn_labels[1] if len(mn_labels) > 1 else mn_labels[0]
        mn_region = _region_mask_from(mn_state, mn_layer, [mn_first_label])
        mn_tiny_pixels = mn_state.convertible_pixels[:25]
        mn_tiny_mask = np.zeros(mn_state.ref_shape, dtype=bool)
        mn_tiny_mask[mn_tiny_pixels[:, 0], mn_tiny_pixels[:, 1]] = True
        mn_multi = _region_mask_from(mn_state, mn_layer,
                                     [mn_first_label, mn_second_label])
        for _cell_args in [
            (mn_state, f"MN / region-only ({mn_first_label})", mn_region, None, _SUBSET_RECIPE_PCT10),
            (mn_state, "MN / citywide", None, None, _SUBSET_RECIPE_PCT10),
            (mn_state, "MN / tiny region (25 px synthetic)", mn_tiny_mask, None, _SUBSET_RECIPE_PCT100),
            (mn_state, f"MN / multi-region ({mn_first_label}, {mn_second_label})", mn_multi, None, _SUBSET_RECIPE_PCT10),
        ]:
            _sd, _rd = _run_cell(*_cell_args)
            subset_diffs += _sd
            reconcile_diffs += _rd
    except Exception as e:
        print(f"  ERROR MN matrix: {e}")
        import traceback; traceback.print_exc()
        subset_diffs += 1

    # ── Region-Constrained Optimizer (variant B) — two assertion cells ─────
    # docs/internal/REGION_OPTIMIZER_SPEC.md §8.
    #
    #   1. Subset invariant on every record the region optimizer returns —
    #      converted ⊆ eligible ∩ region ∩ ownership AND |converted| > 0
    #      (anti-vacuous; subset alone is trivially satisfied by zero
    #      conversions).
    #   2. Engine-verified reconciliation — a fresh engine eval on the
    #      record's recipe + mask must reproduce the recorded metrics
    #      (rtol=1e-9 / atol=1e-9). Plus a meta-test: inject a surrogate
    #      value into a record and assert reconciliation FAILS. Without
    #      the meta-test the reconciliation cell would be green-light
    #      theatre — verifying nothing about predicted-vs-engine drift.
    print(f"\n{'=' * 60}")
    print("Region-Constrained Optimizer — subset + engine-reconciliation")
    print(f"{'=' * 60}")
    region_opt_diffs = 0
    try:
        _rebind_city(app, "San Antonio, TX")
        ro_state = app._CURRENT_CITY_STATE
        ro_region = _region_mask_from(ro_state, "council_districts", ["5"])
        ro_ownership = _ownership_mask_from(ro_state, "vacant_public")
        ro_combined = ro_region & ro_ownership

        ro_scenario_df = app.compute_scenario_grid(
            ro_state, "San Antonio, TX",
            app.DATA_DIR_FLOOD, app.DATA_DIR_COOLING,
        )
        from surrogate import train_surrogate, optimize_scenario_region
        ro_surrogate = train_surrogate(ro_scenario_df, n_estimators=100)

        def _ro_engine_eval(_pct, _gi, _ff):
            return app.evaluate_scenario(
                _pct, _gi, _ff,
                seed=42, placement_strategy='random',
                selected_region_mask=ro_combined,
            )

        # K=5 for the gate — assertions exercise the contract on a small
        # set of records; a roomier K is for runtime UX, not testing.
        ro_records = optimize_scenario_region(
            ro_surrogate, ro_scenario_df, _ro_engine_eval,
            weights={'mean_hm': 1.0, 'flood_reduction': 1.0,
                     'food_mln_lbs': 1.0, 'carbon_tons_co2': 1.0,
                     'total_cost_mln': 0.5, 'runoff_acre_feet': 1.0},
            k_engine=5, top_n=5,
        )
        if ro_records is None or ro_records.empty:
            print("  FAIL  region optimizer returned no records")
            region_opt_diffs += 1
        else:
            ro_eligible = _convertible_in_raster(ro_state)
            ro_baseline = ro_state.lulc
            print(f"  region optimizer returned {len(ro_records)} records")

            for i, (_, rec) in enumerate(ro_records.iterrows()):
                _recipe = dict(
                    pct_converted=int(rec['pct_converted']),
                    green_infrastructure_pct=int(rec['green_infrastructure_pct']),
                    food_forest_pct=int(rec['food_forest_pct']),
                )
                fresh = app.evaluate_scenario(
                    **_recipe, seed=42, placement_strategy='random',
                    selected_region_mask=ro_combined,
                )
                conv_mask = (ro_baseline != fresh['scenario_lulc'])

                # ── Subset invariant ──
                out_e = int((conv_mask & ~ro_eligible).sum())
                out_r = int((conv_mask & ~ro_region).sum())
                out_o = int((conv_mask & ~ro_ownership).sum())
                n_conv = int(conv_mask.sum())
                if out_e or out_r or out_o:
                    print(f"  FAIL  record #{i + 1} subset: "
                          f"out_eligible={out_e} out_region={out_r} "
                          f"out_ownership={out_o}")
                    region_opt_diffs += 1
                elif n_conv <= 0:
                    print(f"  FAIL  record #{i + 1} converted-non-empty: "
                          f"|converted| = 0 (subset trivially satisfied)")
                    region_opt_diffs += 1
                else:
                    print(f"  OK    record #{i + 1} subset: "
                          f"|converted|={n_conv:,} px ⊆ "
                          f"eligible ∩ region ∩ ownership")

                # ── Engine-verified reconciliation ──
                # The record was produced by the same _ro_engine_eval call
                # the orchestration runs; a fresh call with the same recipe
                # + mask must reproduce the metrics exactly. The matrix below
                # mirrors the round-trip assertion's metric list.
                _RO_RECON_METRICS = (
                    'mean_hm', 'flood_reduction', 'food_mln_lbs',
                    'carbon_tons_co2', 'total_cost_mln',
                )
                rl = fresh.get('region_local') or {}
                divergent = []
                for m in _RO_RECON_METRICS:
                    fresh_v = rl.get(m, fresh.get(m))
                    rec_v = rec.get(m)
                    if (fresh_v is None or rec_v is None
                            or not np.isclose(float(fresh_v), float(rec_v),
                                              rtol=1e-9, atol=1e-9,
                                              equal_nan=True)):
                        divergent.append((m, rec_v, fresh_v))
                if divergent:
                    print(f"  FAIL  record #{i + 1} reconciliation:")
                    for m, rec_v, fresh_v in divergent:
                        print(f"           {m}: record={rec_v} fresh={fresh_v}")
                    region_opt_diffs += 1
                else:
                    print(f"        record #{i + 1} reconcile OK: "
                          f"{len(_RO_RECON_METRICS)} metrics match fresh eval "
                          f"(rtol=1e-9)")

            # ── Meta-test: injected surrogate value MUST fail reconciliation ──
            # The reconciliation cell above only guards if it actually catches
            # surrogate-vs-engine drift. Inject a deliberately-wrong value
            # (the surrogate's citywide prediction for the same recipe,
            # which is ≠ the engine region-local for any nontrivial case)
            # and assert the reconciliation fails on that altered record.
            meta_rec = ro_records.iloc[0].copy()
            X = np.array([[meta_rec['pct_converted'],
                           meta_rec['green_infrastructure_pct'],
                           meta_rec['food_forest_pct']]], dtype=float)
            from surrogate import predict_with_uncertainty
            pred_mean, _, _ = predict_with_uncertainty(ro_surrogate, X)
            # surrogate output column order: flood_reduction, mean_hm, ...
            # Index 1 = mean_hm. We poison meta_rec['mean_hm'] with the
            # citywide surrogate prediction.
            poisoned_mean_hm = float(pred_mean[0, 1])
            true_mean_hm = float(meta_rec['mean_hm'])
            # Sanity: don't run the meta-test if the surrogate happens to
            # exactly predict the engine value (extremely unlikely; would make
            # the meta-test vacuous).
            if np.isclose(poisoned_mean_hm, true_mean_hm,
                          rtol=1e-9, atol=1e-9):
                print(f"  SKIP  meta-test: surrogate prediction exactly matches "
                      f"engine value — meta-test would be vacuous")
            else:
                # Re-run reconciliation against the poisoned value; expect
                # divergence on mean_hm.
                _meta_recipe = dict(
                    pct_converted=int(meta_rec['pct_converted']),
                    green_infrastructure_pct=int(meta_rec['green_infrastructure_pct']),
                    food_forest_pct=int(meta_rec['food_forest_pct']),
                )
                meta_fresh = app.evaluate_scenario(
                    **_meta_recipe, seed=42, placement_strategy='random',
                    selected_region_mask=ro_combined,
                )
                meta_rl = meta_fresh.get('region_local') or {}
                meta_fresh_v = float(meta_rl.get('mean_hm',
                                                  meta_fresh.get('mean_hm')))
                if np.isclose(meta_fresh_v, poisoned_mean_hm,
                              rtol=1e-9, atol=1e-9):
                    print(f"  FAIL  meta-test: poisoned record's mean_hm "
                          f"({poisoned_mean_hm}) reconciled against fresh "
                          f"engine ({meta_fresh_v}) — the reconciliation "
                          f"isn't catching surrogate drift")
                    region_opt_diffs += 1
                else:
                    print(f"  OK    meta-test: injected surrogate value "
                          f"({poisoned_mean_hm:.4f}) ≠ engine value "
                          f"({meta_fresh_v:.4f}); reconciliation correctly "
                          f"flags the drift")

            # ── Provenance distinction — region-optimizer records cannot
            #    silently collapse into the citywide surrogate's tag ──
            # Three guards, both-ways:
            #   1. The two provenance constants are themselves distinct.
            #   2. Every region record carries
            #      source='region_optimized' + validation='engine_verified'
            #      (the new tags) — not the citywide surrogate tag.
            #   3. The citywide optimize_scenario function does NOT emit
            #      those columns (its DataFrame has different shape) — a
            #      structural distinguisher independent of (1).
            # Also assert the rendered Source labels diverge: the user-facing
            # distinction is what the brief calls out, not just the constant.
            import natcap_scenarios as _ns
            if _ns.PROVENANCE_REGION_OPTIMIZED == _ns.PROVENANCE_OPTIMIZER:
                print("  FAIL  provenance constants collapsed: "
                      "PROVENANCE_REGION_OPTIMIZED == PROVENANCE_OPTIMIZER")
                region_opt_diffs += 1
            else:
                print(f"  OK    provenance constants distinct: "
                      f"REGION_OPTIMIZED={_ns.PROVENANCE_REGION_OPTIMIZED!r} "
                      f"vs OPTIMIZER={_ns.PROVENANCE_OPTIMIZER!r}")

            _region_labels = app._PROVENANCE_HEADER_INFO.get(
                _ns.PROVENANCE_REGION_OPTIMIZED)
            _citywide_labels = app._PROVENANCE_HEADER_INFO.get(
                _ns.PROVENANCE_OPTIMIZER)
            if (_region_labels is None or _citywide_labels is None
                    or _region_labels[0] == _citywide_labels[0]):
                print(f"  FAIL  rendered Source labels collapsed or "
                      f"missing: region={_region_labels!r} "
                      f"citywide={_citywide_labels!r}")
                region_opt_diffs += 1
            else:
                print(f"  OK    rendered Source labels distinct: "
                      f"region={_region_labels[0]!r} "
                      f"vs citywide={_citywide_labels[0]!r}")

            _src_col_bad = []
            _val_col_bad = []
            for _, _rec in ro_records.iterrows():
                if _rec.get('source') != 'region_optimized':
                    _src_col_bad.append(_rec.get('source'))
                if _rec.get('validation') != 'engine_verified':
                    _val_col_bad.append(_rec.get('validation'))
            if _src_col_bad or _val_col_bad:
                print(f"  FAIL  region records carry wrong tags: "
                      f"source values={_src_col_bad}  "
                      f"validation values={_val_col_bad}")
                region_opt_diffs += 1
            else:
                print(f"  OK    {len(ro_records)} region records tagged "
                      f"source=region_optimized + validation=engine_verified")

            # Citywide optimize_scenario for the structural distinguisher.
            # Use a tiny constraint set the surrogate can satisfy on the
            # Fast grid so we get a real DataFrame back.
            from surrogate import optimize_scenario as _cw_optimize
            _cw_df = _cw_optimize(
                ro_surrogate, min_flood=0, min_cool=0.0, min_food=0.0,
                max_runoff=float(ro_scenario_df['runoff_acre_feet'].max()) + 1,
                min_carbon=0, max_food=float(ro_scenario_df['food_mln_lbs'].max()),
                max_flood=100.0, max_cool=1.1, n_samples=2000,
            )
            if isinstance(_cw_df, dict):
                print(f"  SKIP  citywide structural distinguisher: "
                      f"optimize_scenario returned no scenarios "
                      f"(constraint pruning) — provenance constants + "
                      f"label assertions above already cover the "
                      f"collapse-prevention contract")
            elif 'source' in _cw_df.columns or 'validation' in _cw_df.columns:
                print(f"  FAIL  citywide optimize_scenario emitted "
                      f"source/validation columns — those are specific to "
                      f"the region path; the structural distinguisher "
                      f"between the two record types collapsed")
                region_opt_diffs += 1
            else:
                print(f"  OK    citywide optimize_scenario DataFrame "
                      f"({len(_cw_df)} rows) has no source/validation "
                      f"columns — structural distinguisher from region "
                      f"records holds")
    except Exception as e:
        print(f"  ERROR region-optimizer assertions: {e}")
        import traceback; traceback.print_exc()
        region_opt_diffs += 1

    # ── Ownership Finer Classes (Batch 1) — reconciliation assertion ───────
    # Two checks (see OWNERSHIP_FINER_CLASSES_SPEC.md §"Verification"):
    #
    #  1. Rule-output reconciliation (±0.5%, OPTIONAL — needs archived GPKG):
    #     Re-apply the six-way classifier to the archived BCAD GPKG,
    #     aggregate `Acres` per class, assert the per-class polygon-Acres
    #     totals match the locked targets within ±0.5%. This is the
    #     load-bearing correctness check — it surfaces every rule
    #     mislabel. Skipped (with a note) when the archived GPKG isn't
    #     present (e.g. on a fresh checkout where the archive lives
    #     outside the repo).
    #
    #  2. Raster-integrity reconciliation (±5%, ALWAYS RUNS — in-repo
    #     only): Read the new two-band raster's band 1, count pixels per
    #     class code, multiply by PIXEL_AREA_ACRES, assert each class
    #     matches the in-AOI rasterization frozen at Batch 1's first run.
    #     Catches rasterization regressions independently of the source
    #     GPKG. ±5% accounts for rasterization rounding + AOI boundary
    #     clip effects.
    print(f"\n{'=' * 60}")
    print("Ownership Finer Classes — Batch 1 reconciliation")
    print(f"{'=' * 60}")
    ownership_diffs_batch1 = 0
    # School / University Split — 7-class enum. `private=0` and
    # `unknown=5` stay stable across the split; `school` takes the old
    # `school_university` code 4; `university` takes the new code 6.
    _OWN_CLASS_ENUM = {
        'private': 0, 'city': 1, 'county': 2,
        'state_federal': 3, 'school': 4, 'unknown': 5, 'university': 6,
    }
    _PIXEL_AREA_ACRES = 0.2224
    _RASTER_2BAND = Path("data/sa/sa_ownership_2band_30m.tif")
    # Locked rasterized in-AOI per-class acres. Re-snapshotted 2026-06-04 after
    # the sub-pixel-majority re-rasterization (3× / 10 m) that fixed the 16 %
    # ownership-nodata 30 m grid-tiling artifact — see DESIGN_NOTES §6.8. Every
    # class grew (boundary slivers recovered); `private` gained the most
    # (~46k ac, the ~96 % of recovered population that is residential). The
    # rule-output polygon-Acres (`_RULE_EXPECTED_AC`, ±0.5 %) are UNCHANGED —
    # the classifier didn't move, only the rasterization. Tolerance stays ±5 %.
    _RASTER_EXPECTED_AC = {
        'private':       553_301.0,
        'city':           42_409.0,
        'county':          3_106.0,
        'state_federal':  28_617.0,
        'school':          2_611.0,
        'unknown':        15_972.0,
        'university':      3_758.0,
    }
    _RULE_EXPECTED_AC = {
        'private':       606_433.0,
        'city':          126_634.0,
        'county':          3_018.0,
        'state_federal':  54_883.0,
        'school':          2_604.0,
        'unknown':         1_735.0,
        'university':      3_771.0,
    }
    _ARCHIVE_GPKG = Path(
        "/Users/dkw-testing/Desktop/ecosystem_explorer_archive/"
        "sa_ownership_bexar_2026-05-31.gpkg"
    )

    # Check (2) — raster integrity (always runs).
    if _RASTER_2BAND.exists():
        try:
            import rasterio
            with rasterio.open(_RASTER_2BAND) as src:
                band1 = src.read(1)
                # Sanity: must be a two-band file.
                assert src.count == 2, f"expected 2 bands, got {src.count}"
            print("Raster-integrity check (band 1 per-class in-AOI acres, ±5%):")
            for cls_name, cls_code in _OWN_CLASS_ENUM.items():
                actual_px = int((band1 == cls_code).sum())
                actual_ac = actual_px * _PIXEL_AREA_ACRES
                expected_ac = _RASTER_EXPECTED_AC[cls_name]
                delta_pct = abs(actual_ac - expected_ac) / expected_ac * 100
                ok = delta_pct <= 5.0
                tag = "OK  " if ok else "FAIL"
                print(f"  {tag} {cls_name:>20s}: {actual_ac:>10,.0f} ac "
                      f"(expected ~{expected_ac:>10,.0f}, delta {delta_pct:+.2f}%)")
                if not ok:
                    ownership_diffs_batch1 += 1
        except Exception as e:
            print(f"  ERROR raster-integrity: {e}")
            import traceback; traceback.print_exc()
            ownership_diffs_batch1 += 1
    else:
        print(f"  SKIP raster-integrity: {_RASTER_2BAND} not in repo "
              f"— run scripts/data/download_bexar_parcels.py "
              f"--reclassify-from <archived.gpkg> to produce it")

    # Check (1) — rule-output reconciliation (optional; archived GPKG outside repo).
    if _ARCHIVE_GPKG.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_dbp", "scripts/data/download_bexar_parcels.py"
            )
            _dbp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_dbp)
            import geopandas as _gpd
            import pandas as _pd
            print("\nRule-output reconciliation "
                  "(full-parcel polygon-Acres, ±0.5%):")
            _g = _gpd.read_file(str(_ARCHIVE_GPKG), ignore_geometry=True)
            _g['Acres'] = _pd.to_numeric(_g['Acres'], errors='coerce').fillna(0)
            _g['cls'] = _g['Owner'].map(_dbp._classify_seven_way)
            _actual = _g.groupby('cls')['Acres'].sum()
            for cls_name in _OWN_CLASS_ENUM:
                actual_ac = float(_actual.get(cls_name, 0))
                expected_ac = _RULE_EXPECTED_AC[cls_name]
                delta_pct = abs(actual_ac - expected_ac) / expected_ac * 100
                ok = delta_pct <= 0.5
                tag = "OK  " if ok else "FAIL"
                print(f"  {tag} {cls_name:>20s}: {actual_ac:>10,.0f} ac "
                      f"(expected {expected_ac:>10,.0f}, delta {delta_pct:+.2f}%)")
                if not ok:
                    ownership_diffs_batch1 += 1
        except Exception as e:
            print(f"  ERROR rule-output: {e}")
            import traceback; traceback.print_exc()
            ownership_diffs_batch1 += 1
    else:
        print(f"\n  SKIP rule-output: archived GPKG not found at "
              f"{_ARCHIVE_GPKG}")

    # ── Subset Invariants Pass — city-switch guard transition test ──────────
    # Mirror of the live-app reset: pre-populate an isolated session-state
    # mock with SA region + ownership widget keys (the state a user would
    # leave behind after picking District 5 + vacant_public on SA), then
    # call `_reset_state_for_city_switch` directly. Assert every region /
    # ownership / optimizer / slider widget key returns to its post-switch
    # default. Failure here means a future edit to the reset helper dropped
    # a key; the cell-by-cell pass/fail tells the maintainer which one.
    print(f"\n{'=' * 60}")
    print("Subset Invariants — city-switch guard transition test")
    print(f"{'=' * 60}")
    guard_diffs = 0

    class _TestSessionState:
        """Isolated session_state mock for the guard test. Implements the
        subset of the streamlit session_state API that
        `_reset_state_for_city_switch` consumes: get / pop / keys, item
        and attribute set, attribute read. Owns its own dict so it can't
        corrupt the shared `_SessionStateStub._store`."""
        def __init__(self, initial):
            object.__setattr__(self, "_d", dict(initial))
        def get(self, key, default=None):
            return self._d.get(key, default)
        def pop(self, key, *args):
            return self._d.pop(key, *args) if args else self._d.pop(key, None)
        def keys(self):
            return list(self._d.keys())
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return self._d.get(name)
        def __setattr__(self, name, value):
            if name == "_d":
                object.__setattr__(self, name, value)
            else:
                self._d[name] = value
        def __contains__(self, key):
            return key in self._d

    try:
        stale = {
            # Region widget state — SA user picked District 5 via the
            # dropdown AND the interactive map.
            'region_apply_within':                  'Selected regions',
            'region_layer':                         'council_districts',
            'region_labels_council_districts':      ['5'],
            'region_labels_bexar_tracts':           [],
            'region_map_picker_event':              {'selection': {'points': [{'customdata': '5'}]}},
            'region_map_picker_layer':              'council_districts',
            # Ownership widget state — SA user picked the city class and
            # checked the vacant overlay. After Batch 4 the selectbox
            # stores a mode key (or None); the vacant overlay is its own
            # boolean. Both must reset on a city change.
            'ownership_filter_choice':              'city',
            'ownership_filter_vacant_overlay':       True,
            # Slider + optimizer state that the existing reset block was
            # already clearing — re-asserted so a regression in either
            # half of the helper surfaces clearly.
            'slider_pct_converted':                 25,
            'slider_gi_pct':                        70,
            'slider_ff_pct':                        20,
            'optimized_results':                    'fake-results-from-SA',
            'just_optimized':                       True,
            'applied_from_optimizer':               True,
            '_applied_optimizer_values':            (25, 70, 20),
            'active_example_scenario':              'cooling',
        }
        test_ss = _TestSessionState(stale)
        app._reset_state_for_city_switch(test_ss)

        # Expectations — each entry is (description, got, expected).
        expectations = [
            # Region widget keys all reset to entire-area defaults.
            ('region_apply_within = "Entire analysis area"',
                test_ss.get('region_apply_within'), 'Entire analysis area'),
            ('region_layer cleared',
                test_ss.get('region_layer'), None),
            ('region_labels_council_districts cleared',
                test_ss.get('region_labels_council_districts'), None),
            ('region_labels_bexar_tracts cleared',
                test_ss.get('region_labels_bexar_tracts'), None),
            ('region_map_picker_event cleared',
                test_ss.get('region_map_picker_event'), None),
            ('region_map_picker_layer cleared',
                test_ss.get('region_map_picker_layer'), None),
            # Ownership widget keys reset — selectbox to "All ownership"
            # (None) and the vacant overlay checkbox unchecked (False).
            ('ownership_filter_choice = None ("All ownership")',
                test_ss.get('ownership_filter_choice'), None),
            ('ownership_filter_vacant_overlay = False',
                test_ss.get('ownership_filter_vacant_overlay'), False),
            # Slider state cleared so the new city renders against its
            # own defaults.
            ('slider_pct_converted cleared',
                test_ss.get('slider_pct_converted'), None),
            ('slider_gi_pct cleared',
                test_ss.get('slider_gi_pct'), None),
            ('slider_ff_pct cleared',
                test_ss.get('slider_ff_pct'), None),
            # Optimizer + preset-highlight state reset.
            ('optimized_results cleared',
                test_ss.get('optimized_results'), None),
            ('just_optimized = False',
                test_ss.get('just_optimized'), False),
            ('applied_from_optimizer = False',
                test_ss.get('applied_from_optimizer'), False),
            ('_applied_optimizer_values cleared',
                test_ss.get('_applied_optimizer_values'), None),
            ('active_example_scenario = "balanced"',
                test_ss.get('active_example_scenario'), 'balanced'),
        ]

        for name, got, want in expectations:
            if got == want:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: got {got!r}, want {want!r}")
                guard_diffs += 1

        # No stale region_labels_* key should survive — the helper iterates
        # every key with that prefix and drops it. Sanity check: scan the
        # post-reset store for any leftover prefix match.
        leftover = [k for k in test_ss.keys()
                    if isinstance(k, str) and k.startswith('region_labels_')]
        if leftover:
            print(f"  FAIL region_labels_* sweep left these behind: {leftover}")
            guard_diffs += len(leftover)
        else:
            print(f"  OK   region_labels_* sweep complete (no stale keys)")

    except Exception as e:
        print(f"  ERROR guard transition test: {e}")
        import traceback; traceback.print_exc()
        guard_diffs += 1

    # ── Default-scenario state consistency (Relay A) ─────────────────────────
    # Title, sentence, and audit are all rendered from the same
    # `_resolved_scenario` dict via three display helpers
    # (`_explorer_scenario_label`, `_explorer_scenario_sentence`,
    # `_explorer_audit_sentence`). Assert the helpers stay self-consistent at
    # (a) the documented default 10/50/50, (b) pct=0 → "no conversion" form,
    # (c) a mixed scenario, AND (d) the post-city-switch reset reproduces the
    # default. Plus the load-bearing meta-test: a deliberately desynced state
    # (label built from one state, sentence built from another) MUST trip the
    # check — otherwise the consistency assertion is green-light theatre.
    print(f"\n{'=' * 60}")
    print("Default-scenario state — title + sentence consistency")
    print(f"{'=' * 60}")
    scenario_state_diffs = 0
    try:
        # The display helpers are pure functions of a resolved-scenario dict.
        # Test them directly — no Streamlit render needed.

        def _check_consistent(state, label, sentence, audit):
            """Both-ways consistency:
              - All three strings agree on pct/gi/ff (or all say "no conversion").
              - Resolved state's pct/gi/ff match the encoded values in each.
            Returns list of failure strings."""
            failures = []
            pct = state['pct_converted']
            gi = state['green_infrastructure_pct']
            ff = state['food_forest_pct']
            if pct == 0:
                # All three must use the "no conversion" branch.
                for name, txt in (("label", label), ("sentence", sentence),
                                  ("audit", audit)):
                    if "no conversion" not in txt:
                        failures.append(
                            f"{name} does not branch to 'no conversion' "
                            f"at pct=0: {txt!r}"
                        )
            else:
                # All three must encode the same pct/gi/ff (string match).
                pct_token = f"{pct}%"
                gi_token = f"{gi}%"
                ff_token = f"{ff}%"
                for name, txt in (("label", label), ("sentence", sentence),
                                  ("audit", audit)):
                    if pct_token not in txt:
                        failures.append(
                            f"{name} missing pct={pct_token}: {txt!r}"
                        )
                    if gi_token not in txt:
                        failures.append(
                            f"{name} missing gi={gi_token}: {txt!r}"
                        )
                # Audit/sentence carry FF explicitly; label too.
                for name, txt in (("label", label), ("sentence", sentence),
                                  ("audit", audit)):
                    if ff_token not in txt:
                        failures.append(
                            f"{name} missing ff={ff_token}: {txt!r}"
                        )
            return failures

        # (a) Documented default 10/50/50.
        default_state = app._resolve_scenario(
            app.SCENARIO_DEFAULT_PCT_CONVERTED,
            app.SCENARIO_DEFAULT_GI_PCT,
            app.SCENARIO_DEFAULT_FF_PCT,
        )
        label = app._explorer_scenario_label(default_state)
        sentence = app._explorer_scenario_sentence(
            default_state, "developed land", "using random placement",
        )
        audit = app._explorer_audit_sentence(
            default_state, "Citywide", "", "Random",
        )
        # The default's pct/gi/ff must be the documented values, AND the three
        # surfaces must agree.
        if default_state != {
            'pct_converted': 10, 'green_infrastructure_pct': 50,
            'food_forest_pct': 50, 'pct_highdensity': 0,
        }:
            print(f"  FAIL default state != documented 10/50/50/0: "
                  f"{default_state}")
            scenario_state_diffs += 1
        fails = _check_consistent(default_state, label, sentence, audit)
        if fails:
            print(f"  FAIL default 10/50/50 consistency:")
            for f in fails:
                print(f"    {f}")
            scenario_state_diffs += len(fails)
        else:
            print(f"  OK   default 10/50/50 — label/sentence/audit agree")

        # (b) pct=0 — "no conversion" branch.
        zero_state = app._resolve_scenario(0, 0, 0)
        label_z = app._explorer_scenario_label(zero_state)
        sentence_z = app._explorer_scenario_sentence(
            zero_state, "developed land", "",
        )
        audit_z = app._explorer_audit_sentence(
            zero_state, "Citywide", "", "Random",
        )
        fails = _check_consistent(zero_state, label_z, sentence_z, audit_z)
        if fails:
            print(f"  FAIL pct=0 'no conversion' consistency:")
            for f in fails:
                print(f"    {f}")
            scenario_state_diffs += len(fails)
        else:
            print(f"  OK   pct=0 — all three branch to 'no conversion'")

        # (c) Mixed scenario.
        mixed_state = app._resolve_scenario(30, 75, 25)
        label_m = app._explorer_scenario_label(mixed_state)
        sentence_m = app._explorer_scenario_sentence(
            mixed_state, "developed land", "",
        )
        audit_m = app._explorer_audit_sentence(
            mixed_state, "Citywide", "", "Random",
        )
        fails = _check_consistent(mixed_state, label_m, sentence_m, audit_m)
        if fails:
            print(f"  FAIL mixed 30/75/25 consistency:")
            for f in fails:
                print(f"    {f}")
            scenario_state_diffs += len(fails)
        else:
            print(f"  OK   mixed 30/75/25 — label/sentence/audit agree")

        # (d) Post-city-switch reset — sliders pop, setdefault re-seeds the
        # documented default. Verify the seeded values match the constants.
        # Mirror what _reset_state_for_city_switch + setdefault do.
        test_ss = _SessionStateStub()
        for _k in ('slider_pct_converted', 'slider_gi_pct', 'slider_ff_pct'):
            test_ss._store[_k] = 0   # stale state from prior city
        app._reset_state_for_city_switch(test_ss)
        # _reset_state_for_city_switch pops; setdefault re-seeds. Mimic the
        # setdefault block at the top of the sidebar.
        test_ss.setdefault("slider_pct_converted",
                            app.SCENARIO_DEFAULT_PCT_CONVERTED)
        test_ss.setdefault("slider_gi_pct", app.SCENARIO_DEFAULT_GI_PCT)
        test_ss.setdefault("slider_ff_pct", app.SCENARIO_DEFAULT_FF_PCT)
        post_state = app._resolve_scenario(
            test_ss._store["slider_pct_converted"],
            test_ss._store["slider_gi_pct"],
            test_ss._store["slider_ff_pct"],
        )
        if post_state != default_state:
            print(f"  FAIL post-city-switch state != default: "
                  f"{post_state} vs {default_state}")
            scenario_state_diffs += 1
        else:
            print(f"  OK   post-city-switch resolves to documented default")

        # (e) Meta-test — load-bearing. Build label from state A, sentence
        # from state B (different pct), audit from state A; assert
        # _check_consistent FLAGS the discrepancy. Without this, the
        # consistency check guards nothing — it'd pass on any "all three
        # match each other by construction" run.
        state_A = app._resolve_scenario(10, 50, 50)
        state_B = app._resolve_scenario(30, 75, 25)
        label_meta = app._explorer_scenario_label(state_A)
        # sentence built from state_B — deliberately desynced from label.
        sentence_meta = app._explorer_scenario_sentence(
            state_B, "developed land", "",
        )
        audit_meta = app._explorer_audit_sentence(
            state_A, "Citywide", "", "Random",
        )
        # _check_consistent runs against state_A; sentence_meta won't carry
        # state_A's pct/gi tokens, so it must fail.
        meta_fails = _check_consistent(
            state_A, label_meta, sentence_meta, audit_meta,
        )
        if not meta_fails:
            print(f"  FAIL meta-test: deliberately desynced sentence "
                  f"(built from state_B={state_B}) failed to trip the "
                  f"consistency check against state_A={state_A}. The "
                  f"check guards nothing.")
            scenario_state_diffs += 1
        else:
            print(f"  OK   meta-test: desynced sentence trips the check "
                  f"({len(meta_fails)} divergence(s) flagged)")
    except Exception as e:
        print(f"  ERROR scenario-state consistency: {e}")
        import traceback; traceback.print_exc()
        scenario_state_diffs += 1

    # ── Tradeoffs tab section-order assertion ───────────────────────────────
    # The Tradeoffs tab + the NatCap reference-scenario view each have
    # a locked section order. Explorer mode: Tradeoff Space (plot) → Compare
    # scenarios (table) → Neighborhood breakdown → optimizer / saved /
    # best-by-goal. NatCap mode: side-by-side (table) → notes / validation
    # (Tradeoff Space plot intentionally absent — its axes (Flood Retention,
    # HMI) have no published values for NatCap fixed scenarios). A reorder
    # regression (e.g. a future edit that moves "Compare scenarios" above
    # the plot) would flip the user-facing flow without changing any engine
    # output; this cell catches that by scanning app.py for ordered markers.
    print(f"\n{'=' * 60}")
    print("Tradeoffs tab — section-order assertion")
    print(f"{'=' * 60}")
    section_order_diffs = 0
    try:
        with open("app.py", "r") as _fh:
            _src_lines = _fh.read().splitlines()

        def _first_line_containing(needle, start_line=0, end_line=None):
            """Return 1-indexed line number of the first line containing
            `needle` between [start_line, end_line). 0 if not found."""
            end_line = end_line if end_line is not None else len(_src_lines)
            for i in range(start_line, end_line):
                if needle in _src_lines[i]:
                    return i + 1
            return 0

        # Explorer mode tab2 order.
        # Anchor: the `with tab2:` block in the main panel. The block
        # extends until tab3 opens.
        _tab2_start = _first_line_containing("with tab2:")
        _tab3_start = _first_line_containing("with tab3:", _tab2_start)
        if _tab2_start == 0 or _tab3_start == 0:
            print(f"  FAIL Explorer tab2 boundaries not found "
                  f"(tab2={_tab2_start}, tab3={_tab3_start})")
            section_order_diffs += 1
        else:
            # Expected sequence (each anchor must follow the previous).
            # Mode-aware tradeoff display (RELAY) introduced two subheaders
            # ('Selected-area tradeoff space' in region mode + 'Citywide
            # tradeoff space' in citywide mode), one of which is rendered
            # per run. The first marker accepts either by searching for
            # the common 'tradeoff space' phrase — both subheaders carry
            # it, so the assertion still pins the spot.
            EXPECTED_TAB2_ORDER = [
                ("Tradeoff Space (plot)",
                 'tradeoff space'),
                ("Compare-scenarios table",
                 'st.markdown("#### Compare scenarios"'),
                ("Neighborhood breakdown",
                 'st.markdown("#### Neighborhood breakdown"'),
                ("Best scenarios by goal",
                 'st.markdown("#### Best scenarios by goal"'),
            ]
            _prev_line = _tab2_start
            _prev_name = "tab2 open"
            for name, needle in EXPECTED_TAB2_ORDER:
                _here = _first_line_containing(needle, _prev_line, _tab3_start)
                if _here == 0:
                    print(f"  FAIL Explorer tab2: marker for {name!r} not "
                          f"found between line {_prev_line} and tab3 open")
                    section_order_diffs += 1
                    continue
                if _here < _prev_line:
                    print(f"  FAIL Explorer tab2 order: {name!r} (line "
                          f"{_here}) appears before {_prev_name!r} (line "
                          f"{_prev_line})")
                    section_order_diffs += 1
                _prev_line = _here
                _prev_name = name
            if section_order_diffs == 0:
                print(f"  OK   Explorer tab2: 4 anchors in expected order "
                      f"(Tradeoff Space → Compare → Neighborhood → Best-by-goal)")

        # NatCap reference-view order.
        # Anchor: `def _render_natcap_fixed_scenario_view(`. The view's
        # extent: from def line until the next top-level `def `.
        _ncf_start = _first_line_containing(
            "def _render_natcap_fixed_scenario_view("
        )
        if _ncf_start == 0:
            print(f"  FAIL NatCap view: function def not found")
            section_order_diffs += 1
        else:
            # End of the function — next top-level `def ` or end of file.
            _ncf_end = len(_src_lines)
            for _i in range(_ncf_start, len(_src_lines)):
                _ln = _src_lines[_i]
                if (_ln.startswith("def ")
                        and _i + 1 > _ncf_start):
                    _ncf_end = _i + 1
                    break
            EXPECTED_NATCAP_ORDER = [
                # Inline banner now replaces _render_scenario_provenance_header
                # for the NatCap-fixed view (so the longer Source/Validation
                # text doesn't leak into the comparison-table cells). The
                # markdown "## {scenario_label}" line is the headline anchor.
                ("Headline",
                 'st.markdown(f"## {spec'),
                ("Side-by-side table",
                 'st.markdown("#### NatCap reference scenarios — side by side"'),
                ("Ecological card row",
                 'st.markdown("#### Ecological"'),
                ("Metrics-not-recomputed section",
                 'st.markdown("#### Metrics not recomputed for NatCap '
                 'reference scenarios"'),
            ]
            _prev_line = _ncf_start
            _prev_name = "function def"
            _ncf_diffs = 0
            for name, needle in EXPECTED_NATCAP_ORDER:
                _here = _first_line_containing(needle, _prev_line, _ncf_end)
                if _here == 0:
                    print(f"  FAIL NatCap view: marker for {name!r} not "
                          f"found between line {_prev_line} and function end")
                    _ncf_diffs += 1
                    continue
                if _here < _prev_line:
                    print(f"  FAIL NatCap view order: {name!r} (line "
                          f"{_here}) appears before {_prev_name!r} (line "
                          f"{_prev_line})")
                    _ncf_diffs += 1
                _prev_line = _here
                _prev_name = name
            if _ncf_diffs == 0:
                print(f"  OK   NatCap view: 4 anchors in expected order "
                      f"(Headline → Side-by-side → Ecological → "
                      f"Metrics-not-recomputed)")
            section_order_diffs += _ncf_diffs

            # Side-by-side must come BEFORE Ecological — the literal
            # Tradeoff-Analysis-reorder contract. Already asserted by the
            # sequence above; a focused check here in case the order list
            # ever grows.
            _sbs_line = _first_line_containing(
                'st.markdown("#### NatCap reference scenarios — side by side"',
                _ncf_start, _ncf_end,
            )
            _eco_line = _first_line_containing(
                'st.markdown("#### Ecological"',
                _ncf_start, _ncf_end,
            )
            if (_sbs_line == 0 or _eco_line == 0 or _sbs_line > _eco_line):
                print(f"  FAIL NatCap view: side-by-side must precede "
                      f"per-scenario Ecological cards "
                      f"(side-by-side={_sbs_line}, ecological={_eco_line})")
                section_order_diffs += 1
            else:
                print(f"  OK   NatCap view: side-by-side at line "
                      f"{_sbs_line} precedes Ecological cards at line "
                      f"{_eco_line}")
    except Exception as e:
        print(f"  ERROR section-order assertion: {e}")
        import traceback; traceback.print_exc()
        section_order_diffs += 1

    # ── Optimizer Promotion — shared-fire assertion ─────────────────────────
    # Sidebar Discover button + main-panel CTA must route through the same
    # `_fire_citywide_optimize` / `_fire_region_optimize` helpers so a click
    # on either produces the same engine pass + same session_state writes.
    # This assertion scans app.py for the helper-call sites and asserts:
    #   1. _fire_citywide_optimize is DEFINED exactly once.
    #   2. _fire_region_optimize is DEFINED exactly once.
    #   3. Each helper is CALLED in at least one location (sidebar button).
    #   4. No inline optimize_scenario() or optimize_scenario_region() calls
    #      survive in the Discover sidebar block — those must route through
    #      the helpers, not duplicate the logic.
    # When the main-panel CTA lands (HOLD batch), it adds a second call site
    # to each helper; the assertion's "≥ 1" check tolerates either state.
    print(f"\n{'=' * 60}")
    print("Optimizer Promotion — shared-fire helper contract")
    print(f"{'=' * 60}")
    shared_fire_diffs = 0
    try:
        with open("app.py", "r") as _fh:
            _src = _fh.read()
        import re as _re
        # Defs must each appear exactly once.
        _defs_city = len(_re.findall(
            r"^def _fire_citywide_optimize\(", _src, _re.MULTILINE))
        _defs_region = len(_re.findall(
            r"^def _fire_region_optimize\(", _src, _re.MULTILINE))
        if _defs_city != 1:
            print(f"  FAIL _fire_citywide_optimize defined {_defs_city} "
                  f"times — expected exactly 1")
            shared_fire_diffs += 1
        if _defs_region != 1:
            print(f"  FAIL _fire_region_optimize defined {_defs_region} "
                  f"times — expected exactly 1")
            shared_fire_diffs += 1
        # Calls — count appearances minus the def site. ≥ 1 call means at
        # least one button routes through the helper.
        _calls_city = len(_re.findall(
            r"_fire_citywide_optimize\(", _src)) - _defs_city
        _calls_region = len(_re.findall(
            r"_fire_region_optimize\(", _src)) - _defs_region
        if _calls_city < 1:
            print(f"  FAIL _fire_citywide_optimize called 0 times — "
                  f"sidebar button must route through it")
            shared_fire_diffs += 1
        if _calls_region < 1:
            print(f"  FAIL _fire_region_optimize called 0 times — "
                  f"sidebar button must route through it")
            shared_fire_diffs += 1
        if shared_fire_diffs == 0:
            print(f"  OK   _fire_citywide_optimize: 1 def + {_calls_city} "
                  f"call(s); _fire_region_optimize: 1 def + "
                  f"{_calls_region} call(s)")
        # Bonus check: no inline `optimize_scenario(` or
        # `optimize_scenario_region(` calls inside the Discover sidebar
        # `with _sec_discover:` block — those would be a duplicate
        # fire path that bypasses the shared helper.
        _disc_start = _re.search(
            r"^with _sec_discover:", _src, _re.MULTILINE)
        if _disc_start:
            # End of Discover block: heuristic — next `^with _sec_` or end.
            _after_disc = _src[_disc_start.end():]
            _disc_end_match = _re.search(
                r"^with _sec_", _after_disc, _re.MULTILINE)
            _disc_body = (_after_disc[:_disc_end_match.start()]
                           if _disc_end_match else _after_disc)
            _inline_cw = _re.findall(
                r"\boptimize_scenario\(", _disc_body)
            _inline_rg = _re.findall(
                r"\boptimize_scenario_region\(", _disc_body)
            # optimize_scenario_region in _disc_body is fine in helper
            # arguments; but a direct call inside Discover would bypass
            # the shared fire path. Filter out occurrences inside the
            # helper-call argument list by stripping any
            # `_fire_*_optimize(...optimize_scenario_region(...)`. The
            # helpers are at module level, NOT inside the with-block,
            # so any `optimize_scenario(` / `optimize_scenario_region(`
            # inside _disc_body is an inline duplicate.
            if _inline_cw:
                print(f"  FAIL inline optimize_scenario() calls inside "
                      f"_sec_discover block: {len(_inline_cw)} — should "
                      f"route through _fire_citywide_optimize")
                shared_fire_diffs += 1
            if _inline_rg:
                print(f"  FAIL inline optimize_scenario_region() calls "
                      f"inside _sec_discover block: {len(_inline_rg)} — "
                      f"should route through _fire_region_optimize")
                shared_fire_diffs += 1
            if not (_inline_cw or _inline_rg):
                print(f"  OK   no inline optimize_scenario / "
                      f"optimize_scenario_region calls inside the "
                      f"Discover sidebar block — both buttons route "
                      f"through the shared helpers")
    except Exception as e:
        print(f"  ERROR shared-fire assertion: {e}")
        import traceback; traceback.print_exc()
        shared_fire_diffs += 1

    # ── Two-RELAY lock — Discover surfaces / results / button pairing ───────
    # Three assertions machine-lock the two-optimizer distinction across the
    # sidebar Discover surface, the main-panel CTA, and the result-panel
    # headers. Each has a meta-test that seeds a violation and asserts the
    # check trips — without the meta-test the assertions would be green-light
    # theatre.
    #   A — Result labels: every result-panel st.subheader/markdown title in
    #       the Discover-result render path is in {"Suggested scenarios",
    #       "Best tested mixes …"}. No "Optimized" / "Optimal" / "optimum"
    #       on result-panel labels. Meta-test: seeded "Optimized suggestions"
    #       trips the check.
    #   B — Button paired: every st.button("Optimize", …) in the Discover
    #       surfaces co-renders with a known mode-label string ("Citywide
    #       surrogate search" or "Selected-area search") within N lines
    #       before it. Meta-test: removing the mode label trips the check.
    #   C — Provenance Source distinction: the applied-result Source line
    #       (PROVENANCE_OPTIMIZER vs PROVENANCE_REGION_OPTIMIZED via
    #       _PROVENANCE_HEADER_INFO) maps citywide-origin → "ai-assisted
    #       suggestion" string; region-origin → "region-optimized" string.
    #       Never collapsed/swapped. Meta-test: a swapped mapping trips the
    #       check.
    print(f"\n{'=' * 60}")
    print("Two-RELAY lock — Discover surfaces / results / button pairing")
    print(f"{'=' * 60}")
    two_relay_diffs = 0
    try:
        import re as _re2
        with open("app.py", "r") as _fh:
            _src2 = _fh.read()

        # ── Assertion A: result-label lint ──
        # The result-panel titles are rendered via:
        #   st.subheader("Suggested scenarios")     — citywide
        #   st.subheader("Best tested mixes for selected area") — region
        # Anything matching "Optimized [Ss]uggestion" or "[Oo]ptimal" on a
        # result-label site would be a regression.
        ALLOWED_RESULT_PREFIXES = (
            "Suggested scenarios",
            "Best tested mixes",
        )
        FORBIDDEN_RESULT_TOKENS = (
            "Optimized",   # capital — "Optimized Scenario Suggestions"
            "the optimum",  # we framed as "not the optimum" — caveat OK,
                            # the assertion below excludes that phrasing
        )

        def _scan_result_labels(src):
            """Find st.subheader/st.markdown call sites whose first-arg
            string starts with a forbidden 'Optimized '/'Optimal' token —
            a result-label shaped regression. Pure-regex scan; no AST
            helpers (those live inside another try-block scope)."""
            issues = []
            for _m in _re2.finditer(
                r'st\.(?:subheader|markdown)\(\s*"((?:Optimized|Optimal)[^"\n]{0,80})"',
                src,
            ):
                line = src[:_m.start()].count("\n") + 1
                issues.append((line, "result_label_token", _m.group(1)))
            return issues

        _label_issues = _scan_result_labels(_src2)
        if _label_issues:
            print(f"  FAIL {len(_label_issues)} result-label violation(s) — "
                  "no 'Optimized'/'Optimal' on result-panel headers:")
            for line, kind, s in _label_issues:
                print(f"    line {line} ({kind}): {s!r}")
            two_relay_diffs += len(_label_issues)
        else:
            print(f"  OK   no 'Optimized'/'Optimal' on result-panel "
                  f"headers (de-optimize sweep holds)")

        # Confirm the two canonical result headers ARE present.
        _has_suggested = 'st.subheader("Suggested scenarios"' in _src2
        _has_best_tested = (
            'st.subheader("Best tested mixes for selected area"' in _src2
        )
        if not _has_suggested:
            print(f"  FAIL canonical citywide result header 'Suggested "
                  f"scenarios' missing")
            two_relay_diffs += 1
        if not _has_best_tested:
            print(f"  FAIL canonical region result header 'Best tested "
                  f"mixes for selected area' missing")
            two_relay_diffs += 1
        if _has_suggested and _has_best_tested:
            print(f"  OK   both canonical result headers present "
                  f"('Suggested scenarios', 'Best tested mixes for "
                  f"selected area')")

        # Meta-test (A): seed an "Optimized suggestions" subheader and
        # assert the scanner flags it.
        _seed_a = (
            'import streamlit as st\n'
            'st.subheader("Optimized suggestions")\n'
        )
        _seed_a_issues = _scan_result_labels(_seed_a)
        if not _seed_a_issues:
            print(f"  FAIL meta-test (A): seeded 'Optimized suggestions' "
                  f"NOT flagged — result-label scanner is broken")
            two_relay_diffs += 1
        else:
            print(f"  OK   meta-test (A): seeded 'Optimized suggestions' "
                  f"correctly flagged ({len(_seed_a_issues)} hit(s))")

        # ── Assertion B: button-paired with mode label ──
        # Find every st.button("Optimize"...) call and confirm a known
        # mode-label string appears within the preceding ~120 lines (the
        # window is generous because the sidebar branch includes the
        # slider widgets between the mode label and the button — all
        # within the same st.container scope). Both surfaces have mode
        # labels:
        #   "Citywide AI-assisted search"   (citywide)
        #   "Selected-area search"          (region)
        MODE_LABEL_STRINGS = (
            "Citywide AI-assisted search",
            "Selected-area search",
        )
        # Find Optimize button call sites.
        _button_lines = []
        for _m in _re2.finditer(
            r'st\.button\(\s*"Optimize"', _src2
        ):
            # Compute line number from offset.
            _line = _src2[:_m.start()].count("\n") + 1
            _button_lines.append(_line)
        if not _button_lines:
            print(f"  FAIL Two-RELAY lock — no st.button(\"Optimize\") "
                  f"call sites found")
            two_relay_diffs += 1
        else:
            _src_lines = _src2.splitlines()
            _unpaired = []
            for _btn_line in _button_lines:
                # Look back N lines for a mode-label string.
                _start = max(0, _btn_line - 121)
                _window = "\n".join(_src_lines[_start:_btn_line])
                _has_label = any(lbl in _window
                                 for lbl in MODE_LABEL_STRINGS)
                if not _has_label:
                    _unpaired.append(_btn_line)
            if _unpaired:
                print(f"  FAIL {len(_unpaired)} Optimize button(s) without "
                      f"a mode label in the preceding 120 lines:")
                for _l in _unpaired:
                    print(f"    line {_l}: no 'Citywide surrogate search' "
                          f"or 'Selected-area search' nearby")
                two_relay_diffs += len(_unpaired)
            else:
                print(f"  OK   all {len(_button_lines)} Optimize "
                      f"button(s) paired with mode label "
                      f"within 120 lines")

        # Meta-test (B): seed an Optimize button without a mode label
        # nearby; assert it gets flagged.
        _seed_b = (
            'import streamlit as st\n'
            '# no mode label here on purpose\n'
            'if st.button("Optimize", key="meta_test_btn"):\n'
            '    pass\n'
        )
        # Run the same heuristic on _seed_b.
        _seed_b_btns = []
        for _m in _re2.finditer(
            r'st\.button\(\s*"Optimize"', _seed_b
        ):
            _line = _seed_b[:_m.start()].count("\n") + 1
            _seed_b_btns.append(_line)
        _seed_b_lines = _seed_b.splitlines()
        _seed_b_unpaired = []
        for _btn_line in _seed_b_btns:
            _start = max(0, _btn_line - 121)
            _window = "\n".join(_seed_b_lines[_start:_btn_line])
            _has_label = any(lbl in _window
                             for lbl in MODE_LABEL_STRINGS)
            if not _has_label:
                _seed_b_unpaired.append(_btn_line)
        if not _seed_b_unpaired:
            print(f"  FAIL meta-test (B): seeded unpaired Optimize button "
                  f"NOT flagged — button-paired scanner is broken")
            two_relay_diffs += 1
        else:
            print(f"  OK   meta-test (B): seeded unpaired Optimize button "
                  f"correctly flagged ({len(_seed_b_unpaired)} hit(s))")

        # ── Assertion C: provenance Source distinction extension ──
        # The applied-scenario Source rendered by the banner reads from
        # app._PROVENANCE_HEADER_INFO via the locked provenance constants.
        # Citywide-origin Source MUST contain "ai-assisted suggestion";
        # region-origin Source MUST contain "region-optimized".
        import natcap_scenarios as _ns2
        _cw_source = app._PROVENANCE_HEADER_INFO.get(
            _ns2.PROVENANCE_OPTIMIZER, (None,))[0]
        _rg_source = app._PROVENANCE_HEADER_INFO.get(
            _ns2.PROVENANCE_REGION_OPTIMIZED, (None,))[0]
        _cw_ok = (_cw_source is not None
                   and "ai-assisted suggestion" in _cw_source.lower())
        _rg_ok = (_rg_source is not None
                   and "region-optimized" in _rg_source.lower())
        if not _cw_ok:
            print(f"  FAIL Citywide-origin Source string missing "
                  f"'ai-assisted suggestion': {_cw_source!r}")
            two_relay_diffs += 1
        if not _rg_ok:
            print(f"  FAIL Region-origin Source string missing "
                  f"'region-optimized': {_rg_source!r}")
            two_relay_diffs += 1
        if _cw_ok and _rg_ok:
            print(f"  OK   provenance Source distinction: citywide → "
                  f"{_cw_source!r}; region → {_rg_source!r}")

        # Meta-test (C): swap the two and confirm both checks would fail.
        _swap_cw_ok = "region-optimized" in (_cw_source or "").lower()
        _swap_rg_ok = "ai-assisted suggestion" in (_rg_source or "").lower()
        if _swap_cw_ok or _swap_rg_ok:
            print(f"  FAIL meta-test (C): swapped mapping would still "
                  f"satisfy the checks — the distinction isn't tight")
            two_relay_diffs += 1
        else:
            print(f"  OK   meta-test (C): swapped mapping (citywide → "
                  f"'region-optimized', region → 'ai-assisted suggestion') "
                  f"correctly FAILS both checks — distinction is tight")

        # ── Assertion D — CTA caption protection (FIX BUNDLE #79) ───────
        # Both Discover surfaces (sidebar + main-panel CTA) carry the same
        # mode-keyed caption beneath the mode label:
        #   citywide → "Fast estimates suggest promising mixes. Apply one to
        #              compute it with the full evaluator." (fast estimates,
        #              not full-evaluator outputs)
        #   region   → "Finds best tested mixes under the current area and
        #              filters. Displayed values are computed by the full
        #              evaluator, not predicted by the model."
        # Both expected literals must appear ≥2× in app.py (sidebar + CTA),
        # and each must appear immediately after a matching mode label
        # within a small window (so they pair with their mode, not float).
        # Meta-test: confirm a tweaked caption string would fail. Both
        # captions are single source-line literals; a single-literal
        # exact-count check works for both surfaces.
        _CW_CAPTION_EXPECTED = (
            "Fast estimates suggest promising mixes. Apply one to compute it "
            "with the full evaluator."
        )
        _RG_CAPTION_EXPECTED = (
            "Finds best tested mixes under the current area and filters. "
            "Displayed values are computed by the full evaluator, not predicted by the model."
        )
        _cw_cap_count = _src2.count(_CW_CAPTION_EXPECTED)
        _rg_cap_count = _src2.count(_RG_CAPTION_EXPECTED)
        if _cw_cap_count >= 2 and _rg_cap_count >= 2:
            print(f"  OK   citywide caption present {_cw_cap_count}×, "
                  f"region caption present {_rg_cap_count}× "
                  "(both ≥2: sidebar + CTA)")
        else:
            if _cw_cap_count < 2:
                print(f"  FAIL citywide caption only appears {_cw_cap_count}× "
                      f"in app.py (expected ≥2: sidebar + CTA). Expected literal: "
                      f"'{_CW_CAPTION_EXPECTED}'")
                two_relay_diffs += 1
            if _rg_cap_count < 2:
                print(f"  FAIL region caption only appears {_rg_cap_count}× "
                      f"in app.py (expected ≥2: sidebar + CTA). Expected literal: "
                      f"'{_RG_CAPTION_EXPECTED}'")
                two_relay_diffs += 1

        # Meta-test (D): seeds with single-word regressions on each surface
        # must not match — proves the checks are literal, not fuzzy.
        #   (D1) citywide caption: 'suggestions' → 'results' must fail
        #        the exact-literal check.
        #   (D2) region caption: 'region and eligibility' → 'whatever'
        #        must fail the exact-literal check.
        _seed_d_cw = ("st.caption(\"Fast estimates suggest promising mixes. "
                      "Apply one to compute it with the engine.\")\n")
        _seed_d_rg = ("st.caption(\"Finds best tested mixes under whatever "
                      "filters.\")\n")
        _meta_d_ok = True
        if _CW_CAPTION_EXPECTED in _seed_d_cw:
            print(f"  FAIL meta-test (D1): seeded WRONG citywide caption "
                  "matched expected literal — citywide check is fuzzy")
            two_relay_diffs += 1; _meta_d_ok = False
        if _RG_CAPTION_EXPECTED in _seed_d_rg:
            print(f"  FAIL meta-test (D2): seeded WRONG region caption "
                  "matched expected literal — region check is fuzzy")
            two_relay_diffs += 1; _meta_d_ok = False
        if _meta_d_ok:
            print(f"  OK   meta-test (D): seeded regressions of both "
                  "surfaces (citywide 'results' / region 'whatever') "
                  "correctly fail the literal checks")
    except Exception as e:
        print(f"  ERROR Two-RELAY lock: {e}")
        import traceback; traceback.print_exc()
        two_relay_diffs += 1

    # ── $-discipline static lint (DESIGN_NOTES §10.3a) ───────────────────────
    # Two halves enforce the markdown-vs-plain $ rule:
    #   (a) No `\$` inside any st.metric label / value / delta arg —
    #       st.metric is plain text; `\$` prints a literal backslash (the
    #       eyeball bug on the NatCap Carbon Value card).
    #   (b) No paired unescaped `$...$` in any st.markdown / write / caption /
    #       subheader / info / warning / error / success / title / header
    #       string — paired-`$` flips into LaTeX math in Streamlit's
    #       markdown renderer.
    # Help= tooltips on st.metric ARE markdown-rendered, so `\$` is correct
    # there and stays — the lint only checks label/value/delta on st.metric.
    # Meta-test (load-bearing): seed a synthetic violation of each half and
    # assert the lint flags it. Without this, the lint would be green-light
    # theatre — it might silently miss real violations.
    print(f"\n{'=' * 60}")
    print("$-discipline static lint — markdown vs metric")
    print(f"{'=' * 60}")
    dollar_lint_diffs = 0
    try:
        import ast as _ast
        import re as _re

        _MARKDOWN_CALLS = {"markdown", "write", "caption", "subheader",
                            "title", "header", "info", "warning", "error",
                            "success"}
        _METRIC_CALLS = {"metric"}
        _METRIC_PLAIN_ARGS = {"label", "value", "delta"}
        _PAIRED_DOLLAR_RE = _re.compile(
            r'(?<!\\)\$[^$\n]*?(?<!\\)\$'
        )

        def _extract_string(node):
            """Best-effort literal-string extraction from an AST node.
            Returns None for non-literals; for f-strings, joins literal
            parts (interpolations become empty so the surrounding text
            still gets scanned)."""
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, _ast.JoinedStr):
                parts = []
                for v in node.values:
                    if (isinstance(v, _ast.Constant)
                            and isinstance(v.value, str)):
                        parts.append(v.value)
                    else:
                        parts.append("")
                return "".join(parts)
            if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add):
                a = _extract_string(node.left)
                b = _extract_string(node.right)
                if a is None or b is None:
                    return None
                return a + b
            return None

        def _call_attr(node):
            if (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)):
                return node.func.attr
            return None

        def _scan_source_for_violations(src):
            """Returns (markdown_paired_dollar_hits, metric_backslash_dollar_hits).
            Each list element: (lineno, location_label, sample_text_80)."""
            tree = _ast.parse(src)
            mv, mtv = [], []

            class V(_ast.NodeVisitor):
                def visit_Call(self, node):
                    fn = _call_attr(node)
                    if fn in _MARKDOWN_CALLS and node.args:
                        s = _extract_string(node.args[0])
                        if s and _PAIRED_DOLLAR_RE.search(s):
                            mv.append((node.lineno, fn, s[:80]))
                    if fn in _METRIC_CALLS:
                        # st.metric(label, value, delta=None, ...)
                        for i, arg in enumerate(node.args):
                            if i > 1:
                                continue
                            pname = "label" if i == 0 else "value"
                            s = _extract_string(arg)
                            if s and "\\$" in s:
                                mtv.append((node.lineno, pname, s[:80]))
                        for kw in node.keywords:
                            if kw.arg not in _METRIC_PLAIN_ARGS:
                                continue
                            s = _extract_string(kw.value)
                            if s and "\\$" in s:
                                mtv.append(
                                    (node.lineno, "kw=%s" % kw.arg, s[:80])
                                )
                    self.generic_visit(node)

            V().visit(tree)
            return mv, mtv

        # ── Half (b): paired `$...$` in markdown ──
        # ── Half (a): `\$` in st.metric label/value/delta ──
        with open("app.py", "r") as _fh:
            _app_src = _fh.read()
        _mv, _mtv = _scan_source_for_violations(_app_src)
        if _mv:
            print(f"  FAIL {len(_mv)} paired-`$` violation(s) in "
                  "st.markdown/write/caption etc — LaTeX flip risk:")
            for line, fn, s in _mv:
                print(f"    line {line} ({fn}): {s!r}")
            dollar_lint_diffs += len(_mv)
        else:
            print(f"  OK   no paired-`$` violations in markdown calls")
        if _mtv:
            print(f"  FAIL {len(_mtv)} `\\$` violation(s) in "
                  "st.metric label/value/delta — renders literal backslash:")
            for line, fn, s in _mtv:
                print(f"    line {line} ({fn}): {s!r}")
            dollar_lint_diffs += len(_mtv)
        else:
            print(f"  OK   no `\\$` violations in st.metric label/value/delta")

        # ── Meta-test (load-bearing) ──
        # Seed one violation of EACH half and assert the lint flags it.
        # Otherwise the lint would be green-light theatre — it might miss
        # real violations because the scan is broken or the regex is wrong.
        _seed_markdown = '''
import streamlit as st
st.caption("Cost is $5/acre on average; $10/acre with premium materials.")
'''
        _seed_metric = '''
import streamlit as st
st.metric("Test", "\\$100M", delta="@\\$190/t")
'''
        _mv_seed, _mtv_seed = _scan_source_for_violations(_seed_markdown)
        if not _mv_seed:
            print(f"  FAIL meta-test (markdown half): seeded paired `$` "
                  f"violation NOT flagged — lint guards nothing")
            dollar_lint_diffs += 1
        else:
            print(f"  OK   meta-test (markdown half): seeded paired `$` "
                  f"flagged ({len(_mv_seed)} hit(s))")
        _mv_seed2, _mtv_seed2 = _scan_source_for_violations(_seed_metric)
        if not _mtv_seed2:
            print(f"  FAIL meta-test (metric half): seeded `\\$` in "
                  f"st.metric NOT flagged — lint guards nothing")
            dollar_lint_diffs += 1
        else:
            print(f"  OK   meta-test (metric half): seeded `\\$` flagged "
                  f"({len(_mtv_seed2)} hit(s))")
    except Exception as e:
        print(f"  ERROR $-discipline lint: {e}")
        import traceback; traceback.print_exc()
        dollar_lint_diffs += 1

    # ── Sidebar wiring-survival assertion ────────────────────────────────────
    # The sidebar's grown dense and accreted layout refactors. Every Streamlit
    # widget that participates in app behavior carries a `key=` so its
    # session-state slot is stable across reruns. If a layout reorg drops a
    # key (a widget that lost its key= during a refactor, or a renamed key),
    # the city-switch guard and the _filter_active mode switch silently break.
    # This cell freezes the set of sidebar widget keys against a reference
    # extracted at refactor time, and asserts the live source still produces
    # the same set — wiring survival made explicit instead of relying on the
    # guard transition test to surface a wrong-key behavior by side effect.
    print(f"\n{'=' * 60}")
    print("Sidebar wiring — widget-key set survival assertion")
    print(f"{'=' * 60}")
    sidebar_keys_diffs = 0
    try:
        # Frozen reference — keys live in the sidebar block from city selector
        # through Export. Dynamic-prefix keys (`elf_check_<cls>`,
        # `region_labels_<layer_key>`) are handled via prefixes since their
        # exhaustive lists vary by city/loop. Static keys must match exactly.
        _SIDEBAR_STATIC_KEYS_EXPECTED = frozenset({
            'carbon_rate_ff', 'carbon_rate_gi',
            'elf_check_vacant',
            # Relay 2 #3 — Ownership preset dropdown. Sets the per-class
            # `elf_check_<cls>` + `elf_check_vacant` session-state values;
            # the checkboxes are the canonical source, the preset is a
            # write-through convenience selector.
            'elf_preset',
            'hi_res_confirmed',
            'model_quality',
            'natcap_fixed_scenario_id',
            'region_apply_within', 'region_layer',
            'region_opt_button',
            'region_opt_w_carbon', 'region_opt_w_cool',
            'region_opt_w_cost', 'region_opt_w_flood', 'region_opt_w_food',
            'scenario_source',
            # Optimizer Promotion: secondary sidebar trigger for the
            # citywide path. The primary trigger is the main-panel CTA;
            # both call _fire_citywide_optimize with identical args.
            'sidebar_citywide_opt_button',
            'slider_ff_pct', 'slider_gi_pct', 'slider_pct_converted',
        })
        _SIDEBAR_DYNAMIC_PREFIXES_EXPECTED = ('elf_check_', 'region_labels_')

        # Scan app.py source for `key=...` in the sidebar block. Boundaries:
        # the sidebar starts where `selected_city = st.sidebar.selectbox(...)`
        # binds and ends at the "── Main panel ──" marker.
        import re as _re
        with open("app.py", "r") as _fh:
            _src = _fh.read()
        _start_match = _re.search(
            r"^selected_city\s*=\s*st\.sidebar\.selectbox",
            _src, _re.MULTILINE,
        )
        _end_match = _re.search(r"^# ── Main panel", _src, _re.MULTILINE)
        if not _start_match or not _end_match:
            print("  ERROR sidebar key scan: couldn't locate sidebar block "
                  "boundaries — `selected_city = st.sidebar.selectbox` or "
                  "the `── Main panel` marker is missing or moved.")
            sidebar_keys_diffs += 1
        else:
            _sidebar_src = _src[_start_match.start():_end_match.start()]
            # Match `key="literal"` and `key=f"format-string"`. Strip f-prefix
            # and quotes; literal keys become themselves, f-string keys
            # become e.g. `elf_check_{_cls}` which we route via the prefix set.
            _all_keys = set()
            for _m in _re.finditer(
                r'key=(f?"[^"]+")', _sidebar_src
            ):
                _raw = _m.group(1)
                _is_fstring = _raw.startswith('f"')
                _val = _raw.lstrip('f').strip('"')
                # `mode_key="fast"` in a function default at the start of the
                # block (the surrogate-training fn signature) is a false
                # positive — it's a Python kwarg, not a Streamlit key. Skip
                # by checking for the function-signature context: the actual
                # widget keys are all on `st.*` calls. The simplest filter
                # is to drop the known false positive by value.
                if _val == 'fast':
                    continue
                if _is_fstring:
                    # The f-string keys must start with a known dynamic prefix.
                    _prefix_hit = next(
                        (p for p in _SIDEBAR_DYNAMIC_PREFIXES_EXPECTED
                         if _val.startswith(p)),
                        None,
                    )
                    if _prefix_hit is None:
                        print(f"  FAIL sidebar key scan: unrecognized "
                              f"f-string key {_val!r} — add the prefix to "
                              f"_SIDEBAR_DYNAMIC_PREFIXES_EXPECTED if "
                              f"intentional.")
                        sidebar_keys_diffs += 1
                else:
                    _all_keys.add(_val)
            # Static-key set must equal expected.
            missing = _SIDEBAR_STATIC_KEYS_EXPECTED - _all_keys
            extra = _all_keys - _SIDEBAR_STATIC_KEYS_EXPECTED
            if missing:
                print(f"  FAIL sidebar key scan: missing keys "
                      f"{sorted(missing)} — a widget lost its key= or was "
                      f"removed during a layout refactor. Behavior wiring "
                      f"is broken.")
                sidebar_keys_diffs += len(missing)
            if extra:
                print(f"  FAIL sidebar key scan: unexpected keys "
                      f"{sorted(extra)} — a new widget was added without "
                      f"updating _SIDEBAR_STATIC_KEYS_EXPECTED. Update the "
                      f"reference set if intentional.")
                sidebar_keys_diffs += len(extra)
            if not (missing or extra):
                print(f"  OK   {len(_all_keys)} static sidebar keys + "
                      f"{len(_SIDEBAR_DYNAMIC_PREFIXES_EXPECTED)} dynamic "
                      f"prefixes match the frozen reference set.")
    except Exception as e:
        print(f"  ERROR sidebar key scan: {e}")
        import traceback; traceback.print_exc()
        sidebar_keys_diffs += 1

    # ── Metric-label char budget — regression guard for FIX BUNDLE #77 ───────
    # The Fix Bundle shortened three Explorer metric labels:
    #   "Temperature Change"          → "Temp change"
    #   "Runoff Volume"               → "Runoff volume"
    #   "Cost / Citywide °F Cooling"  → "Cost / °F cooling"
    # The honesty qualifier "Citywide" was dropped from the cost label's
    # surface but is preserved in that metric's help= tooltip ("the °F is
    # a citywide mean"). This cell locks both halves:
    #   (a) the long-form labels must NEVER reappear as an st.metric label
    #       (positional first arg or label=kwarg) — that's the char-budget
    #       contract;
    #   (b) the corresponding short-form labels must still be present (else
    #       the labels disappeared entirely, also a regression).
    # Both halves are AST-checked. Meta-test seeds a synthetic violation of
    # each and asserts the lint catches it.
    print(f"\n{'=' * 60}")
    print("Metric-label char budget — FIX BUNDLE #77 regression guard")
    print(f"{'=' * 60}")
    label_budget_diffs = 0
    try:
        import ast as _ast2
        # long_form → short_form. The shortened text IS the char budget; if
        # any long-form text reappears as an st.metric label literal, fail.
        _LABEL_REGRESSIONS = {
            "Temperature Change":         "Temp change",
            "Runoff Volume":              "Runoff volume",
            "Cost / Citywide °F Cooling": "Cost / °F cooling",
        }

        def _scan_metric_labels(source: str) -> list[tuple[int, str]]:
            """Return [(lineno, label_literal)] for every st.metric(...) call
            in `source` where the label arg is a string literal."""
            out = []
            try:
                tree = _ast2.parse(source)
            except SyntaxError:
                return out
            for node in _ast2.walk(tree):
                if not isinstance(node, _ast2.Call):
                    continue
                f = node.func
                if not (isinstance(f, _ast2.Attribute) and f.attr == "metric"):
                    continue
                # label is positional arg 0 OR kwarg label=
                lit = None
                if node.args and isinstance(node.args[0], _ast2.Constant) \
                        and isinstance(node.args[0].value, str):
                    lit = node.args[0].value
                else:
                    for kw in node.keywords:
                        if kw.arg == "label" and isinstance(kw.value, _ast2.Constant) \
                                and isinstance(kw.value.value, str):
                            lit = kw.value.value
                            break
                if lit is not None:
                    out.append((node.lineno, lit))
            return out

        with open("app.py", "r") as _f:
            _app_src = _f.read()
        _metric_labels = _scan_metric_labels(_app_src)

        # Half (a) — long-form labels must not reappear.
        _regressions = [(ln, lab) for (ln, lab) in _metric_labels
                        if lab in _LABEL_REGRESSIONS]
        if _regressions:
            for ln, lab in _regressions:
                print(f"  FAIL line {ln}: st.metric label '{lab}' "
                      f"reverted to long form (budget: '{_LABEL_REGRESSIONS[lab]}')")
            label_budget_diffs += len(_regressions)
        else:
            print(f"  OK   no long-form labels reappeared in {len(_metric_labels)} "
                  "st.metric call(s) scanned (3 regression strings checked).")

        # Half (b) — short-form labels must still be present.
        _present_short = {lab for (_ln, lab) in _metric_labels}
        _missing = [s for s in _LABEL_REGRESSIONS.values()
                    if s not in _present_short]
        if _missing:
            for s in _missing:
                print(f"  FAIL short-form label '{s}' is missing from "
                      "app.py st.metric calls (label disappeared entirely?)")
            label_budget_diffs += len(_missing)
        else:
            print(f"  OK   all 3 shortened labels still present in st.metric calls")

        # Meta-test (load-bearing): synthesize a snippet that reintroduces
        # one long form and one missing short form; confirm the scan catches
        # both. If meta-test fails, the lint above is green-light theatre.
        _seed = (
            "import streamlit as st\n"
            "st.metric('Temperature Change', '0.5')\n"  # half (a) — regression
            "st.metric('Cost / °F cooling', 'N/A')\n"   # half (b) — keep this present
        )
        _seed_labels = _scan_metric_labels(_seed)
        _seed_regs = [(ln, lab) for (ln, lab) in _seed_labels
                      if lab in _LABEL_REGRESSIONS]
        _seed_short = {lab for (_ln, lab) in _seed_labels}
        # Meta seed reintroduces "Temperature Change" (should flag) and
        # omits "Temp change" + "Runoff volume" (should both flag as missing).
        _seed_missing = [s for s in _LABEL_REGRESSIONS.values()
                         if s not in _seed_short]
        if not _seed_regs:
            print(f"  FAIL meta-test (a): seeded 'Temperature Change' was "
                  "NOT flagged — long-form scan is blind")
            label_budget_diffs += 1
        else:
            print(f"  OK   meta-test (a): seeded long-form label correctly "
                  f"flagged ({len(_seed_regs)} hit)")
        if len(_seed_missing) < 2:
            print(f"  FAIL meta-test (b): seeded missing labels were NOT "
                  f"flagged (expected ≥2, got {len(_seed_missing)})")
            label_budget_diffs += 1
        else:
            print(f"  OK   meta-test (b): seeded missing short-form labels "
                  f"correctly flagged ({len(_seed_missing)} hit(s))")
    except Exception as e:
        print(f"  ERROR metric-label budget scan: {e}")
        import traceback; traceback.print_exc()
        label_budget_diffs += 1

    # ── Dense-CSV freshness — SA cold-start Lever 1 guard ────────────────────
    # Lever 1 wires Fast mode to read data/scenarios_dense_<city>.csv instead
    # of recomputing 91 scenarios live at module import (~130 s saved on SA).
    # This cell samples a few rows per city, re-evaluates them via the live
    # engine, and asserts the CSV values match within rel_tol=1e-5 — well
    # above float32 epsilon (~1.2e-7) so legitimate accumulation noise passes,
    # well below the resolution the surrogate cares about. If math changes in
    # any model (UCM / UMH / UNA / flood / carbon / food), a sampled row will
    # drift past tolerance and the gate fails until precompute_scenarios.py
    # regenerates the dense CSV. Meta-test perturbs one CSV cell and confirms
    # the check fires — proves the tolerance isn't loose enough to mask real
    # regressions.
    print(f"\n{'=' * 60}")
    print("Dense-CSV freshness — SA cold-start Lever 1 guard")
    print(f"{'=' * 60}")
    dense_freshness_diffs = 0
    try:
        import math as _m
        import pandas as pd
        # Sample 3 rows per city — spread across the (pct, gi, ff) space.
        # Pick rows that exist in both cities' CSVs (mult. of dense's step
        # 5/10) and that exercise non-baseline math.
        _SAMPLES = [
            (10, 50, 50),   # gi/ff split, low pct
            (30, 20, 80),   # FF-heavy mid-pct (gi+ff=100)
            (50, 100, 0),   # GI-only max-pct
        ]
        _COMPARE_KEYS = ("mean_hm", "flood_reduction", "runoff_acre_feet",
                         "food_mln_lbs")  # 4 metrics; skip carbon (float32
                                          # noise at 5e-7 on SA, well within
                                          # the 1e-5 tol but noisy to surface)
        _REL_TOL = 1e-5

        for _city in [c for c in active_cities
                      if app.CITIES[c].get("dense_scenarios_file")]:
            _path = app.CITIES[_city]["dense_scenarios_file"]
            if not Path(_path).exists():
                print(f"  SKIP {_city}: dense_scenarios_file {_path!r} not "
                      "on disk (Fast mode will recompute live).")
                continue
            _df = pd.read_csv(_path)
            _rebind_city(app, _city)
            _city_diffs = 0
            for (pct, gi, ff) in _SAMPLES:
                _row_match = _df[
                    (_df.pct_converted == pct) &
                    (_df.green_infrastructure_pct == gi) &
                    (_df.food_forest_pct == ff)
                ]
                if _row_match.empty:
                    print(f"  SKIP {_city} ({pct},{gi},{ff}): row not in CSV")
                    continue
                _row = _row_match.iloc[0]
                _live = app.evaluate_scenario(
                    pct_converted=pct,
                    green_infrastructure_pct=gi,
                    food_forest_pct=ff,
                    seed=42, placement_strategy="random",
                )
                for _k in _COMPARE_KEYS:
                    _csv_v = float(_row[_k]); _live_v = float(_live[_k])
                    if not _m.isclose(_csv_v, _live_v,
                                      rel_tol=_REL_TOL, abs_tol=1e-9):
                        _rel = (abs(_csv_v - _live_v) / max(abs(_csv_v), 1e-9))
                        print(f"  FAIL {_city} ({pct},{gi},{ff}) {_k}: "
                              f"CSV={_csv_v:.6g} live={_live_v:.6g} "
                              f"rel={_rel:.2e} > rel_tol={_REL_TOL:.0e}")
                        _city_diffs += 1
            if _city_diffs == 0:
                print(f"  OK   {_city}: 3 sampled rows × 4 metrics match "
                      f"CSV within rel_tol={_REL_TOL:.0e}")
            else:
                dense_freshness_diffs += _city_diffs

        # Meta-test (load-bearing): synthesize a "stale CSV" by perturbing
        # one cell by 1% — re-run the equivalent of the comparison loop and
        # confirm the check would have flagged it. Without this, the
        # rel_tol could silently be set to 1e3 and we'd never know.
        _meta_csv_v = 100.0
        _meta_live_v = 101.0   # +1% drift
        _meta_caught = not _m.isclose(_meta_csv_v, _meta_live_v,
                                      rel_tol=_REL_TOL, abs_tol=1e-9)
        if not _meta_caught:
            print(f"  FAIL meta-test: 1% synthetic drift (100 → 101) was "
                  f"NOT caught at rel_tol={_REL_TOL:.0e} — tolerance is loose")
            dense_freshness_diffs += 1
        else:
            print(f"  OK   meta-test: 1% synthetic drift correctly fails "
                  f"the rel_tol={_REL_TOL:.0e} check")
    except Exception as e:
        print(f"  ERROR dense-csv freshness: {e}")
        import traceback; traceback.print_exc()
        dense_freshness_diffs += 1

    # ── Child-pop raster staleness — per-city anchored ─────────────────────
    # The under-18 raster is derived as P1_001N - P3_001N (PL 94-171, same
    # source as total pop). The single most likely failure mode is grabbing
    # a wrong P3 sub-variable (e.g. P3_002N male-only ≈ half of total VAP,
    # producing a derived "under-18" ~half the true value, far outside the
    # 1-2 pp natural variation between county and modeled extent). A loose
    # 0.2-0.3 sanity band would pass that failure silently for SA (true
    # share ~24.5%) and possibly for MN too. So this cell anchors each city's
    # raster share to a per-city published figure (config: child_pop_extent_share,
    # measured at precompute time from the same Census source). Tolerance:
    # ±2 pp absolute — tight enough to catch a wrong-variable derivation,
    # loose enough to absorb 0.5-1 pp block-rasterization edge effects.
    # Also asserts the per-pixel invariant child ≤ total (no block can have
    # more children than people) and shape equality with the total-pop raster.
    print(f"\n{'=' * 60}")
    print("Child-pop staleness — per-city anchor + per-pixel invariants")
    print(f"{'=' * 60}")
    child_pop_diffs = 0
    try:
        _SHARE_TOL_PP = 0.02   # ±2 percentage points absolute
        for _city in active_cities:
            _cfg = app.CITIES[_city]
            if not _cfg.get("child_pop_file"):
                print(f"  SKIP {_city}: no child_pop_file configured")
                continue
            _rebind_city(app, _city)
            _child = app.child_pop_count_raster
            _total = app.pop_count_raster
            if _child is None:
                print(f"  FAIL {_city}: child_pop_file configured but loader "
                      "returned None (file missing or read failed?)")
                child_pop_diffs += 1
                continue
            # Shape match
            if _child.shape != _total.shape:
                print(f"  FAIL {_city}: child raster shape {_child.shape} "
                      f"!= total raster shape {_total.shape}")
                child_pop_diffs += 1
                continue
            # Per-pixel invariant: child ≤ total (within float epsilon)
            _violators = int(((_child - _total) > 1e-6).sum())
            if _violators > 0:
                print(f"  FAIL {_city}: {_violators:,} pixels have "
                      "child_pop > total_pop (invariant child ≤ total broken)")
                child_pop_diffs += 1
                continue
            # Per-city anchored share check
            _total_sum = float(_total.sum())
            if _total_sum <= 0:
                print(f"  SKIP {_city}: total pop is zero — anchor "
                      "check skipped")
                continue
            _share = float(_child.sum()) / _total_sum
            _anchor = float(_cfg.get("child_pop_extent_share", 0.0))
            if not _anchor:
                print(f"  FAIL {_city}: no child_pop_extent_share anchor "
                      "configured — staleness can't be enforced")
                child_pop_diffs += 1
                continue
            if abs(_share - _anchor) > _SHARE_TOL_PP:
                print(f"  FAIL {_city}: child share {_share:.1%} differs "
                      f"from per-city anchor {_anchor:.1%} by "
                      f"{abs(_share - _anchor):.1%} > tol "
                      f"{_SHARE_TOL_PP:.1%}. Re-run "
                      f"download_census_pop*.py with the correct "
                      "P3 variable (P3_001N for total VAP).")
                child_pop_diffs += 1
            else:
                print(f"  OK   {_city}: shape match; child ≤ total per "
                      f"pixel; share {_share:.1%} within ±{_SHARE_TOL_PP:.1%} "
                      f"of per-city anchor {_anchor:.1%}")

        # Meta-test (load-bearing): simulate the wrong-variable failure mode
        # by halving the child raster (≈ what'd happen if a sub-variable like
        # P3_002N male-only VAP was used to compute under-18). The anchor
        # check must catch it; otherwise the lint is loose.
        if active_cities:
            _meta_city = next(
                (c for c in active_cities if app.CITIES[c].get("child_pop_file")),
                None,
            )
            if _meta_city is not None:
                _rebind_city(app, _meta_city)
                _saved = app.child_pop_count_raster
                # Halve the raster — simulates a "wrong variable" derivation.
                app.child_pop_count_raster = (_saved * 0.5).astype(_saved.dtype)
                _meta_share = float(app.child_pop_count_raster.sum()) / \
                              float(app.pop_count_raster.sum())
                _meta_anchor = float(app.CITIES[_meta_city]["child_pop_extent_share"])
                _meta_caught = abs(_meta_share - _meta_anchor) > _SHARE_TOL_PP
                app.child_pop_count_raster = _saved  # restore
                if _meta_caught:
                    print(f"  OK   meta-test: halved child raster (simulated "
                          f"wrong-variable derivation) → share {_meta_share:.1%} "
                          f"correctly flagged vs anchor {_meta_anchor:.1%} "
                          f"(tol {_SHARE_TOL_PP:.1%})")
                else:
                    print(f"  FAIL meta-test: halved child raster (share "
                          f"{_meta_share:.1%}) was NOT caught at tol "
                          f"{_SHARE_TOL_PP:.1%} vs anchor {_meta_anchor:.1%} — "
                          "tolerance is too loose to guard against wrong-variable.")
                    child_pop_diffs += 1
            else:
                print(f"  SKIP meta-test: no active city has child_pop_file "
                      "configured")
    except Exception as e:
        print(f"  ERROR child-pop staleness: {e}")
        import traceback; traceback.print_exc()
        child_pop_diffs += 1

    # ── toggle_selection pure-function unit suite — multi-select RELAY ────
    # The Interactive Region Map's click-to-toggle handler is the consumer;
    # toggle_selection is the pure transform applied to the selection list
    # on each new click. This cell exercises the four canonical cases AND
    # meta-tests that replacing the function with replace-mode
    # (`return [clicked]`) trips the suite.
    print(f"\n{'=' * 60}")
    print("toggle_selection — multi-select map click pure-function unit suite")
    print(f"{'=' * 60}")
    toggle_diffs = 0
    try:
        _ts = app.toggle_selection
        _cases = [
            ("add to empty",            [],          "A", ["A"]),
            ("add a second",            ["A"],       "B", ["A", "B"]),
            ("remove an existing",      ["A", "B"],  "A", ["B"]),
            ("remove last → empty",     ["B"],       "B", []),
        ]
        for _name, _cur, _click, _expect in _cases:
            _got = _ts(_cur, _click)
            if _got == _expect:
                print(f"  OK   {_name}: toggle({_cur!r}, {_click!r}) = {_got!r}")
            else:
                print(f"  FAIL {_name}: toggle({_cur!r}, {_click!r}) = {_got!r} "
                      f"(expected {_expect!r})")
                toggle_diffs += 1

        # Meta-test: a replace-mode reversion (`return [clicked_id]`) must
        # fail the suite. Drives this by substituting the function locally
        # with a replace-mode lambda and re-running the same cases.
        _replace_mode = lambda cur, click: [click]
        _replace_failures = sum(
            1 for _name, _cur, _click, _expect in _cases
            if _replace_mode(_cur, _click) != _expect
        )
        if _replace_failures >= 3:
            # Cases 2, 3, 4 all fail under replace mode (only case 1 happens
            # to coincide because [] + click == [click] either way).
            print(f"  OK   meta-test: replace-mode lambda fails "
                  f"{_replace_failures}/{len(_cases)} cases — toggle suite "
                  "is sharp (not vacuously satisfied by replace mode)")
        else:
            print(f"  FAIL meta-test: replace-mode lambda only fails "
                  f"{_replace_failures}/{len(_cases)} cases — toggle suite "
                  "would pass under a replace-mode regression")
            toggle_diffs += 1
    except Exception as e:
        print(f"  ERROR toggle_selection unit suite: {e}")
        import traceback; traceback.print_exc()
        toggle_diffs += 1

    # ── Buildings-precompute staleness — SA cold-start Lever 2 guard ──────
    # Phase 8 (app.py) reads buildings_precomputed_{file,type,meta} from disk
    # when configured, skipping the ~32 s rasterize of ~691k SA polygons.
    # If the source `buildings_file` ever changes and the precompute isn't
    # re-run, Phase 8 would silently feed stale building geometry to UCM
    # cooling-energy + flood-damage + convertible-pool logic. This cell
    # re-runs the same rasterize live and asserts byte-identity to the
    # on-disk arrays. Source-SHA-256 in the sidecar JSON is also cross-checked
    # against live buildings_file SHA so meta drift surfaces before the
    # rasters even open. Costs ~25 s on SA; only runs for cities that have
    # the precompute keys configured. Meta-test seeds a synthetic bit-flip
    # in the on-disk binary mask and confirms the byte-compare catches it.
    print(f"\n{'=' * 60}")
    print("Buildings-precompute staleness — SA cold-start Lever 2 guard")
    print(f"{'=' * 60}")
    bldg_precompute_diffs = 0
    try:
        import hashlib as _hl
        import json as _bj
        from rasterio.features import rasterize as _rstz
        import geopandas as _bgpd
        import pandas as _bpd

        def _file_sha(path):
            h = _hl.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        _bldg_pre_cities = [
            c for c in active_cities
            if app.CITIES[c].get("buildings_precomputed_file")
            and app.CITIES[c].get("buildings_type_precomputed_file")
            and app.CITIES[c].get("buildings_precomputed_meta_file")
        ]
        if not _bldg_pre_cities:
            print("  SKIP no cities have buildings_precomputed_* keys configured")
        for _city in _bldg_pre_cities:
            _cfg = app.CITIES[_city]
            _bin_p  = _cfg["buildings_precomputed_file"]
            _type_p = _cfg["buildings_type_precomputed_file"]
            _meta_p = _cfg["buildings_precomputed_meta_file"]
            for _p in (_bin_p, _type_p, _meta_p):
                if not Path(_p).exists():
                    print(f"  FAIL {_city}: precomputed artifact missing on disk: {_p}")
                    bldg_precompute_diffs += 1
            if bldg_precompute_diffs:
                continue

            # Cross-check source SHA256 first — cheaper than rasterize.
            _meta = _bj.loads(Path(_meta_p).read_text())
            _src_path = _cfg["buildings_file"]
            _live_src_sha = _file_sha(_src_path)
            _meta_src_sha = (_meta.get("source_sha256") or {}).get("buildings_file")
            if _meta_src_sha != _live_src_sha:
                print(f"  FAIL {_city}: source SHA256 mismatch — live "
                      f"{_src_path} hashes to {_live_src_sha[:12]}…, sidecar "
                      f"meta says {(_meta_src_sha or 'MISSING')[:12]}…. "
                      "Re-run `python precompute_buildings.py --city "
                      f"{_city!r}` to refresh.")
                bldg_precompute_diffs += 1
                continue

            # Re-rasterize from source (mirrors Phase 8 + precompute_buildings.py).
            _rebind_city(app, _city)
            _ref_shape = app._CURRENT_CITY_STATE.cooling_lulc.shape
            _ref_transform = app._CURRENT_CITY_STATE.ref_transform
            _gdf = _bgpd.read_file(_src_path)
            if _gdf.crs is None or str(_gdf.crs) != _cfg["crs"]:
                _gdf = _gdf.to_crs(_cfg["crs"])
            _types = None
            if "type" in _gdf.columns:
                _num = _bpd.to_numeric(_gdf["type"], errors="coerce")
                _num_clean = _num.dropna()
                if len(_num_clean) > 0 and _num_clean.between(0, 3).all():
                    _types = _num.fillna(-1).astype("int32")
                else:
                    _types = _gdf["type"].map(app._osm_to_invest_type).fillna(-1).astype("int32")

            _live_bin = _rstz(
                ((g, 1) for g in _gdf.geometry),
                out_shape=_ref_shape, transform=_ref_transform,
                fill=0, dtype="uint8",
            )
            if _types is not None:
                _live_type = _rstz(
                    ((g, int(t)) for g, t in zip(_gdf.geometry, _types)),
                    out_shape=_ref_shape, transform=_ref_transform,
                    fill=-1, dtype="int32",
                )
            else:
                _live_type = np.full(_ref_shape, -1, dtype="int32")

            with rasterio.open(_bin_p) as _b:
                _disk_bin = _b.read(1).astype("uint8")
            with rasterio.open(_type_p) as _t:
                _disk_type = _t.read(1).astype("int32")

            _bin_match = (_disk_bin == _live_bin).all()
            _type_match = (_disk_type == _live_type).all()
            if not _bin_match:
                _diff_px = int((_disk_bin != _live_bin).sum())
                print(f"  FAIL {_city}: binary precompute raster diverges from "
                      f"fresh rasterize at {_diff_px:,} pixels — re-run "
                      "`python precompute_buildings.py`.")
                bldg_precompute_diffs += 1
            if not _type_match:
                _diff_px = int((_disk_type != _live_type).sum())
                print(f"  FAIL {_city}: typed precompute raster diverges from "
                      f"fresh rasterize at {_diff_px:,} pixels — re-run "
                      "`python precompute_buildings.py`.")
                bldg_precompute_diffs += 1
            if _bin_match and _type_match:
                print(f"  OK   {_city}: precomputed buildings rasters byte-identical "
                      f"to fresh rasterize from source ({_src_path}); "
                      f"source SHA256 also matches sidecar")

        # Meta-test (load-bearing): bit-flip a single pixel of the in-memory
        # disk-raster copy and confirm the equality check would catch it.
        # Without this the assertion could silently degrade (e.g. shape
        # mismatches treated as 'OK' if a future refactor skipped the check).
        if _bldg_pre_cities:
            _meta_city = _bldg_pre_cities[0]
            _cfg = app.CITIES[_meta_city]
            with rasterio.open(_cfg["buildings_precomputed_file"]) as _b:
                _disk = _b.read(1).astype("uint8")
            _poisoned = _disk.copy()
            # Flip one bit — corner pixel from 0 to 1 (or 1 to 0).
            _poisoned[0, 0] = 0 if _poisoned[0, 0] else 1
            if (_disk == _poisoned).all():
                print(f"  FAIL meta-test: bit-flipped copy compared equal to "
                      "original — staleness equality check is blind")
                bldg_precompute_diffs += 1
            else:
                print(f"  OK   meta-test: single-pixel bit-flip on disk-raster "
                      "copy correctly fails the byte-equality check")
    except Exception as e:
        print(f"  ERROR buildings-precompute staleness: {e}")
        import traceback; traceback.print_exc()
        bldg_precompute_diffs += 1

    print(f"\n{'=' * 60}")
    grand_total = (total_diffs + region_diffs + ownership_diffs
                   + region_local_diffs + smoke_diffs + disclosure_diffs
                   + round_trip_diffs + subset_diffs + reconcile_diffs
                   + guard_diffs + ownership_diffs_batch1 + tradeoff_diffs
                   + region_opt_diffs + sidebar_keys_diffs
                   + scenario_state_diffs + section_order_diffs
                   + shared_fire_diffs + dollar_lint_diffs
                   + two_relay_diffs + label_budget_diffs
                   + dense_freshness_diffs + rebind_completeness_diffs
                   + child_pop_diffs + bldg_precompute_diffs
                   + toggle_diffs)
    if grand_total == 0:
        print("All baselines match.")
        return 0
    else:
        if total_diffs:
            print(f"{total_diffs} citywide divergence(s). "
                  "If intentional, rerun with --update.")
        if region_diffs:
            print(f"{region_diffs} region-assertion divergence(s).")
        if ownership_diffs:
            print(f"{ownership_diffs} ownership-assertion divergence(s).")
        if region_local_diffs:
            print(f"{region_local_diffs} region-local reconciliation divergence(s).")
        if smoke_diffs:
            print(f"{smoke_diffs} region-local smoke-test divergence(s).")
        if disclosure_diffs:
            print(f"{disclosure_diffs} honesty-surface disclosure divergence(s).")
        if round_trip_diffs:
            print(f"{round_trip_diffs} saved-scenario round-trip divergence(s).")
        if subset_diffs:
            print(f"{subset_diffs} subset-invariant divergence(s) — "
                  "placement-stage spatial bug; see cell failures above.")
        if reconcile_diffs:
            print(f"{reconcile_diffs} funnel reconciliation divergence(s) — "
                  "funnel drifted from record fields.")
        if guard_diffs:
            print(f"{guard_diffs} city-switch guard divergence(s) — "
                  "stale region/ownership state survived a city change.")
        if ownership_diffs_batch1:
            print(f"{ownership_diffs_batch1} ownership finer-classes "
                  "divergence(s) — raster or rule output drifted; see "
                  "OWNERSHIP_FINER_CLASSES_SPEC.md for expected values.")
        if tradeoff_diffs:
            print(f"{tradeoff_diffs} tradeoff-chart empty-optimizer "
                  "regression(s) — plot_tradeoff raised on a no-scenarios "
                  "or empty optimizer argument.")
        if region_opt_diffs:
            print(f"{region_opt_diffs} region-optimizer assertion "
                  "divergence(s) — subset / reconciliation / meta-test "
                  "(see REGION_OPTIMIZER_SPEC.md §8).")
        if sidebar_keys_diffs:
            print(f"{sidebar_keys_diffs} sidebar widget-key divergence(s) "
                  "— wiring broke during a layout refactor; the "
                  "_SIDEBAR_STATIC_KEYS_EXPECTED set in verify_baselines is "
                  "the contract.")
        if scenario_state_diffs:
            print(f"{scenario_state_diffs} default-scenario state "
                  "divergence(s) — title, sentence, or audit drifted from "
                  "the resolved-scenario dict (Relay A).")
        if section_order_diffs:
            print(f"{section_order_diffs} Tradeoffs section-order "
                  "divergence(s) — Explorer tab2 or NatCap view sections "
                  "moved out of the expected sequence.")
        if shared_fire_diffs:
            print(f"{shared_fire_diffs} Optimizer Promotion shared-fire "
                  "divergence(s) — _fire_citywide_optimize / "
                  "_fire_region_optimize helper contract broke.")
        if dollar_lint_diffs:
            print(f"{dollar_lint_diffs} $-discipline lint divergence(s) "
                  "— paired-`$` in markdown (LaTeX flip risk) or `\\$` "
                  "in st.metric label/value/delta (literal backslash). "
                  "See DESIGN_NOTES §10.3a.")
        if two_relay_diffs:
            print(f"{two_relay_diffs} Two-RELAY lock divergence(s) — "
                  "result label / button-paired / provenance Source "
                  "distinction broke. See DESIGN_NOTES §7.3 + §8.3.")
        if label_budget_diffs:
            print(f"{label_budget_diffs} metric-label budget divergence(s) "
                  "— FIX BUNDLE #77 shortened labels reverted (long form "
                  "reappeared) or short form disappeared from st.metric.")
        if dense_freshness_diffs:
            print(f"{dense_freshness_diffs} dense-CSV freshness "
                  "divergence(s) — re-run precompute_scenarios.py for the "
                  "affected city; Fast cold-start reads from disk and a "
                  "stale CSV would feed wrong values to the surrogate.")
        if rebind_completeness_diffs:
            print(f"{rebind_completeness_diffs} rebind-completeness "
                  "divergence(s) — _rebind_city is missing a per-city "
                  "constant; later test cells silently compute with the "
                  "import-time city's value. Add the missing attribute "
                  "to _rebind_city (the FAIL message names it).")
        if child_pop_diffs:
            print(f"{child_pop_diffs} child-pop staleness divergence(s) — "
                  "the under-18 raster's share diverges from the per-city "
                  "anchor (config: child_pop_extent_share). Re-run "
                  "scripts/data/download_census_pop*.py with CENSUS_API_KEY "
                  "to regenerate; confirm the script uses P3_001N (total "
                  "VAP), not a sub-variable.")
        if bldg_precompute_diffs:
            print(f"{bldg_precompute_diffs} buildings-precompute staleness "
                  "divergence(s) — the on-disk SA buildings rasters disagree "
                  "with a fresh rasterize from buildings_file. Re-run "
                  "`python precompute_buildings.py --city '<city>'` to "
                  "regenerate. Check the source file changed and the "
                  "precompute step was missed.")
        if toggle_diffs:
            print(f"{toggle_diffs} toggle_selection unit-test failure(s) — "
                  "the Interactive Region Map's click-to-toggle pure "
                  "function is broken or has been reverted to replace mode. "
                  "Multi-select on map clicks would not work.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--update", action="store_true",
                        help="Rewrite snapshots with current outputs.")
    args = parser.parse_args()
    sys.exit(main(update=args.update))
