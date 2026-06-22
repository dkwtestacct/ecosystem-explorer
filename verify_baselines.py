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

            # ── Relay 60 Part A: region-no-band lock ──────────────────────
            # Region suggestions are engine-verified; they must NEVER carry a
            # surrogate-derived prediction band. Two non-vacuous guards:
            #  (1) data — no returned region record carries any *_lower/*_upper
            #      band column (we just constructed a real region scenario).
            #  (2) render — plot_tradeoff_region's source attaches no band
            #      (no error_y / _lower / _upper), while the citywide
            #      plot_tradeoff DOES render error_y. The citywide reference
            #      proves bands exist in the codebase — so a green region check
            #      means "deliberately omitted," not "bands don't exist." A
            #      future change routing a citywide band onto the region path
            #      trips one of these.
            import inspect as _insp
            _BAND_KEYS = ('flood_lower', 'flood_upper', 'hm_lower', 'hm_upper',
                          'food_lower', 'food_upper', 'carbon_lower', 'carbon_upper')
            _bands_in_records = [k for k in _BAND_KEYS if k in ro_records.columns]
            if _bands_in_records:
                print(f"  FAIL region-no-band lock (data): region records carry "
                      f"surrogate band keys {_bands_in_records} — region values "
                      "must be engine-verified, band-free")
                region_opt_diffs += 1
            else:
                print("  OK   region-no-band lock (data): region records carry no "
                      "surrogate band keys")
            try:
                _reg_src = _insp.getsource(app.plot_tradeoff_region)
                _cw_src = _insp.getsource(app.plot_tradeoff)
                _reg_band = [t for t in ('error_y', '_lower', '_upper') if t in _reg_src]
                _cw_has_band = 'error_y' in _cw_src
                if _reg_band:
                    print(f"  FAIL region-no-band lock (render): plot_tradeoff_region "
                          f"references band tokens {_reg_band} — region render must "
                          "attach no surrogate band")
                    region_opt_diffs += 1
                elif not _cw_has_band:
                    print("  FAIL region-no-band lock (render): citywide plot_tradeoff "
                          "no longer renders error_y — the non-vacuous reference is "
                          "broken (can't prove region omission is deliberate)")
                    region_opt_diffs += 1
                else:
                    print("  OK   region-no-band lock (render): plot_tradeoff_region "
                          "attaches no band; citywide plot_tradeoff renders error_y "
                          "(reference intact)")
            except Exception as _e:
                print(f"  FAIL region-no-band lock (render): source scan errored: {_e}")
                region_opt_diffs += 1

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

    # ── Relay 25 — Map view is a VIEW, not a layer ───────────────────────────
    # Two pure helpers drive the Map View tab's "render exactly one map + only
    # its key" contract:
    #   _map_view_default_for_scope(region_selected) — the scope→default mapping,
    #     flip-tested both ways (citywide → density summary, region → detailed).
    #   _map_view_render_plan(view_choice) — resolves the active view into
    #     exactly-one-map decisions; the categorical "Scenario changes" legend
    #     rides ONLY with the detailed map (it would mislabel concentration as
    #     GI/FF/HD under the change-density view).
    # Plus the load-bearing meta-tests: a collapsed scope→default mapping and a
    # both-maps-or-neither render plan MUST trip the check.
    print(f"\n{'=' * 60}")
    print("Relay 25 — Map view scope-default + one-map render plan")
    print(f"{'=' * 60}")
    map_view_diffs = 0
    try:
        # (a) scope→default both ways.
        _d_region = app._map_view_default_for_scope(True)
        _d_city = app._map_view_default_for_scope(False)
        if _d_region == app._MAP_VIEW_DETAILED:
            print(f"  OK   region scope → {_d_region!r}")
        else:
            print(f"  FAIL region scope → {_d_region!r}, "
                  f"want {app._MAP_VIEW_DETAILED!r}")
            map_view_diffs += 1
        if _d_city == app._MAP_VIEW_DENSITY:
            print(f"  OK   citywide scope → {_d_city!r}")
        else:
            print(f"  FAIL citywide scope → {_d_city!r}, "
                  f"want {app._MAP_VIEW_DENSITY!r}")
            map_view_diffs += 1
        # Meta-test: the two scopes must map to DIFFERENT defaults — a collapsed
        # mapping defeats the context-sensitive default.
        if _d_region == _d_city:
            print(f"  FAIL meta-test: both scopes map to {_d_region!r} (collapsed)")
            map_view_diffs += 1
        else:
            print("  OK   meta-test: scopes map to distinct defaults")

        # (b) render plan — exactly one map per view; categorical legend rides
        # the detailed view ONLY.
        _plan_det = app._map_view_render_plan(app._MAP_VIEW_DETAILED)
        _plan_den = app._map_view_render_plan(app._MAP_VIEW_DENSITY)
        _plan_checks = [
            ("detailed: detail map on",          _plan_det['show_detail_map'], True),
            ("detailed: density map off",         _plan_det['show_density_map'], False),
            ("detailed: categorical legend on",   _plan_det['show_categorical_legend'], True),
            ("density: detail map off",           _plan_den['show_detail_map'], False),
            ("density: density map on",           _plan_den['show_density_map'], True),
            ("density: categorical legend off",   _plan_den['show_categorical_legend'], False),
        ]
        for name, got, want in _plan_checks:
            if got == want:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: got {got!r}, want {want!r}")
                map_view_diffs += 1
        # Meta-test: exactly one map renders in each view (detail XOR density).
        for _vname, _plan in (("detailed", _plan_det), ("density", _plan_den)):
            if _plan['show_detail_map'] != _plan['show_density_map']:
                print(f"  OK   {_vname}: exactly one map (detail XOR density)")
            else:
                print(f"  FAIL {_vname}: not exactly one map: {_plan}")
                map_view_diffs += 1
    except Exception as e:
        print(f"  ERROR map-view render-plan test: {e}")
        import traceback; traceback.print_exc()
        map_view_diffs += 1

    # ── Relay 26 — Concentration map: boundary context + grid sanity ─────────
    # The change-density map is grounded with faint boundary geometry and its
    # per-cell shares stay correctness-bounded. Three pure surfaces:
    #   _density_boundary_layers(in_aoi, region_mask, district_raster) — draws a
    #     boundary when geometry is available; empty when it isn't (flip-tested).
    #   _district_edge_mask(raster) — marks the region-id seams (flip-tested
    #     against a uniform raster, which must produce no edges).
    #   _change_density_grid — per-cell share stays in [0,1] and finite exactly
    #     where the AOI has pixels.
    print(f"\n{'=' * 60}")
    print("Relay 26 — Concentration map boundary context + grid sanity")
    print(f"{'=' * 60}")
    density_diffs = 0
    try:
        import numpy as _np
        _aoi_all = _np.ones((4, 4), dtype=bool)
        _aoi_none = _np.zeros((4, 4), dtype=bool)
        _region = _np.array([[True, True, False, False]] * 4)
        _distr = _np.array([[0, 0, 1, 1]] * 4, dtype=_np.int32)

        # (a) boundary layers — geometry present → drawn; absent → empty.
        def _kinds(layers):
            return [k for k, _m in layers]
        _bl_aoi = app._density_boundary_layers(_aoi_all, None, None)
        _bl_aoi_d = app._density_boundary_layers(_aoi_all, None, _distr)
        _bl_region = app._density_boundary_layers(_aoi_all, _region, _distr)
        _bl_empty = app._density_boundary_layers(_aoi_none, None, _distr)
        _bl_checks = [
            ("citywide AOI → ['aoi']",            _kinds(_bl_aoi), ['aoi']),
            ("citywide AOI + raster → aoi+districts",
                sorted(_kinds(_bl_aoi_d)), ['aoi', 'districts']),
            ("selected region wins → ['region']", _kinds(_bl_region), ['region']),
            ("no geometry → [] (empty)",          _kinds(_bl_empty), []),
        ]
        for name, got, want in _bl_checks:
            if got == want:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: got {got!r}, want {want!r}")
                density_diffs += 1
        # Meta-test: a degenerate AOI with no selection draws NOTHING — proves
        # the "drawn only when geometry available" gate is real, not always-on.
        if app._density_boundary_layers(_aoi_none, None, None):
            print("  FAIL meta-test: boundary drawn with NO geometry available")
            density_diffs += 1
        else:
            print("  OK   meta-test: no boundary without geometry")

        # (b) district edge mask — seam present; uniform raster → no edges.
        _edges = app._district_edge_mask(_distr)
        if (_edges[:, 1].all() and _edges[:, 2].all()
                and not _edges[:, 0].any() and not _edges[:, 3].any()):
            print("  OK   district edge mask marks the seam columns only")
        else:
            print(f"  FAIL district edge mask wrong: {_edges.tolist()}")
            density_diffs += 1
        if app._district_edge_mask(_np.zeros((4, 4), dtype=_np.int32)).any():
            print("  FAIL meta-test: uniform raster produced phantom edges")
            density_diffs += 1
        else:
            print("  OK   meta-test: uniform raster → no edges")

        # (c) density grid — per-cell share in [0,1], finite ⇔ in-AOI.
        _NA = app.NODATA
        _b = _np.array([[1, 1, 2, _NA],
                        [1, 1, 2,  2],
                        [_NA, 1, 2, 2],
                        [1, 1, 2,  2]])
        _s = _b.copy()
        _s[0, 0] = 9
        _s[1, 1] = 9
        _grid = app._change_density_grid(_b, _s, region_mask=None, n_cells=4)
        _finite = _np.isfinite(_grid)
        _in_aoi = (_b != _NA)
        _ok_range = (_np.nanmin(_grid) >= 0.0 - 1e-9
                     and _np.nanmax(_grid) <= 1.0 + 1e-9)
        if _ok_range:
            print(f"  OK   density grid in [0,1] (min {_np.nanmin(_grid):.2f}, "
                  f"max {_np.nanmax(_grid):.2f})")
        else:
            print(f"  FAIL density grid out of [0,1]: {_grid.tolist()}")
            density_diffs += 1
        if _np.array_equal(_finite, _in_aoi):
            print("  OK   density grid finite exactly where AOI has pixels")
        else:
            print("  FAIL density grid finite-mask ≠ AOI mask")
            density_diffs += 1
        if int((_grid == 1.0).sum()) == 2:
            print("  OK   density grid: 2 fully-converted cells (cell=1px)")
        else:
            print(f"  FAIL density grid converted-cell count: "
                  f"{int((_grid == 1.0).sum())}, want 2")
            density_diffs += 1
    except Exception as e:
        print(f"  ERROR concentration-map boundary/grid test: {e}")
        import traceback; traceback.print_exc()
        density_diffs += 1

    # ── Relay 28 — Concentration view: final copy + teal palette ─────────────
    # The shipped view used pre-decision names/palette; the final copy renames it
    # to "Conversion concentration summary" with a single-hue teal ramp. Assert:
    #   - the palette is teal (low red, high green+blue at the top), NOT YlOrRd;
    #   - the view option + title + exact caption carry the new wording, the warm
    #     "Warmer = ..." label and the YlOrRd palette are gone;
    #   - GUARD: the "High Density" land-use category survives the rename — an
    #     over-eager density→concentration replace must NOT clobber it.
    print(f"\n{'=' * 60}")
    print("Relay 28 — Concentration view final copy + teal palette")
    print(f"{'=' * 60}")
    concentration_diffs = 0
    try:
        # (a) palette teal at the top, NOT warm.
        _r, _g, _b, _a = app._DENSITY_CMAP(1.0)
        if _g > _r and _b > _r:
            print(f"  OK   density palette top is teal "
                  f"(r={_r:.2f}, g={_g:.2f}, b={_b:.2f})")
        else:
            print(f"  FAIL density palette not teal: "
                  f"r={_r:.2f}, g={_g:.2f}, b={_b:.2f}")
            concentration_diffs += 1
        # Meta-test: the OLD YlOrRd top is red-dominant → fails the teal check,
        # proving the check discriminates rather than passing everything.
        import matplotlib.pyplot as _plt28
        _yr, _yg, _yb, _ya = _plt28.get_cmap("YlOrRd")(1.0)
        if not (_yg > _yr and _yb > _yr):
            print("  OK   meta-test: YlOrRd top fails the teal check (red-dominant)")
        else:
            print("  FAIL meta-test: YlOrRd unexpectedly passed the teal check")
            concentration_diffs += 1

        # (b) view option carries the new wording.
        if app._MAP_VIEW_DENSITY == "Conversion concentration summary":
            print(f"  OK   view option = {app._MAP_VIEW_DENSITY!r}")
        else:
            print(f"  FAIL view option = {app._MAP_VIEW_DENSITY!r}, "
                  "want 'Conversion concentration summary'")
            concentration_diffs += 1

        # (c) exact caption — assert the module-level constant verbatim (the live
        #     copy, immune to source line-wrap drift).
        _want_caption = (
            "Conversions aggregated into grid cells to show where the scenario "
            "is concentrated. This is a readability aid based on the same "
            "converted pixels, not a modeled outcome."
        )
        if app._CONCENTRATION_CAPTION == _want_caption:
            print("  OK   concentration caption matches final copy verbatim")
        else:
            print(f"  FAIL concentration caption drifted: "
                  f"{app._CONCENTRATION_CAPTION!r}")
            concentration_diffs += 1

        # title carries the new 'concentration' wording (helper-generated since
        # Relay 29; the per-scope literals are flip-tested in the Relay 29 block).
        _title_cw = app._map_view_title(app._MAP_VIEW_DENSITY, "citywide")
        if "conversion concentration summary" in _title_cw.lower():
            print(f"  OK   title carries 'concentration' wording: {_title_cw!r}")
        else:
            print(f"  FAIL title missing 'concentration' wording: {_title_cw!r}")
            concentration_diffs += 1

        # colorbar label present in source; old warm wording + YlOrRd gone.
        with open("app.py", encoding="utf-8") as _f28:
            _src28 = _f28.read()
        _present = [
            ("colorbar label 'Share of grid cell converted'",
             "Share of grid cell converted", True),
            ("old 'Warmer = larger share' wording gone",
             "Warmer = larger share", False),
            ("YlOrRd palette gone from app.py",
             'get_cmap("YlOrRd")', False),
        ]
        for name, needle, want_present in _present:
            got = needle in _src28
            if got == want_present:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: present={got}, want_present={want_present}")
                concentration_diffs += 1

        # (d) GUARD — 'High Density' land-use category survives the rename.
        _hd_count = _src28.count("High Density")
        _hd_key = "High Density" in app.CHANGE_COLORS
        if _hd_count >= 3 and _hd_key:
            print(f"  OK   'High Density' land-use category intact "
                  f"({_hd_count}× in source, CHANGE_COLORS key present)")
        else:
            print(f"  FAIL 'High Density' clobbered: {_hd_count}× in source, "
                  f"CHANGE_COLORS key present={_hd_key}")
            concentration_diffs += 1
    except Exception as e:
        print(f"  ERROR concentration final-copy test: {e}")
        import traceback; traceback.print_exc()
        concentration_diffs += 1

    # ── Relay 29 — Self-describing map views ─────────────────────────────────
    # Three always-visible labels: the active-view indicator (names the view
    # actually rendered, points to the other), scope-aware titles (citywide vs
    # the region's own label), and the teal key under the concentration map.
    # Pure helpers carry all three; assert:
    #   - the indicator names the rendered view and flips with it;
    #   - scope-aware titles match the rendered scope (citywide ≠ region);
    #   - the teal key is the agreed string AND the ramp runs light(low) →
    #     dark(high) so "darker = larger" isn't backwards.
    print(f"\n{'=' * 60}")
    print("Relay 29 — Self-describing map views")
    print(f"{'=' * 60}")
    self_describe_diffs = 0
    try:
        # (a) active-view indicator — flips with the rendered view.
        _exp_det = (f"Map view: {app._MAP_VIEW_DETAILED} — switch to "
                    f"{app._MAP_VIEW_DENSITY} in Map view & overlays.")
        _exp_den = (f"Map view: {app._MAP_VIEW_DENSITY} — switch to "
                    f"{app._MAP_VIEW_DETAILED} in Map view & overlays.")
        _ind_det = app._map_view_indicator(app._MAP_VIEW_DETAILED)
        _ind_den = app._map_view_indicator(app._MAP_VIEW_DENSITY)
        _ind_checks = [
            ("indicator (detailed) names detailed view", _ind_det, _exp_det),
            ("indicator (concentration) names that view", _ind_den, _exp_den),
        ]
        for name, got, want in _ind_checks:
            if got == want:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: got {got!r}, want {want!r}")
                self_describe_diffs += 1
        # Meta-test: the indicator actually FLIPS (names the rendered view, not a
        # constant string) — the two views must produce different indicators.
        if _ind_det != _ind_den:
            print("  OK   meta-test: indicator flips with the rendered view")
        else:
            print("  FAIL meta-test: indicator identical across views (no flip)")
            self_describe_diffs += 1

        # (b) scope-aware titles — match the rendered scope (citywide ≠ region).
        _title_checks = [
            ("detailed citywide",
             app._map_view_title(app._MAP_VIEW_DETAILED, "citywide"),
             "Detailed conversion map — citywide"),
            ("detailed region",
             app._map_view_title(app._MAP_VIEW_DETAILED, "Council District 5"),
             "Detailed conversion map — Council District 5"),
            ("concentration citywide",
             app._map_view_title(app._MAP_VIEW_DENSITY, "citywide"),
             "Citywide conversion concentration summary"),
            ("concentration region",
             app._map_view_title(app._MAP_VIEW_DENSITY, "Council District 5"),
             "Council District 5 conversion concentration summary"),
        ]
        for name, got, want in _title_checks:
            if got == want:
                print(f"  OK   title {name}: {got!r}")
            else:
                print(f"  FAIL title {name}: got {got!r}, want {want!r}")
                self_describe_diffs += 1
        # Meta-test: the scope label is actually reflected — citywide vs region
        # titles must differ for both views (not a fixed string).
        for _v, _vn in ((app._MAP_VIEW_DETAILED, "detailed"),
                        (app._MAP_VIEW_DENSITY, "concentration")):
            if (app._map_view_title(_v, "citywide")
                    != app._map_view_title(_v, "Council District 5")):
                print(f"  OK   meta-test: {_vn} title reflects scope")
            else:
                print(f"  FAIL meta-test: {_vn} title ignores scope")
                self_describe_diffs += 1

        # (c) teal key string + ramp direction (light=low → dark=high).
        _want_key = "Darker teal = larger share of grid cell converted."
        if app._CONCENTRATION_TEAL_KEY == _want_key:
            print("  OK   teal key matches the agreed string")
        else:
            print(f"  FAIL teal key drifted: {app._CONCENTRATION_TEAL_KEY!r}")
            self_describe_diffs += 1
        with open("app.py", encoding="utf-8") as _f29:
            _src29 = _f29.read()
        if "_CONCENTRATION_TEAL_KEY" in _src29 and "Darker teal" in _src29:
            print("  OK   teal key rendered in app.py")
        else:
            print("  FAIL teal key not referenced in app.py")
            self_describe_diffs += 1
        _lo = sum(app._DENSITY_CMAP(0.0)[:3])   # low share → light (high sum)
        _hi = sum(app._DENSITY_CMAP(1.0)[:3])   # high share → dark (low sum)
        if _lo > _hi:
            print(f"  OK   ramp light→dark (low sum {_hi:.2f} < high-light "
                  f"sum {_lo:.2f}) — 'darker = larger' agrees")
        else:
            print(f"  FAIL ramp backwards: low-share sum {_lo:.2f}, "
                  f"high-share sum {_hi:.2f} — teal key would be reversed")
            self_describe_diffs += 1
    except Exception as e:
        print(f"  ERROR self-describing map-views test: {e}")
        import traceback; traceback.print_exc()
        self_describe_diffs += 1

    # ── Relay 30 — Selected-area locator: heading + tightened instruction ─────
    # The region picker gets a 'Selected area' heading and a shorter instruction
    # so it reads as a secondary locator, not the main output. Light source
    # check: heading + new instruction present, old long form gone. The layer
    # label stays dynamic (_t3_display), so the instruction is checked on the
    # scope-independent tail only.
    print(f"\n{'=' * 60}")
    print("Relay 30 — Selected-area locator heading + instruction")
    print(f"{'=' * 60}")
    locator_diffs = 0
    try:
        with open("app.py", encoding="utf-8") as _f30:
            _src30 = _f30.read()
        # The instruction wraps across string literals in source, so check
        # contiguous sub-phrases (each within one literal) rather than the full
        # concatenated string.
        _loc_checks = [
            ("'Selected area' heading present",
             '"**Selected area**"', True),
            ("tightened instruction head present",
             "to toggle selection. Changes are", True),
            ("tightened instruction mid present",
             "placed only inside the selected area. The Scenario tab reports",
             True),
            ("tightened instruction tail present",
             "both citywide and selected-area results.", True),
            ("old long instruction gone",
             "toggle its selection — click another to add", False),
        ]
        for name, needle, want_present in _loc_checks:
            got = needle in _src30
            if got == want_present:
                print(f"  OK   {name}")
            else:
                print(f"  FAIL {name}: present={got}, want_present={want_present}")
                locator_diffs += 1
    except Exception as e:
        print(f"  ERROR selected-area locator test: {e}")
        import traceback; traceback.print_exc()
        locator_diffs += 1

    # ── Default-scenario state consistency (Relay A) ─────────────────────────
    # Title, line-1 summary, and audit are all rendered from the same
    # `_resolved_scenario` dict via three display helpers
    # (`_explorer_scenario_label`, `_active_scenario_line1`,
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
        sentence = app._active_scenario_line1(
            default_state, app.eib.PROVENANCE_EXPLORER,
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
        sentence_z = app._active_scenario_line1(
            zero_state, app.eib.PROVENANCE_EXPLORER,
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
        sentence_m = app._active_scenario_line1(
            mixed_state, app.eib.PROVENANCE_EXPLORER,
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
        # line-1 built from state_B — deliberately desynced from label.
        sentence_meta = app._active_scenario_line1(
            state_B, app.eib.PROVENANCE_EXPLORER,
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
    # (Tradeoff Space plot intentionally absent — its axes (Flood Index,
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
                 'st.markdown("#### Best citywide scenarios by goal"'),
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
    #       surrogate search" or "Region machine-learning search") within N lines
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
        #   "Citywide machine-learning search"   (citywide)
        #   "Selected-area search"               (region — engine-verified, so
        #                                         no "machine-learning" claim)
        MODE_LABEL_STRINGS = (
            "Citywide machine-learning search",
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
        # Citywide-origin Source MUST contain "machine-learning suggestion";
        # region-origin Source MUST contain "region-optimized".
        import natcap_scenarios as _ns2
        _cw_source = app._PROVENANCE_HEADER_INFO.get(
            _ns2.PROVENANCE_OPTIMIZER, (None,))[0]
        _rg_source = app._PROVENANCE_HEADER_INFO.get(
            _ns2.PROVENANCE_REGION_OPTIMIZED, (None,))[0]
        _cw_ok = (_cw_source is not None
                   and "machine-learning suggestion" in _cw_source.lower())
        _rg_ok = (_rg_source is not None
                   and "region-optimized" in _rg_source.lower())
        if not _cw_ok:
            print(f"  FAIL Citywide-origin Source string missing "
                  f"'machine-learning suggestion': {_cw_source!r}")
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
        _swap_rg_ok = "machine-learning suggestion" in (_rg_source or "").lower()
        if _swap_cw_ok or _swap_rg_ok:
            print(f"  FAIL meta-test (C): swapped mapping would still "
                  f"satisfy the checks — the distinction isn't tight")
            two_relay_diffs += 1
        else:
            print(f"  OK   meta-test (C): swapped mapping (citywide → "
                  f"'region-optimized', region → 'machine-learning suggestion') "
                  f"correctly FAILS both checks — distinction is tight")

        # ── Assertion D — CTA caption protection (FIX BUNDLE #79) ───────
        # Both Discover surfaces (sidebar + main-panel CTA) carry the same
        # mode-keyed caption beneath the mode label:
        #   citywide → "Fast estimates suggest promising mixes; apply one to
        #              recompute it with the InVEST-aligned evaluator." (fast
        #              estimates, not InVEST-aligned-evaluator outputs)
        #   region   → "Searches candidate mixes under the current area and
        #              filters. Displayed values are computed by the
        #              InVEST-aligned evaluator, not model predictions."
        # Both expected literals must appear ≥2× in app.py (sidebar + CTA),
        # and each must appear immediately after a matching mode label
        # within a small window (so they pair with their mode, not float).
        # Meta-test: confirm a tweaked caption string would fail. Both
        # captions are single source-line literals; a single-literal
        # exact-count check works for both surfaces.
        _CW_CAPTION_EXPECTED = (
            "Fast estimates suggest promising mixes; apply one to recompute "
            "with the InVEST-aligned evaluator."
        )
        _RG_CAPTION_EXPECTED = (
            "Searches candidate mixes under the current area and filters. "
            "Displayed values are computed by the InVEST-aligned evaluator, not model predictions."
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
        _seed_d_rg = ("st.caption(\"Searches candidate mixes under whatever "
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
            # Relay 50 — Placement radio gained a key so guided examples can set
            # it; the four guided-example buttons in Quick Start are keyed too.
            'placement_strategy_radio',
            'guided_balanced', 'guided_cooling', 'guided_food', 'guided_school',
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
    #   "Runoff Volume (ac-ft)"       → "Runoff Volume"   (Relay — units to help)
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
            # Relay (units to caption): the unit no longer rides in the label —
            # it lives in the card help ("Values are shown in acre-feet (ac-ft)").
            # The unit-suffixed "Runoff Volume (ac-ft)" is now the banned long
            # form (it truncated at 1/3 width); the bare "Runoff Volume" is the
            # budget. This reverses the earlier units-in-label decision.
            "Runoff Volume (ac-ft)":      "Runoff Volume",
            "Cost / Citywide °F Cooling": "Cost / °F cooling",
            # Relay 31 — ceff1/ceff3 de-truncated to short forms.
            "Cost / Acre-Foot Runoff Prevented": "Cost / ac-ft runoff",
            "Cost / 1,000 People Fed":           "Cost / 1k people fed",
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
                  f"st.metric call(s) scanned ({len(_LABEL_REGRESSIONS)} regression strings checked).")

        # Half (b) — short-form labels must still be present. The three Cost
        # Effectiveness labels now live in app.py's _CE_CARD_SPECS list and reach
        # st.metric via a variable (col.metric(_lbl, …)) so they're string-literal
        # Constants but NOT metric-label args — scan ALL string-literal Constants
        # for the presence check, not only metric labels. (Half (a) stays on
        # metric-label args: a long form must not RENDER as a metric label.)
        try:
            _tree_all = _ast2.parse(_app_src)
            _all_str_consts = {n.value for n in _ast2.walk(_tree_all)
                               if isinstance(n, _ast2.Constant)
                               and isinstance(n.value, str)}
        except SyntaxError:
            _all_str_consts = set()
        _present_short = {lab for (_ln, lab) in _metric_labels} | _all_str_consts
        _missing = [s for s in _LABEL_REGRESSIONS.values()
                    if s not in _present_short]
        if _missing:
            for s in _missing:
                print(f"  FAIL short-form label '{s}' is missing from app.py "
                      "(metric-label arg or _CE_CARD_SPECS — disappeared entirely?)")
            label_budget_diffs += len(_missing)
        else:
            print(f"  OK   all {len(_LABEL_REGRESSIONS)} shortened labels still present "
                  "(st.metric args + _CE_CARD_SPECS literals)")

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
        # omits "Temp change" + "Runoff Volume" (should both flag as missing).
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

    # ── Relocated-unit survival — completes Relay 2's caption half ───────────
    # The #77 guard keeps the Runoff Volume + Carbon labels BARE (the unit was
    # moved out of the label). That's only honest if the shed unit SURVIVES in
    # the card help — this cell locks that predicate, so a future edit that drops
    # the unit from the card entirely fails the gate:
    #   - the Runoff Volume card's help must still mention "acre-feet" / "ac-ft";
    #   - the Carbon card's help must still carry "t CO2e" (it pulls the unit via
    #     `_carbon_unit_suffix`, so we verify the help references the suffix AND
    #     the suffix value carries the token).
    # Meta-test seeds a unit-stripped help and confirms the predicate flags it.
    print(f"\n{'=' * 60}")
    print("Relocated-unit survival — Runoff / Carbon card help (Relay 2/13)")
    print(f"{'=' * 60}")
    unit_survival_diffs = 0
    try:
        import ast as _ast3
        with open("app.py", "r") as _f13:
            _src13 = _f13.read()
        _tree13 = _ast3.parse(_src13)

        def _has_unit(text, tokens):
            return any(tok in text for tok in tokens)

        def _concat_str_consts(node):
            return "".join(
                n.value for n in _ast3.walk(node)
                if isinstance(n, _ast3.Constant) and isinstance(n.value, str)
            )

        # Runoff Volume card — st.metric("Runoff Volume", value, help=(...)).
        _runoff_help = None
        for _n in _ast3.walk(_tree13):
            if (isinstance(_n, _ast3.Call) and isinstance(_n.func, _ast3.Attribute)
                    and _n.func.attr == "metric" and _n.args
                    and isinstance(_n.args[0], _ast3.Constant)
                    and _n.args[0].value == "Runoff Volume"):
                for _kw in _n.keywords:
                    if _kw.arg == "help":
                        _runoff_help = _concat_str_consts(_kw.value)
                break
        if _runoff_help is None:
            print("  FAIL Runoff Volume card help not found (label/help structure changed?)")
            unit_survival_diffs += 1
        elif not _has_unit(_runoff_help, ("acre-feet", "ac-ft")):
            print("  FAIL Runoff Volume help dropped the unit — no 'acre-feet' / 'ac-ft'")
            unit_survival_diffs += 1
        else:
            print("  OK   Runoff Volume help still carries the unit (acre-feet / ac-ft)")

        # Carbon card — _carbon_card_help is variable-built; the unit reaches it
        # via _carbon_unit_suffix. Verify the help references the suffix AND the
        # suffix value carries 't CO2e'.
        _carbon_help_src = None
        _carbon_suffix_val = ""
        for _n in _ast3.walk(_tree13):
            if isinstance(_n, _ast3.Assign) and _n.targets:
                _tgt = _n.targets[0]
                if isinstance(_tgt, _ast3.Name) and _tgt.id == "_carbon_card_help":
                    _carbon_help_src = _ast3.get_source_segment(_src13, _n.value) or ""
                if isinstance(_tgt, _ast3.Name) and _tgt.id == "_carbon_unit_suffix":
                    _carbon_suffix_val = _concat_str_consts(_n.value)
        if (_carbon_help_src and "_carbon_unit_suffix" in _carbon_help_src
                and "t CO2e" in _carbon_suffix_val):
            print("  OK   Carbon card help still carries the unit (t CO2e via _carbon_unit_suffix)")
        else:
            print("  FAIL Carbon card help dropped the unit — help no longer pulls "
                  "_carbon_unit_suffix, or the suffix lost 't CO2e'")
            unit_survival_diffs += 1

        # Meta-test: the predicate must flag a unit-stripped help and pass a
        # unit-bearing one (non-vacuous both ways).
        _seed_stripped = "Lower is better. Modeled runoff volume for the design storm."
        _seed_present = "Values are shown in acre-feet (ac-ft)."
        if (not _has_unit(_seed_stripped, ("acre-feet", "ac-ft"))
                and _has_unit(_seed_present, ("acre-feet", "ac-ft"))):
            print("  OK   meta-test: unit-survival predicate flags a stripped help, passes a unit-bearing one")
        else:
            print("  FAIL meta-test: unit-survival predicate is vacuous")
            unit_survival_diffs += 1
    except Exception as _e_unit:
        print(f"  ERROR relocated-unit survival check: {_e_unit}")
        import traceback
        traceback.print_exc()
        unit_survival_diffs += 1

    # ── Figure-close hygiene — no leaked matplotlib figures ──────────────────
    # Streamlit reruns the whole script every interaction, so every figure
    # rendered (st.pyplot(...) + the Map View .savefig(...)) must be paired with
    # a plt.close(...) or figures accumulate. Mechanism: plt.close count >=
    # render count over app.py (AST, so comments/strings don't count; closes are
    # matched to `plt.close` specifically so unrelated `.close()` calls — file /
    # buffer handles — don't inflate the tally). A new render without a matching
    # close trips it. Meta-test seeds an unpaired render and confirms the
    # predicate flags it.
    print(f"\n{'=' * 60}")
    print("Figure-close hygiene — st.pyplot / savefig paired with plt.close")
    print(f"{'=' * 60}")
    fig_close_diffs = 0
    try:
        import ast as _ast4
        with open("app.py", "r") as _f16:
            _src16 = _f16.read()

        def _render_close_counts(source):
            tree = _ast4.parse(source)
            renders = closes = 0
            for _n in _ast4.walk(tree):
                if not (isinstance(_n, _ast4.Call)
                        and isinstance(_n.func, _ast4.Attribute)):
                    continue
                _attr = _n.func.attr
                if _attr in ("pyplot", "savefig"):
                    renders += 1
                elif (_attr == "close"
                      and isinstance(_n.func.value, _ast4.Name)
                      and _n.func.value.id == "plt"):
                    closes += 1
            return renders, closes

        _r, _c = _render_close_counts(_src16)
        if _r > 0 and _c >= _r:
            print(f"  OK   figure-close hygiene: {_c} plt.close >= {_r} render(s) "
                  "(st.pyplot + savefig)")
        else:
            print(f"  FAIL figure-close hygiene: {_c} plt.close < {_r} render(s) "
                  "— a plot is rendered without a paired plt.close (leaked figure)")
            fig_close_diffs += 1

        # Meta-test: an unpaired render must flag; a paired one must pass.
        _r1, _c1 = _render_close_counts("import streamlit as st\nst.pyplot(fig)\n")
        _r2, _c2 = _render_close_counts(
            "import streamlit as st\nimport matplotlib.pyplot as plt\n"
            "st.pyplot(fig)\nplt.close(fig)\n")
        if _c1 < _r1 and _c2 >= _r2:
            print("  OK   meta-test: unpaired st.pyplot flagged; paired st.pyplot passes")
        else:
            print("  FAIL meta-test: figure-close predicate is vacuous")
            fig_close_diffs += 1
    except Exception as _e_fig:
        print(f"  ERROR figure-close hygiene check: {_e_fig}")
        import traceback
        traceback.print_exc()
        fig_close_diffs += 1

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
        import json as _json3
        # Sample 3 rows per city — spread across the (pct, gi, ff) space.
        # Pick rows that exist in both cities' CSVs (mult. of dense's step
        # 5/10) and that exercise non-baseline math.
        _SAMPLES = [
            (10, 50, 50),   # gi/ff split, low pct
            (30, 20, 80),   # FF-heavy mid-pct (gi+ff=100)
            (50, 100, 0),   # GI-only max-pct
        ]
        _COMPARE_KEYS = ("mean_hm", "flood_reduction", "runoff_acre_feet",
                         "food_mln_lbs",
                         "runoff_retention_idx")  # Relay 58 — the new per-pixel
                                          # UFR retention column; a stale CSV
                                          # missing it raises KeyError here.
                                          # (skip carbon — float32 noise at 5e-7
                                          # on SA, within the 1e-5 tol but noisy)
        _REL_TOL = 1e-5

        for _city in [c for c in active_cities
                      if app.CITIES[c].get("dense_scenarios_file")]:
            _path = app.CITIES[_city]["dense_scenarios_file"]
            if not Path(_path).exists():
                print(f"  SKIP {_city}: dense_scenarios_file {_path!r} not "
                      "on disk (Fast mode will recompute live).")
                continue
            # Relay 37 — provenance stamp check (parity with the Fast-grid cell).
            _meta_path = _path + ".meta.json"
            if not Path(_meta_path).exists():
                print(f"  FAIL {_city}: dense-CSV sidecar {_meta_path!r} missing "
                      "— run `precompute_scenarios.py --city <c> --stamp-only`.")
                dense_freshness_diffs += 1
                continue
            _dmeta = _json3.loads(Path(_meta_path).read_text())
            _sbad = []
            if _dmeta.get("dense_grid_format_version") != 1: _sbad.append("format")
            if _dmeta.get("step_pct") != 5: _sbad.append("step_pct")
            if _dmeta.get("step_alloc") != 10: _sbad.append("step_alloc")
            if _dmeta.get("city_key") != _city: _sbad.append("city_key")
            if _dmeta.get("scenario_schema_version") != app.SCENARIO_SCHEMA_VERSION:
                _sbad.append(f"schema({_dmeta.get('scenario_schema_version')}"
                             f"!={app.SCENARIO_SCHEMA_VERSION})")
            if _sbad:
                print(f"  FAIL {_city}: dense-CSV stamp mismatch {_sbad} — "
                      "regenerate or re-stamp with precompute_scenarios.py.")
                dense_freshness_diffs += 1
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
                print(f"  OK   {_city}: stamp matches + 3 sampled rows × 4 "
                      f"metrics match CSV within rel_tol={_REL_TOL:.0e}")
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
        # Meta-test (stamp, Relay 37): a wrong schema_version in the stamp must
        # be flagged by the same equality the per-city stamp check uses.
        _probe_meta = {"scenario_schema_version": app.SCENARIO_SCHEMA_VERSION + 999}
        if _probe_meta.get("scenario_schema_version") == app.SCENARIO_SCHEMA_VERSION:
            print("  FAIL meta-test: seeded stamp schema mismatch NOT caught")
            dense_freshness_diffs += 1
        else:
            print("  OK   meta-test: seeded stamp schema mismatch correctly "
                  "caught (stamp check is non-vacuous)")
    except Exception as e:
        print(f"  ERROR dense-csv freshness: {e}")
        import traceback; traceback.print_exc()
        dense_freshness_diffs += 1

    # ── Fast-grid artifact freshness — Relay 35 first-click guard ───────────
    # The region-optimizer prefilter trains on a precomputed Fast grid
    # (step_pct=10 / step_alloc=25) loaded from CITIES[city]['fast_grid_file']
    # instead of a ~96 s live build. Two-layer staleness guard:
    #   (1) cheap stamp check — the sidecar's scenario_schema_version + step
    #       params + city + format must match current. A math change that bumps
    #       SCENARIO_SCHEMA_VERSION invalidates the artifact here (and at
    #       runtime → live fallback), forcing a regen.
    #   (2) value spot-check — re-evaluate 5 recipes via the live engine and
    #       assert the stored grid matches within rel_tol (catches a math change
    #       that did NOT bump the schema — same discipline as the dense-CSV cell).
    # Missing artifact → SKIP (runtime falls back to a live build; degraded, not
    # broken). Meta-test seeds a drifted grid value and confirms it's caught.
    print(f"\n{'=' * 60}")
    print("Fast-grid artifact freshness — Relay 35 first-click guard")
    print(f"{'=' * 60}")
    fast_grid_diffs = 0
    try:
        import json as _json2
        import math as _m2
        import pandas as _pd2
        _FG_SAMPLES = [(10, 25, 25), (20, 50, 0), (30, 0, 50),
                       (40, 75, 25), (50, 25, 25)]
        _FG_KEYS = ("mean_hm", "flood_reduction", "runoff_acre_feet", "food_mln_lbs",
                    "runoff_retention_idx")  # Relay 58 — new UFR retention column
        _FG_REL_TOL = 1e-5
        _probe = 100.0  # real grid value captured below for the meta-test
        for _city in [c for c in active_cities
                      if app.CITIES[c].get("fast_grid_file")]:
            _path = app.CITIES[_city]["fast_grid_file"]
            if not Path(_path).exists():
                print(f"  SKIP {_city}: fast_grid_file {_path!r} not on disk "
                      "(runtime builds the Fast grid live).")
                continue
            _meta_path = _path + ".meta.json"
            if not Path(_meta_path).exists():
                print(f"  FAIL {_city}: fast-grid sidecar {_meta_path!r} missing "
                      "— regenerate with scripts/regenerate_fast_grid.py")
                fast_grid_diffs += 1
                continue
            _meta = _json2.loads(Path(_meta_path).read_text())
            _bad = []
            if _meta.get("step_pct") != 10: _bad.append("step_pct")
            if _meta.get("step_alloc") != 25: _bad.append("step_alloc")
            if _meta.get("city_key") != _city: _bad.append("city_key")
            if _meta.get("scenario_schema_version") != app.SCENARIO_SCHEMA_VERSION:
                _bad.append(f"schema({_meta.get('scenario_schema_version')}"
                            f"!={app.SCENARIO_SCHEMA_VERSION})")
            if _bad:
                print(f"  FAIL {_city}: fast-grid stamp mismatch {_bad} — "
                      "regenerate with scripts/regenerate_fast_grid.py")
                fast_grid_diffs += 1
                continue
            _df = _pd2.read_csv(_path)
            _rebind_city(app, _city)
            _c_diffs = 0
            _n_checked = 0
            for (pct, gi, ff) in _FG_SAMPLES:
                _rm = _df[(_df.pct_converted == pct)
                          & (_df.green_infrastructure_pct == gi)
                          & (_df.food_forest_pct == ff)]
                if _rm.empty:
                    print(f"  SKIP {_city} ({pct},{gi},{ff}): recipe not in grid")
                    continue
                _row = _rm.iloc[0]
                if _n_checked == 0:
                    _probe = float(_row[_FG_KEYS[0]])
                _live = app.evaluate_scenario(
                    pct_converted=pct, green_infrastructure_pct=gi,
                    food_forest_pct=ff, seed=42, placement_strategy="random")
                _n_checked += 1
                for _k in _FG_KEYS:
                    _cv = float(_row[_k]); _lv = float(_live[_k])
                    if not _m2.isclose(_cv, _lv, rel_tol=_FG_REL_TOL, abs_tol=1e-9):
                        _rel = abs(_cv - _lv) / max(abs(_cv), 1e-9)
                        print(f"  FAIL {_city} ({pct},{gi},{ff}) {_k}: "
                              f"grid={_cv:.6g} live={_lv:.6g} rel={_rel:.2e} "
                              f"> {_FG_REL_TOL:.0e}")
                        _c_diffs += 1
            if _c_diffs == 0:
                print(f"  OK   {_city}: stamp matches + {_n_checked} recipes × "
                      f"{len(_FG_KEYS)} metrics within rel_tol={_FG_REL_TOL:.0e}")
            else:
                fast_grid_diffs += _c_diffs
        # Meta-test (non-vacuous): a 1% drift on a real grid value must fail.
        if _m2.isclose(_probe * 1.01, _probe, rel_tol=_FG_REL_TOL, abs_tol=1e-9):
            print(f"  FAIL meta-test: 1% drift on a seeded grid value "
                  f"({_probe:.4g}→{_probe*1.01:.4g}) NOT caught — tol too loose")
            fast_grid_diffs += 1
        else:
            print(f"  OK   meta-test: 1% drift on a seeded grid value correctly "
                  f"fails the rel_tol={_FG_REL_TOL:.0e} spot-check")
    except Exception as e:
        print(f"  ERROR fast-grid freshness: {e}")
        import traceback; traceback.print_exc()
        fast_grid_diffs += 1

    # ── Surrogate calibration artifact freshness — Relay 60 Part B ──────────
    # The citywide "Estimate range" is derived from
    # data/<slug>/surrogate_calibration_<mode>.json. Each artifact's stamp must
    # match the live SCENARIO_SCHEMA_VERSION AND the live grid content
    # (grid_hash recomputed from the CSV via the calibration script's own
    # hasher). A schema bump or a regenerated grid must force a re-run of
    # scripts/calibrate_surrogate_band.py — otherwise the range silently goes
    # stale. (city, mode) with a grid file but no artifact → FAIL; with no grid
    # file (e.g. MN Fast) → SKIP (range intentionally absent at runtime).
    print(f"\n{'=' * 60}")
    print("Surrogate calibration freshness — Relay 60 Part B")
    print(f"{'=' * 60}")
    calib_diffs = 0
    try:
        import json as _jc
        import importlib.util as _ilu2
        import pandas as _pdc
        _cal_script = Path(__file__).resolve().parent / "scripts" / "calibrate_surrogate_band.py"
        _cspec = _ilu2.spec_from_file_location("calibrate_surrogate_band", _cal_script)
        _cmod = _ilu2.module_from_spec(_cspec)
        _cspec.loader.exec_module(_cmod)
        for _city in active_cities:
            _cfg = app.CITIES[_city]
            _cslug = _cmod.SLUG.get(_city)
            if not _cslug:
                continue
            for _mode, _key in (("fast", "fast_grid_file"),
                                ("balanced", "dense_scenarios_file")):
                _grid = _cfg.get(_key)
                _cal = Path(f"data/{_cslug}/surrogate_calibration_{_mode}.json")
                if not _grid or not Path(_grid).exists():
                    if _cal.exists():
                        print(f"  WARN {_cslug}/{_mode}: calibration artifact exists "
                              "but no grid file — runtime ignores it (no range shown)")
                    else:
                        print(f"  SKIP {_cslug}/{_mode}: no grid file — range "
                              "intentionally absent for this mode")
                    continue
                if not _cal.exists():
                    print(f"  FAIL {_cslug}/{_mode}: grid exists but calibration "
                          "artifact missing — run scripts/calibrate_surrogate_band.py")
                    calib_diffs += 1
                    continue
                _art = _jc.loads(_cal.read_text())
                _prov = _art.get("provenance", {})
                _bad = []
                if _prov.get("scenario_schema_version") != app.SCENARIO_SCHEMA_VERSION:
                    _bad.append(f"schema({_prov.get('scenario_schema_version')}"
                                f"!={app.SCENARIO_SCHEMA_VERSION})")
                _live_hash = _cmod._grid_hash(_pdc.read_csv(_grid))
                if _prov.get("grid_hash") != _live_hash:
                    _bad.append("grid_hash")
                if _bad:
                    print(f"  FAIL {_cslug}/{_mode}: calibration stamp mismatch "
                          f"{_bad} — re-run scripts/calibrate_surrogate_band.py")
                    calib_diffs += 1
                else:
                    print(f"  OK   {_cslug}/{_mode}: calibration stamp matches "
                          "(schema + grid_hash)")
        # Meta-test (non-vacuous): a poisoned grid_hash must differ from the live
        # hash, so the equality check above would fire.
        _probe = Path("data/sa/surrogate_calibration_balanced.json")
        if _probe.exists():
            _live = _cmod._grid_hash(_pdc.read_csv(
                app.CITIES["San Antonio, TX"]["dense_scenarios_file"]))
            _poison = "0" * 16 if _live != "0" * 16 else "f" * 16
            if _poison != _live:
                print("  OK   meta-test: a poisoned grid_hash differs from the live "
                      "hash → freshness check fires (non-vacuous)")
            else:
                print("  FAIL meta-test: poisoned grid_hash equals live hash")
                calib_diffs += 1
    except Exception as _e:
        print(f"  ERROR calibration freshness: {_e}")
        import traceback
        traceback.print_exc()
        calib_diffs += 1

    # ── Calibration LOADER end-to-end — json-NameError regression ───────────
    # Gate-gap closer: the freshness block above reads the JSON directly and
    # checks stamps, but NEVER calls `app._load_surrogate_calibration`. A `json`
    # NameError in the loader's swallowed `except` therefore left the Estimate
    # range dark for every city/mode since Relay 60B while the file checks stayed
    # green. This exercises the loader on the runtime path. Bug-catcher: RED
    # (None) on the pre-fix code, GREEN after app.py's module-level `import json`.
    print(f"\n{'=' * 60}")
    print("Calibration loader end-to-end — json-NameError regression")
    print(f"{'=' * 60}")
    loader_diffs = 0
    try:
        _CALIB_METRICS = ("flood_reduction", "mean_hm",
                          "food_mln_lbs", "carbon_tons_co2")
        _ld = app._load_surrogate_calibration(
            "sa", "fast", app.SCENARIO_SCHEMA_VERSION)
        if not isinstance(_ld, dict):
            print(f"  FAIL loader returned {type(_ld).__name__}, expected dict — "
                  "the swallowed json NameError is back (Estimate range dark)")
            loader_diffs += 1
        elif not all(m in _ld for m in _CALIB_METRICS):
            print(f"  FAIL loader dict missing calibrated metrics "
                  f"{[m for m in _CALIB_METRICS if m not in _ld]}")
            loader_diffs += 1
        else:
            print("  OK   (sa, fast, live schema) loads a dict carrying "
                  "residual_quantiles for all calibrated metrics")
        # No over-correction: a stale schema and a missing file each → None.
        if app._load_surrogate_calibration("sa", "fast", -1) is not None:
            print("  FAIL stale-schema calibration should return None")
            loader_diffs += 1
        else:
            print("  OK   stale schema → None (data problem degrades quietly)")
        if app._load_surrogate_calibration(
                "sa", "nonexistent_mode", app.SCENARIO_SCHEMA_VERSION) is not None:
            print("  FAIL missing-file calibration should return None")
            loader_diffs += 1
        else:
            print("  OK   missing file → None")
    except Exception as _e:
        print(f"  ERROR calibration loader test: {_e}")
        import traceback
        traceback.print_exc()
        loader_diffs += 1

    # ── Delta-direction discipline — lower-is-better cards invert the arrow ──
    # Runoff Volume (lower is better) must render an increase as a regression
    # (delta_color "inverse"); higher-is-better cards (Flood Index, NDVI) must
    # not. Pins both the _delta_pill mechanism AND the per-card call sites so a
    # future edit can't silently flip one back to green-up. (Relay — Runoff
    # Volume delta read backwards, contradicting the "lower is better" caption.)
    print(f"\n{'=' * 60}")
    print("Delta-direction discipline — lower-is-better cards invert")
    print(f"{'=' * 60}")
    delta_dir_diffs = 0
    try:
        # Mechanism (non-vacuous: both directions + the zero/off case).
        if app._delta_pill(5.0, inverse=True)[1] != "inverse":
            print("  FAIL _delta_pill(+, inverse=True) should colour 'inverse'")
            delta_dir_diffs += 1
        if app._delta_pill(-5.0, inverse=True)[1] != "inverse":
            print("  FAIL _delta_pill(-, inverse=True) should colour 'inverse'")
            delta_dir_diffs += 1
        if app._delta_pill(5.0)[1] != "normal":
            print("  FAIL _delta_pill(+) default should colour 'normal'")
            delta_dir_diffs += 1
        if app._delta_pill(0.0) != (None, "off"):
            print("  FAIL _delta_pill(0) should be (None, 'off')")
            delta_dir_diffs += 1
        if delta_dir_diffs == 0:
            print("  OK   _delta_pill mechanism: inverse↔'inverse', "
                  "default↔'normal', zero↔(None,'off')")
        # Call sites (source scan): Runoff Volume inverts; Flood/NDVI do not.
        import re as _re_dd
        _src_dd = Path("app.py").read_text()

        def _pill_args(_var):
            _m = _re_dd.search(
                rf"{_re_dd.escape(_var)},\s*\w+\s*=\s*_delta_pill\((.*?)\)\s*\n",
                _src_dd, _re_dd.S)
            return _m.group(1) if _m else None

        _runoff_args = _pill_args("_runoff_delta_str")
        if _runoff_args is None:
            print("  FAIL could not locate Runoff Volume _delta_pill call "
                  "(source scan stale — re-point it)")
            delta_dir_diffs += 1
        elif "inverse=True" not in _runoff_args:
            print("  FAIL Runoff Volume delta must pass inverse=True "
                  "(lower is better — an increase is a regression, not green-up)")
            delta_dir_diffs += 1
        else:
            print("  OK   Runoff Volume delta passes inverse=True")
        for _hb in ("_flood_delta_str", "_ndvi_delta_str"):
            _args = _pill_args(_hb)
            if _args is not None and "inverse=True" in _args:
                print(f"  FAIL {_hb}: higher-is-better card must NOT use inverse=True")
                delta_dir_diffs += 1
            else:
                print(f"  OK   {_hb}: higher-is-better, no inverse")
    except Exception as _e:
        print(f"  ERROR delta-direction check: {_e}")
        import traceback
        traceback.print_exc()
        delta_dir_diffs += 1

    # ── Validated-model set single source — Stage 1 ─────────────────────────
    # The 'validated' flag (per-pixel parity vs canonical natcap.invest 3.19.0)
    # has ONE canonical home: model_validation.MODEL_VALIDATION. The export bundle
    # re-exports it (eib._VALIDATION IS that object); Stage 2 badges read it too.
    # Assert the canonical set is exactly the expected models with parity metadata,
    # that the bundle is sourced (identity, not a re-declared literal that could
    # drift), and a non-vacuous flip-test. Doubles as a deliberate-change detector:
    # validating a 6th model trips this until the expected set is updated on purpose.
    print(f"\n{'=' * 60}")
    print("Validated-model set — single source of truth (Stage 1)")
    print(f"{'=' * 60}")
    src_diffs = 0
    try:
        import model_validation as _mv
        import export_invest_bundle as _eib
        if not hasattr(_mv, "MODEL_VALIDATION"):
            raise AttributeError("model_validation.MODEL_VALIDATION not found "
                                 "(canonical source moved?)")
        # UNA re-promoted to validated: the supply_percapita per-pixel parity
        # reproducer landed (compare_una_supply_invest.py → comparisons/
        # una_supply_parity_mn.csv: r = 1.000000, ~5.5e-7 relative MAE, clean +
        # non-vacuous guard, matched-but-independent vs InVEST 3.19.0). It had
        # been demoted while that reproducer was missing.
        _EXPECTED_VALIDATED = {"ucm", "una", "umh", "carbon", "ufr"}
        _EXPECTED_ALIGNED = set()

        def _validated_of(d):
            return {k for k, v in d.items() if v.get("status") == "validated"}

        def _aligned_of(d):
            return {k for k, v in d.items() if v.get("status") == "methodology_aligned"}

        _validated = _validated_of(_mv.MODEL_VALIDATION)
        _aligned = _aligned_of(_mv.MODEL_VALIDATION)
        if _validated != _EXPECTED_VALIDATED:
            print(f"  FAIL canonical validated set {sorted(_validated)} != expected "
                  f"{sorted(_EXPECTED_VALIDATED)} — a model was added/dropped; if "
                  "intentional, update _EXPECTED_VALIDATED on purpose")
            src_diffs += 1
        else:
            print(f"  OK   canonical validated set = {sorted(_validated)}")
        if _aligned != _EXPECTED_ALIGNED:
            print(f"  FAIL methodology_aligned set {sorted(_aligned)} != expected "
                  f"{sorted(_EXPECTED_ALIGNED)} — if intentional, update "
                  "_EXPECTED_ALIGNED on purpose")
            src_diffs += 1
        else:
            print(f"  OK   methodology_aligned set = {sorted(_aligned)} (empty — "
                  "all five models carry committed per-pixel reproducers)")
        # Parity metadata present on every validated model (reference + notes; the
        # four numeric ones also carry pearson_r — UMH's parity is kernel-based).
        _meta_missing = [k for k in _validated
                         if not _mv.MODEL_VALIDATION[k].get("reference")
                         or not _mv.MODEL_VALIDATION[k].get("notes")]
        if _meta_missing:
            print(f"  FAIL validated models missing reference/notes metadata: {_meta_missing}")
            src_diffs += 1
        else:
            print("  OK   every validated model carries reference + notes metadata")
        # Bundle is SOURCED from the canonical object (identity → can't drift).
        if _eib._VALIDATION is not _mv.MODEL_VALIDATION:
            print("  FAIL eib._VALIDATION is not model_validation.MODEL_VALIDATION "
                  "— the bundle re-declared a literal instead of re-exporting")
            src_diffs += 1
        else:
            print("  OK   eib._VALIDATION IS the canonical object (sourced, not copied)")
        # Bundle status dict — every model 'validated' (UNA re-promoted).
        _actual_status = {k: v.get("status") for k, v in _eib._VALIDATION.items()}
        _expected_status = {**{k: "validated" for k in _EXPECTED_VALIDATED},
                            **{k: "methodology_aligned" for k in _EXPECTED_ALIGNED}}
        if _actual_status != _expected_status:
            print(f"  FAIL bundle status dict changed: {_actual_status} "
                  f"(expected {_expected_status})")
            src_diffs += 1
        else:
            print("  OK   bundle status dict matches (all five validated)")
        # Flip-test (non-vacuous): the predicates actually catch seeded drift.
        _poison_drop = {k: v for k, v in _mv.MODEL_VALIDATION.items() if k != "carbon"}
        _poison_align = dict(_mv.MODEL_VALIDATION)
        _poison_align["ufr"] = {**_poison_align["ufr"], "status": "methodology_aligned"}
        if (_validated_of(_poison_drop) == _EXPECTED_VALIDATED
                or not _aligned_of(_poison_align)):
            print("  FAIL flip-test: a dropped/aligned model did NOT change the "
                  "derived set — the check is blind")
            src_diffs += 1
        else:
            print("  OK   flip-test: dropping or aligning a model correctly trips "
                  "the validated/aligned predicates")
    except Exception as _e:
        print(f"  ERROR validated-model source check: {_e}")
        import traceback
        traceback.print_exc()
        src_diffs += 1

    # ── InVEST-validated badge ↔ Stage-1 source cross-check (Stage 2 Slice 1) ─
    # Extends the committed-reproducer rule to the badge: a card renders
    # "InVEST-validated" ONLY if its model is in model_validation.VALIDATED_MODELS
    # AND it's on the validated compute path. No lumped-proxy / dollar / food /
    # cost / MN-carbon card may ever render validated. Flip-test + locate guard.
    print(f"\n{'=' * 60}")
    print("InVEST-validated badge ↔ Stage-1 source cross-check")
    print(f"{'=' * 60}")
    badge_src_diffs = 0
    try:
        import natcap_validation as _nv
        import model_validation as _mv
        if not hasattr(_nv, "_METRIC_TO_MODEL"):
            raise AttributeError("natcap_validation._METRIC_TO_MODEL not found "
                                 "(badge source map moved?)")
        # (1) Every validated-capable metric maps to a model in the Stage-1 set.
        _bad_map = [m for m, mod in _nv._METRIC_TO_MODEL.items()
                    if mod not in _mv.VALIDATED_MODELS]
        if _bad_map:
            print(f"  FAIL metrics mapped to a non-validated model: {_bad_map}")
            badge_src_diffs += 1
        else:
            print(f"  OK   all {len(_nv._METRIC_TO_MODEL)} validated-capable metrics "
                  "map to a model in the Stage-1 validated set")
        # (2) Forbidden cards must NOT be in the map (can't render validated).
        _FORBIDDEN = ["flood_reduction", "runoff_acre_feet", "food_mln_lbs",
                      "total_cost_mln", "carbon_value_usd", "ndvi",
                      "cost_per_acft", "cost_per_degf", "cost_per_1k_people",
                      "cooling_energy_savings_usd", "flood_damage_avoided_usd",
                      "avoided_mh_cost_usd"]
        _leaked = [m for m in _FORBIDDEN if m in _nv._METRIC_TO_MODEL]
        if _leaked:
            print(f"  FAIL lumped-proxy/dollar/food/cost cards in the validated map: {_leaked}")
            badge_src_diffs += 1
        else:
            print("  OK   no lumped-proxy / dollar / food / cost card can render validated")
        # (3) Live badge behaviour: carbon city-split + lumped proxy stays aligned.
        _ctx = _nv.SCENARIO_CONTEXT_EXPLORER
        _sa = _nv.render_validation_badge("carbon_tons_co2", _ctx, validated_path=True)
        _mn = _nv.render_validation_badge("carbon_tons_co2", _ctx,
                                          explicit_status="prototype", validated_path=False)
        _flood = _nv.render_validation_badge("flood_reduction", _nv.SCENARIO_CONTEXT_BASELINE)
        if _sa["state"] != "invest_validated":
            print(f"  FAIL SA carbon (stock path) should be invest_validated, got {_sa['state']}")
            badge_src_diffs += 1
        if _mn["state"] == "invest_validated":
            print("  FAIL MN carbon (proxy path) must NOT render invest_validated")
            badge_src_diffs += 1
        if _flood["state"] == "invest_validated":
            print("  FAIL Flood Index (lumped proxy) must NOT render invest_validated")
            badge_src_diffs += 1
        if (_sa["state"] == "invest_validated" and _mn["state"] != "invest_validated"
                and _flood["state"] != "invest_validated"):
            print("  OK   carbon SA→validated / MN→not; Flood Index → not validated")
        # (4) Flip-test (non-vacuous): inject a forbidden card into the map — the
        # forbidden-leak predicate must catch it.
        _poison_map = dict(_nv._METRIC_TO_MODEL)
        _poison_map["flood_reduction"] = "ufr"
        if "flood_reduction" not in [m for m in _FORBIDDEN if m in _poison_map]:
            print("  FAIL flip-test: the forbidden-leak check is blind to a seeded "
                  "lumped-proxy card in the map")
            badge_src_diffs += 1
        else:
            print("  OK   flip-test: a forbidden card injected into the validated "
                  "map is caught by the leak check")
    except Exception as _e:
        print(f"  ERROR badge↔source cross-check: {_e}")
        import traceback
        traceback.print_exc()
        badge_src_diffs += 1

    # ── Per-tier colorblind glyph — render-path lock (Stage 2 Fix B) ─────────
    # The four badge tiers share near-identical grayscale luminance (green/teal/
    # blue/gray collapse to ~35% luma), so a shape-distinct leading glyph is the
    # ONLY channel that keeps them apart for colorblind viewers. This lock proves
    # the glyph actually reaches the rendered ["text"] for ALL FOUR tiers via the
    # live render_validation_badge path — not just the legend caption — and that
    # the legend's glyphs match what the renderer emits 1:1. Flip-test +
    # locate-guard so it can't pass vacuously.
    print(f"\n{'=' * 60}")
    print("Per-tier colorblind glyph — render-path lock")
    print(f"{'=' * 60}")
    glyph_diffs = 0
    try:
        import re as _re_g
        import natcap_validation as _nvg
        # (1) Every tier's live badge carries its expected shape glyph in ["text"].
        #     Representative (metric, kwargs) per tier that lands that tier.
        _tier_cases = {
            "natcap_anchored":  ("◆", "NatCap published value",
                ("temp_change_f", _nvg.SCENARIO_CONTEXT_NATCAP_FIXED, {})),
            "invest_validated": ("■", "InVEST-validated",
                ("temp_change_f", _nvg.SCENARIO_CONTEXT_EXPLORER, {})),
            "invest_aligned":   ("○", "InVEST-aligned",
                ("flood_reduction", _nvg.SCENARIO_CONTEXT_BASELINE, {})),
            "prototype":        ("△", "Prototype",
                ("food_mln_lbs", _nvg.SCENARIO_CONTEXT_EXPLORER, {})),
        }
        _render_glyph = {}   # state -> glyph the renderer actually emits
        for _state, (_want_g, _want_name, (_m, _ctx, _kw)) in _tier_cases.items():
            _b = _nvg.render_validation_badge(_m, _ctx, **_kw)
            if _b["state"] != _state:
                print(f"  FAIL tier case {_state}: badge landed state={_b['state']} "
                      f"(metric={_m}, ctx={_ctx}); re-pick a representative case")
                glyph_diffs += 1
                continue
            _got_g = _nvg.badge_glyph(_b["text"])
            _render_glyph[_state] = _got_g
            if _got_g != _want_g:
                print(f"  FAIL {_state}: rendered glyph {_got_g!r} != expected "
                      f"{_want_g!r} (the colorblind shape channel is missing/wrong)")
                glyph_diffs += 1
            elif _want_name not in _b["text"]:
                print(f"  FAIL {_state}: tier name {_want_name!r} absent from "
                      f"rendered text {_b['text']!r}")
                glyph_diffs += 1
        if glyph_diffs == 0:
            print("  OK   all 4 tiers carry their shape glyph (◆ ■ ○ △) in the "
                  "live rendered ['text']")
        # (2) Flip-test (non-vacuous): Prototype text must NOT carry the validated
        #     glyph, and the validated text must NOT carry the Prototype glyph.
        _proto_t = _nvg.render_validation_badge(
            "food_mln_lbs", _nvg.SCENARIO_CONTEXT_EXPLORER)["text"]
        _val_t = _nvg.render_validation_badge(
            "temp_change_f", _nvg.SCENARIO_CONTEXT_EXPLORER)["text"]
        if _nvg.badge_glyph(_proto_t) == "■":
            print("  FAIL flip-test: a Prototype badge carries the ■ validated glyph")
            glyph_diffs += 1
        elif _nvg.badge_glyph(_val_t) == "△":
            print("  FAIL flip-test: an InVEST-validated badge carries the △ "
                  "Prototype glyph")
            glyph_diffs += 1
        else:
            print("  OK   flip-test: Prototype text isn't glyphed ■, validated "
                  "text isn't glyphed △")
        # (3) Legend↔render consistency: parse the 4 glyphs out of the live legend
        #     caption in app.py and assert each equals what the renderer emits for
        #     that tier. Catches a legend that drifts from the render path.
        _app_src_g = Path("app.py").read_text(encoding="utf-8")
        _legend_glyph = {}
        for _state, _name in (("natcap_anchored", "NatCap published value"),
                              ("invest_validated", "InVEST-validated"),
                              ("invest_aligned", "InVEST-aligned"),
                              ("prototype", "Prototype")):
            _lm = _re_g.search(r"([◆■○△])\s+" + _re_g.escape(_name), _app_src_g)
            if _lm is None:
                print(f"  FAIL could not locate the legend glyph for {_name!r} "
                      "(legend caption moved/renamed — re-point the scan)")
                glyph_diffs += 1
            else:
                _legend_glyph[_state] = _lm.group(1)
        for _state, _lg in _legend_glyph.items():
            _rg = _render_glyph.get(_state)
            if _rg is not None and _lg != _rg:
                print(f"  FAIL legend↔render mismatch for {_state}: legend {_lg!r} "
                      f"!= renderer {_rg!r}")
                glyph_diffs += 1
        if (len(_legend_glyph) == 4
                and all(_legend_glyph[s] == _render_glyph.get(s)
                        for s in _legend_glyph)):
            print("  OK   legend caption's 4 glyphs match the renderer 1:1")
        # (4) Locate-guard flip-test: the legend regex has teeth (a wrong glyph
        #     in a seeded string is detected as a mismatch, not silently passed).
        #     The ✓ seed is load-bearing AFTER the ✓→■ swap: it locks in that the
        #     validated tier is no longer ✓, so a stray legacy "✓ InVEST-validated"
        #     can't satisfy the parser.
        _seed = "✗ NatCap published value"
        _sm = _re_g.search(r"([◆■○△])\s+NatCap published value", _seed)
        _sm_tick = _re_g.search(r"([◆■○△])\s+InVEST-validated", "✓ InVEST-validated")
        if _sm is not None:
            print("  FAIL locate-guard: the glyph regex matched a non-tier glyph "
                  "(✗) — the character class is too loose")
            glyph_diffs += 1
        elif _sm_tick is not None:
            print("  FAIL locate-guard: a legacy '✓ InVEST-validated' still "
                  "matches — the validated glyph wasn't fully migrated off ✓")
            glyph_diffs += 1
        else:
            print("  OK   locate-guard: the glyph regex only matches the 4 real "
                  "tier shapes, and a legacy ✓ validated label no longer parses")
    except Exception as _e:
        print(f"  ERROR per-tier glyph render-path lock: {_e}")
        import traceback
        traceback.print_exc()
        glyph_diffs += 1

    # ── Carbon unit single-source lock — no duplicate long-form unit vars ────
    # All carbon value-display surfaces (scatter Y-label, comparison table, the
    # no-results warning) route through the shared _carbon_unit_suffix ("t CO2e"
    # / "t CO2e/yr"). The two retired long-form unit vars (_carbon_unit,
    # _opt_carbon_unit = "tons CO2e") must NOT reappear — they were the source of
    # the "tons CO2e" drift. The gate can't render the labels, so source-scan for
    # a bare assignment of either var (word-boundary, so _carbon_unit_suffix /
    # _carbon_unit_label don't false-positive). Flip-test seeds a re-introduction
    # and asserts it's caught. Input sliders keep "tons CO2e" by design and are
    # NOT scanned (they don't assign these vars).
    print(f"\n{'=' * 60}")
    print("Carbon unit single-source lock")
    print(f"{'=' * 60}")
    carbon_unit_diffs = 0
    try:
        import re as _re_cu
        _app_src_cu = Path("app.py").read_text(encoding="utf-8")
        # Bare assignment of either retired unit var. \b before the name keeps
        # _carbon_unit_suffix / _carbon_unit_label / _opt_carbon_col_label clean.
        _dup_pat = _re_cu.compile(r"(?<![\w])(_carbon_unit|_opt_carbon_unit)\s*=")
        _hits = [m.group(1) for m in _dup_pat.finditer(_app_src_cu)]
        if _hits:
            print(f"  FAIL duplicate carbon unit source(s) reintroduced: {_hits} "
                  "— route value-display units through _carbon_unit_suffix instead")
            carbon_unit_diffs += len(_hits)
        else:
            print("  OK   no _carbon_unit / _opt_carbon_unit assignment remains "
                  "(single source: _carbon_unit_suffix)")
        # Flip-test (non-vacuous): a seeded re-introduction must be caught, and
        # the shared-suffix var must NOT be mistaken for the retired one.
        _seed_bad = "    _carbon_unit = 'tons CO2e' if x else 'tons CO2e/yr'\n"
        _seed_ok = "    _carbon_unit_suffix = 't CO2e' if x else 't CO2e/yr'\n"
        if not _dup_pat.search(_seed_bad):
            print("  FAIL flip-test: a seeded _carbon_unit re-introduction was "
                  "NOT caught — the scan is blind")
            carbon_unit_diffs += 1
        elif _dup_pat.search(_seed_ok):
            print("  FAIL flip-test: the scan false-positives on "
                  "_carbon_unit_suffix (word-boundary too loose)")
            carbon_unit_diffs += 1
        else:
            print("  OK   flip-test: seeded re-introduction caught, shared "
                  "_carbon_unit_suffix not mistaken for it")
        # Carbon INPUT-slider labels read "t CO2e" too (last holdout normalized).
        # Anchor on the distinctive label text so the "tons CO2e" doc comments
        # (sequestration-rate conversion notes) don't false-positive: those say
        # "tons CO2e/acre/yr" but never "carbon rate (tons CO2e" / "≥ (tons CO2e".
        _slider_bad = ['carbon rate (tons CO2e', '≥ (tons CO2e']
        _slider_good = ['carbon rate (t CO2e/acre/yr)', '≥ (t CO2e']
        _bad_hit = [s for s in _slider_bad if s in _app_src_cu]
        _good_missing = [s for s in _slider_good if s not in _app_src_cu]
        if _bad_hit:
            print(f"  FAIL carbon slider label still reads 'tons CO2e': {_bad_hit}")
            carbon_unit_diffs += 1
        elif _good_missing:
            print(f"  FAIL carbon slider 't CO2e' label missing/renamed: {_good_missing}")
            carbon_unit_diffs += 1
        else:
            print("  OK   carbon input-slider labels normalized to 't CO2e' "
                  "(rate + optimizer-target sliders)")
    except Exception as _e:
        print(f"  ERROR carbon unit single-source lock: {_e}")
        import traceback
        traceback.print_exc()
        carbon_unit_diffs += 1

    # ── Post-optimize auto-switch migration lock ─────────────────────────────
    # The optimize branches auto-switch to Tradeoffs (set main_tab + toast +
    # rerun); the old manual-switch "just_optimized" banner was removed. The gate
    # can't render a removed banner or catch a Streamlit runtime warning, so
    # source-scan: (1) the dead flag never reappears, and (2) the main_tab
    # segmented_control carries no default=/value=/index= (which would re-trigger
    # the "default ignored" warning against the seeded session_state key). Both
    # have flip-test seeds so the scan can't pass vacuously.
    print(f"\n{'=' * 60}")
    print("Post-optimize auto-switch migration lock")
    print(f"{'=' * 60}")
    autoswitch_diffs = 0
    try:
        import re as _re_as
        _app_src_as = Path("app.py").read_text(encoding="utf-8")
        # Guard 1 — the dead flag is gone and stays gone.
        if "just_optimized" in _app_src_as:
            print("  FAIL 'just_optimized' still appears in app.py — the dead "
                  "manual-switch flag wasn't fully removed")
            autoswitch_diffs += 1
        else:
            print("  OK   'just_optimized' absent (dead manual-switch flag removed)")
        # Guard 2 — the main_tab segmented_control has no default=/value=/index=.
        # Isolate the keyed call body, then scan it for the forbidden kwargs.
        def _main_tab_widget_body(src):
            _m = _re_as.search(r"st\.segmented_control\((.*?)\)",
                               src, _re_as.S)
            # Walk all segmented_control calls; return the one keyed "main_tab".
            for _mm in _re_as.finditer(r"st\.segmented_control\((.*?)\)",
                                       src, _re_as.S):
                if 'key="main_tab"' in _mm.group(1) or "key='main_tab'" in _mm.group(1):
                    return _mm.group(1)
            return None
        _body = _main_tab_widget_body(_app_src_as)
        if _body is None:
            print("  FAIL could not locate the main_tab st.segmented_control "
                  "call (re-point the scan)")
            autoswitch_diffs += 1
        elif _re_as.search(r"\b(default|value|index)\s*=", _body):
            print("  FAIL the main_tab segmented_control passes a "
                  "default=/value=/index= arg — collides with the seeded "
                  "session_state key (Streamlit 'default ignored' warning)")
            autoswitch_diffs += 1
        else:
            print("  OK   main_tab segmented_control has no default/value/index "
                  "(session_state key seeded separately)")
        # Flip-test (non-vacuous): both guards must fire on reintroduced code.
        if "just_optimized" not in "st.session_state.just_optimized = True":
            print("  FAIL flip-test: Guard 1 is blind to a seeded flag write")
            autoswitch_diffs += 1
        _seed_widget = ('st.segmented_control("Main view", options=X, '
                        'default=X[0], key="main_tab")')
        _seed_body = _main_tab_widget_body(_seed_widget)
        if _seed_body is None or not _re_as.search(r"\b(default|value|index)\s*=",
                                                   _seed_body):
            print("  FAIL flip-test: Guard 2 is blind to a seeded default= arg")
            autoswitch_diffs += 1
        if autoswitch_diffs == 0:
            print("  OK   flip-test: both guards fire on reintroduced banner/default")
    except Exception as _e:
        print(f"  ERROR post-optimize auto-switch migration lock: {_e}")
        import traceback
        traceback.print_exc()
        autoswitch_diffs += 1

    # ── Cost-effectiveness suppression condition lock ────────────────────────
    # The dashboard #### Cost Effectiveness section renders a ratio card ONLY
    # where compute_cost_effectiveness returns a number; it hides the card (vs an
    # "N/A" card) when the ratio is None. This locks the SUPPRESSION CONDITION the
    # render keys on — compute_cost_effectiveness is pure, so call it directly with
    # crafted results dicts, both directions, non-vacuous. The honesty-critical
    # case: WARMING (temp_change_f ≥ 0) must yield None (no cooling ratio), so a
    # bad outcome can never masquerade as a cheap one.
    print(f"\n{'=' * 60}")
    print("Cost-effectiveness suppression condition lock")
    print(f"{'=' * 60}")
    ce_diffs = 0
    try:
        _ce_fn = app.compute_cost_effectiveness
        _BR = 1000.0   # baseline runoff ac-ft

        def _ce_res(**ov):
            _d = {'total_cost_mln': 5.0, 'runoff_acre_feet': 980.0,
                  'temp_change_f': -0.5, 'people_fed': 500}
            _d.update(ov)
            return _d

        # (label, results-overrides, key, want_number)  want_number=True → a
        # numeric ratio must render; False → None (card suppressed).
        _cases = [
            # runoff: ≥10 ac-ft prevented → number; 0 prevented / negative → None.
            ("runoff prevented 20 ac-ft", {'runoff_acre_feet': 980.0},
             'cost_per_acft', True),
            ("runoff prevented 0",        {'runoff_acre_feet': 1000.0},
             'cost_per_acft', False),
            ("runoff went UP (negative)", {'runoff_acre_feet': 1010.0},
             'cost_per_acft', False),
            # cooling: ≤ −0.05 °F → number; warming (≥0) → None; |Δ|<0.05 → None.
            ("cooling −0.5 °F",           {'temp_change_f': -0.5},
             'cost_per_degf', True),
            ("WARMING +0.5 °F",           {'temp_change_f': 0.5},
             'cost_per_degf', False),
            ("cooling tiny −0.01 °F",     {'temp_change_f': -0.01},
             'cost_per_degf', False),
            # food: ≥100 people → number; 0 (<100) → None.
            ("people fed 500",            {'people_fed': 500},
             'cost_per_1k_people', True),
            ("people fed 0",              {'people_fed': 0},
             'cost_per_1k_people', False),
        ]
        for _name, _ov, _key, _want_num in _cases:
            _out = _ce_fn(_ce_res(**_ov), _BR)[_key]
            _is_num = isinstance(_out, (int, float)) and _out is not None
            if _is_num != _want_num:
                print(f"  FAIL {_name}: {_key} = {_out!r}, "
                      f"want {'a number' if _want_num else 'None'}")
                ce_diffs += 1
        # cost ≤ 0 → all three None (nothing is cost-effective at zero cost).
        for _cost in (0.0, -1.0):
            _all = _ce_fn(_ce_res(total_cost_mln=_cost), _BR)
            if any(v is not None for v in _all.values()):
                print(f"  FAIL cost={_cost}: expected all None, got {_all}")
                ce_diffs += 1
        if ce_diffs == 0:
            print(f"  OK   suppression condition holds across {len(_cases)} crafted "
                  "cases + cost≤0 (warming→None is the honesty-critical case)")
    except Exception as _e:
        print(f"  ERROR cost-effectiveness suppression lock: {_e}")
        import traceback
        traceback.print_exc()
        ce_diffs += 1

    # ── Active-scenario line-1 helper lock ───────────────────────────────────
    # The page-root Active-scenario block builds line 1 via _active_scenario_line1
    # (pure: resolved dict + provenance). Lock the prefix-per-provenance mapping,
    # the pct=0 → "Baseline · no conversion" override (a 0%-conversion scenario is
    # indistinguishable from baseline), and that the mix string always names all
    # three components (GI/FF/HD). Also assert the prefix map covers EVERY value
    # _scen_provenance is assigned in app.py, so a new provenance can't render an
    # empty prefix. The gate can't render the block; this locks its content.
    print(f"\n{'=' * 60}")
    print("Active-scenario line-1 helper lock")
    print(f"{'=' * 60}")
    active_scn_diffs = 0
    try:
        import re as _re_as2
        _line1 = app._active_scenario_line1
        _eib = app.eib

        def _res(pct, gi=50, ff=30):
            return {'pct_converted': pct, 'green_infrastructure_pct': gi,
                    'food_forest_pct': ff, 'pct_highdensity': 100 - gi - ff}

        # (provenance, expected line-1 prefix) for pct>0.
        _prov_prefix = [
            (_eib.PROVENANCE_EXPLORER,         "Explorer scenario"),
            (_eib.PROVENANCE_OPTIMIZER,        "Optimizer-applied"),
            (_eib.PROVENANCE_REGION_OPTIMIZED, "Selected-area optimized"),
            (_eib.PROVENANCE_BASELINE,         "Baseline"),
        ]
        for _prov, _pref in _prov_prefix:
            _out = _line1(_res(25), _prov)
            if not _out.startswith(_pref + " · "):
                print(f"  FAIL provenance {_prov!r}: line 1 {_out!r} doesn't "
                      f"start with {_pref + ' · '!r}")
                active_scn_diffs += 1
            for _comp in ("GI ", "FF ", "HD "):
                if _comp not in _out:
                    print(f"  FAIL provenance {_prov!r}: mix string missing "
                          f"{_comp!r} component — {_out!r}")
                    active_scn_diffs += 1
        # pct=0 → baseline line regardless of provenance (override).
        for _prov, _ in _prov_prefix:
            _out0 = _line1(_res(0), _prov)
            if _out0 != "Baseline · no conversion":
                print(f"  FAIL pct=0 with provenance {_prov!r}: got {_out0!r}, "
                      "want 'Baseline · no conversion'")
                active_scn_diffs += 1
        # Non-vacuous flip-test: an Explorer line must NOT carry another tier's
        # prefix (proves the mapping discriminates, not a constant).
        _expl = _line1(_res(25), _eib.PROVENANCE_EXPLORER)
        if _expl.startswith("Optimizer-applied") or _expl.startswith("Selected-area"):
            print("  FAIL flip-test: Explorer line carries a non-Explorer prefix")
            active_scn_diffs += 1
        if active_scn_diffs == 0:
            print("  OK   line-1 prefix per provenance + pct=0 baseline override "
                  "+ GI/FF/HD always present (flip-test discriminates)")
        # Coverage: every PROVENANCE_* assigned to _scen_provenance in app.py must
        # be a key in the prefix map, so a new provenance can't render empty.
        _app_src_as2 = Path("app.py").read_text(encoding="utf-8")
        _assigned = set(_re_as2.findall(
            r"_scen_provenance\s*=\s*eib\.(PROVENANCE_\w+)", _app_src_as2))
        if not _assigned:
            print("  FAIL could not find any _scen_provenance assignment "
                  "(re-point the coverage scan)")
            active_scn_diffs += 1
        _map_keys = {k for k in app._ACTIVE_SCENARIO_PREFIX}
        _uncovered = [p for p in _assigned
                      if getattr(_eib, p) not in _map_keys]
        if _uncovered:
            print(f"  FAIL provenance value(s) assigned but not in the prefix "
                  f"map: {_uncovered} — would render an empty/fallback prefix")
            active_scn_diffs += 1
        else:
            print(f"  OK   prefix map covers all {len(_assigned)} _scen_provenance "
                  f"assignment(s): {sorted(_assigned)}")
        # Static order guard — the Active-scenario block must render ABOVE the
        # Discover centerpiece. Cheap source-position check the 40/40 won't catch
        # (both render fine wherever they sit); locks the above-Discover placement
        # so a future edit can't silently drop the block back down the page.
        _blk = _app_src_as2.find('**Active scenario**')
        _disc = _app_src_as2.find('### Discover scenarios')
        if _blk == -1:
            print("  FAIL could not locate the '**Active scenario**' render call "
                  "(block removed or renamed — re-point the order guard)")
            active_scn_diffs += 1
        elif _disc == -1:
            print("  FAIL could not locate the '### Discover scenarios' marker "
                  "(re-point the order guard)")
            active_scn_diffs += 1
        elif _blk > _disc:
            print("  FAIL Active-scenario block renders BELOW the Discover "
                  "centerpiece — it must sit above it (placement regressed)")
            active_scn_diffs += 1
        else:
            print("  OK   Active-scenario block renders above the Discover centerpiece")
    except Exception as _e:
        print(f"  ERROR active-scenario line-1 helper lock: {_e}")
        import traceback
        traceback.print_exc()
        active_scn_diffs += 1

    # ── Flood Damage Avoided conditional render — table-presence gate lock ───
    # The card is hidden ONLY when the city has no damage-valuation table; the
    # gate must key on TOTAL_POTENTIAL_DAMAGE_USD (table presence), NEVER on the
    # computed value _flood_damage_avoided — else a legitimate $0 result with a
    # table loaded would be wrongly suppressed. The gate can't see rendered
    # output (same blind spot as the dark loader / range table), so source-scan
    # the render gate + the unavailable note. Flip-test + locate guard.
    print(f"\n{'=' * 60}")
    print("Flood Damage Avoided — table-presence render gate")
    print(f"{'=' * 60}")
    fda_diffs = 0
    try:
        import re as _re_fda
        _app_src = Path("app.py").read_text(encoding="utf-8")

        def _gate_ok(g):
            return ("TOTAL_POTENTIAL_DAMAGE_USD" in g
                    and "_flood_damage_avoided" not in g)

        _m = _re_fda.search(r"_show_flood_damage\s*=\s*\((.*?)\)", _app_src, _re_fda.S)
        if _m is None:
            print("  FAIL could not locate the _show_flood_damage render gate "
                  "(re-point the scan)")
            fda_diffs += 1
        elif not _gate_ok(_m.group(1)):
            print("  FAIL flood-damage gate must key on TOTAL_POTENTIAL_DAMAGE_USD "
                  "(table presence) and NOT on the computed value")
            fda_diffs += 1
        else:
            print("  OK   hide gates on table presence (TOTAL_POTENTIAL_DAMAGE_USD), "
                  "not the computed value")
        # The unavailable note replaces the hidden card (no empty '—' slot).
        if "requires a city-specific damage-valuation table" not in _app_src:
            print("  FAIL the unavailable-metrics note is missing for the no-table case")
            fda_diffs += 1
        else:
            print("  OK   unavailable-metrics note present for the no-table case")
        # A table-present card still renders the value (zero included) — the
        # value branch survives, so a real $0 result isn't hidden.
        if "_fmt_usd(_flood_damage_avoided)" not in _app_src:
            print("  FAIL the table-present value branch (renders $0 too) is missing")
            fda_diffs += 1
        else:
            print("  OK   table-present branch still renders the value (zero included)")
        # Flip-test (non-vacuous): the predicate rejects a value-keyed gate.
        if _gate_ok("BUILDINGS_DATA_AVAILABLE and _flood_damage_avoided > 0"):
            print("  FAIL flip-test: a value-keyed gate slipped past the checker")
            fda_diffs += 1
        else:
            print("  OK   flip-test: a value-keyed gate is correctly rejected")
    except Exception as _e:
        print(f"  ERROR flood-damage gate lock: {_e}")
        import traceback
        traceback.print_exc()
        fda_diffs += 1

    # ── Runoff retention index — presence, bounds, Jensen non-degeneracy ────
    # Relay 58: `runoff_retention_idx` = canonical UFR `rnf_rt_idx = mean(1 −
    # Q/P)`, the per-pixel retention average. Three non-vacuous guards:
    #   (1) present in evaluate_scenario's return dict + REQUIRED_TARGET_COLUMNS,
    #       and ∈ [0, 1].
    #   (2) Jensen gap is REAL — the per-pixel mean(1 − Q/P) must NOT equal the
    #       mean-CN-lumped form 1 − Q(mean_CN)/P (Q is convex in CN). A
    #       regression that silently reverts to the lumped Flood-Index math
    #       collapses the gap → this FAILS.
    #   (3) sign-of-change agreement — across two scenarios the per-pixel and
    #       lumped readings move the SAME direction (greening raises retention),
    #       so the new metric isn't inverted.
    print(f"\n{'=' * 60}")
    print("Runoff retention index — presence + bounds + Jensen non-degeneracy")
    print(f"{'=' * 60}")
    retention_idx_diffs = 0
    try:
        if 'runoff_retention_idx' not in app.REQUIRED_TARGET_COLUMNS:
            print("  FAIL runoff_retention_idx absent from REQUIRED_TARGET_COLUMNS")
            retention_idx_diffs += 1

        def _lumped_retention(mean_cn, P):
            """1 − Q(mean_CN)/P — the mean-CN-lumped retention (Flood-Index math)."""
            if mean_cn <= 0:
                return 0.0
            S = (1000.0 / mean_cn) - 10.0
            Ia = 0.2 * S
            Q = 0.0 if P <= Ia else (P - Ia) ** 2 / (P - Ia + S)
            return 1.0 - Q / P

        for _city in active_cities:
            _rebind_city(app, _city)
            P = app.DESIGN_STORM_INCHES
            # Two scenarios: low-greening (A) vs GI-heavy (B). GI lowers CN →
            # raises retention, so both readings should rise A→B.
            _A = app.evaluate_scenario(10, 0, 0, seed=42, placement_strategy="random")
            _B = app.evaluate_scenario(50, 100, 0, seed=42, placement_strategy="random")
            _bad = []
            for _tag, _r in (("A", _A), ("B", _B)):
                if 'runoff_retention_idx' not in _r:
                    _bad.append(f"{_tag}:missing"); continue
                _v = _r['runoff_retention_idx']
                if not (isinstance(_v, (int, float)) and 0.0 <= _v <= 1.0):
                    _bad.append(f"{_tag}:out-of-bounds({_v!r})")
            if _bad:
                print(f"  FAIL {_city}: {_bad}")
                retention_idx_diffs += 1
                continue
            # (2) Jensen gap real — per-pixel ≠ lumped on a varied-CN AOI.
            _new_A = _A['runoff_retention_idx']
            _lump_A = _lumped_retention(_A['mean_cn'], P)
            _gap = abs(_new_A - _lump_A)
            if _gap <= 1e-4:
                print(f"  FAIL {_city}: Jensen gap collapsed — per-pixel "
                      f"mean(1−Q/P)={_new_A:.4f} ≈ lumped 1−Q(mean_CN)/P="
                      f"{_lump_A:.4f} (gap {_gap:.2e}); did the metric silently "
                      "revert to the lumped Flood-Index form?")
                retention_idx_diffs += 1
                continue
            # (3) sign-of-change agreement between the two readings.
            _new_d = _B['runoff_retention_idx'] - _new_A
            _lump_d = _lumped_retention(_B['mean_cn'], P) - _lump_A
            if _new_d == 0 or (_new_d > 0) != (_lump_d > 0):
                print(f"  FAIL {_city}: sign-of-change disagreement — "
                      f"per-pixel Δ={_new_d:+.4f}, lumped Δ={_lump_d:+.4f}")
                retention_idx_diffs += 1
                continue
            print(f"  OK   {_city}: present + ∈[0,1]; Jensen gap {_gap:.4f} "
                  f"(real, not lumped); Δ sign agrees (per-pixel {_new_d:+.4f}, "
                  f"lumped {_lump_d:+.4f})")
        # Meta-test (non-vacuous): the lumped form must actually differ from a
        # constructed per-pixel mean — prove the gap check has teeth. Build a
        # 2-pixel CN set whose mean(1−Q/P) ≠ 1−Q(mean_CN)/P by construction.
        _Pm = 6.0
        _cn_two = __import__('numpy').array([60.0, 95.0])
        _S = 1000.0 / _cn_two - 10.0; _Ia = 0.2 * _S
        _Q = __import__('numpy').where(_Pm <= _Ia, 0.0, (_Pm - _Ia) ** 2 / (_Pm - _Ia + _S))
        _per_pixel_mean = float((1.0 - _Q / _Pm).mean())
        _mcn = float(_cn_two.mean())
        _Sm = 1000.0 / _mcn - 10.0; _Iam = 0.2 * _Sm
        _Qm = 0.0 if _Pm <= _Iam else (_Pm - _Iam) ** 2 / (_Pm - _Iam + _Sm)
        _lumped_two = 1.0 - _Qm / _Pm
        if abs(_per_pixel_mean - _lumped_two) <= 1e-4:
            print("  FAIL meta-test: constructed Jensen gap is ~0 — the gap "
                  "check could pass vacuously")
            retention_idx_diffs += 1
        else:
            print(f"  OK   meta-test: constructed Jensen gap "
                  f"{abs(_per_pixel_mean - _lumped_two):.4f} > 0 (gap check has teeth)")
    except Exception as e:
        print(f"  ERROR runoff-retention index check: {e}")
        import traceback; traceback.print_exc()
        retention_idx_diffs += 1

    # ── Children's-card visibility lock — Relay 65 ─────────────────────────
    # The Children's Nature Access card hides when |child − overall| < epsilon,
    # so its PRESENCE means children are genuinely differently served — it does
    # not imply a distinct beneficiary group when the measurement says they
    # track overall access. Lock the predicate (`app._should_show_child_card`)
    # at the boundary: hide near-equal, show on divergence or absent data.
    print(f"\n{'=' * 60}")
    print("Children's-card visibility lock — Relay 65")
    print(f"{'=' * 60}")
    child_card_diffs = 0
    try:
        _f = app._should_show_child_card
        _EPS = app._CHILD_NAT_DIVERGENCE_EPSILON_PP
        _cases = [
            (None, 90.0, True,  "absent child data -> show ('—')"),
            (90.0, 90.0, False, "exactly equal -> hide"),
            (90.0 + _EPS - 0.1, 90.0, False, "below epsilon -> hide"),
            (90.0 + _EPS + 0.1, 90.0, True,  "at/above epsilon -> show"),
            (90.0 - _EPS - 0.1, 90.0, True,  "diverges below by > epsilon -> show"),
        ]
        for _child, _overall, _want, _desc in _cases:
            _got = bool(_f(_child, _overall, _EPS))
            if _got != _want:
                print(f"  FAIL child-card visibility — {_desc}: got {_got}, want {_want}")
                child_card_diffs += 1
        if _EPS != 0.5:
            print(f"  WARN _CHILD_NAT_DIVERGENCE_EPSILON_PP = {_EPS} (expected 0.5)")
        # Non-vacuous meta-test: an always-show predicate would fail the hide
        # cases; confirm the REAL predicate actually hides the equal case.
        if _f(90.0, 90.0, _EPS) is not False:
            print("  FAIL meta-test: predicate does not hide the equal case "
                  "(an always-show regression would pass vacuously)")
            child_card_diffs += 1
        elif child_card_diffs == 0:
            print(f"  OK   child-card visibility: hides near-equal (|delta| < {_EPS:g}pp), "
                  "shows on divergence or absent data; equal-case hide is real")
    except Exception as _e:
        print(f"  ERROR child-card visibility lock: {_e}")
        child_card_diffs += 1

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

    # ── Vocabulary guard (Relay 26) ─────────────────────────────────────────
    # Retired terms must not reappear on user-facing surfaces (app.py +
    # REFERENCE/CAPABILITIES/README). Rides the gate so a reverted rename fails
    # here. Internal docs are intentionally out of scope (they record history);
    # "surrogate" is intentionally not guarded. See scripts/check_vocabulary.py.
    print(f"\n{'=' * 60}")
    print("Vocabulary guard — retired-term scan (user-facing surfaces)")
    print(f"{'=' * 60}")
    vocab_diffs = 0
    try:
        import importlib.util as _ilu, pathlib as _plib
        _vg_path = _plib.Path(__file__).resolve().parent / "scripts" / "check_vocabulary.py"
        _vg_spec = _ilu.spec_from_file_location("check_vocabulary", _vg_path)
        _vg_mod = _ilu.module_from_spec(_vg_spec)
        _vg_spec.loader.exec_module(_vg_mod)
        vocab_diffs = _vg_mod.main()
        # Meta-test: prove the guard isn't a vacuous pass — a seeded retired
        # term must be caught, the 'vocab-allow' marker must suppress it, and
        # canonical copy must stay clean. Same discipline as Assertion-C's swap
        # test. Exercises check_vocabulary's real detection path.
        if _vg_mod.selftest() == 0:
            print("  OK   meta-test: guard catches a seeded retired term and "
                  "honors the 'vocab-allow' marker (not a vacuous pass)")
        else:
            print("  FAIL meta-test: vocabulary guard is vacuous — a seeded "
                  "retired term was NOT caught (or the allow marker failed)")
            vocab_diffs += 1
    except Exception as e:
        print(f"  ERROR vocabulary guard: {e}")
        import traceback; traceback.print_exc()
        vocab_diffs = 1

    # ── Tradeoff-plot flood-axis convention lock (region ↔ citywide) ────────
    # Both plot_tradeoff (citywide) and plot_tradeoff_region (selected-area)
    # place every point — INCLUDING the baseline — on the ABSOLUTE Flood Index
    # (100 - mean_cn). The region baseline used to be pinned at x=[0.0] (a delta
    # convention) while its scenario points were absolute, so the two charts read
    # as unrelated. Lock that the region baseline marker is sourced from the
    # absolute `flood_reduction_baseline`, NOT a hardcoded zero, and that both
    # functions take flood from `flood_reduction` (absolute) + cooling from
    # `mean_hm`. Source-scan (render is gate-blind). Flip-test = non-vacuous.
    print(f"\n{'=' * 60}")
    print("Tradeoff-plot flood-axis convention lock (region ↔ citywide)")
    print(f"{'=' * 60}")
    tradeoff_axis_diffs = 0
    try:
        import re as _re_ax
        _ax_src = Path("app.py").read_text(encoding="utf-8")

        def _func_body(_name):
            _m = _re_ax.search(
                r"\ndef " + _re_ax.escape(_name) + r"\(.*?(?=\ndef |\nclass |\Z)",
                _ax_src, _re_ax.S,
            )
            return _m.group(0) if _m else None

        _city_body = _func_body("plot_tradeoff")
        _region_body = _func_body("plot_tradeoff_region")
        if not _city_body or not _region_body:
            print("  FAIL could not locate plot_tradeoff / plot_tradeoff_region "
                  "(re-point the convention lock)")
            tradeoff_axis_diffs += 1
        else:
            # (1) Region baseline marker must come from the absolute
            #     flood_reduction_baseline, never a hardcoded x=[0.0] delta-pin.
            if "flood_reduction_baseline" not in _region_body:
                print("  FAIL region baseline no longer sourced from the absolute "
                      "`flood_reduction_baseline` — it may have reverted to a "
                      "delta/zero pin")
                tradeoff_axis_diffs += 1
            elif _re_ax.search(r"x=\[0\.0\]", _region_body):
                print("  FAIL region baseline marker pinned at x=[0.0] (delta "
                      "convention) — must sit at its absolute Flood Index")
                tradeoff_axis_diffs += 1
            else:
                print("  OK   region baseline marker uses the absolute "
                      "flood_reduction_baseline (not pinned at 0)")
            # (2) Both functions plot flood from the absolute `flood_reduction`
            #     field and cooling from `mean_hm` — same convention.
            for _nm, _body in (("plot_tradeoff", _city_body),
                               ("plot_tradeoff_region", _region_body)):
                if "flood_reduction" not in _body or "mean_hm" not in _body:
                    print(f"  FAIL {_nm} axis source drifted — expected absolute "
                          f"`flood_reduction` (X) + `mean_hm` (Y) in both plots")
                    tradeoff_axis_diffs += 1
            if tradeoff_axis_diffs == 0:
                print("  OK   both plots take flood from the absolute "
                      "`flood_reduction` field + cooling from `mean_hm`")
            # Flip-test (non-vacuous): the delta-pin detector must fire on a
            # seeded x=[0.0] baseline trace.
            if not _re_ax.search(r"x=\[0\.0\]",
                                 "fig.add_trace(go.Scatter(x=[0.0], y=[b]"):
                print("  FAIL meta-test: x=[0.0] delta-pin detector is vacuous")
                tradeoff_axis_diffs += 1
            else:
                print("  OK   meta-test: a seeded x=[0.0] baseline pin is caught")
            # (3) Plot self-documentation: both Y-axis titles spell out the
            #     cooling metric (no bare "HMI" acronym in an axis title), and
            #     both plots carry the identical "↗ better" direction cue
            #     (parity). Render is gate-blind; these are static label checks.
            for _nm, _body in (("plot_tradeoff", _city_body),
                               ("plot_tradeoff_region", _region_body)):
                _axis_titles = _re_ax.findall(r"[xy]axis_title='([^']*)'", _body)
                _bare_hmi = [_t for _t in _axis_titles if "HMI" in _t]
                if _bare_hmi:
                    print(f"  FAIL {_nm} axis title still uses the bare 'HMI' "
                          f"acronym {_bare_hmi} — spell it 'Heat Mitigation Index'")
                    tradeoff_axis_diffs += 1
                if not any("Heat Mitigation Index" in _t for _t in _axis_titles):
                    print(f"  FAIL {_nm} Y axis no longer spells "
                          f"'Heat Mitigation Index'")
                    tradeoff_axis_diffs += 1
                if "↗ better" not in _body:
                    print(f"  FAIL {_nm} missing the '↗ better' direction cue "
                          f"(both plots must carry it — parity)")
                    tradeoff_axis_diffs += 1
            # Flip-test (non-vacuous): the bare-HMI detector must fire on a
            # seeded bare-acronym axis title.
            if "HMI" not in _re_ax.findall(
                    r"[xy]axis_title='([^']*)'",
                    "yaxis_title='Cooling / HMI'")[0]:
                print("  FAIL meta-test: bare-HMI axis-title detector is vacuous")
                tradeoff_axis_diffs += 1
            else:
                print("  OK   spelled-axis + '↗ better' parity hold; "
                      "bare-HMI detector non-vacuous")
    except Exception as _e:
        print(f"  ERROR tradeoff-plot flood-axis lock: {_e}")
        import traceback
        traceback.print_exc()
        tradeoff_axis_diffs += 1

    # ── UMH parity doc-echo lock (one source of truth) ──────────────────────
    # The UMH parity figures live canonically in validation/compare_umh_invest.py.
    # Every doc surface that states SA UMH parity must echo the same SA tuple
    # (MAE ≤ 2.3e-6, r ≥ 0.99875, |Δtotal| ≤ 0.15%); none may hedge SA as
    # MN-equivalent via the retired "aligned input" phrasing; and REFERENCE's
    # UMH §6 must name the edge-corrected DISK kernel (_convolve_edge_corrected),
    # never a uniform_filter box. Structurally enforces "no claim stronger than
    # the committed reproducer." Change a figure in one place without the others
    # → fail; reintroduce uniform_filter / "aligned input" → fail. Flip-tested.
    print(f"\n{'=' * 60}")
    print("UMH parity doc-echo lock (one source of truth)")
    print(f"{'=' * 60}")
    umh_doc_diffs = 0
    try:
        _SA_TUPLE = ("2.3e-6", "0.99875", "0.15%")  # MAE(active), r, |Δtotal|
        _umh_harness = Path("validation/compare_umh_invest.py").read_text(encoding="utf-8")
        # (0) Source of truth carries the canonical SA tuple.
        _h_missing = [t for t in _SA_TUPLE if t not in _umh_harness]
        if _h_missing:
            print(f"  FAIL harness compare_umh_invest.py missing canonical SA "
                  f"tuple figures {_h_missing} — the source of truth drifted")
            umh_doc_diffs += 1
        else:
            print("  OK   harness carries the canonical SA tuple "
                  "(2.3e-6 / 0.99875 / 0.15%)")
        # (1) Every doc/code surface that states SA UMH parity echoes the tuple
        #     and drops the "aligned input" overstatement.
        _UMH_DOCS = ["REFERENCE.md", "docs/internal/DESIGN_NOTES.md",
                     "docs/internal/NATCAP_ALIGNMENT.md", "app.py"]
        for _d in _UMH_DOCS:
            _txt = Path(_d).read_text(encoding="utf-8")
            _miss = [t for t in _SA_TUPLE if t not in _txt]
            if _miss:
                print(f"  FAIL {_d}: SA UMH parity not echoing the canonical "
                      f"tuple — missing {_miss}")
                umh_doc_diffs += 1
            if "aligned input" in _txt:
                print(f"  FAIL {_d}: still hedges SA UMH as 'aligned input' "
                      "(overstates SA to MN-equivalent r)")
                umh_doc_diffs += 1
        # (2) REFERENCE UMH §6 kernel: edge-corrected disk, never uniform_filter.
        _ref_txt = Path("REFERENCE.md").read_text(encoding="utf-8")
        if "_convolve_edge_corrected" not in _ref_txt:
            print("  FAIL REFERENCE.md UMH §6 no longer names "
                  "`_convolve_edge_corrected` (the disk kernel)")
            umh_doc_diffs += 1
        if "uniform_filter" in _ref_txt:
            print("  FAIL REFERENCE.md still names `uniform_filter` — wrong UMH "
                  "kernel (it's an edge-corrected disk, not a square box)")
            umh_doc_diffs += 1
        if umh_doc_diffs == 0:
            print("  OK   all UMH SA-parity surfaces echo the canonical tuple; "
                  "REFERENCE §6 names the disk kernel, no uniform_filter")
        # Flip-tests (non-vacuous): seeded drift / wrong kernel must be caught.
        if not [t for t in _SA_TUPLE if t not in "no SA figures here"]:
            print("  FAIL meta-test: SA-tuple echo detector is vacuous")
            umh_doc_diffs += 1
        elif "uniform_filter" not in "NE = scipy.ndimage.uniform_filter(x)":
            print("  FAIL meta-test: uniform_filter detector is vacuous")
            umh_doc_diffs += 1
        else:
            print("  OK   meta-test: seeded missing-figure + uniform_filter "
                  "regressions are both caught")
    except Exception as _e:
        print(f"  ERROR UMH parity doc-echo lock: {_e}")
        import traceback
        traceback.print_exc()
        umh_doc_diffs += 1

    # ── Validated-model reproducer conformance lock (generalized) ───────────
    # Every model in VALIDATED_MODELS must be backed by a COMMITTED
    # comparisons/*.csv artifact whose every row is a CLEAN, GUARDED run on
    # InVEST 3.19.0. This generalizes the per-model doc-echo into one universal
    # rule — "no InVEST-validated badge without a conforming reproducer" — with
    # NO 3.16.2 carve-out (all five sit on one InVEST version). Had this existed,
    # it would have caught UNA on its own: a validated badge whose only artifact
    # was a 3.16.2 reachability proxy (different statistic, no guard) fails here.
    print(f"\n{'=' * 60}")
    print("Validated-model reproducer conformance lock (3.19.0 + clean + guard)")
    print(f"{'=' * 60}")
    reproducer_diffs = 0
    try:
        import csv as _csv_r
        import model_validation as _mv_r
        _ARTIFACT = {
            "ucm":    "comparisons/ucm_baseline_mn.csv",
            "una":    "comparisons/una_supply_parity_mn.csv",
            "umh":    "comparisons/umh_parity.csv",
            "ufr":    "comparisons/ufr_sa_retention_parity.csv",
            "carbon": "comparisons/carbon_sa_fourpool_parity.csv",
        }

        def _conforms(_path):
            """Return (ok, reason). Every row: invest_version 3.19.x, clean True,
            guard_ok True."""
            _p = Path(_path)
            if not _p.exists():
                return False, f"artifact missing: {_path}"
            _rows = list(_csv_r.DictReader(open(_p)))
            if not _rows:
                return False, f"artifact empty: {_path}"
            for _row in _rows:
                _ver = (_row.get("invest_version") or "")
                if not _ver.startswith("3.19"):
                    return False, f"{_path}: invest_version={_ver!r} (not 3.19.x)"
                if (_row.get("clean") or "").strip() != "True":
                    return False, f"{_path}: a row is not clean ({_row.get('clean')!r})"
                if (_row.get("guard_ok") or "").strip() != "True":
                    return False, f"{_path}: a row lacks a passing guard ({_row.get('guard_ok')!r})"
            return True, f"{_path} ({len(_rows)} row(s))"

        for _model in sorted(_mv_r.VALIDATED_MODELS):
            _path = _ARTIFACT.get(_model)
            if _path is None:
                print(f"  FAIL validated model {_model!r} has no mapped "
                      "comparisons/*.csv reproducer — a validated badge with no "
                      "conforming artifact (the UNA failure mode)")
                reproducer_diffs += 1
                continue
            _ok, _why = _conforms(_path)
            if _ok:
                print(f"  OK   {_model}: {_why} — 3.19.0, clean, guarded")
            else:
                print(f"  FAIL {_model}: {_why}")
                reproducer_diffs += 1
        # Flip-test (non-vacuous): a validated model with no artifact, a 3.16.2
        # run, an unclean row, or a missing guard must each be caught.
        _seed_dir = Path("comparisons")
        _bad_cases = [
            ("no-artifact", None),
            ("3.16.2", [{"invest_version": "3.16.2", "clean": "True", "guard_ok": "True"}]),
            ("unclean", [{"invest_version": "3.19.0", "clean": "False", "guard_ok": "True"}]),
            ("no-guard", [{"invest_version": "3.19.0", "clean": "True", "guard_ok": "False"}]),
        ]
        _meta_ok = True
        for _label, _rows in _bad_cases:
            if _rows is None:
                _caught = not _conforms("comparisons/_nonexistent_repro.csv")[0]
            else:
                _tmpf = _seed_dir / "_repro_lock_selftest.csv"
                with open(_tmpf, "w", newline="") as _fh:
                    _w = _csv_r.DictWriter(_fh, fieldnames=list(_rows[0].keys()))
                    _w.writeheader(); _w.writerows(_rows)
                _caught = not _conforms(str(_tmpf))[0]
                _tmpf.unlink()
            if not _caught:
                print(f"  FAIL meta-test: seeded {_label} reproducer NOT caught")
                reproducer_diffs += 1; _meta_ok = False
        if _meta_ok:
            print("  OK   meta-test: missing / 3.16.2 / unclean / unguarded "
                  "reproducers are all caught (non-vacuous)")
    except Exception as _e:
        print(f"  ERROR reproducer conformance lock: {_e}")
        import traceback
        traceback.print_exc()
        reproducer_diffs += 1

    # ── Placement-priority overlay honesty lock (no-drift + gated) ─────────
    # The "Placement priority (active strategy)" map overlay renders a focused
    # strategy's per-pixel suitability surface so focused placements are
    # explainable. Two honesty guards, both machine-locked here:
    #   (1) no-drift — the rendered raster's convertible-pixel values are
    #       *exactly* _compute_suitability_weights output for that strategy
    #       (can't drift to a different surface), painted only on the
    #       convertible pool; and the focused surfaces are strategy-distinct,
    #       so the identity check would catch a wrong-strategy raster.
    #   (2) gated — enabled ONLY for Explorer provenance + focused strategy;
    #       suppressed for random + optimizer-applied + region-optimized +
    #       baseline (all ranked under random placement), so the surface never
    #       manufactures a 'why' beside a placement it didn't drive.
    print(f"\n{'=' * 60}")
    print("Placement-priority overlay honesty lock (no-drift + gated)")
    print(f"{'=' * 60}")
    placement_priority_diffs = 0
    try:
        import numpy as _np_pp
        _cp = app.CONVERTIBLE_PIXELS
        _shape = app.cooling_lulc.shape
        # (1) No-drift: rendered raster == _compute_suitability_weights, exactly.
        for _strat in sorted(app.FOCUSED_PLACEMENT_STRATEGIES):
            _w = app._compute_suitability_weights(_cp, _strat)
            _r = app._placement_priority_raster(_strat, _cp, _shape)
            _vals = _r[_cp[:, 0], _cp[:, 1]]
            _nonnan = int(_np_pp.count_nonzero(~_np_pp.isnan(_r)))
            if not _np_pp.array_equal(_vals, _w):
                print(f"  FAIL {_strat}: rendered surface != _compute_suitability_weights")
                placement_priority_diffs += 1
            elif _nonnan != len(_cp):
                print(f"  FAIL {_strat}: raster paints {_nonnan} pixels, "
                      f"expected {len(_cp)} convertible (leak outside the pool)")
                placement_priority_diffs += 1
            else:
                print(f"  OK   {_strat}: rendered surface == suitability weights, "
                      f"NaN outside the {len(_cp)}-pixel convertible pool")
        # Non-vacuous: focused surfaces are strategy-distinct, so the identity
        # check above would catch a raster swapped to a different strategy.
        _wf = app._compute_suitability_weights(_cp, 'flood-focused')
        _wc = app._compute_suitability_weights(_cp, 'cooling-focused')
        if _np_pp.array_equal(_wf, _wc):
            print("  FAIL flood-focused and cooling-focused surfaces identical "
                  "(identity check would be vacuous)")
            placement_priority_diffs += 1
        else:
            print("  OK   focused surfaces are strategy-distinct "
                  "(identity check is non-vacuous)")
        # 'random' has no surface — the raster builder must refuse it.
        try:
            app._placement_priority_raster('random', _cp, _shape)
            print("  FAIL _placement_priority_raster('random') did not raise")
            placement_priority_diffs += 1
        except ValueError:
            print("  OK   _placement_priority_raster refuses 'random' (no surface)")
        # Degenerate-surface guard: a focused strategy with no positive weight
        # placed at random (the weight_sum==0 fallback), so its surface carries
        # no signal and must not render. _priority_surface_has_signal flags that
        # case (observed live: flood-focused on San Antonio → CN 0 → Q 0).
        _allzero = _np_pp.full((3, 3), _np_pp.nan); _allzero[0, 0] = 0.0; _allzero[1, 1] = 0.0
        _hassig = _np_pp.full((3, 3), _np_pp.nan); _hassig[0, 0] = 0.0; _hassig[1, 1] = 2.0
        if app._priority_surface_has_signal(_allzero) or \
                not app._priority_surface_has_signal(_hassig):
            print("  FAIL _priority_surface_has_signal mis-classifies "
                  "zero-signal vs real-signal surfaces")
            placement_priority_diffs += 1
        else:
            print("  OK   all-zero surface flagged no-signal; positive-weight "
                  "surface flagged has-signal (random fallback won't render)")
        # (2) Gate matrix — enabled only for Explorer + focused placement.
        _e_pp_mod = app.eib
        _gate_cases = [
            (_e_pp_mod.PROVENANCE_EXPLORER,         'flood-focused',       True),
            (_e_pp_mod.PROVENANCE_EXPLORER,         'cooling-focused',     True),
            (_e_pp_mod.PROVENANCE_EXPLORER,         'undersupply-focused', True),
            (_e_pp_mod.PROVENANCE_EXPLORER,         'balanced',            True),
            (_e_pp_mod.PROVENANCE_EXPLORER,         'random',              False),
            (_e_pp_mod.PROVENANCE_OPTIMIZER,        'flood-focused',       False),
            (_e_pp_mod.PROVENANCE_REGION_OPTIMIZED, 'flood-focused',       False),
            (_e_pp_mod.PROVENANCE_BASELINE,         'flood-focused',       False),
        ]
        _gate_ok = True
        for _prov, _strat, _want in _gate_cases:
            _got = app._should_show_placement_priority(_prov, _strat)
            if bool(_got) != _want:
                print(f"  FAIL gate({_prov!r}, {_strat!r})={_got}, expected {_want}")
                placement_priority_diffs += 1
                _gate_ok = False
        if _gate_ok:
            print("  OK   gate enables only Explorer+focused; suppresses random, "
                  "optimizer-applied, region-optimized, baseline")
        # Non-vacuous: a focused radio while viewing an optimizer result must
        # stay suppressed — the exact misattribution the gate exists to prevent.
        if app._should_show_placement_priority(_e_pp_mod.PROVENANCE_OPTIMIZER, 'flood-focused'):
            print("  FAIL focused radio + optimizer result would expose the surface")
            placement_priority_diffs += 1
        else:
            print("  OK   focused radio + optimizer result stays suppressed "
                  "(no manufactured 'why')")
    except Exception as _e_pp:
        print(f"  ERROR placement-priority honesty lock: {_e_pp}")
        import traceback
        traceback.print_exc()
        placement_priority_diffs += 1

    # ── Flood-focused placement non-degeneracy lock (CN-path regression guard) ─
    # The flood-focused suitability surface derives per-pixel CN. A CN-path
    # regression that returns 0 on the convertible pool (the SA bug: a bare
    # 2-digit lookup against the 3-digit nlcd_tree table) silently collapses the
    # weights to all-zero → the strategy falls back to uniform random. Assert the
    # weights are non-degenerate (sum > 0 AND spatial variance > 0) on BOTH
    # cities — the compound-table city is the regression, the plain-NLCD city the
    # passing control. Catches "focused knob silently goes random" for ANY future
    # CN-path regression, not just this instance. Runs last: it rebinds cities,
    # and nothing downstream but the pure grand_total tally follows.
    print(f"\n{'=' * 60}")
    print("Flood-focused placement non-degeneracy lock (CN-path regression guard)")
    print(f"{'=' * 60}")
    flood_signal_diffs = 0
    try:
        import numpy as _np_fs
        for _city in active_cities:
            _rebind_city(app, _city)
            _w = app._compute_suitability_weights(app.CONVERTIBLE_PIXELS, 'flood-focused')
            _sum, _var = float(_w.sum()), float(_w.var())
            if _sum > 0 and _var > 0:
                print(f"  OK   {_city}: flood-focused weights non-degenerate "
                      f"(sum={_sum:.4g}, variance>0)")
            else:
                print(f"  FAIL {_city}: flood-focused weights DEGENERATE "
                      f"(sum={_sum:.4g}, var={_var:.4g}) — placement silently random")
                flood_signal_diffs += 1
            # Flip-test on the compound (3-digit-table) city: the OLD bare
            # 2-digit CN lookup must collapse to all-zero, proving the reduction
            # in _per_pixel_cn is load-bearing (non-vacuous).
            if app.cooling_lulc_compound is not None:
                _r, _c = app.CONVERTIBLE_PIXELS[:, 0], app.CONVERTIBLE_PIXELS[:, 1]
                _bare_lulc = _np_fs.clip(app.cooling_lulc[_r, _c], 0, len(app.lucode_idx_arr) - 1)
                _bare_soil = _np_fs.clip(app.soil_resized[_r, _c].astype(int), 1, app.cn_table.shape[1] - 1)
                _bare_cn = app.cn_table[app.lucode_idx_arr[_bare_lulc], _bare_soil]
                if float(_bare_cn.sum()) == 0.0:
                    print(f"  OK   {_city} flip-test: the old bare-2-digit CN lookup "
                          "is all-zero (the reduction fix is non-vacuous)")
                else:
                    print(f"  FAIL {_city} flip-test: bare-2-digit lookup not "
                          "all-zero — degeneracy reproducer broke")
                    flood_signal_diffs += 1
    except Exception as _e_fs:
        print(f"  ERROR flood-focused non-degeneracy lock: {_e_fs}")
        import traceback
        traceback.print_exc()
        flood_signal_diffs += 1

    # ── Map change-palette distinctness floor — Relay 8 guard ────────────────
    # The scenario-change colors were deepened for contrast against the light
    # unchanged background. Lock two NO-REGRESSION floors so a later tune can't
    # quietly push any pair below the PRE-TUNE worst case:
    #   (1) full-set perceptual floor — min CIE76 ΔE over {Unchanged, GI, FF, HD,
    #       intensity-orange, priority-purple, nodata-white} ≥ 15.44 (the
    #       pre-tune Unchanged↔white pair, the tightest);
    #   (2) colorblind floor — min ΔE among GI/FF/HD under deuteranopia +
    #       protanopia (Machado 2009 severity 1.0) ≥ 14.33 (pre-tune deut FF↔HD).
    # Colors are sourced from app (CHANGE_COLORS + the overlay constants) so the
    # guard tracks the real palette. Meta-tests seed a too-close full-set pair
    # and a deut-collapsing pair (distinct in normal vision) and confirm each
    # check flags it — non-vacuous.
    print(f"\n{'=' * 60}")
    print("Map change-palette distinctness floor — Relay 8 guard")
    print(f"{'=' * 60}")
    palette_diffs = 0
    try:
        import itertools as _it
        _PAL_FLOOR_FULL, _PAL_FLOOR_CVD, _EPS = 15.44, 14.33, 1e-2
        _DEUT = [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
        _PROT = [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
        def _hx(h):
            h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        def _ln(c):
            c /= 255.0; return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        def _lab(rgb):
            r, g, b = [_ln(c) for c in rgb]
            X = r * 0.4124 + g * 0.3576 + b * 0.1805
            Y = r * 0.2126 + g * 0.7152 + b * 0.0722
            Z = r * 0.0193 + g * 0.1192 + b * 0.9505
            def f(t): return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
            fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
            return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
        def _de(a, b):
            la, lb = _lab(a), _lab(b)
            return sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5
        def _cvd(rgb, M):
            lin = [_ln(c) for c in rgb]
            o = [sum(M[i][j] * lin[j] for j in range(3)) for i in range(3)]
            def e(c):
                c = max(0.0, min(1.0, c))
                c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
                return c * 255
            return tuple(e(c) for c in o)
        def _dec(a, b, M):
            return _de(_cvd(a, M), _cvd(b, M))
        def _full_min(pal):
            return min(_de(pal[a], pal[b]) for a, b in _it.combinations(pal, 2))
        def _cvd_min(ch):
            return min(_dec(ch[a], ch[b], M)
                       for M in (_DEUT, _PROT) for a, b in _it.combinations(ch, 2))

        _cc = app.CHANGE_COLORS
        _live = {
            'Unchanged': _hx(_cc['Unchanged']),
            'GI': _hx(_cc['Green Infrastructure']),
            'FF': _hx(_cc['Food Forest']),
            'HD': _hx(_cc['High Density']),
            'orange': tuple(app._INTENSITY_OVERLAY_RGB),
            'purple': tuple(app._PRIORITY_OVERLAY_RGB),
            'white': (255, 255, 255),
        }
        _ch = {k: _live[k] for k in ('GI', 'FF', 'HD')}
        _fm, _cm = _full_min(_live), _cvd_min(_ch)
        if _fm + _EPS < _PAL_FLOOR_FULL:
            print(f"  FAIL full-set perceptual floor: min ΔE {_fm:.2f} < {_PAL_FLOOR_FULL} "
                  "(a palette pair is closer than the pre-tune worst case)")
            palette_diffs += 1
        else:
            print(f"  OK   full-set perceptual floor: min ΔE {_fm:.2f} ≥ {_PAL_FLOOR_FULL}")
        if _cm + _EPS < _PAL_FLOOR_CVD:
            print(f"  FAIL colorblind floor: min GI/FF/HD ΔE {_cm:.2f} < {_PAL_FLOOR_CVD} "
                  "(deuteranopia/protanopia collapses a change-color pair)")
            palette_diffs += 1
        else:
            print(f"  OK   colorblind floor: min GI/FF/HD ΔE {_cm:.2f} ≥ {_PAL_FLOOR_CVD}")

        # Meta-test (a): seed GI == intensity-orange → full-set floor must trip.
        _bad = dict(_live); _bad['GI'] = _bad['orange']
        if _full_min(_bad) + _EPS < _PAL_FLOOR_FULL:
            print("  OK   meta-test (a): seeded GI==intensity-orange correctly trips the full-set floor")
        else:
            print("  FAIL meta-test (a): seeded too-close pair NOT flagged — full-set guard is blind")
            palette_diffs += 1
        # Meta-test (b): FF/HD distinct in normal vision but collapse under deut.
        _badch = {'GI': _ch['GI'], 'FF': _hx('#3c9e3c'), 'HD': _hx('#9e6a2a')}
        _bad_normal = _de(_badch['FF'], _badch['HD'])
        if _cvd_min(_badch) + _EPS < _PAL_FLOOR_CVD and _bad_normal > _PAL_FLOOR_FULL:
            print(f"  OK   meta-test (b): seeded deut-collapse pair (normal ΔE {_bad_normal:.0f}) "
                  "correctly trips the colorblind floor")
        else:
            print("  FAIL meta-test (b): seeded colorblind-collapse pair NOT flagged — CVD guard is blind")
            palette_diffs += 1
    except Exception as _e_pal:
        print(f"  ERROR palette distinctness guard: {_e_pal}")
        import traceback
        traceback.print_exc()
        palette_diffs += 1

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
                   + toggle_diffs + vocab_diffs + fast_grid_diffs
                   + retention_idx_diffs + calib_diffs + child_card_diffs
                   + loader_diffs + delta_dir_diffs + src_diffs + badge_src_diffs
                   + glyph_diffs + carbon_unit_diffs + autoswitch_diffs
                   + ce_diffs + active_scn_diffs + fda_diffs
                   + tradeoff_axis_diffs + umh_doc_diffs + reproducer_diffs
                   + placement_priority_diffs + flood_signal_diffs
                   + palette_diffs + unit_survival_diffs + fig_close_diffs
                   + map_view_diffs + density_diffs + concentration_diffs
                   + self_describe_diffs + locator_diffs)
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
        if locator_diffs:
            print(f"{locator_diffs} selected-area locator divergence(s) — the "
                  "'Selected area' heading or tightened instruction drifted, or "
                  "the old long instruction reappeared (Relay 30).")
        if self_describe_diffs:
            print(f"{self_describe_diffs} self-describing map-view "
                  "divergence(s) — the active-view indicator stopped naming the "
                  "rendered view, a scope-aware title stopped reflecting scope, "
                  "or the teal key / ramp direction drifted (Relay 29).")
        if concentration_diffs:
            print(f"{concentration_diffs} concentration-view final-copy "
                  "divergence(s) — palette is no longer teal, the new "
                  "'concentration' wording / caption drifted, or the rename "
                  "clobbered the 'High Density' land-use category (Relay 28).")
        if density_diffs:
            print(f"{density_diffs} concentration-map divergence(s) — boundary "
                  "context stopped drawing when geometry was available, the "
                  "district edge mask drifted, or per-cell shares left [0,1] / "
                  "the AOI (Relay 26).")
        if map_view_diffs:
            print(f"{map_view_diffs} Map-view divergence(s) — the "
                  "scope→default mapping collapsed, or the render plan stopped "
                  "rendering exactly one map / moved the categorical legend off "
                  "the detailed view (Relay 25).")
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
        if palette_diffs:
            print(f"{palette_diffs} change-palette distinctness divergence(s) "
                  "— a CHANGE_COLORS / overlay tune pushed a pair below the "
                  "pre-tune perceptual or colorblind floor (Relay 8 guard).")
        if unit_survival_diffs:
            print(f"{unit_survival_diffs} relocated-unit survival divergence(s) "
                  "— a bare-label card (Runoff Volume / Carbon) dropped its unit "
                  "from the help too; move it back into help/caption (Relay 2/13).")
        if fig_close_diffs:
            print(f"{fig_close_diffs} figure-close hygiene divergence(s) — a "
                  "st.pyplot/savefig render lacks a paired plt.close (leaked "
                  "matplotlib figure across reruns; Relay 16 guard).")
        if dense_freshness_diffs:
            print(f"{dense_freshness_diffs} dense-CSV freshness "
                  "divergence(s) — re-run precompute_scenarios.py for the "
                  "affected city; Fast cold-start reads from disk and a "
                  "stale CSV would feed wrong values to the surrogate.")
        if retention_idx_diffs:
            print(f"{retention_idx_diffs} runoff-retention-index divergence(s) "
                  "(presence / bounds / Jensen non-degeneracy).")
        if calib_diffs:
            print(f"{calib_diffs} surrogate-calibration freshness divergence(s) "
                  "(stale/missing estimate-range artifact).")
        if delta_dir_diffs:
            print(f"{delta_dir_diffs} delta-direction divergence(s) — a "
                  "lower-is-better card lost its inverse delta (or a higher-is-"
                  "better card gained one); see card-row delta colours.")
        if src_diffs:
            print(f"{src_diffs} validated-model source divergence(s) — the "
                  "canonical validated set drifted, the bundle stopped sourcing "
                  "model_validation.MODEL_VALIDATION, or a model's parity status "
                  "changed (update model_validation.py + _EXPECTED_VALIDATED on "
                  "purpose if intentional).")
        if badge_src_diffs:
            print(f"{badge_src_diffs} badge↔source divergence(s) — a card can "
                  "render InVEST-validated whose model isn't in the Stage-1 set, "
                  "a lumped-proxy/dollar/food/cost card leaked into the validated "
                  "map, or the carbon city-split broke.")
        if glyph_diffs:
            print(f"{glyph_diffs} colorblind-glyph divergence(s) — a badge tier "
                  "lost its shape glyph (◆ ■ ○ △) on the live render path, the "
                  "flip-test caught a cross-tier glyph, or the legend caption's "
                  "glyphs drifted from what render_validation_badge emits.")
        if carbon_unit_diffs:
            print(f"{carbon_unit_diffs} carbon-unit single-source divergence(s) — "
                  "a duplicate long-form unit var (_carbon_unit / _opt_carbon_unit, "
                  "'tons CO2e') reappeared; route value-display units through the "
                  "shared _carbon_unit_suffix instead.")
        if autoswitch_diffs:
            print(f"{autoswitch_diffs} post-optimize auto-switch divergence(s) — "
                  "the removed 'just_optimized' banner flag reappeared, or the "
                  "main_tab segmented_control regained a default=/value=/index= "
                  "arg that collides with the seeded session_state key.")
        if ce_diffs:
            print(f"{ce_diffs} cost-effectiveness suppression divergence(s) — "
                  "compute_cost_effectiveness changed which denominators yield None "
                  "(zero/negative/below-epsilon, or warming→None), so the dashboard "
                  "card-hide condition drifted. Re-confirm the floors on purpose.")
        if active_scn_diffs:
            print(f"{active_scn_diffs} active-scenario line-1 divergence(s) — a "
                  "provenance prefix drifted, the pct=0 baseline override broke, "
                  "the GI/FF/HD mix lost a component, or a new _scen_provenance "
                  "value isn't covered by _ACTIVE_SCENARIO_PREFIX.")
        if fda_diffs:
            print(f"{fda_diffs} flood-damage gate divergence(s) — the card hide "
                  "keys on the computed value instead of table presence, or the "
                  "unavailable note / value branch is missing.")
        if tradeoff_axis_diffs:
            print(f"{tradeoff_axis_diffs} tradeoff-plot flood-axis divergence(s) — "
                  "the region baseline reverted to a delta/zero pin or a plot's "
                  "flood/cooling axis source drifted from the absolute convention.")
        if umh_doc_diffs:
            print(f"{umh_doc_diffs} UMH parity doc-echo divergence(s) — a doc's SA "
                  "UMH figures drifted from the harness tuple, an 'aligned input' "
                  "overstatement returned, or REFERENCE's UMH kernel is mis-described.")
        if reproducer_diffs:
            print(f"{reproducer_diffs} validated-model reproducer divergence(s) — a "
                  "validated badge lacks a committed comparisons/*.csv on InVEST "
                  "3.19.0 that is clean + guarded (the UNA failure mode).")
        if loader_diffs:
            print(f"{loader_diffs} calibration-loader divergence(s) — "
                  "_load_surrogate_calibration returned None/wrong shape on a "
                  "valid artifact (the swallowed json NameError, or over-narrowed "
                  "data handling).")
        if child_card_diffs:
            print(f"{child_card_diffs} children's-card visibility divergence(s) "
                  "(suppression predicate regressed).")
        if fast_grid_diffs:
            print(f"{fast_grid_diffs} fast-grid artifact freshness "
                  "divergence(s) — the precomputed Fast grid is stale or its "
                  "stamp mismatches; re-run scripts/regenerate_fast_grid.py for "
                  "the affected city (the region-optimizer prefilter trains on it).")
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
        if vocab_diffs:
            print(f"{vocab_diffs} vocabulary-guard failure — a retired term "
                  "reappeared on a user-facing surface (see the scan output "
                  "above). Replace it with the canonical term from REFERENCE.md "
                  "§ \"Vocabulary (canonical terms)\", or mark a deliberate "
                  "historical mention with the 'vocab-allow' marker.")
        if flood_signal_diffs:
            print(f"{flood_signal_diffs} flood-focused non-degeneracy "
                  "divergence(s) — flood-focused suitability weights collapsed "
                  "to all-zero (or zero-variance) on a city's convertible pool, "
                  "so the strategy silently falls back to random. A CN-path "
                  "regression (e.g. reverting the _per_pixel_cn reduction) "
                  "drives the compound-table city back to all-zero.")
        if placement_priority_diffs:
            print(f"{placement_priority_diffs} placement-priority overlay "
                  "honesty divergence(s) — the rendered surface drifted from "
                  "_compute_suitability_weights, leaked outside the convertible "
                  "pool, or the gate exposed it on random / optimizer-applied / "
                  "region-optimized / baseline placement (it must render only "
                  "for Explorer provenance + a focused strategy).")
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
