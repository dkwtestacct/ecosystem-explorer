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
    # (UNA params + kernel were already rebound above, before
    # _load_city_runtime_state, to ensure baseline_una_supply_percapita_raster
    # is built under the correct city's kernel.)

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

    # Set the initial city for module-level import. The first available city
    # will be selected by the stub's selectbox.
    _DESIRED_CITY = "Minneapolis, MN"
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
    # Locked rasterized in-AOI per-class acres (frozen against the
    # post-split rasterization). The pre-split combined `school_university`
    # value (6,030 ac) splits into `school` (2,392) + `university` (3,583);
    # the 55-ac residual is the fall-through to private of names not
    # matching either ISD/SCHOOL DISTRICT or UNIVERSITY/COLLEGE.
    _RASTER_EXPECTED_AC = {
        'private':       507_165.0,
        'city':           41_044.0,
        'county':          2_886.0,
        'state_federal':  28_237.0,
        'school':          2_392.0,
        'unknown':        15_885.0,
        'university':      3_583.0,
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

    # ── Tradeoff Analysis tab section-order assertion ───────────────────────
    # The Tradeoff Analysis tab + the NatCap reference-scenario view each have
    # a locked section order. Explorer mode: Tradeoff Space (plot) → Compare
    # scenarios (table) → Neighborhood breakdown → optimizer / saved /
    # best-by-goal. NatCap mode: side-by-side (table) → notes / validation
    # (Tradeoff Space plot intentionally absent — its axes (Flood Retention,
    # HMI) have no published values for NatCap fixed scenarios). A reorder
    # regression (e.g. a future edit that moves "Compare scenarios" above
    # the plot) would flip the user-facing flow without changing any engine
    # output; this cell catches that by scanning app.py for ordered markers.
    print(f"\n{'=' * 60}")
    print("Tradeoff Analysis tab — section-order assertion")
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
            EXPECTED_TAB2_ORDER = [
                ("Tradeoff Space (plot)",
                 'st.subheader("Tradeoff space: current scenario vs alternatives"'),
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

    print(f"\n{'=' * 60}")
    grand_total = (total_diffs + region_diffs + ownership_diffs
                   + region_local_diffs + smoke_diffs + disclosure_diffs
                   + round_trip_diffs + subset_diffs + reconcile_diffs
                   + guard_diffs + ownership_diffs_batch1 + tradeoff_diffs
                   + region_opt_diffs + sidebar_keys_diffs
                   + scenario_state_diffs + section_order_diffs
                   + shared_fire_diffs + dollar_lint_diffs)
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
            print(f"{section_order_diffs} Tradeoff Analysis section-order "
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
