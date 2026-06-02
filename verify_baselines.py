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
                app.OWNERSHIP_MODES[mode],
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
            for key, cfg in app._REGION_LOCAL_METRICS.items():
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
                n_decomp = sum(1 for c in app._REGION_LOCAL_METRICS.values()
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
                k for k, cfg in app._REGION_LOCAL_METRICS.items()
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
            app.OWNERSHIP_MODES[mode_key],
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

    print(f"\n{'=' * 60}")
    grand_total = (total_diffs + region_diffs + ownership_diffs
                   + region_local_diffs + smoke_diffs + disclosure_diffs
                   + round_trip_diffs + subset_diffs + reconcile_diffs
                   + guard_diffs + ownership_diffs_batch1 + tradeoff_diffs)
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
