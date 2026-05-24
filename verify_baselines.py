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


def _snapshot_from_results(results: dict) -> dict:
    snap = {}
    for k, v in sorted(results.items()):
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
    state = app_mod._load_city_runtime_state(city_name)
    city_cfg = app_mod.CITIES[city_name]

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
    app_mod._CURRENT_CITY_STATE = state

    # City-config scalars (lines 2054–2070 area)
    app_mod.PIXEL_AREA_ACRES     = city_cfg['pixel_area_acres']
    app_mod.FOOD_FOREST_LBS_ACRE = city_cfg['food_forest_lbs_acre']
    app_mod.UHI_MAX_C            = city_cfg['uhi_max_c']
    app_mod.HM_TO_FAHRENHEIT     = city_cfg['uhi_max_c'] * 1.8

    # Derived baselines (lines 1941–1943, 2084–2088)
    app_mod.BASELINE_NATURE_ACCESS_PCT, app_mod.BASELINE_NATURE_QUALITY_SCORE, _ = (
        app_mod.calculate_nature_access(state.cooling_lulc, state.pop_count_raster)
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

    print(f"\n{'=' * 60}")
    if total_diffs == 0:
        print("All baselines match.")
        return 0
    else:
        print(f"{total_diffs} total divergence(s). If intentional, rerun with --update.")
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
