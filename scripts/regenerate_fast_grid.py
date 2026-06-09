"""Regenerate the precomputed Fast scenario grid (Relay 35).

The region-constrained optimizer's prefilter always trains a Fast surrogate on a
~90-recipe grid (step_pct=10, step_alloc=25). Building that grid live is ~96 s on
San Antonio (90 full engine evals) — the first-region-Optimize cliff. This script
builds the grid offline and writes it as a versioned CSV + sidecar stamp, so the
live path loads (≈0 s) and trains the RF (≈0.05 s) instead of rebuilding.

Mirrors the dense Balanced CSV convention (data/scenarios_dense_<city>.csv) +
adds a provenance sidecar the dense CSV lacks (city/params/schema), so staleness
is detectable both at runtime (cheap stamp check → fall back to live build) and at
the gate (verify_baselines spot-checks recipe values against a fresh engine eval).

Usage (same venv + PROJ/GDAL override as verify_baselines):
  PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
  GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
  .venv/bin/python scripts/regenerate_fast_grid.py --city "San Antonio, TX"

Writes the path configured in CITIES[city]['fast_grid_file'] (+ '<path>.meta.json').
"""
import os
import sys
import json
import time
import argparse

# Repo root on sys.path so `import app` (app.py at root) resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--city", default="San Antonio, TX",
                     help="City key (default: San Antonio, TX).")
_parser.add_argument("--output", default=None,
                     help="CSV path (default: CITIES[city]['fast_grid_file']).")
_args = _parser.parse_args()
CITY_KEY = _args.city

# Fast-grid build parameters — must match _cached_fast_scenario_grid in app.py.
STEP_PCT = 10
STEP_ALLOC = 25
FAST_GRID_FORMAT_VERSION = 1


# ── streamlit stub (cache passthrough; same convention as precompute_scenarios) ──
class _SessionStateStub:
    _store = {}
    def get(self, key, default=None): return self._store.get(key, default)
    def pop(self, key, *args): return self._store.pop(key, *args) if args else self._store.pop(key, None)
    def setdefault(self, key, default=None): return self._store.setdefault(key, default)
    def __getattr__(self, name):
        if name == "_store": return object.__getattribute__(self, "_store")
        return self._store.get(name)
    def __getitem__(self, key): return self._store.get(key)
    def __setitem__(self, key, value): self._store[key] = value
    def __setattr__(self, name, value):
        if name == "_store": object.__setattr__(self, name, value)
        else: self._store[name] = value
    def __contains__(self, key): return key in self._store
    def keys(self): return list(self._store.keys())


class _StubSt:
    def __getattr__(self, name):
        if name in ("cache_data", "cache_resource"): return self._cache
        if name == "columns": return self._columns
        if name == "tabs": return self._tabs
        if name == "selectbox":
            def _sb(label, options, **kw):
                if not options: return None
                if "City" in str(label):
                    for o in options:
                        if o == CITY_KEY or o == f"{CITY_KEY} (coming soon)": return o
                return options[0]
            return _sb
        if name == "radio": return lambda label, options, **kw: options[0] if options else None
        if name == "multiselect": return lambda label, options=(), default=None, **kw: list(default or [])
        if name == "slider": return lambda *a, **kw: kw.get("value", a[3] if len(a) >= 4 else 0)
        if name == "number_input": return lambda *a, **kw: kw.get("value", a[3] if len(a) >= 4 else 0)
        if name == "text_input": return lambda *a, **kw: kw.get("value", "")
        if name == "text_area": return lambda *a, **kw: kw.get("value", "")
        if name in ("toggle", "checkbox", "button"): return lambda *a, **kw: False
        if name == "session_state": return _SessionStateStub()
        return self
    def _cache(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs: return args[0]
        return lambda f: f
    def _columns(self, spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_StubSt() for _ in range(n))
    def _tabs(self, labels, *args, **kwargs): return tuple(_StubSt() for _ in labels)
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *e): return False
    def __getitem__(self, k): return self
    def __setitem__(self, k, v): pass
    def __setattr__(self, n, v): pass
    def __contains__(self, k): return False
    def __iter__(self): return iter([])
    def __bool__(self): return True


_SessionStateStub._store['entry_city'] = CITY_KEY
sys.modules["streamlit"] = _StubSt()

print(f"Building Fast grid for {CITY_KEY} (step_pct={STEP_PCT}, step_alloc={STEP_ALLOC})...")
_t0 = time.perf_counter()
import app  # noqa: E402
print(f"  import app: {time.perf_counter() - _t0:.1f}s")

if CITY_KEY not in app.CITIES:
    sys.exit(f"ERROR: unknown city {CITY_KEY!r}. Known: {list(app.CITIES)}")

OUT_PATH = _args.output or app.CITIES[CITY_KEY].get("fast_grid_file")
if not OUT_PATH:
    sys.exit(f"ERROR: no fast_grid_file configured for {CITY_KEY!r} and no --output given.")

state = app._CURRENT_CITY_STATE
_t1 = time.perf_counter()
grid = app.compute_scenario_grid(
    state, CITY_KEY, app.DATA_DIR_FLOOD, app.DATA_DIR_COOLING,
    step_pct=STEP_PCT, step_alloc=STEP_ALLOC,
)
_build_s = time.perf_counter() - _t1
print(f"  grid build: {_build_s:.1f}s  ({len(grid)} recipes)")

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
grid.to_csv(OUT_PATH, index=False)

meta = {
    "fast_grid_format_version": FAST_GRID_FORMAT_VERSION,
    "city_key": CITY_KEY,
    "step_pct": STEP_PCT,
    "step_alloc": STEP_ALLOC,
    "scenario_schema_version": int(app.SCENARIO_SCHEMA_VERSION),
    "n_recipes": int(len(grid)),
    "columns": list(grid.columns),
}
META_PATH = OUT_PATH + ".meta.json"
with open(META_PATH, "w") as _f:
    json.dump(meta, _f, indent=2, sort_keys=True)

print(f"  wrote {OUT_PATH} ({len(grid)} rows) + {META_PATH}")
print(f"  stamp: schema v{meta['scenario_schema_version']}, "
      f"format v{FAST_GRID_FORMAT_VERSION}")
print(f"DONE in {time.perf_counter() - _t0:.1f}s total "
      f"(build {_build_s:.1f}s — this is what the live first-click now skips).")
