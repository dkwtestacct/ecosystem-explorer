"""RELAY 34 profiler (throwaway) — where does the first region-Optimize go?

Run (same venv + PROJ/GDAL override as verify_baselines):
  PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \
  GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \
  .venv/bin/python scripts/profile_region_optimize.py

Stubs streamlit (cache_data/cache_resource are PASSTHROUGH — see item 4 note),
imports app for the default city (San Antonio), and times the phases directly.
No Streamlit loop. Reports wall AND cpu time per phase so a busy-wait (wall >>
cpu) is distinguishable from genuine compute (cpu ≈ wall).
"""
import os
import sys
import time
import io
import cProfile
import pstats

import numpy as np

# Script lives under scripts/; put the repo root on sys.path so `import app`
# (app.py at repo root) resolves the same way it does for precompute_scenarios.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CITY_KEY = "San Antonio, TX"


# ── streamlit stub (copied from precompute_scenarios.py; cache = passthrough) ──
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


# Pre-seed entry_city so the splash/city-switch path resolves to SA at import
# (same trick verify_baselines uses).
_SessionStateStub._store['entry_city'] = CITY_KEY
sys.modules["streamlit"] = _StubSt()


def timed(label, fn):
    w0, c0 = time.perf_counter(), time.process_time()
    r = fn()
    w, c = time.perf_counter() - w0, time.process_time() - c0
    print(f"{label}\n    wall {w:7.2f}s | cpu {c:7.2f}s | cpu/wall {c/max(w,1e-9):5.0%}")
    return r, w, c


print("=" * 70)
print(f"RELAY 34 profile — {CITY_KEY}")
print("=" * 70)

_w0, _c0 = time.perf_counter(), time.process_time()
import app  # noqa: E402
print(f"import app (startup compute): wall {time.perf_counter()-_w0:.1f}s | "
      f"cpu {time.process_time()-_c0:.1f}s")

state = app._CURRENT_CITY_STATE
DF, DC = app.DATA_DIR_FLOOD, app.DATA_DIR_COOLING
print(f"loaded city: {getattr(state, 'city_key', '?')}  "
      f"(LULC shape {getattr(getattr(state,'lulc_arr',None),'shape','?')})")

# ===== 1. GRID BUILD =====
print("\n" + "-" * 70 + "\n1. GRID BUILD\n" + "-" * 70)
grid, gw, gc = timed(
    "compute_scenario_grid(step_pct=10, step_alloc=25)",
    lambda: app.compute_scenario_grid(state, CITY_KEY, DF, DC, step_pct=10, step_alloc=25),
)
n_rows = len(grid)
print(f"    grid rows (engine evals) = {n_rows}  |  avg/eval = {gw/n_rows:.3f}s")
_sur, tw, tc = timed(
    "train_surrogate(n_estimators=100)",
    lambda: app._train_surrogate_fn(grid, n_estimators=100),
)

# ===== 2. PER-EVAL HOTSPOT =====
print("\n" + "-" * 70 + "\n2. PER-EVAL HOTSPOT (cProfile one evaluate_scenario(20,50,0))\n" + "-" * 70)
pr = cProfile.Profile()
pr.enable()
app.evaluate_scenario(20, 50, 0, seed=42)
pr.disable()
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(22)
print("--- by CUMULATIVE time ---")
print(buf.getvalue())
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(12)
print("--- by TOTTIME (self time) ---")
print(buf.getvalue())

# ===== 3. REGION VERIFY =====
print("\n" + "-" * 70 + "\n3. REGION VERIFY\n" + "-" * 70)
layer = ('council_districts' if 'council_districts' in state.region_rasters
         else list(state.region_rasters)[0])
labels = list(state.region_layer_labels[layer])
raster = state.region_rasters[layer]
pick = labels[:3]
idx = [labels.index(l) for l in pick]
region_mask = np.isin(raster, idx)

from ownership import OWNERSHIP_MODES  # noqa: E402
own_cfg = OWNERSHIP_MODES.get('vacant')
if state.ownership_raster is not None:
    own_mask = app._build_ownership_mask(
        state.ownership_raster, state.ownership_vacant_raster, own_cfg)
    opt_mask = region_mask & own_mask
    own_note = f"∩ vacant ownership → opt_px={int(opt_mask.sum()):,}"
else:
    opt_mask = region_mask
    own_note = "(no ownership layer; region-only)"
print(f"    layer={layer}  districts={pick}")
print(f"    region_px={int(region_mask.sum()):,}  {own_note}")
print(f"    (note: per-eval cost is full-AOI compute + region_local aggregation; "
      f"mask size barely changes per-eval time)")

def engine_eval(p, g, f):
    return app.evaluate_scenario(
        p, g, f, seed=42, placement_strategy='random',
        selected_region_mask=opt_mask)

# The "first run only" prep the region-Optimize spinner shows:
(fast_df, fast_sur), pw, pc = timed(
    "_cached_fast_surrogate_for_region (prep: grid + train)",
    lambda: app._cached_fast_surrogate_for_region(state, CITY_KEY, DF, DC),
)
print(f"    fast_df rows = {len(fast_df)}")

import surrogate  # noqa: E402
_count = {'n': 0, 't': 0.0}
_kbox = {'K': None}
def engine_eval_timed(p, g, f):
    t0 = time.perf_counter()
    r = engine_eval(p, g, f)
    _count['t'] += time.perf_counter() - t0
    _count['n'] += 1
    return r
def _prog(i, K): _kbox['K'] = K

weights = {'mean_hm': 1.0, 'flood_reduction': 1.0, 'carbon_tons_co2': 1.0,
           'food_mln_lbs': 1.0, 'total_cost_mln': 1.0, 'runoff_acre_feet': 1.0}
region_out, rw, rc = timed(
    "optimize_scenario_region(k_engine=40, top_n=5)",
    lambda: surrogate.optimize_scenario_region(
        fast_sur, fast_df, engine_eval_timed, weights,
        k_engine=40, top_n=5, progress_cb=_prog),
)
print(f"    engine-verified evals = {_count['n']}  (progress_cb K = {_kbox['K']}; "
      f"K-cap was 40 → {'CAP BOUND' if _count['n'] >= 40 else 'cap NOT bound'})")
print(f"    avg/verify-eval = {_count['t']/max(_count['n'],1):.3f}s  |  "
      f"verify compute ≈ {_count['t']:.1f}s")
print(f"    rows returned = {0 if region_out is None else len(region_out)}")
print(f"    >>> FIRST region-Optimize total ≈ prep {pw:.1f}s + verify {rw:.1f}s "
      f"= {pw+rw:.1f}s")

# ===== 4. CACHE SANITY =====
print("\n" + "-" * 70 + "\n4. CACHE SANITY (grid build run2)\n" + "-" * 70)
_, gw2, _ = timed(
    "compute_scenario_grid run2",
    lambda: app.compute_scenario_grid(state, CITY_KEY, DF, DC, step_pct=10, step_alloc=25),
)
print(f"    run1={gw:.1f}s  run2={gw2:.1f}s")
print("    NOTE: in this stub @st.cache_data/@st.cache_resource are PASSTHROUGH "
      "(no memo), so run2 ≈ run1 here and does NOT reflect the live cache.")
import inspect
for fn_name in ("compute_scenario_grid", "_cached_fast_surrogate_for_region"):
    src = inspect.getsource(getattr(app, fn_name)).splitlines()
    deco = [l.strip() for l in src[:3] if l.strip().startswith("@")]
    sig = next((l.strip() for l in src if l.strip().startswith("def ")), "")
    print(f"    live decorator on {fn_name}: {deco}  |  {sig[:90]}")

print("\n" + "=" * 70)
print(f"SUMMARY: grid {gw:.0f}s ({n_rows} evals, {gw/n_rows:.2f}s/eval) + "
      f"train {tw:.2f}s | verify {_count['n']}×{_count['t']/max(_count['n'],1):.2f}s "
      f"= {rw:.0f}s | first-click ≈ {pw+rw:.0f}s")
print("=" * 70)
