"""
precompute_scenarios.py — one-shot offline build of a dense surrogate-training
grid for the carbon and nature-access metrics.

Writes data/scenarios_dense.csv with one row per (pct_converted, gi_pct, ff_pct)
combination, evaluated at finer steps than app.py's default in-app grid:

  pct_converted: range(0, 51, 5)    -> 11 values
  gi_pct:        range(0, 101, 10)  -> 11 values
  ff_pct:        range(0, 101, 10)  -> 11 values
  constraint:    gi_pct + ff_pct <= 100

That yields 11 * 66 = 726 valid scenarios — roughly 8x the in-app default of 90
(pct step 10, gi/ff step 25). app.py prefers this CSV over its on-the-fly grid
when the file exists.

Implementation note: importing app.py also triggers its module-level startup,
which includes a 2,541-scenario lookup-table build via distance_transform_edt
(~2-4 min). That work is wasted from the precompute's point of view but lets us
reuse evaluate_scenario / calculate_nature_access / pop_count_raster without
duplicating their logic here. The precompute itself runs in roughly another
1-2 min on top.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


# ── 1. Stub streamlit so importing app.py doesn't try to render a UI ──────────
class _SessionStateStub:
    """Mimic st.session_state — `.get(key, default)` must return `default`,
    item/attr reads return None, writes are no-ops."""

    def get(self, key, default=None):
        return default

    def __getattr__(self, name):
        return None

    def __getitem__(self, key):
        return None

    def __setitem__(self, key, value):
        pass

    def __setattr__(self, name, value):
        pass

    def __contains__(self, key):
        return False


class _StubSt:
    """Acts as: no-op callable / context manager / dict / chained attribute."""

    def __getattr__(self, name):
        if name in ("cache_data", "cache_resource"):
            return self._cache
        if name == "columns":
            return self._columns
        if name == "tabs":
            return self._tabs
        # Widget functions need to return sensible defaults so app.py's
        # downstream code (.index(...), arithmetic, etc.) doesn't break.
        if name == "selectbox":
            return lambda label, options, **kw: options[0] if options else None
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
        # Handles both `@st.cache_data` (no parens) and `@st.cache_data(...)`.
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
        # Block writes like `st.session_state.x = ...` from blowing up.
        pass

    def __contains__(self, key):
        return False

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return True


sys.modules["streamlit"] = _StubSt()


# ── 2. Now safe to import app.py ──────────────────────────────────────────────
print("Importing app.py (triggers its module-level startup, ~2-4 min)...")
_t_import = time.time()
import app  # noqa: E402

print(f"  app.py import + startup compute: {time.time() - _t_import:.1f}s")


# ── 3. Build the dense scenario grid ──────────────────────────────────────────
PCT_RANGE = list(range(0, 51, 5))
GI_RANGE  = list(range(0, 101, 10))
FF_RANGE  = list(range(0, 101, 10))

combos = [
    (pct, gi, ff)
    for pct in PCT_RANGE
    for gi  in GI_RANGE
    for ff  in FF_RANGE
    if gi + ff <= 100
]

print(f"\nDense grid: {len(combos):,} valid scenarios")
print(f"  pct: {PCT_RANGE}")
print(f"  gi:  {GI_RANGE}")
print(f"  ff:  {FF_RANGE}")
print(f"In-app sparse grid (for comparison): {len(app.scenario_df):,} scenarios")
print(f"Density factor: {len(combos) / len(app.scenario_df):.1f}x\n")

OUT_PATH = Path("data/scenarios_dense.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = []
_t_loop = time.time()
for i, (pct, gi, ff) in enumerate(combos, start=1):
    result = app.evaluate_scenario(pct, gi, ff, seed=42)
    row = {k: v for k, v in result.items() if k != "scenario_lulc"}
    # Mirror the explicit recomputation in app.compute_scenario_grid so the
    # CSV schema lines up with what train_surrogate expects.
    row["carbon_tons_co2_yr"] = app._compute_carbon(
        row["n_wet"], row["n_for"], row["n_hd"]
    )
    nature_pct, nature_people = app.calculate_nature_access(
        result["scenario_lulc"], app.pop_count_raster
    )
    row["nature_access_pct"] = nature_pct
    row["people_with_nature_access"] = nature_people
    rows.append(row)

    if i % 50 == 0 or i == len(combos):
        elapsed = time.time() - _t_loop
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(combos) - i) / rate if rate > 0 else 0
        print(f"  {i:4d}/{len(combos)} scenarios — "
              f"{elapsed:5.0f}s elapsed, ETA {eta:4.0f}s")

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)

print(f"\nWrote {OUT_PATH}")
print(f"  Rows:    {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Total time (including app import): {time.time() - _t_import:.1f}s")

# ── 4. Verification block ─────────────────────────────────────────────────────
print("\n--- Verification ---")
print(f"Exact row count: {len(df):,}")

print("\nFirst 5 rows (key columns):")
preview_cols = [c for c in [
    "pct_converted", "green_infrastructure_pct", "food_forest_pct",
    "flood_reduction", "mean_hm", "food_mln_lbs", "runoff_acre_feet",
    "carbon_tons_co2_yr", "nature_access_pct", "mean_ndvi",
] if c in df.columns]
print(df[preview_cols].head(5).to_string(index=False))

print("\nMin/max per output column:")
output_cols = [c for c in [
    "flood_reduction", "mean_hm", "food_mln_lbs", "runoff_acre_feet",
    "carbon_tons_co2_yr", "nature_access_pct", "mean_ndvi",
] if c in df.columns]
header = f"  {'column':<25} {'min':>12} {'max':>12}"
print(header)
print("  " + "-" * (len(header) - 2))
for col in output_cols:
    print(f"  {col:<25} {df[col].min():>12.4f} {df[col].max():>12.4f}")
