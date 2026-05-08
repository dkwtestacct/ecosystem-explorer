"""
validate_scenarios.py — quick sanity check for the scenario engine.

Runs five canonical scenarios via app.evaluate_scenario, prints a results
table, then verifies directional expectations (e.g. "All Food Forest beats
Baseline on carbon"). Any violated expectation is flagged with ⚠️.

Like precompute_scenarios.py, this script stubs `streamlit` so it can
`import app` and reuse the real evaluator without duplicating logic.
"""
from __future__ import annotations

import sys
import time


# ── Stub streamlit so importing app.py doesn't try to render UI ───────────────
class _SessionStateStub:
    def get(self, key, default=None): return default
    def __getattr__(self, name):       return None
    def __getitem__(self, key):        return None
    def __setitem__(self, key, value): pass
    def __setattr__(self, name, value): pass
    def __contains__(self, key):       return False


class _StubSt:
    def __getattr__(self, name):
        if name in ("cache_data", "cache_resource"):
            return self._cache
        if name == "columns":
            return self._columns
        if name == "tabs":
            return self._tabs
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
        if name == "text_input" or name == "text_area":
            return lambda *a, **kw: kw.get("value", "")
        if name in ("toggle", "checkbox", "button"):
            return lambda *a, **kw: False
        if name == "session_state":
            return _SessionStateStub()
        return self

    def _cache(self, *a, **kw):
        if a and callable(a[0]) and len(a) == 1 and not kw:
            return a[0]
        return lambda f: f

    def _columns(self, spec, *a, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_StubSt() for _ in range(n))

    def _tabs(self, labels, *a, **kw):
        return tuple(_StubSt() for _ in labels)

    def __call__(self, *a, **kw):  return self
    def __enter__(self):           return self
    def __exit__(self, *exc):      return False
    def __getitem__(self, key):    return self
    def __setitem__(self, key, value): pass
    def __setattr__(self, name, value): pass
    def __contains__(self, key):   return False
    def __iter__(self):            return iter([])
    def __bool__(self):            return True


sys.modules["streamlit"] = _StubSt()


# ── Import app and run scenarios ──────────────────────────────────────────────
print("Importing app.py (triggers module-level startup)...")
_t0 = time.time()
import app  # noqa: E402

print(f"  app imported in {time.time() - _t0:.1f}s\n")

scenarios = [
    {"name": "Baseline",         "pct": 0,  "gi": 0,   "ff": 0},
    {"name": "All Food Forest",  "pct": 50, "gi": 0,   "ff": 100},
    {"name": "All Green Infra",  "pct": 50, "gi": 100, "ff": 0},
    {"name": "All High Density", "pct": 50, "gi": 0,   "ff": 0},
    {"name": "Mixed 50/50",      "pct": 30, "gi": 50,  "ff": 50},
]

# Evaluate each scenario.
print("Running scenarios...\n")
results = {}
for s in scenarios:
    r = app.evaluate_scenario(s["pct"], s["gi"], s["ff"], seed=42)
    results[s["name"]] = r

# ── Results table ─────────────────────────────────────────────────────────────
metrics = [
    ("flood_reduction",     "Flood Idx",  "{:>9.2f}"),
    ("mean_hm",             "Cooling HM", "{:>10.4f}"),
    ("food_mln_lbs",        "Food Mlbs",  "{:>9.3f}"),
    ("runoff_acre_feet",    "Runoff a-f", "{:>10.0f}"),
    ("carbon_tons_co2_yr",  "Carbon t",   "{:>9.0f}"),
    ("nature_access_pct",   "Nature %",   "{:>8.1f}"),
]

name_w = max(len(s["name"]) for s in scenarios)
header = f"{'Scenario'.ljust(name_w)}  " + "  ".join(f"{label:>9}" for _, label, _ in metrics)
print(header)
print("-" * len(header))
for name, r in results.items():
    cells = []
    for key, _, fmt in metrics:
        cells.append(fmt.format(r[key]))
    print(f"{name.ljust(name_w)}  " + "  ".join(cells))

# ── Directional sanity checks ─────────────────────────────────────────────────
b   = results["Baseline"]
ff  = results["All Food Forest"]
gi  = results["All Green Infra"]
hd  = results["All High Density"]

checks = [
    # All Food Forest > Baseline for cooling, food, carbon, nature access
    ("All Food Forest cooling (mean_hm) > Baseline",
     ff["mean_hm"] > b["mean_hm"]),
    ("All Food Forest food production > Baseline",
     ff["food_mln_lbs"] > b["food_mln_lbs"]),
    ("All Food Forest carbon > Baseline",
     ff["carbon_tons_co2_yr"] > b["carbon_tons_co2_yr"]),
    # NOTE: After switching to the InVEST UNA biophysical table, the baseline
    # already saturates at ~100% — the Mississippi River (NLCD 11, score 1.0,
    # 5 km radius) plus existing forest patches blanket the entire model area,
    # so adding more food forest can't improve a metric that's already pinned
    # at the ceiling. This is real, expected behavior of the InVEST radii at
    # this 10.8 × 10.7 km extent — see the "Saturation at full extent" caveat
    # in REFERENCE.md. The check uses ≥ so the validator passes when both
    # scenarios tie at 100%; if a future radius-scaling slider tightens the
    # radii enough to bring baseline below 100%, this check will tighten too.
    ("All Food Forest nature access ≥ Baseline",
     ff["nature_access_pct"] >= b["nature_access_pct"]),

    # All Green Infra > Baseline for flood reduction
    ("All Green Infra flood reduction > Baseline",
     gi["flood_reduction"] > b["flood_reduction"]),

    # All High Density < Baseline for flood, cooling, nature access
    ("All High Density flood reduction < Baseline",
     hd["flood_reduction"] < b["flood_reduction"]),
    ("All High Density cooling (mean_hm) < Baseline",
     hd["mean_hm"] < b["mean_hm"]),
    # NOTE: Nature access uses ≤, not strict <. The model only converts already-
    # developed pixels (NLCD 21–24); existing nature pixels are never removed,
    # so an All-High-Density scenario leaves the nature mask unchanged and the
    # metric ties baseline. This is expected and correct given the current
    # design — see the "High-Density-only conversion ties baseline" caveat in
    # REFERENCE.md. A future model that degrades nearby green-space quality
    # under high-density expansion could turn this into a strict <.
    ("All High Density nature access ≤ Baseline",
     hd["nature_access_pct"] <= b["nature_access_pct"]),

    # Carbon increases with tree cover
    ("Carbon increases with tree cover (FF > Baseline)",
     ff["carbon_tons_co2_yr"] > b["carbon_tons_co2_yr"]),

    # Runoff decreases with GI
    ("Runoff decreases with GI (GI < Baseline)",
     gi["runoff_acre_feet"] < b["runoff_acre_feet"]),
]

print("\nDirection checks:\n")
n_pass = 0
n_fail = 0
for description, ok in checks:
    if ok:
        print(f"  ✅  {description}")
        n_pass += 1
    else:
        print(f"  ⚠️  {description}")
        n_fail += 1

print(f"\n{n_pass}/{len(checks)} checks passed; {n_fail} flagged.")
sys.exit(0 if n_fail == 0 else 1)
