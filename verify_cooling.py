"""Headless verification of the UCM rework.

Imports app.py with a stubbed Streamlit session_state, runs four canonical
scenarios (baseline, 50 % FF, 50 % GI, 50 % HD), and prints the same numbers
the UI metric cards would display. Then re-imports with UHI_MAX_C = 3.0 to
show the calibration sensitivity.
"""

import argparse
import os
import sys
import importlib.util


# `CITY_KEY` controls which `CITIES[...]` entry the headless harness selects.
# Default is the downtown Minneapolis view; pass `--city "Minneapolis Full, MN"`
# (or any other available city) on the command line to override.
CITY_KEY = os.environ.get("VERIFY_COOLING_CITY", "Minneapolis, MN")


def _stub_streamlit():
    """Replace `streamlit` with a permissive shim that returns sensible defaults
    for every widget call so app.py's module-level UI section runs to
    completion. We just need the data-loading + function-definition side
    effects; UI output goes nowhere."""
    import streamlit as real_st  # let real streamlit import first for type machinery
    import functools

    # Defaults for every slider/text_input/selectbox the app calls. Keys are
    # the slider's "key" argument (or, where missing, the label).
    _defaults = {
        # Sliders / numeric inputs
        "slider_pct_converted":     25,
        "slider_gi_pct":            50,
        "slider_ff_pct":            50,
        "carbon_rate_ff":           1.5,
        "carbon_rate_gi":           1.0,
        "cost_gi":                  50_000,
        "cost_ff":                  10_000,
        "cost_hd":                  5_000,
        "wgt_ndvi":                 0.2,
        "wgt_cooling":              0.4,
        "wgt_nature":               0.4,
        "min_cool_f":               0.0,
        "min_food":                 0,
        "min_carbon":               0,
        "max_runoff":               1_000_000_000,
        # Selectbox / radio
        "City":                     "Minneapolis, MN",
        "priority_mode":            "Random",
    }

    class _SS(dict):
        def __getattr__(self, k):  return self.get(k, _defaults.get(k))
        def __setattr__(self, k, v): self[k] = v
        def __contains__(self, k): return dict.__contains__(self, k) or k in _defaults

    ss = _SS()

    def _widget(*args, **kwargs):
        """Return a sensible default for any widget call."""
        # Streamlit widgets accept (label, options=..., value=..., key=...)
        if "value" in kwargs:
            return kwargs["value"]
        opts = kwargs.get("options") or (args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else None)
        if opts:
            label = args[0] if args and isinstance(args[0], str) else ""
            # If this is the city picker, honour CITY_KEY (or its
            # "(coming soon)" variant). Fallback to default index 0.
            if "City" in label:
                want_active = CITY_KEY
                want_unavail = f"{CITY_KEY} (coming soon)"
                for o in opts:
                    if o == want_active or o == want_unavail:
                        return o
            idx = kwargs.get("index", 0) or 0
            return list(opts)[idx]
        if args and isinstance(args[0], str):
            label = args[0]
            return _defaults.get(kwargs.get("key", label), 0)
        return 0

    class _ContextNoop:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, k): return self  # chained calls return self too
        def __call__(self, *a, **kw): return self
        def progress(self, *a, **kw): return self  # st.progress(...).progress(...)
        def empty(self, *a, **kw): return self
        @staticmethod
        def _noop(*a, **kw): return None

    class _Sidebar:
        def __getattr__(self, name):
            if name in ("button", "checkbox", "toggle"): return lambda *a, **kw: False
            if name in ("selectbox", "radio", "slider", "select_slider",
                         "number_input", "text_input", "text_area",
                         "color_picker", "date_input", "time_input"):
                return _widget
            if name in ("expander", "container", "form", "popover", "status",
                         "empty", "tabs", "columns", "chat_message"):
                return lambda *a, **kw: _ContextNoop()
            return _ContextNoop._noop
        def divider(self): return None

    sidebar = _Sidebar()

    _saved_ss = ss
    _saved_sidebar = sidebar

    class _StStub:
        session_state = _saved_ss
        sidebar = _saved_sidebar

        def __getattr__(self, name):
            if name in ("button", "checkbox", "toggle", "form_submit_button"):
                return lambda *a, **kw: False
            if name in ("selectbox", "radio", "slider", "select_slider",
                         "number_input", "text_input", "text_area",
                         "color_picker", "date_input", "time_input"):
                return _widget
            if name == "tabs":
                return lambda labels: [_ContextNoop() for _ in labels]
            if name == "columns":
                return lambda spec, **kw: [_ContextNoop() for _ in
                                            (spec if hasattr(spec, "__len__") else range(spec))]
            if name in ("expander", "container", "form", "popover", "status",
                         "empty", "spinner", "chat_message", "progress"):
                return lambda *a, **kw: _ContextNoop()
            if name == "cache_data":
                return lambda *a, **kw: (a[0] if a and callable(a[0]) else (lambda f: f))
            if name == "cache_resource":
                return lambda *a, **kw: (a[0] if a and callable(a[0]) else (lambda f: f))
            if name == "stop":
                # If app.py reaches st.stop(), raise to abort module load cleanly
                def _stop(): raise RuntimeError("st.stop() called — should not happen for MN")
                return _stop
            return _ContextNoop._noop

    sys.modules["streamlit"] = _StStub()


def load_app(uhi_override=None):
    """Force a fresh import of app.py, optionally patching UHI_MAX_C."""
    sys.modules.pop("app", None)
    _stub_streamlit()
    spec = importlib.util.spec_from_file_location("app", "app.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        # The module fails midway when it tries to render UI widgets; the
        # failure happens AFTER `evaluate_scenario`, `_compute_cc_raster`,
        # `compute_cooling_energy_savings`, and the baseline override are
        # all defined. We don't care about the UI section.
        pass

    if uhi_override is not None and hasattr(m, "UHI_MAX_C"):
        m.UHI_MAX_C = uhi_override
        m.HM_TO_FAHRENHEIT = uhi_override * 1.8
        # Re-run the baseline override since it depends on UHI_MAX_C only
        # indirectly (via _compute_cc_raster, which doesn't use it). The
        # baseline raster itself is unchanged; only the °F mapping changes.
    return m


SCENARIOS = [
    # (name, pct_converted, gi_pct, ff_pct)
    ("Baseline (0 % converted)",         0,  50, 50),
    ("All Food Forest (50 % FF)",       50,   0, 100),
    ("All Green Infrastructure (50 % GI)", 50, 100, 0),
    ("All High Density (50 % HD)",      50,   0, 0),
    # Mid-range mixed scenarios for UI sanity-check
    ("Mid: 10 % converted, 50/50 GI/FF",  10,  50,  50),
    ("Mid: 20 % converted, 50/50 GI/FF",  20,  50,  50),
    ("Mid: 30 % converted, 50/50 GI/FF",  30,  50,  50),
    ("Mid: 30 % converted, 80/20 GI/FF",  30,  80,  20),
    ("Mid: 30 % converted, 20/80 GI/FF",  30,  20,  80),
    ("Mid: 50 % converted, 50/50 GI/FF",  50,  50,  50),
]


def run_one(m, name, pct, gi, ff):
    r = m.evaluate_scenario(
        pct_converted=pct,
        green_infrastructure_pct=gi,
        food_forest_pct=ff,
        seed=42,
    )
    return {
        "name":           name,
        "mean_hm":        r["mean_hm"],
        "cooling_f":      r["cooling_f"],
        "energy_savings": r["cooling_energy_savings_usd"],
        "flood_reduction": r["flood_reduction"],
    }


def report(m, label):
    print(f"\n========== {label} ==========")
    print(f"  UHI_MAX_C = {m.UHI_MAX_C}, HM_TO_FAHRENHEIT = {m.HM_TO_FAHRENHEIT:.3f}")
    print(f"  BASELINE_HM (live mean of smoothed CC) = {m.BASELINE_HM:.4f}")
    print(f"  MAX_ET_REF = {m.MAX_ET_REF:.2f}")
    print(f"  ENERGY_TABLE_AVAILABLE = {m.ENERGY_TABLE_AVAILABLE}, "
          f"BUILDINGS_DATA_AVAILABLE = {m.BUILDINGS_DATA_AVAILABLE}, "
          f"ET_DATA_AVAILABLE = {m.ET_DATA_AVAILABLE}")
    print(f"\n  {'Scenario':<40s} {'mean_CC':>9s} {'ΔCC':>8s} {'cooling_f':>10s} {'$ savings/yr':>15s} {'flood_red':>10s}")
    print(f"  {'-'*40} {'-'*9} {'-'*8} {'-'*10} {'-'*15} {'-'*10}")
    for name, pct, gi, ff in SCENARIOS:
        s = run_one(m, name, pct, gi, ff)
        d_cc = s["mean_hm"] - m.BASELINE_HM
        print(f"  {s['name']:<40s} {s['mean_hm']:>9.4f} {d_cc:>+8.4f} {s['cooling_f']:>+10.2f} ${s['energy_savings']:>13,.0f} {s['flood_reduction']:>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default=CITY_KEY,
                        help="CITIES key to select (default: 'Minneapolis, MN'). "
                             "Override via CLI or VERIFY_COOLING_CITY env var.")
    parser.add_argument("--no-uhi-sweep", action="store_true",
                        help="Skip the UHI=3.0 alternate run.")
    args = parser.parse_args()
    CITY_KEY = args.city  # update module-level so the stub's _widget sees it

    print(f"Loading app.py with default UHI_MAX_C = 2.05, city = {CITY_KEY!r} ...")
    m1 = load_app()
    report(m1, f"DEFAULT: UHI_MAX_C = 2.05 °C  ({CITY_KEY})")

    if not args.no_uhi_sweep:
        print("\nReloading with UHI_MAX_C = 3.0 (upper bound for dense urban areas)…")
        m2 = load_app(uhi_override=3.0)
        report(m2, f"ALTERNATE: UHI_MAX_C = 3.0 °C  ({CITY_KEY})")
