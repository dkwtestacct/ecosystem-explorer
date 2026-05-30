"""
placement_strategy_diagnostic.py — measure what each placement strategy
actually does on each city, across configurations, with controlled RNG.

Three layers of measurement, written to CSV under analysis/placement_diagnostic/:

  Layer 1 (suitability variance):
    For each (city, strategy) pair, summarize the per-pixel suitability
    weight distribution over the convertible pool: mean, std, min, max,
    p25, p50, p75, p95. Tells us whether each strategy has signal to
    work with on each city.

  Layer 2 (chosen-pixel score gap vs overall-pool mean):
    For each (city, strategy, pct, seed), sample N pixels under that
    strategy's weights and report the mean suitability score of the
    chosen subset vs the overall convertible-pool mean. Gap = chosen
    minus overall. Random is evaluated against flood-focused's surface
    so it has a baseline mean to compare against.

  Layer 3+4 (metric outcomes):
    For each (city, strategy, scenario, pct, seed) run
    evaluate_scenario and record flood_reduction, mean_hm,
    food_mln_lbs, carbon_tons_co2, runoff_acre_feet.
    Scenarios: all-GI, all-FF, all-HD. pcts: 10, 25, 50.
    1,350 evaluate_scenario calls in total.

Usage:
    # Orchestrator mode (default) — launches one subprocess per city
    python3 diagnostics/placement_strategy_diagnostic.py

    # Worker mode — runs one city only (used by the orchestrator)
    python3 diagnostics/placement_strategy_diagnostic.py --city "Minneapolis, MN"

CSVs are append-only with a header written on first row. Re-running
skips (city, strategy, scenario, pct, seed) tuples already present.

Implementation note: app.py's module state is per-city (set at the
sidebar selectbox during import). We can't dynamically switch cities
in-process without re-running the module-level loader. The orchestrator
launches a fresh subprocess per city to keep this clean; each
subprocess also drops the full city raster stack on exit, so 8 GB M1
can run the diagnostic without OOM risk.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Configuration ─────────────────────────────────────────────────────────────
CITIES = ["Minneapolis, MN", "Minneapolis Full, MN", "San Antonio, TX"]
STRATEGIES = ["random", "flood-focused", "cooling-focused", "undersupply-focused", "balanced"]
NON_RANDOM_STRATEGIES = ["flood-focused", "cooling-focused", "undersupply-focused", "balanced"]
SCENARIOS = {
    "all_gi": dict(gi=100, ff=0),
    "all_ff": dict(gi=0, ff=100),
    "all_hd": dict(gi=0, ff=0),
}
PCTS = [10, 25, 50]
SEEDS = list(range(10))

OUT_DIR = Path("analysis/placement_diagnostic")
LAYER1_CSV = OUT_DIR / "layer1_suitability_variance.csv"
LAYER2_CSV = OUT_DIR / "layer2_chosen_pixel_scores.csv"
LAYER3_CSV = OUT_DIR / "layer3_metric_outcomes.csv"


# ── CSV helpers ───────────────────────────────────────────────────────────────
def _ensure_csv(path: Path, fieldnames: list[str]) -> None:
    """Create CSV with header if it doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _existing_tuples(path: Path, key_cols: list[str]) -> set[tuple]:
    """Read CSV and return the set of (key_col_1, key_col_2, ...) tuples
    already recorded, so we can skip them on resume."""
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            out.add(tuple(r[c] for c in key_cols))
    return out


# ── Worker mode: run all three layers for one city ────────────────────────────
def run_worker(city_key: str) -> None:
    """Per-city worker. Imports app with the streamlit stub, then runs
    Layers 1, 2, 3 for `city_key`."""
    print(f"\n{'=' * 60}\n  Worker: {city_key}\n{'=' * 60}\n")

    # Stub streamlit (re-using the precompute_scenarios.py mechanism) then
    # import app, which loads the city's runtime state at module level.
    os.environ["PRECOMPUTE_CITY"] = city_key  # picked up by app's stubbed selectbox
    _install_streamlit_stub(city_key)

    _t_import = time.time()
    print(f"[{city_key}] importing app.py …")
    import app  # noqa: E402
    print(f"[{city_key}] app.py imported in {time.time() - _t_import:.1f}s")

    import numpy as np
    import pandas as pd

    convertible = app.CONVERTIBLE_PIXELS
    print(f"[{city_key}] convertible_pool_size={len(convertible):,}")

    # Pre-compute each strategy's per-pixel weight array once. These are
    # reused by Layer 1 (stats) and Layer 2 (chosen-pixel scoring).
    weights_by_strategy = {}
    for strat in NON_RANDOM_STRATEGIES:
        weights_by_strategy[strat] = app._compute_suitability_weights(convertible, strat)
        print(f"[{city_key}] computed weights for {strat}")

    # ── Layer 1: suitability variance ─────────────────────────────────────
    layer1_fields = ["city", "strategy", "n_pixels", "mean", "std", "min",
                     "p25", "p50", "p75", "p95", "max"]
    _ensure_csv(LAYER1_CSV, layer1_fields)
    layer1_done = _existing_tuples(LAYER1_CSV, ["city", "strategy"])

    for strat in NON_RANDOM_STRATEGIES:
        if (city_key, strat) in layer1_done:
            print(f"[{city_key}] layer1 skip (already done): {strat}")
            continue
        w = weights_by_strategy[strat]
        row = {
            "city": city_key,
            "strategy": strat,
            "n_pixels": int(len(w)),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "min": float(w.min()),
            "p25": float(np.percentile(w, 25)),
            "p50": float(np.percentile(w, 50)),
            "p75": float(np.percentile(w, 75)),
            "p95": float(np.percentile(w, 95)),
            "max": float(w.max()),
        }
        _append_row(LAYER1_CSV, layer1_fields, row)
        print(f"[{city_key}] layer1 wrote: {strat} mean={row['mean']:.4g} std={row['std']:.4g}")

    # ── Layer 2: chosen-pixel score gap vs overall pool mean ─────────────
    # For each strategy, pick N pixels under its weights and report the
    # mean weight of the chosen subset vs the mean weight over the whole
    # pool. Random samples uniformly and is scored against flood-focused's
    # surface so it has a meaningful baseline.
    #
    # Pre-compute non-zero counts for each strategy first. The app's
    # _select_pixels_for_conversion crashes when `n_to_convert` exceeds the
    # number of non-zero weights (rng.choice with replace=False can't sample
    # more entries than have positive probability). We detect this here and
    # mark the row as `saturated` rather than crashing the diagnostic.
    nonzero_count_by_strategy = {
        s: int((weights_by_strategy[s] > 0).sum())
        for s in NON_RANDOM_STRATEGIES
    }
    print(f"[{city_key}] non-zero pixel counts by strategy: {nonzero_count_by_strategy}")

    layer2_fields = ["city", "strategy", "pct", "seed", "n_chosen",
                     "chosen_pool_mean_score", "overall_pool_mean_score",
                     "gap", "scoring_surface", "saturated", "nonzero_count"]
    _ensure_csv(LAYER2_CSV, layer2_fields)
    layer2_done = _existing_tuples(LAYER2_CSV, ["city", "strategy", "pct", "seed"])

    pool_size = len(convertible)
    for pct in PCTS:
        n_chosen = int(pool_size * pct / 100)
        for strat in STRATEGIES:
            scoring_surface = strat if strat != "random" else "flood-focused"
            w_score = weights_by_strategy[scoring_surface]
            pool_mean = float(w_score.mean())
            # `saturated` is now a marker on the row rather than a skip
            # condition — the app's _select_pixels_for_conversion handles
            # the saturated case via its non-zero-then-uniform-remainder
            # fallback (see docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md §7).
            saturated = (
                strat in nonzero_count_by_strategy
                and n_chosen > nonzero_count_by_strategy[strat]
            )
            for seed in SEEDS:
                key = (city_key, strat, str(pct), str(seed))
                if key in layer2_done:
                    continue
                rng = np.random.default_rng(seed)
                chosen_idx = app._select_pixels_for_conversion(
                    convertible, n_chosen, strat, rng
                )
                chosen_mean = float(w_score[chosen_idx].mean()) if len(chosen_idx) > 0 else 0.0
                row = {
                    "city": city_key,
                    "strategy": strat,
                    "pct": pct,
                    "seed": seed,
                    "n_chosen": n_chosen,
                    "chosen_pool_mean_score": chosen_mean,
                    "overall_pool_mean_score": pool_mean,
                    "gap": chosen_mean - pool_mean,
                    "scoring_surface": scoring_surface,
                    "saturated": saturated,
                    "nonzero_count": (nonzero_count_by_strategy[strat]
                                      if strat in nonzero_count_by_strategy
                                      else pool_size),
                }
                _append_row(LAYER2_CSV, layer2_fields, row)
    print(f"[{city_key}] layer2 done")

    # ── Layer 3: metric outcomes via evaluate_scenario ───────────────────
    layer3_fields = ["city", "strategy", "scenario", "pct", "seed",
                     "flood_reduction", "mean_hm", "food_mln_lbs",
                     "carbon_tons_co2", "runoff_acre_feet",
                     "elapsed_s", "saturated"]
    _ensure_csv(LAYER3_CSV, layer3_fields)
    layer3_done = _existing_tuples(LAYER3_CSV, ["city", "strategy", "scenario", "pct", "seed"])

    total_combos = len(STRATEGIES) * len(SCENARIOS) * len(PCTS) * len(SEEDS)
    done = sum(1 for c in layer3_done if c[0] == city_key)
    print(f"[{city_key}] layer3 starting; {done}/{total_combos} already done")

    t_layer3 = time.time()
    n_run = 0
    for strat in STRATEGIES:
        for scen_name, scen in SCENARIOS.items():
            for pct in PCTS:
                n_chosen = int(pool_size * pct / 100)
                # `saturated` is now a marker on the row, not a skip — the
                # app's _select_pixels_for_conversion handles the saturated
                # case via its non-zero-then-uniform-remainder fallback.
                # See docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md §7.
                saturated = (
                    strat in nonzero_count_by_strategy
                    and n_chosen > nonzero_count_by_strategy[strat]
                )
                for seed in SEEDS:
                    key = (city_key, strat, scen_name, str(pct), str(seed))
                    if key in layer3_done:
                        continue
                    t0 = time.time()
                    res = app.evaluate_scenario(
                        pct_converted=pct,
                        green_infrastructure_pct=scen["gi"],
                        food_forest_pct=scen["ff"],
                        seed=seed,
                        placement_strategy=strat,
                    )
                    elapsed = time.time() - t0
                    row = {
                        "city": city_key,
                        "strategy": strat,
                        "scenario": scen_name,
                        "pct": pct,
                        "seed": seed,
                        "flood_reduction": res["flood_reduction"],
                        "mean_hm": res["mean_hm"],
                        "food_mln_lbs": res["food_mln_lbs"],
                        "carbon_tons_co2": res["carbon_tons_co2"],
                        "runoff_acre_feet": res["runoff_acre_feet"],
                        "elapsed_s": round(elapsed, 3),
                        "saturated": saturated,
                    }
                    _append_row(LAYER3_CSV, layer3_fields, row)
                    n_run += 1
                    if n_run % 25 == 0:
                        rate = n_run / max(time.time() - t_layer3, 1e-6)
                        remaining = total_combos - done - n_run
                        eta = remaining / rate if rate > 0 else 0
                        print(f"[{city_key}] layer3 {n_run} new ({done + n_run}/{total_combos}) "
                              f"— rate {rate:.2f}/s, ETA {eta:.0f}s")

    print(f"[{city_key}] layer3 done — {n_run} new rows in {time.time() - t_layer3:.1f}s")
    print(f"[{city_key}] worker complete")


# ── Streamlit stub (lifted from precompute_scenarios.py) ─────────────────────
def _install_streamlit_stub(city_key: str) -> None:
    """Install a minimal streamlit stub so `import app` runs without UI."""

    class _SessionStateStub:
        _store: dict = {}

        def get(self, k, d=None): return self._store.get(k, d)
        def pop(self, k, *a):
            return self._store.pop(k, *a) if a else self._store.pop(k, None)
        def setdefault(self, k, d=None): return self._store.setdefault(k, d)
        def __getattr__(self, name):
            if name == "_store": return object.__getattribute__(self, "_store")
            return self._store.get(name)
        def __getitem__(self, k): return self._store.get(k)
        def __setitem__(self, k, v): self._store[k] = v
        def __setattr__(self, n, v):
            if n == "_store": object.__setattr__(self, n, v)
            else: self._store[n] = v
        def __contains__(self, k): return k in self._store

    class _StubSt:
        def __getattr__(self, name):
            if name in ("cache_data", "cache_resource"): return self._cache
            if name == "columns": return self._columns
            if name == "tabs": return self._tabs
            if name == "selectbox":
                def _sb(label, options, **kw):
                    if not options: return None
                    if "City" in str(label):
                        # Return city_key unconditionally for the City picker so the
                        # diagnostic can target cities filtered from the UI list
                        # (e.g. Minneapolis Full, available=False). Downstream code
                        # reads `CITIES[selected_city]` which still works as long
                        # as city_key is in CITIES, regardless of UI filtering.
                        return city_key
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
        def _cache(self, *a, **kw):
            if a and callable(a[0]) and len(a) == 1 and not kw:
                return a[0]
            return lambda f: f
        def _columns(self, spec, *a, **kw):
            n = spec if isinstance(spec, int) else len(spec)
            return tuple(_StubSt() for _ in range(n))
        def _tabs(self, labels, *a, **kw):
            return tuple(_StubSt() for _ in labels)
        def __call__(self, *a, **kw): return self
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def __getitem__(self, k): return self
        def __setitem__(self, k, v): pass
        def __setattr__(self, n, v): pass
        def __contains__(self, k): return False
        def __iter__(self): return iter([])
        def __bool__(self): return True

    sys.modules["streamlit"] = _StubSt()


# ── Orchestrator: one subprocess per city ────────────────────────────────────
def run_orchestrator(cities: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOrchestrator: launching {len(cities)} worker subprocesses\n")
    for city in cities:
        print(f"\n{'#' * 60}\n# Subprocess: {city}\n{'#' * 60}\n")
        result = subprocess.run(
            [sys.executable, __file__, "--city", city],
            check=False,
        )
        if result.returncode != 0:
            print(f"!! Subprocess for {city!r} exited with {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
        # Subprocess termination drops the city's raster stack; no need
        # for explicit gc here.
        gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", help="If set, run worker mode for this city only.")
    parser.add_argument("--cities", nargs="+", default=CITIES,
                        help="Cities for orchestrator mode (default: all three).")
    args = parser.parse_args()

    if args.city:
        run_worker(args.city)
    else:
        run_orchestrator(args.cities)


if __name__ == "__main__":
    main()
