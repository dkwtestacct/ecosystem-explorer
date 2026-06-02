#!/usr/bin/env python3
"""Phase-0 recon for the region-constrained optimizer.

Times one full-engine `evaluate_scenario` call over a handful of region sizes
to answer two questions:

  1. Does per-eval cost actually shrink with region size, or is the engine
     "compute citywide, then clip" — in which case brute-force-over-a-region
     doesn't get cheaper evals?
  2. At the measured per-eval cost, how many candidate evaluations fit under
     a ~2–5 min interactive budget? (sets the brute-force coarseness ceiling)

This is **measurement only** — the script imports production code (app.evaluate_scenario,
app._build_ownership_mask, app._compose_eligible_filter_cfg, region_rasters from
CityState) and never reimplements engine math. No engine output changes; no
gate run needed.

Invocation (PROJ env per the venv pattern, same as verify_baselines):

  PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \\
  GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \\
  .venv/bin/python scripts/phase0_region_eval.py

Picks SA as the test bed (largest grid → harshest measurement). MN is cheap
enough that any reasonable answer falls out of the SA numbers.
"""
from __future__ import annotations

import gc
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

# Make the project root importable when invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Streamlit stub — must be installed before `import app`. Reuses the pattern
# from verify_baselines.py so the recon hits the same engine path the gate
# does (no skew from a different import sequence).
from verify_baselines import _StubSt, _rebind_city  # noqa: E402

sys.modules["streamlit"] = _StubSt()


# ── RSS helper (macOS reports bytes; Linux reports KB) ───────────────────────
def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


# ── Region cases ─────────────────────────────────────────────────────────────
import numpy as np  # noqa: E402

CITY = "San Antonio, TX"
# One representative recipe (mid-range mix); we vary `seed` across reps so
# repeats are independent placement draws on the same engine path. The
# engine itself has no scenario-level caching — every evaluate_scenario call
# does the full raster pass — so cache-hit understatement isn't a risk here.
RECIPE = dict(
    pct_converted=10,
    green_infrastructure_pct=50,
    food_forest_pct=50,
    placement_strategy="random",
)
N_REPS = 3  # median of 3 — enough to wash out OS jitter without ballooning


def _build_synthetic_25px_mask(state) -> np.ndarray:
    """A 25-pixel mask anchored on a known convertible pixel — the tiniest
    realistic region. Matches verify_baselines' 'tiny region (25 px synthetic)'
    matrix cell so the recon hits the same extreme."""
    mask = np.zeros(state.ref_shape, dtype=bool)
    cp = state.convertible_pixels
    if len(cp) == 0:
        raise RuntimeError("no convertible pixels — can't build synthetic mask")
    r0, c0 = int(cp[0, 0]), int(cp[0, 1])
    H, W = state.ref_shape
    r1 = min(r0 + 5, H)
    c1 = min(c0 + 5, W)
    mask[r0:r1, c0:c1] = True
    return mask


def _district_mask(state, layer_key: str, label: str) -> np.ndarray:
    labels_for_layer = state.region_layer_labels[layer_key]
    pos_idx = labels_for_layer.index(label)
    return state.region_rasters[layer_key] == pos_idx


# ── One timed eval ───────────────────────────────────────────────────────────
def _time_one_eval(app_mod, mask, seed: int):
    t0 = time.perf_counter()
    results = app_mod.evaluate_scenario(
        **RECIPE, seed=seed, selected_region_mask=mask,
    )
    wall_s = time.perf_counter() - t0
    return wall_s, results


def _measure_case(app_mod, label: str, mask, eligible_px: int):
    times = []
    for rep in range(N_REPS):
        gc.collect()
        wall_s, results = _time_one_eval(app_mod, mask, seed=42 + rep)
        times.append(wall_s)
    return {
        "label": label,
        "eligible_px": eligible_px,
        "wall_s_median": statistics.median(times),
        "wall_s_min": min(times),
        "wall_s_max": max(times),
        "wall_s_reps": times,
        "peak_rss_mb": _peak_rss_mb(),
        "results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"Phase-0 recon — region-constrained optimizer budget")
    print(f"City: {CITY}")
    print(f"Recipe: {RECIPE}")
    print(f"Reps per case: {N_REPS}\n")

    print("Importing app.py (triggers module-level startup)...")
    t0 = time.time()
    import app  # noqa: E402

    from ownership import OWNERSHIP_MODES  # noqa: E402
    print(f"  app.py import: {time.time() - t0:.1f}s")

    t1 = time.time()
    _rebind_city(app, CITY)
    state = app._CURRENT_CITY_STATE
    print(f"  city load: {time.time() - t1:.1f}s")
    print(f"  grid: {state.ref_shape[0]} × {state.ref_shape[1]} "
          f"= {state.ref_shape[0] * state.ref_shape[1]:,} px")
    print(f"  convertible pool: {len(state.convertible_pixels):,} px\n")

    # ── Build cases ──────────────────────────────────────────────────────
    # SA council districts vary roughly 2-4x in geographic area; pick a small
    # one (D5 — the matrix-cell standard) and a large one (D9 if present,
    # else fall back to the largest by pixel count).
    layer_key = "council_districts"
    labels_for_layer = state.region_layer_labels[layer_key]
    print(f"  council district labels: {labels_for_layer}")

    # Pick small + large districts by actual pixel count.
    district_px_counts = {}
    for lbl in labels_for_layer:
        m = _district_mask(state, layer_key, lbl)
        district_px_counts[lbl] = int(m.sum())
    print(f"  district sizes (px): "
          f"{sorted(district_px_counts.items(), key=lambda kv: kv[1])}")

    small_label = min(district_px_counts, key=district_px_counts.get)
    large_label = max(district_px_counts, key=district_px_counts.get)

    # No-ownership-overlay for the recon — keep one variable (region size)
    # so the table reads cleanly. (Ownership intersection is composable on
    # top via _build_ownership_mask; cost shape would be the same.)
    cases = []

    # Citywide reference (no region mask).
    cases.append(("citywide (no mask)", None,
                  int(len(state.convertible_pixels))))

    # 25-px synthetic.
    synth_mask = _build_synthetic_25px_mask(state)
    synth_eligible = int(
        synth_mask[state.convertible_pixels[:, 0],
                   state.convertible_pixels[:, 1]].sum()
    )
    cases.append((f"synthetic 25px", synth_mask, synth_eligible))

    # Small district.
    small_mask = _district_mask(state, layer_key, small_label)
    small_eligible = int(
        small_mask[state.convertible_pixels[:, 0],
                   state.convertible_pixels[:, 1]].sum()
    )
    cases.append((f"council D{small_label} (small)",
                  small_mask, small_eligible))

    # Large district.
    large_mask = _district_mask(state, layer_key, large_label)
    large_eligible = int(
        large_mask[state.convertible_pixels[:, 0],
                   state.convertible_pixels[:, 1]].sum()
    )
    cases.append((f"council D{large_label} (large)",
                  large_mask, large_eligible))

    # ── Warm-up — first eval pays for any lazy import + JIT-style numpy
    # interning. Discard so the table reflects steady-state cost.
    print("\n  warmup eval (discarded)...")
    t_warm = time.perf_counter()
    app.evaluate_scenario(**RECIPE, seed=999)
    print(f"    {time.perf_counter() - t_warm:.2f}s\n")

    # ── Run cases ────────────────────────────────────────────────────────
    print(f"{'=' * 78}")
    print("Measuring (median of {} reps per case):".format(N_REPS))
    print(f"{'=' * 78}\n")

    rows = []
    for label, mask, eligible_px in cases:
        print(f"  {label}:")
        if mask is not None:
            print(f"    mask True px:           {int(mask.sum()):>10,}")
        print(f"    eligible (∩ convertible): {eligible_px:>10,}")
        row = _measure_case(app, label, mask, eligible_px)
        rows.append(row)
        print(f"    reps (s): "
              f"{', '.join(f'{t:.3f}' for t in row['wall_s_reps'])}")
        print(f"    median:   {row['wall_s_median']:.3f}s  "
              f"(min {row['wall_s_min']:.3f} / max {row['wall_s_max']:.3f})")
        print(f"    peak RSS: {row['peak_rss_mb']:.0f} MB\n")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"{'=' * 78}")
    print("Summary")
    print(f"{'=' * 78}\n")

    print(f"{'case':<32} {'eligible px':>12} "
          f"{'wall s (med)':>13} {'peak RSS MB':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<32} {r['eligible_px']:>12,} "
              f"{r['wall_s_median']:>13.3f} {r['peak_rss_mb']:>12.0f}")

    # ── Premise check ────────────────────────────────────────────────────
    citywide_t = rows[0]["wall_s_median"]
    print(f"\nPremise check: does per-eval cost shrink with region size?")
    for r in rows[1:]:
        ratio = r["wall_s_median"] / citywide_t
        delta_pct = (r["wall_s_median"] - citywide_t) / citywide_t * 100
        print(f"  {r['label']:<32} "
              f"{r['wall_s_median']:.3f}s  ({ratio:.2f}× citywide, "
              f"{delta_pct:+.1f}%)")

    # ── Budget arithmetic ────────────────────────────────────────────────
    print(f"\nImplied candidate-eval budget (median per-eval cost):")
    median_eval = statistics.median(r["wall_s_median"] for r in rows)
    print(f"  median per-eval across cases: {median_eval:.3f}s")
    for budget_s in (120, 180, 300):
        n = int(budget_s / median_eval)
        print(f"  {budget_s:>3}s interactive target → "
              f"~{n:,} candidate evals fit")

    # ── RAM headroom ─────────────────────────────────────────────────────
    peak_rss = max(r["peak_rss_mb"] for r in rows)
    ceiling_mb = 8 * 1024
    headroom = (ceiling_mb - peak_rss) / ceiling_mb * 100
    print(f"\nRAM headroom vs 8 GB local ceiling:")
    print(f"  peak RSS across cases: {peak_rss:.0f} MB "
          f"({peak_rss / 1024:.2f} GB)")
    print(f"  headroom:              {ceiling_mb - peak_rss:.0f} MB "
          f"({headroom:.0f}% free)")
    print(f"  Streamlit Cloud 1 GB worker:   "
          f"{'OVER' if peak_rss > 1024 else 'fits'} "
          f"({1024 - peak_rss:+.0f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
