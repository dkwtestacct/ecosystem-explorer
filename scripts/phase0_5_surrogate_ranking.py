#!/usr/bin/env python3
"""Phase-0.5 recon: does the citywide-trained surrogate rank region-scoped
candidates the same way the full engine ranks them?

The surrogate (`surrogate.py`) maps (pct_converted, GI%, FF%) → six metric
predictions trained on a CITYWIDE scenario grid. It knows nothing about
regions or ownership filters. The region-constrained optimizer (next phase)
needs to know whether a surrogate prefilter would shortlist the engine's
true region-local winners or scramble them — that's the property B-style
(prefilter) needs to be real.

Method: for one region (SA Council District 5 — same small-district choice
as Phase-0), generate ~30 candidate recipes spanning the knob space, score
each two ways:

  Engine    — `evaluate_scenario(recipe, selected_region_mask=D5)` →
              `results['region_local']` (the true region-local metric).
  Surrogate — `predict_with_uncertainty(model, [[pct, gi, ff]])` (citywide,
              region-blind prediction).

Per metric, compute Spearman rank correlation and recall@K — of the engine's
top-5 candidates, how many appear in the surrogate's top-15. Per-metric is
the key axis: if every constituent metric ranks well, any user-weighted
objective they pick also ranks well (the weighting-agnostic property).

Recon class — no engine/app change, no baselines impact, no gate. Standalone
sibling to phase0_region_eval.py.

Invocation (PROJ env per the venv pattern):

  PROJ_DATA=.venv/lib/python3.9/site-packages/rasterio/proj_data \\
  GDAL_DATA=.venv/lib/python3.9/site-packages/rasterio/gdal_data \\
  .venv/bin/python scripts/phase0_5_surrogate_ranking.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from verify_baselines import _StubSt, _rebind_city  # noqa: E402

sys.modules["streamlit"] = _StubSt()


CITY = "San Antonio, TX"
LAYER_KEY = "council_districts"
PRIMARY_REGION_LABEL = "5"   # the small district from Phase-0
FALLBACK_REGION_LABEL = "3"  # the large district, used only if D5 is borderline


# ── Candidate recipe set ─────────────────────────────────────────────────────
# Structured spread — 4 pct levels × 8 allocations covering corners (all-HD,
# all-GI, all-FF), balanced mixes, and asymmetric mixes. 32 candidates total,
# enough that rankings don't cluster.
PCT_LEVELS = [5, 15, 30, 50]
ALLOC_MIXES = [
    (0,   0),    # all HD
    (100, 0),    # all GI
    (0,   100),  # all FF
    (50,  50),   # balanced GI / FF, no HD
    (75,  25),   # GI-heavy
    (25,  75),   # FF-heavy
    (50,  0),    # half GI, half HD
    (0,   50),   # half FF, half HD
]


def _build_candidates():
    return [
        dict(pct_converted=pct, green_infrastructure_pct=gi,
             food_forest_pct=ff, placement_strategy="random")
        for pct in PCT_LEVELS
        for (gi, ff) in ALLOC_MIXES
    ]


def _district_mask(state, layer_key, label):
    labels_for_layer = state.region_layer_labels[layer_key]
    pos_idx = labels_for_layer.index(label)
    return state.region_rasters[layer_key] == pos_idx


# ── Ranking helpers ──────────────────────────────────────────────────────────
# Per-metric direction. "higher" = larger is better (rank by descending).
# "lower" = smaller is better (rank by ascending). For recall@K the engine's
# "top" is direction-aware.
METRIC_DIRECTION = {
    "mean_hm":          "higher",   # cooling — higher HMI = more cooling
    "flood_reduction":  "higher",   # higher = less runoff
    "runoff_acre_feet": "lower",    # less runoff = better
    "carbon_tons_co2":  "higher",   # SA: stock change; MN: annual flow
    "food_mln_lbs":     "higher",
    "total_cost_mln":   "lower",    # cheaper = better
    "nature_access_pct": "higher",
}


def _top_k(values, k, direction):
    """Indices of the top-k by direction."""
    arr = np.asarray(values, dtype=float)
    if direction == "higher":
        order = np.argsort(-arr, kind="stable")
    else:
        order = np.argsort(arr, kind="stable")
    return list(order[:k])


def _spearman(a, b):
    """Returns (rho, p)."""
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


# ── Main ─────────────────────────────────────────────────────────────────────
def _score_region(app_mod, surrogate_model, mask, region_label, candidates):
    """Run engine + surrogate for every candidate. Returns parallel lists:
    engine_metrics[metric] = list of floats, surrogate_metrics[metric] = same."""

    engine_metrics = {m: [] for m in METRIC_DIRECTION}
    surrogate_metrics = {m: [] for m in METRIC_DIRECTION}

    # Surrogate output column order — must match surrogate.train_surrogate.
    SURROGATE_COLUMNS = ["flood_reduction", "mean_hm", "food_mln_lbs",
                         "runoff_acre_feet", "carbon_tons_co2",
                         "nature_access_pct"]

    # Citywide eligible pool (for the "implied cost" prediction — the
    # surrogate doesn't predict cost, but cost is deterministic in the recipe
    # given a pool, so the citywide pool gives the analog citywide cost
    # ranking the surrogate would imply).
    cp_count = int(len(app_mod.CONVERTIBLE_PIXELS))

    print(f"\n  scoring {len(candidates)} candidates on region {region_label!r}…")
    print(f"  region eligible: ~{int(mask[app_mod.CONVERTIBLE_PIXELS[:, 0], app_mod.CONVERTIBLE_PIXELS[:, 1]].sum()):,} px")
    t_all = time.perf_counter()

    for i, recipe in enumerate(candidates, 1):
        # ── Engine (region-local — the truth target) ─────────────────────
        t0 = time.perf_counter()
        results = app_mod.evaluate_scenario(
            **recipe, seed=42, selected_region_mask=mask,
        )
        engine_t = time.perf_counter() - t0
        rl = results["region_local"]
        for m in METRIC_DIRECTION:
            engine_metrics[m].append(float(rl[m]) if rl.get(m) is not None else float("nan"))
        # Drop the heavy LULC arrays so 30+ engine runs don't bloat RSS.
        for k in ("scenario_lulc", "scenario_lulc_ucm",
                  "scenario_lulc_una", "scenario_lulc_carbon"):
            results.pop(k, None)

        # ── Surrogate (citywide prediction — region-blind) ───────────────
        X = np.array([[recipe["pct_converted"],
                       recipe["green_infrastructure_pct"],
                       recipe["food_forest_pct"]]], dtype=float)
        tree_preds = np.array(
            [t.predict(X) for t in surrogate_model.estimators_]
        )
        # shape (n_trees, 1, n_outputs). Mean across trees.
        mean = tree_preds.mean(axis=0)[0]  # (n_outputs,)
        for col_idx, col_name in enumerate(SURROGATE_COLUMNS):
            if col_name in METRIC_DIRECTION:
                surrogate_metrics[col_name].append(float(mean[col_idx]))

        # ── Implied cost (recipe-deterministic; surrogate analog) ────────
        # Recipe → citywide n_*: n_convert_city = cp_count * pct/100.
        # n_wet = n_convert * gi/100; n_for = * ff/100; n_hd = remainder.
        pct = recipe["pct_converted"]
        gi = recipe["green_infrastructure_pct"]
        ff = recipe["food_forest_pct"]
        n_convert_city = int(cp_count * pct / 100)
        n_wet_city = int(n_convert_city * gi / 100)
        n_for_city = int(n_convert_city * ff / 100)
        n_hd_city = n_convert_city - n_wet_city - n_for_city
        cost_city_mln = float(app_mod.compute_cost(
            n_wet_city, n_for_city, n_hd_city,
            app_mod.DEFAULT_COST_GI, app_mod.DEFAULT_COST_FF,
            app_mod.DEFAULT_COST_HD,
        ))
        surrogate_metrics["total_cost_mln"].append(cost_city_mln)

        # Brief status — one line per candidate.
        print(f"   [{i:>2}/{len(candidates)}] "
              f"pct={pct:>2} gi={gi:>3} ff={ff:>3}  engine={engine_t:.2f}s  "
              f"rl.mean_hm={rl['mean_hm']:.3f}  rl.flood_red={rl['flood_reduction']:.1f}")
        gc.collect()

    print(f"  region {region_label!r} scoring done in {time.perf_counter() - t_all:.1f}s")
    return engine_metrics, surrogate_metrics


def _build_ranking_table(engine_metrics, surrogate_metrics, k_top=5, k_pool=15):
    """Per-metric Spearman + recall@K-in-top-pool."""
    rows = []
    for m, direction in METRIC_DIRECTION.items():
        eng = np.array(engine_metrics[m], dtype=float)
        sur = np.array(surrogate_metrics[m], dtype=float)
        # Skip metrics where the engine produced all NaN (e.g. region_local
        # has None for a non-decomposable metric — none in our list, but defensive).
        if np.isnan(eng).any() or np.isnan(sur).any():
            rows.append({"metric": m, "direction": direction,
                         "rho": float("nan"), "p": float("nan"),
                         "recall": float("nan"),
                         "n_eng_unique": 0, "n_sur_unique": 0})
            continue
        rho, p = _spearman(eng, sur)
        eng_top = set(_top_k(eng, k_top, direction))
        sur_pool = set(_top_k(sur, k_pool, direction))
        hits = len(eng_top & sur_pool)
        recall = hits / k_top
        rows.append({
            "metric":       m,
            "direction":    direction,
            "rho":          rho,
            "p":            p,
            "recall":       recall,
            "n_eng_unique": int(len(set(np.round(eng, 6)))),
            "n_sur_unique": int(len(set(np.round(sur, 6)))),
        })
    return rows


def _print_table(rows, region_label, k_top, k_pool):
    print(f"\n  ── Per-metric ranking — region {region_label!r} "
          f"(K_top={k_top}, K_pool={k_pool}) ──")
    print(f"  {'metric':<22} {'dir':<6} {'spearman ρ':>11} "
          f"{'p':>9} {'recall':>8}  {'eng/sur uniq':>14}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['metric']:<22} {r['direction']:<6} "
              f"{r['rho']:>11.3f} {r['p']:>9.2g} "
              f"{r['recall']:>8.2f}  "
              f"{r['n_eng_unique']:>5}/{r['n_sur_unique']:<5}")


def _is_borderline(rows):
    """A region is 'borderline' if any metric's Spearman ρ is in (0.5, 0.8)
    or recall is in (0.4, 0.8) — fuzzy enough that one more region would
    sharpen the read."""
    for r in rows:
        if np.isnan(r["rho"]):
            continue
        if 0.5 < r["rho"] < 0.8:
            return True
        if 0.4 < r["recall"] < 0.8:
            return True
    return False


def main() -> int:
    print(f"Phase-0.5 recon — surrogate vs engine ranking, city={CITY}")
    print(f"Recipes: {len(PCT_LEVELS) * len(ALLOC_MIXES)} candidates "
          f"({len(PCT_LEVELS)} pct × {len(ALLOC_MIXES)} alloc mixes)\n")

    print("Importing app.py (triggers module-level startup)...")
    t0 = time.time()
    import app  # noqa: E402
    print(f"  app.py import: {time.time() - t0:.1f}s")

    t1 = time.time()
    _rebind_city(app, CITY)
    state = app._CURRENT_CITY_STATE
    print(f"  city load: {time.time() - t1:.1f}s")
    print(f"  grid: {state.ref_shape[0]} × {state.ref_shape[1]}; "
          f"convertible pool: {len(state.convertible_pixels):,} px")

    # ── Build the surrogate (Fast prototype mode — 90 scenarios, 100 trees,
    # the default UI mode). Train on the citywide grid; this is exactly what
    # the live app's optimizer queries.
    print("\nBuilding the citywide training grid (Fast prototype, 90 recipes)…")
    t2 = time.time()
    scenario_df = app.compute_scenario_grid(
        state, CITY, app.DATA_DIR_FLOOD, app.DATA_DIR_COOLING,
        step_pct=10, step_alloc=25,
    )
    print(f"  grid built in {time.time() - t2:.1f}s  ({len(scenario_df)} rows)")

    print("Training surrogate (100 trees)…")
    from surrogate import train_surrogate  # noqa: E402
    t3 = time.time()
    surrogate_model = train_surrogate(scenario_df, n_estimators=100)
    print(f"  trained in {time.time() - t3:.1f}s")

    candidates = _build_candidates()
    print(f"\n{'=' * 78}")
    print(f"Scoring candidates on region D{PRIMARY_REGION_LABEL} (small district)")
    print(f"{'=' * 78}")

    primary_mask = _district_mask(state, LAYER_KEY, PRIMARY_REGION_LABEL)
    eng_m, sur_m = _score_region(app, surrogate_model, primary_mask,
                                 PRIMARY_REGION_LABEL, candidates)
    rows_primary = _build_ranking_table(eng_m, sur_m, k_top=5, k_pool=15)
    _print_table(rows_primary, PRIMARY_REGION_LABEL, 5, 15)

    if _is_borderline(rows_primary):
        print(f"\n  D{PRIMARY_REGION_LABEL} borderline — repeating on "
              f"D{FALLBACK_REGION_LABEL} (large district)…")
        fallback_mask = _district_mask(state, LAYER_KEY, FALLBACK_REGION_LABEL)
        eng_m2, sur_m2 = _score_region(app, surrogate_model, fallback_mask,
                                       FALLBACK_REGION_LABEL, candidates)
        rows_fallback = _build_ranking_table(eng_m2, sur_m2,
                                             k_top=5, k_pool=15)
        _print_table(rows_fallback, FALLBACK_REGION_LABEL, 5, 15)
    else:
        print(f"\n  D{PRIMARY_REGION_LABEL} not borderline — "
              f"D{FALLBACK_REGION_LABEL} repeat skipped.")

    # ── Footnote on cost ─────────────────────────────────────────────────
    print(f"\n  Cost note: surrogate has no cost output. The 'surrogate cost' "
          f"column above is the recipe-deterministic citywide cost "
          f"(n_*_city × default rates), the analog the surrogate WOULD predict "
          f"if it had cost as an output. Region engine cost is the same recipe "
          f"applied to the (smaller) region eligible pool, so rank-correlation "
          f"is preserved by construction — a perfect Spearman ρ here is a "
          f"sanity check, not a finding.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
