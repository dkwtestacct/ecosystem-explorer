"""validate_surrogate_predictions.py — spot-check the retrained surrogate.

After regenerating data/scenarios_dense_mpls.csv with the canonical InVEST UNA
metric, verify the surrogate's predictions match live evaluate_scenario output
on 5 representative scenarios spanning the conversion range.

Placement: all spot-checks use random placement (seed 42) — the configuration
the dense CSV and therefore the surrogate were trained on. The surrogate input
is only (pct_converted, gi%, ff%); it is strategy-blind by design (see the
docstring in surrogate.py), so comparing against random placement isolates the
surrogate's own approximation error rather than conflating it with the
placement-geometry divergence the surrogate structurally cannot capture.

Run on the app's .venv:  .venv/bin/python diagnostics/validate_surrogate_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "validation"))

import pandas as pd

import compare_una_invest as cui  # reuse the Streamlit stub

sys.modules["streamlit"] = cui._StubSt()
import app  # noqa: E402

DENSE_CSV = "data/scenarios_dense_mpls.csv"
OUT_CSV = "comparisons/surrogate_validation.csv"

# The 6 Random-Forest target columns, in train_surrogate's output order.
TARGETS = [
    "flood_reduction", "mean_hm", "food_mln_lbs",
    "runoff_acre_feet", "carbon_tons_co2", "nature_access_pct",
]

# 5 scenarios spanning pct_converted 5-50 (the slider's full range) and a
# spread of GI/FF allocation mixes.
SCENARIOS = [
    (5,  60, 40),
    (20, 100, 0),
    (35, 0, 100),
    (45, 50, 50),
    (50, 33, 33),
]

# Pass thresholds for nature_access_pct (the metric this regeneration fixes).
MAE_PP_THRESHOLD = 5.0      # absolute, percentage points
REL_PCT_THRESHOLD = 10.0    # relative, %


def main():
    print("=" * 68)
    print("Surrogate validation — retrained on canonical-UNA dense CSV")
    print("=" * 68)

    df = pd.read_csv(DENSE_CSV)
    # Balanced mode trains with 200 trees (SURROGATE_TREES['Balanced']).
    model = app._train_surrogate_fn(df, n_estimators=200)
    print(f"\nTrained on {DENSE_CSV}: {len(df)} rows, "
          f"{len(model.estimators_)} trees, {model.n_outputs_} outputs")
    na = df["nature_access_pct"]
    print(f"  training-data nature_access_pct range: "
          f"[{na.min():.1f}, {na.max():.1f}]  (canonical 2SFCA)")

    rows = []
    for pct, gi, ff in SCENARIOS:
        live = app.evaluate_scenario(
            pct, gi, ff, seed=42, placement_strategy="random")
        # Predict from a named-column DataFrame so the feature names match
        # those train_surrogate fitted on (pct_converted, gi%, ff%).
        X = pd.DataFrame(
            [[pct, gi, ff]],
            columns=["pct_converted", "green_infrastructure_pct",
                     "food_forest_pct"])
        pred = model.predict(X)[0]
        for i, target in enumerate(TARGETS):
            lv = float(live[target])
            pv = float(pred[i])
            abs_err = abs(pv - lv)
            rel_err = (abs_err / abs(lv) * 100.0) if lv != 0 else float("nan")
            rows.append({
                "pct_converted": pct, "gi_pct": gi, "ff_pct": ff,
                "target": target, "live": round(lv, 4),
                "predicted": round(pv, 4), "abs_error": round(abs_err, 4),
                "rel_error_pct": round(rel_err, 2),
            })

    vdf = pd.DataFrame(rows)
    Path("comparisons").mkdir(exist_ok=True)
    vdf.to_csv(OUT_CSV, index=False)
    print(f"\nPer-scenario detail written to {OUT_CSV}")

    # ── Per-scenario nature_access_pct table ────────────────────────────────
    print("\nnature_access_pct — surrogate vs live evaluate_scenario:")
    print(f"  {'scenario':22s} {'live':>8s} {'pred':>8s} {'abs err':>9s} "
          f"{'rel err':>9s}")
    na_rows = vdf[vdf.target == "nature_access_pct"]
    for _, r in na_rows.iterrows():
        sc = f"pct{r.pct_converted} gi{r.gi_pct} ff{r.ff_pct}"
        print(f"  {sc:22s} {r.live:>8.1f} {r.predicted:>8.1f} "
              f"{r.abs_error:>9.2f} {r.rel_error_pct:>8.1f}%")

    # ── Per-target MAE summary ──────────────────────────────────────────────
    print("\nMAE per target (mean abs error across the 5 scenarios):")
    print(f"  {'target':22s} {'MAE':>12s} {'mean rel err':>14s}")
    for target in TARGETS:
        sub = vdf[vdf.target == target]
        mae = sub["abs_error"].mean()
        mrel = sub["rel_error_pct"].mean()
        print(f"  {target:22s} {mae:>12.4f} {mrel:>13.2f}%")

    # ── Pass / fail on nature_access_pct ────────────────────────────────────
    na_mae = na_rows["abs_error"].mean()
    na_rel = na_rows["rel_error_pct"].mean()
    print("\n" + "-" * 68)
    print(f"nature_access_pct MAE = {na_mae:.3f} pp  "
          f"(threshold < {MAE_PP_THRESHOLD} pp)")
    print(f"nature_access_pct mean relative error = {na_rel:.2f}%  "
          f"(threshold < {REL_PCT_THRESHOLD}%)")
    has_nan = vdf[["live", "predicted", "abs_error"]].isna().any().any()
    ok = (na_mae < MAE_PP_THRESHOLD or na_rel < REL_PCT_THRESHOLD) and not has_nan
    print(f"NaN present: {has_nan}")
    print(f"\nVALIDATION: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
