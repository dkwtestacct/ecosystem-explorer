#!/usr/bin/env python3
"""calibrate_surrogate_band.py — Relay 60 Part B.

Emit a per-city × per-mode CALIBRATED estimate-range artifact: the empirical
10th–90th percentiles of the surrogate's prediction residual against the full
engine, measured by k-fold held-out cross-validation over the precomputed
scenario grid. This replaces the old inter-tree "model disagreement band" with
an interval that has actually been validated against engine-computed values.

Residual convention: `resid = engine_true − surrogate_pred`. The display forms
the interval as `[estimate + p10, estimate + p90]`, so it brackets the engine
truth around the estimate and a systematic under-prediction (carbon/food) shows
as an upward skew rather than a shifted point estimate.

The CV exactly replicates `surrogate.train_surrogate`:
  X = [pct_converted, green_infrastructure_pct, food_forest_pct]
  y = [flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet,
       carbon_tons_co2, nature_access_pct]
  RandomForestRegressor(n_estimators=100, random_state=42)

Modes are keyed to the grid the runtime surrogate trains on:
  fast      -> CITIES[city]['fast_grid_file']      (SA only; MN builds live)
  balanced  -> CITIES[city]['dense_scenarios_file']

Pure CSV + sklearn — no app import, no engine run. Ground truth is the grid's
engine columns. Writes data/<slug>/surrogate_calibration_<mode>.json.

Usage:
  .venv/bin/python scripts/calibrate_surrogate_band.py            # all cities
  .venv/bin/python scripts/calibrate_surrogate_band.py --city "San Antonio, TX"
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

FEATS = ['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']
# Output column order MUST match surrogate.train_surrogate exactly.
TARGETS = ['flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
           'carbon_tons_co2', 'nature_access_pct']
K_FOLDS = 10
CALIB_FORMAT_VERSION = 1

# city key -> data dir slug (mirrors the grid-file naming).
SLUG = {
    'Minneapolis, MN': 'mpls',
    'Minneapolis Full, MN': 'mpls_full',
    'San Antonio, TX': 'sa',
}


def _grid_hash(df):
    """Stable content hash over the feature+target columns (order-independent
    of other columns), so a regenerated grid changes the stamp."""
    cols = [c for c in FEATS + TARGETS if c in df.columns]
    buf = df[cols].round(6).to_numpy(dtype=float).tobytes()
    h = hashlib.sha256()
    h.update(','.join(cols).encode())
    h.update(buf)
    return h.hexdigest()[:16]


def _surrogate_signature(df):
    """Mirror surrogate-cache signature semantics: row count + target sums."""
    cols = [c for c in FEATS + TARGETS if c in df.columns]
    sums = df[cols].fillna(0).sum().to_numpy()
    return f"n{len(df)}_" + hashlib.sha256(
        np.round(sums, 4).tobytes()).hexdigest()[:12]


def calibrate(csv_path, schema_version):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATS + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    X = df[FEATS].to_numpy(float)
    Y = df[TARGETS].to_numpy(float)
    n = len(df)
    k = min(K_FOLDS, n)
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    resid = np.full_like(Y, np.nan)        # engine_true - surrogate_pred
    for tr, te in kf.split(X):
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(X[tr], Y[tr])
        pred = m.predict(X[te])
        resid[te] = Y[te] - pred           # truth minus prediction
    quant = {}
    for j, t in enumerate(TARGETS):
        r = resid[:, j]
        quant[t] = {
            'p10': round(float(np.percentile(r, 10)), 6),
            'p90': round(float(np.percentile(r, 90)), 6),
            'rmse': round(float(np.sqrt(np.mean(r ** 2))), 6),
            'bias': round(float(np.mean(r)), 6),
        }
    stamp = {
        'calib_format_version': CALIB_FORMAT_VERSION,
        'scenario_schema_version': int(schema_version),
        'grid_hash': _grid_hash(df),
        'surrogate_signature': _surrogate_signature(df),
        'n_rows': int(n),
        'n_folds': int(k),
        'rf_n_estimators': 100,
        'rf_random_state': 42,
        'residual_convention': 'engine_true - surrogate_pred',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    return {'provenance': stamp, 'residual_quantiles': quant}


def _schema_of(grid_path):
    meta = grid_path + '.meta.json'
    if os.path.exists(meta):
        return json.loads(open(meta).read()).get('scenario_schema_version')
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--city', default=None, help='single city key; default all available')
    args = ap.parse_args()

    cities = ([args.city] if args.city
              else [c for c, cfg in config.CITIES.items() if cfg.get('available')])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrote, skipped = [], []
    for city in cities:
        cfg = config.CITIES[city]
        slug = SLUG.get(city)
        if not slug:
            skipped.append(f"{city}: no slug mapping"); continue
        out_dir = os.path.join(root, 'data', slug)
        os.makedirs(out_dir, exist_ok=True)
        for mode, key in (('fast', 'fast_grid_file'), ('balanced', 'dense_scenarios_file')):
            grid = cfg.get(key)
            if not grid or not os.path.exists(grid):
                skipped.append(f"{city}/{mode}: no grid ({key}={grid!r}) — "
                               "runtime shows no range for this mode")
                continue
            art = calibrate(grid, _schema_of(grid))
            out = os.path.join(out_dir, f'surrogate_calibration_{mode}.json')
            with open(out, 'w') as f:
                json.dump(art, f, indent=2)
            q = art['residual_quantiles']
            wrote.append(out)
            print(f"  WROTE {out}  (n={art['provenance']['n_rows']}, schema "
                  f"v{art['provenance']['scenario_schema_version']})")
            for t in TARGETS:
                print(f"      {t:20s} p10={q[t]['p10']:+.4g} p90={q[t]['p90']:+.4g} "
                      f"bias={q[t]['bias']:+.4g}")
    print(f"\n{len(wrote)} artifact(s) written; {len(skipped)} skipped.")
    for s in skipped:
        print(f"  SKIP {s}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
