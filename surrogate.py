"""surrogate.py — Random Forest surrogate model + optimizer.

Lifted out of app.py to isolate the model layer from the UI layer.

Public API:
    train_surrogate(scenario_df, n_estimators) -> trained RF model
    predict_with_uncertainty(model, X) -> (mean, lower, upper)
    optimize_scenario(surrogate, min_flood, min_cool, min_food,
                      max_runoff, min_carbon, max_food, max_flood,
                      max_cool, n_samples) -> top suggestions or dict
    compute_pareto(df) -> Pareto-efficient rows
    plot_feature_importance(model) -> matplotlib Figure

The surrogate is trained on the precomputed scenario set (which varies
by model-quality mode — 90 for Fast prototype, dense CSV for Balanced,
2,541 for High resolution). The @st.cache_resource decorator lives at
the call site in app.py, not here, so this module is Streamlit-agnostic
and can be tested standalone.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor


def train_surrogate(scenario_df, n_estimators=100):
    """Train a multi-output Random Forest on the scenario grid.

    Parameters
    ----------
    scenario_df : DataFrame with columns pct_converted, green_infrastructure_pct,
        food_forest_pct, plus the six target columns.
    n_estimators : int — number of RF trees.

    Returns
    -------
    Fitted RandomForestRegressor (multi-output).
    """
    X = scenario_df[['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']]
    # Nature Access is included as a sixth output, but with an important caveat:
    # the surrogate maps (pct, gi%, ff%) -> nature_access_pct, which discards the
    # spatial geometry that drives the metric. Random vs heat-priority placement,
    # and the location of converted pixels relative to existing parks and
    # population centers, all change the actual buffer overlap — but the
    # surrogate cannot see any of that. Treat surrogate predictions of
    # nature_access_pct as an indicative trend, not a precise spatial estimate.
    y = scenario_df[['flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
                      'carbon_tons_co2', 'nature_access_pct']]
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model


def predict_with_uncertainty(model, X):
    """
    Return mean prediction and 10th/90th percentile bands across RF trees.
    X should be shape (n_samples, n_features).
    Returns: mean (n,6), lower (n,6), upper (n,6)
    Columns: [flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet,
              carbon_tons_co2, nature_access_pct]
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds shape: (n_trees, n_samples, n_outputs)
    mean  = tree_preds.mean(axis=0)
    lower = np.percentile(tree_preds, 10, axis=0)
    upper = np.percentile(tree_preds, 90, axis=0)
    return mean, lower, upper


def plot_feature_importance(model):
    """Plotly horizontal bar chart of RF feature importances.

    Returns a plotly.graph_objects.Figure. Caller should use
    st.plotly_chart(fig, use_container_width=True) instead of
    render_matplotlib(...).
    """
    feature_names = ['% Converted', 'Green Infra %', 'Food Forest %']
    colors = ['#8e8e8e', '#2196a0', '#4caf50']
    importances = model.feature_importances_  # shape (n_features,)

    # Reverse so that % Converted renders at the top of the horizontal bars
    # (Plotly's default is bottom-up).
    fig = go.Figure(go.Bar(
        x=importances[::-1],
        y=feature_names[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f'{v:.2f}' for v in importances[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='What drives outcomes most?', font=dict(size=12)),
        xaxis=dict(
            title='Relative Importance',
            range=[0, max(importances) * 1.3],
        ),
        yaxis=dict(title=None),
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=40, b=40),
        hovermode='closest',
    )
    return fig


def compute_pareto(df):
    """Return Pareto-efficient rows (maximize flood_reduction, mean_hm, food_mln_lbs)."""
    cols = [c for c in ['flood_reduction', 'mean_hm', 'food_mln_lbs'] if c in df.columns]
    points = df[cols].values
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(points[is_efficient] > c, axis=1) |
                np.all(points[is_efficient] == c, axis=1)
            )
            is_efficient[i] = True
    return df[is_efficient]


# ── Region-constrained optimizer (variant B) ─────────────────────────────────
# docs/internal/REGION_OPTIMIZER_SPEC.md — surrogate prefilter (citywide,
# ranking-only) → Pareto shortlist capped at K → full-engine verify in-region
# → weighted-sum rank on engine values → greedy knob-distance dedup → top-5.
# Streamlit-agnostic: the engine eval is passed as a callable so this module
# stays off the app/Streamlit dependency tree.

# Direction per metric — "higher" = better, "lower" = better. Cost and runoff
# invert; everything else maximizes. Same convention as the Phase-0.5 recon.
_REGION_OPT_METRIC_DIRECTION = {
    'mean_hm':           'higher',   # cooling
    'flood_reduction':   'higher',
    'runoff_acre_feet':  'lower',
    'carbon_tons_co2':   'higher',
    'food_mln_lbs':      'higher',
    'total_cost_mln':    'lower',
    'nature_access_pct': 'higher',
}


def _compute_pareto_multi(df, cols, directions):
    """Pareto front over an arbitrary set of columns + per-column directions.
    Direction-correct each column to a maximize convention before the dominance
    sweep (lower-better → negate)."""
    pts = np.column_stack([
        df[c].to_numpy() if d == 'higher' else -df[c].to_numpy()
        for c, d in zip(cols, directions)
    ])
    is_efficient = np.ones(pts.shape[0], dtype=bool)
    for i, c in enumerate(pts):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(pts[is_efficient] > c, axis=1)
                | np.all(pts[is_efficient] == c, axis=1)
            )
            is_efficient[i] = True
    return df[is_efficient].copy()


def _direction_correct_minmax(values, direction):
    """Min-max normalize an array, direction-corrected so 1.0 = best regardless
    of metric direction. Returns 0.5 vector when the array is constant (no
    spread → no ranking signal on this axis)."""
    arr = np.asarray(values, dtype=float)
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin <= 0:
        return np.full_like(arr, 0.5, dtype=float)
    if direction == 'higher':
        return (arr - vmin) / (vmax - vmin)
    return (vmax - arr) / (vmax - vmin)


def _knob_distance(a, b):
    """L1 distance in (pct, gi, ff) space — used for the greedy dedup."""
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]))


def optimize_scenario_region(
    surrogate_model,
    candidate_grid,
    engine_eval,
    weights,
    *,
    k_engine=40,
    top_n=5,
    knob_dedup_threshold=10,
    progress_cb=None,
    metrics=('mean_hm', 'flood_reduction', 'runoff_acre_feet',
             'carbon_tons_co2', 'food_mln_lbs', 'total_cost_mln'),
):
    """Region-constrained optimizer (variant B).

    Parameters
    ----------
    surrogate_model : RandomForestRegressor — trained citywide surrogate.
        Output column order matches `train_surrogate` (flood_reduction,
        mean_hm, food_mln_lbs, runoff_acre_feet, carbon_tons_co2,
        nature_access_pct).
    candidate_grid : DataFrame — the surrogate's training grid; supplies
        (pct_converted, green_infrastructure_pct, food_forest_pct) for the
        candidate set. Cost is recomputed per-row using `cost_fn`.
    engine_eval : callable(pct, gi, ff) -> results_dict — the production
        evaluate_scenario wrapped to inject the active region∩ownership mask
        and the live cost slider values. Returns the engine's result dict
        (must carry a `region_local` block with the metric values).
    weights : dict {metric_name: float} — user weight per objective. Missing
        keys default to 0.0; if all weights are zero, falls back to equal.
    k_engine : int — K candidates engine-evaluated. ≈ 40 sized to ~2-min budget.
    top_n : int — distinct records returned after dedup.
    knob_dedup_threshold : int — L1 (pct + gi + ff) distance below which a
        candidate is treated as a near-duplicate of an already-kept record.
    progress_cb : callable(int, int) or None — called as progress_cb(i, K)
        after each engine eval. Lets Streamlit update a progress bar without
        coupling this module to Streamlit.
    metrics : tuple of metric names participating in the Pareto + weighting.

    Returns
    -------
    DataFrame of up to `top_n` rows. Columns:
        pct_converted, green_infrastructure_pct, food_forest_pct,
        pct_highdensity, scenario_name, each metric in `metrics` (engine
        region-local), weighted_score, weights_used (dict serialized per row).
    """
    # ── Prefilter: surrogate score every candidate ────────────────────────
    grid = candidate_grid[
        ['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']
    ].drop_duplicates().reset_index(drop=True)
    X = grid.to_numpy(dtype=float)
    mean_preds, _, _ = predict_with_uncertainty(surrogate_model, X)
    # Output column order from train_surrogate:
    SUR_COLS = ['flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
                'carbon_tons_co2', 'nature_access_pct']
    prefilter = grid.copy()
    for i, c in enumerate(SUR_COLS):
        prefilter[c] = mean_preds[:, i]

    # Cost is recipe-deterministic. We don't have cost rates plumbed in here
    # (those live behind cost sliders), so we approximate prefilter cost by
    # the recipe's own n_*_proxy × default rate weights — purely for ranking,
    # never displayed. Use the recipe's GI/FF/HD shares directly: cost_proxy
    # scales linearly with pct × (gi_weight + ff_weight + hd_weight). The
    # rates here are placeholder weights, not dollars — the engine pass below
    # produces the real cost values.
    _GI_W, _FF_W, _HD_W = 50.0, 8.0, 8.0   # rough relative-cost ratios
    pct_p = prefilter['pct_converted'].to_numpy(dtype=float)
    gi_p = prefilter['green_infrastructure_pct'].to_numpy(dtype=float)
    ff_p = prefilter['food_forest_pct'].to_numpy(dtype=float)
    hd_p = 100.0 - gi_p - ff_p
    prefilter['total_cost_mln'] = (
        pct_p * (gi_p * _GI_W + ff_p * _FF_W + hd_p * _HD_W)
    )

    # Pareto front across the active metric set.
    metrics = tuple(m for m in metrics if m in prefilter.columns)
    directions = [_REGION_OPT_METRIC_DIRECTION[m] for m in metrics]
    pareto = _compute_pareto_multi(prefilter, list(metrics), directions)

    # Cap at K. If the front exceeds K, sample for knob spread via greedy
    # maximin (pick farthest-from-already-kept each step). Cheap and gives
    # better coverage than slicing.
    if len(pareto) > k_engine:
        recipes = pareto[['pct_converted', 'green_infrastructure_pct',
                          'food_forest_pct']].to_numpy(dtype=float)
        kept = [0]
        remaining = set(range(1, len(recipes)))
        while len(kept) < k_engine and remaining:
            best, best_d = None, -1.0
            for idx in remaining:
                d_min = min(np.sum(np.abs(recipes[idx] - recipes[k]))
                            for k in kept)
                if d_min > best_d:
                    best_d, best = d_min, idx
            kept.append(best)
            remaining.discard(best)
        pareto = pareto.iloc[kept].reset_index(drop=True)
    else:
        pareto = pareto.reset_index(drop=True)

    if pareto.empty:
        return pareto

    # ── Engine-verify each candidate in-region ────────────────────────────
    engine_rows = []
    K = len(pareto)
    for i, row in pareto.iterrows():
        res = engine_eval(
            int(row['pct_converted']),
            int(row['green_infrastructure_pct']),
            int(row['food_forest_pct']),
        )
        rl = res.get('region_local') or {}
        engine_row = {
            'pct_converted':            int(row['pct_converted']),
            'green_infrastructure_pct': int(row['green_infrastructure_pct']),
            'food_forest_pct':          int(row['food_forest_pct']),
        }
        for m in metrics:
            v = rl.get(m)
            if v is None:
                v = res.get(m)
            engine_row[m] = float(v) if v is not None else float('nan')
        # Carry the result-derived $-metric labels through so the table can
        # show full-scenario context the user already sees in other cards.
        engine_row['carbon_value_usd'] = float(rl.get('carbon_value_usd')
                                               or res.get('carbon_value_usd', 0))
        engine_row['cooling_energy_savings_usd'] = float(
            rl.get('cooling_energy_savings_usd')
            or res.get('cooling_energy_savings_usd', 0)
        )
        engine_rows.append(engine_row)
        if progress_cb is not None:
            progress_cb(i + 1, K)

    engine_df = pd.DataFrame(engine_rows)

    # ── Rank by weighted sum over min-max normalized engine values ────────
    norm_cols = {}
    for m in metrics:
        norm_cols[m] = _direction_correct_minmax(engine_df[m].to_numpy(),
                                                 _REGION_OPT_METRIC_DIRECTION[m])
    # Normalize / clean weights — drop unknown metrics, zero-clamp negatives,
    # fall back to equal weights if the user-zeroed all of them.
    w = {m: max(0.0, float(weights.get(m, 0.0))) for m in metrics}
    if sum(w.values()) <= 0:
        w = {m: 1.0 for m in metrics}
    score = np.zeros(len(engine_df), dtype=float)
    for m in metrics:
        score += w[m] * norm_cols[m]
    engine_df['weighted_score'] = score
    engine_df['pct_highdensity'] = (
        100 - engine_df['green_infrastructure_pct']
        - engine_df['food_forest_pct']
    )

    # Sort descending — highest weighted score first.
    engine_df = engine_df.sort_values(
        'weighted_score', ascending=False, kind='stable'
    ).reset_index(drop=True)

    # ── Greedy knob-distance dedup → top_n distinct records ───────────────
    kept_rows = []
    for _, row in engine_df.iterrows():
        recipe = (row['pct_converted'], row['green_infrastructure_pct'],
                  row['food_forest_pct'])
        if any(_knob_distance(recipe,
                              (k['pct_converted'], k['green_infrastructure_pct'],
                               k['food_forest_pct'])) < knob_dedup_threshold
               for k in kept_rows):
            continue
        kept_rows.append(row.to_dict())
        if len(kept_rows) >= top_n:
            break

    out = pd.DataFrame(kept_rows)
    if out.empty:
        return out
    out['scenario_name'] = out.apply(
        lambda r: (
            f"{int(r.pct_converted)}% converted — "
            f"GI {int(r.green_infrastructure_pct)}% / "
            f"FF {int(r.food_forest_pct)}%"
        ),
        axis=1,
    )
    # Snapshot the weights used so a record-consumer can read them back.
    out['weights_used'] = [dict(w)] * len(out)
    out['source'] = 'region_optimized'
    out['validation'] = 'engine_verified'
    return out


def optimize_scenario(surrogate, min_flood, min_cool, min_food, max_runoff,
                      min_carbon=0, max_food=1.0, max_flood=100.0,
                      max_cool=1.1, n_samples=10000):
    """Use the surrogate to find efficient tradeoff scenarios meeting the given constraints.

    Parameters
    ----------
    max_food, max_flood, max_cool : normalization constants for the Pareto
        scoring formula. max_food comes from the scenario grid's food column
        max; the other two are fixed ceilings.
    """
    rng = np.random.default_rng(42)
    pct_converted = rng.integers(0, 51, n_samples)
    gi_pct        = rng.integers(0, 101, n_samples)
    ff_pct        = rng.integers(0, 101, n_samples)

    valid = gi_pct + ff_pct <= 100
    pct_converted, gi_pct, ff_pct = pct_converted[valid], gi_pct[valid], ff_pct[valid]

    X = np.column_stack([pct_converted, gi_pct, ff_pct])
    mean_preds, lower_preds, upper_preds = predict_with_uncertainty(surrogate, X)

    meets = (
        (mean_preds[:, 0] >= min_flood) &
        (mean_preds[:, 1] >= min_cool)  &
        (mean_preds[:, 2] >= min_food)  &
        (mean_preds[:, 3] <= max_runoff) &
        (mean_preds[:, 4] >= min_carbon)
    )
    if not meets.any():
        return {
            'found': False,
            'max_flood':  round(float(mean_preds[:, 0].max()), 1),
            'max_cool':   round(float(mean_preds[:, 1].max()), 4),
            'max_food':   round(float(mean_preds[:, 2].max()), 3),
            'max_carbon': round(float(mean_preds[:, 4].max()), 1),
        }

    candidates = pd.DataFrame({
        'pct_converted':            pct_converted[meets],
        'green_infrastructure_pct': gi_pct[meets],
        'food_forest_pct':          ff_pct[meets],
        'flood_reduction':          mean_preds[meets, 0].round(1),
        'flood_lower':              lower_preds[meets, 0].round(1),
        'flood_upper':              upper_preds[meets, 0].round(1),
        'mean_hm':                  mean_preds[meets, 1].round(4),
        'hm_lower':                 lower_preds[meets, 1].round(4),
        'hm_upper':                 upper_preds[meets, 1].round(4),
        'food_mln_lbs':             mean_preds[meets, 2].round(3),
        'food_lower':               lower_preds[meets, 2].round(3),
        'food_upper':               upper_preds[meets, 2].round(3),
        'carbon_tons_co2':       mean_preds[meets, 4].round(1),
        'carbon_lower':             lower_preds[meets, 4].round(1),
        'carbon_upper':             upper_preds[meets, 4].round(1),
    })
    candidates['pct_highdensity'] = (
        100 - candidates['green_infrastructure_pct'] - candidates['food_forest_pct']
    )
    candidates['scenario_name'] = candidates.apply(
        lambda r: f"{int(r.pct_converted)}% converted — GI {int(r.green_infrastructure_pct)}% / FF {int(r.food_forest_pct)}%",
        axis=1
    )

    pareto = compute_pareto(candidates).copy()
    pareto['score'] = (
        pareto['flood_reduction'] / max_flood +
        pareto['mean_hm'] / max_cool +
        pareto['food_mln_lbs'] / (max_food if max_food > 0 else 1)
    )
    pareto = pareto.sort_values('score', ascending=False)

    # Drop near-duplicates in tradeoff space before returning
    pareto['flood_rounded'] = pareto['flood_reduction'].round(-1)
    pareto['hm_rounded']    = pareto['mean_hm'].round(1)
    pareto = pareto.drop_duplicates(subset=['flood_rounded', 'hm_rounded'])
    pareto = pareto.drop(columns=['flood_rounded', 'hm_rounded', 'score'])

    return pareto.head(5)
