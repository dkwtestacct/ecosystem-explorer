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
                      'carbon_tons_co2_yr', 'nature_access_pct']]
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model


def predict_with_uncertainty(model, X):
    """
    Return mean prediction and 10th/90th percentile bands across RF trees.
    X should be shape (n_samples, n_features).
    Returns: mean (n,6), lower (n,6), upper (n,6)
    Columns: [flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet,
              carbon_tons_co2_yr, nature_access_pct]
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
        'carbon_tons_co2_yr':       mean_preds[meets, 4].round(1),
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
