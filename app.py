import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import plotly.graph_objects as go
import rasterio
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR_FLOOD   = 'data/flood'
DATA_DIR_COOLING = 'data/cooling'

BASELINE_CN           = 75.7
BASELINE_HM           = 0.2719
BASELINE_FOOD_MLN_LBS = 0.0

PIXEL_AREA_ACRES     = 0.222
FOOD_FOREST_LBS_ACRE = 11_500

DEVELOPED_CODES   = [21, 22, 23]
CODE_GREEN_INFRA  = 90
CODE_FOOD_FOREST  = 41
CODE_HIGH_DENSITY = 24
NODATA            = -128

# ── Cost defaults ($/acre) ─────────────────────────────────────────────────────
DEFAULT_COST_GI   = 50_000   # Green infrastructure / woody wetlands
DEFAULT_COST_FF   = 10_000   # Food forest
DEFAULT_COST_HD   =  5_000   # High density development

# ── Metric translation constants ───────────────────────────────────────────────
# SCS design storm: 2-inch rainfall event (typical minor storm)
DESIGN_STORM_INCHES   = 2.0
# HM → temperature: each +1.0 HM ≈ 4°F cooling vs fully paved (calibrated for Minneapolis)
HM_TO_FAHRENHEIT      = 4.0
# Food: average American consumes ~2,000 lbs of food per year
LBS_PER_PERSON_YEAR   = 2_000

CHANGE_COLORS = {
    'Unchanged':            '#d3d3d3',
    'Green Infrastructure': '#2196a0',
    'Food Forest':          '#4caf50',
    'High Density':         '#e53935',
}

REF_SCENARIOS = {
    'Baseline':                     {'flood': 24.3,  'cooling': 0.2719, 'color': 'steelblue'},
    'All Food Forest (NLCD 41)':    {'flood': 29.9,  'cooling': 0.8284, 'color': 'green'},
    'All Green Infra (NLCD 90)':    {'flood': 83.0,  'cooling': 0.8633, 'color': 'teal'},
    'All High Density (NLCD 24)':   {'flood': 18.8,  'cooling': 0.1923, 'color': 'red'},
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ecosystem Explorer", layout="wide")
st.title("🌿 Urban Ecosystem Tradeoff Explorer")
st.markdown(
    "Explore how converting developed land into green infrastructure or food forests "
    "changes **flood risk**, **urban cooling**, and **food production** across the city.  \n"
    "_Prototype using Minneapolis, MN data. Food yield estimated from San Antonio NatCap "
    "benchmarks (~11,500 lbs/acre/year for food forests). San Antonio scenarios coming soon._"
)
st.info(
    "Use the sliders to create a scenario, then explore tradeoffs across flood reduction, "
    "cooling, and food production. **Green Infrastructure** converts developed land to woody wetlands "
    "(NLCD code 90) — best for flood retention. **Food Forest** converts to deciduous forest "
    "(NLCD code 41, used as a food production proxy) — best for cooling and food. "
    "**High Density** adds impervious development — worst for all three."
)

# ── Session state ──────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []
if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = None
# Slider apply state — used by "Apply" button from optimizer
if "slider_pct_converted" not in st.session_state:
    st.session_state.slider_pct_converted = 10
if "slider_gi_pct" not in st.session_state:
    st.session_state.slider_gi_pct = 0
if "slider_ff_pct" not in st.session_state:
    st.session_state.slider_ff_pct = 0

# Apply any pending slider values before sliders are rendered
if "_pending_pct" in st.session_state:
    st.session_state.slider_pct_converted = st.session_state.pop("_pending_pct")
    st.session_state.slider_gi_pct        = st.session_state.pop("_pending_gi")
    st.session_state.slider_ff_pct        = st.session_state.pop("_pending_ff")

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    bio = pd.read_csv(f'{DATA_DIR_FLOOD}/UFR_biophysical_table_MN.csv')

    with rasterio.open(f'{DATA_DIR_FLOOD}/LULC_NLCD_2021_MN.tif') as src:
        lulc = src.read(1)
    with rasterio.open(f'{DATA_DIR_FLOOD}/soil_group_MN.tif') as src:
        soil = src.read(1)

    cooling_bio = pd.read_csv(f'{DATA_DIR_COOLING}/biophysical_table_urban_cooling.csv')
    with rasterio.open(f'{DATA_DIR_COOLING}/land_use_2021.tif') as src:
        cooling_lulc = src.read(1)

    developed_pixels = np.argwhere(np.isin(cooling_lulc, DEVELOPED_CODES))

    cn_by_soil = {
        row['lucode']: {1: row['CN_A'], 2: row['CN_B'], 3: row['CN_C'], 4: row['CN_D']}
        for _, row in bio.iterrows()
    }
    all_lucodes = sorted(cn_by_soil.keys())
    lucode_to_idx = {lc: i + 1 for i, lc in enumerate(all_lucodes)}

    cn_table = np.zeros((len(all_lucodes) + 1, 5), dtype=np.float32)
    for lc, soils in cn_by_soil.items():
        for sg, cn_val in soils.items():
            cn_table[lucode_to_idx[lc], sg] = cn_val

    max_raster_lucode = int(max(cooling_lulc.max(), lulc.max(), max(all_lucodes)))
    lucode_idx_arr = np.zeros(max_raster_lucode + 1, dtype=np.int32)
    for lc, idx in lucode_to_idx.items():
        lucode_idx_arr[int(lc)] = idx

    soil_resized = resize(soil, lulc.shape, order=0, preserve_range=True).astype(int)

    cooling_bio['HM'] = (cooling_bio['shade'] + cooling_bio['kc']) / 2
    hm_lookup = dict(zip(cooling_bio['lucode'], cooling_bio['HM']))
    max_hm_lucode = max(hm_lookup.keys())
    hm_arr = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    for lc, hm in hm_lookup.items():
        hm_arr[lc] = hm

    # ── Equity proxy raster ────────────────────────────────────────────────────
    # TODO: replace with real heat vulnerability index (e.g. CDC/ATSDR HVI by census tract)
    # For now: weight developed pixels by land-use intensity as a rough proxy —
    # high-intensity developed (code 23) scores 1.0, medium (22) scores 0.6, low (21) scores 0.3.
    equity_weights = np.zeros(cooling_lulc.shape, dtype=np.float32)
    equity_weights[cooling_lulc == 23] = 1.0   # high-intensity developed → highest need
    equity_weights[cooling_lulc == 22] = 0.6
    equity_weights[cooling_lulc == 21] = 0.3

    return (lulc, soil_resized, cooling_lulc, developed_pixels,
            cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode,
            equity_weights)


(lulc, soil_resized, cooling_lulc, developed_pixels,
 cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode,
 equity_weights) = load_data()


# ── Metric translation helpers ─────────────────────────────────────────────────
def cn_to_runoff_acre_feet(mean_cn, total_developed_acres):
    """
    SCS curve number method: convert mean CN to direct runoff depth for a design storm,
    then scale to total developed area in acre-feet.
    """
    if mean_cn <= 0:
        return 0.0
    P = DESIGN_STORM_INCHES
    S = (1000.0 / mean_cn) - 10.0
    Ia = 0.2 * S  # initial abstraction
    if P <= Ia:
        return 0.0
    Q_inches = (P - Ia) ** 2 / (P - Ia + S)   # runoff depth in inches
    Q_feet   = Q_inches / 12.0
    return round(Q_feet * total_developed_acres, 1)


def hm_to_fahrenheit_cooling(mean_hm):
    """Translate HM index delta vs baseline into approximate °F cooling."""
    delta_hm = mean_hm - BASELINE_HM
    return round(delta_hm * HM_TO_FAHRENHEIT, 1)


def food_to_people_fed(food_mln_lbs):
    """Translate food production (M lbs/yr) to approximate people fed."""
    lbs = food_mln_lbs * 1_000_000
    return int(lbs / LBS_PER_PERSON_YEAR)


def compute_cost(n_wet_pixels, n_for_pixels, n_hd_pixels,
                 cost_gi, cost_ff, cost_hd):
    """Total implementation cost in $M."""
    acres_gi = n_wet_pixels * PIXEL_AREA_ACRES
    acres_ff = n_for_pixels * PIXEL_AREA_ACRES
    acres_hd = n_hd_pixels  * PIXEL_AREA_ACRES
    total = acres_gi * cost_gi + acres_ff * cost_ff + acres_hd * cost_hd
    return round(total / 1_000_000, 2)   # return in $M


def compute_cost_effectiveness(results, baseline_runoff_acft):
    """Return $/unit ratios vs baseline; None where denominator is zero or negative."""
    cost = results['total_cost_mln'] * 1_000_000
    if cost <= 0:
        return {'cost_per_acft': None, 'cost_per_degf': None, 'cost_per_1k_people': None}

    runoff_prevented = baseline_runoff_acft - results['runoff_acre_feet']
    cost_per_acft = round(cost / runoff_prevented) if runoff_prevented > 0 else None

    cooling_f = results['cooling_f']
    cost_per_degf = round(cost / cooling_f) if cooling_f > 0 else None

    people_fed = results['people_fed']
    cost_per_1k_people = round(cost / (people_fed / 1000)) if people_fed > 0 else None

    return {
        'cost_per_acft':       cost_per_acft,
        'cost_per_degf':       cost_per_degf,
        'cost_per_1k_people':  cost_per_1k_people,
    }


def _fmt_ce(val):
    if val is None:
        return "N/A"
    if val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    return f"${val:,.0f}"


# ── Scenario evaluation ────────────────────────────────────────────────────────
def evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct,
                      seed=42, use_heat_priority=False,
                      cost_gi=DEFAULT_COST_GI,
                      cost_ff=DEFAULT_COST_FF,
                      cost_hd=DEFAULT_COST_HD):
    """
    Convert a random (or equity-weighted) sample of developed pixels to the specified
    land use mix, then compute flood risk, urban cooling, food production, and cost.
    """
    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

    scenario_lulc = cooling_lulc.copy()
    n_convert = int(len(developed_pixels) * pct_converted / 100)

    rng = np.random.default_rng(seed)

    if use_heat_priority and n_convert > 0:
        # Weight sampling toward high-need pixels using equity proxy
        weights = equity_weights[developed_pixels[:, 0], developed_pixels[:, 1]].astype(float)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = None
        chosen_idx = rng.choice(len(developed_pixels), size=n_convert, replace=False, p=weights)
    else:
        chosen_idx = rng.choice(len(developed_pixels), size=n_convert, replace=False)

    pixels_to_convert = developed_pixels[chosen_idx]

    n_wet = int(n_convert * green_infrastructure_pct / 100)
    n_for = int(n_convert * food_forest_pct / 100)
    n_hd  = n_convert - n_wet - n_for

    if n_wet > 0:
        p = pixels_to_convert[:n_wet]
        scenario_lulc[p[:, 0], p[:, 1]] = CODE_GREEN_INFRA
    if n_for > 0:
        p = pixels_to_convert[n_wet:n_wet + n_for]
        scenario_lulc[p[:, 0], p[:, 1]] = CODE_FOOD_FOREST
    if n_hd > 0:
        p = pixels_to_convert[n_wet + n_for:]
        scenario_lulc[p[:, 0], p[:, 1]] = CODE_HIGH_DENSITY

    soil_clamped = np.clip(soil_resized, 1, 4)
    lulc_safe    = np.clip(scenario_lulc, 0, len(lucode_idx_arr) - 1)
    lulc_idx     = lucode_idx_arr[lulc_safe]
    cn_scenario  = cn_table[lulc_idx, soil_clamped]
    mean_cn      = float(cn_scenario[cn_scenario > 0].mean().round(2))

    hm_safe  = np.clip(scenario_lulc, 0, len(hm_arr) - 1)
    hm_map   = hm_arr[hm_safe].astype(float)
    hm_map[(scenario_lulc < 0) | (scenario_lulc >= len(hm_arr))] = np.nan
    valid_hm = hm_map[~np.isnan(hm_map) & (scenario_lulc != NODATA)]
    mean_hm  = float(valid_hm.mean().round(4))

    n_food_pixels = int((scenario_lulc == CODE_FOOD_FOREST).sum())
    food_mln_lbs  = round(n_food_pixels * PIXEL_AREA_ACRES * FOOD_FOREST_LBS_ACRE / 1_000_000, 3)

    total_developed_acres = len(developed_pixels) * PIXEL_AREA_ACRES
    total_cost_mln = compute_cost(n_wet, n_for, n_hd, cost_gi, cost_ff, cost_hd)

    return {
        'pct_converted':            pct_converted,
        'green_infrastructure_pct': green_infrastructure_pct,
        'food_forest_pct':          food_forest_pct,
        'pct_highdensity':          pct_highdensity,
        'n_wet':                    n_wet,
        'n_for':                    n_for,
        'n_hd':                     n_hd,
        'mean_cn':                  mean_cn,
        'flood_reduction':          round(100 - mean_cn, 2),
        'runoff_acre_feet':         cn_to_runoff_acre_feet(mean_cn, total_developed_acres),
        'mean_hm':                  mean_hm,
        'cooling_f':                hm_to_fahrenheit_cooling(mean_hm),
        'food_mln_lbs':             food_mln_lbs,
        'people_fed':               food_to_people_fed(food_mln_lbs),
        'total_cost_mln':           total_cost_mln,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
@st.cache_data
def compute_scenario_grid(data_dir_flood, data_dir_cooling):
    rows = [
        {k: v for k, v in evaluate_scenario(pct, gi, ff, seed=42).items() if k != 'scenario_lulc'}
        for pct in range(0, 51, 10)
        for gi  in range(0, 101, 25)
        for ff  in range(0, 101, 25)
        if gi + ff <= 100
    ]
    return pd.DataFrame(rows)


@st.cache_data
def compute_lookup_table(data_dir_flood, data_dir_cooling):
    """Pre-compute results for every valid slider position (step=5) for instant response."""
    table = {}
    for pct in range(0, 51, 5):
        for gi in range(0, 101, 5):
            for ff in range(0, 101, 5):
                if gi + ff <= 100:
                    r = evaluate_scenario(pct, gi, ff, seed=42)
                    table[(pct, gi, ff)] = {k: v for k, v in r.items() if k != 'scenario_lulc'}
    return table


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


with st.spinner("Loading data and pre-computing scenarios..."):
    scenario_df  = compute_scenario_grid(DATA_DIR_FLOOD, DATA_DIR_COOLING)
    lookup_table = compute_lookup_table(DATA_DIR_FLOOD, DATA_DIR_COOLING)

MAX_FOOD  = float(scenario_df['food_mln_lbs'].max())
MAX_FLOOD = 100.0
MAX_COOL  = 1.1

BASELINE_RUNOFF_ACRE_FEET = cn_to_runoff_acre_feet(
    BASELINE_CN, len(developed_pixels) * PIXEL_AREA_ACRES
)

# ── Surrogate model ────────────────────────────────────────────────────────────
@st.cache_resource
def train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling):
    X = _scenario_df[['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']]
    y = _scenario_df[['flood_reduction', 'mean_hm', 'food_mln_lbs']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


surrogate = train_surrogate(scenario_df, DATA_DIR_FLOOD, DATA_DIR_COOLING)


def predict_with_uncertainty(model, X):
    """
    Return mean prediction and 10th/90th percentile bands across RF trees.
    X should be shape (n_samples, n_features).
    Returns: mean (n,3), lower (n,3), upper (n,3)
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds shape: (n_trees, n_samples, n_outputs)
    mean  = tree_preds.mean(axis=0)
    lower = np.percentile(tree_preds, 10, axis=0)
    upper = np.percentile(tree_preds, 90, axis=0)
    return mean, lower, upper

def plot_feature_importance(model):
    """Bar chart of RF feature importances across all three output metrics."""
    feature_names = ['% Converted', 'Green Infra %', 'Food Forest %']
    metric_names  = ['Flood Reduction', 'Cooling (HM)', 'Food Production']
    
    # Each estimator in a MultiOutputRegressor-style RF predicts all outputs
    # feature_importances_ is averaged across all trees
    importances = model.feature_importances_  # shape (n_features,)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = ['#2196a0', '#4caf50', '#e53935']
    bars = ax.barh(feature_names, importances, color=colors)
    ax.set_xlabel('Relative Importance', fontsize=9)
    ax.set_title('What drives outcomes most?', fontsize=10)
    ax.set_xlim(0, max(importances) * 1.3)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    return fig

def optimize_scenario(surrogate, min_flood, min_cool, min_food, n_samples=10000):
    """Use the surrogate to find Pareto-optimal scenarios meeting the given constraints."""
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
        (mean_preds[:, 2] >= min_food)
    )
    if not meets.any():
        return None

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
        pareto['flood_reduction'] / MAX_FLOOD +
        pareto['mean_hm'] / MAX_COOL +
        pareto['food_mln_lbs'] / (MAX_FOOD if MAX_FOOD > 0 else 1)
    )
    pareto = pareto.sort_values('score', ascending=False)
    
    # Drop near-duplicates in tradeoff space before returning
    pareto['flood_rounded'] = pareto['flood_reduction'].round(0)
    pareto['hm_rounded']    = pareto['mean_hm'].round(2)
    pareto = pareto.drop_duplicates(subset=['flood_rounded', 'hm_rounded'])
    pareto = pareto.drop(columns=['flood_rounded', 'hm_rounded', 'score'])
    
    return pareto.head(5)

# ── Plotting helpers ───────────────────────────────────────────────────────────
def render_matplotlib(fig):
    try:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)


# ── Matplotlib plots ───────────────────────────────────────────────────────────
def plot_bars(results):
    """Three bar charts: flood risk, cooling, food production vs baseline."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    ax1.bar(['Baseline', 'This Scenario'], [BASELINE_CN, results['mean_cn']],
            color=['steelblue', 'purple'])
    ax1.axhline(BASELINE_CN, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean Curve Number (lower = less runoff)')
    ax1.set_title(f'Flood Risk  —  CN = {results["mean_cn"]}')
    ax1.set_ylim(0, 100)

    ax2.bar(['Baseline', 'This Scenario'], [BASELINE_HM, results['mean_hm']],
            color=['steelblue', 'purple'])
    ax2.axhline(BASELINE_HM, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Heat Mitigation Index (higher = more cooling)')
    ax2.set_title(f'Urban Cooling  —  HM = {results["mean_hm"]}')
    ax2.set_ylim(0, 1.1)

    ax3.bar(['Baseline', 'This Scenario'], [BASELINE_FOOD_MLN_LBS, results['food_mln_lbs']],
            color=['steelblue', 'purple'])
    ax3.set_ylabel('Food Production (million lbs/year)')
    ax3.set_title(f'Food Production  —  {results["food_mln_lbs"]:.3f}M lbs/yr')
    ax3.set_ylim(0, max(MAX_FOOD * 1.1, 0.01))

    plt.tight_layout()
    return fig


def plot_spatial_map(scenario_lulc, baseline_lulc):
    h, w = scenario_lulc.shape
    rgb = np.full((h, w, 3), mcolors.to_rgb(CHANGE_COLORS['Unchanged']))

    rgb[(baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_GREEN_INFRA)] = \
        mcolors.to_rgb(CHANGE_COLORS['Green Infrastructure'])
    rgb[(baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_FOOD_FOREST)] = \
        mcolors.to_rgb(CHANGE_COLORS['Food Forest'])
    rgb[(baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_HIGH_DENSITY)] = \
        mcolors.to_rgb(CHANGE_COLORS['High Density'])
    rgb[baseline_lulc == NODATA] = (1.0, 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis('off')
    ax.set_title('Land Use Changes', fontsize=12)
    ax.legend(handles=[
        Patch(facecolor=CHANGE_COLORS['Unchanged'],            label='Unchanged'),
        Patch(facecolor=CHANGE_COLORS['Green Infrastructure'], label='→ Green Infrastructure'),
        Patch(facecolor=CHANGE_COLORS['Food Forest'],          label='→ Food Forest'),
        Patch(facecolor=CHANGE_COLORS['High Density'],         label='→ High Density'),
    ], loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


# ── Plotly tradeoff plot ───────────────────────────────────────────────────────
def food_to_size(food_vals, max_food, base=5, scale=120):
    food_vals = np.atleast_1d(np.asarray(food_vals, dtype=float))
    if max_food > 0:
        return base + scale * np.sqrt(food_vals / max_food)
    return np.full(len(food_vals), base)


def convex_hull_trace(df):
    from scipy.spatial import ConvexHull
    points = df[['flood_reduction', 'mean_hm']].values
    try:
        hull = ConvexHull(points)
        hull_pts = points[np.append(hull.vertices, hull.vertices[0])]
        return go.Scatter(
            x=hull_pts[:, 0],
            y=hull_pts[:, 1],
            mode='lines',
            line=dict(color='rgba(180,180,180,0.5)', width=1.5, dash='dot'),
            fill='toself',
            fillcolor='rgba(200,200,200,0.08)',
            hoverinfo='skip',
            name='Feasible space',
            showlegend=True,
        )
    except Exception:
        return None


def plot_tradeoff(results, scenario_df, lookup_table=None, saved=None, optimized=None):
    max_food = scenario_df['food_mln_lbs'].max()
    fig = go.Figure()

    hull_source = pd.DataFrame(lookup_table.values()) if lookup_table else scenario_df
    hull_tr = convex_hull_trace(hull_source)
    if hull_tr:
        fig.add_trace(hull_tr)

    TEXT_POSITIONS = {
        'Baseline':                   None,              # too crowded — use legend/hover
        'All Food Forest (NLCD 41)':  'top left',
        'All Green Infra (NLCD 90)':  'top right',
        'All High Density (NLCD 24)': None,              # too crowded — use legend/hover
    }

    for name, ref in REF_SCENARIOS.items():
        text_pos = TEXT_POSITIONS.get(name, 'top right')
        fig.add_trace(go.Scatter(
            x=[ref['flood']], y=[ref['cooling']],
            mode='markers+text' if text_pos else 'markers',
            marker=dict(size=13, color=ref['color'], opacity=0.85,
                        line=dict(color='white', width=1)),
            text=[name] if text_pos else None,
            textposition=text_pos,
            textfont=dict(size=10),
            hovertemplate=(
                f"<b>{name}</b> (reference benchmark)<br>"
                f"Flood reduction: {ref['flood']} | Cooling HM: {ref['cooling']:.4f}"
                "<extra></extra>"
            ),
            name=name,
        ))

    if saved is not None and len(saved) > 0:
        df_saved = pd.DataFrame(saved)
        sizes = np.clip(food_to_size(df_saved['food_mln_lbs'].values, max_food), 5, 30)
        fig.add_trace(go.Scatter(
            x=df_saved['flood_reduction'],
            y=df_saved['mean_hm'],
            mode='markers',
            marker=dict(size=sizes, color='purple', opacity=0.55,
                        line=dict(color='white', width=1)),
            text=df_saved.apply(
                lambda r: (
                    f"{r.scenario_name}<br>"
                    f"Flood: {r.flood_reduction:.1f} | Cooling: {r.mean_hm:.4f} | "
                    f"Food: {r.food_mln_lbs:.3f}M lbs"
                ), axis=1),
            hoverinfo='text',
            name='Saved scenarios',
        ))
        pareto_df = compute_pareto(df_saved).sort_values('flood_reduction')
        fig.add_trace(go.Scatter(
            x=pareto_df['flood_reduction'],
            y=pareto_df['mean_hm'],
            mode='markers+lines',
            marker=dict(size=14, color='gold', symbol='circle',
                        line=dict(color='black', width=1)),
            line=dict(color='gold', dash='dash', width=1),
            text=pareto_df.apply(
                lambda r: (
                    f"<b>Pareto optimal</b><br>{r.scenario_name}<br>"
                    f"Flood: {r.flood_reduction:.1f} | Cooling: {r.mean_hm:.4f}"
                ), axis=1),
            hoverinfo='text',
            name='Pareto optimal (saved)',
        ))

    if optimized is not None and len(optimized) > 0:
        opt_sizes = np.clip(food_to_size(optimized['food_mln_lbs'].values, max_food), 5, 30)
        # Error bars from uncertainty bands
        flood_err_minus = (optimized['flood_reduction'] - optimized['flood_lower']).values
        flood_err_plus  = (optimized['flood_upper']     - optimized['flood_reduction']).values
        hm_err_minus    = (optimized['mean_hm']         - optimized['hm_lower']).values
        hm_err_plus     = (optimized['hm_upper']        - optimized['mean_hm']).values
        fig.add_trace(go.Scatter(
            x=optimized['flood_reduction'],
            y=optimized['mean_hm'],
            mode='markers',
            marker=dict(size=opt_sizes, color='orange', symbol='diamond',
                        line=dict(color='black', width=1.5)),
            error_x=dict(type='data', symmetric=False,
                    array=flood_err_plus, arrayminus=flood_err_minus,
                    color='rgba(255,165,0,0.2)', thickness=1, width=4),
            error_y=dict(type='data', symmetric=False,
                    array=hm_err_plus, arrayminus=hm_err_minus,
                    color='rgba(255,165,0,0.2)', thickness=1, width=4),
            text=optimized.apply(
                lambda r: (
                    f"<b>Optimized suggestion</b><br>{r.scenario_name}<br>"
                    f"Flood: {r.flood_reduction:.1f} [{r.flood_lower:.1f}–{r.flood_upper:.1f}]<br>"
                    f"Cooling: {r.mean_hm:.4f} [{r.hm_lower:.4f}–{r.hm_upper:.4f}]<br>"
                    f"Food: {r.food_mln_lbs:.3f}M lbs [{r.food_lower:.3f}–{r.food_upper:.3f}]"
                ), axis=1),
            hoverinfo='text',
            name='Optimized suggestions',
        ))

    fig.add_trace(go.Scatter(
        x=[results['flood_reduction']],
        y=[results['mean_hm']],
        mode='markers',
        marker=dict(size=20, color='purple', symbol='star',
                    line=dict(color='white', width=1.5)),
        hovertemplate=(
            f"<b>This Scenario</b><br>"
            f"Flood reduction: {results['flood_reduction']:.1f}<br>"
            f"Cooling HM: {results['mean_hm']:.4f}<br>"
            f"Food: {results['food_mln_lbs']:.3f}M lbs/yr<br>"
            f"Cost: ${results['total_cost_mln']:.1f}M"
            "<extra></extra>"
        ),
        name='This scenario',
    ))

    fig.add_hline(y=results['mean_hm'], line_dash='dot', line_color='purple', opacity=0.25)
    fig.add_vline(x=results['flood_reduction'], line_dash='dot', line_color='purple', opacity=0.25)

    fig.update_layout(
        title='Tradeoff Space  —  bubble size = food production (saved/optimized scenarios only)',
        xaxis_title='Flood Risk Reduction (higher = better)',
        yaxis_title='Heat Mitigation Index (higher = better)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 1.1]),
        height=520,
        legend=dict(orientation='v', x=1.02, y=1,
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        hovermode='closest',
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")

pct_converted = st.sidebar.slider(
    "% of developed land to convert", 0, 50,
    key="slider_pct_converted"
)

st.sidebar.subheader("Conversion Mix")
st.sidebar.caption(
    "Allocate converted land across three uses — must sum to 100%. "
    "High Density auto-fills as the remainder, but it can also be explicitly adjusted."
)

green_infrastructure_pct = st.sidebar.number_input(
    "Green Infrastructure %", 0, 100,
    value=st.session_state.get("slider_gi_pct", 0),
    step=5, key="slider_gi_pct"
)
food_forest_pct = st.sidebar.number_input(
    "Food Forest %", 0, 100,
    value=st.session_state.get("slider_ff_pct", 0),
    step=5, key="slider_ff_pct"
)

auto_hd = 100 - green_infrastructure_pct - food_forest_pct
pct_highdensity = st.sidebar.number_input(
    "High Density %", 0, 100,
    value=max(0, auto_hd),
    step=5
)

total = green_infrastructure_pct + food_forest_pct + pct_highdensity

if total < 100:
    st.sidebar.warning(f"⚠️ Total is {total}% — must equal 100%.")
    st.stop()
elif total > 100:
    st.sidebar.error(f"⚠️ Total is {total}% — reduce one of the values.")
    st.stop()
else:
    st.sidebar.success("✅ Mix sums to 100%.")

st.sidebar.divider()

# ── Cost sliders ──────────────────────────────────────────────────────────────
st.sidebar.subheader("💰 Implementation Costs ($/acre)")
cost_gi = st.sidebar.slider("Green Infrastructure ($/acre)", 5_000, 150_000,
                              DEFAULT_COST_GI, 5_000,
                              help="Typical range: $20k–$100k/acre for constructed wetlands")
cost_ff = st.sidebar.slider("Food Forest ($/acre)", 1_000, 50_000,
                              DEFAULT_COST_FF, 1_000,
                              help="Typical range: $5k–$20k/acre for food forest establishment")
cost_hd = st.sidebar.slider("High Density Infill ($/acre)", 1_000, 50_000,
                              DEFAULT_COST_HD, 1_000,
                              help="Marginal cost of additional impervious development")

st.sidebar.divider()

# ── Equity toggle ─────────────────────────────────────────────────────────────
st.sidebar.subheader("🌡️ Heat Vulnerability-Weighted Conversion")
use_heat_priority = st.sidebar.toggle(
    "Prioritize high heat-burden areas",
    value=False,

    help=(
        "When ON, conversions are weighted toward higher-intensity developed land "
        "(NLCD codes 23 > 22 > 21) as a proxy for heat exposure. "
        "This is a simplified approximation — not a measured temperature or vulnerability index."
    )

)
if use_heat_priority:
    st.sidebar.caption(
        "🌡️ **Heat vulnerability mode active** — conversions skewed toward high-intensity "
        "developed pixels (code 23 > 22 > 21 as heat exposure proxy)."
    )

st.sidebar.divider()

st.sidebar.subheader("Example Scenarios")

if st.sidebar.button("🌳 Food Forest (Cooling + Food Focus)"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 100
    st.rerun()

if st.sidebar.button("🌊 Green Infrastructure (Flood Mitigation)"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 100
    st.session_state._pending_ff = 0
    st.rerun()

if st.sidebar.button("🏙️ High Density Development"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 0
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🔍 Find Best Scenario")
st.sidebar.caption("Set minimum targets and let the surrogate model find optimal inputs.")

st.sidebar.caption(
    "Optimization currently targets flood, cooling, and food only. "
    "Cost and equity mode are not yet included in the surrogate."
)

min_flood = st.sidebar.slider("Min flood reduction", 0, 90, 30, 5)
min_cool  = st.sidebar.slider("Min cooling (HM)", 0.0, 1.0, 0.3, 0.05)
min_food  = st.sidebar.slider("Min food production (M lbs)", 0.0, float(max(MAX_FOOD, 0.1)), 0.0, 0.01)
        
if st.sidebar.button("Optimize"):
    with st.spinner("Searching for optimal scenarios..."):
        st.session_state.optimized_results = optimize_scenario(
            surrogate, min_flood, min_cool, min_food)
    if st.session_state.optimized_results is None:
        st.sidebar.warning("Optimizer returned None — no scenarios met constraints.")
    else:
        st.sidebar.success(f"Found {len(st.session_state.optimized_results)} scenarios.")

st.sidebar.divider()
st.sidebar.caption(
    "**Green Infrastructure** = woody wetlands (NLCD 90) — best for flood retention.  \n"
    "**Food Forest** = deciduous forest (NLCD 41, food production proxy) — best for cooling + food.  \n"
    "**High Density** = paved development (NLCD 24) — worst for all three."
)

# ── Main panel ─────────────────────────────────────────────────────────────────
lookup_key = (pct_converted, green_infrastructure_pct, food_forest_pct)
if lookup_key in lookup_table and not use_heat_priority:
    # Lookup table was computed without equity weighting — only use it in standard mode
    results = lookup_table[lookup_key].copy()
    results['scenario_lulc'] = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
        use_heat_priority=False, cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd
    )['scenario_lulc']
    # Recompute cost with current cost sliders (lookup table used default costs)
    results['total_cost_mln'] = compute_cost(
        results['n_wet'], results['n_for'], results['n_hd'],
        cost_gi, cost_ff, cost_hd
    )
else:
    results = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
        use_heat_priority=use_heat_priority, cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd
    )

# ── Top metric cards ───────────────────────────────────────────────────────────
def _fmt_runoff(af):
    if af >= 1_000:
        return f"{af / 1_000:.1f}K ac-ft"
    return f"{af:.0f} ac-ft"

def _fmt_food(mln_lbs):
    if mln_lbs >= 1:
        return f"{mln_lbs:.2f}M lbs/yr"
    return f"{mln_lbs * 1_000:.1f}K lbs/yr"

def _fmt_people(n):
    if n >= 1_000:
        return f"~{n // 1_000}K people"
    return f"~{n} people"

row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric(
    "Flood Risk Reduction",
    f"{results['flood_reduction']:.1f}",
    delta=f"{results['flood_reduction'] - (100 - BASELINE_CN):.1f} vs baseline",
    delta_color="normal",
    help="SCS Curve Number based. Higher = less runoff."
)
row1_col2.metric(
    "Urban Cooling",
    f"≈{results['cooling_f']:+.1f}°F",
    delta=f"HM {results['mean_hm']:.4f} vs {BASELINE_HM}",
    help="Approximate temperature difference vs baseline, based on Heat Mitigation Index (calibration factor 4°F/HM unit)."
)
row1_col3.metric(
    "Runoff Prevented",
    _fmt_runoff(results['runoff_acre_feet']),
    delta=None,
    help=f"Acre-feet of runoff from a {DESIGN_STORM_INCHES}-inch design storm across all developed land."
)

row2_col1, row2_col2 = st.columns(2)
row2_col1.metric(
    "Food Production",
    _fmt_food(results['food_mln_lbs']),
    delta=_fmt_people(results['people_fed']),
    help="Estimated yield from food forest pixels at 11,500 lbs/acre/year (San Antonio NatCap benchmark)."
)
row2_col2.metric(
    "Est. Implementation Cost",
    f"${results['total_cost_mln']:.1f}M",
    delta=None,
    help="Total cost based on $/acre sliders × converted acreage. Rough order-of-magnitude only."
)

ce = compute_cost_effectiveness(results, BASELINE_RUNOFF_ACRE_FEET)
ce_col1, ce_col2, ce_col3 = st.columns(3)
ce_col1.metric(
    "Cost / Acre-Foot Prevented",
    _fmt_ce(ce['cost_per_acft']),
    delta=None,
    help=f"Implementation cost divided by runoff reduction vs baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). N/A if scenario increases runoff or has no cost."
)
ce_col2.metric(
    "Cost / °F Cooling",
    _fmt_ce(ce['cost_per_degf']),
    delta=None,
    help="Implementation cost divided by degrees F of cooling vs baseline. N/A if no cooling improvement."
)
ce_col3.metric(
    "Cost / 1,000 People Fed",
    _fmt_ce(ce['cost_per_1k_people']),
    delta=None,
    help="Implementation cost divided by (people fed ÷ 1,000). N/A if no food production."
)

mode_text = "prioritizing hotter areas" if use_heat_priority else "using random placement"

st.write(
    f"This scenario converts **{pct_converted}%** of developed land, allocating "
    f"**{green_infrastructure_pct}%** to green infrastructure, "
    f"**{food_forest_pct}%** to food forest, and **{pct_highdensity}%** "
    f"to high-density development, {mode_text}."
)

st.caption(
    "Prototype tool for exploring tradeoffs — outputs are directional and intended for comparison, not precise prediction."
)

st.caption(
    "Flood reduction is derived from curve number, cooling from a heat mitigation index, "
    "and food production from a food-forest yield benchmark. Use these as comparative indicators."
)

st.caption(
    "Higher is better for flood, cooling, and food metrics. "
    "Cooling °F is approximate (±2°F). Runoff uses a 2-inch design storm. "
    "Cost is order-of-magnitude — adjust $/acre sliders in sidebar."
)

with st.expander("Assumptions and limitations"):
    st.markdown(
        "- **Green Infrastructure** is modeled as woody wetlands (NLCD code 90).\n"
        "- **Food Forest** is modeled as deciduous forest (NLCD code 41) as a proxy "
        "for food-producing tree cover; yield estimated at 11,500 lbs/acre/year.\n"
        "- Land conversion is stylized rather than policy-constrained.\n"
        "- Food production uses a benchmark yield estimate and should be treated as directional.\n"
        "- Spatial placement is simplified and not yet corridor- or parcel-specific.\n"
        "- Optimized results come from a surrogate model and should be verified."
    )

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Scenario", "🔀 Tradeoff Analysis", "🗺️ Map View"])

with tab1:
    st.subheader("Outcome Comparison")
    render_matplotlib(plot_bars(results))

with tab2:
    st.subheader("Tradeoff Space")
    st.plotly_chart(plot_tradeoff(
        results, scenario_df,
        lookup_table=lookup_table,
        saved=st.session_state.saved_scenarios,
        optimized=st.session_state.optimized_results
    ), use_container_width=True)

    if st.button("💾 Save this scenario"):
        saved = {k: v for k, v in results.items() if k != 'scenario_lulc'}
        saved["heat_priority"] = use_heat_priority
        saved["cost_gi"] = cost_gi
        saved["cost_ff"] = cost_ff
        saved["cost_hd"] = cost_hd
        _ce = compute_cost_effectiveness(results, BASELINE_RUNOFF_ACRE_FEET)
        saved["cost_per_acft"]      = _ce['cost_per_acft']
        saved["cost_per_degf"]      = _ce['cost_per_degf']
        saved["cost_per_1k_people"] = _ce['cost_per_1k_people']
        st.session_state.saved_scenarios.append(saved)
        st.success(f"Saved: {results['scenario_name']}")

    if st.session_state.optimized_results is not None:
        st.divider()
        st.subheader("🔍 Optimized Scenario Suggestions")
        opt = st.session_state.optimized_results
        if opt is None or len(opt) == 0:
            st.warning("No scenarios found meeting those constraints. Try lowering the targets.")
        else:
            st.caption(
                f"Top scenarios meeting flood ≥ {min_flood}, cooling ≥ {min_cool:.2f}, "
                f"food ≥ {min_food:.3f}M lbs — ranked by balanced score. "
                "Numbers are surrogate model predictions with 10th–90th percentile uncertainty bands."
            )

            # Display table with uncertainty columns
            display_cols = ['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                            'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs']
            # Add uncertainty columns if present
            unc_cols = [c for c in ['flood_lower', 'flood_upper', 'hm_lower', 'hm_upper',
                                    'food_lower', 'food_upper'] if c in opt.columns]

            with st.expander("Show uncertainty bands", expanded=False):
                st.dataframe(opt[display_cols + unc_cols], use_container_width=True)
            st.dataframe(opt[display_cols], use_container_width=True)

            st.caption("**Influence Map** — which input drives outcomes most according to the surrogate model:")
            render_matplotlib(plot_feature_importance(surrogate))

            best = opt.iloc[0]

            # ── Apply button: loads best scenario into sliders ─────────────────
            apply_col, info_col = st.columns([1, 3])
            with apply_col:
                if st.button("▶️ Apply best to sliders", type="primary"):
                    st.session_state._pending_pct = int(round(best.pct_converted / 5) * 5)
                    st.session_state._pending_gi  = int(round(best.green_infrastructure_pct / 5) * 5)
                    st.session_state._pending_ff  = int(round(best.food_forest_pct / 5) * 5)
                    if st.session_state._pending_gi + st.session_state._pending_ff > 100:
                        st.session_state._pending_ff = 100 - st.session_state._pending_gi
                    st.rerun()

            with info_col:
                flood_unc = f" [{best.flood_lower:.1f}–{best.flood_upper:.1f}]" if 'flood_lower' in best else ""
                hm_unc    = f" [{best.hm_lower:.4f}–{best.hm_upper:.4f}]"       if 'hm_lower'    in best else ""
                food_unc  = f" [{best.food_lower:.3f}–{best.food_upper:.3f}]"   if 'food_lower'  in best else ""
                st.info(
                    f"**Best balanced scenario:** Convert {int(best.pct_converted)}% of developed land — "
                    f"{int(best.green_infrastructure_pct)}% Green Infrastructure, "
                    f"{int(best.food_forest_pct)}% Food Forest, "
                    f"{int(best.pct_highdensity)}% High Density.  \n"
                    f"Predicted: flood **{best.flood_reduction:.1f}**{flood_unc} · "
                    f"cooling HM **{best.mean_hm:.4f}**{hm_unc} · "
                    f"food **{best.food_mln_lbs:.3f}M lbs**{food_unc}"
                )

            # Additional apply buttons for all suggestions
            if len(opt) > 1:
                st.caption("Apply other suggestions:")
                btn_cols = st.columns(len(opt))
                for i, (_, row) in enumerate(opt.iterrows()):
                    with btn_cols[i]:
                        label = f"#{i+1}: {int(row.pct_converted)}% conv"
                        if st.button(label, key=f"apply_opt_{i}"):
                            st.session_state._pending_pct = int(round(row.pct_converted / 5) * 5)
                            st.session_state._pending_gi  = int(round(row.green_infrastructure_pct / 5) * 5)
                            st.session_state._pending_ff  = int(round(row.food_forest_pct / 5) * 5)
                            if st.session_state._pending_gi + st.session_state._pending_ff > 100:
                                st.session_state._pending_ff = 100 - st.session_state._pending_gi
                            st.rerun()

    if st.session_state.saved_scenarios:
        st.divider()
        with st.expander(f"📋 Saved Scenarios ({len(st.session_state.saved_scenarios)})", expanded=False):
            df_saved = pd.DataFrame(st.session_state.saved_scenarios)
            show_cols = [c for c in [
                'scenario_name',
                'pct_converted',
                'green_infrastructure_pct',
                'food_forest_pct',
                'heat_priority',
                'flood_reduction',
                'cooling_f',
                'runoff_acre_feet',
                'mean_hm',
                'food_mln_lbs',
                'people_fed',
                'total_cost_mln',
                'cost_per_acft',
                'cost_per_degf',
                'cost_per_1k_people',
                'cost_gi',
                'cost_ff',
                'cost_hd'
            ] if c in df_saved.columns]

            st.dataframe(df_saved[show_cols], use_container_width=True)
            if st.button("🗑 Clear saved scenarios"):
                st.session_state.saved_scenarios = []
                st.rerun()

with tab3:
    st.subheader("Where Changes Happen")
    if use_heat_priority:
        st.info(
        "🌡️ **Heat vulnerability mode active** — conversions concentrated in higher-intensity "
        "developed areas. Notice the spatial pattern shift vs. random allocation."
        )

    render_matplotlib(plot_spatial_map(results['scenario_lulc'], cooling_lulc))
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary."
    )