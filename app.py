import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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

CHANGE_COLORS = {
    'Unchanged':            '#d3d3d3',
    'Green Infrastructure': '#2196a0',
    'Food Forest':          '#4caf50',
    'High Density':         '#e53935',
}

REF_SCENARIOS = {
    'Baseline':             {'flood': 24.3, 'cooling': 0.2719, 'color': 'steelblue'},
    'Food Forest':          {'flood': 29.9, 'cooling': 0.8284, 'color': 'green'},
    'Green Infrastructure': {'flood': 83.0, 'cooling': 0.8633, 'color': 'teal'},
    'High Density':         {'flood': 18.8, 'cooling': 0.1923, 'color': 'red'},
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

# ── Session state ──────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []
if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = None

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

    # Pre-compute developed pixel locations (used in every scenario evaluation)
    developed_pixels = np.argwhere(np.isin(cooling_lulc, DEVELOPED_CODES))

    # ── CN lookup: fast array indexed by [lucode_idx, soil_group] ──────────────
    cn_by_soil = {
        row['lucode']: {1: row['CN_A'], 2: row['CN_B'], 3: row['CN_C'], 4: row['CN_D']}
        for _, row in bio.iterrows()
    }
    all_lucodes = sorted(cn_by_soil.keys())
    lucode_to_idx = {lc: i for i, lc in enumerate(all_lucodes)}
    cn_table = np.zeros((len(all_lucodes), 5), dtype=np.float32)
    for lc, soils in cn_by_soil.items():
        for sg, cn_val in soils.items():
            cn_table[lucode_to_idx[lc], sg] = cn_val

    # Fast lucode→index array (avoids dict lookup per pixel)
    max_lucode = max(lucode_to_idx.keys())
    lucode_idx_arr = np.zeros(max_lucode + 1, dtype=np.int32)
    for lc, idx in lucode_to_idx.items():
        lucode_idx_arr[lc] = idx

    soil_resized = resize(soil, lulc.shape, order=0, preserve_range=True).astype(int)

    # ── HM lookup: fast array indexed by lucode ────────────────────────────────
    cooling_bio['HM'] = (cooling_bio['shade'] + cooling_bio['kc']) / 2
    hm_lookup = dict(zip(cooling_bio['lucode'], cooling_bio['HM']))
    max_hm_lucode = max(hm_lookup.keys())
    hm_arr = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    for lc, hm in hm_lookup.items():
        hm_arr[lc] = hm

    return (lulc, soil_resized, cooling_lulc, developed_pixels,
            cn_table, lucode_idx_arr, hm_arr, max_lucode, max_hm_lucode)


(lulc, soil_resized, cooling_lulc, developed_pixels,
 cn_table, lucode_idx_arr, hm_arr, max_lucode, max_hm_lucode) = load_data()


# ── Scenario evaluation ────────────────────────────────────────────────────────
def evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct, seed=42):
    """
    Convert a random sample of developed pixels to the specified land use mix,
    then compute flood risk, urban cooling, and food production.
    All raster lookups use fast numpy array indexing — no Python loops per pixel.
    """
    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

    scenario_lulc = cooling_lulc.copy()
    n_convert = int(len(developed_pixels) * pct_converted / 100)

    rng = np.random.default_rng(seed)
    pixels_to_convert = developed_pixels[
        rng.choice(len(developed_pixels), size=n_convert, replace=False)
    ]

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

    # Flood risk: fast CN lookup via pre-built arrays
    lulc_idx = lucode_idx_arr[np.clip(scenario_lulc, 0, max_lucode)]
    soil_clamped = np.clip(soil_resized, 1, 4)
    cn_scenario = cn_table[lulc_idx, soil_clamped]
    mean_cn = float(cn_scenario[cn_scenario > 0].mean().round(2))

    # Cooling: fast HM lookup via pre-built array
    hm_map = hm_arr[np.clip(scenario_lulc, 0, max_hm_lucode)].astype(float)
    hm_map[scenario_lulc > max_hm_lucode] = np.nan
    valid_hm = hm_map[~np.isnan(hm_map) & (scenario_lulc != NODATA)]
    mean_hm = float(valid_hm.mean().round(4))

    # Food production: count food forest pixels
    n_food_pixels = int((scenario_lulc == CODE_FOOD_FOREST).sum())
    food_mln_lbs = round(n_food_pixels * PIXEL_AREA_ACRES * FOOD_FOREST_LBS_ACRE / 1_000_000, 3)

    return {
        'pct_converted':            pct_converted,
        'green_infrastructure_pct': green_infrastructure_pct,
        'food_forest_pct':          food_forest_pct,
        'pct_highdensity':          pct_highdensity,
        'mean_cn':                  mean_cn,
        'flood_reduction':          round(100 - mean_cn, 2),
        'mean_hm':                  mean_hm,
        'food_mln_lbs':             food_mln_lbs,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
@st.cache_data
def compute_scenario_grid():
    rows = [
        {k: v for k, v in evaluate_scenario(pct, gi, ff, seed=42).items() if k != 'scenario_lulc'}
        for pct in range(0, 51, 10)
        for gi  in range(0, 101, 25)
        for ff  in range(0, 101, 25)
        if gi + ff <= 100
    ]
    return pd.DataFrame(rows)


@st.cache_data
def compute_lookup_table():
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
    scenario_df  = compute_scenario_grid()
    lookup_table = compute_lookup_table()

MAX_FOOD  = float(scenario_df['food_mln_lbs'].max())
MAX_FLOOD = 100.0
MAX_COOL  = 1.1


# ── Surrogate model ────────────────────────────────────────────────────────────
@st.cache_resource
def train_surrogate(_scenario_df):
    """Train a Random Forest surrogate on the pre-computed scenario grid."""
    X = _scenario_df[['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']]
    y = _scenario_df[['flood_reduction', 'mean_hm', 'food_mln_lbs']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


surrogate = train_surrogate(scenario_df)


def optimize_scenario(surrogate, min_flood, min_cool, min_food, n_samples=10000):
    """Use the surrogate to find Pareto-optimal scenarios meeting the given constraints."""
    rng = np.random.default_rng(42)
    pct_converted = rng.integers(0, 51, n_samples)
    gi_pct        = rng.integers(0, 101, n_samples)
    ff_pct        = rng.integers(0, 101, n_samples)

    valid = gi_pct + ff_pct <= 100
    pct_converted, gi_pct, ff_pct = pct_converted[valid], gi_pct[valid], ff_pct[valid]

    preds = surrogate.predict(np.column_stack([pct_converted, gi_pct, ff_pct]))
    meets = (preds[:, 0] >= min_flood) & (preds[:, 1] >= min_cool) & (preds[:, 2] >= min_food)
    if not meets.any():
        return None

    candidates = pd.DataFrame({
        'pct_converted':            pct_converted[meets],
        'green_infrastructure_pct': gi_pct[meets],
        'food_forest_pct':          ff_pct[meets],
        'flood_reduction':          preds[meets, 0].round(1),
        'mean_hm':                  preds[meets, 1].round(4),
        'food_mln_lbs':             preds[meets, 2].round(3),
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
    return pareto.sort_values('score', ascending=False).head(5).drop(columns='score')


# ── Plotting ───────────────────────────────────────────────────────────────────
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


def plot_tradeoff(results, scenario_df, saved=None, optimized=None):
    """Flood vs cooling scatter plot. Bubble size encodes food production."""
    fig, ax = plt.subplots(figsize=(9, 6))
    max_food = scenario_df['food_mln_lbs'].max()

    def food_to_size(food_vals, base=20, scale=300):
        if max_food > 0:
            return base + scale * (food_vals / max_food)
        return base

    ax.scatter(scenario_df['flood_reduction'], scenario_df['mean_hm'],
               alpha=0.2, color='lightgray',
               s=food_to_size(scenario_df['food_mln_lbs']), zorder=1)

    for name, ref in REF_SCENARIOS.items():
        ax.scatter(ref['flood'], ref['cooling'], color=ref['color'], s=100, zorder=5)
        ax.annotate(name, (ref['flood'], ref['cooling']),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    if saved:
        df_saved = pd.DataFrame(saved)
        ax.scatter(df_saved['flood_reduction'], df_saved['mean_hm'],
                   color='purple', alpha=0.5,
                   s=food_to_size(df_saved['food_mln_lbs']), zorder=4, label='Saved')
        pareto_df = compute_pareto(df_saved)
        ax.scatter(pareto_df['flood_reduction'], pareto_df['mean_hm'],
                   color='gold', s=120, edgecolor='black', zorder=5, label='Pareto optimal')
        pareto_sorted = pareto_df.sort_values('flood_reduction')
        ax.plot(pareto_sorted['flood_reduction'], pareto_sorted['mean_hm'],
                color='gold', linestyle='--', linewidth=1)
        for _, row in df_saved.iterrows():
            ax.annotate(row['scenario_name'], (row['flood_reduction'], row['mean_hm']),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.7)

    if optimized is not None and len(optimized) > 0:
        ax.scatter(optimized['flood_reduction'], optimized['mean_hm'],
                   color='orange', s=120, edgecolor='black', zorder=6,
                   marker='D', label='Optimized suggestions')

    current_size = float(food_to_size(pd.Series([results['food_mln_lbs']]))[0])
    ax.scatter(results['flood_reduction'], results['mean_hm'],
               color='purple', s=max(current_size, 150), zorder=7,
               marker='*', label='This scenario')
    ax.axvline(results['flood_reduction'], linestyle=':', alpha=0.3, color='purple')
    ax.axhline(results['mean_hm'], linestyle=':', alpha=0.3, color='purple')
    ax.annotate('This Scenario', (results['flood_reduction'], results['mean_hm']),
                textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')

    if max_food > 0:
        for fv, lbl in [(0, '0M lbs'), (max_food * 0.5, f'{max_food*0.5:.1f}M'), (max_food, f'{max_food:.1f}M')]:
            ax.scatter([], [], color='gray', alpha=0.4,
                       s=food_to_size(pd.Series([fv]))[0], label=f'Food: {lbl}')

    ax.set_xlabel('Flood Risk Reduction (higher = better)')
    ax.set_ylabel('Heat Mitigation Index (higher = better)')
    ax.set_title('Tradeoff Space  —  bubble size = food production')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    return fig


def plot_spatial_map(scenario_lulc, baseline_lulc):
    """Show which pixels changed and what they changed to."""
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")
pct_converted            = st.sidebar.slider("% of developed land converted", 0, 50, 10, 5)
green_infrastructure_pct = st.sidebar.slider("% → Green Infrastructure (wetlands)", 0, 100, 0, 5)
food_forest_pct          = st.sidebar.slider("% → Food Forest (trees)", 0, 100, 0, 5)
pct_highdensity          = 100 - green_infrastructure_pct - food_forest_pct

if green_infrastructure_pct + food_forest_pct > 100:
    st.sidebar.error("⚠️ Green Infrastructure + Food Forest exceeds 100%")
    st.stop()

st.sidebar.markdown(f"**→ High Density: {pct_highdensity}%** (remainder)")
st.sidebar.divider()

st.sidebar.subheader("Example Scenarios")
if st.sidebar.button("🌳 Tree Planting (Cooling Focus)"):
    pct_converted = 10
    green_infrastructure_pct = 0
    food_forest_pct = 100
if st.sidebar.button("🌊 Flood Mitigation (Wetlands)"):
    pct_converted = 10
    green_infrastructure_pct = 100
    food_forest_pct = 0
if st.sidebar.button("🏙 High Density Development"):
    pct_converted = 10
    green_infrastructure_pct = 0
    food_forest_pct = 0

st.sidebar.divider()
st.sidebar.subheader("🔍 Find Best Scenario")
st.sidebar.caption("Set minimum targets and let the surrogate model find optimal inputs.")
min_flood = st.sidebar.slider("Min flood reduction", 0, 90, 30, 5)
min_cool  = st.sidebar.slider("Min cooling (HM)", 0.0, 1.0, 0.3, 0.05)
min_food  = st.sidebar.slider("Min food production (M lbs)", 0.0, float(max(MAX_FOOD, 0.1)), 0.0, 0.01)

if st.sidebar.button("Optimize"):
    with st.spinner("Searching for optimal scenarios..."):
        st.session_state.optimized_results = optimize_scenario(
            surrogate, min_flood, min_cool, min_food)

st.sidebar.divider()
st.sidebar.caption(
    "**Green Infrastructure** = woody wetlands — best for flood retention.  \n"
    "**Food Forest** = deciduous trees — best for cooling + food.  \n"
    "**High Density** = paved development — worst for all three."
)

# ── Main panel ─────────────────────────────────────────────────────────────────
lookup_key = (pct_converted, green_infrastructure_pct, food_forest_pct)
if lookup_key in lookup_table:
    results = lookup_table[lookup_key].copy()
    results['scenario_lulc'] = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct
    )['scenario_lulc']
else:
    results = evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Flood Risk Reduction", f"{results['flood_reduction']:.1f}",
            delta=f"{results['flood_reduction'] - (100 - BASELINE_CN):.1f} vs baseline",
            delta_color="normal")
col2.metric("Urban Cooling (HM)", f"{results['mean_hm']:.4f}",
            delta=f"{results['mean_hm'] - BASELINE_HM:.4f} vs baseline")
col3.metric("Food Production", f"{results['food_mln_lbs']:.3f}M lbs/yr",
            delta=f"+{results['food_mln_lbs']:.3f}M vs baseline")
col4.metric("Land Converted", f"{pct_converted}%",
            f"GI {green_infrastructure_pct}% · FF {food_forest_pct}% · HD {pct_highdensity}%")

st.caption(
    "Higher is better for all three service metrics. "
    "Food yield is estimated from San Antonio NatCap benchmarks — treat as directional, not precise."
)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Scenario", "🔀 Tradeoff Analysis", "🗺️ Map View"])

with tab1:
    st.subheader("Outcome Comparison")
    bars_fig = plot_bars(results)
    st.pyplot(bars_fig, use_container_width=True)
    plt.close(bars_fig)

with tab2:
    st.subheader("Tradeoff Space")
    tradeoff_fig = plot_tradeoff(
        results, scenario_df,
        saved=st.session_state.saved_scenarios,
        optimized=st.session_state.optimized_results
    )
    st.pyplot(tradeoff_fig, use_container_width=True)
    plt.close(tradeoff_fig)

    if st.button("💾 Save this scenario"):
        saved = {k: v for k, v in results.items() if k != 'scenario_lulc'}
        st.session_state.saved_scenarios.append(saved)
        st.success(f"Saved: {results['scenario_name']}")

    # Optimizer results
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
                "Surrogate model predictions — verify with the sliders above."
            )
            st.dataframe(
                opt[['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                     'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs']],
                use_container_width=True
            )
            best = opt.iloc[0]
            st.info(
                f"**Best balanced scenario:** Convert {int(best.pct_converted)}% of developed land — "
                f"{int(best.green_infrastructure_pct)}% Green Infrastructure, "
                f"{int(best.food_forest_pct)}% Food Forest, "
                f"{int(best.pct_highdensity)}% High Density.  \n"
                f"Predicted: flood reduction **{best.flood_reduction:.1f}** · "
                f"cooling **{best.mean_hm:.4f}** · "
                f"food **{best.food_mln_lbs:.3f}M lbs/yr**"
            )

    # Saved scenarios in expander
    if st.session_state.saved_scenarios:
        st.divider()
        with st.expander(f"📋 Saved Scenarios ({len(st.session_state.saved_scenarios)})", expanded=False):
            df_saved = pd.DataFrame(st.session_state.saved_scenarios)
            st.dataframe(
                df_saved[['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                          'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs']],
                use_container_width=True
            )
            if st.button("🗑 Clear saved scenarios"):
                st.session_state.saved_scenarios = []
                st.rerun()

with tab3:
    st.subheader("Where Changes Happen")
    map_fig = plot_spatial_map(results['scenario_lulc'], cooling_lulc)
    st.pyplot(map_fig, use_container_width=True)
    plt.close(map_fig)
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary."
    )