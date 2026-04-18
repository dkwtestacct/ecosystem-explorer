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

if "pct_converted" not in st.session_state:
    st.session_state.pct_converted = 10
if "green_infrastructure_pct" not in st.session_state:
    st.session_state.green_infrastructure_pct = 0
if "food_forest_pct" not in st.session_state:
    st.session_state.food_forest_pct = 0

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
    lucode_to_idx = {lc: i + 1 for i, lc in enumerate(all_lucodes)}  # reserve row 0 for unknown

    cn_table = np.zeros((len(all_lucodes) + 1, 5), dtype=np.float32)  # row 0 = unknown
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

    return (lulc, soil_resized, cooling_lulc, developed_pixels,
        cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode)


(lulc, soil_resized, cooling_lulc, developed_pixels,
 cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode) = load_data()


# ── Scenario evaluation ────────────────────────────────────────────────────────
def evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct, seed=42):
    """
    Convert a random sample of developed pixels to the specified land use mix,
    then compute flood risk, urban cooling, and food production.
    All raster lookups use fast numpy array indexing.
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

    soil_clamped = np.clip(soil_resized, 1, 4)
    lulc_safe = np.clip(scenario_lulc, 0, len(lucode_idx_arr) - 1)
    lulc_idx = lucode_idx_arr[lulc_safe]
    cn_scenario = cn_table[lulc_idx, soil_clamped]

    mean_cn = float(cn_scenario[cn_scenario > 0].mean().round(2))

    hm_safe = np.clip(scenario_lulc, 0, len(hm_arr) - 1)
    hm_map = hm_arr[hm_safe].astype(float)
    hm_map[(scenario_lulc < 0) | (scenario_lulc >= len(hm_arr))] = np.nan
    valid_hm = hm_map[~np.isnan(hm_map) & (scenario_lulc != NODATA)]

    mean_hm = float(valid_hm.mean().round(4))

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

# ── Surrogate model ────────────────────────────────────────────────────────────
@st.cache_resource
def train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling):
    X = _scenario_df[['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']]
    y = _scenario_df[['flood_reduction', 'mean_hm', 'food_mln_lbs']]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


surrogate = train_surrogate(scenario_df, DATA_DIR_FLOOD, DATA_DIR_COOLING)


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


# ── Plotting helpers ───────────────────────────────────────────────────────────
def render_matplotlib(fig):
    """Render a matplotlib figure in Streamlit and always close it to prevent memory leaks."""
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


# ── Plotly tradeoff plot ───────────────────────────────────────────────────────
def food_to_size(food_vals, max_food, base=8, scale=90):
    food_vals = np.atleast_1d(np.asarray(food_vals, dtype=float))
    if max_food > 0:
        return base + scale * np.sqrt(food_vals / max_food)
    return np.full(len(food_vals), base)


def convex_hull_trace(scenario_df):
    """Return a Plotly Scatter trace of the convex hull of the scenario grid."""
    from scipy.spatial import ConvexHull
    points = scenario_df[['flood_reduction', 'mean_hm']].values
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

    # Use denser lookup table points for hull if available
    hull_source = pd.DataFrame(lookup_table.values()) if lookup_table else scenario_df
    hull_trace = convex_hull_trace(hull_source)
    if hull_trace:
        fig.add_trace(hull_trace)

    # Named reference scenarios (hardcoded benchmarks)
    for name, ref in REF_SCENARIOS.items():
        fig.add_trace(go.Scatter(
            x=[ref['flood']], y=[ref['cooling']],
            mode='markers+text',
            marker=dict(size=13, color=ref['color'], opacity=0.85,
                        line=dict(color='white', width=1)),
            text=[name], textposition='top right',
            textfont=dict(size=10),
            hovertemplate=(
                f"<b>{name}</b> (reference benchmark)<br>"
                f"Flood reduction: {ref['flood']} | Cooling HM: {ref['cooling']:.4f}"
                "<extra></extra>"
            ),
            name=name,
        ))

    # Saved scenarios
    if saved:
        df_saved = pd.DataFrame(saved)
        sizes = food_to_size(df_saved['food_mln_lbs'].values, max_food)
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

    # Optimized suggestions
    if optimized is not None and len(optimized) > 0:
        opt_sizes = food_to_size(optimized['food_mln_lbs'].values, max_food)
        fig.add_trace(go.Scatter(
            x=optimized['flood_reduction'],
            y=optimized['mean_hm'],
            mode='markers',
            marker=dict(size=opt_sizes, color='orange', symbol='diamond',
                        line=dict(color='black', width=1.5)),
            text=optimized.apply(
                lambda r: (
                    f"<b>Optimized suggestion</b><br>{r.scenario_name}<br>"
                    f"Flood: {r.flood_reduction:.1f} | Cooling: {r.mean_hm:.4f} | "
                    f"Food: {r.food_mln_lbs:.3f}M lbs"
                ), axis=1),
            hoverinfo='text',
            name='Optimized suggestions',
        ))

    # Current scenario (always on top)
    current_size = float(food_to_size(np.array([results['food_mln_lbs']]), max_food)[0])
    fig.add_trace(go.Scatter(
        x=[results['flood_reduction']],
        y=[results['mean_hm']],
        mode='markers+text',
        marker=dict(size=max(current_size, 18), color='purple', symbol='star',
                    line=dict(color='white', width=1.5)),
        text=['This Scenario'],
        textposition='top right',
        textfont=dict(size=11, color='purple'),
        hovertemplate=(
            f"<b>This Scenario</b><br>"
            f"Flood reduction: {results['flood_reduction']:.1f}<br>"
            f"Cooling HM: {results['mean_hm']:.4f}<br>"
            f"Food: {results['food_mln_lbs']:.3f}M lbs/yr"
            "<extra></extra>"
        ),
        name='This scenario',
    ))

    fig.add_hline(y=results['mean_hm'], line_dash='dot', line_color='purple', opacity=0.25)
    fig.add_vline(x=results['flood_reduction'], line_dash='dot', line_color='purple', opacity=0.25)

    fig.update_layout(
        title='Tradeoff Space  —  bubble size = food production (larger = more food)',
        xaxis_title='Flood Risk Reduction (higher = better)',
        yaxis_title='Heat Mitigation Index (higher = better)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 1.1]),
        height=520,
        legend=dict(
            orientation='v', x=1.02, y=1,
            bordercolor='rgba(0,0,0,0.1)', borderwidth=1,
        ),
        hovermode='closest',
    )

    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")
st.sidebar.slider("% of developed land converted", 0, 50, 10, 5, key="pct_converted")
st.sidebar.slider("% → Green Infrastructure (wetlands)", 0, 100, 0, 5, key="green_infrastructure_pct")
st.sidebar.slider("% → Food Forest (trees)", 0, 100, 0, 5, key="food_forest_pct")

pct_converted = st.session_state.pct_converted
green_infrastructure_pct = st.session_state.green_infrastructure_pct
food_forest_pct = st.session_state.food_forest_pct

pct_highdensity          = 100 - green_infrastructure_pct - food_forest_pct

if green_infrastructure_pct + food_forest_pct > 100:
    st.sidebar.error("⚠️ Green Infrastructure + Food Forest exceeds 100%")
    st.stop()

if pct_highdensity > 0:
    st.sidebar.info(
        f"**→ High Density: {pct_highdensity}%** of converted land will become high-density "
        f"development — the remainder after Green Infrastructure and Food Forest allocations."
    )
else:
    st.sidebar.success("✅ 100% of converted land allocated to green uses.")
    
st.sidebar.divider()

if st.sidebar.button("🌳 Downtown Tree Corridors"):
    st.session_state.pct_converted = 10
    st.session_state.green_infrastructure_pct = 0
    st.session_state.food_forest_pct = 100

if st.sidebar.button("🌊 Flood Mitigation (Wetlands)"):
    st.session_state.pct_converted = 10
    st.session_state.green_infrastructure_pct = 100
    st.session_state.food_forest_pct = 0

if st.sidebar.button("🏙 High Density Development"):
    st.session_state.pct_converted = 10
    st.session_state.green_infrastructure_pct = 0
    st.session_state.food_forest_pct = 0

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
    render_matplotlib(plot_spatial_map(results['scenario_lulc'], cooling_lulc))
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary."
    )
