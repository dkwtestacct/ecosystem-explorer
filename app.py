import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio
from skimage.transform import resize

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR_FLOOD   = 'data/flood'
DATA_DIR_COOLING = 'data/cooling'

BASELINE_CN = 75.7   # Mean curve number for current Minneapolis land cover
BASELINE_HM = 0.2719 # Mean heat mitigation index for current Minneapolis land cover

# Land cover codes (NLCD)
DEVELOPED_CODES   = [21, 22, 23]  # Low/medium intensity developed, developed open space
CODE_GREEN_INFRA  = 90            # Woody wetlands
CODE_FOOD_FOREST  = 41            # Deciduous forest
CODE_HIGH_DENSITY = 24            # High intensity developed
NODATA            = -128

# Colors for the spatial map
CHANGE_COLORS = {
    'Unchanged':            '#d3d3d3',
    'Green Infrastructure': '#2196a0',
    'Food Forest':          '#4caf50',
    'High Density':         '#e53935',
}

# Named reference scenarios shown in the tradeoff plot
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
    "changes **flood risk** and **urban cooling** across the city.  \n"
    "_Prototype using Minneapolis, MN data. San Antonio scenarios coming soon._"
)

# ── Session state ──────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

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

    # CN lookup: {lucode: {soil_group: curve_number}}
    cn_by_soil = {
        row['lucode']: {1: row['CN_A'], 2: row['CN_B'], 3: row['CN_C'], 4: row['CN_D']}
        for _, row in bio.iterrows()
    }

    # Resize soil raster to match land cover raster dimensions
    soil_resized = resize(soil, lulc.shape, order=0, preserve_range=True).astype(int)

    # Heat mitigation proxy: average of shade fraction and crop coefficient
    cooling_bio['HM'] = (cooling_bio['shade'] + cooling_bio['kc']) / 2
    hm_lookup = dict(zip(cooling_bio['lucode'], cooling_bio['HM']))

    return lulc, soil_resized, cooling_lulc, cn_by_soil, hm_lookup


lulc, soil_resized, cooling_lulc, cn_by_soil, hm_lookup = load_data()


def get_cn(lucode, soil_group):
    """Return curve number for a given land cover code and soil group."""
    return cn_by_soil.get(lucode, {}).get(soil_group, 0)

get_cn_vec = np.vectorize(get_cn)


# ── Scenario evaluation ────────────────────────────────────────────────────────
def evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct, seed=42):
    """
    Convert a random sample of developed pixels to the specified land use mix,
    then compute flood risk (curve number) and urban cooling (heat mitigation index).
    Returns results dict including the modified land cover array for spatial mapping.
    """
    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

    scenario_lulc = cooling_lulc.copy()
    developed_pixels = np.argwhere(np.isin(cooling_lulc, DEVELOPED_CODES))
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

    cn_scenario = get_cn_vec(scenario_lulc, soil_resized)
    mean_cn = float(cn_scenario[cn_scenario > 0].mean().round(2))

    hm_map = np.vectorize(hm_lookup.get)(scenario_lulc, np.nan)
    valid_hm = hm_map[~np.isnan(hm_map) & (scenario_lulc != NODATA)]
    mean_hm = float(valid_hm.mean().round(4))

    return {
        'pct_converted':            pct_converted,
        'green_infrastructure_pct': green_infrastructure_pct,
        'food_forest_pct':          food_forest_pct,
        'pct_highdensity':          pct_highdensity,
        'mean_cn':                  mean_cn,
        'flood_reduction':          round(100 - mean_cn, 2),
        'mean_hm':                  mean_hm,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
    }


# ── Pre-compute scenario grid for tradeoff background ─────────────────────────
@st.cache_data
def compute_scenario_grid():
    # Exclude scenario_lulc from cached grid (too large to store for all scenarios)
    rows = [
        {k: v for k, v in evaluate_scenario(pct, gi, ff, seed=42).items() if k != 'scenario_lulc'}
        for pct in range(0, 51, 10)
        for gi  in range(0, 101, 25)
        for ff  in range(0, 101, 25)
        if gi + ff <= 100
    ]
    return pd.DataFrame(rows)


def compute_pareto(df):
    """Return Pareto-efficient rows (maximize both flood_reduction and mean_hm)."""
    points = df[['flood_reduction', 'mean_hm']].values
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
    scenario_df = compute_scenario_grid()


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_bars(results):
    """Bar charts comparing this scenario to baseline for flood risk and cooling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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

    plt.tight_layout()
    return fig


def plot_tradeoff(results, scenario_df, saved=None):
    """
    Scatter plot showing where this scenario sits in the flood/cooling tradeoff space.
    Overlays saved scenarios and Pareto frontier if any scenarios are saved.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Background: full scenario grid
    ax.scatter(scenario_df['flood_reduction'], scenario_df['mean_hm'],
               alpha=0.2, color='lightgray', s=30, zorder=1)

    # Named reference scenarios
    for name, ref in REF_SCENARIOS.items():
        ax.scatter(ref['flood'], ref['cooling'], color=ref['color'], s=100, zorder=5)
        ax.annotate(name, (ref['flood'], ref['cooling']),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Saved scenarios + Pareto frontier
    if saved and len(saved) > 0:
        df_saved = pd.DataFrame(saved)
        ax.scatter(df_saved['flood_reduction'], df_saved['mean_hm'],
                   color='purple', alpha=0.5, s=60, zorder=4, label='Saved')

        pareto_df = compute_pareto(df_saved)
        ax.scatter(pareto_df['flood_reduction'], pareto_df['mean_hm'],
                   color='gold', s=120, edgecolor='black', zorder=5, label='Pareto optimal')
        pareto_sorted = pareto_df.sort_values('flood_reduction')
        ax.plot(pareto_sorted['flood_reduction'], pareto_sorted['mean_hm'],
                color='gold', linestyle='--', linewidth=1)

        for _, row in pareto_df.iterrows():
            ax.annotate(row['scenario_name'], (row['flood_reduction'], row['mean_hm']),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.7)

    # Current scenario
    ax.scatter(results['flood_reduction'], results['mean_hm'],
               color='purple', s=250, zorder=6, marker='*', label='This scenario')
    ax.axvline(results['flood_reduction'], linestyle=':', alpha=0.3, color='purple')
    ax.axhline(results['mean_hm'], linestyle=':', alpha=0.3, color='purple')
    ax.annotate('This Scenario', (results['flood_reduction'], results['mean_hm']),
                textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')

    ax.set_xlabel('Flood Risk Reduction (higher = better)')
    ax.set_ylabel('Heat Mitigation Index (higher = better)')
    ax.set_title('Tradeoff Space')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig


def plot_spatial_map(scenario_lulc, baseline_lulc):
    """
    Show which pixels changed relative to baseline.
    Unchanged pixels are gray; converted pixels are colored by their new land use.
    """
    h, w = scenario_lulc.shape
    rgb = np.full((h, w, 3), mcolors.to_rgb(CHANGE_COLORS['Unchanged']))

    changed_to_gi = (baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_GREEN_INFRA)
    changed_to_ff = (baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_FOOD_FOREST)
    changed_to_hd = (baseline_lulc != scenario_lulc) & (scenario_lulc == CODE_HIGH_DENSITY)
    nodata_mask   = baseline_lulc == NODATA

    rgb[changed_to_gi] = mcolors.to_rgb(CHANGE_COLORS['Green Infrastructure'])
    rgb[changed_to_ff] = mcolors.to_rgb(CHANGE_COLORS['Food Forest'])
    rgb[changed_to_hd] = mcolors.to_rgb(CHANGE_COLORS['High Density'])
    rgb[nodata_mask]   = (1.0, 1.0, 1.0)  # white outside city boundary

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.axis('off')
    ax.set_title('Land Use Changes', fontsize=12)

    legend_elements = [
        Patch(facecolor=CHANGE_COLORS['Unchanged'],            label='Unchanged'),
        Patch(facecolor=CHANGE_COLORS['Green Infrastructure'], label='→ Green Infrastructure'),
        Patch(facecolor=CHANGE_COLORS['Food Forest'],          label='→ Food Forest'),
        Patch(facecolor=CHANGE_COLORS['High Density'],         label='→ High Density'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")
pct_converted            = st.sidebar.slider("% of developed land converted", 0, 50, 10, 5)
green_infrastructure_pct = st.sidebar.slider("% → Green Infrastructure (wetlands)", 0, 100, 0, 5)
food_forest_pct          = st.sidebar.slider("% → Food Forest (trees)", 0, 100, 0, 5)
pct_highdensity          = 100 - green_infrastructure_pct - food_forest_pct

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
    
if green_infrastructure_pct + food_forest_pct > 100:
    st.sidebar.error("⚠️ Green Infrastructure + Food Forest exceeds 100%")
    st.stop()

st.sidebar.markdown(f"**→ High Density: {pct_highdensity}%** (remainder)")
st.sidebar.divider()
st.sidebar.caption(
    "**Green Infrastructure** = woody wetlands — best for flood retention.  \n"
    "**Food Forest** = deciduous trees — best for cooling.  \n"
    "**High Density** = paved development — worst for both."
)

# ── Main panel ─────────────────────────────────────────────────────────────────
results = evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct)

col1, col2, col3 = st.columns(3)
col1.metric("Flood Risk Reduction", f"{results['flood_reduction']:.1f}",
            delta=f"{results['flood_reduction'] - (100 - BASELINE_CN):.1f} vs baseline",
            delta_color="normal")
col2.metric("Urban Cooling (HM Index)", f"{results['mean_hm']:.4f}",
            delta=f"{results['mean_hm'] - BASELINE_HM:.4f} vs baseline")
col3.metric("Land Converted", f"{pct_converted}%",
            f"GI {green_infrastructure_pct}% · FF {food_forest_pct}% · HD {pct_highdensity}%")

st.caption("Higher is better for both metrics. Baseline = current Minneapolis land cover.")
st.divider()

# Charts on the left, spatial map on the right
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Outcome Comparison")
    bars_fig = plot_bars(results)
    st.pyplot(bars_fig, use_container_width=True)
    plt.close(bars_fig)

    st.subheader("Tradeoff Space")
    tradeoff_fig = plot_tradeoff(results, scenario_df, saved=st.session_state.saved_scenarios)
    st.pyplot(tradeoff_fig, use_container_width=True)
    plt.close(tradeoff_fig)

with right_col:
    st.subheader("Where Changes Happen")
    map_fig = plot_spatial_map(results['scenario_lulc'], cooling_lulc)
    st.pyplot(map_fig, use_container_width=True)
    plt.close(map_fig)

if st.button("💾 Save this scenario"):
    # Exclude the raster array from session state — too large
    saved = {k: v for k, v in results.items() if k != 'scenario_lulc'}
    st.session_state.saved_scenarios.append(saved)
    st.success(f"Saved: {results['scenario_name']}")

# ── Saved scenarios ────────────────────────────────────────────────────────────
if st.session_state.saved_scenarios:
    st.divider()
    st.subheader("Saved Scenarios")
    df_saved = pd.DataFrame(st.session_state.saved_scenarios)
    st.dataframe(
        df_saved[['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                  'food_forest_pct', 'flood_reduction', 'mean_hm']],
        use_container_width=True
    )
    if st.button("🗑 Clear saved scenarios"):
        st.session_state.saved_scenarios = []
        st.rerun()