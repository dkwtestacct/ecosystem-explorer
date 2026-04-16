import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from skimage.transform import resize

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR_FLOOD   = 'data/flood'
DATA_DIR_COOLING = 'data/cooling'

BASELINE_CN = 75.7   # Mean curve number for current Minneapolis land cover
BASELINE_HM = 0.2719 # Mean heat mitigation index for current Minneapolis land cover

# Land cover codes (NLCD)
DEVELOPED_CODES      = [21, 22, 23]  # Low, medium, developed open space
CODE_GREEN_INFRA     = 90            # Woody wetlands
CODE_FOOD_FOREST     = 41            # Deciduous forest
CODE_HIGH_DENSITY    = 24            # High intensity developed
NODATA               = -128

# Reference scenarios for tradeoff plot (pre-computed)
REF_SCENARIOS = {
    'Baseline':            {'flood': 24.3, 'cooling': 0.2719, 'color': 'steelblue'},
    'Food Forest':         {'flood': 29.9, 'cooling': 0.8284, 'color': 'green'},
    'Green Infrastructure':{'flood': 83.0, 'cooling': 0.8633, 'color': 'teal'},
    'High Density':        {'flood': 18.8, 'cooling': 0.1923, 'color': 'red'},
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ecosystem Explorer", layout="wide")
st.title("🌿 Ecosystem Service Tradeoff Explorer — Minneapolis")
st.caption("Explore how land use changes affect flood risk and urban cooling.")

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

    # HM (heat mitigation) proxy: average of shade and crop coefficient
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
    Apply land use conversions to developed pixels and compute ecosystem service scores.
    Returns a dict with flood risk (mean_cn, flood_reduction) and cooling (mean_hm).
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
    mean_cn = cn_scenario[cn_scenario > 0].mean().round(2)

    hm_map = np.vectorize(hm_lookup.get)(scenario_lulc, np.nan)
    valid_hm = hm_map[~np.isnan(hm_map) & (scenario_lulc != NODATA)]
    mean_hm = valid_hm.mean().round(4)

    return {
        'pct_converted':          pct_converted,
        'green_infrastructure_pct': green_infrastructure_pct,
        'food_forest_pct':        food_forest_pct,
        'pct_highdensity':        pct_highdensity,
        'mean_cn':                mean_cn,
        'flood_reduction':        round(100 - mean_cn, 2),
        'mean_hm':                mean_hm,
    }


# ── Pre-compute scenario grid for tradeoff background ─────────────────────────
@st.cache_data
def compute_scenario_grid():
    rows = [
        evaluate_scenario(pct, gi, ff, seed=42)
        for pct in range(0, 51, 10)
        for gi  in range(0, 101, 25)
        for ff  in range(0, 101, 25)
        if gi + ff <= 100
    ]
    return pd.DataFrame(rows)


with st.spinner("Loading data and pre-computing scenarios..."):
    scenario_df = compute_scenario_grid()


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(results, scenario_df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    # Flood risk bar
    ax1.bar(['Baseline', 'This Scenario'], [BASELINE_CN, results['mean_cn']],
            color=['steelblue', 'purple'])
    ax1.axhline(BASELINE_CN, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean Curve Number (lower = less flood risk)')
    ax1.set_title(f'Flood Risk\nCN = {results["mean_cn"]}')
    ax1.set_ylim(0, 100)

    # Cooling bar
    ax2.bar(['Baseline', 'This Scenario'], [BASELINE_HM, results['mean_hm']],
            color=['steelblue', 'purple'])
    ax2.axhline(BASELINE_HM, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Heat Mitigation Index (higher = more cooling)')
    ax2.set_title(f'Urban Cooling\nHM = {results["mean_hm"]}')
    ax2.set_ylim(0, 1.1)

    # Tradeoff space
    ax3.scatter(scenario_df['flood_reduction'], scenario_df['mean_hm'],
                alpha=0.3, color='lightgray', s=30, zorder=1)
    for name, ref in REF_SCENARIOS.items():
        ax3.scatter(ref['flood'], ref['cooling'], color=ref['color'], s=100, zorder=5)
        ax3.annotate(name, (ref['flood'], ref['cooling']),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax3.scatter(results['flood_reduction'], results['mean_hm'],
                color='purple', s=200, zorder=6, marker='*')
    ax3.annotate('This Scenario', (results['flood_reduction'], results['mean_hm']),
                 textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight='bold')
    ax3.set_xlabel('Flood Risk Reduction (higher = better)')
    ax3.set_ylabel('Heat Mitigation Index (higher = better)')
    ax3.set_title('Tradeoff Space')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")
pct_converted          = st.sidebar.slider("% of developed land converted", 0, 50, 10, 5)
green_infrastructure_pct = st.sidebar.slider("% → Green Infrastructure (wetlands)", 0, 100, 0, 5)
food_forest_pct        = st.sidebar.slider("% → Food Forest (trees)", 0, 100, 0, 5)
pct_highdensity        = 100 - green_infrastructure_pct - food_forest_pct

if green_infrastructure_pct + food_forest_pct > 100:
    st.sidebar.error("⚠️ Green Infrastructure + Food Forest exceeds 100%")
    st.stop()

st.sidebar.markdown(f"**→ High Density: {pct_highdensity}%** (remainder)")

# ── Main panel ─────────────────────────────────────────────────────────────────
results = evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct)

col1, col2, col3 = st.columns(3)
col1.metric("Flood Risk (Curve Number)", results['mean_cn'],
            delta=f"{results['mean_cn'] - BASELINE_CN:.1f} vs baseline",
            delta_color="inverse")
col2.metric("Urban Cooling (HM Index)", results['mean_hm'],
            delta=f"{results['mean_hm'] - BASELINE_HM:.4f} vs baseline")
col3.metric("% Land Converted", f"{pct_converted}%",
            f"GI:{green_infrastructure_pct}% FF:{food_forest_pct}% HD:{pct_highdensity}%")

fig = plot_results(results, scenario_df)
st.pyplot(fig)
plt.close()