import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import plotly.graph_objects as go
import os
from pathlib import Path

import rasterio
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor

# ── City configuration ─────────────────────────────────────────────────────────
CITIES = {
    'Minneapolis, MN': {
        'data_dir_flood':       'data/flood',
        'data_dir_cooling':     'data/cooling',
        'cn_table_file':        'UFR_biophysical_table_MN.csv',
        'cooling_table_file':   'biophysical_table_urban_cooling_MN.csv',
        'baseline_cn':          75.7,
        # 0.1859 = mean(smoothed CC) on the MN baseline LULC after the InVEST
        # UCM rework (ET nodata fix, Gaussian convolution, canonical formula).
        # Auto-recomputed at module load from `_BASELINE_HM_RASTER`, so this
        # value is only a documentation placeholder.
        'baseline_hm':          0.1859,
        'pixel_area_acres':     0.222,
        'food_forest_lbs_acre': 11_500,
        'available':            True,
        'crs':                  'EPSG:26915',
        # Reference points plotted on the tradeoff scatter. Recomputed via
        # `verify_cooling.py` (seed=42) after the InVEST UCM rework: ET
        # nodata fix, Gaussian convolution at 450 m, canonical energy
        # formula, UHI_MAX_C = 2.05 °C. Each "All X" scenario is
        # pct_converted=50 with 100 % allocation to that single land cover.
        'ref_scenarios': {
            'Baseline':                     {'flood': 24.3,  'cooling': 0.1859, 'color': 'steelblue'},
            'All Food Forest (NLCD 41)':    {'flood': 26.7,  'cooling': 0.3910, 'color': 'green'},
            'All Green Infra (NLCD 90)':    {'flood': 49.3,  'cooling': 0.3981, 'color': 'teal'},
            'All High Density (NLCD 24)':   {'flood': 20.5,  'cooling': 0.1549, 'color': 'red'},
        },
    },
    'San Antonio, TX': {
        'data_dir_flood':       'data/sa/flood',
        'data_dir_cooling':     'data/sa/cooling',
        'cn_table_file':        'UFR_biophysical_table_SA.csv',
        'cooling_table_file':   'biophysical_table_urban_cooling_SA.csv',
        # Preliminary values from download_sa_data.py: CN uses CN_B as default
        # soil group (pending SSURGO), HM uses 0.6*shade + 0.2*albedo + 0.2*kc
        # proxy (pending reference-ET).
        'baseline_cn':          65.97,
        'baseline_hm':          0.2917,
        'pixel_area_acres':     0.2224,  # NLCD 30 m in EPSG:5070
        'food_forest_lbs_acre': None,    # TODO: use crop-specific SA yields from project report
        'available':            False,   # set True when soil/population/ET inputs are ready
        'crs':                  'EPSG:5070',
        'notes': (
            'Data source: NatCap SA Urban Agriculture Project 2023. '
            'LULC from NLCD 2021. Baseline constants are preliminary — '
            'see data/sa/README.md.'
        ),
        'ref_scenarios': {},
    },
}

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
# Urban heat-island anomaly for Minneapolis: maximum temperature difference
# between fully-paved and rural-reference pixels. Source: InVEST UCM args JSON
# for the MN AOI (`uhi_max=2.05`, `t_ref=23.2`). One CC unit therefore maps to
# UHI_MAX_C in °C, or UHI_MAX_C × 1.8 in °F. When SA's InVEST args are
# retrieved, this should become a per-city `city_cfg['uhi_max_c']`.
UHI_MAX_C             = 2.05
HM_TO_FAHRENHEIT      = UHI_MAX_C * 1.8   # = 3.69 °F per CC unit (MN-calibrated)
# Food: average American consumes ~2,000 lbs of food per year
LBS_PER_PERSON_YEAR   = 2_000

CHANGE_COLORS = {
    'Unchanged':            '#d3d3d3',
    'Green Infrastructure': '#2196a0',
    'Food Forest':          '#4caf50',
    'High Density':         '#e53935',
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ecosystem Explorer", layout="wide")

st.markdown('''
<style>
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #5b8db8;
    border-color: #5b8db8;
    color: white;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #4a7aa6;
    border-color: #4a7aa6;
    color: white;
}
</style>
''', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []
if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = None
if "active_example_scenario" not in st.session_state:
    st.session_state.active_example_scenario = None
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
    # active_example_scenario is set by the button handler before _pending_ keys are written

# ── City selection ─────────────────────────────────────────────────────────────
_city_names  = list(CITIES.keys())
_city_labels = [
    name if CITIES[name]['available'] else f"{name} (coming soon)"
    for name in _city_names
]
_selected_label = st.sidebar.selectbox("City", _city_labels, index=0)
selected_city   = _city_names[_city_labels.index(_selected_label)]
city_cfg        = CITIES[selected_city]
st.sidebar.divider()

# ── City-aware header ──────────────────────────────────────────────────────────
st.title("🌿 Urban Ecosystem Tradeoff Explorer")
st.subheader(f"📍 {selected_city}")

if not city_cfg['available']:
    st.info(
        f"**{selected_city}** data is being prepared. Check back soon — "
        "or contact the team for early access."
    )
    if city_cfg.get('notes'):
        st.caption(city_cfg['notes'])
    st.sidebar.info(f"{selected_city} data coming soon — select Minneapolis, MN for live scenarios.")
    st.stop()

# Runtime constants derived from selected city — functions reference these as globals
DATA_DIR_FLOOD     = city_cfg['data_dir_flood']
DATA_DIR_COOLING   = city_cfg['data_dir_cooling']
CN_TABLE_FILE      = city_cfg['cn_table_file']
COOLING_TABLE_FILE = city_cfg['cooling_table_file']
BASELINE_CN        = city_cfg['baseline_cn']
BASELINE_HM        = city_cfg['baseline_hm']
REF_SCENARIOS      = city_cfg['ref_scenarios']

st.markdown(
    "Explore how converting developed land into green infrastructure or food forests "
    "affects **flood damage risk**, **urban cooling costs**, **food production**, "
    "**nature access**, and **carbon sequestration** across the city — translating "
    "ecological changes into concrete impacts for planners and decision-makers."
)
st.markdown(
    '- **Green Infrastructure (wetlands)** — best for flood  \n'
    '- **Food Forest** — best for cooling + food  \n'
    '- **High Density** — worst for all three  \n'
)

with st.expander("How this prototype works", expanded=False):
    st.markdown(
        "**Green Infrastructure** converts developed land to woody wetlands "
        "(NLCD code 90) — best for flood retention.  \n"
        "**Food Forest** converts to deciduous forest (NLCD code 41, used as a "
        "food production proxy) — best for cooling and food.  \n"
        "**High Density** adds impervious development — worst for all three.  \n"
        "  \n"
        "This is an exploratory tool — numbers are directional, not precise. "
        "Use them to compare strategies, not as final answers.  \n"
        "  \n"
        "Flood reduction is derived from curve number, cooling from a heat "
        "mitigation index, and food production from a food-forest yield "
        "benchmark — use these as comparative indicators.  \n"
        "Cooling °F is approximate (±2°F). Runoff uses a 2-inch design storm. "
        "Cost is order-of-magnitude — adjust $/acre sliders in sidebar."
    )

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(data_dir_flood, data_dir_cooling, cn_table_file, cooling_table_file):
    bio = pd.read_csv(f'{data_dir_flood}/{cn_table_file}')

    with rasterio.open(f'{data_dir_flood}/LULC_NLCD_2021_MN.tif') as src:
        lulc = src.read(1)
    with rasterio.open(f'{data_dir_flood}/soil_group_MN.tif') as src:
        soil = src.read(1)

    cooling_bio = pd.read_csv(f'{data_dir_cooling}/{cooling_table_file}')
    with rasterio.open(f'{data_dir_cooling}/land_use_2021.tif') as src:
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

    # Per-class shade / Kc / albedo arrays for the full InVEST UCM cooling
    # capacity formula: CC = 0.6·shade + 0.2·albedo + 0.2·ETI, where ETI is
    # built per pixel from the ET raster and Kc (see _compute_cc_raster).
    # We also keep a derived `hm_arr` (= the simplified `(shade + kc) / 2`)
    # for any legacy paths that reference it; the live CC pipeline supersedes
    # it everywhere that matters.
    max_hm_lucode = int(cooling_bio['lucode'].max())
    shade_arr  = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    kc_arr     = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    albedo_arr = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    for _, row in cooling_bio.iterrows():
        lc = int(row['lucode'])
        shade_arr[lc]  = row['shade']
        kc_arr[lc]     = row['kc']
        albedo_arr[lc] = row['albedo']
    hm_arr = (shade_arr + kc_arr) / 2  # legacy compatibility

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
            equity_weights, shade_arr, kc_arr, albedo_arr)


(lulc, soil_resized, cooling_lulc, developed_pixels,
 cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode,
 equity_weights, shade_arr, kc_arr, albedo_arr) = load_data(
    DATA_DIR_FLOOD, DATA_DIR_COOLING, CN_TABLE_FILE, COOLING_TABLE_FILE)

# ── Population raster (for Nature Access metric) ───────────────────────────────
# Built by download_census_pop.py from US Census 2020 block-level totals,
# rasterized to the NLCD grid. Falls back to a uniform placeholder if the file
# is missing so the app still launches before the pipeline has run.
def load_population_data(pop_path, target_shape):
    """Load a population-count raster, resampled to target_shape with bilinear."""
    with rasterio.open(pop_path) as src:
        data = src.read(
            1, out_shape=target_shape,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        data = data.astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = 0
        data[data < 0] = 0
        return data


try:
    pop_count_raster = load_population_data(
        "data/population/minneapolis_pop_2020.tif",
        cooling_lulc.shape,
    )
    POPULATION_DATA_AVAILABLE = True
except (FileNotFoundError, rasterio.errors.RasterioIOError):
    pop_count_raster = np.ones(cooling_lulc.shape, dtype=np.float32)
    POPULATION_DATA_AVAILABLE = False


# ── Urban Cooling Model: ET raster + energy savings ──────────────────────────
# Full InVEST UCM cooling-capacity formula:
#     ETI_pixel = (kc[class] × ET_pixel) / max(ET)
#     CC_pixel  = 0.6 × shade[class] + 0.2 × albedo[class] + 0.2 × ETI_pixel
# `mean_hm` reported by `evaluate_scenario` is now the mean CC over valid
# pixels (replacing the legacy `(shade + kc) / 2` lookup). Per-class shade,
# kc, and albedo come from the city's `cooling_table_file` (e.g.
# `biophysical_table_urban_cooling_MN.csv`); ET comes
# from `reference_evapotranspiration_annual.tif` (1 km, bilinear-resampled
# to the 30 m NLCD grid).
try:
    with rasterio.open(
        "data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/"
        "reference_evapotranspiration_annual.tif"
    ) as _et_src:
        _et_raw = _et_src.read(1).astype(float)
        _et_nodata = _et_src.nodata
    # The MN reference-ET raster uses 65535 as a nodata sentinel; np.isfinite
    # treats that as a valid float, so a previous version of this code
    # propagated the sentinel into MAX_ET_REF and zeroed out the ETI term in
    # the InVEST CC formula. Mask explicitly here, before resampling, so
    # bilinear interpolation doesn't bleed sentinel values onto valid pixels.
    if _et_nodata is not None:
        _et_raw[_et_raw == _et_nodata] = np.nan
    _et_raw[_et_raw > 10_000] = np.nan   # belt-and-braces against any other sentinels
    _et_raw[_et_raw < 0]      = np.nan
    ET_RESIZED = resize(_et_raw, cooling_lulc.shape, order=1, preserve_range=True)
    # Fill NaNs introduced by resize() interpolating across nodata pixels with
    # the field median so the convolution downstream sees a smooth ET surface.
    _finite = np.isfinite(ET_RESIZED)
    ET_RESIZED = np.where(_finite, ET_RESIZED, np.nanmedian(ET_RESIZED[_finite]))
    MAX_ET_REF = float(ET_RESIZED.max()) if ET_RESIZED.max() > 0 else 1.0
    ET_DATA_AVAILABLE = True
except Exception:
    ET_RESIZED = np.ones(cooling_lulc.shape, dtype=float)
    MAX_ET_REF = 1.0
    ET_DATA_AVAILABLE = False

# Energy consumption per building type (kWh/m²/year) from the InVEST UCM
# sample table. Used to translate cooling improvement into avoided AC cost.
try:
    _energy_df = pd.read_csv(
        "data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/energy_consumption.csv"
    )
    ENERGY_BY_TYPE = dict(zip(_energy_df["type"], _energy_df["consumption"]))
    ENERGY_TABLE_AVAILABLE = True
except Exception:
    ENERGY_BY_TYPE = {}
    ENERGY_TABLE_AVAILABLE = False

# Cost-per-kWh (US average residential, EIA 2024). Used to convert
# avoided-AC-kWh into $.
COST_PER_KWH_USD = 0.13

# NOTE: there used to be an `AC_KWH_PER_DEG_F = 0.03` fractional-AC-sensitivity
# constant here, applied as an extra multiplier in the energy-savings formula.
# It has been removed: the InVEST UCM `consumption` column is documented as
# kWh/(m²·°C), i.e. the per-degree response is already encoded in the rate.
# Multiplying by an additional 0.03 fraction would double-count. See
# `data/invest/cooling/UCM_AUDIT.md` for the full reasoning.


# InVEST UCM applies a Gaussian convolution to T_air_nomix with kernel radius
# = green_area_cooling_distance (450 m for MN, per the InVEST args JSON), to
# spatially propagate cooling from green pixels onto neighbours. We smooth CC
# directly — equivalent up to constants since T_air_nomix = t_ref + UHI×(1−CC)
# is an affine transform of CC and Gaussian convolution is linear.
GREEN_AREA_COOLING_DISTANCE_M = 450
_CC_SIGMA_PX = GREEN_AREA_COOLING_DISTANCE_M / 30  # 30 m NLCD pixels → σ = 15 px

from scipy.ndimage import gaussian_filter as _gaussian_filter


def _compute_cc_raster(scenario_lulc):
    """Per-pixel cooling-capacity index per InVEST UCM:
        CC_raw_i  = 0.6·shade_i + 0.2·albedo_i + 0.2·ETI_i
        ETI_i     = (kc_i × ET_ref_i) / max(ET_ref)
        CC_i      = gaussian_filter(CC_raw, σ = 450 m / 30 m px = 15 px)
    The Gaussian step approximates the InVEST T_air convolution: cooling
    benefits propagate ~450 m onto neighbouring pixels rather than staying
    pinned to the green pixel itself. NaN where the LULC code is outside the
    biophysical table; NaNs are temporarily filled with the in-AOI mean so
    they don't poison the convolution, then restored on the output."""
    safe = np.clip(scenario_lulc, 0, len(shade_arr) - 1)
    shade  = shade_arr[safe]
    kc     = kc_arr[safe]
    albedo = albedo_arr[safe]
    eti = (kc * ET_RESIZED) / MAX_ET_REF
    cc_raw = 0.6 * shade + 0.2 * albedo + 0.2 * eti
    nan_mask = (scenario_lulc < 0) | (scenario_lulc >= len(shade_arr)) | ~np.isfinite(cc_raw)

    # Fill NaNs with the valid-pixel mean before convolving, then restore. The
    # alternative — letting NaN propagate — would zero out a 15-px ring around
    # every nodata pixel and visibly bleed onto the AOI interior.
    if nan_mask.any():
        fill = float(np.nan_to_num(cc_raw[~nan_mask], nan=0.0).mean()) if (~nan_mask).any() else 0.0
        cc_filled = np.where(nan_mask, fill, cc_raw)
    else:
        cc_filled = cc_raw

    cc = _gaussian_filter(cc_filled.astype(np.float32), sigma=_CC_SIGMA_PX, mode="nearest")
    cc[nan_mask] = np.nan
    return cc


# Nature Access: weighted population-share metric using the official InVEST
# Urban Nature Access (UNA) biophysical table. Each LULC class has its own
# `urban_nature` score (0–1) and `search_radius_m`. For each scenario we take,
# per pixel, the maximum (in_range × score) across all natural classes — a
# pixel "near" multiple nature types gets the highest of their scores — then
# weight population by that score. Replaces the earlier hardcoded
# `NATURE_CODES = [41, 42, 43, 52, 71, 90, 95]` + single-800m-radius approach.
PIXEL_SIZE_M = 30

from scipy.ndimage import distance_transform_edt as _distance_transform_edt

UNA_TABLE_PATH = Path("data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv")
UNA_TABLE = pd.read_csv(UNA_TABLE_PATH)
# Active rows = classes that contribute to nature access (positive score AND
# a defined search radius). Sorted ascending by score so np.maximum in the
# loop below ends up with the highest score across all in-range classes.
UNA_ACTIVE = UNA_TABLE[
    (UNA_TABLE["urban_nature"] > 0) & UNA_TABLE["search_radius_m"].notna()
].copy()
UNA_ACTIVE["lucode"] = UNA_ACTIVE["lucode"].astype(int)

# Pre-compute distance transforms for natural classes whose pixel set never
# changes across scenarios (the model only converts NLCD 21–24 to GI/FF/HD).
# This keeps per-scenario evaluation fast: classes 21, 41, 90 are recomputed
# live; all other natural classes use the pre-built array.
_DYNAMIC_NATURE_LUCODES = {21, 41, 90}
PRECOMPUTED_NATURE_DISTANCES = {}
for _lucode in UNA_ACTIVE["lucode"]:
    if _lucode in _DYNAMIC_NATURE_LUCODES:
        continue
    _mask = (cooling_lulc == _lucode)
    if _mask.any():
        PRECOMPUTED_NATURE_DISTANCES[int(_lucode)] = (
            _distance_transform_edt(~_mask) * PIXEL_SIZE_M
        )


def _compute_access_score_raster(scenario_lulc):
    """0–1 nature-access score raster for the given scenario LULC.

    For each natural class with a positive `urban_nature` score in the InVEST
    UNA biophysical table, mask the scenario_lulc, compute distance-to-class,
    and combine via `np.maximum` (NOT sum) so a pixel near multiple natural
    classes takes the highest single class score — preventing double-counting.
    Pre-computed distance arrays are reused for natural classes whose pixel
    set never changes across scenarios.
    """
    access_score = np.zeros(scenario_lulc.shape, dtype=np.float32)
    for _, row in UNA_ACTIVE.iterrows():
        lucode = int(row["lucode"])
        radius = float(row["search_radius_m"])
        score  = float(row["urban_nature"])
        if lucode in PRECOMPUTED_NATURE_DISTANCES:
            distance = PRECOMPUTED_NATURE_DISTANCES[lucode]
        else:
            mask = (scenario_lulc == lucode)
            if not mask.any():
                continue
            distance = _distance_transform_edt(~mask) * PIXEL_SIZE_M
        in_range = distance <= radius
        np.maximum(access_score, in_range * score, out=access_score)
    return access_score


def calculate_nature_access(scenario_lulc, pop_count_raster):
    """
    Returns (access_pct, weighted_people_with_access).

    For each LULC class with a positive `urban_nature` score in the InVEST UNA
    table, compute the in-range mask using that class's `search_radius_m`.
    Per pixel, the access score is the maximum `score × in_range` across all
    contributing classes. The reported metric is the population-weighted mean
    of that score, expressed as a percentage. With a continuous 0–1 score
    (rather than the previous binary in-or-out flag) this generalizes the old
    "% of residents with access" framing — a pixel near a class-1.0 forest
    and a class-0.5 open-space patch counts toward the higher of the two.

    pop_count_raster must be per-pixel **counts** (not density).

    Returns `(access_pct, quality_score, weighted_people)`:

    - `access_pct` — share of residents with access_score above
      `NATURE_ACCESS_THRESHOLD` (default 0.3). A pedestrian-style headline
      number: "what fraction of residents reach *meaningful* nature?".
    - `quality_score` — population-weighted mean access score (0-1). Captures
      both proximity and nature class quality. The `np.maximum(...)` step
      inside `_compute_access_score_raster` is what prevents double-counting:
      a pixel near multiple natural classes gets the **highest** of their
      per-class scores, not the sum.
    - `weighted_people` — int of `pop × access_score` summed, useful as a
      population-scale companion number.
    """
    valid_pop = pop_count_raster > 0
    total_pop = pop_count_raster[valid_pop].sum()
    if total_pop <= 0:
        return 0.0, 0.0, 0

    access_score = _compute_access_score_raster(scenario_lulc)

    weighted_pop_score = (pop_count_raster * access_score)[valid_pop].sum()
    quality_score = float(weighted_pop_score / total_pop)

    above_threshold = access_score > NATURE_ACCESS_THRESHOLD
    pop_above_thresh = pop_count_raster[above_threshold & valid_pop].sum()
    access_pct = 100 * pop_above_thresh / total_pop

    return (
        round(float(access_pct), 1),
        round(quality_score, 3),
        int(weighted_pop_score),
    )


# Threshold for "Nature Access %" — pixels with access_score above this count
# as having meaningful nature access (binary headline metric). Below this and
# the resident still has *some* nature score, but it's reported via the
# continuous Nature Quality Score instead.
NATURE_ACCESS_THRESHOLD = 0.3


BASELINE_FOOD_MLN_LBS = 0.0
BASELINE_NATURE_ACCESS_PCT, BASELINE_NATURE_QUALITY_SCORE, _ = calculate_nature_access(
    cooling_lulc, pop_count_raster
)


# ── Urban Wellbeing Score ─────────────────────────────────────────────────────
# Composite index combining vegetation (NDVI), urban cooling (mean HM), and
# nature access %. Weights are *provisional* — see REFERENCE.md "Urban
# Wellbeing Score" section for the full caveat. Defaults sum to 1.0.
DEFAULT_WGT_NDVI    = 0.2
DEFAULT_WGT_COOLING = 0.4
DEFAULT_WGT_NATURE  = 0.4


def compute_wellbeing_score(ndvi, hm, nature_quality, w_ndvi, w_cooling, w_nature):
    """Composite wellbeing index. All inputs are 0–1 already:
    NDVI is 0–1, HM is treated as 0–1 (theoretical max = 1.0), and
    `nature_quality` is the population-weighted mean Nature Quality Score
    from `calculate_nature_access`. Returned value rounds to 3 decimals."""
    return round(float(
        w_ndvi   * ndvi
        + w_cooling * hm
        + w_nature * nature_quality
    ), 3)


# ── Metric translation helpers ─────────────────────────────────────────────────
# NDVI proxy: synthetic per-NLCD greenness values (0–1, higher = denser vegetation).
# Not derived from satellite imagery — assigned by land cover type as a placeholder
# until real NDVI rasters are integrated.
NDVI_PROXY = {
    90: 0.70,  # woody wetlands (green infrastructure)
    41: 0.75,  # deciduous forest (food forest proxy)
    24: 0.10,  # developed, high intensity
    23: 0.15,  # developed, medium intensity
    22: 0.20,  # developed, low intensity
    21: 0.30,  # developed, open space
}
NDVI_OTHER_DEVELOPED = 0.25  # any developed code not explicitly listed
NDVI_OTHER_NATURAL   = 0.60  # any non-developed natural cover
_DEVELOPED_ALL = {21, 22, 23, 24}

# Carbon sequestration: counts only converted pixels (consistent with food production).
# Sequestration rates in tons CO2e/acre/year (already converted from carbon to CO2e)
# To convert from tons C to tons CO2e: multiply by 3.667
# Sources: provisional regional USDA/IPCC values for temperate North America
# Food Forest (NLCD 41): 3.5 tons CO2e/acre/yr
# Green Infrastructure (NLCD 90): 2.0 tons CO2e/acre/yr
# These are order-of-magnitude estimates — replace with locally calibrated values
CARBON_SEQ_RATES = {
    CODE_FOOD_FOREST:  3.5,
    CODE_GREEN_INFRA:  2.0,
    CODE_HIGH_DENSITY: 0.0,
}


def compute_mean_ndvi(lulc_array):
    """Area-weighted mean NDVI proxy across all valid (non-NODATA) pixels."""
    valid = lulc_array[lulc_array != NODATA]
    if valid.size == 0:
        return float('nan')
    ndvi_map = np.full(valid.shape, NDVI_OTHER_NATURAL, dtype=np.float32)
    for code in _DEVELOPED_ALL:
        ndvi_map[valid == code] = NDVI_OTHER_DEVELOPED
    for code, val in NDVI_PROXY.items():
        ndvi_map[valid == code] = val
    return float(round(ndvi_map.mean(), 4))


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
                      cost_hd=DEFAULT_COST_HD,
                      carbon_rate_ff=None,
                      carbon_rate_gi=None,
                      w_ndvi=DEFAULT_WGT_NDVI,
                      w_cooling=DEFAULT_WGT_COOLING,
                      w_nature=DEFAULT_WGT_NATURE):
    """
    Convert a random (or equity-weighted) sample of developed pixels to the specified
    land use mix, then compute flood risk, urban cooling, food production, and cost.
    """
    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

    scenario_lulc = cooling_lulc.copy()
    # Sample from the convertible (= developed AND non-building) pool so
    # conversions land on feasible interstitial spaces (parking lots, lawns,
    # vacant land) rather than on top of existing structures. Total developed
    # acreage for runoff baseline scaling still uses the full developed_pixels
    # array — buildings still produce runoff.
    n_convert = int(len(CONVERTIBLE_PIXELS) * pct_converted / 100)

    rng = np.random.default_rng(seed)

    if use_heat_priority and n_convert > 0:
        # Pull weights specifically for the convertible pixels
        weights = equity_weights[CONVERTIBLE_PIXELS[:, 0], CONVERTIBLE_PIXELS[:, 1]].astype(float)

        # Robustness check: Ensure no negative or NaN weights
        weights = np.maximum(weights, 0)
        weight_sum = weights.sum()

        if weight_sum > 0:
            weights /= weight_sum
            chosen_idx = rng.choice(len(CONVERTIBLE_PIXELS), size=n_convert, replace=False, p=weights)
        else:
            chosen_idx = rng.choice(len(CONVERTIBLE_PIXELS), size=n_convert, replace=False)
    else:
        chosen_idx = rng.choice(len(CONVERTIBLE_PIXELS), size=n_convert, replace=False)

    pixels_to_convert = CONVERTIBLE_PIXELS[chosen_idx]

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

    # Full InVEST UCM cooling-capacity index (replaces the legacy
    # `(shade + kc) / 2` lookup). `mean_hm` is now the mean CC value across
    # valid pixels — same scale as before (0–1, higher = more cooling) but
    # now factors albedo and per-pixel ET in addition to shade and Kc.
    cc_map   = _compute_cc_raster(scenario_lulc)
    valid_hm = cc_map[~np.isnan(cc_map) & (scenario_lulc != NODATA)]
    mean_hm  = float(valid_hm.mean().round(4))
    cooling_energy_savings_usd = compute_cooling_energy_savings(cc_map)

    n_food_pixels = int(((scenario_lulc == CODE_FOOD_FOREST) & (cooling_lulc != CODE_FOOD_FOREST)).sum())
    food_mln_lbs  = round(n_food_pixels * PIXEL_AREA_ACRES * FOOD_FOREST_LBS_ACRE / 1_000_000, 3)

    rate_ff = CARBON_SEQ_RATES[CODE_FOOD_FOREST] if carbon_rate_ff is None else carbon_rate_ff
    rate_gi = CARBON_SEQ_RATES[CODE_GREEN_INFRA] if carbon_rate_gi is None else carbon_rate_gi
    carbon_tons_co2_yr = round(
        n_for * PIXEL_AREA_ACRES * rate_ff
        + n_wet * PIXEL_AREA_ACRES * rate_gi
        + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
    )

    nat_pct, nat_quality, nat_people = calculate_nature_access(
        scenario_lulc, pop_count_raster
    )

    mean_ndvi = compute_mean_ndvi(scenario_lulc)
    # Wellbeing uses the continuous Nature Quality Score (0–1) rather than the
    # threshold-based access percentage — quality captures graded proximity
    # better, which is what the composite is meant to summarize.
    wellbeing_score = compute_wellbeing_score(
        mean_ndvi, mean_hm, nat_quality, w_ndvi, w_cooling, w_nature
    )

    total_developed_acres = len(developed_pixels) * PIXEL_AREA_ACRES
    total_cost_mln = compute_cost(n_wet, n_for, n_hd, cost_gi, cost_ff, cost_hd)
    runoff_acft    = cn_to_runoff_acre_feet(mean_cn, total_developed_acres)
    flood_damage_avoided_usd = compute_flood_damage_avoided(runoff_acft)

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
        'runoff_acre_feet':         runoff_acft,
        'mean_hm':                  mean_hm,
        'cooling_f':                hm_to_fahrenheit_cooling(mean_hm),
        'flood_damage_avoided_usd': flood_damage_avoided_usd,
        'cooling_energy_savings_usd': cooling_energy_savings_usd,
        'mean_ndvi':                mean_ndvi,
        'carbon_tons_co2_yr':       carbon_tons_co2_yr,
        'nature_access_pct':        nat_pct,
        'nature_quality_score':     nat_quality,
        'people_with_nature_access': nat_people,
        'wellbeing_score':          wellbeing_score,
        'food_mln_lbs':             food_mln_lbs,
        'people_fed':               food_to_people_fed(food_mln_lbs),
        'total_cost_mln':           total_cost_mln,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
# Bump SCENARIO_SCHEMA_VERSION whenever the surrogate target columns change so
# Streamlit's @st.cache_data automatically invalidates stale grids/tables.
SCENARIO_SCHEMA_VERSION = 9   # bumped: ET nodata sentinel masked → ETI term now active in CC

# Surrogate target columns that downstream code (train_surrogate, optimize_scenario)
# requires. Listed explicitly so a missing column fails loudly instead of leaking
# into a KeyError deep in fit().
REQUIRED_TARGET_COLUMNS = [
    'flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
    'carbon_tons_co2_yr', 'nature_access_pct',
]


def _compute_carbon(n_wet, n_for, n_hd):
    """Carbon sequestration at default rates — used at scenario-grid build time."""
    return round(
        n_for * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_FOOD_FOREST]
        + n_wet * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_GREEN_INFRA]
        + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
    )


@st.cache_data
def compute_scenario_grid(data_dir_flood, data_dir_cooling,
                          step_pct=10, step_alloc=25,
                          schema_version=SCENARIO_SCHEMA_VERSION):
    rows = []
    for pct in range(0, 51, step_pct):
        for gi in range(0, 101, step_alloc):
            for ff in range(0, 101, step_alloc):
                if gi + ff <= 100:
                    result = evaluate_scenario(pct, gi, ff, seed=42)
                    row = {k: v for k, v in result.items() if k != 'scenario_lulc'}
                    # Explicit recomputation guarantees the surrogate-target
                    # columns exist regardless of evaluate_scenario's return.
                    row['carbon_tons_co2_yr'] = _compute_carbon(
                        row['n_wet'], row['n_for'], row['n_hd']
                    )
                    nature_access_pct, nature_quality_score, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc'], pop_count_raster
                    )
                    row['nature_access_pct'] = nature_access_pct
                    row['nature_quality_score'] = nature_quality_score
                    row['people_with_nature_access'] = people_with_nature_access
                    rows.append(row)
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"compute_scenario_grid is missing required columns {missing}; "
            f"check evaluate_scenario's return dict."
        )
    return df


@st.cache_data
def compute_lookup_table(data_dir_flood, data_dir_cooling, schema_version=SCENARIO_SCHEMA_VERSION):
    """Pre-compute results for every valid slider position (step=5) for instant response."""
    # 2,541 valid (pct, gi, ff) combinations × distance_transform_edt is slow,
    # so show a progress bar so the user knows the app hasn't hung.
    total = sum(
        1
        for pct in range(0, 51, 5)
        for gi in range(0, 101, 5)
        for ff in range(0, 101, 5)
        if gi + ff <= 100
    )
    progress_msg = st.empty()
    progress_msg.info(f"Pre-computing {total:,} scenarios (one-time, then cached)...")
    progress = st.progress(0)

    table = {}
    done = 0
    for pct in range(0, 51, 5):
        for gi in range(0, 101, 5):
            for ff in range(0, 101, 5):
                if gi + ff <= 100:
                    result = evaluate_scenario(pct, gi, ff, seed=42)
                    entry = {k: v for k, v in result.items() if k != 'scenario_lulc'}
                    entry['carbon_tons_co2_yr'] = _compute_carbon(
                        entry['n_wet'], entry['n_for'], entry['n_hd']
                    )
                    nature_access_pct, nature_quality_score, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc'], pop_count_raster
                    )
                    entry['nature_access_pct'] = nature_access_pct
                    entry['nature_quality_score'] = nature_quality_score
                    entry['people_with_nature_access'] = people_with_nature_access
                    missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in entry]
                    if missing:
                        raise RuntimeError(
                            f"compute_lookup_table entry missing columns {missing}; "
                            f"check evaluate_scenario's return dict."
                        )
                    table[(pct, gi, ff)] = entry
                    done += 1
                    if done % 50 == 0 or done == total:
                        progress.progress(done / total)

    progress.empty()
    progress_msg.empty()
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


DENSE_SCENARIOS_PATH = "data/scenarios_dense.csv"
# Read the model-quality selection from session_state. The radio that writes
# here lives in the Advanced Settings expander further down — Streamlit reruns
# top-to-bottom on every interaction, so on the next rerun this read picks up
# the new value the radio wrote on the previous run.
# ── Buildings: damage-loss data + footprint mask for siting constraints ──────
# Loads the InVEST UFR sample buildings shapefile + damage-loss table once. The
# shapefile drives both the "Flood Damage Avoided" $ metric (sum of footprint
# area × damage rate, scaled by scenario runoff reduction) AND the
# building-footprint constraint on pixel selection (conversions are placed only
# on developed pixels that DON'T contain a building, targeting feasible
# interstitial spaces — parking lots, lawns, vacant land). Falls back to
# disabled if the shapefile or CSV is missing so the rest of the app still loads.
try:
    import geopandas as _gpd  # noqa: F401  imported lazily so non-flood paths don't pay it
    from rasterio.features import rasterize as _rasterize

    _BUILDINGS_PATH    = Path("data/invest/flood/UFR_sample_data_MN/buildings.shp")
    _DAMAGE_TABLE_PATH = Path("data/invest/flood/UFR_sample_data_MN/Damage_loss_table_MN.csv")
    _buildings = _gpd.read_file(_BUILDINGS_PATH)
    _damage_table = pd.read_csv(_DAMAGE_TABLE_PATH)
    # `type` column in the buildings shapefile (0–3) maps to `Type` in the
    # damage-loss table. Values are damage in $/m² of building footprint.
    _type_to_damage = dict(zip(_damage_table["Type"], _damage_table["Damage"]))
    _buildings["damage_rate_usd_m2"] = _buildings["type"].map(_type_to_damage).fillna(0)
    _buildings["area_m2"] = _buildings.geometry.area
    _buildings["potential_damage_usd"] = (
        _buildings["area_m2"] * _buildings["damage_rate_usd_m2"]
    )
    TOTAL_POTENTIAL_DAMAGE_USD = float(_buildings["potential_damage_usd"].sum())

    # Rasterize building polygons onto the NLCD grid (CRS already matches —
    # both are EPSG:26915). BUILDINGS_RASTER is binary (1 = building, 0 = not);
    # BUILDINGS_TYPE_RASTER carries the building's `type` field (0–3) so the
    # cooling-energy-savings calculation can look up per-class kWh/m²/year.
    # Pixels outside any building → -1 in the type raster.
    with rasterio.open("data/cooling/land_use_2021.tif") as _ref:
        _ref_shape     = (_ref.height, _ref.width)
        _ref_transform = _ref.transform
        BUILDINGS_RASTER = _rasterize(
            ((geom, 1) for geom in _buildings.geometry),
            out_shape=_ref_shape,
            transform=_ref_transform,
            fill=0,
            dtype="uint8",
        )
        BUILDINGS_TYPE_RASTER = _rasterize(
            ((geom, int(t)) for geom, t in zip(_buildings.geometry, _buildings["type"])),
            out_shape=_ref_shape,
            transform=_ref_transform,
            fill=-1,
            dtype="int32",
        )
    BUILDINGS_DATA_AVAILABLE = True
except Exception:
    TOTAL_POTENTIAL_DAMAGE_USD = 0.0
    BUILDINGS_DATA_AVAILABLE = False
    BUILDINGS_RASTER = np.zeros(cooling_lulc.shape, dtype="uint8")
    BUILDINGS_TYPE_RASTER = np.full(cooling_lulc.shape, -1, dtype="int32")

# OSM road network (citywide) — unioned into BUILDINGS_RASTER so the
# convertible-pixel mask excludes both buildings AND streets. The InVEST
# UFR shapefile only covers a small downtown rectangle and includes only
# 277 road polygons; OSM gives the full street network. Built offline by
# `download_osm_roads.py` and committed to the repo.
try:
    _OSM_ROADS_PATH = Path("data/osm/minneapolis_roads.geojson")
    if _OSM_ROADS_PATH.exists():
        _roads_gdf = _gpd.read_file(_OSM_ROADS_PATH)
        if _roads_gdf.crs is None or str(_roads_gdf.crs) != "EPSG:26915":
            _roads_gdf = _roads_gdf.to_crs("EPSG:26915")
        with rasterio.open("data/cooling/land_use_2021.tif") as _ref:
            ROADS_RASTER = _rasterize(
                ((g, 1) for g in _roads_gdf.geometry),
                out_shape=(_ref.height, _ref.width),
                transform=_ref.transform,
                fill=0,
                dtype="uint8",
            )
        # Union into BUILDINGS_RASTER so any downstream code that consumes
        # "non-convertible developed land" (CONVERTIBLE_PIXELS, the spatial
        # map's heat overlay, the per-tract aggregation) sees roads as
        # already excluded — same semantics as buildings.
        BUILDINGS_RASTER = np.maximum(BUILDINGS_RASTER, ROADS_RASTER)
        OSM_ROADS_AVAILABLE = True
    else:
        ROADS_RASTER = np.zeros(cooling_lulc.shape, dtype="uint8")
        OSM_ROADS_AVAILABLE = False
except Exception:
    ROADS_RASTER = np.zeros(cooling_lulc.shape, dtype="uint8")
    OSM_ROADS_AVAILABLE = False

# Per-pixel AC-energy consumption rate (kWh/m²/year) for the cooling-energy
# savings calculation. Pixels not in any building → 0; building types not in
# the energy table → 0. Indexed via BUILDINGS_TYPE_RASTER at runtime.
PIXEL_AREA_M2 = 30 * 30  # NLCD 30 m grid → 900 m² per pixel
if ENERGY_BY_TYPE:
    _max_bldg_type = int(max(BUILDINGS_TYPE_RASTER.max(), max(ENERGY_BY_TYPE.keys())))
    _consumption_lookup = np.zeros(max(_max_bldg_type, 0) + 2, dtype=float)
    for _t, _c in ENERGY_BY_TYPE.items():
        if int(_t) >= 0:
            _consumption_lookup[int(_t)] = float(_c)
    _safe_type = np.clip(BUILDINGS_TYPE_RASTER, 0, len(_consumption_lookup) - 1)
    CONSUMPTION_RATE_PER_PIXEL = np.where(
        BUILDINGS_TYPE_RASTER >= 0,
        _consumption_lookup[_safe_type],
        0.0,
    )
else:
    CONSUMPTION_RATE_PER_PIXEL = np.zeros(cooling_lulc.shape, dtype=float)
    BUILDINGS_RASTER = np.zeros(cooling_lulc.shape, dtype="uint8")

# Convertible pixels = developed land that is NOT a building footprint. Random
# (or heat-weighted) sampling in evaluate_scenario draws from this pool, so
# conversions land on parking lots, lawns, and vacant land rather than on
# top of existing structures. The full developed_pixels array is still used
# for runoff baseline scaling (buildings still produce runoff).
_no_building = BUILDINGS_RASTER[developed_pixels[:, 0], developed_pixels[:, 1]] == 0
CONVERTIBLE_PIXELS = developed_pixels[_no_building]


# ── Census tracts (for neighborhood-level reporting) ──────────────────────────
# Loads the 27 Hennepin County tracts that intersect the model area, rasterizes
# them onto the NLCD grid (each pixel labeled with its tract index, or -1 for
# pixels outside any tract). Used to compute per-tract Nature Access % and
# Temperature Change against the live scenario.
try:
    _TRACTS_PATH = Path("data/invest/flood/UFR_sample_data_MN/admin_boundaries_census_tracts.shp")
    TRACTS = _gpd.read_file(_TRACTS_PATH)
    if TRACTS.crs is None or str(TRACTS.crs) != "EPSG:26915":
        TRACTS = TRACTS.to_crs("EPSG:26915")
    TRACTS = TRACTS.reset_index(drop=True)
    with rasterio.open("data/cooling/land_use_2021.tif") as _ref:
        TRACT_ID_RASTER = _rasterize(
            ((g, i) for i, g in enumerate(TRACTS.geometry)),
            out_shape=(_ref.height, _ref.width),
            transform=_ref.transform,
            fill=-1,
            dtype=np.int32,
        )
    TRACTS_DATA_AVAILABLE = True
except Exception:
    TRACTS = pd.DataFrame()
    TRACT_ID_RASTER = np.full(cooling_lulc.shape, -1, dtype=np.int32)
    TRACTS_DATA_AVAILABLE = False


def _compute_hm_raster(scenario_lulc):
    """Per-pixel HM raster — now uses the full InVEST UCM CC formula via
    `_compute_cc_raster`. Kept under the old `_compute_hm_raster` name so
    callers (per-tract reporting) don't need to change."""
    return _compute_cc_raster(scenario_lulc)


def compute_cooling_energy_savings(scenario_cc_raster):
    """Annual avoided AC cost ($/yr) for buildings under the active scenario,
    using the canonical InVEST UCM energy-valuation formula.

    Per pixel: `ΔT_°C = (CC_scenario − CC_baseline) × UHI_MAX_C`. The InVEST
    `consumption` column is documented as kWh/(m²·°C), so the per-pixel kWh
    saved is `consumption_rate × ΔT_°C × pixel_area_m²`, and the dollar value
    is multiplied by `$/kWh`. Negative ΔT (scenario hotter than baseline) is
    clamped to zero — we only credit cooling, not penalise warming. Sums over
    building pixels and returns $0 when buildings, the energy table, or the
    ET raster are unavailable.

    See `data/invest/cooling/UCM_AUDIT.md` for the divergence-from-canonical
    log: we still apply this per-pixel rather than per-building (no 600 m
    `t_air_average_radius` aggregation), but the per-pixel CC raster is now
    Gaussian-smoothed at 450 m before reaching this function.
    """
    if not (BUILDINGS_DATA_AVAILABLE and ENERGY_TABLE_AVAILABLE and ET_DATA_AVAILABLE):
        return 0.0
    delta_cc = scenario_cc_raster - _BASELINE_HM_RASTER
    delta_t_c = np.clip(delta_cc * UHI_MAX_C, 0.0, None)
    kwh_saved_per_pixel = (
        CONSUMPTION_RATE_PER_PIXEL * delta_t_c * PIXEL_AREA_M2
    )
    usd_per_pixel = kwh_saved_per_pixel * COST_PER_KWH_USD
    valid = (BUILDINGS_TYPE_RASTER >= 0) & np.isfinite(usd_per_pixel)
    return round(float(usd_per_pixel[valid].sum()), 0)


# Pre-compute baseline rasters once at startup so per-tract aggregates only
# need to recompute the scenario side on each rerun.
_BASELINE_ACCESS_SCORE_RASTER = _compute_access_score_raster(cooling_lulc)
_BASELINE_HM_RASTER          = _compute_hm_raster(cooling_lulc)

# Override the static `BASELINE_HM` from CITIES with the actual mean of the
# baseline CC raster — keeps the reference value in sync with whatever the
# current InVEST UCM pipeline produces. The CITIES value is now documentary.
_valid_base_cc = _BASELINE_HM_RASTER[~np.isnan(_BASELINE_HM_RASTER)]
if _valid_base_cc.size > 0:
    BASELINE_HM = float(_valid_base_cc.mean().round(4))


def compute_per_tract_summary(scenario_lulc):
    """DataFrame with one row per tract: baseline + scenario Nature Access %
    and °F change vs the global baseline, plus the difference (improvement)."""
    if not TRACTS_DATA_AVAILABLE or len(TRACTS) == 0:
        return pd.DataFrame()

    access_s_raster = _compute_access_score_raster(scenario_lulc)
    hm_s_raster     = _compute_hm_raster(scenario_lulc)

    above_b = _BASELINE_ACCESS_SCORE_RASTER > NATURE_ACCESS_THRESHOLD
    above_s = access_s_raster > NATURE_ACCESS_THRESHOLD

    rows = []
    for i in range(len(TRACTS)):
        mask = TRACT_ID_RASTER == i
        if not mask.any():
            continue
        pop_in_tract = pop_count_raster[mask].sum()
        if pop_in_tract <= 0:
            continue
        # Population-weighted Nature Access % within the tract
        b_share = pop_count_raster[mask & above_b].sum() / pop_in_tract
        s_share = pop_count_raster[mask & above_s].sum() / pop_in_tract
        # Temperature offset vs city baseline HM, in °F (positive = cooler)
        valid_hm = mask & ~np.isnan(_BASELINE_HM_RASTER) & ~np.isnan(hm_s_raster)
        if not valid_hm.any():
            continue
        b_hm = _BASELINE_HM_RASTER[valid_hm].mean()
        s_hm = hm_s_raster[valid_hm].mean()
        b_temp_f = (b_hm - BASELINE_HM) * HM_TO_FAHRENHEIT
        s_temp_f = (s_hm - BASELINE_HM) * HM_TO_FAHRENHEIT
        rows.append({
            "GEOID":               str(TRACTS.iloc[i].get("GEOID10", i)),
            "Population":          int(pop_in_tract),
            "Baseline Access %":   round(100 * b_share, 1),
            "Scenario Access %":   round(100 * s_share, 1),
            "Access Δ (pp)":       round(100 * (s_share - b_share), 1),
            "Baseline Temp (°F)":  round(b_temp_f, 2),
            "Scenario Temp (°F)":  round(s_temp_f, 2),
            "Temp Δ (°F cooler)":  round(s_temp_f - b_temp_f, 2),
        })
    return pd.DataFrame(rows)

# Baseline runoff for the damage scaling — computed inline here because
# `BASELINE_RUNOFF_ACRE_FEET` (the canonical module-level constant) isn't
# defined until after the lookup table is built. Same formula either way.
_BASELINE_RUNOFF_FOR_DAMAGE = cn_to_runoff_acre_feet(
    BASELINE_CN, len(developed_pixels) * PIXEL_AREA_ACRES
)


def compute_flood_damage_avoided(runoff_acre_feet):
    """Order-of-magnitude $ damage avoided vs baseline.

    Uses the simplification `avoided = total_potential_damage ×
    (runoff_reduction_fraction)`, where `runoff_reduction_fraction` is
    `max(0, baseline - scenario) / baseline`. Caps at zero — scenarios that
    INCREASE runoff are reported as $0 avoided rather than negative dollars
    (those regressions show up via the existing Runoff Volume card).
    """
    if not BUILDINGS_DATA_AVAILABLE or _BASELINE_RUNOFF_FOR_DAMAGE <= 0:
        return 0.0
    reduction = max(0.0, _BASELINE_RUNOFF_FOR_DAMAGE - runoff_acre_feet)
    fraction  = reduction / _BASELINE_RUNOFF_FOR_DAMAGE
    return round(TOTAL_POTENTIAL_DAMAGE_USD * fraction, 0)


MODEL_QUALITY_OPTIONS = ["Fast prototype", "Balanced", "High resolution"]
# Random Forest tree counts per mode — implementation detail, not exposed in UI.
SURROGATE_TREES = {
    "Fast prototype":  100,
    "Balanced":        200,
    "High resolution": 300,
}
_requested_model_quality = st.session_state.get("model_quality", MODEL_QUALITY_OPTIONS[0])
N_ESTIMATORS = SURROGATE_TREES[_requested_model_quality]

with st.spinner("Loading data and pre-computing scenarios..."):
    # The lookup table is built unconditionally — it powers instant slider
    # response throughout the app, and it doubles as the High-resolution
    # training set (no extra compute needed).
    lookup_table = compute_lookup_table(DATA_DIR_FLOOD, DATA_DIR_COOLING)

    if _requested_model_quality == "High resolution":
        scenario_df = pd.DataFrame(list(lookup_table.values()))
        ACTIVE_MODEL_QUALITY = "high"
    elif _requested_model_quality == "Balanced":
        if os.path.exists(DENSE_SCENARIOS_PATH):
            scenario_df = pd.read_csv(DENSE_SCENARIOS_PATH)
        else:
            st.warning(
                f"⚠️ `{DENSE_SCENARIOS_PATH}` not found — recomputing on the fly. "
                "Run `python3 precompute_scenarios.py` once to skip this on future startups."
            )
            scenario_df = compute_scenario_grid(
                DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=5, step_alloc=10
            )
        ACTIVE_MODEL_QUALITY = "balanced"
    else:  # Fast prototype
        scenario_df = compute_scenario_grid(
            DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=10, step_alloc=25
        )
        ACTIVE_MODEL_QUALITY = "fast"

MAX_FOOD  = float(scenario_df['food_mln_lbs'].max())
MAX_FLOOD = 100.0
MAX_COOL  = 1.1

BASELINE_RUNOFF_ACRE_FEET = cn_to_runoff_acre_feet(
    BASELINE_CN, len(developed_pixels) * PIXEL_AREA_ACRES
)

BASELINE_NDVI = compute_mean_ndvi(cooling_lulc)

# ── Surrogate model ────────────────────────────────────────────────────────────
@st.cache_resource
def train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling,
                    mode_key="fast", n_estimators=100):
    # mode_key + n_estimators participate in the cache key so changing the
    # Model quality mode radio in the sidebar automatically retrains on the
    # new training set without needing a manual cache clear.
    X = _scenario_df[['pct_converted', 'green_infrastructure_pct', 'food_forest_pct']]
    # Nature Access is included as a sixth output, but with an important caveat:
    # the surrogate maps (pct, gi%, ff%) → nature_access_pct, which discards the
    # spatial geometry that drives the metric. Random vs heat-priority placement,
    # and the location of converted pixels relative to existing parks and
    # population centers, all change the actual buffer overlap — but the
    # surrogate cannot see any of that. Treat surrogate predictions of
    # nature_access_pct as an indicative trend, not a precise spatial estimate.
    y = _scenario_df[['flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
                      'carbon_tons_co2_yr', 'nature_access_pct']]
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model


surrogate = train_surrogate(
    scenario_df, DATA_DIR_FLOOD, DATA_DIR_COOLING,
    mode_key=ACTIVE_MODEL_QUALITY, n_estimators=N_ESTIMATORS,
)


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
    """Bar chart of RF feature importances across all three output metrics."""
    feature_names = ['% Converted', 'Green Infra %', 'Food Forest %']
    metric_names  = ['Flood Reduction', 'Cooling (CC)', 'Food Production']
    
    # Each estimator in a MultiOutputRegressor-style RF predicts all outputs
    # feature_importances_ is averaged across all trees
    importances = model.feature_importances_  # shape (n_features,)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = ['#8e8e8e', '#2196a0', '#4caf50']
    bars = ax.barh(feature_names, importances, color=colors)
    ax.invert_yaxis()  # % Converted at top, matching sidebar control order
    ax.set_xlabel('Relative Importance', fontsize=9)
    ax.set_title('What drives outcomes most?', fontsize=10)
    ax.set_xlim(0, max(importances) * 1.3)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    return fig

def optimize_scenario(surrogate, min_flood, min_cool, min_food, max_runoff, min_carbon=0, n_samples=10000):
    """Use the surrogate to find efficient tradeoff scenarios meeting the given constraints."""
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
        pareto['flood_reduction'] / MAX_FLOOD +
        pareto['mean_hm'] / MAX_COOL +
        pareto['food_mln_lbs'] / (MAX_FOOD if MAX_FOOD > 0 else 1)
    )
    pareto = pareto.sort_values('score', ascending=False)
    
    # Drop near-duplicates in tradeoff space before returning
    pareto['flood_rounded'] = pareto['flood_reduction'].round(-1)
    pareto['hm_rounded']    = pareto['mean_hm'].round(1)
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
def plot_spatial_map(scenario_lulc, baseline_lulc,
                     heat_overlay=None, overlay_alpha=0.0,
                     tract_value=None, tract_alpha=0.0):
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

    legend_handles = [
        Patch(facecolor=CHANGE_COLORS['Unchanged'],            label='Unchanged'),
        Patch(facecolor=CHANGE_COLORS['Green Infrastructure'], label='→ Green Infrastructure'),
        Patch(facecolor=CHANGE_COLORS['Food Forest'],          label='→ Food Forest'),
        Patch(facecolor=CHANGE_COLORS['High Density'],         label='→ High Density'),
    ]

    # Optional red heat-vulnerability overlay. Per-pixel alpha = overlay_alpha
    # × heat_overlay value (which is 0–1), so low-vulnerability pixels stay
    # transparent and high-vulnerability ones tint red. With overlay_alpha=0
    # the overlay is fully invisible.
    if heat_overlay is not None and overlay_alpha > 0:
        overlay_rgba = np.zeros((h, w, 4))
        overlay_rgba[..., 0] = 1.0  # red channel
        overlay_rgba[..., 3] = overlay_alpha * np.clip(heat_overlay, 0.0, 1.0)
        ax.imshow(overlay_rgba)
        legend_handles.append(Patch(facecolor=(1, 0, 0, 0.6), label='Heat vulnerability'))

    # Optional tract-level improvement overlay. tract_value is a per-pixel
    # float raster (NaN outside any tract); colormap is RdYlGn so positive
    # improvements are green and regressions are red, centered at 0.
    if tract_value is not None and tract_alpha > 0:
        valid = ~np.isnan(tract_value)
        if valid.any():
            vmax = max(float(np.abs(tract_value[valid]).max()), 0.1)
            norm_val = np.zeros_like(tract_value, dtype=float)
            norm_val[valid] = (tract_value[valid] + vmax) / (2 * vmax)  # → 0..1
            cmap_rgba = plt.get_cmap("RdYlGn")(np.clip(norm_val, 0.0, 1.0))
            cmap_rgba[..., 3] = tract_alpha * valid.astype(float)
            ax.imshow(cmap_rgba)
            legend_handles.append(
                Patch(facecolor=(0.0, 0.6, 0.0, 0.6),
                      label="Neighborhood improvement (green = better)")
            )

    ax.axis('off')
    ax.set_title('Land Use Changes', fontsize=12)
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


# ── Plotly tradeoff plot ───────────────────────────────────────────────────────
def food_to_size(food_vals, max_food, base=5, scale=60):
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
            line=dict(color='rgba(180,180,180,0.25)', width=1.5, dash='dot'),
            fill='toself',
            fillcolor='rgba(200,200,200,0.04)',
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
        'Baseline':                   'bottom right',
        'All Food Forest (NLCD 41)':  'middle left',
        'All Green Infra (NLCD 90)':  'top left',
        'All High Density (NLCD 24)': None,
    }
    MARKER_OVERRIDES = {
        'Baseline': dict(size=16, color='steelblue', opacity=1.0,
                         line=dict(color='black', width=2)),
    }

    for name, ref in REF_SCENARIOS.items():
        text_pos = TEXT_POSITIONS.get(name, 'top right')
        m_override = MARKER_OVERRIDES.get(name, {})
        fig.add_trace(go.Scatter(
            x=[ref['flood']], y=[ref['cooling']],
            mode='markers+text' if text_pos else 'markers',
            marker=dict(
                size=m_override.get('size', 10),
                color=m_override.get('color', ref['color']),
                opacity=m_override.get('opacity', 0.6),
                line=m_override.get('line', dict(color='white', width=1)),
            ),
            text=[name] if text_pos else None,
            textposition=text_pos if text_pos else None,
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>{name}</b> (reference benchmark)<br>"
                f"Flood reduction: {ref['flood']} | Cooling CC: {ref['cooling']:.4f}"
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
                    # Prefer the user-given display_name; fall back to scenario_name
                    # for older saves that predate the named-scenarios feature.
                    f"{getattr(r, 'display_name', None) or r.scenario_name}<br>"
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
                    f"<b>Frontier scenario</b><br>{r.scenario_name}<br>"
                    f"Flood: {r.flood_reduction:.1f} | Cooling: {r.mean_hm:.4f}"
                ), axis=1),
            hoverinfo='text',
            name='Most efficient tradeoffs (saved)',
        ))

    if optimized is not None and len(optimized) > 0:
        opt_sizes = np.clip(food_to_size(optimized['food_mln_lbs'].values, max_food), 6, 18)
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
            f"Cooling CC: {results['mean_hm']:.4f}<br>"
            f"Food: {results['food_mln_lbs']:.3f}M lbs/yr<br>"
            f"Cost: ${results['total_cost_mln']:.1f}M"
            "<extra></extra>"
        ),
        name='This scenario',
    ))

    fig.add_hline(y=results['mean_hm'], line_dash='dot', line_color='purple', opacity=0.25)
    fig.add_vline(x=results['flood_reduction'], line_dash='dot', line_color='purple', opacity=0.25)

    fig.update_layout(
        title='Tradeoff Space',
        xaxis_title='Flood Risk Reduction (higher = better)',
        yaxis_title='Cooling Capacity (higher = better)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 0.6]),
        height=520,
        margin=dict(l=60, r=200, t=80, b=60),
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left', yanchor='top',
                    tracegroupgap=4, font=dict(size=11), itemsizing='constant',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        hovermode='closest',
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")

pct_converted = st.sidebar.slider(
    "% of developed land to convert", 0, 50,
    key="slider_pct_converted",
    help="Note: real conversions depend on land availability and existing uses — not all developed land is freely convertible."
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

mix_sum = green_infrastructure_pct + food_forest_pct + pct_highdensity

if mix_sum == 100:
    st.sidebar.success("Mix sums to 100%")
else:
    st.sidebar.error(f"Mix sums to {mix_sum}% — must equal 100%")
    st.stop()

st.sidebar.divider()

# ── Cost sliders ──────────────────────────────────────────────────────────────
st.sidebar.subheader("Implementation Costs ($/acre)")
cost_gi = st.sidebar.slider("Green Infrastructure ($/acre)", 5_000, 150_000,
                              DEFAULT_COST_GI, 5_000,
                              help="Typical range: $20,000–$100,000/acre for constructed wetlands. Default is an illustrative estimate — adjust to reflect local project costs.")
cost_ff = st.sidebar.slider("Food Forest ($/acre)", 1_000, 50_000,
                              DEFAULT_COST_FF, 1_000,
                              help="Typical range: $5,000–$20,000/acre for food forest establishment. Default is an illustrative estimate — adjust to reflect local project costs.")
cost_hd = st.sidebar.slider("High Density Infill ($/acre)", 1_000, 50_000,
                              DEFAULT_COST_HD, 1_000,
                              help="Marginal cost of additional impervious development. Default is an illustrative estimate — adjust to reflect local project costs.")

st.sidebar.divider()

# ── Spatial priority ──────────────────────────────────────────────────────────
st.sidebar.subheader("Heat-Weighted Conversion")
use_heat_priority = st.sidebar.toggle(
    "Target High Heat-Exposure Areas",
    value=False,
    help="When enabled, the model prioritizes converting land in areas with higher heat-exposure intensity."
)

st.sidebar.divider()

st.sidebar.subheader("Quick Start — Try a Scenario")
st.sidebar.caption("Click any button to load a preset scenario instantly.")

# Clear active example if the user has manually changed any slider away from its values
_EXAMPLE_VALUES = {
    'food_forest':  (10,  0, 100),
    'green_infra':  (10, 100,  0),
    'high_density': (10,  0,   0),
}
_active = st.session_state.active_example_scenario
if _active is not None:
    _exp_pct, _exp_gi, _exp_ff = _EXAMPLE_VALUES[_active]
    if (pct_converted != _exp_pct or
            green_infrastructure_pct != _exp_gi or
            food_forest_pct != _exp_ff):
        st.session_state.active_example_scenario = None
        _active = None

if st.sidebar.button("Food Forest (Cooling + Food Focus)",
                     type="primary" if _active == 'food_forest' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 100
    st.session_state.active_example_scenario = 'food_forest'
    st.rerun()

if st.sidebar.button("Green Infrastructure (Flood Mitigation)",
                     type="primary" if _active == 'green_infra' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 100
    st.session_state._pending_ff = 0
    st.session_state.active_example_scenario = 'green_infra'
    st.rerun()

if st.sidebar.button("High Density Development",
                     type="primary" if _active == 'high_density' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 0
    st.session_state.active_example_scenario = 'high_density'
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Find Best Scenario")

st.sidebar.caption(
    "Uses a surrogate model trained on ~90 full-resolution simulations to "
    "search ~10,000 candidate strategies in seconds. Results are approximate "
    "— verify promising scenarios using the main sliders."
)

st.sidebar.caption(
    "Optimization currently targets flood reduction, cooling, food production, and carbon "
    "sequestration. Cost and heat-priority placement are not yet included in the surrogate."
)

with st.sidebar.container(border=True):

    min_flood  = st.slider("Min flood reduction", 0, 90, 30, 5,
        help="Corresponds to the Flood Risk Reduction metric card. Baseline is 24.3. Higher values mean less runoff — increasing this target will also reduce Runoff Volume in ac-ft.")
    min_cool_f = st.slider(
        "Min cooling (°F vs baseline)",
        min_value=-1.0, max_value=round((1.0 - BASELINE_HM) * HM_TO_FAHRENHEIT, 1),
        value=0.1, step=0.1,
        help="Corresponds to the Temperature Change metric card. Set to 0.1 for at least 0.1°F cooler than baseline."
    )
    min_cool   = BASELINE_HM + min_cool_f / HM_TO_FAHRENHEIT   # HM units for surrogate
    min_food   = st.slider("Min food production (M lbs)", 0.0, float(max(MAX_FOOD, 0.1)), 0.0, 0.01,
        help="Corresponds directly to the Food Production metric card value in M lbs/yr.")
    _runoff_min = float(scenario_df['runoff_acre_feet'].min())
    _runoff_max = float(scenario_df['runoff_acre_feet'].max())
    max_runoff = st.slider(
        "Max allowable runoff (ac-ft)",
        min_value=round(_runoff_min),
        max_value=round(_runoff_max),
        value=round(BASELINE_RUNOFF_ACRE_FEET),
        step=100,
        help=f"Scenarios must stay below this runoff volume. Baseline is approximately {BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft."
    )
    min_carbon = st.slider(
        "Min carbon sequestration (tons CO2e/yr)",
        0, int(scenario_df['carbon_tons_co2_yr'].max()), 0, 100,
        help="Corresponds to the Carbon Sequestration metric card. Counts only converted pixels; baseline is 0."
    )

    st.caption(
        "The optimizer uses a surrogate model — a fast approximation trained on pre-computed "
        "scenarios — to search 10,000 candidate strategies in seconds. Results are approximate; "
        "verify promising scenarios using the main sliders."
    )
    st.sidebar.caption(
        "Slider results use a precomputed lookup table for instant response. "
        "The optimizer uses a separate surrogate model to search a much wider range of scenarios."
    )

    if st.button("Optimize"):
        with st.spinner("Searching for most efficient tradeoff scenarios..."):
            st.session_state.optimized_results = optimize_scenario(
                surrogate, min_flood, min_cool, min_food, max_runoff, min_carbon=min_carbon)
        _opt_res = st.session_state.optimized_results
        if _opt_res is None or (isinstance(_opt_res, dict) and not _opt_res.get('found')):
            st.sidebar.warning("No scenarios found — try lowering the targets.")
            st.session_state.just_optimized = False
        else:
            st.sidebar.success("Results ready — open the Tradeoff Analysis tab →")
            st.session_state.just_optimized = True

st.sidebar.divider()

with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    st.slider(
        "Food Forest carbon rate (tons CO2e/acre/yr)",
        0.5, 18.0, 3.5, 0.5,
        key="carbon_rate_ff",
        help="Provisional range 1.76–18.2 (USDA NRCS 2022). Default 3.5 is conservative for a mature system."
    )
    st.slider(
        "Green Infrastructure carbon rate (tons CO2e/acre/yr)",
        0.5, 5.0, 2.0, 0.5,
        key="carbon_rate_gi",
        help="Provisional range for woody wetlands. Default 2.0 tons CO2e/acre/yr."
    )
    st.caption(
        "These are provisional regional estimates. Adjust to reflect locally calibrated "
        "values or sensitivity test assumptions. See Methodology & Data Sources for "
        "sources and caveats."
    )

    st.divider()

    st.radio(
        "Model quality mode",
        options=MODEL_QUALITY_OPTIONS,
        index=0,
        key="model_quality",
        help=(
            "Controls how many full-resolution simulations are used to train the "
            "surrogate model. More simulations improve optimizer suggestions but "
            "take longer to initialize."
        ),
    )
    st.caption(
        "Fast prototype: ~90 training scenarios — quick startup, good for exploration.  \n"
        "Balanced: ~500 scenarios — better coverage, moderate startup time.  \n"
        "High resolution: trains on the full 2,541-entry lookup table — slower startup, better optimizer coverage."
    )
    st.caption(f"Active: {len(scenario_df):,} training scenarios.")

    st.divider()

    st.markdown("**Urban Wellbeing Score weights**")
    st.caption("Weights are provisional — adjust to reflect local research priorities.")
    wgt_ndvi    = st.slider("NDVI weight",         0.0, 1.0, DEFAULT_WGT_NDVI,    0.05, key="wgt_ndvi")
    wgt_cooling = st.slider("Cooling weight",      0.0, 1.0, DEFAULT_WGT_COOLING, 0.05, key="wgt_cooling")
    wgt_nature  = st.slider("Nature Access weight", 0.0, 1.0, DEFAULT_WGT_NATURE,  0.05, key="wgt_nature")
    st.caption(
        f"Current weights sum to {wgt_ndvi + wgt_cooling + wgt_nature:.2f} — "
        "ideally should sum to 1.0."
    )

# ── Main panel ─────────────────────────────────────────────────────────────────
lookup_key = (pct_converted, green_infrastructure_pct, food_forest_pct)
if lookup_key in lookup_table and not use_heat_priority:
    # Lookup table was computed without equity weighting — only use it in standard mode
    results = lookup_table[lookup_key].copy()
    _fresh = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
        use_heat_priority=False, cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd,
        carbon_rate_ff=st.session_state.carbon_rate_ff,
        carbon_rate_gi=st.session_state.carbon_rate_gi,
        w_ndvi=st.session_state.get("wgt_ndvi", DEFAULT_WGT_NDVI),
        w_cooling=st.session_state.get("wgt_cooling", DEFAULT_WGT_COOLING),
        w_nature=st.session_state.get("wgt_nature", DEFAULT_WGT_NATURE),
    )
    results['scenario_lulc'] = _fresh['scenario_lulc']
    # Food values are recomputed live — lookup table may predate the n_food_pixels fix
    results['food_mln_lbs'] = _fresh['food_mln_lbs']
    results['people_fed']   = _fresh['people_fed']
    results['mean_ndvi']    = _fresh['mean_ndvi']
    results['carbon_tons_co2_yr'] = _fresh['carbon_tons_co2_yr']
    results['nature_access_pct']  = _fresh['nature_access_pct']
    results['nature_quality_score'] = _fresh['nature_quality_score']
    results['people_with_nature_access'] = _fresh['people_with_nature_access']
    results['wellbeing_score']    = _fresh['wellbeing_score']
    results['flood_damage_avoided_usd'] = _fresh['flood_damage_avoided_usd']
    results['cooling_energy_savings_usd'] = _fresh['cooling_energy_savings_usd']
    # Recompute cost with current cost sliders (lookup table used default costs)
    results['total_cost_mln'] = compute_cost(
        results['n_wet'], results['n_for'], results['n_hd'],
        cost_gi, cost_ff, cost_hd
    )
else:
    results = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
        use_heat_priority=use_heat_priority, cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd,
        carbon_rate_ff=st.session_state.carbon_rate_ff,
        carbon_rate_gi=st.session_state.carbon_rate_gi,
        w_ndvi=st.session_state.get("wgt_ndvi", DEFAULT_WGT_NDVI),
        w_cooling=st.session_state.get("wgt_cooling", DEFAULT_WGT_COOLING),
        w_nature=st.session_state.get("wgt_nature", DEFAULT_WGT_NATURE),
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

_flood_delta = results['flood_reduction'] - (100 - BASELINE_CN)
_flood_delta_str = (
    "No change vs baseline" if abs(_flood_delta) < 0.1
    else f"+{_flood_delta:.1f} vs baseline" if _flood_delta > 0
    else f"-{abs(_flood_delta):.1f} vs baseline"
)
_cooling_f = results['cooling_f']
_cooling_label = (
    "No change" if abs(_cooling_f) < 0.1
    else f"{_cooling_f:.1f}°F cooler" if _cooling_f > 0
    else f"{abs(_cooling_f):.1f}°F warmer"
)
_hm_delta = results['mean_hm'] - BASELINE_HM
_runoff_prevented = BASELINE_RUNOFF_ACRE_FEET - results['runoff_acre_feet']
_runoff_negligible = abs(_runoff_prevented) < 1.0
_runoff_delta_str = (
    "No change vs base" if _runoff_negligible
    else f"+{_runoff_prevented:,.0f} ac-ft prevented" if _runoff_prevented > 0
    else f"-{abs(_runoff_prevented):,.0f} ac-ft above base"
)
_people_fed = results['people_fed']
_food_delta_str = f"feeds ~{_people_fed:,} people" if _people_fed > 0 else "—"

_carbon_value = results['carbon_tons_co2_yr']

def _fmt_carbon(tons):
    """Compact carbon display — k notation kicks in at 1,000 t to avoid card truncation."""
    if tons >= 1000:
        return f"{tons / 1000:.1f}k t CO2e/yr"
    return f"{tons:,.0f} t CO2e/yr"

_carbon_value_str = _fmt_carbon(_carbon_value)
_carbon_delta_str = (
    f"+{_fmt_carbon(_carbon_value)} vs base" if _carbon_value >= 1.0
    else "no change vs base"
)

st.markdown("#### Ecological")
eco1, eco2, eco3 = st.columns(3)
eco1.metric(
    "Flood Risk Reduction",
    f"{results['flood_reduction']:.1f}",
    delta=_flood_delta_str,
    delta_color="normal" if abs(_flood_delta) >= 0.1 else "off",
    help=(
        "Confidence: Raster-based calculation. "
        "Unitless index (0–100) based on the USDA Curve Number. Higher = less "
        "runoff potential. Baseline is 24.3 for Minneapolis developed land."
    )
)
eco2.metric(
    "Temperature Change",
    _cooling_label,
    delta=None,
    help="Confidence: Raster-based calculation. Approximate temperature change vs baseline. Positive = cooler, negative = warmer. Derived from mean Cooling Capacity (CC) under the InVEST UCM (calibration factor 3.69°F/CC unit from Minneapolis UHI=2.05°C; ±2°F accuracy). Note: this is mean(CC), an approximation of the canonical InVEST Heat Mitigation Index — see UCM_AUDIT.md."
)
if abs(_cooling_f) < 0.05:
    eco2.markdown(
        '<p style="color: gray; font-size: 0.85em;">↔ No change vs baseline</p>',
        unsafe_allow_html=True,
    )
elif _cooling_f > 0:
    eco2.markdown(
        f'<p style="color: green; font-size: 0.85em;">● {_cooling_f:.1f}°F cooler vs baseline</p>',
        unsafe_allow_html=True,
    )
else:
    eco2.markdown(
        f'<p style="color: red; font-size: 0.85em;">● {abs(_cooling_f):.1f}°F warmer vs baseline</p>',
        unsafe_allow_html=True,
    )
eco3.metric(
    "Runoff Volume",
    _fmt_runoff(results['runoff_acre_feet']),
    delta=_runoff_delta_str,
    delta_color="off" if _runoff_negligible else "normal",
    help=(
        "Confidence: Raster-based calculation. "
        f"Acre-feet of runoff generated by a {DESIGN_STORM_INCHES}-inch design storm. "
        f"Delta shows reduction vs baseline ({_fmt_runoff(BASELINE_RUNOFF_ACRE_FEET)}). "
        "Lower volume = more retention."
    )
)

_ndvi_delta = results['mean_ndvi'] - BASELINE_NDVI
_ndvi_delta_str = f"{_ndvi_delta:+.3f} vs base" if abs(_ndvi_delta) >= 0.001 else "no change"

eco4, eco5 = st.columns([2, 1])
eco4.metric(
    "Carbon Sequestration",
    _carbon_value_str,
    delta=_carbon_delta_str,
    delta_color="normal" if _carbon_value >= 1.0 else "off",
    help=(
        "Confidence: Provisional assumption. "
        "Annual CO2e sequestration from converted pixels only. "
        "Uses provisional regional USDA/IPCC rates: Food Forest 3.5 t CO2e/acre/yr, "
        "Green Infrastructure 2.0 t CO2e/acre/yr. "
        "Treat as directional only — refine with locally calibrated values."
    )
)
eco5.metric(
    "NDVI",
    f"{results['mean_ndvi']:.3f}",
    delta=_ndvi_delta_str,
    delta_color="normal" if abs(_ndvi_delta) >= 0.001 else "off",
    help=(
        "Confidence: Synthetic proxy. "
        "Synthetic vegetation index (0–1) estimated from land cover type — not derived from satellite imagery. "
        "Higher = more vegetation. Woody wetlands: 0.70, Food Forest: 0.75, High Density: 0.10–0.30. "
        "Treat as directional only."
    )
)

st.divider()

st.markdown("#### Human & Social")
hs1, hs2, hs3 = st.columns(3)
_nature_delta = results['nature_access_pct'] - BASELINE_NATURE_ACCESS_PCT
_nature_help = (
    "Confidence: Proximity estimate. "
    f"Share of residents in the model area whose access score exceeds "
    f"{NATURE_ACCESS_THRESHOLD} — using the InVEST UNA biophysical table "
    "(per-class urban_nature score and search radius). Model area covers "
    "downtown Minneapolis and near-neighborhoods (~154,000 residents). "
    "Euclidean distance, not street-network walking. Population from US "
    "Census 2020 block data."
) if POPULATION_DATA_AVAILABLE else (
    "Confidence: Proximity estimate. "
    "Currently using placeholder uniform population weighting — run "
    "`download_census_pop.py` to build the real Census-derived raster. "
    "Proximity metric only, not street-network walking distance."
)
hs1.metric(
    "Nature Access",
    f'{results["nature_access_pct"]:.1f}%',
    delta=f"{_nature_delta:+.1f} percentage points vs baseline",
    delta_color="normal" if abs(_nature_delta) >= 0.1 else "off",
    help=_nature_help,
)
if not POPULATION_DATA_AVAILABLE:
    hs1.caption(
        "⚠️ Nature Access currently uses uniform population weighting — "
        "proximity to green space only, not weighted by where people live. "
        "Real population data loading."
    )
if use_heat_priority:
    hs1.caption(
        "Heat-weighted placement concentrates conversions in higher-intensity "
        "developed areas, which tend to have lower existing nature access — "
        "improving equity of green space distribution."
    )

# Nature Quality Score — population-weighted mean access score (0–1). Sits
# alongside Nature Access % to capture the *graded* quality of nearby green
# space rather than a pure binary "in / out of buffer" share.
_nature_quality = results.get('nature_quality_score', 0.0)
_nature_quality_delta = _nature_quality - BASELINE_NATURE_QUALITY_SCORE
hs2.metric(
    "Nature Quality Score",
    f'{_nature_quality:.3f}',
    delta=f"{_nature_quality_delta:+.3f} vs baseline",
    delta_color="normal" if abs(_nature_quality_delta) >= 0.001 else "off",
    help=(
        "Confidence: Composite proxy. Population-weighted mean nature access "
        "quality score (0–1) based on InVEST Urban Nature Access biophysical "
        "table. Reflects both proximity and quality of nearby green space. "
        "Each pixel's score is the MAX (not sum) of `urban_nature × in_range` "
        "across all natural classes — a pixel near multiple nature types gets "
        "the highest single class score, preventing double-counting."
    ),
)

# Urban Wellbeing Score — composite index. Baseline is recomputed each rerun
# using the *current* weights so the delta stays meaningful when the user tunes
# weights in Advanced Settings.
_active_w_ndvi    = st.session_state.get("wgt_ndvi",    DEFAULT_WGT_NDVI)
_active_w_cooling = st.session_state.get("wgt_cooling", DEFAULT_WGT_COOLING)
_active_w_nature  = st.session_state.get("wgt_nature",  DEFAULT_WGT_NATURE)
_baseline_wellbeing = compute_wellbeing_score(
    BASELINE_NDVI, BASELINE_HM, BASELINE_NATURE_QUALITY_SCORE,
    _active_w_ndvi, _active_w_cooling, _active_w_nature,
)
_wellbeing_delta = results['wellbeing_score'] - _baseline_wellbeing
hs3.metric(
    "Urban Wellbeing Score",
    f'{results["wellbeing_score"]:.3f}',
    delta=f"{_wellbeing_delta:+.3f} vs baseline",
    delta_color="normal" if abs(_wellbeing_delta) >= 0.001 else "off",
    help=(
        "Confidence: Composite proxy. "
        "Composite index combining NDVI (vegetation), urban cooling, and nature "
        "access. Weights are provisional and adjustable in Advanced Settings. "
        "Treat as directional only — not a validated mental health model."
    ),
)

st.divider()

st.markdown("#### Economic")
econ1, econ2, econ3, econ4 = st.columns(4)
econ1.metric(
    "Food Production",
    _fmt_food(results['food_mln_lbs']),
    delta=_food_delta_str,
    delta_color="normal" if _people_fed > 0 else "off",
    help="Confidence: Provisional assumption. Counts only food forest pixels created by this scenario (not pre-existing deciduous forest). Yield estimated at 11,500 lbs/acre/year based on NatCap food forest benchmarks — treat as directional only."
)
if results['food_mln_lbs'] == 0:
    econ1.caption(
        "No food forest in this scenario — add Food Forest % to see production estimates."
    )
econ2.metric(
    "Est. Implementation Cost",
    f"${results['total_cost_mln']:.1f}M",
    delta=None,
    help="Confidence: Order-of-magnitude estimate. Total cost based on $/acre sliders × converted acreage."
)
_flood_damage_avoided = results.get('flood_damage_avoided_usd', 0.0)
if BUILDINGS_DATA_AVAILABLE:
    econ3.metric(
        "Flood Damage Avoided",
        f"${_flood_damage_avoided / 1e6:.1f}M",
        delta=(
            f"+${_flood_damage_avoided / 1e6:.1f}M vs baseline"
            if _flood_damage_avoided >= 1e4 else "no avoided damage"
        ),
        delta_color="normal" if _flood_damage_avoided >= 1e4 else "off",
        help=(
            "Confidence: Order-of-magnitude estimate. Estimated reduction in "
            "flood damage costs based on the InVEST damage-loss table by "
            "building type (Roads $40, Commercial $120, Residential $150, "
            "Industrial $100 per m²) joined to a 3,788-building footprint "
            "shapefile. Scales with this scenario's runoff reduction vs "
            f"baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). Capped at $0 "
            "for scenarios that increase runoff."
        ),
    )
else:
    econ3.metric(
        "Flood Damage Avoided",
        "—",
        help=(
            "Confidence: Order-of-magnitude estimate. Buildings shapefile or "
            "damage-loss table not loaded — see "
            "data/invest/flood/UFR_sample_data_MN/."
        ),
    )

_energy_savings = results.get('cooling_energy_savings_usd', 0.0)
_energy_available = (
    BUILDINGS_DATA_AVAILABLE and ENERGY_TABLE_AVAILABLE and ET_DATA_AVAILABLE
)
if _energy_available:
    econ4.metric(
        "Cooling Energy Savings",
        f"${_energy_savings / 1e6:.2f}M/yr",
        delta=(
            f"+${_energy_savings / 1e6:.2f}M/yr vs baseline"
            if _energy_savings >= 1e3 else "no avoided energy cost"
        ),
        delta_color="normal" if _energy_savings >= 1e3 else "off",
        help=(
            "Confidence: Order-of-magnitude estimate. Estimated avoided air "
            "conditioning energy costs from urban cooling improvement. Based "
            "on InVEST Urban Cooling Model: per-pixel ΔT in °F (= ΔCC × 4) "
            "× 3% AC reduction per °F × per-class consumption (kWh/m²/yr from "
            "energy_consumption.csv) × pixel area × $0.13/kWh, summed over "
            "building pixels. Capped at $0 for scenarios that warm the city."
        ),
    )
else:
    econ4.metric(
        "Cooling Energy Savings",
        "—",
        help=(
            "Confidence: Order-of-magnitude estimate. ET raster, energy table, "
            "or buildings shapefile not loaded — see "
            "data/invest/cooling/UrbanCooling_sample_data/."
        ),
    )

st.divider()

ce = compute_cost_effectiveness(results, BASELINE_RUNOFF_ACRE_FEET)
st.markdown("#### Cost Effectiveness")
st.caption(
    "Shows N/A when the scenario performs worse than the baseline on that metric, "
    "or when no land is converted. Try adding more green infrastructure or food "
    "forest to see values appear."
)
ceff1, ceff2, ceff3 = st.columns(3)
ceff1.metric(
    "Cost / Acre-Foot Prevented",
    _fmt_ce(ce['cost_per_acft']),
    delta=None,
    help=f"Confidence: Order-of-magnitude estimate. Implementation cost divided by runoff reduction vs baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). N/A if scenario increases runoff or has no cost."
)
ceff2.metric(
    "Cost / °F Cooling",
    _fmt_ce(ce['cost_per_degf']),
    delta=None,
    delta_color="off" if _cooling_f <= 0 else "normal",
    help="Confidence: Order-of-magnitude estimate. Implementation cost divided by degrees F of cooling vs baseline. N/A if no cooling improvement."
)
ceff3.metric(
    "Cost / 1,000 People Fed",
    _fmt_ce(ce['cost_per_1k_people']),
    delta=None,
    help="Confidence: Order-of-magnitude estimate. Implementation cost divided by (people fed ÷ 1,000). N/A if no food production."
)

st.caption(
    "For outcome metrics, higher is generally better except Runoff Volume, where "
    "lower is better. For cost-effectiveness ratios, lower cost per unit of "
    "benefit is better."
)

with st.expander("Baseline vs Scenario Comparison", expanded=False):
    _baseline_flood = 100 - BASELINE_CN
    _runoff_diff    = results['runoff_acre_feet'] - BASELINE_RUNOFF_ACRE_FEET
    _flood_diff     = results['flood_reduction'] - _baseline_flood

    _flood_damage_avoided = results.get('flood_damage_avoided_usd', 0.0)
    _energy_savings_table = results.get('cooling_energy_savings_usd', 0.0)
    comparison_data = {
        'Metric': [
            'Flood Risk Reduction', 'Runoff Volume', 'Temperature Change',
            'Food Production', 'Carbon Sequestration', 'Nature Access', 'NDVI',
            'Flood Damage Avoided', 'Cooling Energy Savings',
        ],
        'Baseline': [
            f'{_baseline_flood:.1f}',
            f'{BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft',
            'Reference',
            '0 lbs',
            '0 tons CO2e/yr',
            f'{BASELINE_NATURE_ACCESS_PCT:.1f}%',
            f'{BASELINE_NDVI:.3f}',
            '$0',
            '$0/yr',
        ],
        'This Scenario': [
            f'{results["flood_reduction"]:.1f}',
            f'{results["runoff_acre_feet"]:,.0f} ac-ft',
            (
                f'{_cooling_f:.1f}°F cooler' if _cooling_f > 0
                else f'{abs(_cooling_f):.1f}°F warmer' if _cooling_f < 0
                else 'No change'
            ),
            f'{results["food_mln_lbs"] * 1e6:,.0f} lbs/yr',
            f'{results["carbon_tons_co2_yr"]:,.0f} tons CO2e/yr',
            f'{results["nature_access_pct"]:.1f}%',
            f'{results["mean_ndvi"]:.3f}',
            f'${_flood_damage_avoided / 1e6:.1f}M',
            f'${_energy_savings_table / 1e6:.2f}M/yr',
        ],
        'Change': [
            f'{_flood_diff:+.1f}',
            (
                f'+{_runoff_diff:,.0f} ac-ft'         if _runoff_diff > 0
                else f'{abs(_runoff_diff):,.0f} ac-ft prevented' if _runoff_diff < 0
                else '0 ac-ft'
            ),
            f'{_cooling_f:+.1f}°F',
            f'+{results["food_mln_lbs"] * 1e6:,.0f} lbs/yr',
            f'+{results["carbon_tons_co2_yr"]:,.0f} tons CO2e/yr',
            f'{results["nature_access_pct"] - BASELINE_NATURE_ACCESS_PCT:+.1f} pp',
            f'{results["mean_ndvi"] - BASELINE_NDVI:+.3f}',
            f'+${_flood_damage_avoided / 1e6:.1f}M' if _flood_damage_avoided >= 1e4 else '$0',
            f'+${_energy_savings_table / 1e6:.2f}M/yr' if _energy_savings_table >= 1e3 else '$0/yr',
        ],
    }

    _comparison_df = pd.DataFrame(comparison_data)

    def _color_change(val):
        s = str(val)
        # Runoff is inverse — positive change is bad
        if 'ac-ft' in s and s.startswith('+'):
            return 'color: red'
        if s.startswith('+') or 'prevented' in s or 'cooler' in s:
            return 'color: green'
        if s.startswith('-') or 'warmer' in s or 'worse' in s:
            return 'color: red'
        return 'color: gray'

    _styled = _comparison_df.style.map(_color_change, subset=['Change'])
    st.dataframe(_styled, use_container_width=True, hide_index=True)

with st.expander("Assumptions and limitations"):
    _assumption_tabs = st.tabs([
        "Flood & Runoff", "Temperature", "Food", "Carbon",
        "Nature Access", "Wellbeing Score", "Costs",
    ])
    with _assumption_tabs[0]:
        st.markdown(
            "- **Method:** USDA SCS Curve Number method, computed at 30 m raster "
            "resolution from per-pixel CN values × soil hydrologic group lookup. "
            "Reported as `100 − mean_CN` so higher = better.\n"
            "- **Design storm:** 2-inch rainfall — a common minor event for "
            "Minneapolis. Larger storms scale runoff non-linearly; results don't "
            "extrapolate to extreme events.\n"
            "- **Green Infrastructure** is modeled as woody wetlands (NLCD 90). "
            "The broader GI category (rain gardens, bioswales, permeable pavement, "
            "green roofs, urban tree canopy) is not modeled — each would have "
            "different curve numbers."
        )
    with _assumption_tabs[1]:
        st.markdown(
            "- **Method:** InVEST Urban Cooling Model. Per-pixel Cooling "
            "Capacity `CC = 0.6·shade + 0.2·albedo + 0.2·ETI`, then Gaussian-"
            "smoothed over a 450 m kernel so cooling propagates onto "
            "neighbouring pixels (per InVEST `green_area_cooling_distance`).\n"
            "- **Reported value:** mean(CC) across the AOI, labeled CC. This "
            "approximates but is not identical to the canonical InVEST Heat "
            "Mitigation Index (HMI) — see UCM_AUDIT.md.\n"
            "- **Calibration:** 3.69 °F per CC unit, from Minneapolis "
            "`uhi_max = 2.05 °C` in the InVEST args JSON. Not independently "
            "calibrated for MN — treat the °F output as ±2 °F at best.\n"
            "- **Not captured:** wind, humidity, urban geometry, building "
            "materials, anthropogenic heat. The model sees land cover only."
        )
    with _assumption_tabs[2]:
        st.markdown(
            "- **Food Forest** is modeled as deciduous forest (NLCD 41) — the "
            "closest available NLCD class. No NLCD class exists specifically for "
            "agroforestry or food forests.\n"
            "- **Yield benchmark:** 11,500 lbs/acre/year, from NatCap food-forest "
            "studies. Assumes a mature, well-managed system at peak productivity. "
            "Newly established food forests will produce significantly less in "
            "early years.\n"
            "- **Counts only newly converted pixels** — pre-existing deciduous "
            "forest doesn't add to the food production tally."
        )
    with _assumption_tabs[3]:
        st.markdown(
            "- **Method:** newly converted pixel counts × pixel area × per-cover "
            "rate. Existing land cover is not credited and not penalized.\n"
            "- **Default rates** are provisional regional USDA NRCS / IPCC "
            "values: Food Forest 3.5 t CO2e/acre/yr, Green Infrastructure "
            "2.0 t CO2e/acre/yr, High Density 0.0. Wide published ranges (e.g. "
            "1.76–18.2 for managed food forests) — adjust the sliders in "
            "**Advanced Settings → Urban Wellbeing Score weights** Food Forest "
            "and Green Infrastructure carbon rate sliders.\n"
            "- **Not locally calibrated.** Refine with site-specific data when "
            "available."
        )
    with _assumption_tabs[4]:
        st.markdown(
            "- **Proximity proxy, not walkshed:** Euclidean distance from each "
            "pixel to the nearest nature pixel × 30 m, thresholded at 800 m. "
            "Ignores street networks, barriers, slope, and crossings.\n"
            "- **Population data:** US Census 2020 block totals (Hennepin "
            "County, FIPS 27053) joined to TIGER 2020 blocks and rasterized to "
            "the NLCD grid.\n"
            "- **Extent caveat:** the NLCD raster only covers ~10.8 km × "
            "10.7 km of downtown Minneapolis (~154,000 residents in extent), "
            "not the full city.\n"
            "- **Spatial placement is building-footprint-aware.** Conversions "
            "are sampled from developed pixels that do NOT contain a building, "
            "using the InVEST UFR buildings shapefile to mask out structures. "
            "This targets feasible interstitial spaces (parking lots, lawns, "
            "vacant lots) but still ignores parcel ownership, corridor design, "
            "and zoning — placement within the convertible pool is random "
            "(or heat-weighted in Heat-Priority Mode).\n"
            "- **High-density-only conversion ties baseline:** the model never "
            "removes existing nature, so adding HD alone leaves the buffer "
            "unchanged.\n"
            "- **Saturation:** at ≥ 50 % conversion with aggressive green "
            "allocations the metric tops out around 65–66 % as buffers overlap. "
            "Most discriminating at lower conversion percentages."
        )
    with _assumption_tabs[5]:
        st.markdown(
            "- **Composite, not validated:** weighted combination of NDVI "
            "(synthetic), CC (cooling proxy), and Nature Access %. Each input "
            "carries its own uncertainty.\n"
            "- **Weights are arbitrary** until empirically validated against a "
            "local quality-of-life or self-reported wellbeing dataset. "
            "Defaults (0.2 / 0.4 / 0.4) are a reasonable starting point, not a "
            "claim of correctness.\n"
            "- **Not a mental health model.** Useful for cross-scenario "
            "ranking, not for population-health inference.\n"
            "- **Not in the surrogate** — wellbeing is computed live from "
            "current weights, but the optimizer doesn't search over it."
        )
    with _assumption_tabs[6]:
        st.markdown(
            "- **Order-of-magnitude only:** total cost = "
            "`$/acre slider × converted acres`, summed across green "
            "infrastructure, food forest, and high-density development. "
            "Default $/acre ranges come from broad planning literature, not "
            "site-specific bids.\n"
            "- **Cost-effectiveness ratios** divide the cost by a per-unit "
            "benefit (acre-foot prevented, °F cooling, 1,000 people fed). "
            "Returns N/A when the denominator is zero or negative — never "
            "infinite or misleading.\n"
            "- **Buildings and roads excluded.** Conversions never land on "
            "top of existing buildings or road infrastructure — the InVEST "
            "UFR buildings shapefile and a citywide OpenStreetMap road "
            "network (fetched once via `download_osm_roads.py`) are both "
            "rasterized, unioned, and subtracted from the candidate pool. "
            "Both are still part of the runoff calculation (they shed water "
            "like any developed surface), but they're not eligible to be "
            "replaced by GI/FF/HD. Real projects still need site-by-site "
            "feasibility checks (zoning, ownership, soil, infrastructure).\n"
            "- **Optimized scenarios** come from a Random Forest surrogate. "
            "Verify any suggestion by manually applying it to the main "
            "sliders so the full pixel-level simulation runs."
        )

st.divider()

if st.session_state.get("just_optimized"):
    banner_col, dismiss_col = st.columns([5, 1])
    with banner_col:
        st.success(
            "Optimization complete — open the Tradeoff Analysis tab to see results."
        )
    with dismiss_col:
        if st.button("✕", key="dismiss_optimize_banner"):
            st.session_state.just_optimized = False
            st.rerun()

mode_text = "prioritizing high heat-exposure areas" if use_heat_priority else "using random placement"
st.write(
    f"This scenario converts **{pct_converted}%** of developed land, allocating "
    f"**{green_infrastructure_pct}%** to green infrastructure, "
    f"**{food_forest_pct}%** to food forest, and **{pct_highdensity}%** "
    f"to high-density development, {mode_text}."
)

if st.session_state.get("just_optimized"):
    st.info(
        "Click the **Tradeoff Analysis** tab above to see your optimization results."
    )

tab1, tab2, tab3, tab4 = st.tabs(["Scenario", "Tradeoff Analysis", "Map View", "Reference"])

with tab1:
    st.subheader("Outcome Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(['Baseline', 'This Scenario'],
               [BASELINE_CN, results['mean_cn']],
               color=['#5b8db8', '#7b4fa6'])
        ax.axhline(BASELINE_CN, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Flood Risk', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Curve Number\n(lower = less runoff)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(['Baseline', 'This Scenario'],
               [BASELINE_HM, results['mean_hm']],
               color=['#5b8db8', '#7b4fa6'])
        ax.axhline(BASELINE_HM, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Urban Cooling', fontsize=16, fontweight='bold')
        ax.set_ylabel('Cooling Capacity\n(higher = more cooling)', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(['Baseline', 'This Scenario'],
               [BASELINE_FOOD_MLN_LBS, results['food_mln_lbs']],
               color=['#5b8db8', '#7b4fa6'])
        ax.set_title('Food Production', fontsize=16, fontweight='bold')
        ax.set_ylabel('Food Production\n(million lbs/year)', fontsize=12)
        ax.set_ylim(0, max(MAX_FOOD * 1.1, 0.01))
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab2:
    # NOTE: We deliberately do NOT auto-clear `just_optimized` here. Streamlit
    # executes every `with tabX:` block on every rerun (regardless of which
    # tab is visible), so an auto-clear inside this block fires on the next
    # rerun rather than only when the user actually opens this tab — which
    # made the optimization banner vanish prematurely. The dismiss-X button
    # on the banner is now the only way to clear the flag, plus running a
    # new optimization (which sets it back to True or False).
    st.subheader("Tradeoff Space")
    st.caption("Each point is a scenario. Better outcomes are toward the top-right — more cooling and greater flood-risk reduction. Bubble size shows food production for saved and optimized scenarios.")
    st.plotly_chart(plot_tradeoff(
        results, scenario_df,
        lookup_table=lookup_table,
        saved=st.session_state.saved_scenarios,
        optimized=st.session_state.optimized_results
    ), use_container_width=True)

    if TRACTS_DATA_AVAILABLE:
        st.divider()
        st.markdown("#### Neighborhood breakdown")
        st.caption(
            "Top 5 most-improved Census tracts under this scenario, ranked by a "
            "combined score (Nature Access percentage points + temperature change °F). "
            "Population-weighted within each tract."
        )
        _tracts_summary = compute_per_tract_summary(results['scenario_lulc'])
        if not _tracts_summary.empty:
            _tracts_summary["_combined"] = (
                _tracts_summary["Access Δ (pp)"]
                + 5 * _tracts_summary["Temp Δ (°F cooler)"]   # weight °F more strongly
            )
            _top5 = (
                _tracts_summary
                .sort_values("_combined", ascending=False)
                .head(5)
                .drop(columns="_combined")
                .reset_index(drop=True)
            )
            st.dataframe(_top5, use_container_width=True, hide_index=True)
        else:
            st.caption("No tract-level data could be computed for this scenario.")

    st.divider()
    st.markdown("#### Best scenarios by goal")
    st.caption("From the pre-computed scenario library — not surrogate predictions.")

    lookup_df = pd.DataFrame(lookup_table.values())
    _norm_flood = lookup_df['flood_reduction'] / max(lookup_df['flood_reduction'].max(), 1e-9)
    _norm_hm    = lookup_df['mean_hm']         / max(lookup_df['mean_hm'].max(),         1e-9)
    _norm_food  = lookup_df['food_mln_lbs']    / max(lookup_df['food_mln_lbs'].max(),    1e-9)
    _balanced_score = _norm_flood + _norm_hm + _norm_food

    best_by_goal = {
        "Best for flood reduction": lookup_df.loc[lookup_df['flood_reduction'].idxmax()],
        "Best for cooling":         lookup_df.loc[lookup_df['mean_hm'].idxmax()],
        "Best for food production": lookup_df.loc[lookup_df['food_mln_lbs'].idxmax()],
        "Best for carbon":          lookup_df.loc[lookup_df['carbon_tons_co2_yr'].idxmax()],
        "Best balanced":            lookup_df.loc[_balanced_score.idxmax()],
    }

    for i, (goal, row) in enumerate(best_by_goal.items()):
        text_col, btn_col = st.columns([4, 1])
        with text_col:
            st.markdown(
                f"**{goal}:** {int(row.pct_converted)}% converted — "
                f"{int(row.green_infrastructure_pct)}% GI / {int(row.food_forest_pct)}% FF"
            )
        with btn_col:
            if st.button("Apply", key=f"apply_best_goal_{i}"):
                st.session_state._pending_pct = int(round(row.pct_converted / 5) * 5)
                st.session_state._pending_gi  = int(round(row.green_infrastructure_pct / 5) * 5)
                st.session_state._pending_ff  = int(round(row.food_forest_pct / 5) * 5)
                if st.session_state._pending_gi + st.session_state._pending_ff > 100:
                    st.session_state._pending_ff = 100 - st.session_state._pending_gi
                st.session_state._show_apply_toast = True
                st.rerun()

    if st.session_state.get("_show_apply_toast"):
        st.success("Applied — check the Scenario tab to see updated results.")
        st.session_state._show_apply_toast = False

    st.divider()

    if st.button("Save this scenario"):
        st.session_state.show_save_input = True

    if st.session_state.get("show_save_input"):
        scenario_name_input = st.text_input(
            "Name this scenario:",
            placeholder="e.g. High GI / Low Cost",
            key="scenario_name_input",
        )
        confirm_col, cancel_col = st.columns([1, 5])
        with confirm_col:
            confirm_clicked = st.button("Confirm save")
        with cancel_col:
            if st.button("Cancel", key="cancel_save"):
                st.session_state.show_save_input = False
                st.rerun()
        if confirm_clicked and scenario_name_input:
            saved = {k: v for k, v in results.items() if k != 'scenario_lulc'}
            saved["display_name"] = scenario_name_input
            saved["heat_priority"] = use_heat_priority
            saved["cost_gi"] = cost_gi
            saved["cost_ff"] = cost_ff
            saved["cost_hd"] = cost_hd
            _ce = compute_cost_effectiveness(results, BASELINE_RUNOFF_ACRE_FEET)
            saved["cost_per_acft"]      = _ce['cost_per_acft']
            saved["cost_per_degf"]      = _ce['cost_per_degf']
            saved["cost_per_1k_people"] = _ce['cost_per_1k_people']
            st.session_state.saved_scenarios.append(saved)
            st.session_state.show_save_input = False
            st.success(f"Saved: {scenario_name_input}")
            st.rerun()
        elif confirm_clicked and not scenario_name_input:
            st.warning("Please enter a name before saving.")

    if st.session_state.optimized_results is not None:
        st.divider()
        st.subheader("Optimized Scenario Suggestions")
        st.caption("Scroll down to see suggestions and apply them to the sliders.")
        opt = st.session_state.optimized_results
        if isinstance(opt, dict) and not opt.get('found'):
            st.warning(
                f"No scenarios found meeting all targets simultaneously.  \n"
                f"Maximum achievable values across all candidates:  \n"
                f"- Flood reduction: up to **{opt['max_flood']}** (your target: {min_flood})  \n"
                f"- Cooling: up to **{opt['max_cool']:.4f} HM** (your target: {min_cool:.4f})  \n"
                f"- Food: up to **{opt['max_food']:.3f}M lbs** (your target: {min_food:.3f})  \n"
                f"- Carbon: up to **{opt['max_carbon']:,.0f} tons CO2e/yr** (your target: {min_carbon:,})  \n"
                f"Try lowering the target for whichever metric is furthest from its maximum."
            )
        else:
            st.caption(
                f"Top scenarios meeting flood ≥ {min_flood}, cooling ≥ {min_cool_f:+.1f}°F, "
                f"food ≥ {min_food:.3f}M lbs, carbon ≥ {min_carbon:,} tons CO2e/yr "
                "— ranked by balanced score. "
                "Numbers are surrogate model predictions with 10th–90th percentile uncertainty bands."
            )

            # Display table with uncertainty columns
            display_cols = ['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                            'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs',
                            'carbon_tons_co2_yr']
            # Add uncertainty columns if present
            unc_cols = [c for c in ['flood_lower', 'flood_upper', 'hm_lower', 'hm_upper',
                                    'food_lower', 'food_upper',
                                    'carbon_lower', 'carbon_upper'] if c in opt.columns]
            _col_rename = {
                'scenario_name':            'Scenario',
                'pct_converted':            'Total Conversion (%)',
                'green_infrastructure_pct': 'Green Infra %',
                'food_forest_pct':          'Food Forest %',
                'flood_reduction':          'Flood Index',
                'mean_hm':                  'Cooling HM',
                'food_mln_lbs':             'Food (M lbs)',
                'carbon_tons_co2_yr':       'Carbon (tons CO2e/yr)',
            }

            st.markdown("#### Candidate scenarios")
            st.caption(
                "These are surrogate model predictions. Click Apply to run a "
                "full pixel-level simulation and verify the result."
            )
            with st.expander("Show uncertainty bands", expanded=False):
                st.dataframe(opt[display_cols + unc_cols].rename(columns=_col_rename),
                             use_container_width=True, hide_index=True)
            st.dataframe(opt[display_cols].rename(columns=_col_rename),
                         use_container_width=True, hide_index=True)
            st.caption(
                "Note: suggestions with small amounts of High Density (2–10%) may "
                "reflect surrogate approximation — consider setting HD to 0% when applying."
            )

            st.markdown("#### What drives the surrogate?")
            st.caption("**Influence Map** — which input drives outcomes most according to the surrogate model:")
            render_matplotlib(plot_feature_importance(surrogate))

            st.markdown("#### Apply a suggestion")
            st.caption(
                "Suggestions are ranked by balanced score across flood, cooling, "
                "and food metrics. #1 is the top-ranked scenario."
            )

            btn_cols = st.columns(len(opt))
            for i, (_, row) in enumerate(opt.iterrows()):
                with btn_cols[i]:
                    prefix = "✓ " if st.session_state.get("applied_suggestion") == i else ""
                    label = f"{prefix}#{i+1}: {int(row.pct_converted)}% conv"
                    if st.button(label, key=f"apply_opt_{i}"):
                        st.session_state._pending_pct = int(round(row.pct_converted / 5) * 5)
                        st.session_state._pending_gi  = int(round(row.green_infrastructure_pct / 5) * 5)
                        st.session_state._pending_ff  = int(round(row.food_forest_pct / 5) * 5)
                        if st.session_state._pending_gi + st.session_state._pending_ff > 100:
                            st.session_state._pending_ff = 100 - st.session_state._pending_gi
                        st.session_state.applied_suggestion = i
                        st.session_state._show_apply_toast = True
                        st.rerun()

            # One-shot confirmation toast: rendered on the rerun immediately
            # following an Apply click, then cleared so it doesn't persist
            # through unrelated reruns.
            if st.session_state.get("_show_apply_toast"):
                st.success("Applied — check the Scenario tab to see updated results.")
                st.session_state._show_apply_toast = False

            st.divider()

    if st.session_state.saved_scenarios:
        st.divider()
        st.caption(
            "The Pareto frontier shows the most efficient tradeoff scenarios — ones where you "
            "cannot improve flood reduction, cooling, or food production without making at least "
            "one of the others worse."
        )
        with st.expander(f"Saved Scenarios ({len(st.session_state.saved_scenarios)})", expanded=False):
            df_saved = pd.DataFrame(st.session_state.saved_scenarios)
            # Older saves predate display_name; backfill from scenario_name so the
            # column is always present and never NaN in the table or hover labels.
            if 'display_name' not in df_saved.columns:
                df_saved['display_name'] = df_saved.get('scenario_name', '')
            else:
                df_saved['display_name'] = df_saved['display_name'].fillna('').replace('', np.nan)
                df_saved['display_name'] = df_saved['display_name'].fillna(df_saved['scenario_name'])

            show_cols = [c for c in [
                'display_name',
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

            csv = df_saved[show_cols].to_csv(index=False)
            st.download_button(
                "Download saved scenarios as CSV",
                csv,
                "ecosystem_explorer_scenarios.csv",
                "text/csv",
                type="primary",
            )

            st.dataframe(df_saved[show_cols], use_container_width=True, hide_index=True)

            st.caption(
                "Note: saved scenarios are lost on page refresh — download the CSV to keep them."
            )

            if st.button("Clear saved scenarios"):
                st.session_state.saved_scenarios = []
                st.rerun()

with tab3:
    st.subheader("Where Changes Happen")
    if use_heat_priority:
        st.info(
        "**Heat-exposure mode active** — conversions concentrated in higher-intensity "
        "developed areas. Notice the spatial pattern shift vs. random allocation."
        )

    overlay_opacity = st.slider(
        "Heat vulnerability overlay opacity",
        0.0, 1.0, 0.3, 0.05,
        help=(
            "Tint developed pixels red in proportion to their heat vulnerability "
            "(NLCD development intensity proxy). 0 hides the overlay; 1 makes "
            "the highest-intensity-23 pixels fully red."
        ),
    )

    # Optional neighborhood-improvement overlay — colors each tract by its
    # population-weighted Nature Access change (pp), using a diverging RdYlGn
    # colormap centered at 0. Only rendered when the toggle below is on.
    _tract_overlay_value = None
    _tract_overlay_alpha = 0.0
    if TRACTS_DATA_AVAILABLE:
        _show_tracts = st.toggle(
            "Show neighborhood improvement overlay",
            value=False,
            help=(
                "Color each Census tract by its Nature Access change vs baseline "
                "(green = improved, red = worse). Computed live for the current "
                "scenario."
            ),
        )
        if _show_tracts:
            _tract_overlay_alpha = st.slider(
                "Neighborhood overlay opacity",
                0.0, 1.0, 0.5, 0.05,
            )
            _tracts_summary_map = compute_per_tract_summary(results['scenario_lulc'])
            if not _tracts_summary_map.empty:
                _imp_lookup = np.zeros(len(TRACTS), dtype=np.float32)
                _geoid_to_idx = {str(g): i for i, g in enumerate(TRACTS["GEOID10"])}
                for _, _srow in _tracts_summary_map.iterrows():
                    _tidx = _geoid_to_idx.get(str(_srow["GEOID"]))
                    if _tidx is not None:
                        _imp_lookup[_tidx] = float(_srow["Access Δ (pp)"])
                # Per-pixel: pixels in a tract get that tract's improvement
                # value; pixels outside any tract become NaN so the overlay
                # only renders where it has data.
                _tract_overlay_value = np.where(
                    TRACT_ID_RASTER >= 0,
                    _imp_lookup[np.maximum(TRACT_ID_RASTER, 0)],
                    np.nan,
                )

    render_matplotlib(plot_spatial_map(
        results['scenario_lulc'], cooling_lulc,
        heat_overlay=equity_weights, overlay_alpha=overlay_opacity,
        tract_value=_tract_overlay_value, tract_alpha=_tract_overlay_alpha,
    ))
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary. Red wash = heat vulnerability proxy "
        "(NLCD development intensity), with opacity controlled by the slider above.  \n"
        "Conversions target feasible interstitial spaces — building footprints "
        "and road infrastructure are excluded citywide using OpenStreetMap "
        "road network data unioned with the InVEST UFR buildings shapefile. "
        "The remaining candidate pool covers parking lots, lawns, and vacant "
        "land within the NLCD-21/22/23/24 developed mask. Placement within "
        "that pool is random, or weighted toward higher-intensity-developed "
        "pixels when Heat-Priority Mode is on. Real implementation would "
        "still require site-specific siting analysis (zoning, ownership, "
        "soil, infrastructure)."
    )

with tab4:
    st.markdown("## Methodology & Data Sources")
    try:
        with open("REFERENCE.md", "r") as f:
            reference_content = f.read()
        st.markdown(reference_content)
    except FileNotFoundError:
        st.error("REFERENCE.md not found.")

with st.expander("Intended Use", expanded=False):
    st.markdown(
        "**This tool is designed for:**\n"
        "- Comparing alternative land-use allocation strategies\n"
        "- Exploring tradeoffs across multiple ecosystem services\n"
        "- Identifying candidate scenarios for deeper analysis\n\n"
        "**It is not intended for:**\n"
        "- Parcel-level siting decisions\n"
        "- Precise impact prediction\n"
        "- Final policy or investment decisions without further analysis"
    )