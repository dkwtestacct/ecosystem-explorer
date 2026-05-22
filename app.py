import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import plotly.graph_objects as go
import os
from pathlib import Path
from typing import NamedTuple, Any, Optional

import rasterio
from rasterio.features import rasterize as _rasterize
from skimage.transform import resize
import geopandas as _gpd
from scipy.ndimage import gaussian_filter as _gaussian_filter
from scipy.ndimage import distance_transform_edt as _distance_transform_edt
from scipy.ndimage import zoom as _zoom
from scipy.signal import fftconvolve as _fftconvolve

from config import CITIES, DEFAULT_COST_GI, DEFAULT_COST_FF, DEFAULT_COST_HD
from surrogate import (
    train_surrogate as _train_surrogate_fn,
    predict_with_uncertainty,
    plot_feature_importance,
    optimize_scenario,
    compute_pareto,
)

PIXEL_AREA_ACRES     = 0.2224  # 30 m × 30 m = 900 m² ÷ 4046.86 m²/acre. Same in EPSG:26915 (UTM) and EPSG:5070 (Albers); UTM ground-area distortion at MN is ~0.05 %, well within rounding.
# FOOD_FOREST_LBS_ACRE is city-dependent — see "── City-derived constants ──" below.

DEVELOPED_CODES   = [21, 22, 23]
CODE_GREEN_INFRA  = 90
CODE_FOOD_FOREST  = 41
CODE_HIGH_DENSITY = 24
NODATA            = -128

# ── Metric translation constants ───────────────────────────────────────────────
# SCS design storm: 2-inch rainfall event (typical minor storm)
DESIGN_STORM_INCHES   = 2.0
# UHI_MAX_C, HM_TO_FAHRENHEIT, and FOOD_FOREST_LBS_ACRE are city-dependent and
# initialized after city_cfg is built (see "── City-derived constants ──" below).
# Food: average American consumes ~2,000 lbs of food per year
LBS_PER_PERSON_YEAR   = 2_000

CHANGE_COLORS = {
    'Unchanged':            '#d3d3d3',
    'Green Infrastructure': '#2196a0',
    'Food Forest':          '#4caf50',
    'High Density':         '#e53935',
}

# ── "Recently / Coming up" in-app changelog ────────────────────────────────────
# A small changelog for returning visitors. Surfaces user-facing changes and
# direction signals only — architecture / testing / refactoring / internal-doc
# changes are deliberately left out. Edit WHATS_NEW whenever something
# user-visible ships.
WHATS_NEW = """
### What's new
- **Comprehensive OSM building footprints for the Minneapolis placement mask.** The placement-strategy non-convertible mask now unions the existing InVEST UFR sample buildings (downtown-core typed) with comprehensive Geofabrik OSM building footprints (~113k city-wide, untyped). Conversions can no longer be placed on any OSM building anywhere in the MN AOI — previously only downtown-core buildings were masked. The InVEST UFR sample buildings continue to drive Cooling Energy Savings and Flood Damage Avoided dollar metrics (which need the typed data), so those metrics are unchanged. Aligns with NatCap's recommendation to separate placement-constraint inputs from model-input data.
- **Canonical InVEST Urban Nature Access (UNA) implemented for Minneapolis.** The Nature Access metric card returns, now reporting `pct_pop_supply_ge_demand` from a canonical InVEST UNA two-step floating catchment area (2SFCA) calculation — validated by direct comparison against `natcap.invest.urban_nature_access.execute()`. Parameters per `DESIGN_NOTES.md` (16.7 m²/capita demand, 800 m uniform search radius, dichotomy decay). Reports the % of modelable-extent population (~43% of MN total); the remainder sit on cooling-LULC nodata pixels the model cannot evaluate.
- **Placement strategy picker** in the sidebar — five options for where conversions get sited.
- **Confidence badges** on every metric card (High / Medium / Prototype).
- **InVEST alignment section** in the methodology docs, with metric tooltips linking directly to the relevant InVEST user guides.
- **Interactive Input Influence chart** on the Tradeoff Analysis tab.
- **Cooling-model gap closed** — Temperature Change values now match canonical InVEST exactly, validated by direct comparison against `natcap.invest.urban_cooling_model.execute()`. Energy-cost values use the same canonical cooling input (per-pixel aggregation gap still open).
- **Nature Access and Nature Quality Score removed from the dashboard.** Phase 1 InVEST comparison and sensitivity testing showed neither metric meaningfully discriminates between scenarios at the MN downtown scale. Both metric cards, the per-tract map overlay, and the in-app methodology tab have been removed. The underlying calculations remain (used by the lookup table and surrogate); a redesigned nature-access metric is an open design question. See `UNA_DIVERGENCE_CASE_STUDIES.md`, `UNA_METHODOLOGY_CROSS_CHECK.md`, and `UNA_QUALITY_SCORE_SENSITIVITY.md`.

### Working on now
_Nothing in flight at the moment._

### On the radar
- **San Antonio as a fuller pilot** once more data is in place.
- **AlphaEarth Foundations satellite embeddings** as a future land-cover source — [feasibility research here](https://github.com/dkwtestacct/ecosystem-explorer/blob/main/ALPHAEARTH_FEASIBILITY.md).
"""

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
# Apply any pending slider values before sliders are rendered
if "_pending_pct" in st.session_state:
    st.session_state.slider_pct_converted = st.session_state.pop("_pending_pct")
    st.session_state.slider_gi_pct        = st.session_state.pop("_pending_gi")
    st.session_state.slider_ff_pct        = st.session_state.pop("_pending_ff")
    # active_example_scenario is set by the button handler before _pending_ keys are written

# ── City selection ─────────────────────────────────────────────────────────────
# Only available cities surface in the dropdown. Unavailable entries (e.g.
# Minneapolis Full) stay in the CITIES dict so scripts/tests can still
# reference them by key, but they are hidden from the UI.
_city_names = [name for name, cfg in CITIES.items() if cfg['available']]
selected_city = st.sidebar.selectbox("City", _city_names, index=0)
city_cfg = CITIES[selected_city]

# Reset scenario sliders when the city changes so a new city renders against
# its own defaults instead of inheriting the previous city's slider state.
# Runs BEFORE the sidebar widgets are instantiated. Preset buttons set
# `_pending_*` and trigger `st.rerun()`; on that rerun the city has not
# changed so this branch is skipped and the preset wins.
if st.session_state.get('_prev_city_key') != selected_city:
    for _k in ('slider_pct_converted', 'slider_gi_pct', 'slider_ff_pct'):
        st.session_state.pop(_k, None)
    st.session_state.active_example_scenario = None
    st.session_state._prev_city_key = selected_city

# ── City-derived constants ────────────────────────────────────────────────────
# Values that depend on the active city's climate / project parameters.
#   uhi_max_c — InVEST UCM urban heat-island anomaly (°C). MN: 2.05 from the
#     InVEST args JSON for the MN AOI; SA: 3.5 estimate for Köppen BSh.
#   food_forest_lbs_acre — annual yield benchmark for the food-forest land
#     cover. MN: 11,500 (NatCap MN benchmark); SA: 8,500 placeholder pending
#     project-report numbers for the pecan/fig/mulberry/nopal mix.
UHI_MAX_C            = city_cfg['uhi_max_c']
HM_TO_FAHRENHEIT     = UHI_MAX_C * 1.8
FOOD_FOREST_LBS_ACRE = city_cfg['food_forest_lbs_acre']

_CITY_CAPTIONS = {
    "Minneapolis, MN":      "Downtown and near-neighborhoods — 123 km², ~154k residents.",
    "Minneapolis Full, MN": "Full city boundary — 204 km², ~464k residents.",
    "San Antonio, TX":      "Bexar County area — ~3,060 km², ~1.9M residents.",
}
_caption = _CITY_CAPTIONS.get(selected_city)
if _caption:
    st.sidebar.caption(_caption)
st.sidebar.divider()



# Per-city biophysical-table provenance for the Temperature assumptions tab.
# Surfaces the SA Köppen-BSh tuning to users without hiding the
# medium-confidence framing.
_COOLING_BIOPHYSICAL_SOURCE_TEXT = {
    "Minneapolis, MN": (
        "Biophysical table from the InVEST UCM args JSON for the MN AOI "
        "(humid continental Köppen Dfa)."
    ),
    "Minneapolis Full, MN": (
        "Biophysical table from the InVEST UCM args JSON for the MN AOI "
        "(humid continental Köppen Dfa)."
    ),
    "San Antonio, TX": (
        "Biophysical table tuned for Köppen BSh (hot semi-arid) climate on "
        "four high-impact NLCD classes (Shrub/Scrub, Evergreen Forest, "
        "Deciduous Forest, Hay/Pasture); medium-confidence interim values "
        "pending a SA-calibrated InVEST UCM args run."
    ),
}


def _cooling_biophysical_source(city_key: str) -> str:
    return _COOLING_BIOPHYSICAL_SOURCE_TEXT.get(
        city_key,
        "Biophysical table sourced from the active city's configured "
        "`cooling_table_file`.",
    )

# ── City-aware header ──────────────────────────────────────────────────────────
st.title("🌿 Urban Ecosystem Tradeoff Explorer")

# In-app changelog for returning visitors — expanded by default so it's seen on
# reload; collapsible once read. Sits between the title and the city subheader.
# Wrapped in a bordered container for card-like visual separation.
with st.container(border=True):
    with st.expander("What's new / Coming up", expanded=True):
        st.markdown(WHATS_NEW)

st.subheader(selected_city)

def _preflight_data_check(city_cfg, city_name):
    """Verify all *required* input files referenced by the active city's
    config exist on disk before load_data() runs. Surfaces missing files
    with a clear st.error() + st.stop() instead of a cryptic rasterio /
    geopandas exception 50 lines deep in the data-loading pipeline.

    Files with graceful-degradation fallbacks elsewhere (population, OSM
    roads / buildings, ET, tracts, energy table, dense scenarios CSV,
    damage table) are NOT in the required list — they have try/except
    paths that disable the corresponding metric or feature when missing.
    """
    missing = []

    # Tables that resolve via _resolve_table (try city dir first, then
    # the project-shared data/flood or data/cooling fallback).
    for key, fallback_dir in [
        ("cn_table_file",      "data/flood"),
        ("cooling_table_file", "data/cooling"),
    ]:
        fname = city_cfg.get(key)
        if not fname:
            missing.append(f"`{key}` is not configured for {city_name}")
            continue
        candidates = [
            f"{city_cfg['data_dir_flood']}/{fname}",
            f"{city_cfg['data_dir_cooling']}/{fname}",
            f"{fallback_dir}/{fname}",
        ]
        if not any(Path(p).exists() for p in candidates):
            missing.append(f"`{key}` ({fname}): tried {candidates}")

    # Direct-path or dir+file rasters that load_data() opens unconditionally.
    for key, base_key in [
        ("lulc_file",         "data_dir_flood"),
        ("soil_file",         "data_dir_flood"),
        ("cooling_lulc_file", "data_dir_cooling"),
    ]:
        fname = city_cfg.get(key)
        base  = city_cfg.get(base_key)
        if not (fname and base):
            missing.append(f"`{key}` or `{base_key}` is not configured")
            continue
        path = Path(f"{base}/{fname}").resolve()
        if not path.exists():
            missing.append(f"`{key}` resolves to a missing file: {base}/{fname}")

    # The InVEST UNA biophysical table is required (no graceful fallback;
    # `calculate_nature_access` reads it at module load).
    una = city_cfg.get("una_table_file")
    if not una or not Path(una).exists():
        missing.append(f"`una_table_file` missing: {una}")

    if missing:
        st.error(
            f"**Cannot load {city_name}** — required input files are missing.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nFix the paths in `CITIES['{city}']` or run the corresponding "
              f"`download_*` / `process_*` script. The selected city is marked "
              f"`available=True` but the loader can't find what it needs."
        )
        st.stop()


_preflight_data_check(city_cfg, selected_city)


# Runtime constants derived from selected city — functions reference these as globals
DATA_DIR_FLOOD     = city_cfg['data_dir_flood']
DATA_DIR_COOLING   = city_cfg['data_dir_cooling']
CN_TABLE_FILE      = city_cfg['cn_table_file']
COOLING_TABLE_FILE = city_cfg['cooling_table_file']
LULC_FILE          = city_cfg['lulc_file']
SOIL_FILE          = city_cfg['soil_file']
COOLING_LULC_FILE  = city_cfg['cooling_lulc_file']
CITY_CRS           = city_cfg['crs']
REF_SCENARIOS      = city_cfg['ref_scenarios']
# NOTE: BASELINE_HM and BASELINE_CN are NOT initialised here from city_cfg.
# The city runtime state (_CURRENT_CITY_STATE) is the single source of truth
# for these scalars — they are computed live from the unmodified LULC raster
# inside _load_city_runtime_state and read as `_CURRENT_CITY_STATE.baseline_hm`
# / `.baseline_cn` everywhere downstream. Keeping them off the module-global
# fast-path prevents silent staleness when switching cities mid-session.


# ── City runtime state container ──────────────────────────────────────────────
# All large per-city allocations (rasters, distance fields, baseline images)
# live in this immutable NamedTuple. Built once per (city, session) by
# `_load_city_runtime_state`, which is decorated with @st.cache_resource so the
# heavy work survives Streamlit reruns instead of re-allocating on every widget
# interaction. Module-level globals named in `_apply_state_to_module_globals`
# are aliased to the matching state members on each rerun for backward compat
# with existing function bodies that read them as bare names — those aliases
# are pointer rebindings, not copies, so they cost nothing.
class CityState(NamedTuple):
    # From load_data() (already @st.cache_data)
    lulc: np.ndarray
    soil_resized: np.ndarray
    cooling_lulc: np.ndarray
    developed_pixels: np.ndarray
    cn_table: np.ndarray
    lucode_idx_arr: np.ndarray
    hm_arr: np.ndarray
    max_raster_lucode: int
    max_hm_lucode: int
    equity_weights: np.ndarray
    shade_arr: np.ndarray
    kc_arr: np.ndarray
    albedo_arr: np.ndarray
    green_area_arr: np.ndarray
    # Population
    pop_count_raster: np.ndarray
    population_data_available: bool
    # Reference ET
    et_resized: np.ndarray
    max_et_ref: float
    et_data_available: bool
    # Energy table (small dict)
    energy_by_type: dict
    energy_table_available: bool
    # Rasterization template
    ref_shape: tuple
    ref_transform: Any
    # Buildings
    buildings_raster: np.ndarray
    buildings_type_raster: np.ndarray
    buildings_data_available: bool
    buildings_have_types: bool
    buildings_type_coverage: float  # 0..1, fraction of building pixels with InVEST type > 0
    total_potential_damage_usd: float
    # Roads
    roads_raster: np.ndarray
    osm_roads_available: bool
    # OSM placement-mask buildings (supplements the typed buildings_file)
    osm_buildings_available: bool
    # Per-pixel AC consumption rate
    consumption_rate_per_pixel: np.ndarray
    # Convertible-pixel pool (developed minus buildings/roads)
    convertible_pixels: np.ndarray
    # Tracts
    tracts: pd.DataFrame
    tract_id_raster: np.ndarray
    tracts_data_available: bool
    # Baseline rasters
    baseline_access_score_raster: np.ndarray
    baseline_hm_raster: np.ndarray
    baseline_ne_raster: np.ndarray
    # Baseline scalars — read via _CURRENT_CITY_STATE only (not aliased)
    baseline_hm: float
    baseline_cn: float


# Module-level escape-hatch handle to the active city runtime state. Populated
# by the call to `_load_city_runtime_state(selected_city)` further down. Used
# sparingly — the canonical access path in newly-written code is the `state`
# parameter or the matching module-level alias rebinding. Two scalars
# (BASELINE_HM, BASELINE_CN) are accessed via this handle by design (see
# CityState comment above).
_CURRENT_CITY_STATE: Optional[CityState] = None

st.markdown(
    "Explore how converting developed land into green infrastructure or food forests "
    "affects **flood damage risk**, **urban cooling costs**, **food production**, "
    "**nature access**, **carbon sequestration**, and **mental-health proxy outcomes** across the city — translating "
    "ecological changes into concrete impacts for planners and decision-makers."
)
st.markdown(
    '- **Green Infrastructure (wetlands)** — best for flood  \n'
    '- **Food Forest** — best for cooling + food  \n'
    '- **High Density** — worst for ecological and nature-access outcomes  \n'
)

with st.expander("How this prototype works", expanded=False):
    st.markdown(
        "**Green Infrastructure** converts developed land to woody wetlands "
        "(NLCD code 90) — best for flood retention.  \n"
        "**Food Forest** converts to deciduous forest (NLCD code 41, used as a "
        "food production proxy) — best for cooling and food.  \n"
        "**High Density** adds impervious development — worst for ecological and nature-access outcomes.  \n"
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
    st.markdown(
        "**Confidence tiers** — each metric card displays one of three badges "
        "under its value:  \n"
        "  \n"
        "- **High confidence** — Direct raster outputs grounded in published "
        "methodology (USDA SCS curve numbers, InVEST UCM Cooling Capacity). "
        "Numerical value reflects pixel-level simulation; uncertainty is in "
        "the input data, not the method.  \n"
        "  \n"
        "- **Medium confidence** — Model-based estimates or order-of-magnitude "
        "calculations with empirical grounding. Includes InVEST UMH "
        "preventable cases (peer-reviewed effect sizes, synthetic NDVI input), "
        "canonical InVEST Urban Nature Access (2SFCA), and $-valued cards backed by "
        "per-building or per-acre lookup tables. Use for directional planning; "
        "verify with locally calibrated data for final decisions.  \n"
        "  \n"
        "- **Prototype** — Synthetic proxies or unvalidated assumptions. "
        "Includes the NDVI raster (assigned per NLCD class, not "
        "satellite-derived), carbon sequestration rates (regional benchmarks, "
        "not site-calibrated), and food forest yield (benchmark for mature "
        "managed systems). Treat as directional only; not suitable for "
        "site-specific or quantitative decisions.  \n"
        "  \n"
        "A tier reflects the *method's* confidence, not whether the number is "
        "large or small. A Prototype-badged card showing a precise number is "
        "still a prototype number."
    )

# ── Data loading ───────────────────────────────────────────────────────────────
def _resolve_table(data_dir, filename, *fallback_dirs):
    """Try `data_dir/filename` first, fall back to each `fallback_dirs/filename`.
    Used so cities pointing at custom data_dirs (e.g. data/minneapolis_expanded)
    can still reference the project-shared biophysical tables in data/flood
    or data/cooling without a copy. Raises FileNotFoundError if none match."""
    candidates = [f'{data_dir}/{filename}'] + [f'{d}/{filename}' for d in fallback_dirs]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"could not find {filename}; tried: {candidates}")


@st.cache_data
def load_data(data_dir_flood, data_dir_cooling, cn_table_file, cooling_table_file,
              lulc_file, soil_file, cooling_lulc_file):
    bio = pd.read_csv(_resolve_table(data_dir_flood, cn_table_file, "data/flood"))

    with rasterio.open(f'{data_dir_flood}/{lulc_file}') as src:
        lulc = src.read(1)
    with rasterio.open(f'{data_dir_flood}/{soil_file}') as src:
        soil = src.read(1)

    cooling_bio = pd.read_csv(_resolve_table(data_dir_cooling, cooling_table_file, "data/cooling"))
    with rasterio.open(f'{data_dir_cooling}/{cooling_lulc_file}') as src:
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
    # built per pixel from the ET raster and Kc (see _compute_cc_raw_pure).
    # `green_area_arr` is the per-class green-area flag (0/1) the InVEST UCM
    # HMI uses to source park cooling. We also keep a derived `hm_arr`
    # (= the simplified `(shade + kc) / 2`) for any legacy paths that
    # reference it; the live CC pipeline supersedes it everywhere that matters.
    max_hm_lucode = int(cooling_bio['lucode'].max())
    shade_arr      = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    kc_arr         = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    albedo_arr     = np.full(max_hm_lucode + 1, np.nan, dtype=np.float32)
    green_area_arr = np.zeros(max_hm_lucode + 1, dtype=np.float32)
    for _, row in cooling_bio.iterrows():
        lc = int(row['lucode'])
        shade_arr[lc]      = row['shade']
        kc_arr[lc]         = row['kc']
        albedo_arr[lc]     = row['albedo']
        green_area_arr[lc] = row['green_area']
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
            equity_weights, shade_arr, kc_arr, albedo_arr, green_area_arr)


# ── Population raster loader (for Nature Access metric) ──────────────────────
# Helper used by `_load_city_runtime_state` below. Built offline by
# download_census_pop.py from US Census 2020 block-level totals, rasterized to
# the NLCD grid. The loader falls back to a uniform placeholder if the file is
# missing so the app still launches before the pipeline has run.
def load_population_data(pop_path, target_shape):
    """Load a population-count raster, resampled to target_shape with bilinear."""
    with rasterio.open(pop_path) as src:
        data = src.read(
            1, out_shape=target_shape,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        data = data.astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = 0
        data[data < 0] = 0
        return data

# Cost-per-kWh (US average residential, EIA 2024). Used to convert
# avoided-AC-kWh into $.
COST_PER_KWH_USD = 0.13

# EPA Social Cost of Carbon — central estimate, 2 % discount rate, 2030
# emissions, EPA 2023 final rule "Methodology for Estimating the Social
# Cost of Greenhouse Gases" (Nov 2023). Multiplied by `carbon_tons_co2_yr`
# to get an "avoided-damage" dollar value at the federal-guideline rate.
# This is a deterministic linear function of carbon, so it's NOT added to
# REQUIRED_TARGET_COLUMNS (the surrogate already learns carbon; we
# multiply by this constant post-hoc).
EPA_SOCIAL_COST_CARBON = 190

# NOTE: there used to be an `AC_KWH_PER_DEG_F = 0.03` fractional-AC-sensitivity
# constant here, applied as an extra multiplier in the energy-savings formula.
# It has been removed: the InVEST UCM `consumption` column is documented as
# kWh/(m²·°C), i.e. the per-degree response is already encoded in the rate.
# Multiplying by an additional 0.03 fraction would double-count. See
# `data/invest/cooling/UCM_AUDIT.md` for the full reasoning.


# ── InVEST Urban Cooling Model — canonical Heat Mitigation Index (HMI) ───────
# HMI = max(CC_local, CC_park): a pixel's heat-mitigation value is its own
# cooling capacity, lifted to the distance-weighted park cooling CC_park when
# the pixel is within reach of enough green space. This is a faithful port of
# natcap.invest.urban_cooling_model's pipeline:
#   calc_cc_op_factors      → per-pixel CC          (_compute_cc_raw_pure)
#   mask_cc_green_areas_op  → CC kept only in green pixels
#   convolve_2d_by_exponential → CC_park            (_compute_cc_park_raster)
#   dichotomous convolution → green_area_sum        (_compute_green_area_sum)
#   hm_op                   → HMI = max where eligible  (_compute_hmi_raster*)
# Parameters are hardcoded at InVEST canonical values — not user-configurable.
GREEN_AREA_COOLING_DISTANCE_M = 450        # InVEST `green_area_cooling_distance`
_HMI_PIXEL_SIZE_M = 30                     # NLCD grid resolution
# Decay distance in pixels — InVEST: int(round(d_cool / cell_size)).
_HMI_DECAY_PX = int(round(GREEN_AREA_COOLING_DISTANCE_M / _HMI_PIXEL_SIZE_M))   # 15
# The 2-hectare green-area trigger, as a pixel count — InVEST: 2e4 / cell_size².
_HMI_GREEN_THRESHOLD_PX = 2e4 / _HMI_PIXEL_SIZE_M ** 2                          # 22.22


def _build_hmi_kernels():
    """Construct the two InVEST UCM convolution kernels, matching the geometry
    of pygeoprocessing.kernels._create_distance_kernel (square, side
    2·floor(max_dist)+1, euclidean distance measured from the centre pixel).

    - Exponential decay kernel (CC_park): exp(-d / decay_px), truncated to 0
      beyond 5·decay_px, then normalized to sum 1 — mirrors
      convolve_2d_by_exponential, which calls exponential_decay_kernel with
      ``normalize`` defaulting to True.
    - Dichotomous kernel (green_area_sum): 1 within decay_px, 0 outside, left
      unnormalized — InVEST passes ``normalize=False`` for the area kernel.
    """
    def _dist(max_px):
        apothem = int(np.floor(max_px))
        coords = np.arange(-apothem, apothem + 1, dtype=np.float64)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        return np.hypot(yy, xx)

    exp_dist = _dist(_HMI_DECAY_PX * 5)
    exp_k = np.exp(-exp_dist / _HMI_DECAY_PX)
    exp_k[exp_dist > _HMI_DECAY_PX * 5] = 0.0
    exp_k /= exp_k.sum()

    dich_dist = _dist(_HMI_DECAY_PX)
    dich_k = (dich_dist <= _HMI_DECAY_PX).astype(np.float64)
    return exp_k, dich_k


_HMI_EXP_KERNEL, _HMI_DICH_KERNEL = _build_hmi_kernels()
_HMI_EXP_KERNEL_SUM = float(_HMI_EXP_KERNEL.sum())
_HMI_DICH_KERNEL_SUM = float(_HMI_DICH_KERNEL.sum())


def _compute_cc_raw_pure(scenario_lulc, shade_arr, kc_arr, albedo_arr, et_resized, max_et_ref):
    """Per-pixel raw cooling-capacity index — no spatial propagation:
        CC_i  = 0.6·shade_i + 0.2·albedo_i + 0.2·ETI_i
        ETI_i = (kc_i × ET_ref_i) / max(ET_ref)
    Matches InVEST's `calc_cc_op_factors`. Returns float32, NaN where the LULC
    code is nodata or off the biophysical table. Pure variant — all per-city
    deps explicit, so `_load_city_runtime_state` can call it before the
    module-level aliases for `shade_arr`, `kc_arr`, etc. are rebound.

    Memory-tight: peak transient is ~3 full-AOI float32 buffers (cc + tmp +
    transient indexing result) plus one int32 safe-index array."""
    safe = np.clip(scenario_lulc, 0, len(shade_arr) - 1)
    # Build cc in-place: cc = 0.6*shade + 0.2*albedo + 0.2*eti.
    # Fancy indexing returns fresh arrays, so cc and tmp are uniquely owned
    # and safe to mutate.
    cc = shade_arr[safe]
    cc *= 0.6
    tmp = albedo_arr[safe]
    tmp *= 0.2
    cc += tmp
    np.multiply(kc_arr[safe], et_resized, out=tmp)   # tmp = kc * et_resized
    tmp /= max_et_ref                                 # tmp = eti
    tmp *= 0.2                                        # tmp = 0.2 * eti
    cc += tmp
    del tmp
    if cc.dtype != np.float32:
        cc = cc.astype(np.float32)
    nan_mask = (scenario_lulc < 0) | (scenario_lulc >= len(shade_arr)) | ~np.isfinite(cc)
    cc[nan_mask] = np.nan
    return cc


def _convolve_edge_corrected(signal, kernel, valid_mask, kernel_sum):
    """Linear convolution normalized for nodata and AOI edges — a numpy port
    of pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True,
    normalize_kernel=False), which InVEST UCM uses for both CC_park and
    green_area_sum.

    `signal` must be float64 with nodata cells already zeroed; `valid_mask` is
    True where the signal is valid. The raw convolution is divided by the
    kernel weight that actually overlapped valid data — so edge and nodata
    pixels are not artificially darkened — then rescaled by `kernel_sum`. With
    a sum-1 (normalized) kernel this yields a weighted average; with an
    unnormalized kernel, a true weighted sum. Output is 0 outside
    `valid_mask`."""
    numer = _fftconvolve(signal, kernel, mode="same")
    denom = _fftconvolve(valid_mask.astype(np.float64), kernel, mode="same")
    out = np.zeros_like(numer)
    ok = valid_mask & (denom > 1e-12)
    out[ok] = numer[ok] / denom[ok] * kernel_sum
    return out


def _compute_cc_park_raster(cc_raster, green_mask, valid_mask):
    """CC_park — the exponentially distance-weighted CC sourced from green
    space. Mirrors InVEST's mask_cc_green_areas_op (CC retained inside green
    pixels, zeroed elsewhere) followed by convolve_2d_by_exponential."""
    cc_masked_green = np.where(green_mask, cc_raster, 0.0).astype(np.float64)
    return _convolve_edge_corrected(
        cc_masked_green, _HMI_EXP_KERNEL, valid_mask, _HMI_EXP_KERNEL_SUM)


def _compute_green_area_sum(green_mask, valid_mask):
    """green_area_sum — the edge-corrected count of green pixels within the
    cooling distance of each pixel (InVEST's dichotomous-kernel convolution of
    the green-area reclassification). Compared against the 2-hectare threshold
    to decide whether a pixel is eligible for park cooling."""
    return _convolve_edge_corrected(
        green_mask.astype(np.float64), _HMI_DICH_KERNEL, valid_mask,
        _HMI_DICH_KERNEL_SUM)


def _compute_hmi_raster_pure(scenario_lulc, shade_arr, kc_arr, albedo_arr,
                             et_resized, max_et_ref, green_area_arr):
    """Canonical InVEST UCM Heat Mitigation Index for a scenario LULC.

    HMI = CC_park where a pixel holds ≥ 2 ha of green space within the cooling
    distance AND CC_park exceeds the pixel's local CC; otherwise HMI = local
    CC. This is InVEST's `hm_op`. NaN where the LULC code is nodata or
    off-table. Pure variant — per-city deps explicit, called by
    `_load_city_runtime_state` for the baseline raster before module aliases
    are bound."""
    cc = _compute_cc_raw_pure(scenario_lulc, shade_arr, kc_arr, albedo_arr,
                              et_resized, max_et_ref)
    valid = np.isfinite(cc)
    safe = np.clip(scenario_lulc, 0, len(green_area_arr) - 1)
    green_mask = valid & (green_area_arr[safe] > 0)

    cc64 = np.where(valid, cc, 0.0).astype(np.float64)
    cc_park = _compute_cc_park_raster(cc64, green_mask, valid)
    green_area_sum = _compute_green_area_sum(green_mask, valid)

    # hm_op: take CC_park only where it cools more AND enough green is in reach.
    use_park = valid & (cc_park > cc64) & (green_area_sum >= _HMI_GREEN_THRESHOLD_PX)
    hmi = np.where(use_park, cc_park, cc64).astype(np.float32)
    hmi[~valid] = np.nan
    return hmi


def _compute_hmi_raster(scenario_lulc):
    """Canonical InVEST UCM Heat Mitigation Index — wrapper that pulls
    per-city deps from the module-level aliases populated by
    `_load_city_runtime_state`."""
    return _compute_hmi_raster_pure(
        scenario_lulc, shade_arr, kc_arr, albedo_arr, ET_RESIZED, MAX_ET_REF,
        green_area_arr,
    )


# Nature Access: weighted population-share metric using the official InVEST
# Urban Nature Access (UNA) biophysical table. Each LULC class has its own
# `urban_nature` score (0–1) and `search_radius_m`. For each scenario we take,
# per pixel, the maximum (in_range × score) across all natural classes — a
# pixel "near" multiple nature types gets the highest of their scores — then
# weight population by that score. Replaces the earlier hardcoded
# `NATURE_CODES = [41, 42, 43, 52, 71, 90, 95]` + single-800m-radius approach.
PIXEL_SIZE_M = 30

# Per-city UNA biophysical table path is small + cheap, so it stays
# module-level (kept off CityState to avoid threading a Path through
# functions that don't otherwise need state).
UNA_TABLE_PATH = Path(city_cfg["una_table_file"])

# Cap on the InVEST UNA `search_radius_m` field. Defaults of 5000 m for
# water / forest / wetland classes treat those as "regional" amenities —
# appropriate for a county-scale recreation study, but for an urban walking-
# distance access metric a single water pixel would mark essentially every
# other pixel as "has nature access" (the 100 % baseline bug). 1000 m =
# ~12-minute walk, matches InVEST's own value for "Developed, Open Space".
NATURE_RADIUS_CAP_M = 1000

# Pre-compute distance transforms for natural classes whose pixel set never
# changes across scenarios (the model only converts NLCD 21–24 to GI/FF/HD).
# Those three lucodes are recomputed live; all other natural classes use the
# pre-built array stored on the city runtime state.
_DYNAMIC_NATURE_LUCODES = {21, 41, 90}


def _compute_access_score_raster_pure(scenario_lulc, una_active, precomputed_nature_distances):
    """Pure variant — explicit deps. Used by `_load_city_runtime_state` to
    compute the baseline access-score raster before module aliases exist."""
    access_score = np.zeros(scenario_lulc.shape, dtype=np.float32)
    for _, row in una_active.iterrows():
        lucode = int(row["lucode"])
        radius = float(row["search_radius_m"])
        score  = float(row["urban_nature"])
        if lucode in precomputed_nature_distances:
            distance = precomputed_nature_distances[lucode]
        else:
            mask = (scenario_lulc == lucode)
            if not mask.any():
                continue
            # Cast to float32 immediately — `_distance_transform_edt` returns
            # float64, doubling the per-call transient on a hot path that runs
            # ~3 times per scenario for the dynamic lucodes (21/41/90).
            distance = (_distance_transform_edt(~mask) * PIXEL_SIZE_M).astype(np.float32, copy=False)
        in_range = distance <= radius
        np.maximum(access_score, in_range * score, out=access_score)
    return access_score


# ── Canonical InVEST Urban Nature Access (UNA) — numpy 2SFCA ─────────────────
# Re-implements natcap.invest.urban_nature_access (uniform search radius,
# dichotomy decay) in numpy — the same approach `_compute_hmi_raster` takes for
# the InVEST UCM. The model runs inside the app's own environment (no
# natcap.invest runtime dependency); the numpy result is validated offline
# against `natcap.invest.urban_nature_access.execute()`. Parameter rationale is
# in DESIGN_NOTES.md.
UNA_DEMAND_M2_PER_CAPITA = 16.7   # per-capita supply standard (NatCap SA study)
UNA_SEARCH_RADIUS_M      = 800    # uniform search radius, ~10-min walk

# urban_nature proportion (0-1 of pixel area) per LULC code, from the InVEST
# UNA biophysical table. Codes absent from the table contribute no nature.
URBAN_NATURE_PROPORTION = {
    int(r.lucode): float(r.urban_nature)
    for r in pd.read_csv(UNA_TABLE_PATH).itertuples()
}

# Dichotomy-decay kernel: a binary disk of radius `search_radius / pixel_size`
# pixels, built exactly as pygeoprocessing.kernels.dichotomous_kernel does
# (apothem = floor(radius_px); kernel side = 2·apothem + 1; a pixel is 1 where
# its euclidean distance from the centre <= radius_px, else 0; un-normalized).
_UNA_RADIUS_PX = UNA_SEARCH_RADIUS_M / PIXEL_SIZE_M
_UNA_APOTHEM   = int(np.floor(_UNA_RADIUS_PX))
_una_yy, _una_xx = np.mgrid[
    -_UNA_APOTHEM:_UNA_APOTHEM + 1, -_UNA_APOTHEM:_UNA_APOTHEM + 1]
_UNA_KERNEL = (np.hypot(_una_yy, _una_xx) <= _UNA_RADIUS_PX).astype(np.float32)
del _una_yy, _una_xx


def _una_convolve(signal):
    """Zero-padded 2-D convolution with the dichotomy disk kernel, matching
    `pygeoprocessing.convolve_2d(ignore_nodata_and_edges=False)` as InVEST UNA
    uses it: edges are zero-padded (not edge-corrected), then the negative-value
    clamp of InVEST's `_convolve_and_set_lower_bound` is applied."""
    out = _fftconvolve(signal, _UNA_KERNEL, mode="same")
    np.clip(out, 0.0, None, out=out)
    return out


def _una_supply_percapita(scenario_lulc, pop_count_raster):
    """InVEST UNA `urban_nature_supply_percapita` raster via two-step floating
    catchment area (2SFCA), re-implemented in numpy.

    Returns `(supply_percapita, valid_mask)`. `supply_percapita` is m² of urban
    nature available per capita reachable from each pixel; `valid_mask` is the
    modelable extent (valid-LULC pixels — InVEST masks LULC and population to
    their common valid extent before convolving)."""
    valid = (scenario_lulc != NODATA)
    pixel_area_m2 = float(PIXEL_SIZE_M * PIXEL_SIZE_M)

    # Population masked to the modelable extent; off-extent population counts as
    # 0, exactly as InVEST's `masked_population` feeds the convolution.
    pop = np.where(valid, np.asarray(pop_count_raster, dtype=np.float64), 0.0)

    # Urban-nature area per pixel = urban_nature_proportion × pixel area.
    nature_area = np.zeros(scenario_lulc.shape, dtype=np.float64)
    for lucode, proportion in URBAN_NATURE_PROPORTION.items():
        if proportion > 0:
            nature_area[scenario_lulc == lucode] = proportion * pixel_area_m2

    # 2SFCA step 1 — decay-weighted population within the search radius.
    decayed_pop = _una_convolve(pop)

    # 2SFCA step 1b — R_j, the urban-nature/population ratio. Mirrors InVEST's
    # `_urban_nature_population_ratio`: nature_area / decayed_pop, except where
    # the reachable population is <= 1 person the ratio is set to nature_area
    # (InVEST science-team rule — avoids a divide-by-near-zero blow-up).
    nature_pixels = nature_area > 0
    ratio = np.zeros(scenario_lulc.shape, dtype=np.float64)
    pop_le_one = decayed_pop <= 1.0
    sel = nature_pixels & pop_le_one
    ratio[sel] = nature_area[sel]
    sel = nature_pixels & ~pop_le_one
    ratio[sel] = nature_area[sel] / decayed_pop[sel]

    # 2SFCA step 2 — supply per capita = decay-weighted sum of R_j.
    return _una_convolve(ratio), valid


def _invest_una_pct_pop_supply_ge_demand(scenario_lulc, pop_count_raster):
    """Headline UNA metric: the share of the modelable-extent population whose
    per-capita urban-nature supply meets `UNA_DEMAND_M2_PER_CAPITA`.

    Returns `(pct, modelable_pop, people_supplied)`. The modelable extent is the
    population on valid-LULC pixels; InVEST cannot model supply for residents on
    LULC nodata (a large share of the prototype's downtown MN AOI)."""
    supply_percapita, valid = _una_supply_percapita(
        scenario_lulc, pop_count_raster)
    pop = np.asarray(pop_count_raster, dtype=np.float64)
    modelable_pop = float(pop[valid].sum())
    if modelable_pop <= 0:
        return 0.0, 0.0, 0.0
    adequate = valid & (supply_percapita >= UNA_DEMAND_M2_PER_CAPITA)
    people_supplied = float(pop[adequate].sum())
    return 100.0 * people_supplied / modelable_pop, modelable_pop, people_supplied


def calculate_nature_access(scenario_lulc, pop_count_raster):
    """Canonical InVEST Urban Nature Access for the given scenario LULC.

    Re-implements `natcap.invest.urban_nature_access` (uniform 800 m search
    radius, dichotomy decay, 16.7 m²/capita demand — see
    DESIGN_NOTES.md) in numpy via two-step floating catchment area
    (2SFCA). The headline metric is `pct_pop_supply_ge_demand`: the share of the
    modelable-extent population whose per-capita nature supply meets the demand
    standard.

    `pop_count_raster` must be per-pixel population **counts** (not density).

    Returns a 3-tuple `(access_pct, _legacy_slot, people_with_access)`:

    - `access_pct` — pct_pop_supply_ge_demand, 0-100, rounded to 0.1.
    - `_legacy_slot` — always 0.0. Formerly the Nature Quality Score (removed);
      the slot is retained so existing three-value call sites are unaffected.
    - `people_with_access` — integer headcount, access_pct/100 × modelable-
      extent population.
    """
    pct, _modelable_pop, people_supplied = _invest_una_pct_pop_supply_ge_demand(
        scenario_lulc, pop_count_raster
    )
    return round(float(pct), 1), 0.0, int(round(people_supplied))


# Baseline food production is zero by definition (no conversions means no
# food forest).
BASELINE_FOOD_MLN_LBS = 0.0


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


def _lulc_to_ndvi_raster(lulc_array):
    """Per-pixel NDVI proxy raster (same shape as lulc, dtype float32). Used
    by compute_mean_ndvi (which then takes the mean) and by the InVEST UMH
    pipeline (which Gaussian-smooths it within a 300 m search radius).
    Pixels with NODATA become NDVI_OTHER_NATURAL (a benign default — the UMH
    delta zeros out at NODATA pixels because both baseline and scenario use
    the same fill there)."""
    ndvi_map = np.full(lulc_array.shape, NDVI_OTHER_NATURAL, dtype=np.float32)
    for code in _DEVELOPED_ALL:
        ndvi_map[lulc_array == code] = NDVI_OTHER_DEVELOPED
    for code, val in NDVI_PROXY.items():
        ndvi_map[lulc_array == code] = val
    return ndvi_map


def compute_mean_ndvi(lulc_array, ndvi_raster=None):
    """Area-weighted mean NDVI proxy across all valid (non-NODATA) pixels.
    Pass `ndvi_raster=` to reuse a precomputed raster (saves the per-call
    allocation when the caller already has one in hand)."""
    valid_mask = lulc_array != NODATA
    if not valid_mask.any():
        return float('nan')
    if ndvi_raster is None:
        ndvi_raster = _lulc_to_ndvi_raster(lulc_array)
    return float(round(ndvi_raster[valid_mask].mean(), 4))


# ── InVEST Urban Mental Health Model (v3.19.0) ────────────────────────────────
# Implements the canonical InVEST UMH preventable-cases formula:
#   NE_i = gaussian_filter(NDVI_i, sigma=search_radius/pixel_size)  per-pixel exposure
#   ΔNE_i = NE_scenario_i − NE_baseline_i
#   RR_i = exp( ln(RR_0.1) × 10 × ΔNE_i )               relative risk
#   PF_i = 1 − RR_i                                     preventable fraction
#   PC_i = PF_i × BIR × population_i                    preventable cases
#   $    = Σ PC_i × cost_per_case
#
# Constants below are the user-supplied defaults at the time of integration:
#   RR per 0.1 NDVI from Liu et al. 2023 meta-analysis (the InVEST UMH
#   reference) — depression 0.96 (4 % reduction per 0.1 NDVI), anxiety 0.97.
#   Baseline incidence/prevalence from CDC 2023; cost-of-illness figures are
#   plausible mid-range values (InVEST docs cite $11,000 USD-PPP/case as a
#   default — our values are slightly lower, US-only nominal). All values
#   should be replaced with locally-calibrated numbers for production work.
RR_0_1_NDVI_DEPRESSION       = 0.96
RR_0_1_NDVI_ANXIETY          = 0.97
BIR_DEPRESSION               = 0.21
BIR_ANXIETY                  = 0.19
COST_PER_DEPRESSION_CASE_USD = 8467
COST_PER_ANXIETY_CASE_USD    = 5765
_MH_CASES_PILL_EPSILON       = 1     # cases threshold for pill suppression
_MH_COST_PILL_EPSILON        = 1000  # USD threshold for pill suppression
UMH_SEARCH_RADIUS_M          = 300   # Li et al. 2025; ~10 px at 30 m NLCD

# InVEST UMH uses Gaussian-smoothed NDVI exposure within the search radius.
# `sigma_pixels = UMH_SEARCH_RADIUS_M / PIXEL_SIZE_M` matches the canonical
# InVEST behavior (search radius interpreted as kernel σ).
_UMH_SIGMA_PX = UMH_SEARCH_RADIUS_M / PIXEL_SIZE_M     # = 10.0 at 30 m / 300 m
_UMH_LN_RR_DEPRESSION = float(np.log(RR_0_1_NDVI_DEPRESSION))
_UMH_LN_RR_ANXIETY    = float(np.log(RR_0_1_NDVI_ANXIETY))


def calculate_mental_health_impact(scenario_lulc, baseline_ne_raster, pop_count, ndvi_raster=None):
    """Return (preventable_mh_cases, avoided_mh_cost_usd) for the scenario.

    `baseline_ne_raster` is the smoothed NE raster for the unmodified LULC
    (precomputed once at module load — see _BASELINE_NE_RASTER below). We
    compute the scenario-side NE on the fly, take ΔNE, apply the InVEST UMH
    formula per pixel, and sum population-weighted preventable cases. Returns
    (0.0, 0.0) if the population raster isn't loaded — there's nothing to
    weight by.

    Pass `ndvi_raster=` to reuse a precomputed scenario NDVI raster (saves
    one full-AOI allocation when `evaluate_scenario` already built one for
    `compute_mean_ndvi`)."""
    if not POPULATION_DATA_AVAILABLE:
        return 0.0, 0.0
    if ndvi_raster is None:
        ndvi_raster = _lulc_to_ndvi_raster(scenario_lulc)
    ne_scenario = _gaussian_filter(
        ndvi_raster, sigma=_UMH_SIGMA_PX, mode="nearest"
    )
    delta_ne = ne_scenario - baseline_ne_raster

    rr_dep = np.exp(_UMH_LN_RR_DEPRESSION * 10 * delta_ne)
    rr_anx = np.exp(_UMH_LN_RR_ANXIETY    * 10 * delta_ne)
    pf_dep = 1.0 - rr_dep
    pf_anx = 1.0 - rr_anx

    pc_dep = pf_dep * BIR_DEPRESSION * pop_count
    pc_anx = pf_anx * BIR_ANXIETY    * pop_count
    total_pc = float((pc_dep + pc_anx).sum())
    avoided_cost = float((
        pc_dep * COST_PER_DEPRESSION_CASE_USD
        + pc_anx * COST_PER_ANXIETY_CASE_USD
    ).sum())
    return round(total_pc, 1), round(avoided_cost, 0)


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
    # read from state to avoid silent-staleness if city switches
    delta_hm = mean_hm - _CURRENT_CITY_STATE.baseline_hm
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


# ── Placement strategies ──────────────────────────────────────────────────────
# Five named strategies for selecting which convertible pixels to convert.
# 'random' is the default and reproduces the prior uniform-sampling behavior.
# The others weight the sampling toward pixels where conversion yields the
# highest benefit per the INVEST_PLACEMENT.md research. UI exposure is deferred
# to a future session; for now these are Python-API-only.
PLACEMENT_STRATEGIES = {
    'random':          'Uniform random sampling',
    'flood-focused':   'Prioritize pixels with highest runoff reduction potential',
    'cooling-focused': 'Prioritize pixels in hot areas near buildings',
    'equity-focused':  'Prioritize pixels in nature-deficit areas with high population',
    'balanced':        'Weighted combination of flood, cooling, and equity signals',
}

# Human-readable labels for the sidebar radio. Keys must match PLACEMENT_STRATEGIES.
PLACEMENT_STRATEGY_LABELS = {
    'random':          'Random placement',
    'flood-focused':   'Prioritize flood-prone areas',
    'cooling-focused': 'Prioritize hot areas near buildings',
    'equity-focused':  'Prioritize underserved areas',
    'balanced':        'Balanced approach',
}


def _compute_suitability_weights(convertible_pixels, strategy):
    """Compute per-pixel suitability weights for the given strategy.

    Returns a 1-D array (same length as convertible_pixels) of non-negative
    weights. Higher = more suitable for conversion. The caller normalizes
    to a probability distribution before sampling.

    Each strategy combines one or more module-level rasters evaluated at
    the convertible-pixel coordinates. If a required raster is unavailable,
    the component falls back to uniform (ones).
    """
    rows = convertible_pixels[:, 0]
    cols = convertible_pixels[:, 1]
    n = len(convertible_pixels)

    if strategy == 'flood-focused':
        # Higher baseline CN = more runoff from this pixel = more benefit
        # from converting it to GI/FF. Extract per-pixel CN from the
        # baseline LULC using the same lookup as evaluate_scenario.
        lulc_vals = np.clip(cooling_lulc[rows, cols], 0, len(lucode_idx_arr) - 1)
        soil_vals = np.clip(soil_resized[rows, cols].astype(int), 1, cn_table.shape[1] - 1)
        pixel_cn = cn_table[lucode_idx_arr[lulc_vals], soil_vals].astype(np.float64)
        # CN ranges ~30-98; use it directly as the weight (higher CN = higher priority).
        weights = np.maximum(pixel_cn, 0.0)

    elif strategy == 'cooling-focused':
        # Two signals: (a) heat exposure = 1 - baseline CC (hotter pixels
        # benefit more from greening), and (b) building proximity (pixels
        # near buildings save more AC energy when cooled).
        # Heat exposure from baseline CC raster (smoothed, 0-1 range).
        heat = 1.0 - _BASELINE_HM_RASTER[rows, cols].astype(np.float64)
        heat = np.maximum(heat, 0.0)

        # Building proximity: pixels ON or adjacent to buildings are high-value.
        # Use the buildings raster directly: 1 where building exists, 0 otherwise.
        # A pixel adjacent to a building is high-value but isn't itself a building
        # (buildings are excluded from CONVERTIBLE_PIXELS). So use a distance-based
        # signal: distance_to_nearest_building, inverted. For efficiency, use the
        # existing equity_weights raster (NLCD 23→1.0, 22→0.6, 21→0.3) as a
        # development-intensity proxy — higher-intensity developed pixels are
        # closer to buildings and commercial areas.
        intensity = equity_weights[rows, cols].astype(np.float64)
        intensity = np.maximum(intensity, 0.0)

        # Combine: multiply heat exposure by development intensity.
        # Both are non-negative; product emphasizes pixels that are both hot
        # and in high-intensity areas.
        weights = heat * (intensity + 0.1)  # +0.1 floor so low-intensity pixels aren't zeroed out

    elif strategy == 'equity-focused':
        # Two signals: (a) population density (more people = more benefit),
        # and (b) nature deficit (1 - access score; underserved pixels need
        # nature more). Product maximizes benefit to underserved populations.
        if POPULATION_DATA_AVAILABLE:
            pop = pop_count_raster[rows, cols].astype(np.float64)
            pop = np.maximum(pop, 0.0)
        else:
            pop = np.ones(n, dtype=np.float64)

        # Nature deficit: 1 - access_score. Pixels far from nature with low
        # quality score get higher weight.
        access = _BASELINE_ACCESS_SCORE_RASTER[rows, cols].astype(np.float64)
        deficit = np.maximum(1.0 - access, 0.0)

        # Combine: population × deficit. A high-pop, low-access pixel is the
        # highest-priority target.
        weights = pop * (deficit + 0.01)  # +0.01 floor so zero-deficit pixels aren't impossible

    elif strategy == 'balanced':
        # Equal-weight combination of the three focused strategies.
        # Normalize each component to sum to 1, then average.
        flood_w = _compute_suitability_weights(convertible_pixels, 'flood-focused')
        cool_w = _compute_suitability_weights(convertible_pixels, 'cooling-focused')
        equity_w = _compute_suitability_weights(convertible_pixels, 'equity-focused')

        def _safe_normalize(w):
            s = w.sum()
            return w / s if s > 0 else np.ones_like(w) / len(w)

        weights = (_safe_normalize(flood_w)
                   + _safe_normalize(cool_w)
                   + _safe_normalize(equity_w)) / 3.0
    else:
        raise ValueError(f"Unknown placement strategy: {strategy!r}. "
                         f"Valid: {list(PLACEMENT_STRATEGIES)}")

    return weights


def _select_pixels_for_conversion(convertible_pixels, n_to_convert, strategy, rng):
    """Select which convertible pixels to convert based on the placement strategy.

    Returns an array of indices into convertible_pixels.
    """
    if n_to_convert <= 0:
        return np.array([], dtype=int)

    if strategy == 'random':
        return rng.choice(len(convertible_pixels), size=n_to_convert, replace=False)

    weights = _compute_suitability_weights(convertible_pixels, strategy)
    weights = np.maximum(weights, 0.0)
    weight_sum = weights.sum()

    if weight_sum > 0:
        weights /= weight_sum
        return rng.choice(len(convertible_pixels), size=n_to_convert, replace=False, p=weights)
    else:
        # Fallback to uniform random if all weights are zero
        return rng.choice(len(convertible_pixels), size=n_to_convert, replace=False)


# ── Scenario evaluation ────────────────────────────────────────────────────────
def evaluate_scenario(pct_converted, green_infrastructure_pct, food_forest_pct,
                      seed=42, use_heat_priority=False,
                      placement_strategy='random',
                      cost_gi=DEFAULT_COST_GI,
                      cost_ff=DEFAULT_COST_FF,
                      cost_hd=DEFAULT_COST_HD,
                      carbon_rate_ff=None,
                      carbon_rate_gi=None):
    """
    Convert a sample of developed pixels to the specified land use mix,
    then compute flood risk, urban cooling, food production, and cost.

    Placement is controlled by `placement_strategy` (default 'random').
    The legacy `use_heat_priority=True` flag is translated to
    `placement_strategy='cooling-focused'` for backward compatibility.
    """
    # Legacy backward compat: use_heat_priority=True maps to cooling-focused.
    # If both are specified, use_heat_priority takes precedence (conservative —
    # existing callers that pass it should get the behavior they expect).
    if use_heat_priority:
        placement_strategy = 'cooling-focused'

    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

    scenario_lulc = cooling_lulc.copy()
    # Sample from the convertible (= developed AND non-building) pool so
    # conversions land on feasible interstitial spaces (parking lots, lawns,
    # vacant land) rather than on top of existing structures. Total developed
    # acreage for runoff baseline scaling still uses the full developed_pixels
    # array — buildings still produce runoff.
    n_convert = int(len(CONVERTIBLE_PIXELS) * pct_converted / 100)

    rng = np.random.default_rng(seed)

    chosen_idx = _select_pixels_for_conversion(
        CONVERTIBLE_PIXELS, n_convert, placement_strategy, rng)

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

    # Canonical InVEST UCM Heat Mitigation Index — HMI = max(CC_local,
    # CC_park). `mean_hm` is the mean HMI across valid pixels (0–1 scale,
    # higher = more cooling); the per-pixel value factors shade, albedo,
    # per-pixel ET, and exponentially distance-weighted cooling from green
    # areas ≥ 2 ha within the 450 m cooling distance.
    hmi_map  = _compute_hmi_raster(scenario_lulc)
    valid_hm = hmi_map[~np.isnan(hmi_map) & (scenario_lulc != NODATA)]
    mean_hm  = float(valid_hm.mean().round(4))
    cooling_energy_savings_usd = compute_cooling_energy_savings(hmi_map)

    n_food_pixels = int(((scenario_lulc == CODE_FOOD_FOREST) & (cooling_lulc != CODE_FOOD_FOREST)).sum())
    food_mln_lbs  = round(n_food_pixels * PIXEL_AREA_ACRES * FOOD_FOREST_LBS_ACRE / 1_000_000, 3)

    rate_ff = CARBON_SEQ_RATES[CODE_FOOD_FOREST] if carbon_rate_ff is None else carbon_rate_ff
    rate_gi = CARBON_SEQ_RATES[CODE_GREEN_INFRA] if carbon_rate_gi is None else carbon_rate_gi
    carbon_tons_co2_yr = round(
        n_for * PIXEL_AREA_ACRES * rate_ff
        + n_wet * PIXEL_AREA_ACRES * rate_gi
        + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
    )
    avoided_carbon_cost_usd = round(carbon_tons_co2_yr * EPA_SOCIAL_COST_CARBON, 0)

    nat_pct, _nat_quality, nat_people = calculate_nature_access(
        scenario_lulc, pop_count_raster
    )

    # Build the scenario NDVI raster once and pass to both consumers. Saves
    # one full-AOI float32 allocation per evaluate_scenario call.
    scenario_ndvi = _lulc_to_ndvi_raster(scenario_lulc)
    mean_ndvi = compute_mean_ndvi(scenario_lulc, ndvi_raster=scenario_ndvi)

    total_developed_acres = len(developed_pixels) * PIXEL_AREA_ACRES
    total_cost_mln = compute_cost(n_wet, n_for, n_hd, cost_gi, cost_ff, cost_hd)
    runoff_acft    = cn_to_runoff_acre_feet(mean_cn, total_developed_acres)
    flood_damage_avoided_usd = compute_flood_damage_avoided(runoff_acft)

    # InVEST UMH preventable mental health cases + avoided cost (depression +
    # anxiety, NDVI-mediated). Returns (0, 0) if population data isn't loaded.
    preventable_mh_cases, avoided_mh_cost_usd = calculate_mental_health_impact(
        scenario_lulc, _BASELINE_NE_RASTER, pop_count_raster,
        ndvi_raster=scenario_ndvi,
    )

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
        'avoided_carbon_cost_usd':  avoided_carbon_cost_usd,
        'nature_access_pct':        nat_pct,
        'people_with_nature_access': nat_people,
        'preventable_mh_cases':     preventable_mh_cases,
        'avoided_mh_cost_usd':      avoided_mh_cost_usd,
        'food_mln_lbs':             food_mln_lbs,
        'people_fed':               food_to_people_fed(food_mln_lbs),
        'total_cost_mln':           total_cost_mln,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
# Bump SCENARIO_SCHEMA_VERSION whenever the surrogate target columns change so
# Streamlit's @st.cache_data automatically invalidates stale grids/tables.
SCENARIO_SCHEMA_VERSION = 17  # bumped: revert SA class 21 Kc to MN's 0.516 — was incorrectly tuned in v16 despite explicit Stage-3 instruction to leave it alone. Authorized SA-tuned classes are now 41, 42, 52, 81 only. See data/sa/cooling/biophysical_table_sources.md.

# Surrogate target columns that downstream code (train_surrogate, optimize_scenario)
# requires. Listed explicitly so a missing column fails loudly instead of leaking
# into a KeyError deep in fit().
REQUIRED_TARGET_COLUMNS = [
    'flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
    'carbon_tons_co2_yr', 'nature_access_pct',
    'preventable_mh_cases', 'avoided_mh_cost_usd',
]


def _compute_carbon(n_wet, n_for, n_hd):
    """Carbon sequestration at default rates — used at scenario-grid build time."""
    return round(
        n_for * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_FOOD_FOREST]
        + n_wet * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_GREEN_INFRA]
        + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
    )


@st.cache_data
def compute_scenario_grid(_state, city_key, data_dir_flood, data_dir_cooling,
                          step_pct=10, step_alloc=25,
                          schema_version=SCENARIO_SCHEMA_VERSION):
    """Build the scenario training grid. `_state` is leading-underscore so
    Streamlit skips hashing the heavy NamedTuple; `city_key` is the explicit
    cache discriminator. Read per-city arrays from `_state` rather than from
    `_CURRENT_CITY_STATE` — the cached call could fire under a stale module
    global during an in-flight rerun."""
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
                    nature_access_pct, _nature_quality, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc'], _state.pop_count_raster
                    )
                    row['nature_access_pct'] = nature_access_pct
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
def compute_lookup_table(_state, city_key, data_dir_flood, data_dir_cooling, schema_version=SCENARIO_SCHEMA_VERSION):
    """Pre-compute results for every valid slider position (step=5) for instant response.

    `_state` skip-hashed; `city_key` is the cache discriminator. Reads per-city
    arrays via `_state` to avoid stale-module-global hazards under in-flight
    reruns."""
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
                    nature_access_pct, _nature_quality, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc'], _state.pop_count_raster
                    )
                    entry['nature_access_pct'] = nature_access_pct
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


DENSE_SCENARIOS_PATH = city_cfg.get("dense_scenarios_file") or "data/scenarios_dense.csv"
# Read the model-quality selection from session_state. The radio that writes
# here lives in the Advanced Settings expander further down — Streamlit reruns
# top-to-bottom on every interaction, so on the next rerun this read picks up
# the new value the radio wrote on the previous run.

# NLCD 30 m grid — 900 m² per pixel. Used by the InVEST UCM energy-valuation
# formula in `compute_cooling_energy_savings` (kWh/(m²·°C) × ΔT × m²).
PIXEL_AREA_M2 = 30 * 30


def compute_cooling_energy_savings(scenario_hmi_raster):
    """Annual avoided AC cost ($/yr) for buildings under the active scenario,
    using the canonical InVEST UCM energy-valuation formula.

    Per pixel: `ΔT_°C = (HMI_scenario − HMI_baseline) × UHI_MAX_C`. The InVEST
    `consumption` column is documented as kWh/(m²·°C), so the per-pixel kWh
    saved is `consumption_rate × ΔT_°C × pixel_area_m²`, and the dollar value
    is multiplied by `$/kWh`. Negative ΔT (scenario hotter than baseline) is
    clamped to zero — we only credit cooling, not penalise warming. Sums over
    building pixels and returns $0 when buildings, the energy table, or the
    ET raster are unavailable.

    See `data/invest/cooling/UCM_AUDIT.md` for the divergence-from-canonical
    log: we still apply this per-pixel rather than per-building (no 600 m
    `t_air_average_radius` aggregation), but the per-pixel raster is now the
    canonical HMI = max(CC_local, CC_park).
    """
    # BUILDINGS_HAVE_TYPES gates the per-type kWh/(m²·°C) lookup. Without it
    # (e.g. OSM-only buildings for the expanded MN view) the cooling-energy-
    # savings dollar metric isn't meaningful — return $0 cleanly.
    if not (BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES
            and ENERGY_TABLE_AVAILABLE and ET_DATA_AVAILABLE):
        return 0.0
    # Single scratch buffer reused through the entire chain. Original
    # code allocated four full-AOI float32 buffers (delta_cc, delta_t_c,
    # kwh_saved_per_pixel, usd_per_pixel) — this collapses to one.
    buf = scenario_hmi_raster - _BASELINE_HM_RASTER      # buf = delta_hmi
    np.multiply(buf, UHI_MAX_C, out=buf)                  # buf = delta_cc * UHI
    np.clip(buf, 0.0, None, out=buf)                      # buf = delta_t_c
    np.multiply(buf, CONSUMPTION_RATE_PER_PIXEL, out=buf) # buf = delta_t * consumption
    np.multiply(buf, PIXEL_AREA_M2, out=buf)              # buf = kwh saved
    np.multiply(buf, COST_PER_KWH_USD, out=buf)           # buf = usd per pixel
    valid = (BUILDINGS_TYPE_RASTER >= 0) & np.isfinite(buf)
    return round(float(buf[valid].sum()), 0)


# ── OSM building-tag → InVEST type-code mapping ──────────────────────────────
# Geofabrik OSM building extracts (used for San Antonio) carry the OSM
# `building=*` tag value in the `type` column as strings, not the integer
# codes InVEST UCM expects (0=other, 1=commercial, 2=residential, 3=industrial).
# This table is an approximation: peer cooling-energy profiles of the
# canonical OSM building values onto the three InVEST categories. Untyped
# polygons (NaN or `building=yes`) get 0 and are excluded from the per-pixel
# kWh/(m²·°C) lookup downstream. See CLAUDE.md "Buildings — typing" section.
_OSM_BUILDING_TO_INVEST_TYPE = {
    # Residential → 2
    'house': 2, 'residential': 2, 'apartments': 2, 'detached': 2,
    'semidetached_house': 2, 'terrace': 2, 'bungalow': 2, 'dormitory': 2,
    'cabin': 2, 'farm': 2, 'static_caravan': 2, 'houseboat': 2,
    # Residential outbuildings → 2 (typically attached / proximate to homes)
    'garage': 2, 'carport': 2, 'shed': 2, 'barn': 2,
    'farm_auxiliary': 2, 'hut': 2,
    # Commercial → 1
    'commercial': 1, 'retail': 1, 'office': 1, 'supermarket': 1,
    'kiosk': 1, 'shop': 1, 'hotel': 1, 'restaurant': 1,
    # Industrial → 3
    'industrial': 3, 'warehouse': 3, 'factory': 3, 'manufacture': 3,
    # Public / institutional → treat as commercial (1) — closest cooling
    # profile among the three InVEST categories
    'school': 1, 'university': 1, 'hospital': 1, 'church': 1,
    'public': 1, 'civic': 1, 'government': 1, 'kindergarten': 1,
    'college': 1, 'cathedral': 1, 'chapel': 1, 'mosque': 1,
    'synagogue': 1, 'temple': 1, 'fire_station': 1, 'train_station': 1,
    # Explicitly ambiguous — return -1 ("no usable type"), NOT 0. Type 0
    # = "other" in energy_consumption.csv carries a real 10 kWh/(m²·°C)
    # rate, so writing 0 here would silently charge these polygons that
    # rate in compute_cooling_energy_savings. -1 lines up with the
    # rasterize fill sentinel and excludes them cleanly.
    'yes': -1, 'roof': -1, 'service': -1, 'storage_tank': -1,
}


def _osm_to_invest_type(tag_value) -> int:
    """Map an OSM `building=*` tag value to the InVEST integer code, or -1
    if the value is unrecognized / missing. Safe on NaN, None, or any input
    type — coerces to lowercase string before lookup. -1 means "no usable
    type" and is the same sentinel rasterize uses for "no building"."""
    if tag_value is None:
        return -1
    try:
        key = str(tag_value).lower().strip()
    except Exception:
        return -1
    if not key or key == 'nan':
        return -1
    return _OSM_BUILDING_TO_INVEST_TYPE.get(key, -1)


# ── City runtime state loader ─────────────────────────────────────────────────
# All heavy per-city allocations (rasters, distance fields, baseline images)
# happen inside this @st.cache_resource function. Cached on `city_key`, so:
#   - First call per (city, session) runs the full ~1.5 GB transient pipeline.
#   - Subsequent reruns within the session return the cached CityState
#     instantly — no re-allocation, no GeoPandas re-load.
# This is the single architectural fix for the 1011 keepalive OOM: every
# Streamlit widget interaction would previously re-execute the module-level
# allocations; now they happen at most once per session per city.
@st.cache_resource(show_spinner="Loading city data — first interaction may take a minute…")
def _load_city_runtime_state(city_key: str) -> CityState:
    cfg = CITIES[city_key]

    # ── Phase 1: cached load_data outputs ────────────────────────────────────
    (l_lulc, l_soil_resized, l_cooling_lulc, l_developed_pixels,
     l_cn_table, l_lucode_idx_arr, l_hm_arr, l_max_raster_lucode, l_max_hm_lucode,
     l_equity_weights, l_shade_arr, l_kc_arr, l_albedo_arr,
     l_green_area_arr) = load_data(
        cfg['data_dir_flood'], cfg['data_dir_cooling'],
        cfg['cn_table_file'], cfg['cooling_table_file'],
        cfg['lulc_file'], cfg['soil_file'], cfg['cooling_lulc_file'])

    # ── Phase 2: Population raster ──────────────────────────────────────────
    pop_file = cfg.get("pop_file")
    try:
        if pop_file is None:
            raise FileNotFoundError("pop_file not configured")
        pop_count_raster = load_population_data(pop_file, l_cooling_lulc.shape)
        population_data_available = True
    except (FileNotFoundError, rasterio.errors.RasterioIOError, TypeError):
        pop_count_raster = np.ones(l_cooling_lulc.shape, dtype=np.float32)
        population_data_available = False

    # ── Phase 3: Reference ET ───────────────────────────────────────────────
    et_file = cfg.get("et_file")
    try:
        if et_file is None:
            raise FileNotFoundError("et_file not configured")
        with rasterio.open(et_file) as et_src:
            et_raw = et_src.read(1).astype(np.float32)
            et_nodata = et_src.nodata
        if et_nodata is not None:
            et_raw[et_raw == et_nodata] = np.nan
        et_raw[et_raw > 10_000] = np.nan
        et_raw[et_raw < 0]      = np.nan
        et_resized = resize(et_raw, l_cooling_lulc.shape, order=1, preserve_range=True).astype(np.float32)
        finite = np.isfinite(et_resized)
        et_resized = np.where(finite, et_resized, np.nanmedian(et_resized[finite])).astype(np.float32)
        max_et_ref = float(et_resized.max()) if et_resized.max() > 0 else 1.0
        et_data_available = True
    except Exception:
        et_resized = np.ones(l_cooling_lulc.shape, dtype=np.float32)
        max_et_ref = 1.0
        et_data_available = False

    # ── Phase 4: Energy table (small dict) ──────────────────────────────────
    energy_table_file = cfg.get("energy_table_file")
    try:
        if energy_table_file is None:
            raise FileNotFoundError("energy_table_file not configured")
        energy_df = pd.read_csv(energy_table_file)
        energy_by_type = dict(zip(energy_df["type"], energy_df["consumption"]))
        energy_table_available = True
    except Exception:
        energy_by_type = {}
        energy_table_available = False

    # ── Phase 5: UNA biophysical (small DataFrame) ──────────────────────────
    una_table = pd.read_csv(Path(cfg["una_table_file"]))
    una_active = una_table[
        (una_table["urban_nature"] > 0) & una_table["search_radius_m"].notna()
    ].copy()
    una_active["lucode"] = una_active["lucode"].astype(int)
    una_active["search_radius_m"] = una_active["search_radius_m"].clip(upper=NATURE_RADIUS_CAP_M)

    # ── Phase 6: precomputed nature-distance fields (.npy disk cache) ───────
    precomp_dir = cfg.get("precomputed_dir")
    if precomp_dir:
        Path(precomp_dir).mkdir(parents=True, exist_ok=True)
    static_lucodes = [
        int(lc) for lc in una_active["lucode"]
        if int(lc) not in _DYNAMIC_NATURE_LUCODES
    ]
    precomputed_nature_distances = {}
    computed_any = False
    for lucode in static_lucodes:
        cache_file = (
            Path(precomp_dir) / f"nature_distance_{lucode}.npy"
            if precomp_dir else None
        )
        if cache_file is not None and cache_file.exists():
            try:
                arr = np.load(cache_file)
                if arr.shape == l_cooling_lulc.shape and arr.dtype == np.float32:
                    precomputed_nature_distances[lucode] = arr
                    continue
            except Exception:
                pass
        mask = (l_cooling_lulc == lucode)
        if not mask.any():
            continue
        arr = (_distance_transform_edt(~mask) * PIXEL_SIZE_M).astype(np.float32)
        precomputed_nature_distances[lucode] = arr
        computed_any = True
        if cache_file is not None:
            try:
                np.save(cache_file, arr)
            except Exception:
                pass
    # ── Phase 7: Rasterization template ─────────────────────────────────────
    with rasterio.open(f"{cfg['data_dir_cooling']}/{cfg['cooling_lulc_file']}") as ref:
        ref_shape     = (ref.height, ref.width)
        ref_transform = ref.transform

    # ── Phase 8: Buildings (BIG transient on SA — 345k polygons) ───────────
    buildings_have_types = False
    buildings_type_coverage = 0.0
    buildings_file = cfg.get("buildings_file")
    damage_table_file = cfg.get("damage_table_file")
    try:
        if not buildings_file:
            raise FileNotFoundError("buildings_file not configured")
        buildings_gdf = _gpd.read_file(buildings_file)
        if buildings_gdf.crs is None or str(buildings_gdf.crs) != cfg['crs']:
            buildings_gdf = buildings_gdf.to_crs(cfg['crs'])

        # Two paths for typing the polygons:
        #   (a) Numeric `type` column with values in {0,1,2,3} — InVEST sample
        #       shapefiles (MN downtown). Used directly.
        #   (b) String `type` column with OSM `building=*` values (SA from
        #       Geofabrik). Mapped via `_osm_to_invest_type`.
        invest_types = None
        if "type" in buildings_gdf.columns:
            numeric = pd.to_numeric(buildings_gdf["type"], errors="coerce")
            numeric_clean = numeric.dropna()
            if len(numeric_clean) > 0 and numeric_clean.between(0, 3).all():
                # Path (a): MN-style integer codes.
                invest_types = numeric.fillna(-1).astype("int32")
                buildings_have_types = True
            else:
                # Path (b): OSM string tags. _osm_to_invest_type returns -1
                # for any value that doesn't map to a real InVEST category;
                # fillna(-1) catches actual NaN cells. The -1 sentinel
                # matches the rasterize `fill=-1` and is excluded by the
                # `BUILDINGS_TYPE_RASTER >= 0` mask downstream — untyped
                # polygons are NOT charged the type-0 ("other") 10 kWh rate.
                invest_types = (
                    buildings_gdf["type"].map(_osm_to_invest_type).fillna(-1).astype("int32")
                )
                # buildings_have_types is set later from raster coverage.

        if buildings_have_types and damage_table_file:
            damage_table = pd.read_csv(damage_table_file)
            type_to_damage = dict(zip(damage_table["Type"], damage_table["Damage"]))
            buildings_gdf["damage_rate_usd_m2"] = (
                buildings_gdf["type"].map(type_to_damage).fillna(0)
            )
            buildings_gdf["area_m2"] = buildings_gdf.geometry.area
            buildings_gdf["potential_damage_usd"] = (
                buildings_gdf["area_m2"] * buildings_gdf["damage_rate_usd_m2"]
            )
            total_potential_damage_usd = float(buildings_gdf["potential_damage_usd"].sum())
        else:
            total_potential_damage_usd = 0.0

        buildings_raster = _rasterize(
            ((geom, 1) for geom in buildings_gdf.geometry),
            out_shape=ref_shape, transform=ref_transform,
            fill=0, dtype="uint8",
        )
        if invest_types is not None:
            # Burn the (possibly OSM-mapped) integer codes. fill=-1 means
            # "no building here"; 0 means "building present but untyped".
            buildings_type_raster = _rasterize(
                ((geom, int(t)) for geom, t in zip(buildings_gdf.geometry, invest_types)),
                out_shape=ref_shape, transform=ref_transform,
                fill=-1, dtype="int32",
            )
        else:
            buildings_type_raster = np.full(ref_shape, -1, dtype="int32")

        # Pixel-level coverage check + flag. Pixel-level matters more than
        # polygon-level because large typed buildings outweigh small untyped
        # ones. `buildings_have_types` is set based on whether *anything*
        # got a type code > 0 — protects against future cities with OSM data
        # whose tag values fall entirely outside the mapping.
        total_building_pixels = int(np.sum(buildings_raster > 0))
        typed_pixels = int(np.sum(buildings_type_raster > 0))
        if total_building_pixels > 0:
            buildings_type_coverage = typed_pixels / total_building_pixels
        if not buildings_have_types:
            # Path (b) — only flip the flag on if the OSM mapping actually
            # produced typed pixels. Empty mapping → leave at False, dollar
            # metrics fall back to blank.
            buildings_have_types = typed_pixels > 0
        print(
            f"[BUILDINGS] {city_key}: {typed_pixels:,}/{total_building_pixels:,} "
            f"building pixels typed ({buildings_type_coverage:.1%} coverage)"
        )

        buildings_data_available = True
        # Drop the GeoDataFrame ASAP — it's the single biggest transient on SA.
        del buildings_gdf
    except Exception:
        total_potential_damage_usd = 0.0
        buildings_data_available = False
        buildings_raster = np.zeros(l_cooling_lulc.shape, dtype="uint8")
        buildings_type_raster = np.full(l_cooling_lulc.shape, -1, dtype="int32")

    # ── Phase 9: Roads ──────────────────────────────────────────────────────
    roads_file = cfg.get("roads_file")
    try:
        if not roads_file or not Path(roads_file).exists():
            raise FileNotFoundError(f"roads_file not configured or missing: {roads_file}")
        roads_gdf = _gpd.read_file(roads_file)
        if roads_gdf.crs is None or str(roads_gdf.crs) != cfg['crs']:
            roads_gdf = roads_gdf.to_crs(cfg['crs'])
        roads_raster = _rasterize(
            ((g, 1) for g in roads_gdf.geometry),
            out_shape=ref_shape, transform=ref_transform,
            fill=0, dtype="uint8",
        )
        buildings_raster = np.maximum(buildings_raster, roads_raster)
        osm_roads_available = True
        del roads_gdf
    except Exception:
        roads_raster = np.zeros(l_cooling_lulc.shape, dtype="uint8")
        osm_roads_available = False

    # ── Phase 9b: OSM buildings mask union ──────────────────────────────────
    # Supplements the typed `buildings_raster` from `buildings_file` for
    # placement-mask purposes only: `mask_buildings_file` is untyped OSM data
    # and does NOT feed `buildings_type_raster` (the $ metrics still use the
    # typed raster from `buildings_file`). Mirrors the Phase 9 roads union;
    # runs before Phase 10 so the final `buildings_raster` is the union of
    # typed buildings + roads + OSM mask buildings. Cities without
    # `mask_buildings_file` configured (SA, Mpls Full) skip this cleanly.
    mask_buildings_file = cfg.get("mask_buildings_file")
    try:
        if not mask_buildings_file or not Path(mask_buildings_file).exists():
            raise FileNotFoundError(
                f"mask_buildings_file not configured or missing: {mask_buildings_file}"
            )
        mask_buildings_gdf = _gpd.read_file(mask_buildings_file)
        if mask_buildings_gdf.crs is None or str(mask_buildings_gdf.crs) != cfg['crs']:
            mask_buildings_gdf = mask_buildings_gdf.to_crs(cfg['crs'])
        mask_buildings_raster = _rasterize(
            ((g, 1) for g in mask_buildings_gdf.geometry),
            out_shape=ref_shape, transform=ref_transform,
            fill=0, dtype="uint8",
        )
        _mask_px_before = int(np.sum(buildings_raster > 0))
        buildings_raster = np.maximum(buildings_raster, mask_buildings_raster)
        _mask_px_after = int(np.sum(buildings_raster > 0))
        osm_buildings_available = True
        print(
            f"[MASK] {city_key}: non-convertible mask {_mask_px_before:,} px "
            f"-> {_mask_px_after:,} px after OSM buildings union "
            f"(+{_mask_px_after - _mask_px_before:,})"
        )
        del mask_buildings_gdf
    except Exception:
        osm_buildings_available = False

    # ── Phase 10: Per-pixel AC consumption rate ─────────────────────────────
    if energy_by_type:
        max_bldg_type = int(max(buildings_type_raster.max(), max(energy_by_type.keys())))
        consumption_lookup = np.zeros(max(max_bldg_type, 0) + 2, dtype=np.float32)
        for t, c in energy_by_type.items():
            if int(t) >= 0:
                consumption_lookup[int(t)] = float(c)
        safe_type = np.clip(buildings_type_raster, 0, len(consumption_lookup) - 1)
        consumption_rate_per_pixel = np.where(
            buildings_type_raster >= 0,
            consumption_lookup[safe_type],
            np.float32(0.0),
        ).astype(np.float32)
    else:
        consumption_rate_per_pixel = np.zeros(l_cooling_lulc.shape, dtype=np.float32)
        buildings_raster = np.zeros(l_cooling_lulc.shape, dtype="uint8")

    # ── Phase 11: Convertible-pixel pool ────────────────────────────────────
    no_building = buildings_raster[l_developed_pixels[:, 0], l_developed_pixels[:, 1]] == 0
    convertible_pixels = l_developed_pixels[no_building]

    # ── Phase 12: Tracts ────────────────────────────────────────────────────
    tracts_file = cfg.get("tracts_file")
    try:
        if not tracts_file:
            raise FileNotFoundError("tracts_file not configured")
        tracts = _gpd.read_file(tracts_file)
        if tracts.crs is None or str(tracts.crs) != cfg['crs']:
            tracts = tracts.to_crs(cfg['crs'])
        tracts = tracts.reset_index(drop=True)
        tract_id_raster = _rasterize(
            ((g, i) for i, g in enumerate(tracts.geometry)),
            out_shape=ref_shape, transform=ref_transform,
            fill=-1, dtype=np.int32,
        )
        tracts_data_available = True
    except Exception:
        tracts = pd.DataFrame()
        tract_id_raster = np.full(l_cooling_lulc.shape, -1, dtype=np.int32)
        tracts_data_available = False

    # ── Phase 13: Baseline rasters (use *_pure helpers because module aliases
    # haven't been rebound to this state's arrays yet — we're inside the
    # cache_resource call that produces the state). ─────────────────────────
    baseline_access_score_raster = _compute_access_score_raster_pure(
        l_cooling_lulc, una_active, precomputed_nature_distances,
    )
    baseline_hm_raster = _compute_hmi_raster_pure(
        l_cooling_lulc, l_shade_arr, l_kc_arr, l_albedo_arr, et_resized, max_et_ref,
        l_green_area_arr,
    )
    baseline_ne_raster = _gaussian_filter(
        _lulc_to_ndvi_raster(l_cooling_lulc), sigma=_UMH_SIGMA_PX, mode="nearest",
    )

    # ── Phase 14: Baseline scalars ──────────────────────────────────────────
    valid_base_cc = baseline_hm_raster[~np.isnan(baseline_hm_raster)]
    baseline_hm = (
        float(valid_base_cc.mean().round(4))
        if valid_base_cc.size > 0 else float(cfg['baseline_hm'] or 0.0)
    )
    baseline_lulc_safe = np.clip(l_cooling_lulc, 0, len(l_lucode_idx_arr) - 1)
    baseline_lulc_idx  = l_lucode_idx_arr[baseline_lulc_safe]
    baseline_soil      = np.clip(l_soil_resized, 1, 4)
    baseline_cn_grid   = l_cn_table[baseline_lulc_idx, baseline_soil]
    valid_base_cn      = baseline_cn_grid[baseline_cn_grid > 0]
    baseline_cn = (
        float(valid_base_cn.mean().round(2))
        if valid_base_cn.size > 0 else float(cfg['baseline_cn'] or 0.0)
    )

    return CityState(
        lulc=l_lulc, soil_resized=l_soil_resized, cooling_lulc=l_cooling_lulc,
        developed_pixels=l_developed_pixels, cn_table=l_cn_table,
        lucode_idx_arr=l_lucode_idx_arr, hm_arr=l_hm_arr,
        max_raster_lucode=l_max_raster_lucode, max_hm_lucode=l_max_hm_lucode,
        equity_weights=l_equity_weights,
        shade_arr=l_shade_arr, kc_arr=l_kc_arr, albedo_arr=l_albedo_arr,
        green_area_arr=l_green_area_arr,
        pop_count_raster=pop_count_raster,
        population_data_available=population_data_available,
        et_resized=et_resized, max_et_ref=max_et_ref,
        et_data_available=et_data_available,
        energy_by_type=energy_by_type,
        energy_table_available=energy_table_available,
        ref_shape=ref_shape, ref_transform=ref_transform,
        buildings_raster=buildings_raster,
        buildings_type_raster=buildings_type_raster,
        buildings_data_available=buildings_data_available,
        buildings_have_types=buildings_have_types,
        buildings_type_coverage=buildings_type_coverage,
        total_potential_damage_usd=total_potential_damage_usd,
        roads_raster=roads_raster, osm_roads_available=osm_roads_available,
        osm_buildings_available=osm_buildings_available,
        consumption_rate_per_pixel=consumption_rate_per_pixel,
        convertible_pixels=convertible_pixels,
        tracts=tracts, tract_id_raster=tract_id_raster,
        tracts_data_available=tracts_data_available,
        baseline_access_score_raster=baseline_access_score_raster,
        baseline_hm_raster=baseline_hm_raster,
        baseline_ne_raster=baseline_ne_raster,
        baseline_hm=baseline_hm, baseline_cn=baseline_cn,
    )


# ── Build (or fetch from cache) the city runtime state, then alias members
# to module-level globals. This is the seam between the cached state and the
# rest of app.py: downstream functions that read these as bare globals
# (cooling_lulc, ET_RESIZED, BUILDINGS_RASTER, ...) continue to work without
# any threading because the names rebind to the cached state's arrays on each
# rerun. Two scalars (BASELINE_HM, BASELINE_CN) are deliberately NOT aliased
# — they're read from `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` so a
# silent staleness bug on city switch is impossible.
_CURRENT_CITY_STATE = _load_city_runtime_state(selected_city)
state = _CURRENT_CITY_STATE  # short alias for use inside cached helpers below

# load_data outputs
lulc                = _CURRENT_CITY_STATE.lulc
soil_resized        = _CURRENT_CITY_STATE.soil_resized
cooling_lulc        = _CURRENT_CITY_STATE.cooling_lulc
developed_pixels    = _CURRENT_CITY_STATE.developed_pixels
cn_table            = _CURRENT_CITY_STATE.cn_table
lucode_idx_arr      = _CURRENT_CITY_STATE.lucode_idx_arr
hm_arr              = _CURRENT_CITY_STATE.hm_arr
max_raster_lucode   = _CURRENT_CITY_STATE.max_raster_lucode
max_hm_lucode       = _CURRENT_CITY_STATE.max_hm_lucode
equity_weights      = _CURRENT_CITY_STATE.equity_weights
shade_arr           = _CURRENT_CITY_STATE.shade_arr
kc_arr              = _CURRENT_CITY_STATE.kc_arr
albedo_arr          = _CURRENT_CITY_STATE.albedo_arr
green_area_arr      = _CURRENT_CITY_STATE.green_area_arr
# Population
pop_count_raster          = _CURRENT_CITY_STATE.pop_count_raster
POPULATION_DATA_AVAILABLE = _CURRENT_CITY_STATE.population_data_available
# ET
ET_RESIZED        = _CURRENT_CITY_STATE.et_resized
MAX_ET_REF        = _CURRENT_CITY_STATE.max_et_ref
ET_DATA_AVAILABLE = _CURRENT_CITY_STATE.et_data_available
# Energy
ENERGY_BY_TYPE         = _CURRENT_CITY_STATE.energy_by_type
ENERGY_TABLE_AVAILABLE = _CURRENT_CITY_STATE.energy_table_available
# Rasterization template
_REF_SHAPE     = _CURRENT_CITY_STATE.ref_shape
_REF_TRANSFORM = _CURRENT_CITY_STATE.ref_transform
# Buildings
BUILDINGS_RASTER           = _CURRENT_CITY_STATE.buildings_raster
BUILDINGS_TYPE_RASTER      = _CURRENT_CITY_STATE.buildings_type_raster
BUILDINGS_DATA_AVAILABLE   = _CURRENT_CITY_STATE.buildings_data_available
BUILDINGS_HAVE_TYPES       = _CURRENT_CITY_STATE.buildings_have_types
BUILDINGS_TYPE_COVERAGE    = _CURRENT_CITY_STATE.buildings_type_coverage
TOTAL_POTENTIAL_DAMAGE_USD = _CURRENT_CITY_STATE.total_potential_damage_usd
# Roads
ROADS_RASTER        = _CURRENT_CITY_STATE.roads_raster
OSM_ROADS_AVAILABLE = _CURRENT_CITY_STATE.osm_roads_available
OSM_BUILDINGS_AVAILABLE = _CURRENT_CITY_STATE.osm_buildings_available
# Energy + buildings derived
CONSUMPTION_RATE_PER_PIXEL = _CURRENT_CITY_STATE.consumption_rate_per_pixel
# Convertible-pixel pool
CONVERTIBLE_PIXELS = _CURRENT_CITY_STATE.convertible_pixels
# Tracts
TRACTS                = _CURRENT_CITY_STATE.tracts
TRACT_ID_RASTER       = _CURRENT_CITY_STATE.tract_id_raster
TRACTS_DATA_AVAILABLE = _CURRENT_CITY_STATE.tracts_data_available
# Baseline rasters (module-level for legacy reads; same array as state.X)
_BASELINE_ACCESS_SCORE_RASTER = _CURRENT_CITY_STATE.baseline_access_score_raster
_BASELINE_HM_RASTER           = _CURRENT_CITY_STATE.baseline_hm_raster
_BASELINE_NE_RASTER           = _CURRENT_CITY_STATE.baseline_ne_raster
# NOTE: BASELINE_HM and BASELINE_CN are intentionally NOT aliased here. Read
# them as `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` everywhere
# downstream — see CityState comment above.


def compute_per_tract_summary(scenario_lulc):
    """DataFrame with one row per tract: baseline + scenario temperature (°F)
    vs the global baseline, plus the difference (improvement)."""
    if not TRACTS_DATA_AVAILABLE or len(TRACTS) == 0:
        return pd.DataFrame()

    hm_s_raster = _compute_hmi_raster(scenario_lulc)

    rows = []
    for i in range(len(TRACTS)):
        mask = TRACT_ID_RASTER == i
        if not mask.any():
            continue
        pop_in_tract = pop_count_raster[mask].sum()
        if pop_in_tract <= 0:
            continue
        # Temperature offset vs city baseline HM, in °F (positive = cooler)
        valid_hm = mask & ~np.isnan(_BASELINE_HM_RASTER) & ~np.isnan(hm_s_raster)
        if not valid_hm.any():
            continue
        b_hm = _BASELINE_HM_RASTER[valid_hm].mean()
        s_hm = hm_s_raster[valid_hm].mean()
        # read from state to avoid silent-staleness if city switches
        b_temp_f = (b_hm - _CURRENT_CITY_STATE.baseline_hm) * HM_TO_FAHRENHEIT
        s_temp_f = (s_hm - _CURRENT_CITY_STATE.baseline_hm) * HM_TO_FAHRENHEIT
        rows.append({
            "GEOID":               str(TRACTS.iloc[i].get("GEOID10", i)),
            "Population":          int(pop_in_tract),
            "Baseline Temp (°F)":  round(b_temp_f, 2),
            "Scenario Temp (°F)":  round(s_temp_f, 2),
            "Temp Δ (°F cooler)":  round(s_temp_f - b_temp_f, 2),
        })
    return pd.DataFrame(rows)

# Baseline runoff for the damage scaling — computed inline here because
# `BASELINE_RUNOFF_ACRE_FEET` (the canonical module-level constant) isn't
# defined until after the lookup table is built. Same formula either way.
# read from state to avoid silent-staleness if city switches
_BASELINE_RUNOFF_FOR_DAMAGE = cn_to_runoff_acre_feet(
    _CURRENT_CITY_STATE.baseline_cn, len(developed_pixels) * PIXEL_AREA_ACRES
)


def compute_flood_damage_avoided(runoff_acre_feet):
    """Order-of-magnitude $ damage avoided vs baseline.

    Uses the simplification `avoided = total_potential_damage ×
    (runoff_reduction_fraction)`, where `runoff_reduction_fraction` is
    `max(0, baseline - scenario) / baseline`. Caps at zero — scenarios that
    INCREASE runoff are reported as $0 avoided rather than negative dollars
    (those regressions show up via the existing Runoff Volume card).
    """
    # Per-type damage rates from Damage_loss_table_MN.csv keyed on the
    # buildings shapefile `type` column. Without per-building type codes
    # we can't compute potential damage at all — return $0.
    if not (BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES) or _BASELINE_RUNOFF_FOR_DAMAGE <= 0:
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
    # The lookup table is the most expensive thing the app ever computes —
    # 2,541 scenarios × per-pixel rasters can take 25–50 minutes for SA
    # (3.4 M pixels). On Streamlit Cloud free tier (1 GB RAM, ~5 min
    # health-check window) that's fatal. So we now build it ONLY when the
    # user explicitly picks High Resolution mode. Fast prototype (default)
    # and Balanced both skip it entirely.
    #
    # The slider-response path (`if lookup_key in lookup_table` further
    # down) gracefully falls through to a fresh evaluate_scenario call when
    # the table is empty — slightly slower per-slider but functional. The
    # "Best scenarios by goal" section also falls back to scenario_df when
    # lookup_table is empty.
    if _requested_model_quality == "High resolution":
        lookup_table = compute_lookup_table(_CURRENT_CITY_STATE, selected_city, DATA_DIR_FLOOD, DATA_DIR_COOLING)
        scenario_df = pd.DataFrame(list(lookup_table.values()))
        ACTIVE_MODEL_QUALITY = "high"
    elif _requested_model_quality == "Balanced":
        lookup_table = {}
        _dense_configured = city_cfg.get("dense_scenarios_file")
        if _dense_configured and os.path.exists(_dense_configured):
            scenario_df = pd.read_csv(_dense_configured)
        else:
            if not _dense_configured:
                st.warning(
                    f"⚠️ Balanced mode: no `dense_scenarios_file` configured for "
                    f"{selected_city!r} — recomputing on the fly. Add the path to "
                    f"the CITIES entry once you've run "
                    f"`python3 precompute_scenarios.py --city '{selected_city}' "
                    f"--output data/scenarios_dense_<city>.csv`."
                )
            else:
                st.warning(
                    f"⚠️ Balanced mode: `{_dense_configured}` not found — "
                    f"recomputing on the fly. Run "
                    f"`python3 precompute_scenarios.py --city '{selected_city}' "
                    f"--output {_dense_configured}` once to skip this on future startups."
                )
            scenario_df = compute_scenario_grid(
                _CURRENT_CITY_STATE, selected_city,
                DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=5, step_alloc=10,
            )
        ACTIVE_MODEL_QUALITY = "balanced"
    else:  # Fast prototype
        lookup_table = {}
        scenario_df = compute_scenario_grid(
            _CURRENT_CITY_STATE, selected_city,
            DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=10, step_alloc=25,
        )
        ACTIVE_MODEL_QUALITY = "fast"

MAX_FOOD  = float(scenario_df['food_mln_lbs'].max())
MAX_FLOOD = 100.0
MAX_COOL  = 1.1

# read from state to avoid silent-staleness if city switches
BASELINE_RUNOFF_ACRE_FEET = cn_to_runoff_acre_feet(
    _CURRENT_CITY_STATE.baseline_cn, len(developed_pixels) * PIXEL_AREA_ACRES
)

BASELINE_NDVI = compute_mean_ndvi(cooling_lulc)

# ── Surrogate model ────────────────────────────────────────────────────────────
# Training, prediction, Pareto, and optimizer logic live in surrogate.py.
# The @st.cache_resource wrapper stays here so surrogate.py is Streamlit-agnostic.
@st.cache_resource
def _cached_train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling,
                            mode_key="fast", n_estimators=100):
    # mode_key + n_estimators participate in the cache key so changing the
    # Model quality mode radio in the sidebar automatically retrains on the
    # new training set without needing a manual cache clear.
    return _train_surrogate_fn(_scenario_df, n_estimators=n_estimators)


surrogate = _cached_train_surrogate(
    scenario_df, DATA_DIR_FLOOD, DATA_DIR_COOLING,
    mode_key=ACTIVE_MODEL_QUALITY, n_estimators=N_ESTIMATORS,
)

# ── Plotting helpers ───────────────────────────────────────────────────────────
def render_matplotlib(fig):
    try:
        st.pyplot(fig, width='stretch')
    finally:
        plt.close(fig)


# ── Matplotlib plots ───────────────────────────────────────────────────────────
# Cap on the dimension matplotlib actually rasterises for the spatial map.
# Streamlit displays the figure at ~600 px wide regardless, so rendering the
# full 1713×1984 SA AOI (which produced ~378 MB transient per imshow call
# under the previous float64 RGBA layers) is wasted work. Aspect ratio is
# preserved by `scale = _PLOT_MAX_DIM / max(h, w)`.
_PLOT_MAX_DIM = 1024


def _downsample_for_plot(arr, order):
    """Aspect-preserving downsample for the spatial-map renderer.

    `order=0` (nearest neighbor) for the integer LULC raster — category
    integrity must be preserved, no averaging across lucodes. `order=1`
    (bilinear) for continuous overlays (heat alpha, tract score). Returns
    the input unchanged when both dimensions are already within
    `_PLOT_MAX_DIM`."""
    h, w = arr.shape[:2]
    if max(h, w) <= _PLOT_MAX_DIM:
        return arr
    scale = _PLOT_MAX_DIM / max(h, w)
    return _zoom(arr, scale, order=order)


def plot_spatial_map(scenario_lulc, baseline_lulc,
                     heat_overlay=None, overlay_alpha=0.0,
                     tract_value=None, tract_alpha=0.0):
    # Downsample once, then build all layers from the downsampled rasters.
    # Doing this after layer construction would defeat the memory savings.
    scenario_lulc = _downsample_for_plot(scenario_lulc, order=0)
    baseline_lulc = _downsample_for_plot(baseline_lulc, order=0)
    h, w = scenario_lulc.shape

    # uint8 RGB triples in [0,255]. matplotlib imshow accepts uint8 natively
    # and skips the float64 → display-byte conversion path, cutting the layer
    # array from ~82 MB to ~10 MB at full SA resolution and ~3 MB after the
    # 1024px cap.
    def _rgb_u8(name):
        r, g, b = mcolors.to_rgb(CHANGE_COLORS[name])
        return np.array([round(r * 255), round(g * 255), round(b * 255)], dtype=np.uint8)

    rgb = np.full((h, w, 3), _rgb_u8('Unchanged'), dtype=np.uint8)
    changed = (baseline_lulc != scenario_lulc)
    rgb[changed & (scenario_lulc == CODE_GREEN_INFRA)] = _rgb_u8('Green Infrastructure')
    rgb[changed & (scenario_lulc == CODE_FOOD_FOREST)] = _rgb_u8('Food Forest')
    rgb[changed & (scenario_lulc == CODE_HIGH_DENSITY)] = _rgb_u8('High Density')
    rgb[baseline_lulc == NODATA] = (255, 255, 255)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    legend_handles = [
        Patch(facecolor=CHANGE_COLORS['Unchanged'],            label='Unchanged'),
        Patch(facecolor=CHANGE_COLORS['Green Infrastructure'], label='→ Green Infrastructure'),
        Patch(facecolor=CHANGE_COLORS['Food Forest'],          label='→ Food Forest'),
        Patch(facecolor=CHANGE_COLORS['High Density'],         label='→ High Density'),
    ]

    # Optional heat-vulnerability overlay (orange, was red — see commit
    # notes). Per-pixel alpha = overlay_alpha × heat_overlay value (which is
    # 0–1), so low-vulnerability pixels stay transparent and high-vulnerability
    # ones tint orange. With overlay_alpha=0 the overlay is fully invisible.
    if heat_overlay is not None and overlay_alpha > 0:
        heat_overlay_ds = _downsample_for_plot(heat_overlay, order=1)
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Orange channel mix (R=255, G=140, B=0) — avoids collision with the
        # "→ High Density" red. Alpha still encodes the HV gradient per pixel.
        overlay_rgba[..., 0] = 255  # red
        overlay_rgba[..., 1] = 140  # green
        overlay_rgba[..., 2] = 0    # blue (explicit even though default is 0)
        alpha_f = overlay_alpha * np.clip(heat_overlay_ds, 0.0, 1.0)
        overlay_rgba[..., 3] = (alpha_f * 255).astype(np.uint8)
        ax.imshow(overlay_rgba)
        legend_handles.append(Patch(facecolor=(1.0, 140/255, 0.0, 0.6), label='Heat vulnerability'))

    # Optional tract-level improvement overlay. tract_value is a per-pixel
    # float raster (NaN outside any tract); colormap is RdYlGn so positive
    # improvements are green and regressions are red, centered at 0. Cmap
    # output kept float32 because fractional alpha matters for the blend.
    if tract_value is not None and tract_alpha > 0:
        # NaNs don't survive bilinear interpolation, so downsample the
        # validity mask separately (nearest-neighbor) and use it to gate the
        # cmap alpha.
        tract_value_ds = _downsample_for_plot(np.nan_to_num(tract_value, nan=0.0), order=1)
        valid_full = (~np.isnan(tract_value)).astype(np.uint8)
        valid_ds = _downsample_for_plot(valid_full, order=0).astype(bool)
        if valid_ds.any():
            vmax = max(float(np.abs(tract_value_ds[valid_ds]).max()), 0.1)
            norm_val = np.zeros_like(tract_value_ds, dtype=np.float32)
            norm_val[valid_ds] = (tract_value_ds[valid_ds] + vmax) / (2 * vmax)  # → 0..1
            cmap_rgba = plt.get_cmap("RdYlGn")(np.clip(norm_val, 0.0, 1.0)).astype(np.float32)
            cmap_rgba[..., 3] = tract_alpha * valid_ds.astype(np.float32)
            ax.imshow(cmap_rgba)
            legend_handles.append(
                Patch(facecolor=(0.0, 0.6, 0.0, 0.6),
                      label="Neighborhood improvement (green = better)")
            )

    ax.axis('off')
    # Title removed — section H2 "Where Changes Happen" already provides context
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
        'Baseline':                   'top right',
        'All Food Forest (NLCD 41)':  'middle right',
        'All Green Infra (NLCD 90)':  'top left',
        'All High Density (NLCD 24)': 'bottom right',
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
        title='',
        xaxis_title='Flood Risk Reduction (higher = better)',
        yaxis_title='Cooling Capacity (higher = better)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 0.6]),
        height=520,
        margin=dict(l=60, r=200, t=30, b=60),
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left', yanchor='top',
                    tracegroupgap=4, font=dict(size=11), itemsizing='constant',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        hovermode='closest',
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Land Use Scenario")

# Seed slider defaults via session_state (not via widget `value=` kwarg) so
# the city-change reset above composes cleanly and Streamlit does not warn
# about a key being set both via the widget default and the Session State API.
st.session_state.setdefault("slider_pct_converted", 10)
st.session_state.setdefault("slider_gi_pct", 0)
st.session_state.setdefault("slider_ff_pct", 0)

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
    step=5, key="slider_gi_pct",
    help="Share of converted land allocated to green infrastructure (woody wetlands, NLCD 90)."
)
food_forest_pct = st.sidebar.number_input(
    "Food Forest %", 0, 100,
    step=5, key="slider_ff_pct",
    help="Share of converted land allocated to food forest (deciduous forest, NLCD 41)."
)

auto_hd = 100 - green_infrastructure_pct - food_forest_pct
pct_highdensity = st.sidebar.number_input(
    "High Density %", 0, 100,
    value=max(0, auto_hd),
    step=5,
    help="Share of converted land allocated to high-density development (NLCD 24). Auto-fills as remainder."
)

mix_sum = green_infrastructure_pct + food_forest_pct + pct_highdensity

if mix_sum == 100:
    st.sidebar.success("Mix sums to 100%")
else:
    st.sidebar.error(f"Mix sums to {mix_sum}% — must equal 100%")
    st.stop()

st.sidebar.divider()

# ── Quick Start — preset scenarios ───────────────────────────────────────────
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

if st.sidebar.button("Green Infrastructure",
                     type="primary" if _active == 'green_infra' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 100
    st.session_state._pending_ff = 0
    st.session_state.active_example_scenario = 'green_infra'
    st.rerun()
st.sidebar.caption("Flood mitigation focus")

if st.sidebar.button("Food Forest",
                     type="primary" if _active == 'food_forest' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 100
    st.session_state.active_example_scenario = 'food_forest'
    st.rerun()
st.sidebar.caption("Cooling + food production focus")

if st.sidebar.button("High Density",
                     type="primary" if _active == 'high_density' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 0
    st.session_state._pending_ff = 0
    st.session_state.active_example_scenario = 'high_density'
    st.rerun()
st.sidebar.caption("Control case — no green conversion")

st.sidebar.divider()
st.sidebar.subheader("Find Best Scenario")

st.sidebar.caption(
    "Uses a surrogate model trained on ~90 full-resolution simulations to "
    "search ~10,000 candidate strategies in seconds. Results are approximate "
    "— verify promising scenarios using the main sliders."
)

st.sidebar.caption(
    "Optimization currently targets flood reduction, cooling, food production, and carbon "
    "sequestration. Cost and placement strategy are not yet included in the surrogate."
)

with st.sidebar.container(border=True):

    min_flood  = st.slider("Min flood reduction", 0, 90, 30, 5,
        help="Corresponds to the Flood Risk Reduction metric card. Baseline is 24.3. Higher values mean less runoff — increasing this target will also reduce Runoff Volume in ac-ft.")
    # read from state to avoid silent-staleness if city switches
    _baseline_hm_local = _CURRENT_CITY_STATE.baseline_hm
    min_cool_f = st.slider(
        "Min cooling (°F vs baseline)",
        min_value=-1.0, max_value=round((1.0 - _baseline_hm_local) * HM_TO_FAHRENHEIT, 1),
        value=0.1, step=0.1,
        help="Corresponds to the Temperature Change metric card. Set to 0.1 for at least 0.1°F cooler than baseline."
    )
    min_cool   = _baseline_hm_local + min_cool_f / HM_TO_FAHRENHEIT   # HM units for surrogate
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
                surrogate, min_flood, min_cool, min_food, max_runoff,
                min_carbon=min_carbon, max_food=MAX_FOOD,
                max_flood=MAX_FLOOD, max_cool=MAX_COOL)
        _opt_res = st.session_state.optimized_results
        if _opt_res is None or (isinstance(_opt_res, dict) and not _opt_res.get('found')):
            st.sidebar.warning("No scenarios found — try lowering the targets.")
            st.session_state.just_optimized = False
        else:
            st.sidebar.success("Results ready — open the Tradeoff Analysis tab →")
            st.session_state.just_optimized = True

st.sidebar.divider()

# ── Placement strategy ────────────────────────────────────────────────────────
st.sidebar.subheader("Placement Strategy")
placement_strategy = st.sidebar.radio(
    "Which pixels get converted",
    options=list(PLACEMENT_STRATEGY_LABELS.keys()),
    format_func=lambda key: PLACEMENT_STRATEGY_LABELS[key],
    index=0,
    help=(
        "Which pixels get converted. Random samples uniformly across "
        "convertible developed pixels. Focused strategies bias placement "
        "toward pixels where conversion yields the most benefit for the "
        "chosen objective. Balanced combines flood, cooling, and equity "
        "signals equally."
    ),
    label_visibility="collapsed",
)
# Legacy alias kept for backward compatibility with saved scenarios.
use_heat_priority = (placement_strategy == 'cooling-focused')

st.sidebar.divider()

# ── Cost sliders (collapsed expander) ────────────────────────────────────────
with st.sidebar.expander("Implementation Costs ($/acre)", expanded=False):
    cost_gi = st.slider("Green Infrastructure ($/acre)", 5_000, 150_000,
                        DEFAULT_COST_GI, 5_000,
                        help="Typical range: $20,000–$100,000/acre for constructed wetlands. Default is an illustrative estimate — adjust to reflect local project costs.")
    cost_ff = st.slider("Food Forest ($/acre)", 1_000, 50_000,
                        DEFAULT_COST_FF, 1_000,
                        help="Typical range: $5,000–$20,000/acre for food forest establishment. Default is an illustrative estimate — adjust to reflect local project costs.")
    cost_hd = st.slider("High Density Infill ($/acre)", 1_000, 50_000,
                        DEFAULT_COST_HD, 1_000,
                        help="Marginal cost of additional impervious development. Default is an illustrative estimate — adjust to reflect local project costs.")

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

# ── Main panel ─────────────────────────────────────────────────────────────────
lookup_key = (pct_converted, green_infrastructure_pct, food_forest_pct)
if lookup_key in lookup_table and placement_strategy == 'random':
    # Lookup table was computed with random placement — only use it in random mode
    results = lookup_table[lookup_key].copy()
    _fresh = evaluate_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
        placement_strategy='random', cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd,
        carbon_rate_ff=st.session_state.carbon_rate_ff,
        carbon_rate_gi=st.session_state.carbon_rate_gi,
    )
    results['scenario_lulc'] = _fresh['scenario_lulc']
    # Food values are recomputed live — lookup table may predate the n_food_pixels fix
    results['food_mln_lbs'] = _fresh['food_mln_lbs']
    results['people_fed']   = _fresh['people_fed']
    results['mean_ndvi']    = _fresh['mean_ndvi']
    results['carbon_tons_co2_yr'] = _fresh['carbon_tons_co2_yr']
    results['avoided_carbon_cost_usd'] = _fresh['avoided_carbon_cost_usd']
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
        placement_strategy=placement_strategy, cost_gi=cost_gi, cost_ff=cost_ff, cost_hd=cost_hd,
        carbon_rate_ff=st.session_state.carbon_rate_ff,
        carbon_rate_gi=st.session_state.carbon_rate_gi,
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

def _delta_pill(value_delta, *, fmt="", suffix="vs baseline", epsilon=0.05):
    """Consistent delta string + color for st.metric cards.

    Returns (delta_str, delta_color).
    - Zero-delta  →  (None, "off")  — pill suppressed entirely
    - Positive    →  ("+{value} {suffix}", "normal")  — green ↑
    - Negative    →  ("-{abs(value)} {suffix}", "normal")  — red ↓

    Sign convention: the helper does not invert signs for lower-is-better
    metrics. Callers pass the delta pre-flipped for direction-of-goodness
    (e.g. runoff_prevented = baseline − scenario, so positive = good).

    For zero-baselined metrics (currently Carbon), callers may pass the raw
    value rather than a computed delta — the helper treats them equivalently.
    If a non-zero baseline is ever introduced, call sites must switch to a
    true delta.
    """
    if abs(value_delta) < epsilon:
        return None, "off"
    if value_delta > 0:
        return f"+{value_delta:{fmt}} {suffix}", "normal"
    return f"-{abs(value_delta):{fmt}} {suffix}", "normal"

_CONFIDENCE_BADGES = {
    "high":      "High confidence",
    "medium":    "Medium confidence",
    "prototype": "Prototype",
}

def _confidence_caption(col, tier):
    """Render the confidence badge under a metric card.
    tier ∈ {'high', 'medium', 'prototype'} — see 'How this prototype works'
    expander for tier definitions."""
    col.caption(_CONFIDENCE_BADGES[tier])

# read from state to avoid silent-staleness if city switches
_flood_delta = results['flood_reduction'] - (100 - _CURRENT_CITY_STATE.baseline_cn)
_flood_delta_str, _flood_delta_color = _delta_pill(_flood_delta, fmt=".1f", epsilon=0.1)
_cooling_f = results['cooling_f']
_cooling_label = (
    "No change" if abs(_cooling_f) < 0.1
    else f"{_cooling_f:.1f}°F cooler" if _cooling_f > 0
    else f"{abs(_cooling_f):.1f}°F warmer"
)
_hm_delta = results['mean_hm'] - _CURRENT_CITY_STATE.baseline_hm
_runoff_prevented = BASELINE_RUNOFF_ACRE_FEET - results['runoff_acre_feet']
_runoff_delta_str, _runoff_delta_color = _delta_pill(
    _runoff_prevented, fmt=",.0f",
    suffix="ac-ft vs baseline",
    epsilon=1.0,
)
_people_fed = results['people_fed']
_food_delta_str = f"feeds ~{_people_fed:,} people" if _people_fed > 0 else None

_carbon_value = results['carbon_tons_co2_yr']

def _fmt_carbon(tons):
    """Compact carbon display — k notation kicks in at 1,000 t to avoid card truncation."""
    if tons >= 1000:
        return f"{tons / 1000:.1f}k t CO2e/yr"
    return f"{tons:,.0f} t CO2e/yr"

_carbon_value_str = _fmt_carbon(_carbon_value)
_carbon_delta_str, _carbon_delta_color = _delta_pill(_carbon_value, fmt=",.0f", suffix="t CO2e/yr from conversions", epsilon=1.0)

if placement_strategy != 'random':
    st.caption(f"Placement: {PLACEMENT_STRATEGY_LABELS[placement_strategy]}")

st.markdown("#### Ecological")
eco1, eco2, eco3 = st.columns(3)
eco1.metric(
    "Flood Risk Reduction",
    f"{results['flood_reduction']:.1f}",
    delta=_flood_delta_str,
    delta_color=_flood_delta_color,
    help=(
        "Confidence: High — see 'How this prototype works' for tier definitions. "
        "Unitless index (0–100) based on the USDA Curve Number. Higher = less "
        "runoff potential. Baseline is 24.3 for Minneapolis developed land. "
        "Underlying model: [InVEST Urban Flood Risk Mitigation]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)."
    )
)
_confidence_caption(eco1, "high")
eco2.metric(
    "Temperature Change",
    _cooling_label,
    delta=None,
    delta_color="off",
    help="Confidence: High — see 'How this prototype works' for tier definitions. Approximate temperature change vs baseline. Positive = cooler, negative = warmer. Derived from mean Cooling Capacity (CC) under the InVEST UCM (calibration factor 3.69°F/CC unit from Minneapolis UHI=2.05°C; ±2°F accuracy). Note: this is mean(CC), an approximation of the canonical InVEST Heat Mitigation Index — see UCM_AUDIT.md. Underlying model: [InVEST Urban Cooling Model](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
)
_confidence_caption(eco2, "high")
eco3.metric(
    "Runoff Volume",
    _fmt_runoff(results['runoff_acre_feet']),
    delta=_runoff_delta_str,
    delta_color=_runoff_delta_color,
    help=(
        "Confidence: High — see 'How this prototype works' for tier definitions. "
        f"Acre-feet of runoff generated by a {DESIGN_STORM_INCHES}-inch design storm. "
        f"Delta shows reduction vs baseline ({_fmt_runoff(BASELINE_RUNOFF_ACRE_FEET)}). "
        "Lower volume = more retention. "
        "Underlying model: [InVEST Urban Flood Risk Mitigation]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)."
    )
)
_confidence_caption(eco3, "high")

_ndvi_delta = results['mean_ndvi'] - BASELINE_NDVI
_ndvi_delta_str, _ndvi_delta_color = _delta_pill(_ndvi_delta, fmt=".3f", suffix="vs baseline", epsilon=0.001)

eco4, eco5 = st.columns([2, 1])
eco4.metric(
    "Carbon Sequestration",
    _carbon_value_str,
    delta=_carbon_delta_str,
    delta_color=_carbon_delta_color,
    help=(
        "Confidence: Prototype — see 'How this prototype works' for tier definitions. "
        "Annual CO2e sequestration from converted pixels only. "
        "Uses provisional regional USDA/IPCC rates: Food Forest 3.5 t CO2e/acre/yr, "
        "Green Infrastructure 2.0 t CO2e/acre/yr. "
        "Treat as directional only — refine with locally calibrated values. "
        "Loosely related model: [InVEST Carbon Storage and Sequestration]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/carbonstorage.html)."
    )
)
_confidence_caption(eco4, "prototype")
eco5.metric(
    "NDVI",
    f"{results['mean_ndvi']:.3f}",
    delta=_ndvi_delta_str,
    delta_color=_ndvi_delta_color,
    help=(
        "Confidence: Prototype — see 'How this prototype works' for tier definitions. "
        "Synthetic vegetation index (0–1) estimated from land cover type — not derived from satellite imagery. "
        "Higher = more vegetation. Woody wetlands: 0.70, Food Forest: 0.75, High Density: 0.10–0.30. "
        "Treat as directional only."
    )
)
_confidence_caption(eco5, "prototype")

st.divider()

st.markdown("#### Human & Social")

# InVEST Urban Mental Health (v3.19.0): two cards in a second row.
# Both are zero at the unmodified baseline by construction (ΔNE = 0 → PF = 0 → PC = 0).
# Sign convention: preventable_mh_cases > 0 means the scenario PREVENTS cases
# (good — green ↑); < 0 means the scenario INDUCES cases (bad — red ↑). Same
# direction-of-goodness for avoided_mh_cost_usd. Streamlit's delta color
# combines with the leading sign of the delta string: to get a red ↑, we feed
# a positive-signed delta ("+X cases induced") with color="inverse".
# MH cards use bespoke pill rendering instead of _delta_pill: positive-
# signed strings with delta_color="inverse" give red ↑ for "induced
# cases" / "added in costs", matching healthcare-burden semantics. The
# rest of the app's metric pills use _delta_pill which produces red ↓
# for negative deltas. Both are internally consistent answers to
# st.metric's sign-parses-arrow constraint; the MH framing is the
# right one for healthcare burden specifically.
hs_na, hs3, hs4 = st.columns(3)

# Nature Access — canonical InVEST Urban Nature Access (2SFCA), re-implemented
# in numpy by `calculate_nature_access`. See DESIGN_NOTES.md.
_nature_access = results.get('nature_access_pct', 0.0)
hs_na.metric(
    "Nature Access",
    f'{_nature_access:.1f}%',
    help=(
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        "% of MN population whose per-capita nature supply meets the 16.7 m²/capita "
        "demand standard, computed via canonical InVEST Urban Nature Access (2SFCA "
        "methodology). Reports only the modelable-extent population (~43% of MN total) — "
        "the remainder sits on cooling-LULC nodata pixels InVEST cannot model. "
        "Parameters: 800m uniform search radius, dichotomy decay. See "
        "DESIGN_NOTES.md for parameter rationale. "
        "Underlying model: [InVEST Urban Nature Access]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_nature_access.html)."
    ),
)
_confidence_caption(hs_na, "medium")

_mh_cases = results.get('preventable_mh_cases', 0.0)
_mh_cost  = results.get('avoided_mh_cost_usd', 0.0)
if _mh_cases >= _MH_CASES_PILL_EPSILON:
    _mh_cases_delta = f"+{_mh_cases:,.0f} cases prevented"
    _mh_cases_color = "normal"      # green ↑
elif _mh_cases <= -_MH_CASES_PILL_EPSILON:
    _mh_cases_delta = f"+{abs(_mh_cases):,.0f} cases induced"
    _mh_cases_color = "inverse"     # red ↑
else:
    _mh_cases_delta = None
    _mh_cases_color = "off"
hs3.metric(
    "Preventable MH Cases",
    f'{_mh_cases:,.0f}',
    delta=_mh_cases_delta,
    delta_color=_mh_cases_color,
    help=(
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        "Estimated preventable depression "
        "and anxiety cases from the scenario's NDVI exposure change. Based on "
        "the InVEST Urban Mental Health model (v3.19.0): per-pixel "
        "ΔNE = NE_scenario − NE_baseline (smoothed at 300 m), "
        "RR = exp(ln(RR₀.₁) × 10 × ΔNE), "
        "PC = (1 − RR) × baseline_prevalence × population. "
        "Effect sizes from Liu et al. 2023 meta-analysis; baseline prevalence "
        "from CDC 2023 (depression 21 %, anxiety 19 %). Returns 0 at baseline "
        "and for scenarios with no greenness change. Negative values mean the "
        "scenario INDUCED cases (e.g. converting open space to high-density "
        "development) — shown in red. "
        "Underlying model: [InVEST Urban Mental Health]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_mental_health.html)."
    ),
)
_confidence_caption(hs3, "medium")
hs3.caption("cases prevented" if _mh_cases >= 0 else "cases induced")
if _mh_cost >= _MH_COST_PILL_EPSILON:
    _mh_cost_value = f'${_mh_cost / 1e6:.2f}M/yr'
    _mh_cost_delta = f"+${_mh_cost / 1e6:.2f}M/yr avoided"
    _mh_cost_color = "normal"
elif _mh_cost <= -_MH_COST_PILL_EPSILON:
    _mh_cost_value = f'-${abs(_mh_cost) / 1e6:.2f}M/yr'
    _mh_cost_delta = f"+${abs(_mh_cost) / 1e6:.2f}M/yr added in costs"
    _mh_cost_color = "inverse"
else:
    _mh_cost_value = f'${_mh_cost / 1e6:.2f}M/yr'
    _mh_cost_delta = None
    _mh_cost_color = "off"
hs4.metric(
    "Avoided MH Costs",
    _mh_cost_value,
    delta=_mh_cost_delta,
    delta_color=_mh_cost_color,
    help=(
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        "Avoided healthcare cost = "
        "preventable_cases × per-case cost-of-illness. Per-case costs: "
        f"\\${COST_PER_DEPRESSION_CASE_USD:,}/depression, "
        f"\\${COST_PER_ANXIETY_CASE_USD:,}/anxiety (US nominal; InVEST default "
        "is ~\\$11K USD-PPP/case). Sums depression + anxiety. Order-of-"
        "magnitude — see REFERENCE.md for full caveats. "
        "Underlying model: [InVEST Urban Mental Health]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_mental_health.html)."
    ),
)
_confidence_caption(hs4, "medium")
hs4.caption("avoided MH costs/yr" if _mh_cost >= 0 else "added MH costs/yr")

st.divider()

st.markdown("#### Economic")
# Row 1: Food Production + Implementation Cost (the two scenario-input-driven cards)
econ1, econ2 = st.columns(2)
econ1.metric(
    "Food Production",
    _fmt_food(results['food_mln_lbs']),
    delta=_food_delta_str,
    delta_color="normal" if _people_fed > 0 else "off",
    help=(
        "Confidence: Prototype — see 'How this prototype works' for tier definitions. "
        "Counts only food forest pixels "
        "created by this scenario (not pre-existing deciduous forest). "
        f"Yield estimated at {FOOD_FOREST_LBS_ACRE:,} lbs/acre/year for "
        f"{selected_city} — "
        + (
            "NatCap MN food forest benchmark."
            if selected_city.startswith("Minneapolis")
            else "placeholder for the pecan/fig/mulberry/nopal mix per the "
                 "NatCap SA Urban Agriculture project report; below the MN "
                 "benchmark to reflect hot semi-arid productivity. Replace "
                 "with project-published weighted average when available."
        )
        + " Treat as directional only."
    )
)
_confidence_caption(econ1, "prototype")
if results['food_mln_lbs'] == 0:
    econ1.caption(
        "No food forest in this scenario — add Food Forest % to see production estimates."
    )
econ2.metric(
    "Est. Implementation Cost",
    f"${results['total_cost_mln']:.1f}M",
    delta=None,
    help="Confidence: Medium — see 'How this prototype works' for tier definitions. Total cost based on $/acre sliders × converted acreage."
)
_confidence_caption(econ2, "medium")

# Row 2: the three model-derived dollar metrics (each computed downstream
# from the scenario, not directly from the user's sliders).
econ3, econ4, econ5 = st.columns(3)

_flood_damage_avoided = results.get('flood_damage_avoided_usd', 0.0)
# Render dollars only when the city has *both* per-building types AND a
# damage table — `TOTAL_POTENTIAL_DAMAGE_USD > 0` is the single signal that
# covers both. SA has types now (OSM mapping) but no damage table, so the
# total is still $0 and the card must render "—", not "$0.0M".
if BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES and TOTAL_POTENTIAL_DAMAGE_USD > 0:
    econ3.metric(
        "Flood Damage Avoided",
        f"${_flood_damage_avoided / 1e6:.1f}M",
        delta=(
            f"+${_flood_damage_avoided / 1e6:.1f}M vs baseline"
            if _flood_damage_avoided >= 1e4 else "no avoided damage"
        ),
        delta_color="normal" if _flood_damage_avoided >= 1e4 else "off",
        help=(
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "Estimated reduction in "
            "flood damage costs based on the InVEST damage-loss table by "
            "building type (Roads $40, Commercial $120, Residential $150, "
            "Industrial $100 per m²) joined to a 3,788-building footprint "
            "shapefile. Scales with this scenario's runoff reduction vs "
            f"baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). Capped at $0 "
            "for scenarios that increase runoff. "
            "Underlying model: [InVEST Urban Flood Risk Mitigation]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)."
        ),
    )
    _confidence_caption(econ3, "medium")
else:
    if BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES:
        # Has typed buildings but no damage table (SA today). Be honest
        # about the specific gap rather than implying buildings are
        # missing.
        _help_text = (
            "Damage rates per building type not available for this city "
            "(no $/m² damage table sourced yet). The per-pixel building "
            "mask and InVEST type codes are loaded — adding a city-specific "
            "damage-loss CSV would light this card up."
        )
    elif BUILDINGS_DATA_AVAILABLE and not BUILDINGS_HAVE_TYPES:
        _help_text = (
            "Building-type data not available for this extent — requires "
            "per-building type codes (1=commercial, 2=residential, "
            "3=industrial) to look up damage rates."
        )
    else:
        _help_text = (
            "Buildings shapefile or damage-loss table not loaded — see "
            "data/invest/flood/UFR_sample_data_MN/."
        )
    econ3.metric(
        "Flood Damage Avoided",
        "—",
        help="Confidence: Medium — see 'How this prototype works' for tier definitions. " + _help_text + " Underlying model: [InVEST Urban Flood Risk Mitigation](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html).",
    )
    _confidence_caption(econ3, "medium")

_energy_savings = results.get('cooling_energy_savings_usd', 0.0)
_energy_available = (
    BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES
    and ENERGY_TABLE_AVAILABLE and ET_DATA_AVAILABLE
)
if _energy_available:
    # Per-typed-building-pixel rate. City totals scale with the building
    # footprint (MN: InVEST sample shapefile ~447 px / ~0.4 km² downtown;
    # SA: county-wide OSM ~36,860 typed px / ~33 km²), which makes the
    # absolute dollar values incomparable across cities. The per-pixel
    # rate strips out the footprint-scope difference and is the
    # apples-to-apples cross-city number. "Per typed building pixel" not
    # "per building" — multi-pixel buildings get counted by pixel — but
    # pixel size is identical (30 m NLCD) in both cities so the rate is
    # comparable.
    _typed_px = int(np.sum(BUILDINGS_TYPE_RASTER > 0))
    _per_pixel_cooling_usd = (
        _energy_savings / _typed_px if _typed_px > 0 else None
    )

    def _fmt_per_pixel_rate(usd):
        if usd is None:
            return None
        if usd >= 1000:
            return f"~${round(usd / 10) * 10:,.0f}/yr per typed building"
        if usd >= 100:
            return f"~${usd:,.0f}/yr per typed building"
        return f"~${usd:,.2f}/yr per typed building"

    # When typing coverage is partial (currently SA), append a one-sentence
    # caveat so the headline number is interpreted as a lower bound.
    if BUILDINGS_TYPE_COVERAGE < 0.95:
        _help_text = (
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "Estimated avoided air-conditioning costs from urban cooling "
            "effects, computed over building pixels with mapped commercial/"
            f"residential/industrial types. For this city, ~"
            f"{BUILDINGS_TYPE_COVERAGE:.0%} of building pixels carry a "
            "recognized type tag; untyped buildings (e.g. OSM `building=yes`) "
            "are excluded. Conservative lower-bound estimate. The "
            "per-typed-building rate is roughly comparable across cities "
            "even when total building footprints differ; the city total is "
            "sensitive to the size of the typed-building dataset. "
            "Underlying model: [InVEST Urban Cooling Model]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
        )
    else:
        _help_text = (
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "Estimated avoided air-conditioning costs from urban cooling "
            "effects, computed over building pixels typed as commercial/"
            "residential/industrial. Order-of-magnitude estimate. The "
            "per-typed-building rate is roughly comparable across cities "
            "even when total building footprints differ; the city total is "
            "sensitive to the size of the typed-building dataset. "
            "Underlying model: [InVEST Urban Cooling Model]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
        )
    econ4.metric(
        "Cooling Energy Savings",
        f"${_energy_savings / 1e6:.2f}M/yr",
        delta=(
            f"+${_energy_savings / 1e6:.2f}M/yr vs baseline"
            if _energy_savings >= 1e3 else "no avoided energy cost"
        ),
        delta_color="normal" if _energy_savings >= 1e3 else "off",
        help=_help_text,
    )
    _confidence_caption(econ4, "medium")
    # Per-pixel rate as a small secondary caption — only when the city
    # total is meaningful. Suppresses at HD-only scenarios where there's
    # no cooling delta to amortize.
    _rate_str = _fmt_per_pixel_rate(_per_pixel_cooling_usd)
    if _rate_str is not None and _energy_savings >= 1e3:
        econ4.caption(_rate_str)
else:
    if BUILDINGS_DATA_AVAILABLE and not BUILDINGS_HAVE_TYPES:
        _help_text = (
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "Building-type data not "
            "available for this extent — requires per-building type codes "
            "(InVEST sample uses 0=other, 1=commercial, 2=residential, "
            "3=industrial) to look up energy_consumption.csv kWh/(m²·°C) rates. "
            "OSM-only buildings don't carry these codes. Spatial placement "
            "mask is still active. "
            "Underlying model: [InVEST Urban Cooling Model]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
        )
    else:
        _help_text = (
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "ET raster, energy table, "
            "or buildings shapefile not loaded — see "
            "data/invest/cooling/UrbanCooling_sample_data/. "
            "Underlying model: [InVEST Urban Cooling Model]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
        )
    econ4.metric("Cooling Energy Savings", "—", help=_help_text)
    _confidence_caption(econ4, "medium")

# Avoided Carbon Cost — deterministic from carbon_tons_co2_yr × EPA SCC.
# Always available regardless of buildings/ET data, since it's purely a
# function of the converted-pixel carbon flux.
_avoided_carbon = results.get('avoided_carbon_cost_usd', 0.0)
econ5.metric(
    "Avoided Carbon Cost",
    f"${_avoided_carbon / 1e6:.2f}M/yr" if abs(_avoided_carbon) >= 1e4 else f"${_avoided_carbon:,.0f}/yr",
    delta=(
        f"+${_avoided_carbon / 1e6:.2f}M/yr vs baseline" if _avoided_carbon >= 1e4
        else "$0/yr vs baseline" if abs(_avoided_carbon) < 1
        else f"${_avoided_carbon:,.0f}/yr vs baseline"
    ),
    delta_color="normal" if _avoided_carbon >= 1 else "off",
    help=(
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        f"Annual carbon value at EPA Social Cost of Carbon (${EPA_SOCIAL_COST_CARBON}/ton CO2e, "
        "EPA 2023 final rule, 2 % discount rate, 2030 emissions). Represents "
        "the estimated economic damage avoided per ton of CO2e sequestered "
        "based on federal guidelines. Linear in `carbon_tons_co2_yr` so "
        "scales directly with the carbon-rate sliders in Advanced Settings."
    ),
)
_confidence_caption(econ5, "medium")

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
    help=f"Confidence: Medium — see 'How this prototype works' for tier definitions. Implementation cost divided by runoff reduction vs baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). N/A if scenario increases runoff or has no cost."
)
_confidence_caption(ceff1, "medium")
ceff2.metric(
    "Cost / °F Cooling",
    _fmt_ce(ce['cost_per_degf']),
    delta=None,
    delta_color="off" if _cooling_f <= 0 else "normal",
    help="Confidence: Medium — see 'How this prototype works' for tier definitions. Implementation cost divided by degrees F of cooling vs baseline. N/A if no cooling improvement."
)
_confidence_caption(ceff2, "medium")
ceff3.metric(
    "Cost / 1,000 People Fed",
    _fmt_ce(ce['cost_per_1k_people']),
    delta=None,
    help="Confidence: Medium — see 'How this prototype works' for tier definitions. Implementation cost divided by (people fed ÷ 1,000). N/A if no food production."
)
_confidence_caption(ceff3, "medium")

st.caption(
    "For outcome metrics, higher is generally better except Runoff Volume, where "
    "lower is better. For cost-effectiveness ratios, lower cost per unit of "
    "benefit is better."
)

with st.expander("Baseline vs Scenario Comparison", expanded=False):
    # read from state to avoid silent-staleness if city switches
    _baseline_flood = 100 - _CURRENT_CITY_STATE.baseline_cn
    _runoff_diff    = results['runoff_acre_feet'] - BASELINE_RUNOFF_ACRE_FEET
    _flood_diff     = results['flood_reduction'] - _baseline_flood

    _flood_damage_avoided = results.get('flood_damage_avoided_usd', 0.0)
    _energy_savings_table = results.get('cooling_energy_savings_usd', 0.0)
    _avoided_carbon_table = results.get('avoided_carbon_cost_usd', 0.0)
    comparison_data = {
        'Metric': [
            'Flood Risk Reduction', 'Runoff Volume', 'Temperature Change',
            'Food Production', 'Carbon Sequestration', 'NDVI',
            'Flood Damage Avoided', 'Cooling Energy Savings', 'Avoided Carbon Cost',
        ],
        'Baseline': [
            f'{_baseline_flood:.1f}',
            f'{BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft',
            'Reference',
            '0 lbs',
            '0 tons CO2e/yr',
            f'{BASELINE_NDVI:.3f}',
            '$0',
            '$0/yr',
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
            f'{results["mean_ndvi"]:.3f}',
            f'${_flood_damage_avoided / 1e6:.1f}M',
            f'${_energy_savings_table / 1e6:.2f}M/yr',
            f'${_avoided_carbon_table / 1e6:.2f}M/yr' if abs(_avoided_carbon_table) >= 1e4 else f'${_avoided_carbon_table:,.0f}/yr',
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
            f'{results["mean_ndvi"] - BASELINE_NDVI:+.3f}',
            f'+${_flood_damage_avoided / 1e6:.1f}M' if _flood_damage_avoided >= 1e4 else '$0',
            f'+${_energy_savings_table / 1e6:.2f}M/yr' if _energy_savings_table >= 1e3 else '$0/yr',
            f'+${_avoided_carbon_table / 1e6:.2f}M/yr' if _avoided_carbon_table >= 1e4 else f'+${_avoided_carbon_table:,.0f}/yr' if _avoided_carbon_table >= 1 else '$0/yr',
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
    st.dataframe(_styled, width='stretch', hide_index=True)

with st.expander("Assumptions and limitations"):
    _assumption_tabs = st.tabs([
        "Flood & Runoff", "Temperature", "Food", "Carbon",
        "Mental Health", "Costs",
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
        _temp_calibration = (
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per CC unit. "
            f"Values come from the InVEST UCM args JSON for the Minneapolis AOI "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`, humid continental Köppen Dfa). "
            "Treat the °F output as ±2 °F at best.\n"
            if selected_city.startswith("Minneapolis") else
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per CC unit. "
            f"No published InVEST args exist for hot semi-arid Köppen BSh; "
            f"values are an estimate from regional UHI literature "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`). "
            "Treat the °F output as ±2 °F at best — uncertainty is larger "
            "here than for MN.\n"
        )
        st.markdown(
            "- **Method:** InVEST Urban Cooling Model. Per-pixel Cooling "
            "Capacity `CC = 0.6·shade + 0.2·albedo + 0.2·ETI`, then Gaussian-"
            "smoothed over a 450 m kernel so cooling propagates onto "
            "neighbouring pixels (per InVEST `green_area_cooling_distance`).\n"
            "- **Reported value:** mean(CC) across the AOI, labeled CC. This "
            "approximates but is not identical to the canonical InVEST Heat "
            "Mitigation Index (HMI) — see UCM_AUDIT.md.\n"
            + _temp_calibration +
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
            "1.76–18.2 for managed food forests) — adjust the Food Forest "
            "and Green Infrastructure carbon-rate sliders in **Advanced Settings**.\n"
            "- **Not locally calibrated.** Refine with site-specific data when "
            "available."
        )
    with _assumption_tabs[4]:
        st.markdown(
            "- **Method:** InVEST Urban Mental Health Model (v3.19.0). "
            "Per-pixel `ΔNE = NE_scenario − NE_baseline` (NE = NDVI Gaussian-"
            "smoothed with σ = 300 m / 30 m px = 10 px, matching InVEST canonical "
            "behavior), "
            "`RR = exp(ln(RR₀.₁) × 10 × ΔNE)`, "
            "`PC = (1 − RR) × baseline_prevalence × population`. Two outcomes "
            "are summed: depression and anxiety.\n"
            "- **Effect sizes** from Liu et al. 2023 meta-analysis on green "
            "space and mental health: RR per 0.1 NDVI = 0.96 (depression) / "
            "0.97 (anxiety) — i.e. 4 % / 3 % reduction per 0.1 NDVI gain.\n"
            "- **Baseline prevalence (US):** 21 % depression, 19 % anxiety "
            "(CDC 2023). These are best interpreted as ever-diagnosed / "
            "lifetime prevalence; using them with the InVEST formula treats "
            "them as the at-risk pool.\n"
            "- **Cost-of-illness:** \\$8,467/depression case, \\$5,765/anxiety "
            "case (US nominal). InVEST docs cite ~\\$11K USD-PPP/case as a "
            "default — our values are slightly lower.\n"
            "- **Caveats:** NDVI is a synthetic per-NLCD-class proxy here, "
            "not satellite-derived; baseline-vs-scenario comparison assumes "
            "the population raster is unchanged across scenarios; the model "
            "captures only the *direct* exposure pathway, not air-quality or "
            "social-cohesion mechanisms.\n"
            "- **Not in the surrogate** — UMH outputs are computed live but "
            "are now in the surrogate target list (REQUIRED_TARGET_COLUMNS), "
            "so future training cycles will pick them up."
        )
    with _assumption_tabs[5]:
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
    _applied_idx = st.session_state.get("applied_suggestion")
    if _applied_idx is not None:
        _banner_msg = (
            f"Sliders updated to match suggestion #{_applied_idx + 1}. "
            "Switch to Tradeoff Analysis to verify."
        )
    else:
        _banner_msg = (
            "Optimization complete — switch to the Tradeoff Analysis tab to see results."
        )
    banner_col, dismiss_col = st.columns([5, 1])
    with banner_col:
        st.info(_banner_msg)
    with dismiss_col:
        if st.button("✕", key="dismiss_optimize_banner"):
            st.session_state.just_optimized = False
            st.rerun()

mode_text = f"using {PLACEMENT_STRATEGY_LABELS[placement_strategy].lower()}"
st.write(
    f"This scenario converts **{pct_converted}%** of developed land, allocating "
    f"**{green_infrastructure_pct}%** to green infrastructure, "
    f"**{food_forest_pct}%** to food forest, and **{pct_highdensity}%** "
    f"to high-density development, {mode_text}."
)

tab1, tab2, tab3, tab4 = st.tabs(["Scenario", "Tradeoff Analysis", "Map View", "Reference"])

with tab1:
    st.subheader("Outcome Comparison")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # read from state to avoid silent-staleness if city switches
        _baseline_cn_local = _CURRENT_CITY_STATE.baseline_cn
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(['Baseline', 'This Scenario'],
               [_baseline_cn_local, results['mean_cn']],
               color=['#5b8db8', '#7b4fa6'])
        ax.axhline(_baseline_cn_local, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Flood Risk', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Curve Number\n(lower = less runoff)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col2:
        # read from state to avoid silent-staleness if city switches
        _baseline_hm_local = _CURRENT_CITY_STATE.baseline_hm
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(['Baseline', 'This Scenario'],
               [_baseline_hm_local, results['mean_hm']],
               color=['#5b8db8', '#7b4fa6'])
        ax.axhline(_baseline_hm_local, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Urban Cooling', fontsize=16, fontweight='bold')
        ax.set_ylabel('Cooling Capacity\n(higher = more cooling)', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
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
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(5, 5))
        _max_carbon = max(scenario_df['carbon_tons_co2_yr'].max() * 1.1, 1.0)
        ax.bar(['Baseline', 'This Scenario'],
               [0, results['carbon_tons_co2_yr']],
               color=['#5b8db8', '#7b4fa6'])
        ax.set_title('Carbon Sequestration', fontsize=16, fontweight='bold')
        ax.set_ylabel('Carbon (tons CO2e/year)\n(higher = more sequestration)', fontsize=12)
        ax.set_ylim(0, _max_carbon)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
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
    ), width='stretch')

    if TRACTS_DATA_AVAILABLE:
        st.divider()
        st.markdown("#### Neighborhood breakdown")
        st.caption(
            "Top 5 most-improved Census tracts under this scenario, ranked by "
            "temperature change (°F cooler). Population-weighted within each tract."
        )
        _tracts_summary = compute_per_tract_summary(results['scenario_lulc'])
        if not _tracts_summary.empty:
            _top5 = (
                _tracts_summary
                .sort_values("Temp Δ (°F cooler)", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            st.dataframe(_top5, width='stretch', hide_index=True)
        else:
            st.caption("No tract-level data could be computed for this scenario.")

    st.divider()
    st.markdown("#### Best scenarios by goal")
    st.caption("From the pre-computed scenario library — not surrogate predictions.")

    # Best-scenarios-by-goal uses the lookup table when High Resolution mode
    # built one; otherwise falls back to the scenario_df the active mode is
    # using (Fast prototype: ~90 scenarios; Balanced: ~726 scenarios).
    lookup_df = pd.DataFrame(lookup_table.values()) if lookup_table else scenario_df
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
            saved["placement_strategy"] = placement_strategy
            saved["heat_priority"] = use_heat_priority  # backward compat for older saves
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
                             width='stretch', hide_index=True)
            st.dataframe(opt[display_cols].rename(columns=_col_rename),
                         width='stretch', hide_index=True)
            st.caption(
                "Note: suggestions with small amounts of High Density (2–10%) may "
                "reflect surrogate approximation — consider setting HD to 0% when applying."
            )

            st.markdown("#### Input Influence")
            st.caption("**Influence Map** — which input drives outcomes most according to the surrogate model:")
            st.plotly_chart(plot_feature_importance(surrogate), use_container_width=True)

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
                'placement_strategy',
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

            st.dataframe(df_saved[show_cols], width='stretch', hide_index=True)

            st.caption(
                "Note: saved scenarios are lost on page refresh — download the CSV to keep them."
            )

            if st.button("Clear saved scenarios"):
                st.session_state.saved_scenarios = []
                st.rerun()

with tab3:
    st.subheader("Where Changes Happen")
    if placement_strategy != 'random':
        st.info(
        f"**{PLACEMENT_STRATEGY_LABELS[placement_strategy]}** — conversions weighted "
        "toward higher-suitability pixels. Notice the spatial pattern shift vs. random allocation."
        )

    overlay_opacity = st.slider(
        "Heat vulnerability overlay opacity",
        0.0, 0.5, 0.2, 0.05,
        help=(
            "Transparency of the heat vulnerability overlay on the map. "
            "Currently uses high-intensity developed pixels (NLCD class 23) "
            "as a proxy for heat-vulnerable areas — this is a placeholder "
            "for a future CDC/ATSDR Heat Vulnerability Index by census "
            "tract. Set to 0 to hide."
        ),
    )

    render_matplotlib(plot_spatial_map(
        results['scenario_lulc'], cooling_lulc,
        heat_overlay=equity_weights, overlay_alpha=overlay_opacity,
    ))
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary. Orange wash = heat vulnerability proxy "
        "(darker orange = higher NLCD development intensity: 23 > 22 > 21), "
        "opacity controlled by the slider above."
    )

    with st.expander("Assumptions and limitations", expanded=False):
        st.markdown(
            "Conversions target feasible interstitial spaces — building footprints "
            "and road infrastructure are excluded citywide using OpenStreetMap "
            "road network data unioned with the city's buildings shapefile. "
            "The remaining candidate pool covers parking lots, lawns, and vacant "
            "land within the NLCD-21/22/23/24 developed mask. Placement within "
            "that pool is random by default, or weighted toward specific objectives "
            "(flood, cooling, equity, or balanced) via the sidebar placement picker. "
            "Real implementation would "
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