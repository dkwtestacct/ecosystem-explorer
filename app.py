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
# SCS design storm depth — per-city, set after city_cfg is built (see
# "── City-derived constants ──" below alongside UHI_MAX_C and FOOD_FOREST_LBS_ACRE).
# Brief 23: MN uses 3.94" (100 mm per NatCap MN args.json); SA uses 6.18" (157 mm
# per NatCap SA README). Kept in inches because the SCS-CN runoff formula is
# imperial-form (S = 1000/CN - 10). DESIGN_STORM_MM is the derived display form.
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

# ── "What's new" in-app changelog ──────────────────────────────────────────────
# A small changelog for returning visitors. Each entry clears a strict bar:
# the change *happened* (not queued/upcoming), is from the past ~7 days, would
# be noticed by a returning user, and reads as one line without internal
# vocabulary or specific parameter values. Forward-looking work goes in
# UNDERWAY_ENTRIES, which renders only when non-empty.
WHATS_NEW_ENTRIES = [
    "San Antonio flood estimates now use NatCap's San Antonio Curve Numbers instead of Minneapolis values. Under SA's design-storm conditions, Green Infrastructure scenarios show minimal flood mitigation — GI's primary benefits in SA are heat, nature access, and carbon. (NatCap, 2023.)",
    "San Antonio land cover now uses NatCap's San Antonio data.",
    "San Antonio carbon now uses NatCap's four-pool storage framework — reported as one-time storage value rather than annual rate.",
]

UNDERWAY_ENTRIES = []

ON_THE_RADAR = """\
- **AlphaEarth Foundations satellite embeddings** as a future land-cover source — [feasibility research here](https://github.com/dkwtestacct/ecosystem-explorer/blob/main/ALPHAEARTH_FEASIBILITY.md).
"""

def _build_whats_new():
    sections = []
    if WHATS_NEW_ENTRIES:
        sections.append("### What's new\n" + "\n".join(f"- {e}" for e in WHATS_NEW_ENTRIES))
    if UNDERWAY_ENTRIES:
        sections.append("### Underway\n" + "\n".join(f"- {e}" for e in UNDERWAY_ENTRIES))
    sections.append("### On the radar\n" + ON_THE_RADAR.rstrip())
    return "\n\n".join(sections)

WHATS_NEW = _build_whats_new()

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
    st.session_state.active_example_scenario = 'balanced'
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
    # First paint and every city switch reset the sliders to their defaults
    # (10 / 50 / 50 via the setdefaults below), which is the Balanced preset —
    # so re-highlight that button rather than clearing the active example.
    st.session_state.active_example_scenario = 'balanced'
    # Brief A.2: also reset cross-city optimizer state. Without these, an MN
    # optimizer result visibly persists into the SA dashboard view (the
    # post-optimize success banner stays up, and the Optimized Scenario
    # Suggestions section keeps rendering MN's results table).
    st.session_state.optimized_results = None
    st.session_state.just_optimized = False
    st.session_state._prev_city_key = selected_city

# ── City-derived constants ────────────────────────────────────────────────────
# Values that depend on the active city's climate / project parameters.
#   uhi_max_c — InVEST UCM urban heat-island anomaly (°C). MN: 2.05 from the
#     InVEST args JSON for the MN AOI; SA: 11 from NatCap SA README canonical
#     (Brief 14 migration; replaced the prior 3.5 °C estimate).
#   food_forest_lbs_acre — annual yield benchmark for the food-forest land
#     cover. MN: 11,500 (NatCap MN benchmark); SA: 8,500 placeholder pending
#     project-report numbers for the pecan/fig/mulberry/nopal mix.
UHI_MAX_C            = city_cfg['uhi_max_c']
HM_TO_FAHRENHEIT     = UHI_MAX_C * 1.8
FOOD_FOREST_LBS_ACRE = city_cfg['food_forest_lbs_acre']
DESIGN_STORM_INCHES  = float(city_cfg['design_storm_inches'])
DESIGN_STORM_MM      = DESIGN_STORM_INCHES * 25.4  # derived display form (round to 0.1 in tooltips)

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
        "Biophysical table is NatCap's compound NLCD×NLUD×tree-canopy "
        "lookup (`ucm__nlcd_nlud_tree.csv`, 1,984 rows), keyed on the "
        "compound LULC raster. San Antonio UNA and Carbon also use "
        "NatCap compound-keyed biophysical tables."
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

# In-app changelog for returning visitors — collapsed by default so the
# dashboard (sliders + metric cards) is the first view, not a changelog.
# Sits between the title and the city subheader. Wrapped in a bordered
# container for card-like visual separation.
with st.container(border=True):
    with st.expander("What's new", expanded=False):
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

    # Optional compound LULC + crosswalk (Brief 27). Both must exist when
    # `compound_lulc_file` is set; otherwise neither is required.
    compound_lulc = city_cfg.get("compound_lulc_file")
    crosswalk    = city_cfg.get("crosswalk_file")
    if compound_lulc or crosswalk:
        if not (compound_lulc and crosswalk):
            missing.append(
                "`compound_lulc_file` and `crosswalk_file` must be set together"
            )
        else:
            cl_path = Path(f"{city_cfg['data_dir_flood']}/{compound_lulc}").resolve()
            cw_path = Path(f"{city_cfg['data_dir_flood']}/{crosswalk}").resolve()
            if not cl_path.exists():
                missing.append(f"`compound_lulc_file` resolves to a missing file: {cl_path}")
            if not cw_path.exists():
                missing.append(f"`crosswalk_file` resolves to a missing file: {cw_path}")
            for k in ('default_ff_lucode', 'default_gi_lucode', 'default_hd_lucode'):
                if city_cfg.get(k) is None:
                    missing.append(f"`{k}` must be set when compound LULC is used")

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
    nlcd_intensity_weights: np.ndarray
    shade_arr: np.ndarray
    kc_arr: np.ndarray
    albedo_arr: np.ndarray
    green_area_arr: np.ndarray
    # InVEST UNA per-LULC `urban_nature` proportion (Brief 29). Sized
    # `max_una_lucode + 1`; for SA that's 1,984 (compound), for MN ~96
    # (NLCD). Indexed by `scenario_lulc_una` — compound for SA, NLCD for MN.
    urban_nature_arr: np.ndarray
    # InVEST Carbon four-pool stock per LULC (Brief 30, SA only — None for
    # MN). Each pool in tons C/ha, sized to `max_carbon_lucode + 1` (1,984
    # for SA's compound table). Indexed by `scenario_lulc_carbon`, the
    # carbon-view scenario raster (compound for SA, NLCD for MN). MN uses
    # the single-rate annual proxy via `CARBON_SEQ_RATES` and so leaves
    # these four fields at None.
    c_above_arr: Optional[np.ndarray]
    c_below_arr: Optional[np.ndarray]
    c_soil_arr: Optional[np.ndarray]
    c_dead_arr: Optional[np.ndarray]
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
    baseline_hm_raster: np.ndarray
    baseline_ne_raster: np.ndarray
    # Per-Brief-9 placement strategy reformulation: canonical baseline rasters
    # consumed by `_compute_suitability_weights`.
    baseline_una_supply_percapita_raster: np.ndarray  # InVEST UNA `urban_nature_supply_percapita.tif`
    buildings_distance_raster: np.ndarray             # distance (px) to nearest building from BUILDINGS_RASTER
    # Baseline scalars — read via _CURRENT_CITY_STATE only (not aliased)
    baseline_hm: float
    baseline_cn: float
    # NatCap compound LULC framework (Brief 27, SA only — None for cities
    # without a `compound_lulc_file`). The full compound raster sits on the
    # prototype's 30 m grid; the reduction is already baked into
    # `cooling_lulc` (and `lulc`) above so Brief 27 metrics route unchanged.
    # The three `compound_after_*` arrays carry, per source compound lucode,
    # the target compound lucode that preserves NLUD+tree-canopy while
    # swapping NLCD to the conversion target (or DEFAULT_<target>_LUCODE
    # fallback). Briefs 28–30 will consume them inside evaluate_scenario.
    cooling_lulc_compound: Optional[np.ndarray]
    compound_to_nlcd: Optional[np.ndarray]
    # compound → NLCD×tree-canopy (3-digit `nlcd*10+tier`) lookup for the SA
    # flood CN path; None for cities without a compound LULC. See
    # `load_lulc_crosswalk` for the tier = max(tree, 1) mapping rationale.
    compound_to_nlcd_tree: Optional[np.ndarray]
    compound_after_ff: Optional[np.ndarray]
    compound_after_gi: Optional[np.ndarray]
    compound_after_hd: Optional[np.ndarray]
    # Brief B: parallel boolean arrays indexed by source compound lucode.
    # True = the source pixel's (NLUD, tree-canopy) had no matching row
    # in the crosswalk for the target NLCD; conversion fell back to
    # DEFAULT_<target>_LUCODE. Per-scenario fallback counts feed the
    # `*_fellback_pixels` result-dict keys and the SA dashboard's
    # "Conversion fidelity" panel.
    compound_after_ff_was_default: Optional[np.ndarray]
    compound_after_gi_was_default: Optional[np.ndarray]
    compound_after_hd_was_default: Optional[np.ndarray]


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
        f"Cooling °F is approximate (±2°F). Runoff uses a city-specific design "
        f"storm ({DESIGN_STORM_MM:.0f} mm / {DESIGN_STORM_INCHES:.2f} inches for "
        f"{selected_city}; NatCap per-city canonical). Cost is order-of-magnitude — "
        f"adjust $/acre sliders in sidebar."
    )
    st.markdown(
        "**Confidence tiers** — each metric card displays one of three badges "
        "under its value:  \n"
        "  \n"
        "- **High confidence** — Direct raster outputs grounded in published "
        "methodology (USDA SCS curve numbers, InVEST UCM Heat Mitigation Index). "
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


# ── NatCap compound LULC crosswalk (Brief 27, foundational) ──────────────────
# Loads `lulc_crosswalk.csv` and builds vectorized lookup arrays:
#   compound_to_nlcd[lucode]       → NLCD code (the reduction layer)
#   compound_after_<target>[lucode] → compound code of the same (NLUD,
#       tree-canopy) bin with NLCD swapped to the conversion target. Built
#       only for the three conversion-target NLCDs (FF=41, GI=90, HD=24).
#       When the source pixel's (NLUD, tree) tuple has no matching row for
#       the target NLCD, the array carries the configured
#       `DEFAULT_<target>_LUCODE` fallback. Currently used only at load
#       time for the reduction; Brief 28+ will consume the compound_after_*
#       arrays inside evaluate_scenario when per-model tables go compound-
#       keyed. See SA_INTEGRATION_PLAN.md Decision 2 and DESIGN_NOTES.md.
# NLCD classes the SA flood CN table (biophys_floodmitig_sa.csv) represents
# with a single, non-tiered 2-digit code rather than three canopy tiers. Per
# NatCap's QA canopy rules, water/ice are "None canopy only" and forests are
# "never None" — each therefore has a single canopy state, so the CN table
# carries one row (11, 12, 41, 42, 43) instead of a 1/2/3 tier triplet.
# `compound_to_nlcd_tree` emits the bare 2-digit code for these classes so the
# CN lookup resolves (emitting e.g. 411 would miss the table → CN 0).
_CN_FLOOD_SINGLE_CODE_NLCD = (11, 12, 41, 42, 43)


@st.cache_data
def load_lulc_crosswalk(crosswalk_path, default_ff, default_gi, default_hd):
    df = pd.read_csv(crosswalk_path)
    max_lucode = int(df['lucode'].max())

    compound_to_nlcd = np.full(max_lucode + 1, -1, dtype=np.int16)
    compound_to_nlcd[df['lucode'].astype(int).values] = df['nlcd'].astype(int).values

    # Build compound → NLCD×tree-canopy lookup for the SA flood CN path.
    # Encoding: nlcd_tree = nlcd * 10 + tier, where tier ∈ {1, 2, 3} maps from
    # the crosswalk's `tree` column (4 classes: None=0 / Low=1 / Medium=2 / High=3).
    #
    # Mapping choice: tier = max(tree, 1)
    #   tree=0 (None,    0% canopy) → tier 1
    #   tree=1 (Low,    15% canopy) → tier 1
    #   tree=2 (Medium, 40% canopy) → tier 2
    #   tree=3 (High,   66% canopy) → tier 3
    #
    # Why max(tree, 1):
    # The NatCap-provided CN table (biophys_floodmitig_sa.csv) has 3 tiers per
    # NLCD class, where tier 1 reproduces the TR-55 baseline CN (no canopy
    # benefit) and tier 3 is the maximum canopy-modulated reduction (e.g.,
    # Developed High Intensity tier 1 = 98, tier 3 = 77.4). The crosswalk has
    # 4 canopy classes. With one more class than tiers, the choice is whether
    # to collapse None+Low together or Medium+High together.
    #
    # Collapsing None+Low → tier 1 is the conservative choice (wet-side error):
    # 15% canopy is a marginal interception signal, and treating it as baseline
    # means we underclaim flood mitigation rather than overclaim it. The CN
    # reductions in the data support this — tier 1 values match the unmodified
    # TR-55 baseline, suggesting tier 1 was intended for "no meaningful canopy."
    #
    # Authoritative source caveat: the NatCap data delivery does not document
    # the tier↔canopy-class mapping explicitly. The SA README points to
    # `Ben NDR and Flood Mar_2023.pptx` for flood methodology, but that file is
    # not in the shared folders. The mapping above is a defensible methodology
    # choice pending NatCap clarification (logged as a question for the next
    # NatCap conversation; see NATCAP_COLLABORATION.md).
    #
    # Two notes on edge cases (not action items here):
    # - Barren (31): the QA notes say "Barren is only allowed None canopy," yet
    #   the CN file gives Barren 3 tiers (311/312/313). With max(tree, 1),
    #   nearly all Barren pixels → tier 1 anyway. Consistent with the QA intent.
    # - SA scenario codes 997/998/999 (Food Forest / Gardens) appear in the CN
    #   file but are never produced by the prototype's conversion path (which
    #   produces standard NLCD 41/90/24). Not currently reachable; noted for
    #   the record.
    nlcd_vals = df['nlcd'].astype(int).values
    tier_vals = np.maximum(df['tree'].astype(int).values, 1)
    nlcd_tree_vals = nlcd_vals * 10 + tier_vals
    single_mask = np.isin(nlcd_vals, _CN_FLOOD_SINGLE_CODE_NLCD)
    nlcd_tree_vals[single_mask] = nlcd_vals[single_mask]
    compound_to_nlcd_tree = np.full(max_lucode + 1, -1, dtype=np.int16)
    compound_to_nlcd_tree[df['lucode'].astype(int).values] = nlcd_tree_vals

    # Per-source-pixel "convert to target" lookups. Build once for each
    # conversion target by grouping crosswalk rows on (nlud_simple, tree)
    # and picking, for each (NLUD, tree) tuple present, the lucode row with
    # the target NLCD — preferring is_realistic_to_create=yes when multiple
    # rows match. Pixels whose (NLUD, tree) tuple has no row for the target
    # NLCD fall back to DEFAULT_<target>_LUCODE.
    #
    # Brief B: alongside each `compound_after_*` array, build a parallel
    # boolean `was_default_*` array indexed by source compound lucode.
    # True = this source pixel's (NLUD, tree-canopy) had no matching row
    # in the crosswalk for the target NLCD and the conversion fell back
    # to the configured DEFAULT_<target>_LUCODE. Used by conversion sites
    # in evaluate_scenario to count per-scenario fallback fractions.
    df['_create_ok'] = df['is_realistic_to_create'].astype(str).str.lower() == 'yes'
    targets = [(41, default_ff), (90, default_gi), (24, default_hd)]
    compound_after = {}
    was_default = {}
    for target_nlcd, fallback in targets:
        # First-match per (NLUD, tree) → target compound lucode, preferring
        # create_ok=yes rows, then ascending lucode as a deterministic
        # tiebreaker.
        target_rows = (df[df['nlcd'] == target_nlcd]
                       .sort_values(by=['_create_ok', 'lucode'],
                                    ascending=[False, True])
                       .drop_duplicates(subset=['nlud_simple', 'tree'],
                                        keep='first'))
        # Vectorized fill: every source row's (NLUD, tree) tuple is looked
        # up in the target's first-match map; pixels whose tuple has no
        # match fall back to the configured DEFAULT_<target>_LUCODE.
        key_to_lucode = dict(
            zip(zip(target_rows['nlud_simple'].astype(int),
                    target_rows['tree'].astype(int)),
                target_rows['lucode'].astype(int))
        )
        out = np.full(max_lucode + 1, fallback, dtype=np.int16)
        # was_default starts True for every index — only real source
        # lucodes whose (NLUD, tree) matched the crosswalk get flipped to
        # False below. Indices outside the actual lucode space stay True;
        # the conversion-site indexing only ever touches real source
        # lucodes (from the LULC raster) so those padding True values
        # never get counted.
        was_default_arr = np.ones(max_lucode + 1, dtype=bool)
        src_keys = list(zip(df['nlud_simple'].astype(int),
                            df['tree'].astype(int)))
        src_lucodes = df['lucode'].astype(int).values
        for i, key in enumerate(src_keys):
            if key in key_to_lucode:
                out[src_lucodes[i]] = key_to_lucode[key]
                was_default_arr[src_lucodes[i]] = False
        compound_after[target_nlcd] = out
        was_default[target_nlcd] = was_default_arr

    return (df, compound_to_nlcd, compound_to_nlcd_tree,
            compound_after[41], compound_after[90], compound_after[24],
            was_default[41], was_default[90], was_default[24])


def reduce_compound_to_nlcd(compound_raster, compound_to_nlcd):
    """Map a compound LULC raster to its per-NLCD reduction.

    Compound nodata (-1) is rewritten to the prototype's module-wide
    `NODATA` (-128) sentinel so downstream `(scenario_lulc != NODATA)`
    masks continue to work. Returned dtype is int16. Vectorized — no
    per-pixel loop."""
    max_lc = compound_to_nlcd.shape[0] - 1
    safe = np.where((compound_raster >= 0) & (compound_raster <= max_lc),
                    compound_raster, 0)
    return np.where(compound_raster == -1, NODATA,
                    compound_to_nlcd[safe]).astype(np.int16)


def reduce_compound_to_nlcd_tree(compound_raster, compound_to_nlcd_tree):
    """Map a compound LULC raster to its per-NLCD×tree-canopy reduction.

    Direct analogue of `reduce_compound_to_nlcd`, but the lookup yields the
    3-digit `nlcd*10 + tier` code (or bare 2-digit code for the single-canopy
    classes 11/12/41/42/43) used by SA's flood CN table. Compound nodata (-1)
    is rewritten to `NODATA` (-128) so downstream masks continue to work.
    Returned dtype is int16. Vectorized — no per-pixel loop."""
    max_lc = compound_to_nlcd_tree.shape[0] - 1
    safe = np.where((compound_raster >= 0) & (compound_raster <= max_lc),
                    compound_raster, 0)
    return np.where(compound_raster == -1, NODATA,
                    compound_to_nlcd_tree[safe]).astype(np.int16)


def _assert_raster_crs(src, expected_crs, file_path):
    """Assert that a loaded raster's CRS matches the city's canonical CRS.

    Raises ValueError with a clear message naming the offending file if
    the CRS doesn't match. Defense-in-depth against future data-
    integration mistakes — the prototype's area math assumes equal-area
    projections (EPSG:26915 for MN, EPSG:5070 for SA) and would silently
    produce wrong numbers if a 3857 raster (or any non-equal-area CRS)
    were introduced. See ARCHITECTURE.md "CRS handling" for the
    rationale. Comparison via `rasterio.crs.CRS.from_user_input` handles
    both EPSG-code and WKT-string representations consistently."""
    if src.crs is None:
        raise ValueError(
            f"Raster {file_path} has no CRS metadata. Expected {expected_crs}. "
            f"All rasters must declare their CRS explicitly."
        )
    expected = rasterio.crs.CRS.from_user_input(expected_crs)
    if src.crs != expected:
        raise ValueError(
            f"Raster {file_path} has CRS {src.crs}, expected {expected_crs}. "
            f"All rasters must be in the city's canonical equal-area CRS for "
            f"PIXEL_AREA_ACRES math to be correct. If this raster genuinely "
            f"belongs in a different CRS, reproject it at preparation time "
            f"(not runtime) — see ARCHITECTURE.md 'CRS handling'."
        )


@st.cache_data
def load_data(data_dir_flood, data_dir_cooling, cn_table_file, cooling_table_file,
              lulc_file, soil_file, cooling_lulc_file,
              una_table_file,
              expected_crs,
              compound_lulc_file=None, crosswalk_file=None,
              default_ff_lucode=None, default_gi_lucode=None, default_hd_lucode=None,
              carbon_table_file=None):
    bio = pd.read_csv(_resolve_table(data_dir_flood, cn_table_file, "data/flood"))

    cooling_bio = pd.read_csv(_resolve_table(data_dir_cooling, cooling_table_file, "data/cooling"))

    # InVEST UNA biophysical table (Brief 29: per-city). For MN it's the
    # NLCD-keyed sample bundle (~14 rows); for SA it's NatCap's compound
    # NLCD×NLUD×tree-canopy table (1,984 rows). The `urban_nature_arr`
    # array sized to `max_una_lucode + 1` enables a vectorized
    # `urban_nature_arr[scenario_lulc_una]` lookup that works at either
    # cardinality — the prior per-class boolean-mask loop (~14 passes
    # for MN, ~1,984 passes for SA) becomes one indexed read.
    una_bio = pd.read_csv(una_table_file)
    max_una_lucode = int(una_bio['lucode'].max())
    urban_nature_arr = np.zeros(max_una_lucode + 1, dtype=np.float32)
    for _, row in una_bio.iterrows():
        urban_nature_arr[int(row['lucode'])] = float(row['urban_nature'])

    # InVEST Carbon four-pool table (Brief 30: SA only). Each pool in tons
    # C/ha keyed on the compound `lucode` (0-1983 for SA). Four arrays sized
    # to `max_carbon_lucode + 1` enable a vectorized per-pixel stock lookup:
    # `(c_above_arr + c_below_arr + c_soil_arr + c_dead_arr)[scenario_lulc_carbon]`.
    # MN keeps the single-rate annual proxy via `CARBON_SEQ_RATES` — these
    # four arrays are None for cities without a `carbon_table_file`.
    c_above_arr = c_below_arr = c_soil_arr = c_dead_arr = None
    if carbon_table_file is not None:
        carbon_bio = pd.read_csv(carbon_table_file)
        max_carbon_lucode = int(carbon_bio['lucode'].max())
        c_above_arr = np.zeros(max_carbon_lucode + 1, dtype=np.float32)
        c_below_arr = np.zeros(max_carbon_lucode + 1, dtype=np.float32)
        c_soil_arr  = np.zeros(max_carbon_lucode + 1, dtype=np.float32)
        c_dead_arr  = np.zeros(max_carbon_lucode + 1, dtype=np.float32)
        for _, row in carbon_bio.iterrows():
            lc = int(row['lucode'])
            c_above_arr[lc] = float(row['c_above'])
            c_below_arr[lc] = float(row['c_below'])
            c_soil_arr[lc]  = float(row['c_soil'])
            c_dead_arr[lc]  = float(row['c_dead'])

    # Compound-LULC path (SA post-Brief 27): load NatCap's compound raster +
    # crosswalk, then derive the NLCD-reduced view that downstream consumers
    # see. Both rasters are kept in the city state; until per-model tables
    # go compound-keyed (Briefs 28–30) the reduced view drives every metric.
    cooling_lulc_compound = None
    compound_to_nlcd = None
    compound_to_nlcd_tree = None
    compound_after_ff = compound_after_gi = compound_after_hd = None
    compound_after_ff_was_default = None
    compound_after_gi_was_default = None
    compound_after_hd_was_default = None
    if compound_lulc_file is not None and crosswalk_file is not None:
        _compound_path = f'{data_dir_flood}/{compound_lulc_file}'
        with rasterio.open(_compound_path) as src:
            _assert_raster_crs(src, expected_crs, _compound_path)
            cooling_lulc_compound = src.read(1).astype(np.int16)
        # crosswalk_file is given relative to data_dir_flood (e.g.
        # '../natcap_2024/lulc_crosswalk.csv'), matching the existing
        # cooling_lulc_file convention.
        cw_path = f'{data_dir_flood}/{crosswalk_file}'
        (_xwalk_df, compound_to_nlcd, compound_to_nlcd_tree,
         compound_after_ff, compound_after_gi, compound_after_hd,
         compound_after_ff_was_default,
         compound_after_gi_was_default,
         compound_after_hd_was_default) = load_lulc_crosswalk(
             cw_path,
             int(default_ff_lucode),
             int(default_gi_lucode),
             int(default_hd_lucode))
        reduced = reduce_compound_to_nlcd(cooling_lulc_compound, compound_to_nlcd)
        # The reduced NLCD view replaces both `lulc` (flood) and
        # `cooling_lulc` since SA's prior config has them pointing at the
        # same raster. Downstream consumers continue to see NLCD codes.
        lulc = reduced.copy()
        cooling_lulc = reduced
    else:
        _lulc_path = f'{data_dir_flood}/{lulc_file}'
        with rasterio.open(_lulc_path) as src:
            _assert_raster_crs(src, expected_crs, _lulc_path)
            lulc = src.read(1)
        _cooling_path = f'{data_dir_cooling}/{cooling_lulc_file}'
        with rasterio.open(_cooling_path) as src:
            _assert_raster_crs(src, expected_crs, _cooling_path)
            cooling_lulc = src.read(1)

    _soil_path = f'{data_dir_flood}/{soil_file}'
    with rasterio.open(_soil_path) as src:
        _assert_raster_crs(src, expected_crs, _soil_path)
        soil = src.read(1)

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

    # ── NLCD intensity proxy raster ────────────────────────────────────────────
    # Used by the Map View "heat vulnerability" overlay. NLCD 23→1.0, 22→0.6, 21→0.3.
    # TODO: replace with real heat vulnerability index (e.g. CDC/ATSDR HVI by census tract).
    # Renamed from `equity_weights` in Brief 9 — was previously misused as a
    # building-proximity proxy in cooling-focused; that path now uses the real
    # distance-to-buildings raster (see _BUILDINGS_DISTANCE_RASTER).
    nlcd_intensity_weights = np.zeros(cooling_lulc.shape, dtype=np.float32)
    nlcd_intensity_weights[cooling_lulc == 23] = 1.0   # high-intensity developed
    nlcd_intensity_weights[cooling_lulc == 22] = 0.6
    nlcd_intensity_weights[cooling_lulc == 21] = 0.3

    return (lulc, soil_resized, cooling_lulc, developed_pixels,
            cn_table, lucode_idx_arr, hm_arr, max_raster_lucode, max_hm_lucode,
            nlcd_intensity_weights, shade_arr, kc_arr, albedo_arr, green_area_arr,
            urban_nature_arr,
            c_above_arr, c_below_arr, c_soil_arr, c_dead_arr,
            cooling_lulc_compound, compound_to_nlcd, compound_to_nlcd_tree,
            compound_after_ff, compound_after_gi, compound_after_hd,
            compound_after_ff_was_default,
            compound_after_gi_was_default,
            compound_after_hd_was_default)


# ── Population raster loader (for Nature Access metric) ──────────────────────
# Helper used by `_load_city_runtime_state` below. Built offline by
# download_census_pop.py from US Census 2020 block-level totals, rasterized to
# the NLCD grid. The loader falls back to a uniform placeholder if the file is
# missing so the app still launches before the pipeline has run.
def load_population_data(pop_path, target_shape, expected_crs):
    """Load a population-count raster, resampled to target_shape with bilinear."""
    with rasterio.open(pop_path) as src:
        _assert_raster_crs(src, expected_crs, pop_path)
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
# Cost of Greenhouse Gases" (Nov 2023). Multiplied by `carbon_tons_co2`
# to get a carbon-value dollar metric at the federal-guideline rate. SA
# uses one-time stock value × SC-CO2 (Vibrant Land framing); MN uses
# annual flow × SC-CO2 (legacy avoided-damage framing). See Brief 30.
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

# (Retired Brief 9: `_DYNAMIC_NATURE_LUCODES` and `_compute_access_score_raster_pure`
# supported a homegrown "reachability proxy" `_BASELINE_ACCESS_SCORE_RASTER` that
# the equity-focused placement strategy consumed. After Brief 9 reformulated
# equity-focused as `undersupply-focused` driven by the canonical InVEST UNA
# `urban_nature_supply_percapita` raster, the proxy chain had zero consumers and
# was deleted. The canonical UNA pipeline below is the only consumer of UNA
# data in the app.)


# ── Canonical InVEST Urban Nature Access (UNA) — numpy 2SFCA ─────────────────
# Re-implements natcap.invest.urban_nature_access (uniform search radius +
# configurable decay) in numpy — the same approach `_compute_hmi_raster` takes
# for the InVEST UCM. The model runs inside the app's own environment (no
# natcap.invest runtime dependency); the numpy result is validated offline
# against `natcap.invest.urban_nature_access.execute()`. Parameter rationale is
# in DESIGN_NOTES.md.
#
# Per-city parameters (Brief 22): NatCap maintains two project framings — MN
# project uses `demand=250 / radius=1000m / decay=exponential` (aspirational
# targets); SA project uses `demand=16.7 / radius=800m / decay=dichotomy`
# (WHO-minimum, heat-wave). Per-city alignment is the NatCap pattern.
UNA_DEMAND_M2_PER_CAPITA = float(city_cfg['una_demand_m2_per_capita'])
UNA_SEARCH_RADIUS_M      = float(city_cfg['una_search_radius_m'])
UNA_DECAY_FUNCTION       = str(city_cfg['una_decay_function'])

# Brief 29: `URBAN_NATURE_PROPORTION` (Python dict) retired in favour of the
# vectorized `urban_nature_arr` (np.float32) built inside `load_data` and
# carried on `CityState`. The dict pattern was fine at MN's ~14 codes but
# untenable at SA's 1,984 compound lucodes — the prior per-class boolean-mask
# loop would do 1,984 raster-wide comparisons per `_una_supply_percapita`
# call. The module-level alias is rebound by the `_CURRENT_CITY_STATE`
# fan-out block below; `_una_supply_percapita` reads it as a bare name.

# 2SFCA convolution kernel. Built per the configured decay function exactly
# as natcap.invest.urban_nature_access calls pygeoprocessing.kernels:
#   * dichotomy  — binary disk of radius `search_radius / pixel_size` pixels.
#                  pygeoprocessing.kernels.dichotomous_kernel(
#                      max_distance=search_radius_in_pixels, normalize=False)
#   * exponential — k(d) = exp(-d / expected_distance) for d ≤ max_distance else 0,
#                  where expected_distance = search_radius_in_pixels and
#                  max_distance = ceil(search_radius_in_pixels) * 2 + 1
#                  (matches pygeoprocessing.kernels.exponential_decay_kernel as
#                  natcap.invest.urban_nature_access calls it).
_UNA_RADIUS_PX = UNA_SEARCH_RADIUS_M / PIXEL_SIZE_M
if UNA_DECAY_FUNCTION == 'dichotomy':
    _UNA_APOTHEM = int(np.floor(_UNA_RADIUS_PX))
    _una_yy, _una_xx = np.mgrid[
        -_UNA_APOTHEM:_UNA_APOTHEM + 1, -_UNA_APOTHEM:_UNA_APOTHEM + 1]
    _UNA_KERNEL = (np.hypot(_una_yy, _una_xx) <= _UNA_RADIUS_PX).astype(np.float32)
    del _una_yy, _una_xx
elif UNA_DECAY_FUNCTION == 'exponential':
    _UNA_MAX_DIST = int(np.ceil(_UNA_RADIUS_PX)) * 2 + 1
    _UNA_APOTHEM  = int(np.ceil(_UNA_MAX_DIST))
    _una_yy, _una_xx = np.mgrid[
        -_UNA_APOTHEM:_UNA_APOTHEM + 1, -_UNA_APOTHEM:_UNA_APOTHEM + 1]
    _una_d = np.hypot(_una_yy, _una_xx)
    _UNA_KERNEL = np.where(
        _una_d <= _UNA_MAX_DIST,
        np.exp(-_una_d / _UNA_RADIUS_PX),
        0.0
    ).astype(np.float32)
    del _una_yy, _una_xx, _una_d
else:
    raise ValueError(
        f"Unknown UNA decay function {UNA_DECAY_FUNCTION!r}; "
        f"valid: {{'dichotomy', 'exponential'}}. Check `city_cfg['una_decay_function']`."
    )


def _una_convolve(signal):
    """Zero-padded 2-D convolution with the dichotomy disk kernel, matching
    `pygeoprocessing.convolve_2d(ignore_nodata_and_edges=False)` as InVEST UNA
    uses it: edges are zero-padded (not edge-corrected), then the negative-value
    clamp of InVEST's `_convolve_and_set_lower_bound` is applied."""
    out = _fftconvolve(signal, _UNA_KERNEL, mode="same")
    np.clip(out, 0.0, None, out=out)
    return out


def _una_supply_percapita_pure(scenario_lulc, pop_count_raster, urban_nature_arr):
    """Pure variant of `_una_supply_percapita` — takes `urban_nature_arr`
    explicitly so the loader can call it before the module-level alias is
    rebound. The zero-deps wrapper below reads the module alias. Mirrors
    the `_compute_hmi_raster` / `_compute_hmi_raster_pure` pattern from
    Brief 28b (see CLAUDE.md "Pure-variant helpers").

    `scenario_lulc` is the UNA-view raster (compound for SA post-Brief-29,
    NLCD for MN); `urban_nature_arr` is sized to match (compound for SA,
    NLCD for MN) so a single vectorized indexed read
    `urban_nature_arr[safe]` retrieves the per-pixel proportion regardless
    of cardinality. Both nodata sentinels are handled via `>= 0`: NLCD
    rasters use -128, compound rasters use -1 (see `reduce_compound_to_nlcd`).

    Returns `(supply_percapita, valid_mask)`. `supply_percapita` is m² of urban
    nature available per capita reachable from each pixel; `valid_mask` is the
    modelable extent (valid-LULC pixels — InVEST masks LULC and population to
    their common valid extent before convolving)."""
    valid = (scenario_lulc >= 0)
    pixel_area_m2 = float(PIXEL_SIZE_M * PIXEL_SIZE_M)

    # Population masked to the modelable extent; off-extent population counts as
    # 0, exactly as InVEST's `masked_population` feeds the convolution.
    pop = np.where(valid, np.asarray(pop_count_raster, dtype=np.float64), 0.0)

    # Urban-nature area per pixel = urban_nature_proportion × pixel area.
    # Vectorized lookup (Brief 29) — was a Python for-loop over a dict of
    # ~14 NLCD codes; with SA's compound table the dict pattern would have
    # done 1,984 raster-wide boolean comparisons per call.
    safe = np.clip(scenario_lulc, 0, len(urban_nature_arr) - 1)
    nature_area = urban_nature_arr[safe].astype(np.float64) * pixel_area_m2
    nature_area[~valid] = 0.0

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


def _una_supply_percapita(scenario_lulc, pop_count_raster):
    """Zero-deps wrapper that reads the module-level `urban_nature_arr` alias
    populated by the post-cache_resource fan-out. Downstream code (everything
    except the in-loader baseline computation) calls this variant."""
    return _una_supply_percapita_pure(
        scenario_lulc, pop_count_raster, urban_nature_arr)


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

    Re-implements `natcap.invest.urban_nature_access` (uniform search
    radius + configurable decay, with per-city parameters — see
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
    'random':              'Uniform random sampling',
    'flood-focused':       'Prioritize pixels with highest per-pixel runoff Q_{p,i} (InVEST UFR canonical)',
    'cooling-focused':     'Prioritize pixels with low HMI near buildings (canonical HMI + distance to BUILDINGS_RASTER)',
    'undersupply-focused': 'Prioritize pixels with the largest per-capita UNA supply deficit (InVEST UNA canonical)',
    'balanced':            'Equal-weight normalized combination of the three focused strategies',
}

# Human-readable labels for the sidebar radio. Keys must match PLACEMENT_STRATEGIES.
PLACEMENT_STRATEGY_LABELS = {
    'random':              'Random placement',
    'flood-focused':       'Prioritize flood-prone areas',
    'cooling-focused':     'Prioritize hot areas near buildings',
    'undersupply-focused': 'Prioritize areas with unmet nature demand',
    'balanced':            'Balanced approach',
}

# Backward-compatibility shim — saved scenarios from before the Brief 9
# reformulation may carry the legacy key 'equity-focused'. Map transparently
# to the canonical reformulated key on read; remove after one schema cycle.
_LEGACY_PLACEMENT_STRATEGY_ALIASES = {
    'equity-focused': 'undersupply-focused',
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
        # Per-pixel runoff Q_{p,i} for the design storm — InVEST UFR's
        # canonical signal for "this pixel produces a lot of runoff."
        # Higher runoff = more potential benefit from greening (greening
        # lowers CN, which lowers Q). See InVEST UFR user guide eq. 127.
        lulc_vals = np.clip(cooling_lulc[rows, cols], 0, len(lucode_idx_arr) - 1)
        soil_vals = np.clip(soil_resized[rows, cols].astype(int), 1, cn_table.shape[1] - 1)
        pixel_cn = cn_table[lucode_idx_arr[lulc_vals], soil_vals].astype(np.float64)
        # SCS-CN runoff equation. CN is dimensionless; output Q is in mm.
        s_max = 25400.0 / np.maximum(pixel_cn, 1e-6) - 254.0  # mm
        p_mm = DESIGN_STORM_INCHES * 25.4  # mm
        ia = 0.2 * s_max
        q_mm = np.where(p_mm > ia, (p_mm - ia) ** 2 / (p_mm + 0.8 * s_max), 0.0)
        weights = np.maximum(q_mm, 0.0)

    elif strategy == 'cooling-focused':
        # Two signals: (a) heat exposure = (1 − HMI) using the canonical
        # InVEST UCM Heat Mitigation Index raster (validated at MAE=0 against
        # natcap.invest.urban_cooling_model.execute()), and (b) real
        # distance-to-buildings from BUILDINGS_RASTER via a distance
        # transform precomputed at module load (see Stage D in Brief 9).
        # Pixels closer to buildings save more AC energy when cooled.
        heat = 1.0 - _BASELINE_HM_RASTER[rows, cols].astype(np.float64)
        heat = np.maximum(heat, 0.0)

        bldg_dist = _BUILDINGS_DISTANCE_RASTER[rows, cols].astype(np.float64)
        proximity = 1.0 / (1.0 + bldg_dist)  # 1.0 on a building pixel; decays with distance

        weights = heat * proximity

    elif strategy == 'undersupply-focused':
        # Per-pixel unmet nature demand per InVEST UNA's canonical framing:
        # `urban_nature_balance_percapita = supply_percapita − demand` per
        # pixel (UNA user guide; SUP_DEM_{i,cap}). Pixels with negative
        # balance are "undersupplied" — those are the candidates for new
        # nature. Weight = magnitude of the per-capita deficit (zero where
        # balance ≥ 0). No population multiplier — adequacy of per-capita
        # access is what InVEST UNA measures.
        supply = _BASELINE_UNA_SUPPLY_PERCAPITA_RASTER[rows, cols].astype(np.float64)
        deficit = np.maximum(UNA_DEMAND_M2_PER_CAPITA - supply, 0.0)
        weights = deficit

    elif strategy == 'balanced':
        # Equal-weight combination of the three focused strategies.
        # Normalize each component to sum to 1, then average.
        flood_w = _compute_suitability_weights(convertible_pixels, 'flood-focused')
        cool_w = _compute_suitability_weights(convertible_pixels, 'cooling-focused')
        undersupply_w = _compute_suitability_weights(convertible_pixels, 'undersupply-focused')

        def _safe_normalize(w):
            s = w.sum()
            return w / s if s > 0 else np.ones_like(w) / len(w)

        weights = (_safe_normalize(flood_w)
                   + _safe_normalize(cool_w)
                   + _safe_normalize(undersupply_w)) / 3.0
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

    if weight_sum == 0:
        # Fallback to uniform random if all weights are zero
        return rng.choice(len(convertible_pixels), size=n_to_convert, replace=False)

    # Saturation fallback: when a strategy's suitability surface has fewer
    # non-zero-weighted pixels than n_to_convert (e.g. equity-focused at
    # pct >= ~35% on Minneapolis downtown, where ~64% of convertible pixels
    # have pop=0), `rng.choice(replace=False, p=weights)` would raise
    # ValueError. We take all the non-zero pixels first (preserving strategy
    # intent for the pixels the strategy can rank — sampled in weighted-
    # priority order, same convention as the non-saturated path) and fill
    # the remainder uniformly from the zero-weighted pool.
    # See PLACEMENT_STRATEGY_DIAGNOSTIC.md §3 and §7.
    nonzero_mask = weights > 0
    nonzero_count = int(nonzero_mask.sum())
    if nonzero_count < n_to_convert:
        nonzero_idx = np.flatnonzero(nonzero_mask)
        zero_idx = np.flatnonzero(~nonzero_mask)
        nonzero_weights = weights[nonzero_mask]
        nonzero_weights = nonzero_weights / nonzero_weights.sum()
        chosen_nonzero = rng.choice(
            nonzero_idx, size=nonzero_count, replace=False, p=nonzero_weights
        )
        n_remainder = n_to_convert - nonzero_count
        chosen_zero = rng.choice(zero_idx, size=n_remainder, replace=False)
        return np.concatenate([chosen_nonzero, chosen_zero])

    # Normal weighted-sample path
    weights /= weight_sum
    return rng.choice(len(convertible_pixels), size=n_to_convert, replace=False, p=weights)


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
    # Brief 9 backward compat: saved scenarios from before the placement-strategy
    # reformulation may carry the legacy 'equity-focused' key. Map transparently
    # to the canonical reformulated key on entry. Remove after one schema cycle.
    placement_strategy = _LEGACY_PLACEMENT_STRATEGY_ALIASES.get(
        placement_strategy, placement_strategy
    )

    pct_highdensity = 100 - green_infrastructure_pct - food_forest_pct

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

    # Brief 28b: branch on whether this city has a NatCap compound LULC view
    # (SA after Brief 27) or only the NLCD view (MN). For SA, UCM consumes
    # the compound view and conversions map source compound codes → target
    # compound codes via the `COMPOUND_AFTER_*` lookups so the (NLUD,
    # tree-canopy) bin is preserved. UFR / UNA / food / NDVI / MH still
    # operate on the NLCD reduction; for SA that's derived from the converted
    # compound raster via `reduce_compound_to_nlcd`. For MN, both views are
    # the same NLCD raster.
    # Brief B: per-target counts of conversions whose source pixel's
    # (NLUD, tree-canopy) tuple had no matching crosswalk row and fell
    # back to DEFAULT_<target>_LUCODE. Stay at 0 for MN (no compound
    # conversion path) so the result-dict schema is consistent across
    # cities. Surfaced in the SA dashboard's Conversion fidelity panel
    # and in `evaluate_scenario`'s return dict as `*_fellback_pixels`.
    ff_fellback_pixels = 0
    gi_fellback_pixels = 0
    hd_fellback_pixels = 0
    if cooling_lulc_compound is not None:
        scenario_lulc_compound = cooling_lulc_compound.copy()
        if n_wet > 0:
            p = pixels_to_convert[:n_wet]
            src = scenario_lulc_compound[p[:, 0], p[:, 1]]
            scenario_lulc_compound[p[:, 0], p[:, 1]] = COMPOUND_AFTER_GI[src]
            gi_fellback_pixels = int(COMPOUND_AFTER_GI_WAS_DEFAULT[src].sum())
        if n_for > 0:
            p = pixels_to_convert[n_wet:n_wet + n_for]
            src = scenario_lulc_compound[p[:, 0], p[:, 1]]
            scenario_lulc_compound[p[:, 0], p[:, 1]] = COMPOUND_AFTER_FF[src]
            ff_fellback_pixels = int(COMPOUND_AFTER_FF_WAS_DEFAULT[src].sum())
        if n_hd > 0:
            p = pixels_to_convert[n_wet + n_for:]
            src = scenario_lulc_compound[p[:, 0], p[:, 1]]
            scenario_lulc_compound[p[:, 0], p[:, 1]] = COMPOUND_AFTER_HD[src]
            hd_fellback_pixels = int(COMPOUND_AFTER_HD_WAS_DEFAULT[src].sum())
        scenario_lulc = reduce_compound_to_nlcd(scenario_lulc_compound, COMPOUND_TO_NLCD)
        scenario_lulc_ucm = scenario_lulc_compound
        scenario_lulc_una = scenario_lulc_compound
        scenario_lulc_carbon = scenario_lulc_compound
    else:
        scenario_lulc = cooling_lulc.copy()
        if n_wet > 0:
            p = pixels_to_convert[:n_wet]
            scenario_lulc[p[:, 0], p[:, 1]] = CODE_GREEN_INFRA
        if n_for > 0:
            p = pixels_to_convert[n_wet:n_wet + n_for]
            scenario_lulc[p[:, 0], p[:, 1]] = CODE_FOOD_FOREST
        if n_hd > 0:
            p = pixels_to_convert[n_wet + n_for:]
            scenario_lulc[p[:, 0], p[:, 1]] = CODE_HIGH_DENSITY
        scenario_lulc_compound = None
        scenario_lulc_ucm = scenario_lulc
        scenario_lulc_una = scenario_lulc
        scenario_lulc_carbon = scenario_lulc

    soil_clamped = np.clip(soil_resized, 1, 4)
    # CN lookup key: SA's flood CN table is keyed by NLCD × tree-canopy 3-digit
    # codes (NatCap NLCD×tree-canopy compound framework, see Ben NDR and Flood
    # Mar_2023.pptx). Reduce the compound raster to that space.
    # MN's table is plain 2-digit NLCD, so use scenario_lulc directly. Only
    # this CN lookup uses the tree-reduced view; every other SA metric continues
    # to use scenario_lulc / the compound views as before.
    if scenario_lulc_compound is not None:
        cn_lookup_lulc = reduce_compound_to_nlcd_tree(scenario_lulc_compound, COMPOUND_TO_NLCD_TREE)
    else:
        cn_lookup_lulc = scenario_lulc
    lulc_safe    = np.clip(cn_lookup_lulc, 0, len(lucode_idx_arr) - 1)
    lulc_idx     = lucode_idx_arr[lulc_safe]
    cn_scenario  = cn_table[lulc_idx, soil_clamped]
    mean_cn      = float(cn_scenario[cn_scenario > 0].mean().round(2))

    # Canonical InVEST UCM Heat Mitigation Index — HMI = max(CC_local,
    # CC_park). `mean_hm` is the mean HMI across valid pixels (0–1 scale,
    # higher = more cooling); the per-pixel value factors shade, albedo,
    # per-pixel ET, and exponentially distance-weighted cooling from green
    # areas ≥ 2 ha within the 450 m cooling distance.
    # Brief 28b: `scenario_lulc_ucm` is the compound view for SA (indexes
    # the compound-keyed `shade_arr` etc.) and the NLCD view for MN.
    hmi_map  = _compute_hmi_raster(scenario_lulc_ucm)
    valid_hm = hmi_map[~np.isnan(hmi_map) & (scenario_lulc != NODATA)]
    mean_hm  = float(valid_hm.mean().round(4))
    cooling_energy_savings_usd = compute_cooling_energy_savings(hmi_map)

    n_food_pixels = int(((scenario_lulc == CODE_FOOD_FOREST) & (cooling_lulc != CODE_FOOD_FOREST)).sum())
    food_mln_lbs  = round(n_food_pixels * PIXEL_AREA_ACRES * FOOD_FOREST_LBS_ACRE / 1_000_000, 3)

    # Brief 30: SA uses NatCap's four-pool stock framework (one-time stock
    # change in t CO2 from the LULC delta) per the Vibrant Land methodology;
    # MN keeps the per-conversion-type single-rate annual proxy. The
    # `carbon_tons_co2` return key is unified across cities — its temporal
    # framing (annual flow for MN, one-time stock for SA) is documented in
    # the dashboard card label, the schema log, and DESIGN_NOTES. Carbon-rate
    # sliders are MN-only by design (no rate per pool for SA — the stock is
    # the table's data, not a user input).
    if c_above_arr is not None:
        carbon_tons_co2 = _compute_carbon_four_pool(
            scenario_lulc_carbon, cooling_lulc_compound,
        )
    else:
        rate_ff = CARBON_SEQ_RATES[CODE_FOOD_FOREST] if carbon_rate_ff is None else carbon_rate_ff
        rate_gi = CARBON_SEQ_RATES[CODE_GREEN_INFRA] if carbon_rate_gi is None else carbon_rate_gi
        carbon_tons_co2 = round(
            n_for * PIXEL_AREA_ACRES * rate_ff
            + n_wet * PIXEL_AREA_ACRES * rate_gi
            + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
        )
    carbon_value_usd = round(carbon_tons_co2 * EPA_SOCIAL_COST_CARBON, 0)

    # Brief 29: `scenario_lulc_una` is the compound view for SA (indexes
    # the compound-keyed `urban_nature_arr`) and the NLCD view for MN.
    nat_pct, _nat_quality, nat_people = calculate_nature_access(
        scenario_lulc_una, pop_count_raster
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
        # Brief 30: unified field name (Option D.1). Semantics differ per
        # city — annual flow (t CO2e/yr) for MN, one-time stock change
        # (t CO2) for SA. Temporal framing surfaced via metric labels.
        'carbon_tons_co2':          carbon_tons_co2,
        'carbon_value_usd':         carbon_value_usd,
        'nature_access_pct':        nat_pct,
        'people_with_nature_access': nat_people,
        'preventable_mh_cases':     preventable_mh_cases,
        'avoided_mh_cost_usd':      avoided_mh_cost_usd,
        'food_mln_lbs':             food_mln_lbs,
        'people_fed':               food_to_people_fed(food_mln_lbs),
        'total_cost_mln':           total_cost_mln,
        # Brief B: per-target fallback-pixel counts (compound-conversion
        # diagnostic). 0 for MN (no compound conversion path); for SA, the
        # subset of `n_for` / `n_wet` / `n_hd` whose source pixel's (NLUD,
        # tree-canopy) had no matching crosswalk row for the target NLCD
        # and fell back to DEFAULT_<target>_LUCODE. Dashboard derives the
        # fallback fraction (`*_fellback_pixels / n_*`). Not a surrogate
        # target — pure metadata about the conversion.
        'ff_fellback_pixels':       ff_fellback_pixels,
        'gi_fellback_pixels':       gi_fellback_pixels,
        'hd_fellback_pixels':       hd_fellback_pixels,
        'scenario_name':            f"{pct_converted}% converted — GI {green_infrastructure_pct}% / FF {food_forest_pct}%",
        'scenario_lulc':            scenario_lulc,
        # Brief 28b: the UCM-view scenario raster. Compound (0–1983 lucodes)
        # for SA; the same array as `scenario_lulc` (NLCD) for MN. Consumers
        # that re-run the UCM helpers — chiefly `compute_per_tract_summary`
        # — must use this view so per-pixel `shade_arr[...]` lookups land in
        # the right lucode space.
        'scenario_lulc_ucm':        scenario_lulc_ucm,
        # Brief 29: the UNA-view scenario raster. Compound for SA, NLCD for
        # MN. Mirrors `scenario_lulc_ucm`'s role for UCM. Consumers that
        # re-run UNA helpers downstream of `evaluate_scenario` (currently
        # the lookup-refresh paths in `compute_scenario_grid` /
        # `compute_lookup_table` / `precompute_scenarios.py`) must pass
        # this view so per-pixel `urban_nature_arr[...]` lookups land in
        # the right lucode space.
        'scenario_lulc_una':        scenario_lulc_una,
        # Brief 30: the Carbon-view scenario raster. Compound for SA, NLCD
        # for MN. Mirrors the `scenario_lulc_ucm` / `_una` pattern. MUST
        # be stripped in all three CSV/dict consumers (`compute_scenario_grid`,
        # `compute_lookup_table`, `precompute_scenarios.py`) — see
        # CLAUDE.md "Interface changes require auditing all consumers".
        'scenario_lulc_carbon':     scenario_lulc_carbon,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
# Bump SCENARIO_SCHEMA_VERSION whenever the surrogate target columns change so
# Streamlit's @st.cache_data automatically invalidates stale grids/tables.
SCENARIO_SCHEMA_VERSION = 26  # bumped: Brief B adds three per-target fallback-pixel diagnostic keys to `evaluate_scenario`'s return dict — `ff_fellback_pixels`, `gi_fellback_pixels`, `hd_fellback_pixels`. For SA these count converted pixels whose source (NLUD, tree-canopy) tuple had no matching crosswalk row and fell back to DEFAULT_<target>_LUCODE (1310 / 122 / 341). For MN they're always 0 (no compound conversion path). Not surrogate targets — pure metadata about the conversion, surfaced in the SA dashboard's Conversion fidelity panel. Brief 30 was at 25.

# Surrogate target columns that downstream code (train_surrogate, optimize_scenario)
# requires. Listed explicitly so a missing column fails loudly instead of leaking
# into a KeyError deep in fit().
REQUIRED_TARGET_COLUMNS = [
    'flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
    'carbon_tons_co2', 'nature_access_pct',
    'preventable_mh_cases', 'avoided_mh_cost_usd',
]


def _compute_carbon(n_wet, n_for, n_hd):
    """Carbon sequestration at default rates — used at scenario-grid build time.

    MN-style annual-flow proxy: per-conversion-type CARBON_SEQ_RATES (3.5 t
    CO2e/acre/yr for FF, 2.0 for GI, 0 for HD) × converted area. Returns an
    annual rate. SA uses the four-pool stock framework via
    `_compute_carbon_four_pool` instead — see Brief 30.
    """
    return round(
        n_for * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_FOOD_FOREST]
        + n_wet * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_GREEN_INFRA]
        + n_hd  * PIXEL_AREA_ACRES * CARBON_SEQ_RATES[CODE_HIGH_DENSITY], 1
    )


# Pixel area in hectares — used by the four-pool stock formula (which takes
# t C/ha rates). NLCD 30 m grid: 900 m² = 0.09 ha. Computed alongside
# PIXEL_AREA_ACRES (0.2224 ac) and PIXEL_AREA_M2 (900 m²).
PIXEL_AREA_HA = 0.09


def _compute_carbon_four_pool_pure(
    scenario_lulc_carbon, baseline_lulc_carbon,
    c_above_arr, c_below_arr, c_soil_arr, c_dead_arr,
):
    """One-time stock-delta carbon between scenario and baseline LULC.

    Per InVEST canonical four-pool framework and NatCap's Vibrant Land
    methodology for SA. Sums above/below/soil/dead carbon per pixel under
    both scenario and baseline LULC, takes the delta, aggregates across
    all valid pixels, multiplies by pixel area (hectares) and the
    atomic-mass ratio (44/12) for tons CO2-equivalent.

    Returns total stock change in t CO2 (positive = gained, negative = lost).
    This is *not* an annual rate — it is the one-time stock change when
    land use changes (matches Vibrant Land Appendix 2: "we analyzed
    landscape carbon storage using the InVEST Carbon model").

    Pure variant — takes the four pool arrays explicitly so the loader can
    call it before the module-level aliases are rebound. Downstream code
    uses the zero-deps wrapper `_compute_carbon_four_pool` below.
    """
    n = len(c_above_arr)
    valid = (scenario_lulc_carbon >= 0) & (baseline_lulc_carbon >= 0)
    scen_safe = np.clip(scenario_lulc_carbon, 0, n - 1)
    base_safe = np.clip(baseline_lulc_carbon, 0, n - 1)

    scen_total = (c_above_arr[scen_safe] + c_below_arr[scen_safe]
                  + c_soil_arr[scen_safe] + c_dead_arr[scen_safe])
    base_total = (c_above_arr[base_safe] + c_below_arr[base_safe]
                  + c_soil_arr[base_safe] + c_dead_arr[base_safe])

    delta_t_C_per_ha = np.where(valid, scen_total - base_total, 0.0)
    total_t_C = float(delta_t_C_per_ha.sum()) * PIXEL_AREA_HA
    total_t_CO2 = total_t_C * (44.0 / 12.0)
    return round(total_t_CO2, 1)


def _compute_carbon_four_pool(scenario_lulc_carbon, baseline_lulc_carbon):
    """Zero-deps wrapper reading the module-level pool-array aliases populated
    by the post-cache_resource fan-out. Downstream code calls this variant."""
    return _compute_carbon_four_pool_pure(
        scenario_lulc_carbon, baseline_lulc_carbon,
        c_above_arr, c_below_arr, c_soil_arr, c_dead_arr,
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
                    # Brief 28b: also strip `scenario_lulc_ucm` — for SA it's
                    # a separate full-AOI compound raster; for MN it's the
                    # same object as `scenario_lulc` and stripping is a no-op.
                    # Brief 29: same logic for `scenario_lulc_una`.
                    # Brief 30: same logic for `scenario_lulc_carbon`.
                    row = {k: v for k, v in result.items()
                           if k not in ('scenario_lulc', 'scenario_lulc_ucm',
                                        'scenario_lulc_una', 'scenario_lulc_carbon')}
                    # Explicit recomputation guarantees the surrogate-target
                    # columns exist regardless of evaluate_scenario's return.
                    # Brief 30: MN re-normalises to `_compute_carbon` defaults
                    # (overriding any session-state rate slider); SA's stock
                    # value from `evaluate_scenario` is already canonical
                    # (no rate sliders apply — the four-pool table is the data).
                    if _state.c_above_arr is None:
                        row['carbon_tons_co2'] = _compute_carbon(
                            row['n_wet'], row['n_for'], row['n_hd']
                        )
                    nature_access_pct, _nature_quality, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc_una'], _state.pop_count_raster
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
                    # Brief 28b/29/30: strip `scenario_lulc_ucm`,
                    # `scenario_lulc_una`, and `scenario_lulc_carbon`
                    # alongside `scenario_lulc` (same logic as
                    # `compute_scenario_grid` — see comment there).
                    entry = {k: v for k, v in result.items()
                             if k not in ('scenario_lulc', 'scenario_lulc_ucm',
                                          'scenario_lulc_una', 'scenario_lulc_carbon')}
                    # Brief 30: MN re-normalises to defaults; SA's four-pool
                    # stock value is already canonical (see
                    # `compute_scenario_grid` comment).
                    if _state.c_above_arr is None:
                        entry['carbon_tons_co2'] = _compute_carbon(
                            entry['n_wet'], entry['n_for'], entry['n_hd']
                        )
                    nature_access_pct, _nature_quality, people_with_nature_access = calculate_nature_access(
                        result['scenario_lulc_una'], _state.pop_count_raster
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
#
# max_entries=1 forces eviction of the previously-cached city on switch.
# Without this cap, both cities' ~1.5 GB transient pipelines can be cached
# simultaneously, risking OOM on Streamlit Cloud's ~1 GB ceiling during
# rapid city-switching. Trade-off: every city switch becomes a cold load
# (~minute wait) rather than an instant cache hit. We prefer reliability
# over speed for the second-switch case.
@st.cache_resource(
    max_entries=1,
    show_spinner="Loading city data — first interaction may take a minute…",
)
def _load_city_runtime_state(city_key: str) -> CityState:
    cfg = CITIES[city_key]

    # ── Phase 1: cached load_data outputs ────────────────────────────────────
    (l_lulc, l_soil_resized, l_cooling_lulc, l_developed_pixels,
     l_cn_table, l_lucode_idx_arr, l_hm_arr, l_max_raster_lucode, l_max_hm_lucode,
     l_nlcd_intensity_weights, l_shade_arr, l_kc_arr, l_albedo_arr,
     l_green_area_arr,
     l_urban_nature_arr,
     l_c_above_arr, l_c_below_arr, l_c_soil_arr, l_c_dead_arr,
     l_cooling_lulc_compound, l_compound_to_nlcd, l_compound_to_nlcd_tree,
     l_compound_after_ff, l_compound_after_gi, l_compound_after_hd,
     l_compound_after_ff_was_default,
     l_compound_after_gi_was_default,
     l_compound_after_hd_was_default) = load_data(
        cfg['data_dir_flood'], cfg['data_dir_cooling'],
        cfg['cn_table_file'], cfg['cooling_table_file'],
        cfg['lulc_file'], cfg['soil_file'], cfg['cooling_lulc_file'],
        cfg['una_table_file'],
        cfg['crs'],
        compound_lulc_file=cfg.get('compound_lulc_file'),
        crosswalk_file=cfg.get('crosswalk_file'),
        default_ff_lucode=cfg.get('default_ff_lucode'),
        default_gi_lucode=cfg.get('default_gi_lucode'),
        default_hd_lucode=cfg.get('default_hd_lucode'),
        carbon_table_file=cfg.get('carbon_table_file'))

    # ── Phase 2: Population raster ──────────────────────────────────────────
    pop_file = cfg.get("pop_file")
    try:
        if pop_file is None:
            raise FileNotFoundError("pop_file not configured")
        pop_count_raster = load_population_data(pop_file, l_cooling_lulc.shape, cfg['crs'])
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
            _assert_raster_crs(et_src, cfg['crs'], et_file)
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

    # ── Phase 5: (deleted — was UNA biophysical filtering for the
    # `_BASELINE_ACCESS_SCORE_RASTER` homegrown reachability proxy, retired
    # in Brief 9 when undersupply-focused was reformulated to use the
    # canonical UNA per-capita supply deficit instead.) ──────────────────────

    # ── Phase 6: (deleted — was precomputed nature-distance .npy disk cache,
    # also part of the retired access-score raster pipeline.) ───────────────

    # ── Phase 7: Rasterization template ─────────────────────────────────────
    _ref_path = f"{cfg['data_dir_cooling']}/{cfg['cooling_lulc_file']}"
    with rasterio.open(_ref_path) as ref:
        _assert_raster_crs(ref, cfg['crs'], _ref_path)
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
    # Brief 28b: for cities with a NatCap compound UCM table (SA),
    # `l_shade_arr`/`l_kc_arr`/`l_albedo_arr`/`l_green_area_arr` are sized
    # to the compound lucode space (0–1983) and must be indexed by the
    # compound raster, not the NLCD-reduced view. For cities without a
    # compound view (MN), the NLCD raster is the right input.
    _ucm_baseline_lulc = (
        l_cooling_lulc_compound
        if l_cooling_lulc_compound is not None
        else l_cooling_lulc
    )
    baseline_hm_raster = _compute_hmi_raster_pure(
        _ucm_baseline_lulc, l_shade_arr, l_kc_arr, l_albedo_arr, et_resized, max_et_ref,
        l_green_area_arr,
    )
    baseline_ne_raster = _gaussian_filter(
        _lulc_to_ndvi_raster(l_cooling_lulc), sigma=_UMH_SIGMA_PX, mode="nearest",
    )
    # Canonical InVEST UNA `urban_nature_supply_percapita` at baseline — for
    # the undersupply-focused placement strategy (Brief 9 Stage D). Reuses
    # the canonical 2SFCA implementation via its `_pure` variant because
    # module aliases haven't been rebound yet (see CLAUDE.md "Pure-variant
    # helpers"). Brief 29: for cities with a NatCap compound UNA table
    # (SA), `l_urban_nature_arr` is sized to the compound lucode space
    # (0-1983) and MUST be indexed by the compound raster — same parity
    # with how `_ucm_baseline_lulc` selects above.
    _una_baseline_lulc = (
        l_cooling_lulc_compound
        if l_cooling_lulc_compound is not None
        else l_cooling_lulc
    )
    baseline_una_supply_percapita_raster, _ = _una_supply_percapita_pure(
        _una_baseline_lulc, pop_count_raster, l_urban_nature_arr,
    )
    baseline_una_supply_percapita_raster = baseline_una_supply_percapita_raster.astype(np.float32)
    # Distance-to-buildings raster (pixel units) — for the cooling-focused
    # placement strategy. `_distance_transform_edt` returns float64; cast
    # immediately to float32 to halve the per-AOI memory cost on SA.
    buildings_distance_raster = _distance_transform_edt(
        ~buildings_raster.astype(bool)
    ).astype(np.float32)

    # ── Phase 14: Baseline scalars ──────────────────────────────────────────
    valid_base_cc = baseline_hm_raster[~np.isnan(baseline_hm_raster)]
    baseline_hm = (
        float(valid_base_cc.mean().round(4))
        if valid_base_cc.size > 0 else float(cfg['baseline_hm'] or 0.0)
    )
    # Baseline CN must use the same lookup key as evaluate_scenario: SA's CN
    # table is NLCD × tree-canopy keyed (NatCap framework), so reduce the
    # compound baseline LULC to that space; MN uses plain 2-digit NLCD.
    if l_cooling_lulc_compound is not None:
        baseline_cn_lulc = reduce_compound_to_nlcd_tree(l_cooling_lulc_compound, l_compound_to_nlcd_tree)
    else:
        baseline_cn_lulc = l_cooling_lulc
    baseline_lulc_safe = np.clip(baseline_cn_lulc, 0, len(l_lucode_idx_arr) - 1)
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
        nlcd_intensity_weights=l_nlcd_intensity_weights,
        shade_arr=l_shade_arr, kc_arr=l_kc_arr, albedo_arr=l_albedo_arr,
        green_area_arr=l_green_area_arr,
        urban_nature_arr=l_urban_nature_arr,
        c_above_arr=l_c_above_arr, c_below_arr=l_c_below_arr,
        c_soil_arr=l_c_soil_arr, c_dead_arr=l_c_dead_arr,
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
        baseline_hm_raster=baseline_hm_raster,
        baseline_ne_raster=baseline_ne_raster,
        baseline_una_supply_percapita_raster=baseline_una_supply_percapita_raster,
        buildings_distance_raster=buildings_distance_raster,
        baseline_hm=baseline_hm, baseline_cn=baseline_cn,
        cooling_lulc_compound=l_cooling_lulc_compound,
        compound_to_nlcd=l_compound_to_nlcd,
        compound_to_nlcd_tree=l_compound_to_nlcd_tree,
        compound_after_ff=l_compound_after_ff,
        compound_after_gi=l_compound_after_gi,
        compound_after_hd=l_compound_after_hd,
        compound_after_ff_was_default=l_compound_after_ff_was_default,
        compound_after_gi_was_default=l_compound_after_gi_was_default,
        compound_after_hd_was_default=l_compound_after_hd_was_default,
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
nlcd_intensity_weights = _CURRENT_CITY_STATE.nlcd_intensity_weights
shade_arr           = _CURRENT_CITY_STATE.shade_arr
kc_arr              = _CURRENT_CITY_STATE.kc_arr
albedo_arr          = _CURRENT_CITY_STATE.albedo_arr
green_area_arr      = _CURRENT_CITY_STATE.green_area_arr
urban_nature_arr    = _CURRENT_CITY_STATE.urban_nature_arr
# Brief 30: InVEST Carbon four-pool arrays. None for cities without a
# `carbon_table_file` (MN); compound-sized (1,984) for SA. Indexed by the
# carbon-view scenario raster — compound for SA, NLCD for MN.
c_above_arr         = _CURRENT_CITY_STATE.c_above_arr
c_below_arr         = _CURRENT_CITY_STATE.c_below_arr
c_soil_arr          = _CURRENT_CITY_STATE.c_soil_arr
c_dead_arr          = _CURRENT_CITY_STATE.c_dead_arr
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
_BASELINE_HM_RASTER                     = _CURRENT_CITY_STATE.baseline_hm_raster
_BASELINE_NE_RASTER                     = _CURRENT_CITY_STATE.baseline_ne_raster
_BASELINE_UNA_SUPPLY_PERCAPITA_RASTER   = _CURRENT_CITY_STATE.baseline_una_supply_percapita_raster
_BUILDINGS_DISTANCE_RASTER              = _CURRENT_CITY_STATE.buildings_distance_raster
# NOTE: BASELINE_HM and BASELINE_CN are intentionally NOT aliased here. Read
# them as `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` everywhere
# downstream — see CityState comment above.
# NatCap compound LULC (Brief 27). All five are None for cities without a
# `compound_lulc_file`. Aliased for legacy bare-name reads; Brief 28+ will
# extend evaluate_scenario to consume `cooling_lulc_compound` and the three
# `COMPOUND_AFTER_*` arrays when per-model tables go compound-keyed.
cooling_lulc_compound = _CURRENT_CITY_STATE.cooling_lulc_compound
COMPOUND_TO_NLCD      = _CURRENT_CITY_STATE.compound_to_nlcd
COMPOUND_TO_NLCD_TREE = _CURRENT_CITY_STATE.compound_to_nlcd_tree
COMPOUND_AFTER_FF     = _CURRENT_CITY_STATE.compound_after_ff
COMPOUND_AFTER_GI     = _CURRENT_CITY_STATE.compound_after_gi
COMPOUND_AFTER_HD     = _CURRENT_CITY_STATE.compound_after_hd
# Brief B: parallel boolean arrays (same indexing as COMPOUND_AFTER_*).
# Consumed by evaluate_scenario's conversion sites to count per-scenario
# fallback fractions. None for cities without a crosswalk (MN).
COMPOUND_AFTER_FF_WAS_DEFAULT = _CURRENT_CITY_STATE.compound_after_ff_was_default
COMPOUND_AFTER_GI_WAS_DEFAULT = _CURRENT_CITY_STATE.compound_after_gi_was_default
COMPOUND_AFTER_HD_WAS_DEFAULT = _CURRENT_CITY_STATE.compound_after_hd_was_default
# Brief B: dashboard gate — co-extensive with `_CARBON_IS_STOCK` for
# current cities (only SA has compound conversion + four-pool carbon)
# but semantically more precise. Used to hide MN-irrelevant UI like the
# Conversion fidelity panel.
_COMPOUND_CONVERSION_ACTIVE = COMPOUND_AFTER_FF is not None

# Brief 30: city-conditional carbon framing flag — True when SA's four-pool
# stock framework is active (one-time t CO2 stock change per the Vibrant
# Land methodology); False for MN's per-conversion-type single-rate annual
# proxy. Drives dashboard card labels, unit suffixes, and delta strings
# wherever carbon appears. Read in the sidebar, metric cards, comparison
# table, radar plot, and optimizer panel — so it's defined here (once,
# right after the alias rebinding) rather than re-derived per call site.
_CARBON_IS_STOCK = c_above_arr is not None


def compute_per_tract_summary(scenario_lulc_ucm):
    """DataFrame with one row per tract: baseline + scenario temperature (°F)
    vs the global baseline, plus the difference (improvement).

    Brief 28b: takes the UCM-view scenario raster (compound for SA, NLCD for
    MN). The caller has both views in `results`; pick
    `results['scenario_lulc_ucm']`. Re-running `_compute_hmi_raster` against
    the NLCD view for SA would index the compound-keyed `shade_arr` with
    NLCD codes and silently produce wrong numbers."""
    if not TRACTS_DATA_AVAILABLE or len(TRACTS) == 0:
        return pd.DataFrame()

    hm_s_raster = _compute_hmi_raster(scenario_lulc_ucm)

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
# Brief C: High Resolution mode is an opt-in gate. The expensive
# compute_lookup_table call below would otherwise fire on the first
# rerun after a radio click, before the user can see any warning
# (the radio widget itself is rendered ~600 lines below this point,
# inside Advanced Settings). So we downgrade to Balanced for the
# actual compute until the user explicitly checks a confirmation box.
# The radio still shows the user's selection; the checkbox is consent
# to the 25-50 minute build.
_hi_res_confirmed = st.session_state.get("hi_res_confirmed", False)
_effective_model_quality = _requested_model_quality
if _requested_model_quality == "High resolution" and not _hi_res_confirmed:
    _effective_model_quality = "Balanced"
N_ESTIMATORS = SURROGATE_TREES[_effective_model_quality]

with st.spinner("Loading data and pre-computing scenarios..."):
    # The lookup table is the most expensive thing the app ever computes —
    # 2,541 scenarios × per-pixel rasters can take 25–50 minutes for SA
    # (3.4 M pixels). On Streamlit Cloud free tier (1 GB RAM, ~5 min
    # health-check window) that's fatal. So we now build it ONLY when the
    # user explicitly picks High Resolution mode AND confirms via the
    # opt-in checkbox in Advanced Settings. Fast prototype (default) and
    # Balanced both skip it entirely.
    #
    # The slider-response path (`if lookup_key in lookup_table` further
    # down) gracefully falls through to a fresh evaluate_scenario call when
    # the table is empty — slightly slower per-slider but functional. The
    # "Best scenarios by goal" section also falls back to scenario_df when
    # lookup_table is empty.
    if _effective_model_quality == "High resolution":
        lookup_table = compute_lookup_table(_CURRENT_CITY_STATE, selected_city, DATA_DIR_FLOOD, DATA_DIR_COOLING)
        scenario_df = pd.DataFrame(list(lookup_table.values()))
        ACTIVE_MODEL_QUALITY = "high"
    elif _effective_model_quality == "Balanced":
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
def _scenario_signature(df):
    """Lightweight signature for cache invalidation. Captures row count,
    column set, and per-column numeric sums to detect content changes
    without hashing the full dataframe."""
    cols = [
        'pct_converted',
        'green_infrastructure_pct',
        'food_forest_pct',
        'flood_reduction',
        'mean_hm',
        'food_mln_lbs',
        'runoff_acre_feet',
        'carbon_tons_co2',
        'nature_access_pct',
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return (len(df), ())
    sums = df[cols].fillna(0).sum().to_numpy()
    return (
        len(df),
        tuple(cols),
        tuple(np.round(sums, 4).tolist()),
    )


@st.cache_resource
def _cached_train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling,
                            scenario_signature,
                            mode_key="fast", n_estimators=100):
    # Brief A.1: scenario_signature is the cache-key-visible representation of
    # _scenario_df's contents. Without it, the leading-underscore _scenario_df
    # is skipped by Streamlit's hasher, so a regenerated dense CSV (same city,
    # same mode) would silently return a stale surrogate trained on old data.
    # mode_key + n_estimators participate in the cache key so changing the
    # Model quality mode radio in the sidebar automatically retrains on the
    # new training set without needing a manual cache clear.
    return _train_surrogate_fn(_scenario_df, n_estimators=n_estimators)


surrogate = _cached_train_surrogate(
    scenario_df, DATA_DIR_FLOOD, DATA_DIR_COOLING,
    _scenario_signature(scenario_df),
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
        legend_handles.append(Patch(facecolor=(1.0, 140/255, 0.0, 0.6), label='Development-intensity heat proxy'))

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
                    f"Flood: {r.flood_reduction:.1f} | HMI: {r.mean_hm:.4f} | "
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
                    f"Flood: {r.flood_reduction:.1f} | HMI: {r.mean_hm:.4f}"
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
                    f"HMI: {r.mean_hm:.4f} [{r.hm_lower:.4f}–{r.hm_upper:.4f}]<br>"
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
        xaxis_title='Flood Retention (higher = better)',
        yaxis_title='Heat Mitigation Index (higher = better)',
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
st.session_state.setdefault("slider_gi_pct", 50)
st.session_state.setdefault("slider_ff_pct", 50)

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

st.sidebar.caption(
    "Default view illustrates a balanced 50/50 mix at 10% conversion. "
    "Adjust the sliders or use a Quick Start preset to explore alternatives."
)

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
    'balanced':     (10, 50,  50),
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

if st.sidebar.button("Balanced",
                     type="primary" if _active == 'balanced' else "secondary"):
    st.session_state._pending_pct = 10
    st.session_state._pending_gi = 50
    st.session_state._pending_ff = 50
    st.session_state.active_example_scenario = 'balanced'
    st.rerun()
st.sidebar.caption("Default view — 50/50 nature-based mix")

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

st.sidebar.caption(
    "Set targets the optimizer must satisfy. The sliders define minimum "
    "acceptable performance (flood reduction, cooling, food, carbon) or "
    "cap an unwanted outcome (runoff). The optimizer searches for "
    "scenarios that meet all targets at once."
)

with st.sidebar.container(border=True):

    # Flood slider max uses the precomputed grid's actual achievable maximum
    # rather than the theoretical 0–100 ceiling, so the slider range
    # represents reachable targets. Round up to the next 5 for headroom.
    _flood_achievable_max = int(scenario_df['flood_reduction'].max())
    _flood_slider_max = ((_flood_achievable_max + 4) // 5) * 5
    _flood_default = max(0, _flood_slider_max - 10)
    min_flood  = st.slider(
        "Flood reduction ≥",
        0, _flood_slider_max, _flood_default, 5,
        help=f"Corresponds to the Flood Retention metric card. Baseline is {100 - _CURRENT_CITY_STATE.baseline_cn:.1f}. Higher values mean less runoff — increasing this target will also reduce Runoff Volume in ac-ft.",
    )
    # read from state to avoid silent-staleness if city switches
    _baseline_hm_local = _CURRENT_CITY_STATE.baseline_hm
    # Cooling slider max uses the precomputed grid's actual achievable maximum
    # rather than the theoretical CC ceiling, so the slider range represents
    # reachable targets. +0.2 °F headroom.
    _cool_achievable_max = (scenario_df['mean_hm'].max() - _baseline_hm_local) * HM_TO_FAHRENHEIT
    _cool_slider_max = round(_cool_achievable_max + 0.2, 1)
    min_cool_f = st.slider(
        "Cooling ≥ (°F vs baseline)",
        min_value=-1.0, max_value=_cool_slider_max,
        value=0.1, step=0.1,
        help="Corresponds to the Temperature Change metric card. Set to 0.1 for at least 0.1°F cooler than baseline."
    )
    min_cool   = _baseline_hm_local + min_cool_f / HM_TO_FAHRENHEIT   # HM units for surrogate
    min_food   = st.slider("Food production ≥ (M lbs)", 0.0, float(max(MAX_FOOD, 0.1)), 0.0, 0.01,
        help="Corresponds directly to the Food Production metric card value in M lbs/yr.")
    _runoff_min = float(scenario_df['runoff_acre_feet'].min())
    _runoff_max = float(scenario_df['runoff_acre_feet'].max())
    max_runoff = st.slider(
        "Runoff ≤ (ac-ft)",
        min_value=round(_runoff_min),
        max_value=round(_runoff_max),
        value=round(BASELINE_RUNOFF_ACRE_FEET),
        step=100,
        help=f"Scenarios must stay below this runoff volume. Baseline is approximately {BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft."
    )
    # Brief 30: SA framing = stock change (t CO2e); MN framing = annual flow.
    _opt_carbon_label = (
        "Carbon storage change ≥ (tons CO2e)" if _CARBON_IS_STOCK
        else "Carbon sequestration ≥ (tons CO2e/yr)"
    )
    _opt_carbon_help = (
        "Corresponds to the Carbon Storage Change metric card (one-time stock value). "
        "Baseline is 0."
        if _CARBON_IS_STOCK else
        "Corresponds to the Carbon Sequestration metric card. Counts only converted pixels; baseline is 0."
    )
    min_carbon = st.slider(
        _opt_carbon_label,
        0, int(scenario_df['carbon_tons_co2'].max()), 0, 100,
        help=_opt_carbon_help,
    )

    st.caption(
        "The optimizer uses a surrogate model — a fast approximation trained on pre-computed "
        "scenarios — to search 10,000 candidate strategies in seconds. Results are approximate; "
        "verify promising scenarios using the main sliders."
    )
    if lookup_table:
        st.sidebar.caption(
            "Slider results use a precomputed lookup table for faster response. "
            "The optimizer uses a separate surrogate model to search a much wider range of scenarios."
        )
    else:
        st.sidebar.caption(
            "Slider results are computed live in the current model-quality mode. "
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
        "chosen criterion. Balanced combines flood, cooling, and equity "
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
    # Brief C.1: carbon-rate sliders apply only to MN's per-conversion-type
    # annual proxy (`CARBON_SEQ_RATES`). SA's Carbon uses NatCap's
    # four-pool stock table directly — no per-pool override is exposed.
    # Hide the sliders for SA and seed session_state with the MN defaults
    # so downstream `st.session_state.carbon_rate_*` reads still work.
    if not _CARBON_IS_STOCK:
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
    else:
        st.session_state.setdefault("carbon_rate_ff", 3.5)
        st.session_state.setdefault("carbon_rate_gi", 2.0)
        st.caption(
            "Carbon for this city uses NatCap's four-pool carbon storage table. "
            "Annual sequestration-rate sliders are hidden because they don't apply."
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
    # Brief C.2: High Resolution mode is gated behind an explicit opt-in
    # checkbox. The compute_lookup_table build takes 25–50 minutes on SA
    # and is fatal on Streamlit Cloud's 1 GB tier. Until confirmed, the
    # app silently runs in Balanced mode (the `_effective_model_quality`
    # downgrade earlier in the script). The radio still shows the user's
    # selection so intent is preserved; the checkbox is consent to the
    # expensive build.
    if _requested_model_quality == "High resolution":
        st.warning(
            "High resolution rebuilds a 2,541-entry lookup table that takes "
            "25–50 minutes on San Antonio and is not recommended on Streamlit "
            "Cloud (1 GB worker tier). Use Balanced unless you're running locally."
        )
        st.checkbox(
            "Yes, build the high-resolution lookup table (~25–50 min)",
            key="hi_res_confirmed",
        )
        if not _hi_res_confirmed:
            st.caption("Running in Balanced mode until you confirm.")
    st.caption(f"Active: {len(scenario_df):,} training scenarios.")

# ── Main panel ─────────────────────────────────────────────────────────────────
# Lookup-overlay safety contract (Brief A.4):
#   compute_lookup_table is @st.cache_data-decorated with
#   `schema_version=SCENARIO_SCHEMA_VERSION` as a cache-key parameter, so any
#   bump to SCENARIO_SCHEMA_VERSION invalidates every cached entry. That means
#   the fields LOADED from the lookup row (flood_reduction, mean_hm,
#   runoff_acre_feet, nature_access_pct, n_wet/n_for/n_hd, MH fields, etc.)
#   are guaranteed schema-current — no defensive overwrite needed for them.
#   The overwrites below (scenario_lulc, food, NDVI, carbon, dollar metrics,
#   total_cost) are for fields that legitimately depend on per-rerun state
#   (cost sliders, carbon-rate sliders, fresh rasters), NOT for staleness
#   protection. Future devs: do NOT add new defensive overwrites for
#   surrogate-target fields — bump SCENARIO_SCHEMA_VERSION instead. Full
#   contract in DESIGN_NOTES.md "Lookup-overlay safety contract".
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
    # Brief 28b: also restore the UCM-view raster from _fresh so downstream
    # consumers (compute_per_tract_summary) can re-run the HMI helpers on the
    # right lucode-space view (compound for SA, NLCD for MN).
    results['scenario_lulc_ucm'] = _fresh['scenario_lulc_ucm']
    # Food values are recomputed live. The historical reason was that the
    # lookup table predated an n_food_pixels fix; that defense is now
    # redundant under the schema-version contract above, but kept for
    # stability — food is also cheap to recompute.
    results['food_mln_lbs'] = _fresh['food_mln_lbs']
    results['people_fed']   = _fresh['people_fed']
    results['mean_ndvi']    = _fresh['mean_ndvi']
    results['carbon_tons_co2']   = _fresh['carbon_tons_co2']
    results['carbon_value_usd']  = _fresh['carbon_value_usd']
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
    "high":             "High confidence",
    "medium":           "Medium confidence",
    "prototype":        "Prototype",
    # Methodology-specific descriptor (not a confidence tier) for SA carbon,
    # which uses NatCap's four-pool stock framework (Brief 30) rather than
    # the MN single-rate proxy. Brief 2.
    "natcap_four_pool": "Four-pool stock (NatCap framework)",
}

def _confidence_caption(col, tier):
    """Render the badge under a metric card.
    tier ∈ {'high', 'medium', 'prototype'} for confidence tiers, or a
    methodology descriptor key like 'natcap_four_pool' — see 'How this
    prototype works' expander for tier definitions."""
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

_carbon_value = results['carbon_tons_co2']

# Brief 30: `_CARBON_IS_STOCK` is set once after the city-state aliasing
# above; here we derive the dependent display strings.
_carbon_unit_suffix = "t CO2e" if _CARBON_IS_STOCK else "t CO2e/yr"

def _fmt_carbon(tons):
    """Compact carbon display — k notation kicks in at 1,000 t to avoid card truncation."""
    if abs(tons) >= 1000:
        return f"{tons / 1000:.1f}k {_carbon_unit_suffix}"
    return f"{tons:,.0f} {_carbon_unit_suffix}"

# Brief 2 (Approach Y): the SA four-pool stock card is bespoke, mirroring
# Brief 1's signed-card pattern — flip to a "Loss" label with a positive
# magnitude and a red ↑ delta when conversions reduce stored carbon. MN's
# annual sequestration flow is always ≥ 0, so it keeps the shared `_delta_pill`
# path and the "Carbon Sequestration" label. Lifting only the SA branch out of
# `_delta_pill` leaves the other three callers (flood, runoff, NDVI) untouched.
_CARBON_PILL_EPSILON = 1.0
if _CARBON_IS_STOCK:
    if _carbon_value < -_CARBON_PILL_EPSILON:
        _carbon_card_label = "Carbon Storage Loss"
        _carbon_value_str = _fmt_carbon(abs(_carbon_value))
        _carbon_delta_str = f"+{abs(_carbon_value):,.0f} t CO2e lost from conversions"
        _carbon_delta_color = "inverse"
    elif _carbon_value > _CARBON_PILL_EPSILON:
        _carbon_card_label = "Carbon Storage Change"
        _carbon_value_str = _fmt_carbon(_carbon_value)
        _carbon_delta_str = f"+{_carbon_value:,.0f} t CO2e stock change from conversions"
        _carbon_delta_color = "normal"
    else:
        _carbon_card_label = "Carbon Storage Change"
        _carbon_value_str = _fmt_carbon(_carbon_value)
        _carbon_delta_str = None
        _carbon_delta_color = "off"
else:
    _carbon_card_label = "Carbon Sequestration"
    _carbon_value_str = _fmt_carbon(_carbon_value)
    _carbon_delta_str, _carbon_delta_color = _delta_pill(
        _carbon_value, fmt=",.0f", suffix="t CO2e/yr from conversions", epsilon=1.0,
    )

if placement_strategy != 'random':
    st.caption(f"Placement: {PLACEMENT_STRATEGY_LABELS[placement_strategy]}")

st.markdown("#### Ecological")
eco1, eco2, eco3 = st.columns(3)
eco1.metric(
    "Flood Retention",
    f"{results['flood_reduction']:.1f}",
    delta=_flood_delta_str,
    delta_color=_flood_delta_color,
    help=(
        "Confidence: High — see 'How this prototype works' for tier definitions. "
        "Unitless index (0–100) based on the USDA Curve Number. Higher = less "
        f"runoff potential. Baseline is {100 - _CURRENT_CITY_STATE.baseline_cn:.1f} "
        "for the current AOI's developed land. "
        "Note: this is the app's CN-inversion index (100 − mean_CN), monotone "
        "with but not identical to InVEST UFR's canonical runoff retention "
        "index `rnf_rt_idx = mean(1 − Q/P)`. "
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
    help=f"Confidence: High — see 'How this prototype works' for tier definitions. Approximate temperature change vs baseline. Positive = cooler, negative = warmer. Derived from mean Heat Mitigation Index (HMI) under the InVEST UCM (calibration factor {HM_TO_FAHRENHEIT:.2f}°F/HMI unit, UHI_max = {UHI_MAX_C:.2f}°C; ±2°F accuracy). HMI is the canonical InVEST UCM output, validated at MAE = 0.0000 against `natcap.invest.urban_cooling_model.execute()`. Underlying model: [InVEST Urban Cooling Model](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)."
)
_confidence_caption(eco2, "high")
eco3.metric(
    "Runoff Volume",
    _fmt_runoff(results['runoff_acre_feet']),
    delta=_runoff_delta_str,
    delta_color=_runoff_delta_color,
    help=(
        "Confidence: High — see 'How this prototype works' for tier definitions. "
        f"Acre-feet of runoff generated by a {DESIGN_STORM_MM:.0f}-mm design storm ({DESIGN_STORM_INCHES:.2f} inches; NatCap per-city canonical). "
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
# `_carbon_card_label` is set above alongside the value/delta (Brief 2,
# Approach Y) so the SA loss-flip label survives.
_carbon_card_help = (
    (
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        "One-time stock change in landscape carbon storage from the LULC delta, "
        "computed via the InVEST four-pool framework (above-ground biomass + "
        "below-ground biomass + soil + dead organic matter), keyed on NatCap's "
        "compound NLCD×NLUD×tree-canopy biophysical table. This is a stock value "
        "in t CO2e — not an annual rate. Matches NatCap's Vibrant Land (2023) "
        "methodology for San Antonio. "
        "Underlying model: [InVEST Carbon Storage and Sequestration]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/carbonstorage.html)."
    )
    if _CARBON_IS_STOCK else
    (
        "Confidence: Prototype — see 'How this prototype works' for tier definitions. "
        "Annual CO2e sequestration from converted pixels only. "
        "Uses provisional regional USDA/IPCC rates: Food Forest 3.5 t CO2e/acre/yr, "
        "Green Infrastructure 2.0 t CO2e/acre/yr. "
        "Treat as directional only — refine with locally calibrated values. "
        "Loosely related model: [InVEST Carbon Storage and Sequestration]"
        "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/carbonstorage.html)."
    )
)
eco4.metric(
    _carbon_card_label,
    _carbon_value_str,
    delta=_carbon_delta_str,
    delta_color=_carbon_delta_color,
    help=_carbon_card_help,
)
_confidence_caption(eco4, "natcap_four_pool" if _CARBON_IS_STOCK else "prototype")
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
        f"Confidence: Medium — see 'How this prototype works' for tier definitions. "
        f"% of the selected city's modelable-extent population whose per-capita "
        f"nature supply meets the {UNA_DEMAND_M2_PER_CAPITA:g} m²/capita demand standard, "
        f"computed via canonical InVEST Urban Nature Access (2SFCA methodology). "
        f"Reports only the modelable-extent population — the remainder sits on "
        f"cooling-LULC nodata pixels InVEST cannot model. "
        f"Parameters: {UNA_SEARCH_RADIUS_M:g}m uniform search radius, "
        f"{UNA_DECAY_FUNCTION} decay (per the active city's NatCap project framing — "
        f"see DESIGN_NOTES.md). "
        f"Underlying model: [InVEST Urban Nature Access]"
        f"(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_nature_access.html)."
    ),
)
_confidence_caption(hs_na, "medium")

_mh_cases = results.get('preventable_mh_cases', 0.0)
_mh_cost  = results.get('avoided_mh_cost_usd', 0.0)
if _mh_cases >= _MH_CASES_PILL_EPSILON:
    _mh_cases_label = "Preventable MH Cases"
    _mh_cases_value = f'{_mh_cases:,.0f}'
    _mh_cases_delta = f"+{_mh_cases:,.0f} cases prevented"
    _mh_cases_color = "normal"      # green ↑
elif _mh_cases <= -_MH_CASES_PILL_EPSILON:
    _mh_cases_label = "Additional MH Cases"
    _mh_cases_value = f'{abs(_mh_cases):,.0f}'
    _mh_cases_delta = f"+{abs(_mh_cases):,.0f} cases induced"
    _mh_cases_color = "inverse"     # red ↑
else:
    _mh_cases_label = "Preventable MH Cases"
    _mh_cases_value = f'{_mh_cases:,.0f}'
    _mh_cases_delta = None
    _mh_cases_color = "off"
hs3.metric(
    _mh_cases_label,
    _mh_cases_value,
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
    _mh_cost_label = "Avoided MH Costs"
    _mh_cost_value = f'${_mh_cost / 1e6:.2f}M/yr'
    _mh_cost_delta = f"+${_mh_cost / 1e6:.2f}M/yr avoided"
    _mh_cost_color = "normal"
elif _mh_cost <= -_MH_COST_PILL_EPSILON:
    _mh_cost_label = "Added MH Costs"
    _mh_cost_value = f'${abs(_mh_cost) / 1e6:.2f}M/yr'
    _mh_cost_delta = f"+${abs(_mh_cost) / 1e6:.2f}M/yr added in costs"
    _mh_cost_color = "inverse"
else:
    _mh_cost_label = "Avoided MH Costs"
    _mh_cost_value = f'${_mh_cost / 1e6:.2f}M/yr'
    _mh_cost_delta = None
    _mh_cost_color = "off"
hs4.metric(
    _mh_cost_label,
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
        "Matches InVEST UMH's `preventable_cost.tif` output (paired with "
        "`preventable_cases.tif`). The card title 'Avoided MH Costs' is the "
        "app's framing; InVEST uses 'preventable cost' as the canonical name "
        "for the same quantity. "
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
    _n_typed_buildings = int(np.sum(BUILDINGS_TYPE_RASTER > 0))
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
            f"Industrial $100 per m²) joined to {_n_typed_buildings:,} typed "
            "building pixels. Scales with this scenario's runoff reduction vs "
            f"baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). Capped at $0 "
            "for scenarios that increase runoff. "
            "Note: InVEST UFR's analogous `serv_blt` indicator is explicitly "
            "described in InVEST docs as only an indicator of service in "
            "currency·m³ units, not an actual measure of damage or savings. "
            "This card converts to dollars by scaling potential damage "
            "(`aff_bld`) by the runoff retention fraction — a stronger framing "
            "than InVEST itself makes. Treat as directional. "
            "Underlying model: [InVEST Urban Flood Risk Mitigation]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)."
        ),
    )
    _confidence_caption(econ3, "medium")
elif BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES:
    # Typed buildings but no damage table (SA today). Brief 33 / Path C:
    # match NatCap's Vibrant Land (Guerry et al. 2023) methodology —
    # they used InVEST UFRM for SA but explicitly did not enable
    # `infrastructure_damage_loss_table_path`, reporting flood mitigation
    # as percent volume reduction instead. The card label, value, and
    # help text shift accordingly; the underlying
    # `avoided_flood_damage_usd` field stays at $0 (surrogate-training
    # compatibility, no schema change).
    econ3.metric(
        "Flood Retention",
        f"{results['flood_reduction']:.1f}%",
        delta=(
            f"+{results['flood_reduction']:.1f}% vs baseline"
            if results['flood_reduction'] > 0 else "no reduction"
        ),
        delta_color="normal" if results['flood_reduction'] > 0 else "off",
        help=(
            "Confidence: Medium — see 'How this prototype works' for tier definitions. "
            "Percent reduction in flood volume during the city's design storm "
            f"({DESIGN_STORM_INCHES:.1f} inches over 24 hours), computed via the "
            "SCS Curve Number method. "
            "NatCap's Vibrant Land report (Guerry et al. 2023) used InVEST UFRM "
            "for San Antonio but explicitly did not enable damage valuation; "
            "they reported flood mitigation as percent volume reduction. The "
            "prototype matches this methodology for SA. "
            "Underlying model: [InVEST Urban Flood Risk Mitigation]"
            "(https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html)."
        ),
    )
    _confidence_caption(econ3, "medium")
else:
    if BUILDINGS_DATA_AVAILABLE and not BUILDINGS_HAVE_TYPES:
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

# Brief 30: Carbon dollar metric. For SA = one-time stock value
# (Vibrant Land framing); for MN = annual avoided-cost flow. Label and
# value-suffix branch on the same `_CARBON_IS_STOCK` flag as the
# Carbon-quantity card above.
_carbon_value_dollars = results.get('carbon_value_usd', 0.0)
_dollar_period_suffix = "" if _CARBON_IS_STOCK else "/yr"

# Brief 1: when the scenario loses carbon (negative dollar value), flip the
# label to the loss/cost framing and show a positive magnitude, mirroring the
# signed MH cards. (MN's carbon is always >= 0, so the loss labels only
# surface for SA's four-pool stock model.)
if _carbon_value_dollars < 0:
    _carbon_dollar_label = "Carbon Storage Loss" if _CARBON_IS_STOCK else "Added Carbon Cost"
else:
    _carbon_dollar_label = "Carbon Storage Value" if _CARBON_IS_STOCK else "Avoided Carbon Cost"

def _fmt_carbon_dollars(usd):
    if abs(usd) >= 1e4:
        return f"${usd / 1e6:.2f}M{_dollar_period_suffix}"
    return f"${usd:,.0f}{_dollar_period_suffix}"

_carbon_dollar_value = _fmt_carbon_dollars(abs(_carbon_value_dollars))

if _carbon_value_dollars >= 1e4:
    _carbon_dollar_delta = f"+${_carbon_value_dollars / 1e6:.2f}M{_dollar_period_suffix} vs baseline"
elif abs(_carbon_value_dollars) < 1:
    _carbon_dollar_delta = f"$0{_dollar_period_suffix} vs baseline"
else:
    _carbon_dollar_delta = f"${_carbon_value_dollars:,.0f}{_dollar_period_suffix} vs baseline"

# Align color with the MH cards: green for benefit, red ("inverse") for loss,
# neutral ("off") only near zero.
if _carbon_value_dollars >= 1:
    _carbon_dollar_color = "normal"
elif _carbon_value_dollars <= -1:
    _carbon_dollar_color = "inverse"
else:
    _carbon_dollar_color = "off"

_carbon_dollar_help = (
    (
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        f"One-time landscape carbon value at EPA Social Cost of Carbon "
        f"(${EPA_SOCIAL_COST_CARBON}/ton CO2e, EPA 2023 final rule, 2 % discount "
        "rate, 2030 emissions). Computed as the InVEST four-pool stock change "
        "(NatCap's Vibrant Land methodology for San Antonio) × SC-CO2. Note "
        "the temporal framing differs from MN's annual carbon flow — SA matches "
        "the published NatCap SA methodology; MN matches what the prototype "
        "currently has."
    )
    if _CARBON_IS_STOCK else
    (
        "Confidence: Medium — see 'How this prototype works' for tier definitions. "
        f"Annual carbon value at EPA Social Cost of Carbon (${EPA_SOCIAL_COST_CARBON}/ton CO2e, "
        "EPA 2023 final rule, 2 % discount rate, 2030 emissions). Represents "
        "the estimated economic damage avoided per ton of CO2e sequestered "
        "based on federal guidelines. Linear in `carbon_tons_co2` so "
        "scales directly with the carbon-rate sliders in Advanced Settings."
    )
)
econ5.metric(
    _carbon_dollar_label,
    _carbon_dollar_value,
    delta=_carbon_dollar_delta,
    delta_color=_carbon_dollar_color,
    help=_carbon_dollar_help,
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
    "Cost / Acre-Foot Runoff Prevented",
    _fmt_ce(ce['cost_per_acft']),
    delta=None,
    help=f"Confidence: Medium — see 'How this prototype works' for tier definitions. Implementation cost divided by runoff reduction vs baseline ({BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft). N/A if scenario increases runoff or has no cost."
)
_confidence_caption(ceff1, "medium")
ceff2.metric(
    "Cost / Citywide °F Cooling",
    _fmt_ce(ce['cost_per_degf']),
    delta=None,
    delta_color="off" if _cooling_f <= 0 else "normal",
    help="Confidence: Medium — see 'How this prototype works' for tier definitions. Implementation cost divided by degrees F of city-average cooling vs baseline (the °F is a citywide mean, not a per-person or per-site value). N/A if no cooling improvement. InVEST UCM canonical units are °C — to translate, this is approximately (Cost / °F) × 1.8 per °C."
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
    _carbon_value_table = results.get('carbon_value_usd', 0.0)
    _carbon_tons_table = results.get('carbon_tons_co2', 0.0)
    # Brief 30: SA = stock change (one-time); MN = annual sequestration.
    _carbon_metric_label = 'Carbon Storage Change' if _CARBON_IS_STOCK else 'Carbon Sequestration'
    _carbon_dollar_label_table = 'Carbon Storage Value' if _CARBON_IS_STOCK else 'Avoided Carbon Cost'
    _carbon_unit = 'tons CO2e' if _CARBON_IS_STOCK else 'tons CO2e/yr'
    _carbon_dollar_period = '' if _CARBON_IS_STOCK else '/yr'
    # Brief 33: per-city flood-damage rendering. Cities with a damage table
    # (MN) show monetized damage avoided; cities without (SA — matches
    # Vibrant Land methodology) show percent volume reduction.
    _flood_damage_monetized = (
        BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES and TOTAL_POTENTIAL_DAMAGE_USD > 0
    )
    if _flood_damage_monetized:
        _flood_label_table = 'Flood Damage Avoided'
        _flood_baseline_table = '$0'
        _flood_scenario_table = f'${_flood_damage_avoided / 1e6:.1f}M'
        _flood_change_table = (
            f'+${_flood_damage_avoided / 1e6:.1f}M'
            if _flood_damage_avoided >= 1e4 else '$0'
        )
    else:
        _flood_label_table = 'Flood Retention'
        _flood_baseline_table = '0%'
        _flood_scenario_table = f'{results["flood_reduction"]:.1f}%'
        _flood_change_table = f'+{results["flood_reduction"]:.1f}%'
    comparison_data = {
        'Metric': [
            'Flood Retention', 'Runoff Volume', 'Temperature Change',
            'Food Production', _carbon_metric_label, 'NDVI',
            _flood_label_table, 'Cooling Energy Savings', _carbon_dollar_label_table,
        ],
        'Baseline': [
            f'{_baseline_flood:.1f}',
            f'{BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft',
            'Reference',
            '0 lbs',
            f'0 {_carbon_unit}',
            f'{BASELINE_NDVI:.3f}',
            _flood_baseline_table,
            '$0/yr',
            f'$0{_carbon_dollar_period}',
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
            f'{_carbon_tons_table:,.0f} {_carbon_unit}',
            f'{results["mean_ndvi"]:.3f}',
            _flood_scenario_table,
            f'${_energy_savings_table / 1e6:.2f}M/yr',
            f'${_carbon_value_table / 1e6:.2f}M{_carbon_dollar_period}' if abs(_carbon_value_table) >= 1e4 else f'${_carbon_value_table:,.0f}{_carbon_dollar_period}',
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
            f'{_carbon_tons_table:+,.0f} {_carbon_unit}',
            f'{results["mean_ndvi"] - BASELINE_NDVI:+.3f}',
            _flood_change_table,
            f'+${_energy_savings_table / 1e6:.2f}M/yr' if _energy_savings_table >= 1e3 else '$0/yr',
            f'+${_carbon_value_table / 1e6:.2f}M{_carbon_dollar_period}' if _carbon_value_table >= 1e4 else f'+${_carbon_value_table:,.0f}{_carbon_dollar_period}' if _carbon_value_table >= 1 else f'$0{_carbon_dollar_period}',
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
    if selected_city.startswith("San Antonio"):
        st.info(
            "**SA Land Cover:** Using NatCap's compound NLCD×NLUD×tree-canopy "
            "LULC framework (1,984 compound lucodes; foundational adoption "
            "landed Brief 27). UCM, UNA, and Carbon all consume the "
            "compound-keyed biophysical tables directly (Briefs 28b, 29, 30). "
            "See `SA_INTEGRATION_PLAN.md` for the brief sequence."
        )
    # Brief B: Conversion fidelity panel — SA-only. Shows what fraction
    # of this scenario's converted pixels resolved via the documented
    # default-lucode fallback (because the source pixel's (NLUD, tree-
    # canopy) tuple had no matching row in NatCap's crosswalk for the
    # target NLCD). Surfaces a methodology question that would otherwise
    # be invisible. Hidden for MN (no compound conversion).
    if _COMPOUND_CONVERSION_ACTIVE:
        _ff_n = int(results.get('n_for', 0))
        _gi_n = int(results.get('n_wet', 0))
        _hd_n = int(results.get('n_hd', 0))
        _ff_fb = int(results.get('ff_fellback_pixels', 0))
        _gi_fb = int(results.get('gi_fellback_pixels', 0))
        _hd_fb = int(results.get('hd_fellback_pixels', 0))
        _conv_lines = ["**Conversion fidelity (SA)**", ""]
        for label, n_total, n_fb in [
            ("Green infrastructure", _gi_n, _gi_fb),
            ("Food forest",          _ff_n, _ff_fb),
            ("High density",         _hd_n, _hd_fb),
        ]:
            if n_total == 0:
                _conv_lines.append(f"- **{label}:** no conversions in this scenario.")
            else:
                _pct = 100.0 * n_fb / n_total
                _conv_lines.append(
                    f"- **{label}:** {n_fb:,} of {n_total:,} converted pixels "
                    f"({_pct:.1f} %) used the default target lucode because the "
                    f"source pixel's (NLUD, tree-canopy) context had no matching "
                    f"row in NatCap's crosswalk."
                )
        _conv_lines.append("")
        _conv_lines.append(
            "Default target lucodes: FF = 1310 (Deciduous Forest × Timber × "
            "medium canopy), GI = 122 (Woody Wetlands × Wetland × medium canopy), "
            "HD = 341 (Developed High Intensity × Residential × low canopy). "
            "See REFERENCE.md \"Land-use alignment\" for the conversion mechanism."
        )
        st.markdown("\n".join(_conv_lines))
    _assumption_tabs = st.tabs([
        "Flood & Runoff", "Temperature", "Food", "Carbon",
        "Mental Health", "Costs",
    ])
    with _assumption_tabs[0]:
        st.markdown(
            "- **Method:** USDA SCS Curve Number method, computed at 30 m raster "
            "resolution from per-pixel CN values × soil hydrologic group lookup. "
            "Reported as `100 − mean_CN` so higher = better.\n"
            f"- **Design storm:** {DESIGN_STORM_INCHES:.2f}-inch / "
            f"{DESIGN_STORM_MM:.0f}-mm rainfall — the NatCap per-city canonical "
            f"value for {selected_city} (MN: 100 mm / 3.94\", SA: 157 mm / 6.18\"; "
            f"migrated to per-city in Brief 23). Larger storms scale runoff "
            f"non-linearly; results don't extrapolate to extreme events.\n"
            "- **Green Infrastructure** is modeled as woody wetlands (NLCD 90). "
            "The broader GI category (rain gardens, bioswales, permeable pavement, "
            "green roofs, urban tree canopy) is not modeled — each would have "
            "different curve numbers.\n"
            "- **Relationship to InVEST UFR's runoff retention index.** The "
            "app reports `100 − mean_CN`, monotone with but not identical to "
            "InVEST UFR's canonical `rnf_rt_idx = mean(1 − Q/P)`. See "
            "REFERENCE.md's Flood Risk Reduction section for the relationship."
        )
    with _assumption_tabs[1]:
        _temp_calibration = (
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per HMI unit. "
            f"Values come from the InVEST UCM args JSON for the Minneapolis AOI "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`, humid continental Köppen Dfa). "
            "Treat the °F output as ±2 °F at best.\n"
            if selected_city.startswith("Minneapolis") else
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per HMI unit. "
            f"No published InVEST args exist for hot semi-arid Köppen BSh; "
            f"values are an estimate from regional UHI literature "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`). "
            "Treat the °F output as ±2 °F at best — uncertainty is larger "
            "here than for MN.\n"
        )
        st.markdown(
            "- **Method:** InVEST Urban Cooling Model. Per-pixel "
            "Cooling Capacity `CC = 0.6·shade + 0.2·albedo + 0.2·ETI`. "
            "The canonical Heat Mitigation Index `HMI = max(CC_local, "
            "CC_park)`, where `CC_park` is the exponentially distance-"
            "weighted average of CC values from green areas ≥2 hectares "
            "within `d_cool = 450 m` (per InVEST UCM eq. 118: "
            "`e^(-d/d_cool)`).\n"
            "- **Reported value:** mean(HMI) across valid pixels — "
            "validated against `natcap.invest.urban_cooling_model."
            "execute()` at MAE = 0.0000.\n"
            + _temp_calibration +
            "- **Not captured:** wind, humidity, urban geometry, building "
            "materials, anthropogenic heat. The model sees land cover only."
        )
    with _assumption_tabs[2]:
        _food_yield_line = (
            f"- **Yield benchmark:** {FOOD_FOREST_LBS_ACRE:,} lbs/acre/year, "
            "from NatCap food-forest studies. Assumes a mature, well-managed "
            "system at peak productivity. Newly established food forests will "
            "produce significantly less in early years.\n"
            if selected_city.startswith("Minneapolis") else
            f"- **Yield benchmark:** {FOOD_FOREST_LBS_ACRE:,} lbs/acre/year, "
            "from the NatCap SA Urban Agriculture project (2023) — conservative "
            "placeholder for hot semi-arid climate, below the MN benchmark to "
            "reflect lower productivity. Replace with project-published "
            "weighted average when available.\n"
        )
        st.markdown(
            "- **Food Forest** is modeled as deciduous forest (NLCD 41) — the "
            "closest available NLCD class. No NLCD class exists specifically for "
            "agroforestry or food forests.\n"
            + _food_yield_line +
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
            "- **Not in the surrogate.** UMH outputs are computed "
            "deterministically inside `evaluate_scenario` from the scenario's "
            "NDVI exposure — the surrogate doesn't need to predict them. They "
            "appear in the precomputed grid columns alongside the RF targets, "
            "but are recomputed live for any scenario the optimizer surfaces."
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
            "top of existing buildings or road infrastructure — building "
            "footprints and OSM road networks are both rasterized, unioned, "
            "and subtracted from the candidate pool. "
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
        ax.set_ylabel('Heat Mitigation Index\n(higher = more cooling)', fontsize=12)
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
        _max_carbon = max(scenario_df['carbon_tons_co2'].max() * 1.1, 1.0)
        # Brief 30: SA = one-time stock change (Vibrant Land framework);
        # MN = annual sequestration rate. Title + Y-label branch on framing.
        _carbon_title = 'Carbon Storage Change' if _CARBON_IS_STOCK else 'Carbon Sequestration'
        _carbon_ylabel = (
            'Carbon stock change (tons CO2e)\n(higher = more carbon stored)'
            if _CARBON_IS_STOCK
            else 'Carbon (tons CO2e/year)\n(higher = more sequestration)'
        )
        ax.bar(['Baseline', 'This Scenario'],
               [0, results['carbon_tons_co2']],
               color=['#5b8db8', '#7b4fa6'])
        ax.set_title(_carbon_title, fontsize=16, fontweight='bold')
        ax.set_ylabel(_carbon_ylabel, fontsize=12)
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

    # Brief A.3: filter saved scenarios to the active city. The .get("city",
    # selected_city) default is backward-compatible — in-memory saves from
    # before A.3 lacked the `city` key; treat them as belonging to the
    # current city rather than orphaning them.
    _saved_for_city = [
        s for s in st.session_state.saved_scenarios
        if s.get("city", selected_city) == selected_city
    ]

    st.subheader("Tradeoff Space")
    st.caption("Each point is a scenario. Better outcomes are toward the top-right — more cooling and greater flood-risk reduction. Bubble size shows food production for saved and optimized scenarios.")
    st.plotly_chart(plot_tradeoff(
        results, scenario_df,
        lookup_table=lookup_table,
        saved=_saved_for_city,
        optimized=st.session_state.optimized_results
    ), width='stretch')

    if TRACTS_DATA_AVAILABLE:
        st.divider()
        st.markdown("#### Neighborhood breakdown")
        # Brief 31: SA uses ACS block-group polygons (NatCap-canonical, matches
        # Vibrant Land Figure 10 framing); MN uses Census tracts. The
        # aggregation code is polygon-name-agnostic — only the user-facing
        # caption changes per city.
        _polygon_unit_plural = (
            "Census block groups" if selected_city.startswith("San Antonio")
            else "Census tracts"
        )
        _polygon_unit_singular = (
            "block group" if selected_city.startswith("San Antonio") else "tract"
        )
        st.caption(
            f"Top 5 most-improved {_polygon_unit_plural} under this scenario, ranked by "
            f"temperature change (°F cooler). Population-weighted within each {_polygon_unit_singular}."
        )
        _tracts_summary = compute_per_tract_summary(results['scenario_lulc_ucm'])
        if not _tracts_summary.empty:
            _top5 = (
                _tracts_summary
                .sort_values("Temp Δ (°F cooler)", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            st.dataframe(_top5, width='stretch', hide_index=True)
        else:
            st.caption(f"No {_polygon_unit_singular}-level data could be computed for this scenario.")

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
        "Best for carbon":          lookup_df.loc[lookup_df['carbon_tons_co2'].idxmax()],
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
            # Brief A.3: tag the city this scenario was saved in. Display sites
            # filter by active city so MN saves don't show up in SA's view.
            saved["city"] = selected_city
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
        # Brief 30: SA optimizer reports stock-change; MN reports annual flow.
        _opt_carbon_unit = "tons CO2e" if _CARBON_IS_STOCK else "tons CO2e/yr"
        _opt_carbon_col_label = (
            "Carbon (tons CO2e stock)" if _CARBON_IS_STOCK
            else "Carbon (tons CO2e/yr)"
        )
        if isinstance(opt, dict) and not opt.get('found'):
            st.warning(
                f"No scenarios found meeting all targets simultaneously.  \n"
                f"Maximum achievable values across all candidates:  \n"
                f"- Flood reduction: up to **{opt['max_flood']}** (your target: {min_flood})  \n"
                f"- Cooling: up to **{opt['max_cool']:.4f} HMI** (your target: {min_cool:.4f})  \n"
                f"- Food: up to **{opt['max_food']:.3f}M lbs** (your target: {min_food:.3f})  \n"
                f"- Carbon: up to **{opt['max_carbon']:,.0f} {_opt_carbon_unit}** (your target: {min_carbon:,})  \n"
                f"Try lowering the target for whichever metric is furthest from its maximum."
            )
        else:
            st.caption(
                f"Top scenarios meeting flood ≥ {min_flood}, cooling ≥ {min_cool_f:+.1f}°F, "
                f"food ≥ {min_food:.3f}M lbs, carbon ≥ {min_carbon:,} {_opt_carbon_unit} "
                "— ranked by balanced score. "
                "Numbers are surrogate model predictions with 10th–90th percentile uncertainty bands."
            )

            # Display table with uncertainty columns
            display_cols = ['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                            'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs',
                            'carbon_tons_co2']
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
                'carbon_tons_co2':          _opt_carbon_col_label,
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

    if _saved_for_city:
        st.divider()
        st.caption(
            "The Pareto frontier shows the most efficient tradeoff scenarios — ones where you "
            "cannot improve flood reduction, cooling, or food production without making at least "
            "one of the others worse."
        )
        with st.expander(f"Saved Scenarios ({len(_saved_for_city)})", expanded=False):
            df_saved = pd.DataFrame(_saved_for_city)
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
        "Development-intensity heat proxy opacity",
        0.0, 0.5, 0.2, 0.05,
        help=(
            "Transparency of the development-intensity heat-proxy overlay on the "
            "map. Currently uses developed-land intensity as a proxy for "
            "heat-vulnerable areas — NLCD 23 (high-intensity) weighted 1.0, "
            "NLCD 22 (medium) 0.6, NLCD 21 (low) 0.3. This is a placeholder "
            "for a future CDC/ATSDR Heat Vulnerability Index by census tract. "
            "Set to 0 to hide."
        ),
    )

    render_matplotlib(plot_spatial_map(
        results['scenario_lulc'], cooling_lulc,
        heat_overlay=nlcd_intensity_weights, overlay_alpha=overlay_opacity,
    ))
    st.caption(
        "Gray = unchanged developed land. Colors show where conversions occur. "
        "White = outside city boundary. Orange wash = development-intensity heat proxy "
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