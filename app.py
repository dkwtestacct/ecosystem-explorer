import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import plotly.graph_objects as go
import os
import json
import math
from pathlib import Path
from typing import NamedTuple, Any, Optional

import rasterio
from rasterio.features import rasterize as _rasterize
from skimage.transform import resize
import geopandas as _gpd
from scipy.ndimage import distance_transform_edt as _distance_transform_edt
from scipy.ndimage import zoom as _zoom
from scipy.signal import fftconvolve as _fftconvolve

from config import CITIES, DEFAULT_CITY, DEFAULT_COST_GI, DEFAULT_COST_FF, DEFAULT_COST_HD
from surrogate import (
    train_surrogate as _train_surrogate_fn,
    predict_with_uncertainty,
    plot_feature_importance,
    optimize_scenario,
    optimize_scenario_region,
    compute_pareto,
)
import export_invest_bundle as eib   # Brief D1 — InVEST export bundle
import natcap_validation as nv       # Brief B2 (revised) — validation badges
import natcap_scenarios as ns        # Brief B1 + B2 — fixed-scenario loader/flood helper
from ownership import OWNERSHIP_MODES, ELIGIBLE_FILTER_PRIMARY_MODES
from region_local_metrics import _REGION_LOCAL_METRICS

PIXEL_AREA_ACRES     = 0.2224  # 30 m × 30 m = 900 m² ÷ 4046.86 m²/acre. Same in EPSG:26915 (UTM) and EPSG:5070 (Albers); UTM ground-area distortion at MN is ~0.05 %, well within rounding.
# FOOD_FOREST_LBS_ACRE is city-dependent — see "── City-derived constants ──" below.

DEVELOPED_CODES   = [21, 22, 23]
CODE_GREEN_INFRA  = 90
CODE_FOOD_FOREST  = 41
CODE_HIGH_DENSITY = 24
NODATA            = -128

# OWNERSHIP_MODES + ELIGIBLE_FILTER_PRIMARY_MODES — data-only ownership
# mode tables — live in `ownership.py` (Constants Refactor / Task #52).
# The mask builder, the composite resolver, the normalizer, and every
# downstream consumer (export bundle, comparison table, audit expander,
# CSV export) stay here in app.py.


def _resolve_eligible_filter_mode(primary: str, vacant_overlay: bool) -> "Optional[str]":
    """Map the (primary class, vacant overlay) UI state to an
    OWNERSHIP_MODES key. (Legacy single-class resolver, kept for callers
    not yet on the multi-class UI; Batch 4 v2's checkbox panel uses the
    composite resolver below.)"""
    if primary is None:
        return 'vacant' if vacant_overlay else None
    if primary == 'public':
        return 'vacant_public' if vacant_overlay else 'public'
    return f"{primary}_vacant" if vacant_overlay else primary


# ── Eligible-land-filter multi-class resolvers (Batch 4 v2) ──────────────────
# The sidebar exposes 5 finer-class checkboxes + a vacant overlay. Single-
# class selections collapse to existing OWNERSHIP_MODES keys for backward
# compat with saved scenarios; multi-class selections persist as a small
# dict `{'classes': [...], 'vacant': bool}` that `_normalize_ownership_filter`
# below de-shapes uniformly for display and the export bundle.

def _compose_eligible_filter_cfg(class_names, vacant_overlay):
    """Build an OWNERSHIP_MODES-compatible mode_cfg for an arbitrary
    union of finer classes (+ optional vacant). The composed cfg has
    band1_in (the enum values for the checked classes) and optionally
    band2_eq=1; `_build_ownership_mask` consumes it identically to a
    static mode dict."""
    enum_vals = []
    for cls in class_names:
        cfg = OWNERSHIP_MODES.get(cls) or {}
        if 'band1_eq' in cfg:
            enum_vals.append(int(cfg['band1_eq']))
    result = {}
    if enum_vals:
        result['band1_in'] = tuple(sorted(set(enum_vals)))
    if vacant_overlay:
        result['band2_eq'] = 1
    return result


def _build_composite_ownership_label(class_names, vacant_overlay):
    """Display label for a multi-class composite. Uses each class's
    OWNERSHIP_MODES label with the trailing " land" stripped, joined by
    " + ", with "(vacant only)" appended when the overlay is on."""
    if not class_names:
        return "Vacant land" if vacant_overlay else "All ownership"
    parts = []
    for cls in sorted(class_names):
        cfg = OWNERSHIP_MODES.get(cls) or {}
        label = cfg.get('label', cls)
        # Strip trailing " land" so "City-owned land" + "School ... land"
        # → "City-owned + School district (K-12 public) (vacant only)".
        if label.endswith(' land'):
            label = label[:-len(' land')]
        parts.append(label)
    joined = ' + '.join(parts)
    if vacant_overlay:
        joined += ' (vacant only)'
    return joined


def _build_composite_ownership_short(class_names, vacant_overlay):
    """Terse variant for the provenance-bar suffix. Uses each class's
    OWNERSHIP_MODES `short` with the trailing " land" stripped, joined
    by " + ", with " (vacant)" appended when the overlay is on. Mirrors
    `_build_composite_ownership_label`'s structure on the short field."""
    if not class_names:
        return "vacant land" if vacant_overlay else "all ownership"
    parts = []
    for cls in sorted(class_names):
        cfg = OWNERSHIP_MODES.get(cls) or {}
        short = cfg.get('short', cls)
        if short.endswith(' land'):
            short = short[:-len(' land')]
        parts.append(short)
    joined = ' + '.join(parts)
    if vacant_overlay:
        joined += ' (vacant)'
    return joined


def _resolve_eligible_filter_state(classes_checked, vacant_overlay):
    """Returns (storage_value, mode_cfg, label, allowed_band1_values):
      storage_value: what to stamp onto results['ownership_filter']
        — None, an OWNERSHIP_MODES key string, or a composite dict.
      mode_cfg: dict for `_build_ownership_mask`.
      label: display string.
      allowed_band1_values: list[int] for the export bundle's
        `allowed_classes` field.

    Single-class selections (with or without vacant overlay) collapse
    to existing string mode keys so saved scenarios from Batch 4 v1
    round-trip identically. Multi-class selections persist as a small
    dict `{'classes': [list], 'vacant': bool}`."""
    classes_checked = list(classes_checked or [])
    if not classes_checked and not vacant_overlay:
        return None, None, None, []
    if not classes_checked and vacant_overlay:
        cfg = OWNERSHIP_MODES['vacant']
        return 'vacant', cfg, cfg['label'], []
    if len(classes_checked) == 1:
        cls = classes_checked[0]
        mode_key = f"{cls}_vacant" if vacant_overlay else cls
        if mode_key in OWNERSHIP_MODES:
            cfg = OWNERSHIP_MODES[mode_key]
            return (mode_key, cfg, cfg['label'],
                    _ownership_allowed_band1_values(cfg))
    # Multi-class (or single-class without a pre-baked _vacant entry) —
    # synthesize a composite.
    cfg = _compose_eligible_filter_cfg(classes_checked, vacant_overlay)
    label = _build_composite_ownership_label(classes_checked, vacant_overlay)
    storage = {'classes': sorted(classes_checked), 'vacant': bool(vacant_overlay)}
    return storage, cfg, label, list(cfg.get('band1_in', ()))


def _normalize_ownership_filter(value):
    """Normalize results['ownership_filter'] to a canonical record:
      {classes: [...], vacant_only: bool, mode_key: str|None,
       label: str, short: str}
    Accepts the three shapes the storage path produces:
      - None (no filter)
      - str (a single OWNERSHIP_MODES key — Batch 4 v1 + earlier)
      - dict (composite from Batch 4 v2's checkbox UI)
    Returns None when the input doesn't resolve to a known mode."""
    if value is None:
        return None
    if isinstance(value, str):
        cfg = OWNERSHIP_MODES.get(value)
        if cfg is None:
            return None
        # Derive class list from the band1 selector. For a single-class
        # mode (band1_eq), look up the matching primary class key. For
        # a rollup (band1_in), collect every primary class whose
        # band1_eq matches one of the rollup's values.
        classes = []
        if 'band1_eq' in cfg:
            for k, v in OWNERSHIP_MODES.items():
                if (k in ELIGIBLE_FILTER_PRIMARY_MODES
                        and v.get('band1_eq') == cfg['band1_eq']):
                    classes.append(k)
                    break
        elif 'band1_in' in cfg:
            for ev in cfg['band1_in']:
                for k, v in OWNERSHIP_MODES.items():
                    if (k in ELIGIBLE_FILTER_PRIMARY_MODES
                            and v.get('band1_eq') == ev):
                        classes.append(k)
                        break
        return {
            'classes':     classes,
            'vacant_only': cfg.get('band2_eq') == 1,
            'mode_key':    value,
            'label':       cfg['label'],
            'short':       cfg.get('short', cfg['label']),
        }
    if isinstance(value, dict):
        classes = list(value.get('classes', []))
        vacant = bool(value.get('vacant', False))
        return {
            'classes':     classes,
            'vacant_only': vacant,
            'mode_key':    None,
            'label':       _build_composite_ownership_label(classes, vacant),
            'short':       _build_composite_ownership_short(classes, vacant),
        }
    return None


def _build_ownership_mask(band1, band2, mode_cfg) -> "np.ndarray":
    """Compose the boolean ownership mask for a given OWNERSHIP_MODES
    config dict. Combines (where present) band1_eq / band1_in for the
    class-enum filter and band2_eq for the vacant overlay. Absent keys
    are unconstrained on that axis (the mask starts True and ANDs each
    present criterion). Both bands must be the same shape."""
    import numpy as _np
    mask = _np.ones_like(band1, dtype=bool)
    if 'band1_eq' in mode_cfg:
        mask &= (band1 == mode_cfg['band1_eq'])
    if 'band1_in' in mode_cfg:
        mask &= _np.isin(band1, list(mode_cfg['band1_in']))
    if 'band2_eq' in mode_cfg:
        mask &= (band2 == mode_cfg['band2_eq'])
    return mask


def _ownership_allowed_band1_values(mode_cfg) -> "list[int]":
    """Return the list of band-1 (class enum) values that satisfy the
    mode's class-enum filter, or [] if the mode is class-unconstrained
    (e.g. `vacant`, which keys only on band 2). Used by the export
    bundle's rich `ownership_filter` block."""
    if 'band1_eq' in mode_cfg:
        return [int(mode_cfg['band1_eq'])]
    if 'band1_in' in mode_cfg:
        return [int(v) for v in mode_cfg['band1_in']]
    return []


def toggle_selection(current, clicked_id):
    """Pure function — toggle `clicked_id` in `current` selection list.

    Returns a NEW list: `clicked_id` removed if already present, appended if
    absent. No Streamlit calls inside — pure data transform so tests can
    exercise it in isolation (verify_baselines has a four-case unit suite +
    meta-test that reverting this to 'return [clicked_id]' replace-mode
    must fail).

    Used by the Interactive Region Map's click-to-toggle handler: every
    new map-click event from Plotly tells us WHICH district was clicked
    (event.selection.points[0].customdata); the producer detects new
    clicks by event-signature de-dup; this function applies the toggle to
    the existing selection state read from session_state. Same source of
    truth as the sidebar multiselect — both read/write the same
    `region_labels_<layer>` key.

    Limitation: Plotly's selection_mode='points' delivers identical event
    payloads when the user clicks the SAME district twice in a row (the
    selection state doesn't change), so toggling a district off requires
    an intervening click on a different district. Click A → click B →
    click A correctly leaves {B} selected; click A → click A in immediate
    succession is a no-op (no rerun fires). Documented in the help-text
    caption."""
    if clicked_id in current:
        return [x for x in current if x != clicked_id]
    return list(current) + [clicked_id]


# ── Default-scenario state + display unification (Relay A) ─────────────────
# Documented default — the app's own copy says "Default view illustrates a
# balanced 50/50 mix at 10% conversion," so on load and after a city switch
# the resolved scenario lands at (pct=10, GI=50, FF=50, HD=0).
SCENARIO_DEFAULT_PCT_CONVERTED      = 10
SCENARIO_DEFAULT_GI_PCT             = 50
SCENARIO_DEFAULT_FF_PCT             = 50


def _resolve_scenario(pct_converted, green_infrastructure_pct, food_forest_pct):
    """Single resolved-scenario dict that every display surface reads from.

    The HD share is canonical `100 - GI - FF`. The display helpers below take
    this dict as their only input so the banner title, the main-panel
    sentence, and the audit expander can't desync from each other or from
    the engine's input arguments.
    """
    pct = int(pct_converted)
    gi = int(green_infrastructure_pct)
    ff = int(food_forest_pct)
    return {
        'pct_converted':            pct,
        'green_infrastructure_pct': gi,
        'food_forest_pct':          ff,
        'pct_highdensity':          100 - gi - ff,
    }


def _explorer_scenario_label(resolved):
    """Banner title for an Explorer-source scenario.

    When `pct_converted == 0` no pixels convert, so the standard
    "{pct}% converted — GI {gi}% / FF {ff}%" form is misleading (it would
    advertise allocation knobs that don't fire). Use the "no conversion"
    label instead; downstream code uses this consistently with the engine's
    no-op behavior at pct=0.
    """
    pct = resolved['pct_converted']
    if pct == 0:
        return "Explorer scenario · no conversion"
    return (f"Explorer scenario · {pct}% converted — "
            f"GI {resolved['green_infrastructure_pct']}% / "
            f"FF {resolved['food_forest_pct']}%")


# Active-scenario line-1 prefix per provenance — keyed on the PROVENANCE_* value
# so it shares its source with `_scen_provenance` and the provenance header. pct=0
# overrides to a baseline line (a 0%-conversion scenario is indistinguishable from
# baseline), so the PROVENANCE_BASELINE entry is only a safety net; the fallback
# covers any unmapped provenance so a new value can't render an empty prefix.
_ACTIVE_SCENARIO_PREFIX = {
    eib.PROVENANCE_EXPLORER:         "Explorer scenario",
    eib.PROVENANCE_OPTIMIZER:        "Optimizer-applied",
    eib.PROVENANCE_REGION_OPTIMIZED: "Selected-area optimized",
    eib.PROVENANCE_BASELINE:         "Baseline",
}


def _active_scenario_line1(resolved, provenance):
    """Line 1 of the page-root Active-scenario block: provenance prefix + mix.

    pct=0 → "Baseline · no conversion" regardless of provenance (a no-conversion
    scenario reads as baseline). Otherwise "{prefix} · {pct}% converted — GI {gi}%
    / FF {ff}% / HD {hd}%", always showing all three mix components (they sum to
    100 on their face)."""
    pct = resolved['pct_converted']
    if pct == 0:
        return "Baseline · no conversion"
    prefix = _ACTIVE_SCENARIO_PREFIX.get(provenance, "Scenario")
    return (
        f"{prefix} · {pct}% converted — "
        f"GI {resolved['green_infrastructure_pct']}% / "
        f"FF {resolved['food_forest_pct']}% / "
        f"HD {resolved['pct_highdensity']}%"
    )


def _explorer_audit_sentence(resolved, area_phrase, ownership_clause,
                             strategy_label):
    """Audit-expander sentence — same 0-conversion branch as the main panel
    sentence, with the audit's additional area + ownership context."""
    pct = resolved['pct_converted']
    if pct == 0:
        return (
            f"This scenario makes no conversions in **{area_phrase}**"
            f"{ownership_clause}; the metrics reflect the baseline."
        )
    return (
        f"This scenario converts **{pct}%** of the eligible "
        f"convertible pool in **{area_phrase}**{ownership_clause}, "
        f"allocating **{resolved['green_infrastructure_pct']}%** "
        f"to green infrastructure, "
        f"**{resolved['food_forest_pct']}%** to food forest, "
        f"and **{resolved['pct_highdensity']}%** to high-density "
        f"development, using **{strategy_label}**."
    )


# _REGION_LOCAL_METRICS — per-metric treatment table (data-only) — lives
# in `region_local_metrics.py` (Constants Refactor / Task #52). The
# reconciliation invariant (region_local over the entire AOI ==
# citywide) is asserted in verify_baselines.py.

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
# Organized into capability sections — placement (where conversions can land)
# vs. validation/handoff (how to read and ship results). Data/model updates
# (curve numbers, land-cover sources, carbon framing) are implementation
# history; they live in the changelog / "On the radar", not here.
WHATS_NEW_SECTIONS = [
    ("Spatial scenario placement", [
        "Apply land-use changes inside selected council districts or census tracts, then compare selected-area and citywide impacts.",
        "See how much land remains eligible after roads, buildings, existing natural land, and ownership filters are excluded.",
    ]),
    ("Ownership-aware scenarios", [
        "In San Antonio, restrict conversions to public, vacant, school, university, city, county, or state/federal land.",
        "These are planning-screen filters — they do not verify parcel availability or legal feasibility.",
        "School-related scenarios — in San Antonio, restrict conversions to school-related parcels and evaluate nature access for residents and children, and nature access at schools, plus cooling and Urban Mental Health outcomes.",
    ]),
    ("Scenario discovery", [
        "Search citywide with a fast machine-learning model that suggests promising mixes.",
        "For selected areas, the displayed impacts are computed by the InVEST-aligned evaluator, not predicted by the machine-learning model. The search returns the best tested mixes under your filters — best found, not guaranteed optima.",
    ]),
    ("Validation & handoff", [
        "Compare NatCap reference, current, and saved scenarios with source and validation labels.",
        "Export a runnable San Antonio bundle for canonical InVEST.",
    ]),
]

UNDERWAY_ENTRIES = []

ON_THE_RADAR = """\
- AlphaEarth-derived land-cover inputs, pixel-level spatial optimization, and nutrient retention (NDR) if canonical inputs become available.
- Street-tree scenarios — explore planting along sidewalks and rights-of-way at street scale.
"""

def _build_whats_new():
    sections = []
    if WHATS_NEW_SECTIONS:
        parts = ["### What's new"]
        for _title, _entries in WHATS_NEW_SECTIONS:
            if _entries:
                parts.append(f"\n**{_title}**\n")
                parts.extend(f"- {e}" for e in _entries)
        sections.append("\n".join(parts))
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
/* District/region multiselect chips: match the primary button blue instead of
   BaseWeb's default red (which read as an error state). Selector targets the
   BaseWeb tag inside stMultiSelect; baseweb class names shift by Streamlit
   version, so this is an ASYNC eyeball item — confirm the rendered chip is blue. */
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #5b8db8;
    border-color: #5b8db8;
    color: white;
}
</style>
''', unsafe_allow_html=True)

# ── First-load splash ─────────────────────────────────────────────────────────
# Module-level guard before any heavy city loaders run. Loading a city's
# rasters/tables is ~18 s for Minneapolis but ~220 s cold for San Antonio
# (1713×1984 grid + ownership + NatCap reference scenarios; measured
# 2026-06-02 — see PART B measurement note). @st.cache_resource(max_entries=1)
# on _load_city_runtime_state caps memory at one city at a time (Streamlit
# Cloud's 1 GB ceiling rules out keeping both warm), so the cold cost can't
# be hidden behind caching alone — first visit *will* pay it. The splash
# lets the user pick which city to pay for, and frames the speed difference
# honestly. Once `entry_city` is in session_state the splash never re-renders
# this session; the sidebar selectbox owns the city after that. The
# verify_baselines harness pre-seeds entry_city via its stub so the gate
# bypasses the splash flow entirely.
if 'entry_city' not in st.session_state:
    st.title("🌿 Ecosystem Explorer")
    st.markdown("Urban land-use tradeoff prototype, canonical-InVEST-aligned.")
    st.markdown(
        "Choose a starting city. San Antonio is the flagship demo and may "
        "take up to about a minute to load the first time; Minneapolis is "
        "a faster lightweight demo. You can switch cities later from the "
        "sidebar."
    )
    st.markdown("&nbsp;")  # spacer
    _splash_col_a, _splash_col_b = st.columns(2)
    with _splash_col_a:
        if st.button("Explore San Antonio — flagship demo",
                     type="primary", width="stretch",
                     key="splash_pick_sa"):
            st.session_state['entry_city'] = "San Antonio, TX"
            st.rerun()
        st.caption(
            # Non-breaking spaces within phrases + plain spaces around the
            # `·` separators → wraps land at separators only, never mid-phrase.
            # The detail line previously broke awkwardly like
            # 'NatCap reference / scenarios' or clipped 'InVEST exp…'
            # in narrow viewports.
            "San Antonio regional extent · "
            "ownership and school-land filters · "
            "council districts · "
            "NatCap reference scenarios · "
            "InVEST export"
        )
    with _splash_col_b:
        if st.button("Explore Minneapolis — lightweight demo",
                     width="stretch", key="splash_pick_mn"):
            st.session_state['entry_city'] = "Minneapolis, MN"
            st.rerun()
        st.caption(
            "Downtown Minneapolis extent · "
            "census-tract targeting · "
            "children's & school nature-access detail"
        )
    st.markdown("&nbsp;")  # spacer
    # What you can do — a plain-language capability teaser for first-time
    # visitors. Each line leads with its emphasized phrase and describes a
    # capability in the visitor's words; the in-app badge vocab (NatCap
    # published value / InVEST-validated / etc.) stays one layer in, not on
    # this landing. Mirrors the synced CAPABILITIES.md so the two don't drift.
    # Bordered container visually separates the teaser from the primary
    # city-picker decision above.
    with st.container(border=True):
        st.markdown(
            "**What you can do**  \n"
            "• **Build a land-use scenario** — convert developed land into green "
            "infrastructure, food forest, or higher-density development.  \n"
            "• **Target where it happens** — citywide, in selected council "
            "districts, or on eligible public, vacant, school, or other land.  \n"
            # "five" is sourced from model_validation.VALIDATED_MODELS (UCM, UNA,
            # UMH, UFR, Carbon). If that set changes, update this copy — the
            # verify_baselines validated-set check pins the count at 5.
            "• **Evaluate impacts** — flood, cooling, carbon, greenness, nature "
            "access, mental health, food, and cost, built around five InVEST "
            "model pathways. Five InVEST model pathways provide the core "
            "ecological and social outcomes; prototype food and cost modules add "
            "planning-screening context.  \n"
            "• **Optimize & compare** — find promising citywide mixes or "
            "best-tested mixes for a selected area, then compare tradeoffs.  \n"
            "• **Trust the results** — each result shows where it came from "
            "and how it was checked, with audit detail underneath."
        )
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []
if "optimized_results" not in st.session_state:
    st.session_state.optimized_results = None
# Region-constrained optimizer (variant B). Distinct slot from the citywide
# `optimized_results` so the tradeoff tab can pick the right render branch:
# citywide shows surrogate-predicted values + calibrated estimate ranges; region-active
# shows engine-true region_local values (no bands — values are real, not
# quantiles). See docs/internal/REGION_OPTIMIZER_SPEC.md.
if "region_optimized_results" not in st.session_state:
    st.session_state.region_optimized_results = None
if "active_example_scenario" not in st.session_state:
    st.session_state.active_example_scenario = 'balanced'
# Brief #4 — track whether the active scenario was applied from an optimizer
# suggestion (vs manually constructed via Explorer sliders or hit baseline).
# Survives the rerun the Apply button triggers; cleared when the user moves
# sliders away from the applied values (manual edit, preset click, etc.) so
# the OPTIMIZER provenance never stays stale.
if "applied_from_optimizer" not in st.session_state:
    st.session_state.applied_from_optimizer = False
if "_applied_optimizer_values" not in st.session_state:
    st.session_state._applied_optimizer_values = None
# Region-constrained optimizer (variant B) — distinct from the citywide
# `applied_from_optimizer` flag because the displayed metrics are engine-true
# region-local, not surrogate predictions. The provenance / Save / Export
# branches key off this independently so a region-optimized scenario can't
# silently surface as "machine-learning suggestion." See
# docs/internal/REGION_OPTIMIZER_SPEC.md §4.
if "applied_from_region_optimizer" not in st.session_state:
    st.session_state.applied_from_region_optimizer = False
if "_applied_region_optimizer_values" not in st.session_state:
    st.session_state._applied_region_optimizer_values = None
# Apply any pending slider values before sliders are rendered
if "_pending_pct" in st.session_state:
    st.session_state.slider_pct_converted = st.session_state.pop("_pending_pct")
    st.session_state.slider_gi_pct        = st.session_state.pop("_pending_gi")
    st.session_state.slider_ff_pct        = st.session_state.pop("_pending_ff")
    # active_example_scenario is set by the button handler before _pending_ keys are written
# Relay 50 — a guided example may also stage a placement strategy; land it on the
# radio's key before that radio renders (mirrors the mix _pending_* transfer).
if "_pending_placement" in st.session_state:
    st.session_state.placement_strategy_radio = st.session_state.pop("_pending_placement")
# Brief #4 — auto-clear the Applied-from-Optimizer flag whenever the current
# slider state diverges from the values that were applied (manual user edit,
# preset button, Best-by-Goal Apply, etc.). The optimizer Apply path sets
# both the slider values AND _applied_optimizer_values, so a match means we
# are still on the just-applied optimizer scenario.
if st.session_state.get("applied_from_optimizer"):
    _cur_slider_vals = (
        st.session_state.get("slider_pct_converted"),
        st.session_state.get("slider_gi_pct"),
        st.session_state.get("slider_ff_pct"),
    )
    _applied_vals = st.session_state.get("_applied_optimizer_values")
    if _applied_vals is None or _cur_slider_vals != _applied_vals:
        st.session_state.applied_from_optimizer = False
        st.session_state._applied_optimizer_values = None
# Same auto-clear for the region-optimizer flag — mirrors the citywide
# logic so manual edits / preset clicks reset region-optimized provenance
# back to Explorer.
if st.session_state.get("applied_from_region_optimizer"):
    _cur_slider_vals_r = (
        st.session_state.get("slider_pct_converted"),
        st.session_state.get("slider_gi_pct"),
        st.session_state.get("slider_ff_pct"),
    )
    _applied_vals_r = st.session_state.get("_applied_region_optimizer_values")
    if _applied_vals_r is None or _cur_slider_vals_r != _applied_vals_r:
        st.session_state.applied_from_region_optimizer = False
        st.session_state._applied_region_optimizer_values = None

# ── City selection ─────────────────────────────────────────────────────────────
# Only available cities surface in the dropdown. Unavailable entries (e.g.
# Minneapolis Full) stay in the CITIES dict so scripts/tests can still
# reference them by key, but they are hidden from the UI. Initial selection
# precedence: session_state['entry_city'] (set by the splash on first load)
# → config.DEFAULT_CITY (San Antonio, TX, the flagship) → first available
# city as a final fallback. After first render the selectbox's own widget
# state owns the city — switching via the sidebar works the same way
# whichever entry path was taken.
_city_names = [name for name, cfg in CITIES.items() if cfg['available']]
_entry_city = st.session_state.get('entry_city')
if _entry_city in _city_names:
    _default_city_index = _city_names.index(_entry_city)
elif DEFAULT_CITY in _city_names:
    _default_city_index = _city_names.index(DEFAULT_CITY)
else:
    _default_city_index = 0
selected_city = st.sidebar.selectbox("City", _city_names,
                                     index=_default_city_index)
city_cfg = CITIES[selected_city]

# Reset scenario sliders when the city changes so a new city renders against
# its own defaults instead of inheriting the previous city's slider state.
# Runs BEFORE the sidebar widgets are instantiated. Preset buttons set
# `_pending_*` and trigger `st.rerun()`; on that rerun the city has not
# changed so this branch is skipped and the preset wins.
def _reset_state_for_city_switch(session_state) -> None:
    """Clear every session_state key that doesn't survive a city change.

    Sliders + optimizer flags: a new city renders against its own defaults
    rather than inheriting the previous city's slider state, and an MN
    optimizer result doesn't visibly persist into the SA dashboard.

    Region + ownership widget keys (Subset Invariants Pass): regions are
    city-specific (SA council_districts ≠ MN downtown_tracts — rasters,
    label spaces, and display names all differ) and ownership data is
    SA-only, so a stale widget-key value carried across a city switch
    would render an empty multiselect at best and a labels-mismatch at
    worst. Clearing the widget state here, BEFORE the sidebar renders,
    means the new city starts on "entire-area / no filter" defaults. The
    mask-rebuild path in the sidebar already null-defaults the derived
    masks every rerun; this helper clears the WIDGET state that drives
    the rebuild.

    Called from the top of the script on city change AND from
    verify_baselines.py's guard transition test so both paths exercise
    the same contract. Caller is responsible for updating
    `_prev_city_key` afterwards.
    """
    for _k in ('slider_pct_converted', 'slider_gi_pct', 'slider_ff_pct'):
        session_state.pop(_k, None)
    session_state.active_example_scenario = 'balanced'
    session_state.optimized_results = None
    session_state.region_optimized_results = None
    session_state.applied_from_optimizer = False
    session_state._applied_optimizer_values = None
    session_state.applied_from_region_optimizer = False
    session_state._applied_region_optimizer_values = None
    session_state.region_apply_within = "Entire analysis area"
    session_state.pop('region_layer', None)
    session_state.pop('region_map_picker_event', None)
    session_state.pop('region_map_picker_layer', None)
    for _stale_key in [k for k in list(session_state.keys())
                       if isinstance(k, str) and k.startswith('region_labels_')]:
        session_state.pop(_stale_key, None)
    # Eligible Land Filter (Batch 4 v2) — the panel switched to per-class
    # checkboxes plus a vacant-overlay toggle. Reset all of them. The
    # earlier-batch `ownership_filter_choice` selectbox + the legacy
    # `ownership_filter_vacant_overlay` are also reset (they may still be
    # in session_state from a prior session before the UI change).
    for _elf_key in ('elf_check_city', 'elf_check_county',
                     'elf_check_state_federal', 'elf_check_school',
                     'elf_check_university', 'elf_check_vacant'):
        session_state.pop(_elf_key, None)
    session_state.ownership_filter_choice = None
    session_state.ownership_filter_vacant_overlay = False


if st.session_state.get('_prev_city_key') != selected_city:
    _reset_state_for_city_switch(st.session_state)
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
    "San Antonio, TX":      "San Antonio regional extent — ~3,060 km², ~1.9M residents.",
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
st.title("🌿 Ecosystem Explorer")
# Descriptor — matches the splash + CAPABILITIES.md so the product
# carries one consistent self-description across surfaces.
st.caption("Urban land-use tradeoff prototype, canonical-InVEST-aligned.")

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
    # Children's nature access (RELAY) — under-18 population per pixel,
    # uniform-block-spread from Census 2020 PL 94-171 P1 - P3. Used ONLY
    # to weight the access share in calculate_nature_access (not for the
    # 2SFCA supply/demand on total pop, and not for UMH). None for cities
    # without a child_pop_file configured — calculate_nature_access then
    # falls back to a None children's-access return, surfaced as "—" in
    # the UI.
    child_pop_count_raster: Optional[np.ndarray]
    child_population_data_available: bool
    # Nature Access at Schools — K-12 school points (public + charter +
    # private; NCES CCD/PSS/EDGE) clipped to the modelable extent and
    # projected to pixel coordinates. Powers the destination-based access
    # metric (sample the 2SFCA adequate mask at each school). None for
    # cities without schools_file configured.
    schools_pixels: Optional[np.ndarray]   # (N, 2) int64 (row, col)
    schools_sectors: Optional[np.ndarray]  # (N,) object: 'public'/'charter'/'private'
    schools_metadata: Optional[dict]       # source, vintage, sector counts, etc.
    schools_data_available: bool
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
    # Region Selection Phase 1 — per-city region-layer artifacts. Each layer
    # key (e.g. 'council_districts', 'bexar_tracts') maps to:
    #   region_rasters[key]:        int32 raster with positional indices
    #                               (0..N-1) and -1 fill, on the active grid
    #   region_layer_labels[key]:   list[str] of labels (label_field values)
    #                               indexed by raster value
    #   region_layer_display_names[key]: singular noun for UI captions
    #
    # **Contract — positional vs label.**
    #   - Positional indices (0..N-1) are INTERNAL: only the raster carries
    #     them, only the mask-build site references them.
    #   - Everything USER-FACING and METADATA-FACING uses the real label_field
    #     values via region_layer_labels — the UI selects "District 5", the
    #     `region_selection['selected_ids']` block in `results` carries
    #     ["5"] (label values), the export bundle's metadata.json carries
    #     ["5"]. Positional indices never leak past the mask-construction
    #     chokepoint.
    #
    # See `REGION_SELECTION_PHASE1_SPEC.md`. The selected_region_mask
    # consumed by evaluate_scenario is built at the caller by translating
    # label-strings -> positional indices via
    # `[labels.index(lbl) for lbl in selected_labels]`, then
    # `np.isin(region_rasters[key], positional_indices)`.
    region_rasters: dict
    region_layer_labels: dict
    region_layer_display_names: dict
    # Ownership Integration — Finer Ownership Classes Pass uses a two-band
    # raster. `ownership_raster` (band 1) carries the class enum 0-5
    # (private / city / county / state-federal / school-university /
    # unknown), nodata=-1. `ownership_vacant_raster` (band 2) carries the
    # is_vacant flag 0/1, nodata=-1. Both are None for cities without an
    # `ownership_layer` config (MN). The caller derives boolean masks via
    # `_build_ownership_mask(ownership_raster, ownership_vacant_raster,
    # OWNERSHIP_MODES[mode])`.
    ownership_raster: Optional[np.ndarray]
    ownership_vacant_raster: Optional[np.ndarray]
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

# Un-bury the optimizer — the description paragraph + the three bullets
# move INTO "How this prototype works" below; they're "how it works"
# context, not the "what can I do" surface. The flood-changes-may-be-small
# expectation-setting caveat rides with them (it's in the GI bullet).
# Resulting top: title → [What's new / How this prototype works /
# Validation status, all collapsed] → scenario sentence → Discover CTA →
# metrics.
with st.expander("How this prototype works", expanded=False):
    st.markdown(
        "Explore how converting developed land into green infrastructure or food forests "
        "affects **flood damage risk**, **urban cooling costs**, **food production**, "
        "**nature access**, **carbon sequestration**, and **Urban Mental Health outcomes** (per-pixel parity with InVEST UMH; NDVI input is a land-cover-derived proxy) across the city — translating "
        "ecological changes into concrete impacts for planners and decision-makers."
    )
    st.markdown(
        "**Five InVEST model pathways.** Urban Cooling, Urban Flood Risk "
        "Mitigation, Urban Nature Access, Urban Mental Health, and Carbon "
        "Storage/Sequestration — evaluated by a prototype evaluator validated "
        "against canonical InVEST 3.19.0 where comparable (the **where comparable** "
        "hedge honestly excludes the cases that aren't, notably MN carbon, which "
        "runs a proxy rather than the four-pool stock model). The app does not run "
        "canonical InVEST live; export is available for formal InVEST handoff. "
        "These five pathways provide the core ecological and social outcomes; "
        "prototype food and cost modules add planning-screening context."
    )
    st.markdown(
        '- **Green Infrastructure (wetlands)** — strongest per-pixel improvement in the runoff / curve-number indicators; citywide changes may be small  \n'
        '- **Food Forest** — best for cooling + food  \n'
        '- **High Density** — worst for ecological and nature-access outcomes  \n'
    )
    st.markdown(
        "**Green Infrastructure** converts developed land to woody wetlands "
        "(NLCD code 90) — strongest for Flood Index and runoff-volume indicators.  \n"
        "**Food Forest** is modeled as deciduous forest (NLCD code 41) with a "
        "food-yield benchmark — best for cooling and food.  \n"
        "**High Density** is modeled as developed, high-intensity / impervious "
        "land (NLCD code 24) — worst for ecological and nature-access outcomes.  \n"
        "  \n"
        "This is an exploratory tool — numbers are directional, not precise. "
        "Use them to compare strategies, not as final answers.  \n"
        "  \n"
        "Flood Index is derived from curve number, cooling from a heat "
        "mitigation index, and food production from a food-forest yield "
        "benchmark — use these as comparative indicators.  \n"
        f"Cooling °F is approximate (±2°F). Runoff uses a city-specific design "
        f"storm ({DESIGN_STORM_MM:.0f} mm / {DESIGN_STORM_INCHES:.2f} inches for "
        f"{selected_city}; NatCap per-city canonical). Cost is order-of-magnitude — "
        f"adjust \\$/acre sliders in sidebar."
    )
    st.markdown(
        "- **InVEST-aligned evaluator** — the prototype's numpy reimplementation "
        "of the InVEST urban models, verified against canonical InVEST where "
        "comparable. Not canonical InVEST running live.  \n"
    )
    st.markdown(
        "**Each scenario shows two validation surfaces.** A *Source / Validation* "
        "header above the metric cards describes the scenario as a whole; each "
        "individual metric card has a small badge under its value.  \n"
        "  \n"
        "**Scenario provenance header** — one of four sources:  \n"
        "  \n"
        "- **Baseline** — the unmodified city LULC; prototype evaluator, verified against canonical InVEST where comparable.  \n"
        "- **NatCap published reference** — the value is displayed directly from "
        "NatCap's published outputs (fixed-scenario reference view).  \n"
        "- **Explorer-generated** — a slider-built scenario; engine-validated; "
        "scenario not NatCap-published.  \n"
        "- **machine-learning suggestion** — an applied citywide search suggestion; "
        "engine-validated; exploratory candidate.  \n"
        "  \n"
        "**Per-card badge** — one of four states:  \n"
        "  \n"
        "- **`◆ NatCap published value`** (green) — fires only in the fixed-scenario "
        "reference view for metrics with NatCap-published values. *Displays "
        "NatCap's number*; not a reproduction claim.  \n"
        "- **`■ InVEST-validated`** (teal) — the card's InVEST model has measured "
        "per-pixel parity against canonical natcap.invest 3.19.0 via a committed "
        "comparison harness: Temperature/cooling (UCM), Nature Access (UNA), "
        "Preventable MH (UMH), Runoff Retention (UFR), and SA four-pool Carbon.  \n"
        "- **`○ InVEST-aligned`** (blue) — canonical InVEST methodology, but "
        "per-pixel parity isn't measured for this output (Flood Index, Runoff "
        "Volume, and the dollar / derived cards).  \n"
        "- **`△ Prototype`** (gray) — no canonical InVEST analog (synthetic NDVI "
        "proxy, food yield benchmark, implementation-cost sliders, "
        "cost-effectiveness ratios, MN carbon proxy).  \n"
        "  \n"
        "The leading glyph (◆ ■ ○ △) is a shape-distinct, hue-independent marker so "
        "the tiers stay distinguishable when color alone doesn't.  \n"
        "  \n"
        "**Context-switch rule.** A `NatCap published value`-class metric shows "
        "the green badge *only* in the fixed-scenario reference view; in "
        "Baseline / Explorer / Optimizer scenarios it shows its everyday tier "
        "(InVEST-validated for the parity-measured models, else InVEST-aligned).  \n"
        "  \n"
        "The badges describe the *method's* trustworthiness, not whether the "
        "number is large or small. A `Prototype`-badged card showing a precise "
        "number is still a prototype number.  \n"
        "  \n"
        "**Conditional outputs.** *Flood Damage Avoided* "
        "requires a city-specific damage-valuation table, so it is shown only "
        "for cities that have one; San Antonio reports Flood Index, Runoff "
        "Retention, and Runoff Volume instead."
    )

with st.expander("Validation status", expanded=False):
    st.markdown(
        "**How outputs are validated.** Ecosystem Explorer verifies its model "
        "engine against canonical `natcap.invest` execution wherever the inputs "
        "to do so exist — InVEST's UCM, UNA, and UMH match per-pixel. NatCap's "
        "published project values are shown as reference points. Some of those "
        "project values are displayed as published references rather than "
        "reproduced from source, because the exact scenario land-cover rasters, "
        "aggregation scripts, or model arguments behind NatCap's published "
        "figures weren't available to us. Scenarios you build, or that the "
        "optimizer suggests, are computed by the validated engine but are not "
        "NatCap-published."
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
#       keyed. See docs/archive/SA_INTEGRATION_PLAN_2026-05.md Decision 2 and docs/internal/DESIGN_NOTES.md.
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
    # NatCap conversation; see docs/internal/NATCAP_COLLABORATION.md).
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
    were introduced. See docs/internal/ARCHITECTURE.md "CRS handling" for the
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
            f"(not runtime) — see docs/internal/ARCHITECTURE.md 'CRS handling'."
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
# in docs/internal/DESIGN_NOTES.md.
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


def _sample_schools_access(adequate, schools_pixels, schools_sectors):
    """Sample the 2SFCA `adequate` per-pixel mask at each school point.

    Returns a dict with the total + per-sector counts and the headline %.
    Used by both the citywide path and the region-local clip — the caller
    passes the appropriate `adequate` mask (citywide vs region-clipped)
    and the same schools_pixels each time. No new threshold introduced:
    `adequate` is the SAME mask the residential Nature Access metric uses
    (supply_percapita >= UNA_DEMAND_M2_PER_CAPITA, restricted to valid LULC).
    """
    if schools_pixels is None or len(schools_pixels) == 0:
        return {"pct": None, "n_with_access": 0, "n_total": 0,
                "by_sector": {}}
    rs = schools_pixels[:, 0]
    cs = schools_pixels[:, 1]
    school_access = adequate[rs, cs]
    n_total = int(len(schools_pixels))
    n_with = int(school_access.sum())
    by_sector = {}
    for sec in ("public", "charter", "private"):
        sec_mask = (schools_sectors == sec)
        sec_n = int(sec_mask.sum())
        if sec_n == 0:
            by_sector[sec] = {"pct": None, "n_with_access": 0, "n_total": 0}
        else:
            sec_with = int(school_access[sec_mask].sum())
            by_sector[sec] = {
                "pct": round(100.0 * sec_with / sec_n, 1),
                "n_with_access": sec_with,
                "n_total": sec_n,
            }
    return {
        "pct": round(100.0 * n_with / n_total, 1) if n_total else None,
        "n_with_access": n_with,
        "n_total": n_total,
        "by_sector": by_sector,
    }


def _invest_una_pct_pop_supply_ge_demand(scenario_lulc, pop_count_raster,
                                         mask=None,
                                         child_pop_count_raster=None):
    """Headline UNA metric: the share of the modelable-extent population whose
    per-capita urban-nature supply meets `UNA_DEMAND_M2_PER_CAPITA`.

    Returns `(pct, modelable_pop, people_supplied)`. The modelable extent is the
    population on valid-LULC pixels; InVEST cannot model supply for residents on
    LULC nodata (a large share of the prototype's downtown MN AOI).

    `mask` (Region-Local Metrics Commit 1) intersects with the valid filter so
    the returned numbers are population-clipped to the masked pixels — the
    locked UNA region-local treatment is "clip to population inside region".
    `mask=None` reproduces the citywide behavior exactly.

    Children's nature access RELAY — when `child_pop_count_raster` is provided,
    ALSO returns child access metrics computed against the **same adequate
    mask** (which is built on the SAME 2SFCA supply, demand, and valid-LULC
    rules using TOTAL pop). The only difference is the weighting: child access
    asks "what fraction of under-18 residents live where total-pop per-capita
    supply meets demand", not "what's the supply when only kids are counted as
    demand." Per the brief: keep 2SFCA on total pop; the child weighting is
    in the *access share*, never in the supply calculation. Returns 6-tuple
    `(adult_pct, adult_modelable, adult_supplied, child_pct, child_modelable,
    child_supplied)`. UMH is NOT child-weighted (adult-calibrated incidence
    + effect sizes; child × adult is meaningless).
    """
    supply_percapita, valid = _una_supply_percapita(
        scenario_lulc, pop_count_raster)
    pop = np.asarray(pop_count_raster, dtype=np.float64)
    if mask is not None:
        valid = valid & mask
    modelable_pop = float(pop[valid].sum())
    if modelable_pop <= 0:
        if child_pop_count_raster is None:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    adequate = valid & (supply_percapita >= UNA_DEMAND_M2_PER_CAPITA)
    people_supplied = float(pop[adequate].sum())
    adult_pct = 100.0 * people_supplied / modelable_pop
    if child_pop_count_raster is None:
        return adult_pct, modelable_pop, people_supplied
    # Children's access — same adequate mask, weighted by under-18 population.
    child = np.asarray(child_pop_count_raster, dtype=np.float64)
    child_modelable = float(child[valid].sum())
    if child_modelable <= 0:
        return adult_pct, modelable_pop, people_supplied, 0.0, 0.0, 0.0
    child_supplied = float(child[adequate].sum())
    child_pct = 100.0 * child_supplied / child_modelable
    return (adult_pct, modelable_pop, people_supplied,
            child_pct, child_modelable, child_supplied)


def calculate_nature_access(scenario_lulc, pop_count_raster, mask=None,
                            child_pop_count_raster=None):
    """Canonical InVEST Urban Nature Access for the given scenario LULC.

    Re-implements `natcap.invest.urban_nature_access` (uniform search
    radius + configurable decay, with per-city parameters — see
    docs/internal/DESIGN_NOTES.md) in numpy via two-step floating catchment area
    (2SFCA). The headline metric is `pct_pop_supply_ge_demand`: the share of the
    modelable-extent population whose per-capita nature supply meets the demand
    standard.

    `pop_count_raster` must be per-pixel population **counts** (not density).

    `mask` (Region-Local Metrics Commit 1) — when provided, the returned
    pct + people_with_access are population-clipped to the masked pixels
    (denominator = population at mask ∩ valid; numerator = population at
    mask ∩ valid ∩ supply-adequate). Per the locked UNA treatment, this is
    a population-clip not a pixel-clip. `mask=None` reproduces citywide.

    Returns a 3-tuple `(access_pct, _legacy_slot, people_with_access)` when
    `child_pop_count_raster` is None (preserves all existing 4 call sites).
    When `child_pop_count_raster` is provided, returns a 5-tuple
    `(access_pct, _legacy_slot, people_with_access,
      children_access_pct, children_with_access)`.
    """
    if child_pop_count_raster is None:
        pct, _modelable_pop, people_supplied = _invest_una_pct_pop_supply_ge_demand(
            scenario_lulc, pop_count_raster, mask=mask
        )
        return round(float(pct), 1), 0.0, int(round(people_supplied))
    (pct, _modelable_pop, people_supplied,
     child_pct, _child_modelable, child_supplied) = (
        _invest_una_pct_pop_supply_ge_demand(
            scenario_lulc, pop_count_raster, mask=mask,
            child_pop_count_raster=child_pop_count_raster,
        )
    )
    return (round(float(pct), 1), 0.0, int(round(people_supplied)),
            round(float(child_pct), 1), int(round(child_supplied)))


def calculate_schools_nature_access(scenario_lulc, pop_count_raster,
                                    schools_pixels, schools_sectors,
                                    mask=None):
    """Destination-based UNA metric: % of K-12 school points sitting on
    pixels where the 2SFCA `adequate` mask is True. Children's daytime
    location proxy — the residential metric (Nature Access /
    Children's Nature Access) answers 'do residents live where supply
    meets demand'; this answers 'do schools sit where supply meets
    demand'. Same 2SFCA pipeline, same per-city UNA_DEMAND_M2_PER_CAPITA
    threshold, same valid-LULC restriction — no new threshold introduced.

    Sampling at the existing `adequate` mask makes the school metric
    consistent with the residential metric by construction. Different
    quantity (count of school points, not population-weighted), same
    aligned-method validation tier.

    `schools_pixels` is an (N, 2) int64 ndarray of (row, col) pixel
    coordinates; `schools_sectors` is an (N,) object array with values in
    {'public', 'charter', 'private'}. Both pre-computed at city load by
    `_load_city_runtime_state` Phase 2c. `mask=None` reproduces the
    citywide behavior; an Mxshape boolean mask restricts adequate to the
    intersection (region-local treatment).

    Returns a dict (NOT a tuple — different from the residential
    function's signature) so the sector breakdowns + count fields stay
    discoverable. `pct` is None when no schools are configured/in-extent."""
    if schools_pixels is None or len(schools_pixels) == 0:
        return {"pct": None, "n_with_access": 0, "n_total": 0,
                "by_sector": {}}
    supply_percapita, valid = _una_supply_percapita(scenario_lulc,
                                                     pop_count_raster)
    if mask is not None:
        valid = valid & mask
    adequate = valid & (supply_percapita >= UNA_DEMAND_M2_PER_CAPITA)
    return _sample_schools_access(adequate, schools_pixels, schools_sectors)


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

# Neighborhood NDVI exposure (NE): a uniform BUFFER-MEAN of the NDVI proxy over
# a flat disk of radius `search_radius / pixel_size`, matching canonical InVEST
# UMH 3.19.0's NE kernel exactly — a binary-disk convolution with edge
# correction, i.e. pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True).
# Validated to per-pixel parity against `natcap.invest.urban_mental_health.execute()`
# via compare_umh_invest.py: MN MAE(active) ≤ 1.1e-9 cases/px, r = 1.000000 on
# both outcomes; SA MAE(active) ≤ 2.3e-6 cases/px, r ≥ 0.998 (SA residual is
# canonical's radius padding + edge-crop alignment, not a kernel divergence).
# Locked by the harness's parity-assert pass criterion (MAE < 1e-5, r > 0.99,
# |Δtotal|/total < 0.5%) with a --meta-test mode that perturbs proto by +0.5%
# to prove the assert is sharp. See docs/internal/DESIGN_NOTES.md §6.3.
# (Brief A's Gaussian σ=radius kernel diverged per-pixel; Brief B switched it.)
_UMH_RADIUS_PX = UMH_SEARCH_RADIUS_M / PIXEL_SIZE_M     # = 10.0 at 30 m / 300 m
_UMH_APOTHEM   = int(np.floor(_UMH_RADIUS_PX))
_umh_yy, _umh_xx = np.mgrid[-_UMH_APOTHEM:_UMH_APOTHEM + 1, -_UMH_APOTHEM:_UMH_APOTHEM + 1]
_UMH_KERNEL = (np.hypot(_umh_yy, _umh_xx) <= _UMH_RADIUS_PX).astype(np.float64)
del _umh_yy, _umh_xx
_UMH_LN_RR_DEPRESSION = float(np.log(RR_0_1_NDVI_DEPRESSION))
_UMH_LN_RR_ANXIETY    = float(np.log(RR_0_1_NDVI_ANXIETY))


def _umh_neighborhood_exposure(ndvi_raster):
    """Neighborhood NDVI exposure (NE) = edge-corrected mean of the NDVI proxy
    over the UMH buffer disk — canonical InVEST UMH's `ndvi_*_buffer_mean`. The
    NDVI proxy is fully filled (no nodata), so valid_mask is all-True; the edge
    correction handles the AOI boundary exactly as canonical's radius padding
    does. `kernel_sum=1.0` makes `_convolve_edge_corrected` return numer/denom —
    the local mean rather than a weighted sum."""
    valid = np.ones(ndvi_raster.shape, dtype=bool)
    return _convolve_edge_corrected(
        ndvi_raster.astype(np.float64), _UMH_KERNEL, valid, 1.0)


def calculate_mental_health_impact(scenario_lulc, baseline_ne_raster, pop_count, ndvi_raster=None, mask=None):
    """Return (preventable_mh_cases, avoided_mh_cost_usd) for the scenario.

    `baseline_ne_raster` is the buffer-mean NE raster for the unmodified LULC
    (precomputed once at module load — see _BASELINE_NE_RASTER below). We
    compute the scenario-side NE on the fly, take ΔNE, apply the InVEST UMH
    formula per pixel, and sum population-weighted preventable cases. Returns
    (0.0, 0.0) if the population raster isn't loaded — there's nothing to
    weight by.

    Pass `ndvi_raster=` to reuse a precomputed scenario NDVI raster (saves
    one full-AOI allocation when `evaluate_scenario` already built one for
    `compute_mean_ndvi`).

    `mask` (Region-Local Metrics Commit 1) — when provided, the sum runs only
    over masked pixels (population-clip per the locked UMH treatment).
    `mask=None` reproduces the citywide sum."""
    if not POPULATION_DATA_AVAILABLE:
        return 0.0, 0.0
    if ndvi_raster is None:
        ndvi_raster = _lulc_to_ndvi_raster(scenario_lulc)
    ne_scenario = _umh_neighborhood_exposure(ndvi_raster)
    delta_ne = ne_scenario - baseline_ne_raster

    rr_dep = np.exp(_UMH_LN_RR_DEPRESSION * 10 * delta_ne)
    rr_anx = np.exp(_UMH_LN_RR_ANXIETY    * 10 * delta_ne)
    pf_dep = 1.0 - rr_dep
    pf_anx = 1.0 - rr_anx

    pc_dep = pf_dep * BIR_DEPRESSION * pop_count
    pc_anx = pf_anx * BIR_ANXIETY    * pop_count
    if mask is None:
        total_pc = float((pc_dep + pc_anx).sum())
        avoided_cost = float((
            pc_dep * COST_PER_DEPRESSION_CASE_USD
            + pc_anx * COST_PER_ANXIETY_CASE_USD
        ).sum())
    else:
        total_pc = float((pc_dep + pc_anx)[mask].sum())
        avoided_cost = float((
            pc_dep * COST_PER_DEPRESSION_CASE_USD
            + pc_anx * COST_PER_ANXIETY_CASE_USD
        )[mask].sum())
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


def cn_array_to_retention_index(cn_arr, valid_mask):
    """Mean per-pixel runoff-retention index — the canonical InVEST UFR
    reading `rnf_rt_idx = mean(1 − Q/P)`. Vectorizes the SAME SCS-CN chain
    `cn_to_runoff_acre_feet` applies to the lumped mean CN, but PER PIXEL,
    then averages the retained fraction over the valid mask.

    Because Q is convex in CN, this is NOT equal to `1 − Q(mean_CN)/P`
    (Jensen's inequality) — it is the faithful per-pixel retention average,
    distinct from the mean-CN-lumped form the Flood Index (`100 − mean_CN`)
    uses. Q ≤ P and Q ≥ 0 by construction, so the result is in [0, 1];
    returns 0.0 when the mask is empty. `valid_mask` must match the masking
    used for `mean_cn` (i.e. `cn_arr > 0`, optionally ∩ region mask).
    """
    cn = cn_arr[valid_mask].astype(np.float64)
    if cn.size == 0:
        return 0.0
    P = DESIGN_STORM_INCHES
    S = (1000.0 / cn) - 10.0
    Ia = 0.2 * S
    Q = np.where(P <= Ia, 0.0, (P - Ia) ** 2 / (P - Ia + S))
    rnf = 1.0 - Q / P
    return round(float(rnf.mean()), 4)


def hm_to_temp_change_f(mean_hm):
    """Translate an HM-index delta vs baseline into an approximate °F
    temperature change: ΔT = T_after − T_before.

    Sign convention: positive = WARMER, negative = cooler. A higher Heat
    Mitigation Index means more cooling (a lower air temperature), so the
    index delta is negated before scaling to °F. This matches the universal
    physical ΔT convention and avoids the "negative cooling" oxymoron — the
    display layer (`_fmt_temp_change`) translates the sign back into
    "X°F cooler" / "X°F warmer" so users never see a bare signed number.
    """
    # read from state to avoid silent-staleness if city switches
    delta_hm = mean_hm - _CURRENT_CITY_STATE.baseline_hm
    return round(-delta_hm * HM_TO_FAHRENHEIT, 1)


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
    """Return $/unit ratios vs baseline; None where the denominator is
    zero, negative, OR below the per-metric "too small to divide"
    epsilon. Small-denominator floors are screening thresholds — when a
    region-constrained scenario reduces runoff by only a fraction of an
    acre-foot or cools by a hundredth of a degree, dividing the
    implementation cost by that nearly-zero benefit produces a
    spuriously precise ratio (e.g. "$961k per ac-ft") that reads as a
    real number but isn't informative. The floors collapse those cases
    to N/A. Citywide scenarios typically clear them; region cells
    sometimes don't, which is the signal the floors are designed to
    catch (see UI feedback #2)."""
    cost = results['total_cost_mln'] * 1_000_000
    if cost <= 0:
        return {'cost_per_acft': None, 'cost_per_degf': None, 'cost_per_1k_people': None}

    # Epsilon floors: below these the ratio's precision is illusory.
    _RUNOFF_EPS_ACFT     = 10.0   # ~10 ac-ft of runoff reduction
    _COOLING_EPS_DEGF    = 0.05   # ~0.05 °F of citywide cooling
    _PEOPLE_EPS_HEADS    = 100    # 100 people fed (= 0.1 thousand)

    runoff_prevented = baseline_runoff_acft - results['runoff_acre_feet']
    cost_per_acft = (round(cost / runoff_prevented)
                     if runoff_prevented >= _RUNOFF_EPS_ACFT else None)

    # Cost per °F of cooling. Under the ΔT convention cooling is a NEGATIVE
    # temp_change_f (positive = warmer), so the ratio is only defined when the
    # scenario actually cools; divide cost by the cooling magnitude.
    temp_change_f = results['temp_change_f']
    cost_per_degf = (round(cost / -temp_change_f)
                     if (temp_change_f < 0
                         and abs(temp_change_f) >= _COOLING_EPS_DEGF) else None)

    people_fed = results['people_fed']
    cost_per_1k_people = (round(cost / (people_fed / 1000))
                          if people_fed >= _PEOPLE_EPS_HEADS else None)

    return {
        'cost_per_acft':       cost_per_acft,
        'cost_per_degf':       cost_per_degf,
        'cost_per_1k_people':  cost_per_1k_people,
    }


def _fmt_sig(x, sig=3):
    """Screening-precision number — the SINGLE source of value precision across
    cards, plot hover, and the suggestion tables. Shows `sig` significant figures
    with a floating k/M/B unit and no fixed decimal places, so precision tracks
    the magnitude rather than an arbitrary dp count:
        3,095,697 → '3.10M'   559,410,000 → '559M'   16.93 → '16.9'
        79,300,000 → '79.3M'  0.4670 → '0.467'        100,800 → '101k'
    Callers append the unit (or move it into the label). For °F/% the caller
    layers the unit; Temp keeps a bespoke 1-dp form (sub-unit, see card)."""
    if x is None or not math.isfinite(x) or x == 0:
        return "0"
    a = abs(x)
    div, suf = 1.0, ""
    if a >= 1e9:
        div, suf = 1e9, "B"
    elif a >= 1e6:
        div, suf = 1e6, "M"
    elif a >= 1e3:
        div, suf = 1e3, "k"
    m = x / div
    intdigits = int(math.floor(math.log10(abs(m)))) + 1
    dp = max(0, sig - intdigits)
    return f"{m:.{dp}f}{suf}"


def _fmt_ce(val):
    if val is None:
        return "N/A"
    if val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    return f"${val:,.0f}"


def _fmt_usd(v):
    """USD card-VALUE formatter — single-sourced on _fmt_sig so dollar precision
    matches every other card (3 sig figs, floating k/M): 559,410,000 → '$559M',
    870,000 → '$870k'. Value-only — labels stay $-free. Carbon keeps its own
    _fmt_carbon_dollars (lower $M threshold) so its '$0.87M' display is unaffected."""
    return f"${_fmt_sig(v)}"


# ── Placement strategies ──────────────────────────────────────────────────────
# Five named strategies for selecting which convertible pixels to convert.
# 'random' is the default and reproduces the prior uniform-sampling behavior.
# The others weight the sampling toward pixels where conversion yields the
# highest benefit per the docs/research/INVEST_PLACEMENT.md research. UI exposure is deferred
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
    # See docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md §3 and §7.
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
                      carbon_rate_gi=None,
                      selected_region_mask=None):
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
    #
    # Region Selection Phase 1 — when `selected_region_mask` is provided,
    # filter the convertible pool to only pixels inside the mask. The
    # suitability machinery (`_compute_suitability_weights` /
    # `_select_pixels_for_conversion`) ranks within whatever (N, 2) array of
    # (row, col) pairs it's given, so swapping in `region_convertible_pixels`
    # ranks within the masked set with no changes to the strategy code.
    # `selected_region_mask=None` is byte-identical to prior behavior.
    if selected_region_mask is not None:
        _cp_rows = CONVERTIBLE_PIXELS[:, 0]
        _cp_cols = CONVERTIBLE_PIXELS[:, 1]
        _in_region = selected_region_mask[_cp_rows, _cp_cols]
        region_convertible_pixels = CONVERTIBLE_PIXELS[_in_region]
    else:
        region_convertible_pixels = CONVERTIBLE_PIXELS

    n_convert = int(len(region_convertible_pixels) * pct_converted / 100)

    rng = np.random.default_rng(seed)

    chosen_idx = _select_pixels_for_conversion(
        region_convertible_pixels, n_convert, placement_strategy, rng)

    pixels_to_convert = region_convertible_pixels[chosen_idx]

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
    # Children's nature access RELAY — when child_pop_count_raster is
    # available, the same call also returns the under-18 access share
    # (same adequate mask, child-weighted). None when no child raster
    # configured for this city.
    if child_pop_count_raster is not None:
        (nat_pct, _nat_quality, nat_people,
         children_nat_pct, children_nat_people) = calculate_nature_access(
            scenario_lulc_una, pop_count_raster,
            child_pop_count_raster=child_pop_count_raster,
        )
    else:
        nat_pct, _nat_quality, nat_people = calculate_nature_access(
            scenario_lulc_una, pop_count_raster
        )
        children_nat_pct = None
        children_nat_people = None

    # Nature Access at Schools (RELAY) — destination-based readout
    # sampled at the SAME 2SFCA adequate mask as the residential metric.
    # None when no schools_file is configured for this city; otherwise
    # returns a dict with pct + per-sector counts. See
    # docs/internal/DESIGN_NOTES §6.7.
    schools_access = calculate_schools_nature_access(
        scenario_lulc_una, pop_count_raster,
        SCHOOLS_PIXELS, SCHOOLS_SECTORS,
    )

    # Build the scenario NDVI raster once and pass to both consumers. Saves
    # one full-AOI float32 allocation per evaluate_scenario call.
    scenario_ndvi = _lulc_to_ndvi_raster(scenario_lulc)
    mean_ndvi = compute_mean_ndvi(scenario_lulc, ndvi_raster=scenario_ndvi)

    total_developed_acres = len(developed_pixels) * PIXEL_AREA_ACRES
    total_cost_mln = compute_cost(n_wet, n_for, n_hd, cost_gi, cost_ff, cost_hd)
    runoff_acft    = cn_to_runoff_acre_feet(mean_cn, total_developed_acres)
    # Additive third flood reading (Relay 58): canonical InVEST UFR per-pixel
    # retention index `rnf_rt_idx = mean(1 − Q/P)`. Engine-computed (NOT a
    # surrogate target — deterministic from CN, like UMH). Same masking as
    # `mean_cn`. Distinct from the lumped Flood Index by Jensen's inequality.
    runoff_retention_idx = cn_array_to_retention_index(cn_scenario, cn_scenario > 0)
    flood_damage_avoided_usd = compute_flood_damage_avoided(runoff_acft)

    # InVEST UMH preventable mental health cases + avoided cost (depression +
    # anxiety, NDVI-mediated). Returns (0, 0) if population data isn't loaded.
    preventable_mh_cases, avoided_mh_cost_usd = calculate_mental_health_impact(
        scenario_lulc, _BASELINE_NE_RASTER, pop_count_raster,
        ndvi_raster=scenario_ndvi,
    )

    # ── Region-Local Metrics (REGION_LOCAL_METRICS_SPEC.md) ──────────────────
    # Region-clipped values for every metric in `_REGION_LOCAL_METRICS`, using
    # the locked per-model treatment from the spec (pixel-clip vs population-
    # clip). None for citywide / non-region scenarios. The verify_baselines
    # reconciliation assertion guarantees: for any key, region_local over the
    # entire AOI must equal the citywide value.
    if selected_region_mask is not None:
        rm = selected_region_mask
        # Region developed-acre denominator for the runoff closed form (mirrors
        # how citywide uses `total_developed_acres`).
        _rl_developed = np.zeros_like(rm, dtype=bool)
        _rl_developed[developed_pixels[:, 0], developed_pixels[:, 1]] = True
        _rl_developed_in_region = _rl_developed & rm
        _rl_developed_acres = float(_rl_developed_in_region.sum()) * PIXEL_AREA_ACRES

        # Means — same per-pixel arrays as the citywide aggregations, just
        # masked. Empty intersections fall back to the citywide scalar.
        _rl_cn_valid = (cn_scenario > 0) & rm
        _rl_mean_cn = float(cn_scenario[_rl_cn_valid].mean().round(2)) if _rl_cn_valid.any() else mean_cn
        _rl_hm_valid = (~np.isnan(hmi_map)) & (scenario_lulc != NODATA) & rm
        _rl_mean_hm = float(hmi_map[_rl_hm_valid].mean().round(4)) if _rl_hm_valid.any() else mean_hm
        _rl_ndvi_valid = (scenario_lulc != NODATA) & rm
        _rl_mean_ndvi = float(round(scenario_ndvi[_rl_ndvi_valid].mean(), 4)) if _rl_ndvi_valid.any() else mean_ndvi

        # Region baseline HM for the temp_change_f delta — hm_to_temp_change_f
        # uses _CURRENT_CITY_STATE.baseline_hm (citywide scalar), which gives
        # a mixed reading when the scenario mean is regional. Compute the
        # region baseline mean directly so the region delta is fully regional.
        _rl_base_hm_raster = _CURRENT_CITY_STATE.baseline_hm_raster
        _rl_base_hm_valid = (~np.isnan(_rl_base_hm_raster)) & rm
        _rl_base_mean_hm = float(_rl_base_hm_raster[_rl_base_hm_valid].mean().round(4)) if _rl_base_hm_valid.any() else float(_CURRENT_CITY_STATE.baseline_hm)
        _rl_delta_hm = _rl_mean_hm - _rl_base_mean_hm
        _rl_temp_change_f = round(-_rl_delta_hm * HM_TO_FAHRENHEIT, 1)

        # Closed-form runoff + derived flood_reduction.
        _rl_flood_reduction = round(100 - _rl_mean_cn, 2)
        _rl_runoff_acft = cn_to_runoff_acre_feet(_rl_mean_cn, _rl_developed_acres)
        _rl_flood_damage_avoided_usd = compute_flood_damage_avoided(_rl_runoff_acft)
        # Per-pixel retention index over the SAME region-masked CN pixels
        # (`_rl_cn_valid` = (cn_scenario > 0) & rm), reusing the masking above.
        _rl_runoff_retention_idx = cn_array_to_retention_index(cn_scenario, _rl_cn_valid)

        # Cooling energy savings — per-pixel kWh × $/kWh sum, masked.
        _rl_cooling_energy_savings_usd = compute_cooling_energy_savings(hmi_map, mask=rm)

        # UNA population-clip + UMH population-clip. Children's nature access
        # RELAY — pass child_pop too so the region-clipped child access
        # share is computed under the same adequate mask. Strongest pairing
        # is with the school-land ownership filter (the brief's intended use).
        if child_pop_count_raster is not None:
            (_rl_nat_pct, _, _rl_nat_people,
             _rl_children_nat_pct, _rl_children_nat_people) = (
                calculate_nature_access(
                    scenario_lulc_una, pop_count_raster, mask=rm,
                    child_pop_count_raster=child_pop_count_raster,
                )
            )
        else:
            _rl_nat_pct, _, _rl_nat_people = calculate_nature_access(
                scenario_lulc_una, pop_count_raster, mask=rm,
            )
            _rl_children_nat_pct = None
            _rl_children_nat_people = None
        _rl_preventable_mh_cases, _rl_avoided_mh_cost_usd = calculate_mental_health_impact(
            scenario_lulc, _BASELINE_NE_RASTER, pop_count_raster,
            ndvi_raster=scenario_ndvi, mask=rm,
        )

        # Sums / counts — pixels_to_convert was filtered to region_convertible_pixels
        # at the top of evaluate_scenario, so n_wet / n_for / n_hd / food / cost /
        # fellback are already region-local. Mirror them.
        _rl_food_mln_lbs = food_mln_lbs
        _rl_people_fed = food_to_people_fed(_rl_food_mln_lbs)
        _rl_total_cost_mln = total_cost_mln

        # Carbon — recompute the region-clipped four-pool stock delta for SA;
        # for MN the per-conversion-type analytical value is already region-local.
        if c_above_arr is not None:
            _rl_c_valid = ((scenario_lulc_carbon >= 0) & (cooling_lulc_compound >= 0) & rm)
            _n_c = len(c_above_arr)
            _scen_safe = np.clip(scenario_lulc_carbon, 0, _n_c - 1)
            _base_safe = np.clip(cooling_lulc_compound, 0, _n_c - 1)
            _scen_total = (c_above_arr[_scen_safe] + c_below_arr[_scen_safe]
                           + c_soil_arr[_scen_safe] + c_dead_arr[_scen_safe])
            _base_total = (c_above_arr[_base_safe] + c_below_arr[_base_safe]
                           + c_soil_arr[_base_safe] + c_dead_arr[_base_safe])
            _rl_delta = np.where(_rl_c_valid, _scen_total - _base_total, 0.0)
            _rl_carbon_tons_co2 = round(float(_rl_delta.sum()) * PIXEL_AREA_HA * (44.0 / 12.0), 1)
        else:
            _rl_carbon_tons_co2 = carbon_tons_co2
        _rl_carbon_value_usd = round(_rl_carbon_tons_co2 * EPA_SOCIAL_COST_CARBON, 0)

        region_local = {
            'mean_cn':              _rl_mean_cn,
            'flood_reduction':      _rl_flood_reduction,
            'runoff_acre_feet':     _rl_runoff_acft,
            'runoff_retention_idx': _rl_runoff_retention_idx,
            'flood_damage_avoided_usd': _rl_flood_damage_avoided_usd,
            'mean_hm':              _rl_mean_hm,
            'temp_change_f':        _rl_temp_change_f,
            'cooling_energy_savings_usd': _rl_cooling_energy_savings_usd,
            'mean_ndvi':            _rl_mean_ndvi,
            'n_wet':                n_wet,
            'n_for':                n_for,
            'n_hd':                 n_hd,
            'ff_fellback_pixels':   ff_fellback_pixels,
            'gi_fellback_pixels':   gi_fellback_pixels,
            'hd_fellback_pixels':   hd_fellback_pixels,
            'food_mln_lbs':         _rl_food_mln_lbs,
            'people_fed':           _rl_people_fed,
            'total_cost_mln':       _rl_total_cost_mln,
            'carbon_tons_co2':      _rl_carbon_tons_co2,
            'carbon_value_usd':     _rl_carbon_value_usd,
            'nature_access_pct':            _rl_nat_pct,
            'people_with_nature_access':    _rl_nat_people,
            'children_nature_access_pct':   _rl_children_nat_pct,
            'children_with_nature_access':  _rl_children_nat_people,
            'preventable_mh_cases':         _rl_preventable_mh_cases,
            'avoided_mh_cost_usd':          _rl_avoided_mh_cost_usd,
        }
    else:
        region_local = None

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
        # Relay 58 — canonical InVEST UFR per-pixel retention index
        # `rnf_rt_idx = mean(1 − Q/P)`, in [0, 1]. Additive sibling to the
        # Flood Index (which is unchanged). Engine-only; not a surrogate target.
        'runoff_retention_idx':     runoff_retention_idx,
        'mean_hm':                  mean_hm,
        'temp_change_f':            hm_to_temp_change_f(mean_hm),
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
        # Children's nature access RELAY — under-18 share of access. None
        # on cities without a child_pop_file configured. Same adequate
        # mask as nature_access_pct; just child-pop-weighted.
        'children_nature_access_pct':   children_nat_pct,
        'children_with_nature_access':  children_nat_people,
        # Nature Access at Schools (RELAY) — destination-based metric. None
        # on cities without schools_file. Flat scalar headlines (snapshotted
        # by verify_baselines) PLUS the full dict (kept for UI breakdowns +
        # tooltip).  Source: NCES CCD/PSS/EDGE 2021-22.
        'schools_nature_access_pct':    schools_access.get('pct'),
        'schools_n_total':              schools_access.get('n_total'),
        'schools_n_with_access':        schools_access.get('n_with_access'),
        'schools_public_pct':           (schools_access.get('by_sector') or {}).get('public', {}).get('pct'),
        'schools_charter_pct':          (schools_access.get('by_sector') or {}).get('charter', {}).get('pct'),
        'schools_private_pct':          (schools_access.get('by_sector') or {}).get('private', {}).get('pct'),
        'schools_nature_access':        schools_access,
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
        # Region Selection Phase 1 — structured block carrying the
        # mask-derivable fields. The descriptive fields (`layer`,
        # `selected_ids`) are stamped onto results['region_selection'] by the
        # caller (the chokepoint that built the mask from a (layer, ids)
        # selection). The block is complete by the time `results` leaves the
        # call site.
        #
        # **Contract — `selected_ids` carries LABEL VALUES, not positional
        # indices.** The caller translates labels → positional indices to
        # build the mask, then stamps the LABEL list onto `selected_ids`. So
        # downstream readers (export metadata.json, saved scenarios, the
        # provenance header in Commit 5) see ["5", "7"] for "District 5 and
        # District 7", not [4, 6]. Positional indices stay internal.
        #
        # `verify_baselines._snapshot_from_results` whitelist-skips this key
        # (the load-bearing scalar `eligible_pixels_in_region` gets a
        # targeted assertion in Commit 6's region baseline).
        'region_selection': {
            'mode': (
                'selected_regions' if selected_region_mask is not None
                else 'entire_aoi'
            ),
            'eligible_pixels_in_region': int(len(region_convertible_pixels)),
            'selected_area_acres': (
                float(int(selected_region_mask.sum()) * PIXEL_AREA_ACRES)
                if selected_region_mask is not None else None
            ),
            # Scenario Record Pass — converted_acres is the third leg of the
            # placement-funnel trio (selected_area / eligible / converted).
            # Citywide scenarios carry it too (eligible_pixels_in_region is
            # already populated to the citywide convertible count), so the
            # field is uniformly shaped across both modes.
            'converted_acres': float((n_wet + n_for + n_hd) * PIXEL_AREA_ACRES),
            'layer':        None,  # caller stamps with layer_key (e.g. 'council_districts')
            'selected_ids': None,  # caller stamps with label list (e.g. ['5', '7']) — NOT positional indices
        },
        # Region-Local Metrics (REGION_LOCAL_METRICS_SPEC.md) — region-clipped
        # per-metric values when a region mask is active, None otherwise.
        # Decomposable slots carry numeric values; non-decomposable slots are
        # explicit None (the UI consults `_REGION_LOCAL_METRICS` for reasons).
        # `verify_baselines._snapshot_from_results` whitelist-skips this key;
        # the reconciliation assertion in verify_baselines is the
        # baseline-safety guard.
        'region_local': region_local,
    }


# ── Scenario grid and lookup table ─────────────────────────────────────────────
# Bump SCENARIO_SCHEMA_VERSION whenever the surrogate target columns change so
# Streamlit's @st.cache_data automatically invalidates stale grids/tables.
SCENARIO_SCHEMA_VERSION = 36  # bumped: Relay 58 — evaluate_scenario's return dict gains one new field `runoff_retention_idx`: the canonical InVEST UFR per-pixel runoff-retention index `rnf_rt_idx = mean(1 - Q/P)`, in [0, 1], computed by `cn_array_to_retention_index` over the same `cn_scenario > 0` mask as `mean_cn` (region_local applies the same mask & region). Engine-computed only - NOT added to the surrogate y-columns (deterministic from CN, like UMH). Additive third flood reading; the Flood Index (`100 - mean_CN`) is UNCHANGED. Distinct from the lumped form by Jensen's inequality (Q is convex in CN). Lands as `aligned_method` (blue) - no published NatCap SA flood value to match. All 40 SA + MN baselines re-snapshotted; every scenarios_fast_*/dense_* CSV regenerated + re-stamped to carry the new column; reconciliation invariant (region_local over full AOI == citywide) holds for the new field. (35 was Nature Access at Schools RELAY - six new schools_* fields + the schools_nature_access dict; NCES/EDGE/PSS school points, SA 647 / MN 60.) (34 was Children's nature access RELAY - children_nature_access_pct + children_with_nature_access.) (33 was Batch 4 v2 of Finer Ownership Classes - composite ownership_filter dict shape.)

# Surrogate target columns that downstream code (train_surrogate, optimize_scenario)
# requires. Listed explicitly so a missing column fails loudly instead of leaking
# into a KeyError deep in fit().
REQUIRED_TARGET_COLUMNS = [
    'flood_reduction', 'mean_hm', 'food_mln_lbs', 'runoff_acre_feet',
    'runoff_retention_idx',  # Relay 58 — per-pixel UFR retention index
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
                                        'scenario_lulc_una', 'scenario_lulc_carbon',
                                        'region_selection', 'region_local')}
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
                                          'scenario_lulc_una', 'scenario_lulc_carbon',
                                          'region_selection', 'region_local')}
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


def compute_cooling_energy_savings(scenario_hmi_raster, mask=None):
    """Annual avoided AC cost ($/yr) for buildings under the active scenario,
    using the canonical InVEST UCM energy-valuation formula.

    Per pixel: `ΔT_°C = (HMI_scenario − HMI_baseline) × UHI_MAX_C`. The InVEST
    `consumption` column is documented as kWh/(m²·°C), so the per-pixel kWh
    saved is `consumption_rate × ΔT_°C × pixel_area_m²`, and the dollar value
    is multiplied by `$/kWh`. Negative ΔT (scenario hotter than baseline) is
    clamped to zero — we only credit cooling, not penalise warming. Sums over
    building pixels and returns $0 when buildings, the energy table, or the
    ET raster are unavailable.

    `mask` (Region-Local Metrics Commit 1) — when provided, the sum runs only
    over masked building pixels. `mask=None` reproduces the citywide sum.

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
    if mask is not None:
        valid = valid & mask
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

    # ── Phase 2b: Child population raster (RELAY) ──────────────────────────
    # Under-18 per pixel, same source/vintage/resolution/CRS as pop_file.
    # Used ONLY by calculate_nature_access to compute the access share
    # weighted by child population; 2SFCA supply/demand stays on total
    # pop, UMH stays on total pop. None for cities without a child_pop_file
    # configured.
    child_pop_file = cfg.get("child_pop_file")
    try:
        if child_pop_file is None:
            raise FileNotFoundError("child_pop_file not configured")
        child_pop_count_raster = load_population_data(
            child_pop_file, l_cooling_lulc.shape, cfg['crs']
        )
        child_population_data_available = True
    except (FileNotFoundError, rasterio.errors.RasterioIOError, TypeError):
        child_pop_count_raster = None
        child_population_data_available = False

    # ── Phase 2c: Schools points (RELAY — Nature Access at Schools) ────────
    # Load the per-city K-12 school points (public + charter + private from
    # NCES CCD/PSS/EDGE), project to the LULC CRS, and convert each to
    # (row, col) pixel coordinates via the LULC's affine inverse. Schools
    # whose pixel falls outside the raster are dropped (the per-city prep
    # script already clips to the bbox, so this is a safety net for floor
    # edges). None for cities without a schools_file configured.
    schools_file = cfg.get("schools_file")
    try:
        if schools_file is None:
            raise FileNotFoundError("schools_file not configured")
        _schools_gdf = _gpd.read_file(schools_file)
        if _schools_gdf.crs is None or str(_schools_gdf.crs) != cfg['crs']:
            _schools_gdf = _schools_gdf.to_crs(cfg['crs'])
        # Convert (x, y) → (row, col) via the rasterization template.
        # ref_transform is set in Phase 7 below; we need it earlier here, so
        # open the LULC briefly to read its transform.
        with rasterio.open(
            f"{cfg['data_dir_cooling']}/{cfg['cooling_lulc_file']}"
        ) as _src:
            _t = _src.transform
            _h, _w = _src.height, _src.width
        _rows, _cols = [], []
        _sectors = []
        for geom, sector in zip(_schools_gdf.geometry, _schools_gdf["sector"]):
            r, c = rasterio.transform.rowcol(_t, geom.x, geom.y)
            if 0 <= r < _h and 0 <= c < _w:
                _rows.append(int(r))
                _cols.append(int(c))
                _sectors.append(str(sector))
        schools_pixels = np.array(list(zip(_rows, _cols)), dtype=np.int64) \
            if _rows else np.empty((0, 2), dtype=np.int64)
        schools_sectors = np.array(_sectors, dtype=object)
        _sec_counts = {
            s: int((schools_sectors == s).sum())
            for s in ("public", "charter", "private")
        }
        schools_metadata = {
            "source_file": schools_file,
            "vintage": "NCES CCD 2022-23 + EDGE 2021-22 + PSS 2021-22",
            "n_total": int(len(schools_pixels)),
            "sector_counts": _sec_counts,
        }
        schools_data_available = True
        print(f"[SCHOOLS] {city_key}: {schools_metadata['n_total']:,} schools "
              f"on-extent (public={_sec_counts['public']}, "
              f"charter={_sec_counts['charter']}, private={_sec_counts['private']})")
        del _schools_gdf
    except Exception:
        schools_pixels = None
        schools_sectors = None
        schools_metadata = None
        schools_data_available = False

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
    # SA cold-start Lever 2: if buildings_precomputed_* are configured AND all
    # three files exist (binary raster, typed raster, meta sidecar), read them
    # from disk and skip the rasterize. Saves ~32 s on SA. The gate runs a
    # fresh rasterize and asserts byte-identity vs the on-disk files; if the
    # source `buildings_file` changes without a precompute re-run, the gate
    # fails on the staleness cell. Generated by `precompute_buildings.py`.
    # MN keeps the live rasterize path (its building polygon count is small).
    buildings_have_types = False
    buildings_type_coverage = 0.0
    buildings_file = cfg.get("buildings_file")
    damage_table_file = cfg.get("damage_table_file")
    _b_pre_bin  = cfg.get("buildings_precomputed_file")
    _b_pre_type = cfg.get("buildings_type_precomputed_file")
    _b_pre_meta = cfg.get("buildings_precomputed_meta_file")
    _b_precomputed_available = bool(
        _b_pre_bin  and Path(_b_pre_bin).exists() and
        _b_pre_type and Path(_b_pre_type).exists() and
        _b_pre_meta and Path(_b_pre_meta).exists()
    )

    if _b_precomputed_available:
        try:
            with rasterio.open(_b_pre_bin) as _src_b:
                _assert_raster_crs(_src_b, cfg['crs'], _b_pre_bin)
                buildings_raster = _src_b.read(1).astype("uint8")
            with rasterio.open(_b_pre_type) as _src_t:
                _assert_raster_crs(_src_t, cfg['crs'], _b_pre_type)
                buildings_type_raster = _src_t.read(1).astype("int32")
            import json as _json
            _meta = _json.loads(Path(_b_pre_meta).read_text())
            total_potential_damage_usd = float(_meta.get("total_potential_damage_usd", 0.0))
            total_building_pixels = int(np.sum(buildings_raster > 0))
            typed_pixels = int(np.sum(buildings_type_raster > 0))
            if total_building_pixels > 0:
                buildings_type_coverage = typed_pixels / total_building_pixels
            buildings_have_types = bool(_meta.get("buildings_have_types", typed_pixels > 0))
            if not buildings_have_types:
                buildings_have_types = typed_pixels > 0
            print(
                f"[BUILDINGS] {city_key}: {typed_pixels:,}/{total_building_pixels:,} "
                f"building pixels typed ({buildings_type_coverage:.1%} coverage) "
                "[precomputed]"
            )
            buildings_data_available = True
        except Exception as _e:
            # Corrupt file / CRS mismatch / missing meta key → log and fall
            # through to the live rasterize. Gate's staleness cell catches
            # the real cause in CI.
            print(f"[BUILDINGS] {city_key}: precomputed load failed ({_e}); "
                  "falling back to live rasterize")
            _b_precomputed_available = False

    if not _b_precomputed_available:
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

    # ── Phase 12b: Region-selection polygon layers (Phase 1) ─────────────────
    # For each declared region layer, rasterize the polygons to an int32 raster
    # carrying positional indices (0..N-1) with -1 fill, mirroring the tracts
    # rasterize above. Caller-side mask construction:
    #     mask = np.isin(region_rasters[layer_key], selected_positional_indices)
    # See REGION_SELECTION_PHASE1_SPEC.md.
    region_rasters = {}
    region_layer_labels = {}
    region_layer_display_names = {}
    for _layer_key, _layer_cfg in (cfg.get("region_layers") or {}).items():
        try:
            _layer_gdf = _gpd.read_file(_layer_cfg["path"])
            if _layer_gdf.crs is None or str(_layer_gdf.crs) != cfg["crs"]:
                _layer_gdf = _layer_gdf.to_crs(cfg["crs"])
            _layer_gdf = _layer_gdf.reset_index(drop=True)
            _region_raster = _rasterize(
                ((g, i) for i, g in enumerate(_layer_gdf.geometry)),
                out_shape=ref_shape, transform=ref_transform,
                fill=-1, dtype=np.int32,
            )
            region_rasters[_layer_key] = _region_raster
            region_layer_labels[_layer_key] = (
                _layer_gdf[_layer_cfg["label_field"]].astype(str).tolist()
            )
            region_layer_display_names[_layer_key] = _layer_cfg["display_name"]
        except Exception as exc:
            print(
                f"  WARN: region layer {_layer_key!r} failed to load "
                f"({exc!r}); skipping."
            )

    # ── Phase 12c: Ownership raster (Finer Ownership Classes Pass) ───────────
    # Two-band int8 raster on the active grid:
    #   Band 1 = ownership class enum 0-5 (private / city / county /
    #            state-federal / school-university / unknown); nodata=-1.
    #   Band 2 = is_vacant 0/1; nodata=-1.
    # Built by `scripts/data/download_bexar_parcels.py` per
    # OWNERSHIP_FINER_CLASSES_SPEC.md. SA-only; MN has no `ownership_layer`
    # config. `state.ownership_raster` retains its name for backward compat
    # and now holds BAND 1 (the class enum); `state.ownership_vacant_raster`
    # holds BAND 2. The CRS assertion stays the safety net.
    ownership_raster: Optional[np.ndarray] = None
    ownership_vacant_raster: Optional[np.ndarray] = None
    _ownership_cfg = cfg.get("ownership_layer")
    if _ownership_cfg is not None:
        try:
            _own_path = _ownership_cfg["path"]
            with rasterio.open(_own_path) as _own_src:
                _assert_raster_crs(_own_src, cfg["crs"], _own_path)
                ownership_raster = _own_src.read(1)
                if _own_src.count >= 2:
                    ownership_vacant_raster = _own_src.read(2)
                else:
                    # Legacy single-band file detected — config still points
                    # to it on some old branches. Surface loudly because the
                    # mode→mask path now expects two bands.
                    print(
                        f"  WARN: ownership raster {_own_path!r} is single-band; "
                        "the Finer Ownership Classes Pass requires the two-band "
                        "file produced by `download_bexar_parcels.py "
                        "--reclassify-from <gpkg>`. Ownership disabled for "
                        "this city."
                    )
                    ownership_raster = None
            if ownership_raster is not None and ownership_raster.shape != ref_shape:
                print(
                    f"  WARN: ownership raster shape {ownership_raster.shape} != "
                    f"ref_shape {ref_shape}; ownership disabled for this city."
                )
                ownership_raster = None
                ownership_vacant_raster = None
        except Exception as exc:
            print(f"  WARN: ownership raster failed to load ({exc!r}); skipping.")
            ownership_raster = None
            ownership_vacant_raster = None

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
    baseline_ne_raster = _umh_neighborhood_exposure(
        _lulc_to_ndvi_raster(l_cooling_lulc))
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
        child_pop_count_raster=child_pop_count_raster,
        child_population_data_available=child_population_data_available,
        schools_pixels=schools_pixels,
        schools_sectors=schools_sectors,
        schools_metadata=schools_metadata,
        schools_data_available=schools_data_available,
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
        region_rasters=region_rasters,
        region_layer_labels=region_layer_labels,
        region_layer_display_names=region_layer_display_names,
        ownership_raster=ownership_raster,
        ownership_vacant_raster=ownership_vacant_raster,
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


# ── Record-display helpers (shared across tabs + the audit expander) ─────────
# Used by the Compare scenarios table (tab2), the Scenario audit expander
# (above the metric cards), and the Scenario CSV export (tab2). Compose at
# render time from the fields evaluate_scenario stamps onto results — the
# underlying record stays minimal; the rich Area / Ownership view lives in
# this composition rule. Defined at module scope (right after
# _CURRENT_CITY_STATE is bound, since they read its region_layer_display_names)
# so every downstream block can call them without duplicating the logic. Both
# functions accept either a results dict or a saved-scenario dict (same shape
# sans `scenario_lulc`).
def _cs_area_for_row(row):
    rs = (row or {}).get('region_selection') or {}
    if rs.get('mode') != 'selected_regions' or rs.get('layer') is None:
        return "Citywide"
    layer = rs['layer']
    ids = rs.get('selected_ids') or []
    display = _CURRENT_CITY_STATE.region_layer_display_names.get(layer, "region")
    n = len(ids)
    if n == 1:
        return f"{display} {ids[0]}"
    if 1 < n <= 3:
        return f"{n} selected {display}s ({', '.join(ids)})"
    return f"{n} selected {display}s"


def _cs_ownership_for_row(row):
    """Display label for `row['ownership_filter']`. Handles all three
    shapes the storage path produces: None, a single OWNERSHIP_MODES
    key (Batch 4 v1 + earlier saved scenarios), or a composite dict
    {'classes': [...], 'vacant': bool} (Batch 4 v2's multi-class UI)."""
    norm = _normalize_ownership_filter((row or {}).get('ownership_filter'))
    return norm['label'] if norm else "None"

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
child_pop_count_raster          = _CURRENT_CITY_STATE.child_pop_count_raster
CHILD_POPULATION_DATA_AVAILABLE = _CURRENT_CITY_STATE.child_population_data_available
SCHOOLS_PIXELS         = _CURRENT_CITY_STATE.schools_pixels
SCHOOLS_SECTORS        = _CURRENT_CITY_STATE.schools_sectors
SCHOOLS_METADATA       = _CURRENT_CITY_STATE.schools_metadata
SCHOOLS_DATA_AVAILABLE = _CURRENT_CITY_STATE.schools_data_available
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
    """One row per tract: baseline and scenario temperature vs the city-wide
    average (°F), plus the scenario's own temperature change.

    Sign convention (ΔT = T_after − T_before): positive = WARMER, negative =
    cooler. The `vs city avg` columns give each polygon's mean temperature
    relative to the city-wide baseline mean (positive = warmer than average).
    `_change_f` is the scenario's effect (positive = warmer) used for sorting
    and color coding; the display layer renders it as natural-language text.

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
        # Temperature vs the city-wide baseline mean, in °F, under the
        # ΔT = T_after − T_before convention (positive = WARMER than the city
        # average, negative = cooler). Higher HM = more cooling = cooler, so
        # the HM offset is negated.
        valid_hm = mask & ~np.isnan(_BASELINE_HM_RASTER) & ~np.isnan(hm_s_raster)
        if not valid_hm.any():
            continue
        b_hm = _BASELINE_HM_RASTER[valid_hm].mean()
        s_hm = hm_s_raster[valid_hm].mean()
        # read from state to avoid silent-staleness if city switches
        b_anom_f = (_CURRENT_CITY_STATE.baseline_hm - b_hm) * HM_TO_FAHRENHEIT
        s_anom_f = (_CURRENT_CITY_STATE.baseline_hm - s_hm) * HM_TO_FAHRENHEIT
        rows.append({
            "GEOID":                      str(TRACTS.iloc[i].get("GEOID10", i)),
            "Population":                 int(pop_in_tract),
            "Baseline vs city avg (°F)":  round(b_anom_f, 2),
            "Scenario vs city avg (°F)":  round(s_anom_f, 2),
            "_change_f":                  round(s_anom_f - b_anom_f, 2),
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
# Relay 37 — dense Balanced-grid provenance (parity with the Fast grid). The
# dense CSV is pct step 5 / gi+ff step 10, built by precompute_scenarios.py.
_DENSE_GRID_STEP_PCT = 5
_DENSE_GRID_STEP_ALLOC = 10
_DENSE_GRID_FORMAT_VERSION = 1


def _load_dense_grid_artifact(city_key):
    """Return the precomputed dense Balanced grid for `city_key`, or None.

    Validates CITIES[city_key]['dense_scenarios_file'] + its '<path>.meta.json'
    sidecar (city / step params / format / SCENARIO_SCHEMA_VERSION + the
    surrogate-relevant REQUIRED_TARGET_COLUMNS + recipe count) before loading.
    Any mismatch / missing file / read error returns None so the caller keeps
    its existing live-rebuild fallback — degrades to slow, never breaks. Mirrors
    _load_fast_grid_artifact; numeric staleness is also caught by the
    verify_baselines dense-CSV freshness spot-check."""
    import json as _json
    path = (CITIES.get(city_key) or {}).get('dense_scenarios_file')
    if not path or not os.path.exists(path):
        return None
    meta_path = path + '.meta.json'
    try:
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as _f:
            meta = _json.load(_f)
        if (meta.get('dense_grid_format_version') != _DENSE_GRID_FORMAT_VERSION
                or meta.get('city_key') != city_key
                or meta.get('step_pct') != _DENSE_GRID_STEP_PCT
                or meta.get('step_alloc') != _DENSE_GRID_STEP_ALLOC
                or meta.get('scenario_schema_version') != SCENARIO_SCHEMA_VERSION):
            return None
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in df.columns]
        if missing or len(df) != meta.get('n_recipes'):
            return None
        return df
    except Exception:
        return None


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
        _dense_df = _load_dense_grid_artifact(selected_city)
        if _dense_df is not None:
            scenario_df = _dense_df
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
                    f"⚠️ Balanced mode: `{_dense_configured}` missing or stale "
                    f"(provenance stamp mismatch) — recomputing on the fly. Run "
                    f"`python3 precompute_scenarios.py --city '{selected_city}' "
                    f"--output {_dense_configured}` once to skip this on future startups."
                )
            scenario_df = compute_scenario_grid(
                _CURRENT_CITY_STATE, selected_city,
                DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=5, step_alloc=10,
            )
        ACTIVE_MODEL_QUALITY = "balanced"
    else:  # Fast prototype
        # Lever 1 — Fast mode reads the precomputed dense grid when it
        # exists (same source as Balanced). The 726-row dense grid is
        # strictly richer than Fast's historical 91-row coarse grid (pct
        # step 5/10 vs 10/25); the Fast surrogate trains on the same data
        # as Balanced but with fewer trees (N_ESTIMATORS distinction
        # preserved in the SURROGATE_TREES dict). Containment: changes
        # ONLY the surrogate's training source — the validated full-engine
        # apply path is untouched, and the 91-row historical surrogate
        # was itself an exploratory predictor, not an authoritative metric.
        # Cold start savings on SA: ~130 s of compute_scenario_grid removed.
        # Falls back to live grid if the CSV is missing (printed warning,
        # not a hard fail — keeps cold start usable in dev workflows).
        lookup_table = {}
        _dense_configured = city_cfg.get("dense_scenarios_file")
        _dense_df = _load_dense_grid_artifact(selected_city)
        if _dense_df is not None:
            scenario_df = _dense_df
        else:
            if _dense_configured:
                print(
                    f"[FAST] dense_scenarios_file {_dense_configured!r} missing "
                    f"or stale (provenance stamp mismatch) for {selected_city!r} "
                    f"— recomputing on the fly. Run `python3 "
                    f"precompute_scenarios.py --city {selected_city!r} --output "
                    f"{_dense_configured}` once to skip this on future cold starts."
                )
            else:
                print(
                    f"[FAST] no dense_scenarios_file configured for "
                    f"{selected_city!r} — recomputing on the fly."
                )
            scenario_df = compute_scenario_grid(
                _CURRENT_CITY_STATE, selected_city,
                DATA_DIR_FLOOD, DATA_DIR_COOLING, step_pct=10, step_alloc=25,
            )
        ACTIVE_MODEL_QUALITY = "fast"

MAX_FOOD  = float(scenario_df['food_mln_lbs'].max())
MAX_FLOOD = 100.0
MAX_COOL  = 1.1

# ── Relay 60 Part B — calibrated estimate ranges ────────────────────────────
# The citywide suggestion "Estimate range" is the empirically-calibrated 10th–
# 90th residual interval [estimate + p10, estimate + p90] from
# scripts/calibrate_surrogate_band.py (k-fold CV vs the engine grid). Residual
# convention is engine_true − surrogate_pred, so the interval brackets the
# engine truth and carbon/food under-prediction shows as upward skew. Loaded
# per (city slug, active mode); None → ranges are suppressed for that mode
# (e.g. MN Fast, which has no precomputed grid file to calibrate against).
# Region suggestions never get a range (engine-verified — Relay 60 Part A).
_CALIB_MODE = {"fast": "fast", "balanced": "balanced", "high": "balanced"}.get(
    ACTIVE_MODEL_QUALITY, "balanced")
# City slug derived from the dense-grid filename (config-driven, no hardcoded
# city names): 'data/scenarios_dense_sa.csv' → 'sa'.
_dense_file = city_cfg.get("dense_scenarios_file") or ""
_CITY_SLUG = (os.path.basename(_dense_file)
              .replace("scenarios_dense_", "").replace(".csv", "")) if _dense_file else None
# Band metric → (lower_col, upper_col) on the optimizer DataFrame. Nature access
# is deliberately absent: it is SUPPRESSED (lattice CV understates its true,
# placement-driven uncertainty) — it keeps its caveat, not a calibrated band.
_CALIB_BAND_METRICS = {
    "flood_reduction": ("flood_lower", "flood_upper"),
    "mean_hm":         ("hm_lower", "hm_upper"),
    "food_mln_lbs":    ("food_lower", "food_upper"),
    "carbon_tons_co2": ("carbon_lower", "carbon_upper"),
}


@st.cache_data(show_spinner=False)
def _load_surrogate_calibration(slug, mode, schema_version):
    """Load data/<slug>/surrogate_calibration_<mode>.json, or None. Validates
    the stamp's schema against the live SCENARIO_SCHEMA_VERSION so a stale
    artifact is ignored (runtime falls back to no range — never a wrong one).

    Failure policy: only *data* problems degrade to None — missing file,
    unreadable file, malformed JSON, wrong-shaped or missing keys, stale stamp.
    *Code* problems (NameError / AttributeError / TypeError) PROPAGATE loudly.
    Hence the narrow `except (OSError, ValueError)` around the read+parse plus
    explicit dict-shape guards — NOT a bare `except Exception`, which previously
    swallowed a `json` NameError and left the Estimate range dark for every
    city/mode since Relay 60B."""
    if not slug:
        return None
    path = Path(f"data/{slug}/surrogate_calibration_{mode}.json")
    if not path.exists():
        return None
    try:
        art = json.loads(path.read_text())
    except (OSError, ValueError):
        return None  # unreadable file or malformed JSON — a data problem
    if not isinstance(art, dict):
        return None
    prov = art.get("provenance", {})
    if not isinstance(prov, dict) or prov.get("scenario_schema_version") != schema_version:
        return None
    rq = art.get("residual_quantiles")
    return rq if isinstance(rq, dict) and rq else None


_ACTIVE_CALIBRATION = _load_surrogate_calibration(
    _CITY_SLUG, _CALIB_MODE, SCENARIO_SCHEMA_VERSION)


def _apply_calibrated_ranges(opt_df, calib):
    """Replace the optimizer's inter-tree bands with the calibrated 10th–90th
    residual interval [est + p10, est + p90] per metric. With no calibration
    for the active mode, DROP the band columns so every downstream surface
    suppresses ranges (no error bars, no range columns). Returns opt_df."""
    if not isinstance(opt_df, pd.DataFrame) or opt_df.empty:
        return opt_df
    q = calib or {}
    for est_col, (lo_col, hi_col) in _CALIB_BAND_METRICS.items():
        mq = q.get(est_col)
        if mq and est_col in opt_df.columns:
            opt_df[lo_col] = (opt_df[est_col] + mq["p10"])
            opt_df[hi_col] = (opt_df[est_col] + mq["p90"])
        else:
            opt_df.drop(columns=[c for c in (lo_col, hi_col)
                                 if c in opt_df.columns],
                        inplace=True, errors="ignore")
    return opt_df

# read from state to avoid silent-staleness if city switches
BASELINE_RUNOFF_ACRE_FEET = cn_to_runoff_acre_feet(
    _CURRENT_CITY_STATE.baseline_cn, len(developed_pixels) * PIXEL_AREA_ACRES
)

# Relay 58 — baseline per-pixel retention index. Every pct_converted=0 grid row
# is the unconverted baseline (no conversions regardless of gi/ff), so the
# no-conversion row's `runoff_retention_idx` is the baseline reading. Guarded
# for the transient pre-regen state where the column may be absent.
try:
    _b0_ret = scenario_df.loc[scenario_df['pct_converted'] == 0, 'runoff_retention_idx']
    BASELINE_RUNOFF_RETENTION_IDX = float(_b0_ret.iloc[0]) if len(_b0_ret) else None
except (KeyError, IndexError):
    BASELINE_RUNOFF_RETENTION_IDX = None

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


# ── Fast-only surrogate for the region-constrained optimizer ────────────────
# Phase-0.5 validated the prefilter at the Fast configuration only (~90
# recipes, 100 trees). Balanced and High Resolution surrogates haven't been
# checked for the same ranking quality vs the engine on region-scoped
# candidates. To keep the validated configuration in use regardless of the
# Model-quality radio, the region optimizer ALWAYS prefilters with a Fast
# surrogate — built lazily and cached per city so a Balanced / High user
# pays the ~3-minute Fast-grid build once per session, not per click.
# Relay 35 — Fast-grid build parameters. Shared by the precompute script
# (scripts/regenerate_fast_grid.py), the artifact-load validator, and the
# live-build fallback so all three agree on the grid shape. Bump
# _FAST_GRID_FORMAT_VERSION if the artifact format/columns change in a way the
# stamp must reject.
_FAST_GRID_STEP_PCT = 10
_FAST_GRID_STEP_ALLOC = 25
_FAST_GRID_FORMAT_VERSION = 1


def _load_fast_grid_artifact(city_key):
    """Return a precomputed Fast-grid DataFrame for `city_key`, or None.

    Loads CITIES[city_key]['fast_grid_file'] (built by
    scripts/regenerate_fast_grid.py) only when it AND its '<path>.meta.json'
    sidecar exist and the stamp matches the current city / step params /
    format / SCENARIO_SCHEMA_VERSION, and the CSV carries the required columns
    and recipe count. Any mismatch / missing file / read error returns None, so
    the caller degrades to a live build — never breaks. Numeric staleness from a
    math change that did NOT bump the schema is caught by the verify_baselines
    Fast-grid spot-check (mirrors the dense-CSV guard)."""
    import json as _json
    path = (CITIES.get(city_key) or {}).get('fast_grid_file')
    if not path or not os.path.exists(path):
        return None
    meta_path = path + '.meta.json'
    try:
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as _f:
            meta = _json.load(_f)
        if (meta.get('fast_grid_format_version') != _FAST_GRID_FORMAT_VERSION
                or meta.get('city_key') != city_key
                or meta.get('step_pct') != _FAST_GRID_STEP_PCT
                or meta.get('step_alloc') != _FAST_GRID_STEP_ALLOC
                or meta.get('scenario_schema_version') != SCENARIO_SCHEMA_VERSION):
            return None
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in df.columns]
        if missing or len(df) != meta.get('n_recipes'):
            return None
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_fast_scenario_grid(_state, city_key,
                               data_dir_flood, data_dir_cooling):
    _loaded = _load_fast_grid_artifact(city_key)
    if _loaded is not None:
        return _loaded
    return compute_scenario_grid(
        _state, city_key, data_dir_flood, data_dir_cooling,
        step_pct=_FAST_GRID_STEP_PCT, step_alloc=_FAST_GRID_STEP_ALLOC,
    )


@st.cache_resource(show_spinner=False)
def _cached_fast_surrogate_for_region(_state, city_key,
                                      data_dir_flood, data_dir_cooling):
    """Return (fast_scenario_df, fast_surrogate_model) for the region
    optimizer's prefilter. Independent of ACTIVE_MODEL_QUALITY so the
    Phase-0.5-validated configuration (Fast 90, 100 trees) is the only
    thing ever used to rank region candidates."""
    fast_df = _cached_fast_scenario_grid(
        _state, city_key, data_dir_flood, data_dir_cooling)
    fast_model = _train_surrogate_fn(fast_df, n_estimators=100)
    return fast_df, fast_model


# ── Shared optimize triggers (Optimizer Promotion) ──────────────────────────
# The sidebar Discover button AND the main-panel CTA must fire the same
# optimize logic on the same config so the user can trust that clicking
# either produces the same result. These helpers are the single fire site;
# both buttons call them with the same explicit args (no implicit module
# globals). verify_baselines asserts both call sites pass identical
# argument expressions to lock in the contract.
def _fire_citywide_optimize(
    surrogate_model,
    min_flood, min_cool, min_food, max_runoff, min_carbon,
    max_food, max_flood_const, max_cool_const,
):
    """Run the citywide surrogate optimizer. Writes to
    `st.session_state.optimized_results`. Surfaces success / no-result
    feedback inline."""
    with st.spinner("Searching for most efficient tradeoff scenarios..."):
        st.session_state.optimized_results = _apply_calibrated_ranges(
            optimize_scenario(
                surrogate_model, min_flood, min_cool, min_food, max_runoff,
                min_carbon=min_carbon, max_food=max_food,
                max_flood=max_flood_const, max_cool=max_cool_const,
            ),
            _ACTIVE_CALIBRATION,
        )
    _opt_res = st.session_state.optimized_results
    if _opt_res is None or (
            isinstance(_opt_res, dict) and not _opt_res.get('found')):
        st.warning("No scenarios found — try lowering the targets.")
    else:
        # On success, jump directly to Tradeoffs instead of nudging
        # the user to switch tabs manually. Setting the segmented_control's
        # session_state key + st.rerun() makes the switch happen on the
        # very next rerun (before any tab body renders this turn). Toast
        # replaces the prior 'open the Tradeoffs tab →' success
        # banner since the switch is the actual confirmation.
        st.session_state['main_tab'] = "Tradeoffs"
        st.toast("Results ready — opening Tradeoffs ↓")
        st.rerun()


def _fire_region_optimize(
    state, city_key, data_dir_flood, data_dir_cooling,
    region_mask, ownership_mask,
    cost_gi_val, cost_ff_val, cost_hd_val,
    weights,
):
    """Run the region-prefilter + engine-verify optimizer. Writes to
    `st.session_state.region_optimized_results`. Composes the
    region∩ownership mask, defines the engine-eval closure, pulls the
    cached Fast surrogate, and orchestrates the engine-verify pass under
    a progress bar — same code as the sidebar's prior inline block,
    factored so the main-panel CTA can call the same path."""
    # Compose the active region∩ownership mask the engine consumes.
    if region_mask is not None and ownership_mask is not None:
        opt_mask = region_mask & ownership_mask
    elif ownership_mask is not None:
        opt_mask = ownership_mask
    else:
        opt_mask = region_mask

    def _engine_eval(_pct, _gi, _ff):
        return evaluate_scenario(
            _pct, _gi, _ff,
            seed=42, placement_strategy='random',
            cost_gi=cost_gi_val, cost_ff=cost_ff_val, cost_hd=cost_hd_val,
            carbon_rate_ff=st.session_state.carbon_rate_ff,
            carbon_rate_gi=st.session_state.carbon_rate_gi,
            selected_region_mask=opt_mask,
        )

    with st.spinner("Preparing Fast prefilter (first run only)…"):
        fast_df, fast_surrogate = _cached_fast_surrogate_for_region(
            state, city_key, data_dir_flood, data_dir_cooling,
        )

    _prog = st.progress(0.0, text="Engine-verifying candidates 0 / 0…")
    def _progress(i, K):
        _prog.progress(i / K,
                       text=f"Engine-verifying candidates {i} / {K}…")
    try:
        region_out = optimize_scenario_region(
            fast_surrogate, fast_df, _engine_eval,
            weights,
            k_engine=40, top_n=5,
            progress_cb=_progress,
        )
        _prog.empty()
    except Exception as _e:
        _prog.empty()
        st.error(f"Optimization failed: {_e}")
        region_out = None

    if region_out is None or region_out.empty:
        st.warning(
            "No scenarios found — try widening the weights or "
            "selecting a different region."
        )
        st.session_state.region_optimized_results = None
    else:
        st.session_state.region_optimized_results = region_out
        # Auto-switch to Tradeoffs on success — same pattern as the
        # citywide branch above. See its comment for why.
        st.session_state['main_tab'] = "Tradeoffs"
        st.toast("Results ready — opening Tradeoffs ↓")
        st.rerun()


# ── Optimize-button help (Relay 33) — one string per mode, shared by the
# sidebar Optimize button and the main-CTA Optimize button so hovering either
# explains the same thing. Both buttons route through the same _fire_*_optimize
# helpers; the disambiguator is the sidebar "Same search…" caption added beside
# each sidebar button.
_OPTIMIZE_HELP_CITYWIDE = (
    "Searches citywide for promising mixes with the fast machine-learning "
    "model; apply one to recompute with the InVEST-aligned evaluator."
)
_OPTIMIZE_HELP_REGION = (
    "Finds best-tested mixes under your current area and filters."
)


# ── Plotting helpers ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_region_polygons_for_plotly(path: str, label_field: str):
    """Cache the polygon coordinate arrays for the interactive selector. Returns
    a list of `(label, [(xs, ys), ...])` tuples where xs/ys are exterior coords
    for each polygon ring in the region. EPSG:5070 (equal-area) so the visual
    shape is preserved without basemap distortion. Plain-cartesian plotly
    Scatter polygons + click events = zero new deps (Interactive Region Map
    Spec, Path C decision).
    """
    import geopandas as _gpd
    gdf = _gpd.read_file(path)
    if gdf.crs is None or str(gdf.crs) != "EPSG:5070":
        gdf = gdf.to_crs("EPSG:5070")
    out = []
    for _, row in gdf.iterrows():
        label = str(row[label_field])
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        rings = []
        for poly in polys:
            xs, ys = poly.exterior.xy
            rings.append((list(xs), list(ys)))
        out.append((label, rings))
    return out


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
                     tract_value=None, tract_alpha=0.0,
                     selected_region_mask=None):
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
        # Relay 38 — conversions always show through: zero the overlay alpha on
        # changed pixels so the orange tints only unchanged land. `changed` is
        # already at plot resolution (the LULC rasters were downsampled above),
        # so it aligns with alpha_f without a second downsample.
        alpha_f[changed] = 0.0
        overlay_rgba[..., 3] = (alpha_f * 255).astype(np.uint8)
        ax.imshow(overlay_rgba)
        legend_handles.append(Patch(facecolor=(1.0, 140/255, 0.0, 0.6), label='Urban intensity overlay'))

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

    # Region Selection Phase 1 (Commit 4) — display-only outline of the
    # selected region(s) on the existing map. Read-only highlight; no click
    # handling. Drawn last so it sits above the LULC + heat + tract layers.
    # Color choice: matplotlib default blue (#1f77b4) — doesn't collide with
    # GI/FF green, HD red, or the heat-overlay orange.
    if selected_region_mask is not None:
        region_mask_ds = _downsample_for_plot(
            selected_region_mask.astype(np.uint8), order=0
        ).astype(bool)
        if region_mask_ds.any():
            ax.contour(
                region_mask_ds.astype(np.uint8),
                levels=[0.5],
                colors=['#1f77b4'],
                linewidths=1.8,
            )
            legend_handles.append(
                Patch(facecolor='none', edgecolor='#1f77b4', linewidth=1.8,
                      label="Selected region")
            )
            # Relay 38 — auto-fit the view to the selected region (+~12% pad) so
            # its conversions fill the frame. y inverted for image origin.
            # selected_region_mask=None or empty mask → full extent (unchanged).
            _rows, _cols = np.where(region_mask_ds)
            _rmin, _rmax = int(_rows.min()), int(_rows.max())
            _cmin, _cmax = int(_cols.min()), int(_cols.max())
            _pad = max(2.0, 0.12 * max(_rmax - _rmin, _cmax - _cmin))
            ax.set_xlim(_cmin - _pad, _cmax + _pad)
            ax.set_ylim(_rmax + _pad, _rmin - _pad)

    ax.axis('off')
    # Title removed — section H2 "Where land-cover changes happen" already provides context
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
    # Shorter on-plot / legend display labels (the config keys carry the NLCD
    # code, which overlaps on dense plots). Keyed by the config name so the
    # TEXT_POSITIONS / MARKER_OVERRIDES lookups above still resolve.
    SHORT_REF_LABELS = {
        'Baseline':                   'Baseline',
        'All Food Forest (NLCD 41)':  'All food forest',
        'All Green Infra (NLCD 90)':  'All green infra',
        'All High Density (NLCD 24)': 'All high density',
    }

    for name, ref in REF_SCENARIOS.items():
        text_pos = TEXT_POSITIONS.get(name, 'top right')
        m_override = MARKER_OVERRIDES.get(name, {})
        display = SHORT_REF_LABELS.get(name, name)
        fig.add_trace(go.Scatter(
            x=[ref['flood']], y=[ref['cooling']],
            mode='markers+text' if text_pos else 'markers',
            marker=dict(
                size=m_override.get('size', 13),
                color=m_override.get('color', ref['color']),
                opacity=m_override.get('opacity', 0.6),
                line=m_override.get('line', dict(color='white', width=1)),
            ),
            text=[display] if text_pos else None,
            textposition=text_pos if text_pos else None,
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>{display}</b> (reference benchmark)<br>"
                f"Flood Index: {ref['flood']} | Cooling CC: {ref['cooling']:.4f}"
                "<extra></extra>"
            ),
            name=display,
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

    # Defensive backstop: `optimize_scenario` returns a Pareto-frontier
    # DataFrame on success but a `{'found': False, 'max_*': ...}` dict
    # when no candidate scenarios meet the targets. The dict has
    # len > 0 (it has keys) and would slip past a `len(optimized) > 0`
    # check, then explode at the DataFrame-style `['food_mln_lbs']`
    # access. Skip the overlay unless `optimized` is actually a
    # DataFrame with rows.
    if (isinstance(optimized, pd.DataFrame)
            and len(optimized) > 0
            and 'food_mln_lbs' in optimized.columns):
        opt_sizes = np.clip(food_to_size(optimized['food_mln_lbs'].values, max_food), 6, 18)
        # Relay 60 Part B — error bars are the CALIBRATED estimate range
        # [est + p10, est + p90] (set by _apply_calibrated_ranges). When the
        # active mode has no calibration artifact, the band columns were dropped
        # → no error bars + estimate-only hover. (Region never reaches here.)
        _has_range = ('flood_lower' in optimized.columns
                      and 'hm_lower' in optimized.columns)
        _err_x = _err_y = None
        if _has_range:
            flood_err_minus = (optimized['flood_reduction'] - optimized['flood_lower']).values
            flood_err_plus  = (optimized['flood_upper']     - optimized['flood_reduction']).values
            hm_err_minus    = (optimized['mean_hm']         - optimized['hm_lower']).values
            hm_err_plus     = (optimized['hm_upper']        - optimized['mean_hm']).values
            _err_x = dict(type='data', symmetric=False,
                          array=flood_err_plus, arrayminus=flood_err_minus,
                          color='rgba(255,165,0,0.2)', thickness=1, width=4)
            _err_y = dict(type='data', symmetric=False,
                          array=hm_err_plus, arrayminus=hm_err_minus,
                          color='rgba(255,165,0,0.2)', thickness=1, width=4)

        def _opt_hover(r):
            if _has_range:
                return (
                    f"<b>Suggested scenario</b><br>{r.scenario_name}<br>"
                    f"Flood — fast estimate: {_fmt_sig(r.flood_reduction)}; "
                    f"estimate range: {_fmt_sig(r.flood_lower)}–{_fmt_sig(r.flood_upper)}<br>"
                    f"HMI — fast estimate: {_fmt_sig(r.mean_hm)}; "
                    f"estimate range: {_fmt_sig(r.hm_lower)}–{_fmt_sig(r.hm_upper)}<br>"
                    f"Calibrated from evaluator-comparison errors"
                )
            return (
                f"<b>Suggested scenario</b><br>{r.scenario_name}<br>"
                f"Flood — fast estimate: {_fmt_sig(r.flood_reduction)}<br>"
                f"HMI — fast estimate: {_fmt_sig(r.mean_hm)}"
            )
        fig.add_trace(go.Scatter(
            x=optimized['flood_reduction'],
            y=optimized['mean_hm'],
            mode='markers',
            marker=dict(size=opt_sizes, color='orange', symbol='diamond',
                        line=dict(color='black', width=1.5)),
            error_x=_err_x,
            error_y=_err_y,
            text=optimized.apply(_opt_hover, axis=1),
            hoverinfo='text',
            name='Suggested scenarios',
        ))

    fig.add_trace(go.Scatter(
        x=[results['flood_reduction']],
        y=[results['mean_hm']],
        mode='markers',
        marker=dict(size=20, color='purple', symbol='star',
                    line=dict(color='white', width=1.5)),
        hovertemplate=(
            f"<b>This Scenario</b><br>"
            f"Flood Index: {results['flood_reduction']:.1f}<br>"
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
        xaxis_title='Flood Index (higher = better)',
        yaxis_title='Cooling / Heat Mitigation Index (higher = better)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 0.6]),
        height=600,
        margin=dict(l=60, r=200, t=55, b=60),
        # Legend stays vertical-outside-right rather than flipping to
        # horizontal-below: with the feasible-space hull, four reference
        # benchmarks, the current scenario, and the conditional saved /
        # frontier / optimized traces it can carry 6-9 items — too many to
        # read in a horizontal strip. It already sits in the reserved right
        # margin, not over the data.
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left', yanchor='top',
                    tracegroupgap=4, font=dict(size=11), itemsizing='constant',
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        hovermode='closest',
    )
    fig.add_annotation(
        text="Current scenario shown as the purple star; dotted lines mark its position",
        xref='paper', yref='paper', x=0, y=1.06,
        xanchor='left', yanchor='bottom',
        showarrow=False, font=dict(size=11, color='gray'),
    )
    return fig


def _apply_region_optimizer_mix(row, index):
    """Apply a tested mix from the region optimizer as the current scenario.

    Single source of truth for both the 'best tested mixes' table buttons
    and the click-to-apply on the SELECTED-AREA tradeoff scatter. Rounds
    (pct, gi, ff) to the nearest 5 to align with slider granularity,
    clamps gi+ff to 100, sets the applied_from_region_optimizer
    provenance flag so headers / Save / Export route to
    PROVENANCE_REGION_OPTIMIZED, and stashes the prior scenario state for
    the mis-click revert affordance.

    Caller is responsible for calling st.rerun() — this helper just
    writes session_state."""
    pct = int(round(row.pct_converted / 5) * 5)
    gi = int(round(row.green_infrastructure_pct / 5) * 5)
    ff = int(round(row.food_forest_pct / 5) * 5)
    if gi + ff > 100:
        ff = 100 - gi
    # Stash prior state for the mis-click revert. Read from the current
    # slider session_state keys (where the live UI values live).
    st.session_state['_region_apply_prev'] = {
        'pct': st.session_state.get('slider_pct_converted'),
        'gi':  st.session_state.get('slider_gi_pct'),
        'ff':  st.session_state.get('slider_ff_pct'),
    }
    st.session_state._pending_pct = pct
    st.session_state._pending_gi = gi
    st.session_state._pending_ff = ff
    st.session_state.applied_suggestion = index
    st.session_state.applied_from_region_optimizer = True
    st.session_state._applied_region_optimizer_values = (pct, gi, ff)
    st.session_state.applied_from_optimizer = False
    st.session_state._applied_optimizer_values = None
    st.session_state._show_apply_toast = True


def _render_apply_toast():
    """One-shot, mode-aware confirmation toast after an Apply (Relay 51).

    Citywide machine-learning suggestions, selected-area candidates, and best-by-goal
    applies get handoff text matching their path. Read-only on the apply/
    provenance flags — it does NOT set them (the apply helpers own that); it just
    reads them to choose the message, then clears the one-shot toast flag."""
    if not st.session_state.get("_show_apply_toast"):
        return
    if st.session_state.get("applied_from_optimizer"):
        st.success(
            "Applied the machine-learning suggestion — recomputed with the "
            "InVEST-aligned evaluator. Map, metrics, and comparison table updated."
        )
    elif st.session_state.get("applied_from_region_optimizer"):
        st.success(
            "Applied as the active scenario — under your selected area and "
            "filters. Map and metrics updated."
        )
    else:
        st.success(
            "Applied as the active scenario — recomputed with the "
            "InVEST-aligned evaluator. Map and metrics updated."
        )
    st.session_state._show_apply_toast = False


def _load_guided_example(name, pct, gi, ff, *, placement=None,
                         ownership_preset=None):
    """Load a guided-example recipe (Relay 50). Sets ONLY this recipe's defining
    knobs via the same _pending_* path the mix presets use, plus the eligibility
    section's elf_preset write-through for ownership. NEVER touches region
    selection (selected_region_mask / region_selection) or the placement MODE
    (_filter_active) — a recipe applies citywide or within the user's current
    region. Goals are deliberately not set (a scenario's displayed metrics are
    goal-independent; goals only weight a later Optimize). Caller is a button
    handler; this stages session_state and triggers the rerun."""
    st.session_state._pending_pct = pct
    st.session_state._pending_gi = gi
    st.session_state._pending_ff = ff
    if placement is not None:
        st.session_state._pending_placement = placement
    if ownership_preset is not None:
        # Re-use the eligibility section's preset write-through: set the
        # selectbox value and clear its applied-tracker so the checkboxes land
        # on rerun.
        st.session_state["elf_preset"] = ownership_preset
        st.session_state.pop("_elf_preset_applied", None)
    st.session_state._example_toast = name
    st.rerun()


def plot_tradeoff_region(results, region_optimized_df, baseline_hm_region):
    """Selected-area tradeoff scatter — region-local axes only.

    Plots the current scenario star + engine-verified tested mixes (from
    the region-optimizer's top 5) + the region-local baseline marker, all
    on region-local Flood Index × Cooling/HMI axes. Excludes the
    citywide surrogate diamonds and NatCap reference scenarios — those
    have no region basis (NatCap refs are citywide-only; surrogate is
    trained on citywide grid) so plotting them on region-local axes would
    be apples-to-oranges.

    Region-local pure-allocation refs (All food forest / All green infra /
    All high density at the region) are NOT computed for v1 — they'd need
    separate engine runs at pct=50 with each pure mix under the active
    region∩ownership mask. Omitted per the brief ('If not readily
    available, omit them from the region scatter for v1').

    Returns a plotly figure. The companion 'Citywide context' expander
    below this chart on the dashboard renders the existing plot_tradeoff
    function with all citywide content."""
    fig = go.Figure()
    region_local = results.get('region_local') or {}
    cur_flood = float(region_local.get('flood_reduction') or 0.0)
    cur_hm = float(region_local.get('mean_hm') or 0.0)

    # Region-local baseline marker — flood_reduction = 0 by definition (no
    # change from baseline); mean_hm is the per-pixel baseline HMI averaged
    # inside the active mask.
    if baseline_hm_region is not None:
        fig.add_trace(go.Scatter(
            x=[0.0], y=[float(baseline_hm_region)],
            mode='markers+text',
            marker=dict(size=16, color='steelblue', opacity=1.0,
                        line=dict(color='black', width=2)),
            text=['Region baseline'],
            textposition='top right',
            textfont=dict(size=10),
            hovertemplate=(
                "<b>Region baseline</b><br>"
                f"Flood Index: 0.0 (no conversion)<br>"
                f"Cooling HMI: {float(baseline_hm_region):.4f}"
                "<extra></extra>"
            ),
            name='Region baseline',
        ))

    # Engine-verified tested mixes — already region-local because the
    # region-optimizer's _engine_eval ran evaluate_scenario with the active
    # mask. Plotted as orange-rimmed squares (distinct from the citywide
    # surrogate diamonds) so the visual encoding tracks 'engine-verified'.
    # Click-to-apply: each marker carries customdata = its 0-based index
    # into region_optimized_df, so the chart click handler can map a
    # click back to the row to apply. Hovertemplate includes "Click to
    # apply this mix" to invite the action.
    if (isinstance(region_optimized_df, pd.DataFrame)
            and len(region_optimized_df) > 0
            and 'flood_reduction' in region_optimized_df.columns
            and 'mean_hm' in region_optimized_df.columns):
        opt = region_optimized_df.reset_index(drop=True).copy()
        opt['rank'] = opt.index + 1
        fig.add_trace(go.Scatter(
            x=opt['flood_reduction'],
            y=opt['mean_hm'],
            mode='markers+text',
            marker=dict(size=14, color='orange', symbol='square',
                        line=dict(color='black', width=1.5)),
            text=opt['rank'].astype(str),
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            customdata=opt.index.tolist(),  # 0-based row index for click handler
            hovertemplate=opt.apply(
                lambda r: (
                    f"<b>Tested mix #{int(r['rank'])}</b> "
                    "(engine-verified, region-local)<br>"
                    f"{int(r.get('pct_converted', 0))}% conv — "
                    f"GI {int(r.get('green_infrastructure_pct', 0))}% / "
                    f"FF {int(r.get('food_forest_pct', 0))}%<br>"
                    f"Flood Index: {r.get('flood_reduction', 0):.1f}<br>"
                    f"Cooling HMI: {r.get('mean_hm', 0):.4f}<br>"
                    "<i>Click to apply this mix</i>"
                ), axis=1,
            ).tolist(),
            hoverinfo='text',
            name='Tested mixes (engine-verified)',
        ))

    # Current scenario star — same purple symbol as the citywide chart so
    # the eye tracks 'this scenario' identically across both views.
    fig.add_trace(go.Scatter(
        x=[cur_flood], y=[cur_hm],
        mode='markers',
        marker=dict(size=20, color='purple', symbol='star',
                    line=dict(color='white', width=1.5)),
        hovertemplate=(
            "<b>This scenario</b> (region-local)<br>"
            f"Flood Index: {cur_flood:.1f}<br>"
            f"Cooling HMI: {cur_hm:.4f}"
            "<extra></extra>"
        ),
        name='This scenario',
    ))

    fig.update_layout(
        title='',
        xaxis_title='Flood Index — region (higher = better)',
        yaxis_title='Cooling / HMI — region (higher = better)',
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        height=500,
        margin=dict(l=60, r=200, t=40, b=60),
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left',
                    yanchor='top', font=dict(size=11),
                    bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        hovermode='closest',
    )
    return fig


# ── Brief B2 (revised, 2026-05-29): per-metric VALIDATION badges ──────────────
# Replaces the previous High/Medium/Prototype confidence tiers. Sourced from
# data/sa/natcap_reference_outputs.csv via `natcap_validation.py`:
#  - green  NatCap published value — fixed reference view; displayed value IS a
#    NatCap published number (read from the CSV directly).
#  - teal   InVEST-validated  — everyday view; the card's model has measured
#    per-pixel parity (model_validation.VALIDATED_MODELS) and is on the
#    validated compute path (carbon stock, not the MN proxy).
#  - blue   InVEST-aligned    — canonical method, parity not measured for this
#    output (lumped Flood Index / Runoff Volume, dollar / derived cards).
#  - gray   Prototype          — exploratory metric, no canonical InVEST analog.
# See docs/internal/DESIGN_NOTES.md §8.1 "Two-surface validation vocabulary — locked" for the non-CSV-card curated map.

_VALIDATION_BADGE_COLOR_HEX = {
    "green": "#1a7f37",   # NatCap published value (fixed reference view)
    "teal":  "#0f766e",   # InVEST-validated (per-pixel parity, Stage-1 source)
    "blue":  "#0969da",   # InVEST-aligned (canonical method, parity not measured here)
    "gray":  "#6e7681",   # Prototype
}


def _render_validation_caption(col, metric_name, scenario_context,
                               explicit_status=None, validated_path=True):
    """Render the per-card validation badge as a colored caption with a
    scenario-aware tooltip. Replaces the previous `_confidence_caption`.

    `metric_name` is the canonical CSV name (e.g. `temp_change_f`) when the
    metric is in `natcap_reference_outputs.csv`; otherwise `explicit_status`
    must be passed for the non-CSV-card curated map (`'natcap_published'` /
    `'aligned_method'` / `'prototype'`).

    `validated_path` gates the InVEST-validated tier on the compute path actually
    being the validated one — pass `_CARBON_IS_STOCK` for carbon (SA four-pool =
    True, MN proxy = False). Defaults True for cards validated on every city.
    """
    badge = nv.render_validation_badge(metric_name, scenario_context,
                                       explicit_status=explicit_status,
                                       validated_path=validated_path)
    if badge["text"] is None:
        col.caption("—")
        return
    color = _VALIDATION_BADGE_COLOR_HEX.get(badge["color"], "#6e7681")
    col.caption(
        f'<span style="color: {color}">{badge["text"]}</span>',
        help=badge["tooltip"], unsafe_allow_html=True,
    )


# Brief #3 (2026-05-29) — scenario-level Source + Validation header. Sits
# above each scenario's metric grid so provenance is impossible to miss.
# Per-metric card badges (above) carry per-metric nuance; this header
# describes the scenario as a whole. Drives off the same PROVENANCE_*
# taxonomy from natcap_scenarios.py (re-exported via eib).
_PROVENANCE_HEADER_INFO = {
    eib.PROVENANCE_BASELINE: (
        "Baseline",
        "prototype evaluator, verified against canonical InVEST where comparable; absolute NatCap citywide "
        "figures not reproduced",
        "blue",
    ),
    eib.PROVENANCE_NATCAP_FIXED: (
        "NatCap published reference",
        "displayed from NatCap output; exact scenario raster / aggregation "
        "not available",
        "green",
    ),
    eib.PROVENANCE_EXPLORER: (
        "Explorer-generated",
        "InVEST-aligned evaluator; verified where comparable; scenario not NatCap-published",
        "blue",
    ),
    # Brief #4 — Applied-from-Optimizer flag is now plumbed through
    # session_state, so the OPTIMIZER provenance only ever fires on a scenario
    # that has been Applied and therefore evaluated by the InVEST-aligned
    # evaluator. The validation line reflects that: "evaluated on apply" rules
    # out the misread that the displayed cards are still surrogate predictions.
    eib.PROVENANCE_OPTIMIZER: (
        # Two-RELAY lock — applied-result Source line. The applied citywide
        # scenario is engine-evaluated on apply (the main panel reruns
        # evaluate_scenario with the applied recipe), so the Source surface
        # frames it precisely: "Citywide machine-learning suggestion —
        # engine-evaluated on apply." Distinct from the region path's
        # "Engine-verified — region-optimized" below (Assertion C in
        # verify_baselines machine-locks the distinction).
        "Citywide machine-learning suggestion — engine-evaluated on apply",
        "engine-validated — exploratory candidate "
        "for further validation",
        "blue",
    ),
    # Region-constrained optimizer (variant B). Distinct from
    # PROVENANCE_OPTIMIZER because the values displayed are engine-true
    # region-local — the surrogate's role stopped at shortlisting. The
    # label calls that out: "Engine-verified" (the data shown is real),
    # "region-optimized" (the search scope was the active region∩ownership
    # filter), with the search-completeness caveat in the validation line.
    eib.PROVENANCE_REGION_OPTIMIZED: (
        "Engine-verified — region-optimized",
        "engine-true region-local values; machine-learning-shortlisted "
        "candidates (shortlist may not be exhaustive)",
        "blue",
    ),
}


def _ownership_source_suffix(results_or_saved) -> str:
    """Return ' · <terse ownership>' when an Explorer scenario carries an
    active ownership_filter, else ''. Uses the OWNERSHIP_MODES `short`
    variant so the visible provenance bar reads e.g.
    ' · school land' (full 'school district land (K-12 public)' stays
    in the audit expander, comparison table, and export bundle).

    Reads `ownership_filter` from a results dict OR a saved-scenario dict
    (both expose the field after Commit 1 — pre-29 saves return None safely
    via .get()). Handles Batch 4 v2's composite dict shape via the shared
    normalizer.
    """
    if not results_or_saved:
        return ""
    norm = _normalize_ownership_filter(results_or_saved.get('ownership_filter'))
    return f" · {norm['short']}" if norm else ""


def _render_scenario_provenance_header(provenance, scenario_label=None,
                                       scenario_id=None,
                                       trailing_caption=None,
                                       source_suffix="",
                                       show_scenario_label=True):
    """Render a prominent Source + Validation header for the active scenario.

    `provenance` is one of `eib.PROVENANCE_*`. `scenario_label` is the
    scenario's human-facing title (rendered as an `##` heading above the
    badge); `scenario_id` is the canonical id where one exists (NatCap
    scenarios) and shown inline in the Source line. `trailing_caption` is an
    optional small caption rendered just below the badge (used by the
    fixed-scenario reference view to keep the "flip to Explorer" hint).
    `source_suffix` augments the Source line text (used by Region Selection
    Phase 1 to render 'Explorer-generated · selected region').
    """
    info = _PROVENANCE_HEADER_INFO.get(
        provenance,
        ("Unknown", "provenance not recorded", "gray"),
    )
    source, validation, color_key = info
    source = f"{source}{source_suffix}"
    color = _VALIDATION_BADGE_COLOR_HEX.get(color_key, "#6e7681")
    id_caption = f" · <code>scenario_id={scenario_id}</code>" if scenario_id else ""
    if scenario_label and show_scenario_label:
        st.markdown(f"## {scenario_label}")
    st.markdown(
        f'<div style="margin: 0.2em 0 1.0em 0; padding: 0.5em 0.75em; '
        f'border-left: 4px solid {color}; background: #f6f8fa; '
        f'color: #24292f; font-size: 0.92em; line-height: 1.4;">'
        f'<strong>Source:</strong> {source}{id_caption}<br/>'
        f'<strong>Validation:</strong> {validation}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if trailing_caption:
        st.caption(trailing_caption)


def _render_natcap_fixed_scenario_view(scenario_id):
    """B2 (revised) Phase 3 — dedicated reference view for a NatCap SA fixed
    scenario. Reads NatCap's published temp/carbon from
    `natcap_reference_outputs.csv`, computes flood via the B1 helper, shows
    explicit 'not available' states for compound-gated metrics. Does NOT
    route through `evaluate_scenario`.

    Flood reconcile (per B2-revised, docs/internal/OPEN_QUESTIONS.md): the prototype's
    flood-on-native-NLCD×tree vs flood-on-compound-reduced baseline gap is
    mostly derivation artifact; NatCap's documented SA finding is flood ≈
    scenario-invariant under design-storm saturation. So the delta pill
    reflects '≈ invariant.'
    """
    spec = ns.SA_NATCAP_FIXED_SCENARIOS.get(scenario_id)
    if spec is None:
        st.error(f"Unknown NatCap scenario_id: {scenario_id!r}")
        return
    ctx_for_metrics = (
        nv.SCENARIO_CONTEXT_BASELINE if scenario_id == "baseline"
        else nv.SCENARIO_CONTEXT_NATCAP_FIXED
    )

    def _fmt_dt(dt, threshold=0.1):
        if abs(dt) < threshold:
            return "No change"
        return f"{abs(dt):.1f}°F warmer" if dt > 0 else f"{abs(dt):.1f}°F cooler"

    # ── Header — NatCap reference-view banner ───────────────────────────
    # Inline banner (not _render_scenario_provenance_header) so the Source
    # + Validation text can be the longer NatCap-specific honesty framing
    # without changing _PROVENANCE_HEADER_INFO globally (the constant
    # still drives shorter strings in the comparison table). scenario_id
    # moves off the headline → audit expander below.
    st.markdown(f"## {spec['label']}")
    _natcap_banner_color = _VALIDATION_BADGE_COLOR_HEX.get("green", "#1a7f37")
    st.markdown(
        f'<div style="margin: 0.2em 0 1.0em 0; padding: 0.5em 0.75em; '
        f'border-left: 4px solid {_natcap_banner_color}; background: #f6f8fa; '
        f'color: #24292f; font-size: 0.92em; line-height: 1.4;">'
        f'<strong>Source:</strong> NatCap published baseline reference<br/>'
        f'<strong>Validation:</strong> prototype evaluator, verified against canonical InVEST where '
        f'inputs allow; NatCap published values shown as references.'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Sidebar source = NatCap reference scenario. "
        "Flip to Explorer for custom scenarios."
    )

    # NatCap fix #4 — anchor-and-launch framing. These are the scenarios
    # NatCap already studied; the Explorer goes beyond them. Honesty
    # guardrails baked in: "validate" = InVEST-aligned-evaluator verification of
    # Explorer scenarios (not reproduction of NatCap's published figures);
    # "optimize" = "best tested mixes," not the global optimum. The
    # framing must not imply the app reproduces or validates NatCap's
    # numbers — only that the Explorer goes beyond NatCap's static
    # reference set with its own engine.
    st.markdown(
        "**These are the NatCap reference scenarios for San Antonio** — "
        "published project values used as anchors for comparison, not "
        "Explorer-recomputed numbers. The Explorer lets you go beyond "
        "them: explore variations on the sliders, compare against these "
        "anchors, optimize for your selected area (best tested mixes — "
        "not the global optimum), and validate Explorer scenarios with "
        "the InVEST-aligned evaluator. The app does not reproduce or validate "
        "NatCap's published figures themselves."
    )

    with st.expander("Scenario audit", expanded=False):
        st.markdown(
            f"- **scenario_id:** `{scenario_id}`  \n"
            f"- **Provenance constant:** "
            f"`{spec['provenance']}` "
            f"({_PROVENANCE_HEADER_INFO.get(spec['provenance'], ('Unknown',))[0]})  \n"
            f"- **Scope:** NatCap published reference (San Antonio Vibrant "
            f"Land fixed scenario)  \n"
            f"- **Engine:** `evaluate_scenario` is NOT routed; published "
            f"temperature + carbon shown as references; flood is computed "
            f"by the prototype's `flood_reduction_from_nlcd_tree` helper "
            f"on the NatCap-shipped flood raster (if present)."
        )

    # ── Side-by-side (Tradeoffs reorder) ────────────────────────
    # Placed first under the banner so the user lands on the
    # cross-scenario overview before the per-scenario detail. Tradeoff
    # Space plot is intentionally NOT rendered here — the plot's axes
    # (Flood Index, Heat Mitigation Index) don't have published
    # values for NatCap fixed scenarios beyond baseline.
    st.markdown("#### NatCap reference scenarios — side by side")
    st.caption(
        "All values from NatCap's published scenario outputs "
        "(`nootenboom_results/citywide_results_UPDATED.xlsx` → "
        "`natcap_reference_outputs.csv`). Flood is excluded because the "
        "Explorer and NatCap flood derivations differ."
    )
    # Columns drop the "change" suffix per spec — Temperature / Carbon
    # stock / Carbon value. Baseline row shows absolute published values;
    # alternative rows show +Δ from baseline.
    _comp_rows = []
    for _sid in ns.SA_NATCAP_FIXED_SCENARIOS.keys():
        _, _bv_t_s, _dT_s = nv.published_delta(selected_city, _sid, "temp_change_f")
        _, _bv_c_s, _dC_s = nv.published_delta(selected_city, _sid, "carbon_tons_co2")
        _label = ns.SA_NATCAP_FIXED_SCENARIOS[_sid]["label"]
        if _sid == scenario_id:
            _label = f"▶ {_label}"
        if _sid == "baseline":
            _t_str = f"{_bv_t_s:.2f} °F" if _bv_t_s is not None else "—"
            _c_str = (f"{_bv_c_s / 1e6:.2f}M t CO2e"
                      if _bv_c_s is not None else "—")
            # DataFrame cells render plain — $ is safe unescaped.
            _cv_str = (f"${_bv_c_s * EPA_SOCIAL_COST_CARBON / 1e9:.2f}B"
                       if _bv_c_s is not None else "—")
        else:
            _t_str = (f"{_fmt_dt(_dT_s)} ({_dT_s:+.3f} °F)"
                      if _dT_s is not None else "—")
            _c_str = (f"{_dC_s / 1e6:+.2f}M t CO2e"
                      if _dC_s is not None else "—")
            _cv_str = (f"${_dC_s * EPA_SOCIAL_COST_CARBON / 1e6:+.0f}M"
                       if _dC_s is not None else "—")
        _comp_rows.append({
            "Scenario":    _label,
            "Temperature": _t_str,
            "Carbon stock": _c_str,
            "Carbon value": _cv_str,
        })
    st.dataframe(pd.DataFrame(_comp_rows),
                 width="stretch", hide_index=True)
    st.caption(
        "Baseline shows absolute published values; alternatives show "
        "change from NatCap's baseline. Carbon \\$ is derived as carbon "
        f"Δ × \\${EPA_SOCIAL_COST_CARBON}/t CO2e — not a NatCap-published "
        "dollar value. Flood is excluded because the Explorer and NatCap "
        "flood derivations differ."
    )
    st.divider()

    # ── Compute flood on the loaded scenario raster (B1 helper) ──
    flood_red = mean_cn = None
    flood_source_label = None
    if scenario_id == "baseline":
        lulc_nlcdtree = reduce_compound_to_nlcd_tree(
            cooling_lulc_compound, COMPOUND_TO_NLCD_TREE
        ).astype(np.int16)
        flood_source_label = "prototype baseline (compound → NLCD×tree)"
    else:
        try:
            lulc_nlcdtree, _ld_meta = ns.load_natcap_fixed_scenario(
                scenario_id, "data/sa/flood/land_use_compound_sa.tif"
            )
            flood_source_label = (
                f"NatCap-shipped flood-encoded raster "
                f"(`{os.path.basename(_ld_meta['source_path'])}`)"
            )
        except FileNotFoundError as e:
            st.warning(
                f"Could not load the NatCap scenario raster for "
                f"`{scenario_id}` — source not present on this machine. "
                f"Flood card will be unavailable.\n\n`{e}`"
            )
            lulc_nlcdtree = None
            flood_source_label = "unavailable (source raster missing)"
    if lulc_nlcdtree is not None:
        _soil_clamped = np.clip(soil_resized, 1, 4)
        mean_cn, flood_red = ns.flood_reduction_from_nlcd_tree(
            lulc_nlcdtree, _soil_clamped, cn_table, lucode_idx_arr
        )

    # ── Validation claim (B2 revised Phase 4 — conservative reframe) ──
    # The original Phase-4 ambition was a "we reproduce NatCap's baseline"
    # match panel (prototype absolute vs NatCap absolute + Δ + ✓/✗). The
    # investigation (2026-05-29) established that NatCap's published citywide
    # absolutes — `avg_temp_f` = 90.08°F and `c_sequestration` = 107.32M
    # t CO2e — are NOT reproducible from the disk content: their UCM args
    # aren't shipped, and the carbon aggregation script behind the spreadsheet
    # isn't either. Per the conservative-floor directive: no prototype-vs-
    # NatCap absolute side-by-side (even captioned, it reads as a miss and
    # they aren't comparable quantities). One plain line instead.
    if scenario_id == "baseline":
        st.markdown("#### Validation claim")
        st.markdown(
            "**Validated:** per-pixel parity vs canonical InVEST "
            "(HMI MAE 0.0000, Brief 28b; UMH MAE ≈ 0, Brief B).  \n"
            "**Not established:** reproduction of NatCap's published "
            "citywide figures — their UCM args and carbon-aggregation script "
            "aren't recoverable from disk. See `docs/internal/OPEN_QUESTIONS.md`."
        )
        st.divider()

    # ── Card row — Ecological ──
    # Card-label truncation fix: titles drop the "(baseline)" suffix and
    # use compact names that fit in the metric chrome (Mean Air Temp,
    # Carbon Stock, Carbon Value, Flood Index). The baseline-vs-Δ
    # distinction lives in the metric value (absolute on baseline rows;
    # signed Δ on alternative rows), never in the title or badge string.
    # The validation badge underneath each card is one of the locked four
    # (NatCap published value / InVEST-validated / InVEST-aligned /
    # Prototype) — render_validation_badge returns the locked text, and
    # the "baseline" framing never appears in the badge.
    st.markdown("#### Ecological")
    # NatCap fix #2 — card truncation. 4 cards in one row at sidebar-typical
    # widths cut off the value + delta strings ("No ch…", "+0.43…", "↑ Δ vs
    # baseline: -…"). Split into 2x2 — row 1 carries the NatCap-published
    # values (green badges); row 2 carries derived/computed values.
    # Each card now has ~50% width, enough for the value + delta to read.
    eco_a, eco_b = st.columns(2)

    sv_t, bv_t, dT = nv.published_delta(selected_city, scenario_id, "temp_change_f")
    if scenario_id == "baseline" and bv_t is not None:
        eco_a.metric("Mean Air Temp", f"{bv_t:.2f}°F",
                     delta=None, delta_color="off")
    elif dT is not None:
        eco_a.metric("Mean Air Temp", _fmt_dt(dT),
                     delta=f"Δ {dT:+.3f} °F vs baseline",
                     delta_color="off")
    else:
        eco_a.metric("Mean Air Temp", "—")
    _render_validation_caption(eco_a, "temp_change_f", ctx_for_metrics)

    sv_c, bv_c, dC = nv.published_delta(selected_city, scenario_id, "carbon_tons_co2")
    if scenario_id == "baseline" and bv_c is not None:
        eco_b.metric("Carbon Stock",
                     f"{bv_c / 1e6:.1f}M t CO2e",
                     delta=None, delta_color="off")
    elif dC is not None:
        _sign = "+" if dC >= 0 else ""
        eco_b.metric("Carbon Stock",
                     f"{_sign}{dC / 1e6:.2f}M t CO2e",
                     delta="Δ vs baseline",
                     delta_color="off")
    else:
        eco_b.metric("Carbon Stock", "—")
    _render_validation_caption(eco_b, "carbon_tons_co2", ctx_for_metrics)

    # Row 2 — derived/computed cards. Carbon Value (InVEST-aligned) +
    # Flood Index (Prototype).
    eco_c, eco_d = st.columns(2)

    if scenario_id == "baseline" and bv_c is not None:
        eco_c.metric("Carbon Value",
                     f"${bv_c * EPA_SOCIAL_COST_CARBON / 1e9:.1f}B",
                     delta=f"@ ${EPA_SOCIAL_COST_CARBON}/t (EPA 2023)",
                     delta_color="off")
    elif dC is not None:
        _usd = dC * EPA_SOCIAL_COST_CARBON
        _sign = "+" if _usd >= 0 else ""
        eco_c.metric("Carbon Value",
                     f"{_sign}${_usd / 1e6:.0f}M",
                     delta=f"@ ${EPA_SOCIAL_COST_CARBON}/t (EPA 2023)",
                     delta_color="off")
    else:
        eco_c.metric("Carbon Value", "—")
    # Carbon Value badge: derived from NatCap-published carbon × EPA SC-CO2
    # — NOT a NatCap-published dollar value. Explicit_status='aligned_method'
    # surfaces as "InVEST-aligned" (the third locked badge). The
    # "derived from NatCap carbon" framing lives in the tooltip below,
    # never in the badge string itself.
    _render_validation_caption(eco_c, "carbon_value_usd", ctx_for_metrics,
                               explicit_status="aligned_method")

    if flood_red is not None:
        eco_d.metric(
            "Flood Index", f"{flood_red:.1f}",
            # NatCap fix #2 — shortened delta. The "design-storm saturation,
            # NatCap finding" detail lives in the help tooltip below, not
            # in the delta-line which truncates aggressively at 2x2 width.
            delta=(None if scenario_id == "baseline"
                   else "≈ invariant"),
            delta_color="off",
            help=(
                "Flood Index — a unitless curve-number-based indicator (100 − mean CN); "
                "not a direct measure of flood volume or damage. Computed by the prototype "
                "on the loaded scenario raster via the canonical SCS-CN "
                "method (B1 flood helper). NatCap's documented SA finding: "
                "under the 24-hour 100-year design storm, soil infiltration "
                "is exceeded and the flood metric is essentially scenario-"
                "invariant — GI's primary SA benefits are heat, nature "
                "access, and carbon, not flood (NatCap, 2023). The literal "
                "scenario − baseline delta is suppressed here because of a "
                "~5-pt CN gap between the native NLCD×tree path and the "
                "prototype's compound-reduced baseline (mostly derivation "
                "artifact; see docs/internal/OPEN_QUESTIONS.md → \"Native NLCD×tree "
                "baseline flood raster\")."
            ),
        )
    else:
        eco_d.metric("Flood Index", "—")
    # Flood badge: NatCap published no SA flood metric (UFRM without
    # damage valuation, no published number); the prototype's flood
    # value is a Prototype computation, not "InVEST-aligned." Override
    # the CSV-derived status to keep the locked vocab honest here.
    _render_validation_caption(eco_d, "flood_reduction", ctx_for_metrics,
                               explicit_status="prototype")

    # ── Metrics not recomputed for NatCap reference scenarios ────────────
    st.divider()
    st.markdown("#### Metrics not recomputed for NatCap reference scenarios")
    st.caption(
        "Some metrics cannot be recomputed for NatCap reference scenarios "
        "because the per-scenario compound LULC rasters weren't available "
        "in the shared data. The app shows NatCap's published temperature "
        "and carbon reference values where available."
    )
    st.markdown(
        "- **Nature Access** (UNA)  \n"
        "- **Mental Health** — preventable cases + avoided costs  \n"
        "- **Cooling Energy Savings**  \n"
        "- **Food Production**  \n"
        "- **NDVI** (land-cover-derived)  \n"
        "- **Implementation Cost** & **Cost-Effectiveness** ratios"
    )
    with st.expander("Why are these unavailable?", expanded=False):
        st.markdown(
            "These cards require the **compound** "
            "(NLCD × NLUD × tree-canopy) scenario inputs, which NatCap "
            "built as unsaved pipeline intermediates "
            "(see `docs/internal/OPEN_QUESTIONS.md` → "
            "\"Per-scenario compound LULC inputs\"). Baseline reproduction "
            "+ display of NatCap's published reference values for "
            "temperature and carbon is intact."
        )

    # ── Source / methodology footer ──
    st.divider()
    st.caption(
        f"**Flood input:** {flood_source_label}.  \n"
        f"**Temperature & carbon:** NatCap's published values from "
        f"`nootenboom_results/citywide_results_UPDATED.xlsx`, surfaced via "
        f"`data/sa/natcap_reference_outputs.csv`.  \n"
        f"**Validation state per metric:** see `docs/internal/NATCAP_ALIGNMENT.md`."
    )
    st.caption(
        "**Validation status (summary).** Engine: verified against canonical "
        "InVEST where inputs allow. NatCap project values: shown as published "
        "references. Exact scenario reproduction: unavailable (NatCap's "
        "scenario rasters / aggregation scripts / args weren't shipped)."
    )


# ── Sidebar: scenario-source selector (Brief B2 revised, Phase 3) ─────────────
# SA-only. When the user picks a NatCap fixed scenario, the main panel routes
# to a dedicated reference view (populated from `natcap_reference_outputs.csv`
# + the B1 flood helper); the Explorer sidebar controls below are skipped via
# `st.stop()` so the sidebar shows only the source selector + scenario picker.
st.session_state.setdefault("scenario_source", "Explorer")
st.session_state.setdefault("natcap_fixed_scenario_id", "baseline")

if selected_city.startswith("San Antonio"):
    # NatCap fix #3 — relabel "project" → "reference" everywhere user-
    # facing. The radio's STORAGE VALUE stays 'NatCap project scenario'
    # (session_state key 'scenario_source' carries it; the mode-switch
    # check below compares against it) so a saved session with the old
    # value still wires through; format_func renders the new "reference"
    # label without orphaning state.
    _SCENARIO_SOURCE_LABELS = {
        "Explorer": "Explorer",
        "NatCap project scenario": "NatCap reference scenario",
    }
    _src = st.sidebar.radio(
        "Scenario source",
        options=["Explorer", "NatCap project scenario"],
        format_func=lambda v: _SCENARIO_SOURCE_LABELS[v],
        key="scenario_source",
        help=(
            "Explorer: build a custom scenario with the sliders below. "
            "NatCap reference scenarios are published project values used "
            "as anchors for comparison — not recomputed Explorer scenarios."
        ),
    )
    if _src == "NatCap project scenario":
        _fixed_ids = list(ns.SA_NATCAP_FIXED_SCENARIOS.keys())
        _fixed_labels = {sid: ns.SA_NATCAP_FIXED_SCENARIOS[sid]["label"]
                         for sid in _fixed_ids}
        _picked = st.sidebar.selectbox(
            "NatCap reference scenario",
            options=_fixed_ids,
            format_func=lambda sid: _fixed_labels[sid],
            key="natcap_fixed_scenario_id",
        )
        _render_natcap_fixed_scenario_view(_picked)
        st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
# Sidebar Reorg — five collapsible sections: Scenario / Spatial targeting /
# Eligibility filters / Discover scenarios / Export. The City selector at the
# top and the SA scenario-source picker (above) stay outside the expanders;
# they're app-level picks, not scenario controls.
#
# Seed slider defaults via session_state (not via widget `value=` kwarg) so
# the city-change reset above composes cleanly and Streamlit does not warn
# about a key being set both via the widget default and the Session State API.
st.session_state.setdefault("slider_pct_converted",
                            SCENARIO_DEFAULT_PCT_CONVERTED)
st.session_state.setdefault("slider_gi_pct",
                            SCENARIO_DEFAULT_GI_PCT)
st.session_state.setdefault("slider_ff_pct",
                            SCENARIO_DEFAULT_FF_PCT)

# ── Sidebar visual order ───────────────────────────────────────────────────
# Pre-create the section expanders in VISUAL (workflow) order — the user reads
# the sidebar top-to-bottom as build → target → filter → search → tune →
# export:
#   Scenario (Build) → Quick Start → Choose area → Eligible land →
#   Placement Strategy → Search goals → Cost assumptions →
#   Advanced: model quality → Export / handoff.
# Quick Start sits at position 2 (next to Build) because presets are the fast
# way to *build* a scenario — onboarding, not an afterthought (this reverses
# the earlier "Quick Start at the bottom" rationale). Carbon valuation is no
# longer a standalone section: it's folded into Cost assumptions as an MN-only
# subsection, so this is nine sections, not ten.
#
# The expander DEFINITION order below sets the visual order (Streamlit renders
# each container where it's defined). The `with _sec_*:` population blocks are
# UNMOVED and run in DEPENDENCY order (Scenario → Quick Start → Placement →
# Cost → Where → Eligibility → Search goals → Advanced → Export), so the Search
# goals block still reads the cost-slider / region-mask / ownership-mask state
# the populate-earlier blocks set in the same rerun — no one-rerun lag, and the
# Assertion-B button/mode-label pairings stay intact.
_where_expanded = (
    st.session_state.get('selected_region_mask') is not None
)
_eligibility_available = _CURRENT_CITY_STATE.ownership_raster is not None
_carbon_rates_available = not _CARBON_IS_STOCK
_sec_scenario          = st.sidebar.expander("Scenario", expanded=True)
# Quick Start promoted next to Build — presets are the fast way to build.
_sec_quick_start       = st.sidebar.expander("Quick Start", expanded=False)
_sec_where             = st.sidebar.expander("Choose area",
                                              expanded=_where_expanded)
_sec_eligibility       = (
    st.sidebar.expander("Eligible land", expanded=False)
    if _eligibility_available else None
)
_sec_placement         = st.sidebar.expander("Placement Strategy",
                                              expanded=False)
# The capstone "find better options", in the search slot after the setup
# sections. Calm default: collapsed like the others — only Scenario opens.
_sec_discover          = st.sidebar.expander("Search goals",
                                              expanded=False)
_sec_costs             = st.sidebar.expander(
    "Cost assumptions", expanded=False,
)
# Power-user detail, demoted beneath the search/tune sections.
_sec_advanced_quality  = st.sidebar.expander("Advanced: model quality",
                                              expanded=False)
_sec_export            = st.sidebar.expander("Export / handoff", expanded=False)

# ── Sidebar section: Scenario ──────────────────────────────────────────────
# Base scenario controls — conversion mix, presets, placement strategy,
# implementation costs (folded in from the prior standalone expander), and
# carbon-rate sliders (MN only — folded in from the prior Advanced Settings
# expander). Expanded by default since this is the primary configuration
# surface.
with _sec_scenario:
    pct_converted = st.slider(
        "% of developed land to convert", 0, 50,
        key="slider_pct_converted",
        help="Note: real conversions depend on land availability and existing uses — not all developed land is freely convertible."
    )

    st.subheader("Conversion Mix")
    st.caption(
        "Allocate converted land across three uses — must sum to 100%. "
        "High Density auto-fills as the remainder, but it can also be explicitly adjusted."
    )

    green_infrastructure_pct = st.number_input(
        "Green Infrastructure %", 0, 100,
        step=5, key="slider_gi_pct",
        help="Share of converted land allocated to green infrastructure (woody wetlands, NLCD 90)."
    )
    food_forest_pct = st.number_input(
        "Food Forest %", 0, 100,
        step=5, key="slider_ff_pct",
        help="Share of converted land allocated to food forest (deciduous forest, NLCD 41)."
    )

    auto_hd = 100 - green_infrastructure_pct - food_forest_pct
    pct_highdensity = st.number_input(
        "High Density %", 0, 100,
        value=max(0, auto_hd),
        step=5,
        help="Share of converted land allocated to high-density development (NLCD 24). Auto-fills as remainder."
    )

    mix_sum = green_infrastructure_pct + food_forest_pct + pct_highdensity

    st.caption(
        "Default view illustrates a balanced 50/50 mix at 10% conversion. "
        "Adjust the sliders or use a Quick Start preset to explore alternatives."
    )

    if mix_sum == 100:
        st.success("Mix sums to 100%")
    else:
        st.error(f"Mix sums to {mix_sum}% — must equal 100%")
        st.stop()

    # ── Resolved scenario state (Relay A) ────────────────────────────────
    # Single source of truth for the banner title, the page-root Active-scenario
    # block, the audit-expander sentence, and the metric-card label flow. The
    # display helpers (_explorer_scenario_label / _active_scenario_line1
    # / _explorer_audit_sentence) read this dict — they do not interpolate
    # the raw module-level variables. HD is canonical 100 - GI - FF here
    # (the slider isn't keyed and can drift; this fixes the source).
    _resolved_scenario = _resolve_scenario(
        pct_converted, green_infrastructure_pct, food_forest_pct,
    )
    pct_highdensity = _resolved_scenario['pct_highdensity']
# End of Scenario expander — Quick Start / Placement / Costs / Carbon rates
# move to their own sibling expanders below.

# ── Sidebar section: Quick Start ───────────────────────────────────────────
# Extracted out of Scenario per "un-bury the optimizer" so Scenario stays
# compact (% + Conversion Mix only) and the Discover CTA sits directly
# under it. Quick Start preset buttons set `_pending_pct/gi/ff` and rerun
# — the resolved-state aliases at the top of the script pick those up
# before slider widgets render.
with _sec_quick_start:
    st.caption("Click any button to load a preset scenario instantly.")

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

    if st.button("Balanced",
                 type="primary" if _active == 'balanced' else "secondary"):
        st.session_state._pending_pct = 10
        st.session_state._pending_gi = 50
        st.session_state._pending_ff = 50
        st.session_state.active_example_scenario = 'balanced'
        st.rerun()
    st.caption("Default view — 50/50 nature-based mix")

    if st.button("Green Infrastructure",
                 type="primary" if _active == 'green_infra' else "secondary"):
        st.session_state._pending_pct = 10
        st.session_state._pending_gi = 100
        st.session_state._pending_ff = 0
        st.session_state.active_example_scenario = 'green_infra'
        st.rerun()
    st.caption("Runoff reduction focus")

    if st.button("Food Forest",
                 type="primary" if _active == 'food_forest' else "secondary"):
        st.session_state._pending_pct = 10
        st.session_state._pending_gi = 0
        st.session_state._pending_ff = 100
        st.session_state.active_example_scenario = 'food_forest'
        st.rerun()
    st.caption("Cooling + food production focus")

    if st.button("High Density",
                 type="primary" if _active == 'high_density' else "secondary"):
        st.session_state._pending_pct = 10
        st.session_state._pending_gi = 0
        st.session_state._pending_ff = 0
        st.session_state.active_example_scenario = 'high_density'
        st.rerun()
    st.caption("Control case — no green conversion")

    # ── Guided examples (Relay 50) ──────────────────────────────────────────
    # Portable recipes: each sets ONLY its defining knobs (mix, and where noted a
    # placement strategy or land filter) via the same _pending_* path the mix
    # presets use, plus the existing elf_preset write-through for ownership.
    # Region selection and placement MODE are NEVER touched — a recipe applies
    # citywide if no region is selected, or within the user's selected region if
    # one is. Goals are deliberately not set (a scenario's displayed metrics are
    # goal-independent; goals only weight a later Optimize).
    st.markdown("---")
    st.markdown("**Guided examples**")
    st.caption(
        "Starting points — each sets a coherent mix (and, where noted, a "
        "placement strategy or land filter), within your current region or "
        "citywide selection. Adjust the sliders or run Optimize from here."
    )

    if st.button("Balanced", key="guided_balanced"):
        _load_guided_example("Balanced", 10, 50, 50)
    st.caption("50/50 nature-based mix.")
    if st.button("Cooling-focused", key="guided_cooling"):
        _load_guided_example("Cooling-focused", 10, 80, 20,
                             placement="cooling-focused")
    st.caption("Green/tree-leaning mix, placed toward hot areas near buildings.")
    if st.button("Food forest", key="guided_food"):
        _load_guided_example("Food forest", 10, 0, 100)
    st.caption("Food-forest-leaning mix.")
    if st.button("School-land greening", key="guided_school"):
        _load_guided_example("School-land greening", 10, 50, 50,
                             ownership_preset="School land")
    st.caption("Nature mix limited to school land.")

    if st.session_state.get("_example_toast"):
        st.toast(
            f"Loaded the {st.session_state.pop('_example_toast')} example. "
            "Adjust sliders or run Optimize to explore alternatives."
        )

# ── Sidebar section: Placement Strategy ────────────────────────────────────
# Extracted out of Scenario. Placement shapes the current scenario; users
# configure it alongside the conversion mix, then optionally optimize.
with _sec_placement:
    placement_strategy = st.radio(
        "Which pixels get converted",
        options=list(PLACEMENT_STRATEGY_LABELS.keys()),
        format_func=lambda key: PLACEMENT_STRATEGY_LABELS[key],
        index=0,
        key="placement_strategy_radio",  # Relay 50 — settable by guided examples
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

# ── Sidebar section: Implementation Costs (\$/acre) ────────────────────────
# Extracted out of Scenario. Must populate BEFORE the Discover expander so
# the region-optimizer's _fire_region_optimize closure captures live cost
# values at click time.
with _sec_costs:
    cost_gi = st.slider(
        "Green Infrastructure (\\$/acre)", 5_000, 150_000,
        DEFAULT_COST_GI, 5_000,
        help="Typical range: \\$20,000–\\$100,000/acre for constructed wetlands. Default is an illustrative estimate — adjust to reflect local project costs.",
    )
    cost_ff = st.slider(
        "Food Forest (\\$/acre)", 1_000, 50_000,
        DEFAULT_COST_FF, 1_000,
        help="Typical range: \\$5,000–\\$20,000/acre for food forest establishment. Default is an illustrative estimate — adjust to reflect local project costs.",
    )
    cost_hd = st.slider(
        "High Density Infill (\\$/acre)", 1_000, 50_000,
        DEFAULT_COST_HD, 1_000,
        help="Marginal cost of additional impervious development. Default is an illustrative estimate — adjust to reflect local project costs.",
    )

    # ── Carbon valuation (MN only) — folded in from the former standalone
    # "Carbon rates" expander. SA's Carbon uses NatCap's four-pool stock table
    # directly (no per-pool override exposed), so this subsection renders only
    # for MN; SA seeds the same session_state defaults outside any expander
    # below so downstream `st.session_state.carbon_rate_*` reads still work.
    # A small divider keeps the two groups ($/acre costs vs carbon $ valuation)
    # visually distinct.
    if _carbon_rates_available:
        st.markdown("**Carbon valuation**")
        st.slider(
            "Food Forest carbon rate (t CO2e/acre/yr)",
            0.5, 18.0, 3.5, 0.5,
            key="carbon_rate_ff",
            help="Provisional range 1.76–18.2 (USDA NRCS 2022). Default 3.5 is conservative for a mature system.",
        )
        st.slider(
            "Green Infrastructure carbon rate (t CO2e/acre/yr)",
            0.5, 5.0, 2.0, 0.5,
            key="carbon_rate_gi",
            help="Provisional range for woody wetlands. Default 2.0 t CO2e/acre/yr.",
        )
        st.caption(
            "These are provisional regional estimates. Adjust to reflect locally calibrated "
            "values or sensitivity test assumptions. See Methodology & Data Sources for "
            "sources and caveats."
        )

# SA (and any stock-carbon city) seeds carbon-rate defaults outside any expander
# — no MN carbon subsection renders — so downstream reads still resolve.
if not _carbon_rates_available:
    st.session_state.setdefault("carbon_rate_ff", 3.5)
    st.session_state.setdefault("carbon_rate_gi", 2.0)

# ── Interactive Region Map: sync clicks → sidebar multiselect (top-of-script) ─
# Reads `region_map_picker_event` (stashed by tab3) and TOGGLES the clicked
# district(s) into the multiselect's session_state slot BEFORE the sidebar
# reads it. Tab3 runs after the sidebar, so a naive write would land one
# rerun late; tab3's signature-de-duped st.rerun() forces this handler to
# fire on the very next rerun — sidebar multiselect, mask, scenario
# sentence, metric cards, and the optimizer guard all reflect the click
# without a second interaction.
#
# Multi-select RELAY: replaces the prior 'overwrite with picked_ids' write
# (which made each click replace the selection — a bug because the
# sidebar dropdown already supports multi). Plotly's selection_mode=
# 'points' WITHOUT modifier forwarding gives the clicked district id in
# event.selection.points[0].customdata; we toggle that id against the
# current selection. Source of truth is `region_labels_<layer>` —
# multiselect and map both read/write it.
_picker_event = st.session_state.get("region_map_picker_event")
if _picker_event is not None:
    _clicked_ids = [
        p.get("customdata") for p in
        (_picker_event.get("selection") or {}).get("points", [])
        if p.get("customdata")
    ]
    _picker_layer = st.session_state.get("region_map_picker_layer")
    if _picker_layer is not None and _clicked_ids:
        _ms_key = f"region_labels_{_picker_layer}"
        _current = list(st.session_state.get(_ms_key, []) or [])
        for _id in _clicked_ids:
            _current = toggle_selection(_current, _id)
        # Multiselect-keyed session_state must be written BEFORE the widget
        # renders (Streamlit raises if you mutate a widget-keyed value
        # after instantiation on the same run). Sorting keeps the
        # rendered list stable.
        if sorted(_current) != sorted(st.session_state.get(_ms_key, []) or []):
            st.session_state[_ms_key] = sorted(_current)
    # Consume the event so the next rerun starts clean.
    st.session_state["region_map_picker_event"] = None

# ── Sidebar section: Spatial targeting (Sidebar Reorg) ────────────────────
# Placed between Scenario and Eligibility filters so a planner picks
# WHERE conversions can land (region) → narrows by WHAT KIND (ownership)
# → optionally searches for promising mixes (Discover).
_region_layers_available = bool(_CURRENT_CITY_STATE.region_rasters)
# Relay B: Where-changes-happen default-expand by mode. Citywide (no
# region mask) → collapsed; region active → expanded so the user can see
# the current selection at a glance. NatCap mode is short-circuited
# earlier via st.stop(), so the expander never renders in that mode.
_where_expanded = (
    st.session_state.get('selected_region_mask') is not None
)
with _sec_where:
    _apply_within = st.radio(
        "Apply changes within",
        options=["Entire analysis area", "Selected regions"],
        index=0,
        help=(
            "Constrain conversions to inside selected polygons (council districts "
            "or census tracts), instead of citywide. The per-pixel InVEST-aligned evaluator is the "
            "same validated math; the where is planner-chosen. Conversions exclude "
            "roads, buildings, and existing nature; the Selected-region impact "
            "table compares in-region vs citywide."
        ),
        disabled=not _region_layers_available,
        key="region_apply_within",
    )

    # Default: clear any active region state. selected_region_layer is
    # re-set below when the user enters 'Selected regions' mode (regardless
    # of polygon selection) so the Map View tab can render the base layer.
    # selected_region_mask + selected_region_ids stay None until at least
    # one polygon is picked — that's what the engine reads.
    st.session_state['selected_region_mask']  = None
    st.session_state['selected_region_layer'] = None
    st.session_state['selected_region_ids']   = None
    # Ownership Integration Commit 1 — default reset; the Commit 2 UI block will
    # set these conditionally when the user picks a non-default ownership filter.
    # Until that lands, both stay None and the citywide path is byte-identical.
    st.session_state['selected_ownership_mask'] = None
    st.session_state['selected_ownership_mode'] = None

    if _apply_within == "Selected regions" and _region_layers_available:
        _layer_keys = list(_CURRENT_CITY_STATE.region_rasters.keys())
        _layer_key = st.selectbox(
            "Region layer",
            options=_layer_keys,
            format_func=lambda k: _CURRENT_CITY_STATE.region_layer_display_names.get(k, k),
            index=0,
            key="region_layer",
            help="Pick a polygon layer to select from.",
        )
        # Surface the active layer key as soon as the user enters 'Selected
        # regions' mode, even before any polygons are picked. This lets the
        # Map View tab render the base map (district boundaries, no
        # conversions) instead of nothing — a bare opacity slider with no
        # canvas above it is worse than seeing the choices to click. Mask +
        # ids stay None until labels are picked (so the engine still runs
        # citywide until the user actually selects).
        st.session_state['selected_region_layer'] = _layer_key
        _labels = _CURRENT_CITY_STATE.region_layer_labels[_layer_key]
        _display = _CURRENT_CITY_STATE.region_layer_display_names[_layer_key]
        _selected_labels = st.multiselect(
            f"{_display}s",
            options=_labels,
            default=[],
            key=f"region_labels_{_layer_key}",
            help=f"Select one or more {_display.lower()}s to constrain placement.",
        )
        if _selected_labels:
            # Build mask via positional indices (the locked contract: the raster
            # carries positional indices internally; label values are what the
            # user, the metadata, and session_state see).
            _pos_indices = [_labels.index(lbl) for lbl in _selected_labels]
            _raster = _CURRENT_CITY_STATE.region_rasters[_layer_key]
            _region_mask = np.isin(_raster, _pos_indices)
            # Live placement denominator — convertible ∩ region. The spec
            # specifies eligible_pixels_in_region (convertible subset), NOT
            # selected_area_acres (total region polygon area).
            _cp = _CURRENT_CITY_STATE.convertible_pixels
            _eligible_count = int(_region_mask[_cp[:, 0], _cp[:, 1]].sum())
            _eligible_acres = _eligible_count * PIXEL_AREA_ACRES
            _plural = "s" if len(_selected_labels) > 1 else ""
            if _eligible_count == 0:
                # Region Selection Phase 1 (Commit 6) — zero-convertible edge case.
                # The region has no convertible pixels (e.g. all buildings + roads
                # + existing nature inside the selected polygon). n_convert would
                # be 0 → no conversion → metrics == baseline; surface that
                # explicitly so the user understands the dashboard isn't broken.
                st.warning(
                    f"No convertible pixels inside the selected "
                    f"{_display.lower()}{_plural} — conversions can't land here. "
                    f"Try a larger region, a different layer, or clear the "
                    f"selection."
                )
            else:
                st.caption(
                    f"**Eligible for placement:** {_eligible_count:,} pixels "
                    f"(~{_eligible_acres:,.0f} acres) inside the selected "
                    f"{_display.lower()}{_plural}."
                )
                # One-line capability + the load-bearing caveat stays visible;
                # exclusion mechanics + the in-region-vs-citywide table detail
                # moved into the "Apply changes within" radio help above.
                st.caption(
                    "Conversions stay inside the selected area. "
                    "**Main cards remain citywide.**"
                )
            # Push to session_state for Commit 2's lookup bypass and Commit 1's
            # caller-stamping of results['region_selection']. Note: even with
            # _eligible_count == 0 we still set the mask — evaluate_scenario
            # degrades gracefully (n_convert == 0 → no-op conversion → metrics
            # equal baseline), and the provenance label still augments correctly.
            st.session_state['selected_region_mask']  = _region_mask
            st.session_state['selected_region_layer'] = _layer_key
            st.session_state['selected_region_ids']   = list(_selected_labels)  # label values, not positional
        else:
            st.caption(f"Pick one or more {_display.lower()}s above.")
    elif _apply_within == "Selected regions" and not _region_layers_available:
        st.info("No region layers configured for this city.")

# ── Sidebar section: Where conversions can be placed (Sidebar Reorg) ───────
# Batch 4 of OWNERSHIP_FINER_CLASSES_SPEC.md. The panel reads as "where can
# conversions land": exclusion items (always-on) at the top, then the
# selectable ownership class + vacant overlay below. Composes with the
# Where-changes-happen region selection: the combined eligible denominator
# reflects `region ∩ ownership ∩ convertible` when both are active. SA-only
# (MN has no ownership raster — the expander is hidden for that city).
# _eligibility_available was computed at the top of the sidebar for
# pre-creation of _sec_eligibility; reused here as the populate-or-skip
# guard. Alias preserved for any downstream readers below the sidebar.
_ownership_available = _eligibility_available
if _eligibility_available:
    with _sec_eligibility:
        # Always-on exclusions — display only; the per-pixel engine enforces
        # these via the convertible-pixel pool (developed minus
        # buildings/roads), so they aren't selectable.
        st.markdown(
            "**Conversions can never be placed on:**\n"
            "- Building footprints (always excluded)\n"
            "- Roads (always excluded)\n"
            "- Existing natural land (always excluded)"
        )
        # Locked planning-screen caveat — Relay 2 #2 short version stays
        # visible. The longer derivation detail moves under the
        # "How ownership classes are derived" expander below.
        st.caption(
            "Ownership filters are planning-screen constraints. They limit "
            "where conversions can be placed but do not verify legal "
            "availability."
        )
        # Caveat specific to school + university classes (split-preset RELAY).
        # Surfaces the title-verification caveat at the panel level so the
        # caveat is visible without opening the derivation popover.
        st.caption(
            "School and university classes are planning-screen filters, "
            "not title-verified ownership."
        )
        # KNOWN_DIVERGENCES honesty surface — same caveat as the export
        # bundle's metadata.json (entry id `ownership_rule_derived`,
        # asserted complete by verify_baselines). Tucked into a popover
        # (not an expander — Streamlit disallows nested expanders, and
        # this block already lives inside _sec_eligibility) so the
        # visible caption above stays short; the full derivation detail
        # (BCAD rules, school/university approximate, public rollup
        # excludes university) is one click away rather than five lines
        # of preamble.
        with st.popover("How ownership classes are derived"):
            st.markdown(
                "Classes are derived from BCAD owner-name and exemption "
                "rules. **School** and **university** classes are "
                "approximate and not title-verified — `School` matches "
                "ISD / SCHOOL DISTRICT name patterns only (charters and "
                "private K-12 schools fall through to private); "
                "`University` spans both public (UT / A&M / Alamo) and "
                "private (Trinity / St. Mary's / OLLU) campuses and is "
                "kept OUT of the Public rollup for that reason.  \n"
                "  \n"
                "- School land is inferred from BCAD owner-name / "
                "exemption patterns and may miss private schools or "
                "include some district-owned non-school parcels.  \n"
                "- College / university land spans public and private "
                "campuses and is likewise name-pattern–inferred, not "
                "title-verified.  \n"
                "  \n"
                "See `data/sa/sa_ownership_2band_30m.tif` provenance in "
                "`docs/internal/DATA_INVENTORY.md`."
            )
        # Preset dropdown (Relay 2 #3). Replaces the 3 narrow quick-set
        # buttons with a single selector — cleaner at sidebar width and
        # keeps the most-used coarse rollups one click away. Custom keeps
        # the per-class checkboxes visible for manual control. The preset
        # writes to the same `elf_check_<cls>` / `elf_check_vacant` session
        # state keys the checkboxes own, so the composite-filter wiring
        # (mask = union of checked classes ∩ vacant overlay; single-class
        # collapses to OWNERSHIP_MODES key; multi-class persists as
        # composite dict) is unchanged — the dropdown is a way to SET the
        # checkboxes, not a parallel filter source.
        _elf_finer = ('city', 'county', 'state_federal', 'school', 'university')

        _PRESET_OPTIONS = [
            "None",
            "Public land",
            "Vacant land",
            "Vacant + public",
            "School land",
            "College / university land",
            "Custom",
        ]
        # Default to None on first render; the preset selectbox isn't
        # session_state-keyed because the checkboxes ARE the canonical
        # state — the preset writes through to them and then renders the
        # checkbox values.
        st.markdown("**Preset:**")
        _elf_preset = st.selectbox(
            "Ownership preset", options=_PRESET_OPTIONS, index=0,
            label_visibility="collapsed",
            help=(
                "Quick-set the checkboxes below. Public rollup = "
                "city + county + state-federal + school (university is "
                "kept out — mixed public + private). Vacant overlays the "
                "vacancy filter. Pick 'Custom' to set the checkboxes "
                "individually."
            ),
            key="elf_preset",
        )
        # Apply the preset's check pattern by writing the checkboxes'
        # session_state. The checkboxes render below and pick up the new
        # values. Tracking the prior preset lets us write-through only on
        # change so manual checkbox edits aren't clobbered every rerun.
        _prev_preset = st.session_state.get("_elf_preset_applied")
        if _elf_preset != _prev_preset and _elf_preset != "Custom":
            if _elf_preset == "None":
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = False
                st.session_state["elf_check_vacant"] = False
            elif _elf_preset == "Public land":
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = (
                        _cls != "university"
                    )
                st.session_state["elf_check_vacant"] = False
            elif _elf_preset == "Vacant land":
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = False
                st.session_state["elf_check_vacant"] = True
            elif _elf_preset == "Vacant + public":
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = (
                        _cls != "university"
                    )
                st.session_state["elf_check_vacant"] = True
            elif _elf_preset == "School land":
                # Single-class preset — school only, no vacant overlay.
                # The two classes are intentionally separately presetable
                # (no combined "School + university" preset); to combine
                # them, pick Custom and check both checkboxes.
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = (_cls == "school")
                st.session_state["elf_check_vacant"] = False
            elif _elf_preset == "College / university land":
                # Single-class preset — university only, no vacant overlay.
                for _cls in _elf_finer:
                    st.session_state[f"elf_check_{_cls}"] = (_cls == "university")
                st.session_state["elf_check_vacant"] = False
            st.session_state["_elf_preset_applied"] = _elf_preset

        # Checkboxes — visible only under "Custom" to keep the panel
        # uncluttered for the coarse rollups. Their session_state still
        # exists across modes (the preset just writes through), so the
        # composite-filter resolver below sees the same values either way.
        _own_classes_checked = []
        if _elf_preset == "Custom":
            st.markdown("**Restrict to (check classes to include):**")
            for _cls in _elf_finer:
                _label = OWNERSHIP_MODES[_cls]['label']
                if st.checkbox(_label, value=False, key=f"elf_check_{_cls}"):
                    _own_classes_checked.append(_cls)
            _vacant_overlay = st.checkbox(
                "Limit to vacant parcels only",
                value=False,
                key="elf_check_vacant",
                help=(
                    "Narrow the selection above to parcels flagged as vacant "
                    "(no improvement value, exempt-keyed; see the vacancy "
                    "methodology in DATA_INVENTORY). Composable with any "
                    "checked class — e.g. School district + vacant = vacant "
                    "ISD parcels."
                ),
            )
        else:
            # Non-Custom: read the session_state values the preset set
            # (or that were left over from a previous Custom session).
            for _cls in _elf_finer:
                if st.session_state.get(f"elf_check_{_cls}", False):
                    _own_classes_checked.append(_cls)
            _vacant_overlay = st.session_state.get("elf_check_vacant", False)

        # Resolve the (checked classes, vacant overlay) UI state. Storage
        # value collapses to a single OWNERSHIP_MODES key when possible
        # (single class with or without overlay), else a composite dict.
        _own_mode, _own_cfg, _own_label, _own_allowed = _resolve_eligible_filter_state(
            _own_classes_checked, _vacant_overlay,
        )
        if _own_mode is not None:
            # `_own_cfg` from the resolver is OWNERSHIP_MODES-compatible whether
            # the mode is a string key or a composite dict — pass it directly,
            # don't re-look-up by key (composite dicts aren't in OWNERSHIP_MODES).
            _own_mask = _build_ownership_mask(
                _CURRENT_CITY_STATE.ownership_raster,
                _CURRENT_CITY_STATE.ownership_vacant_raster,
                _own_cfg,
            )
            # Eligible-under-all-constraints denominator. If the region UI above
            # already set a mask, intersect; otherwise use ownership alone.
            _region_mask_active = st.session_state.get('selected_region_mask')
            _combined_for_eligible = (
                _own_mask & _region_mask_active
                if _region_mask_active is not None else _own_mask
            )
            _cp = _CURRENT_CITY_STATE.convertible_pixels
            _own_eligible_count = int(
                _combined_for_eligible[_cp[:, 0], _cp[:, 1]].sum()
            )
            _own_eligible_acres = _own_eligible_count * PIXEL_AREA_ACRES
            _own_combined_label = (
                "within the selected region(s) AND on " if _region_mask_active is not None
                else "on "
            ) + _own_label.lower()
            if _own_eligible_count == 0:
                st.warning(
                    f"No convertible pixels {_own_combined_label} — conversions "
                    f"can't land here. Try a broader filter or clear the "
                    f"region/ownership selection."
                )
            else:
                st.caption(
                    f"**Eligible for placement:** {_own_eligible_count:,} pixels "
                    f"(~{_own_eligible_acres:,.0f} acres) {_own_combined_label}."
                )
            st.caption(
                "Ownership is derived from parcel records and approximate at "
                "30 m resolution — reliable for large parcels (parks, civic "
                "campuses, big public tracts), pixelated for small lots."
            )
            st.session_state['selected_ownership_mask'] = _own_mask
            st.session_state['selected_ownership_mode'] = _own_mode

# ── Sidebar section: Discover scenarios (Sidebar Reorg) ────────────────────
# The optimizer panel — citywide-surrogate mode (no filter) vs region-
# prefilter + engine-verify mode (filter active). The model-quality controls
# (folded in from the prior Advanced Settings expander) live at the bottom.
# See docs/internal/REGION_OPTIMIZER_SPEC.md §2 for the mode-switch contract.

# Filter-active drives the mode switch between the citywide surrogate
# optimizer (no filter) and the region-prefilter + engine-verify path
# (filter active). docs/internal/REGION_OPTIMIZER_SPEC.md §2.
_filter_active = (
    st.session_state.get('selected_region_mask') is not None
    or st.session_state.get('selected_ownership_mask') is not None
)

# Defaults — defined before the mode branch so the citywide-results render
# block in the Tradeoffs tab (which interpolates `min_flood` /
# `min_cool` / etc. into a caption) never NameErrors if the user activated a
# filter after an earlier citywide Optimize run left `optimized_results`
# behind in session state. The citywide branch below overwrites these.
min_flood = 0
min_cool = _CURRENT_CITY_STATE.baseline_hm
min_cool_f = 0.0
min_food = 0.0
max_runoff = float(BASELINE_RUNOFF_ACRE_FEET)
min_carbon = 0

with _sec_discover:
    if not _filter_active:
        # Two-RELAY lock — citywide mode label + caption + Optimize button
        # co-render in the same block (Assertion B in verify_baselines).
        # Mode label is promoted markdown (visible), not a faint caption.
        st.markdown("**Citywide machine-learning search**")
        st.caption("Fast estimates suggest promising mixes; apply one to recompute with the InVEST-aligned evaluator.")
        with st.popover("How this works"):
            st.markdown(
                "_The fast model is a random forest trained on "
                "evaluator-computed scenarios. It is used for screening, "
                "not final results._  \n"
                "  \n"
                "The model is trained on the prototype's pre-computed "
                "scenario library (~90 full-resolution runs in Fast "
                "mode; more in the higher-quality modes). It explores "
                "combinations of conversion percentage and conversion mix "
                "far faster than the InVEST-aligned evaluator — but each returned "
                "scenario is a **fast estimate**, not a full evaluation. "
                "It targets the Flood Index, cooling, food production, and "
                "carbon; cost and placement strategy are not part of the "
                "model.  \n"
                "  \n"
                "Set the minimum performance each slider below must meet "
                "(or cap runoff); the search returns scenarios that "
                "satisfy all targets at once. Use the controls above to "
                "verify any suggested scenario in detail."
            )

        with st.container(border=True):
            # Group caption — scoped to the UPPER LIMIT only: the slider maxes come
            # from the scenario library's achievable range (+ small headroom), but
            # the floors (0 / −1) and defaults are fixed, so don't claim those.
            st.caption(
                "Each target's upper limit is the best the scenario library "
                "reached, with a little headroom — reachable screening targets, "
                "not physical limits."
            )
            # Flood slider max uses the precomputed grid's actual achievable maximum
            # rather than the theoretical 0–100 ceiling, so the slider range
            # represents reachable targets. Round up to the next 5 for headroom.
            _flood_achievable_max = int(scenario_df['flood_reduction'].max())
            _flood_slider_max = ((_flood_achievable_max + 4) // 5) * 5
            _flood_default = max(0, _flood_slider_max - 10)
            min_flood  = st.slider(
                "Flood Index ≥",
                0, _flood_slider_max, _flood_default, 5,
                help=f"Corresponds to the Flood Index metric card. Baseline is {100 - _CURRENT_CITY_STATE.baseline_cn:.1f}. Higher values mean less runoff — increasing this target will also reduce Runoff Volume in ac-ft.",
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
                "Carbon storage change ≥ (t CO2e)" if _CARBON_IS_STOCK
                else "Carbon sequestration ≥ (t CO2e/yr)"
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

            if lookup_table:
                st.caption(
                    "Slider results use a precomputed lookup table for faster response. "
                    "Suggested scenarios come from the fast machine-learning model. Apply a suggestion to recompute it with the InVEST-aligned evaluator."
                )
            else:
                st.caption(
                    "Slider results are computed live in the current model-quality mode. "
                    "Suggested scenarios come from the fast machine-learning model. Apply a suggestion to recompute it with the InVEST-aligned evaluator."
                )

            # Two-RELAY lock — sidebar "Optimize" trigger. Co-renders with
            # the mode label + caption above in the same st.container
            # (Assertion B in verify_baselines locks the pairing).
            # width="stretch" gives the button full sidebar
            # width so the short "Optimize" verb can't wrap.
            if st.button("Optimize",
                         key="sidebar_citywide_opt_button",
                         width="stretch",
                         help=_OPTIMIZE_HELP_CITYWIDE):
                _fire_citywide_optimize(
                    surrogate, min_flood, min_cool, min_food, max_runoff,
                    min_carbon, MAX_FOOD, MAX_FLOOD, MAX_COOL,
                )
            st.caption("Same search as the Optimize button in Discover scenarios — uses your current goals, region, and filters.")
    else:
        # ── Region-constrained optimizer (variant B) ─────────────────────
        # Mode-switch path. Replaces the min-target sliders + Optimize button
        # with weight sliders + Optimize-selected-area. The fast surrogate
        # shortlists candidate mixes; the full engine evaluates the finalists
        # on the active region∩ownership mask. Displayed values are engine-true
        # region-local — no surrogate predictions surface. See
        # docs/internal/REGION_OPTIMIZER_SPEC.md.
        # Two-RELAY lock — selected-area mode label + caption + Optimize
        # button co-render in the same block (Assertion B). Mode label is
        # promoted markdown (visible), not a faint caption.
        st.markdown("**Selected-area search**")
        st.caption(
            "Searches candidate mixes under the current area and filters. Displayed values are computed by the InVEST-aligned evaluator, not model predictions."
        )
        with st.popover("How this works"):
            st.markdown(
                "When a region or ownership filter is active, the search "
                "runs in two stages. **Stage 1** — the fast machine-learning "
                "model ranks every candidate mix and picks a Pareto-efficient "
                "shortlist (≈ 40 candidates). **Stage 2** — each shortlisted "
                "mix is evaluated by the InVEST-aligned evaluator inside your "
                "selected area. Values shown on each returned scenario are "
                "InVEST-aligned evaluator region-local (not model predictions). To "
                "re-rank under new weights, click Optimize again (v1 reruns "
                "the full pipeline).\n\n"
                "For selected-area optimization, displayed values are computed "
                "by the InVEST-aligned evaluator, not predicted by the machine-learning "
                "model. A validation check confirmed the prefilter "
                "did not miss the best tested mix across the tested selections "
                "and goal weights. This would need rechecking if the full "
                "evaluator becomes more spatially detailed."
            )

        with st.container(border=True):
            st.markdown("**Weight each objective** (0 = ignore, 1 = full weight)")
            st.caption(
                "Weights set how much each objective matters when ranking the "
                "best-tested mixes — they are not performance targets. The flood "
                "weight also covers runoff."
            )
            w_cool = st.slider("Cooling", 0.0, 1.0, 1.0, 0.1, key="region_opt_w_cool")
            w_flood = st.slider("Flood Index", 0.0, 1.0, 1.0, 0.1,
                                key="region_opt_w_flood")
            w_carbon = st.slider("Carbon", 0.0, 1.0, 1.0, 0.1,
                                 key="region_opt_w_carbon")
            w_food = st.slider("Food production", 0.0, 1.0, 1.0, 0.1,
                               key="region_opt_w_food")
            w_cost = st.slider("Cost (lower = better)", 0.0, 1.0, 0.5, 0.1,
                               key="region_opt_w_cost")
            _region_opt_weights = {
                'mean_hm':          w_cool,
                'flood_reduction':  w_flood,
                'carbon_tons_co2':  w_carbon,
                'food_mln_lbs':     w_food,
                'total_cost_mln':   w_cost,
                'runoff_acre_feet': w_flood,  # piggyback flood weight onto runoff
            }

            # Two-RELAY lock — sidebar region "Optimize" trigger paired
            # with the mode label + caption above (Assertion B).
            if st.button("Optimize",
                         key="region_opt_button",
                         width="stretch",
                         help=_OPTIMIZE_HELP_REGION):
                _fire_region_optimize(
                    _CURRENT_CITY_STATE, selected_city,
                    DATA_DIR_FLOOD, DATA_DIR_COOLING,
                    st.session_state.get('selected_region_mask'),
                    st.session_state.get('selected_ownership_mask'),
                    cost_gi, cost_ff, cost_hd,
                    _region_opt_weights,
                )
            st.caption("Same search as the Optimize button in Discover scenarios — uses your current goals, region, and filters.")

# ── Sidebar section: Advanced model quality (Two-RELAY lock) ──────────────
# Model-quality controls extracted out of Discover into their own collapsed
# expander below Discover. The Discover surface stays focused on the
# action (mode label / caption / Optimize). The "Active: N training
# scenarios" indicator stays as a quiet honest-mode disclosure inside this
# expander — N reflects the current scenario_df (live), not a hardcoded
# value. Brief C.2: High Resolution opt-in checkbox preserved.
with _sec_advanced_quality:
    st.radio(
        "Model quality mode",
        options=MODEL_QUALITY_OPTIONS,
        index=0,
        key="model_quality",
        help=(
            "Controls how many full-resolution simulations are used to train the "
            "machine-learning model. More simulations improve optimizer suggestions but "
            "take longer to initialize."
        ),
        label_visibility="collapsed",
    )
    st.caption(
        "Fast prototype: ~90 training scenarios — quick startup, good for exploration.  \n"
        "Balanced: ~500 scenarios — better coverage, moderate startup time.  \n"
        "High resolution: trains on the full 2,541-entry lookup table — slower startup, better optimizer coverage."
    )
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
#   contract in docs/internal/DESIGN_NOTES.md "Lookup-overlay safety contract".
lookup_key = (pct_converted, green_infrastructure_pct, food_forest_pct)
# Region Selection Phase 1 (Commit 2): region-selected scenarios bypass the
# lookup table and run live. The lookup encodes citywide-convertible-pool
# math; a region mask shrinks that pool, so the cached citywide aggregates
# would be incorrect for the region-relative scenario. Off-grid scenarios
# are an advanced feature; the live cost is acceptable.
_selected_region_mask = st.session_state.get('selected_region_mask')
_selected_ownership_mask = st.session_state.get('selected_ownership_mask')
# Ownership Integration Commit 1 — `evaluate_scenario` takes a single
# placement-constraint mask. Compose `region ∩ ownership` here so the engine
# is unchanged. Either or both may be None.
if _selected_region_mask is not None and _selected_ownership_mask is not None:
    _combined_mask = _selected_region_mask & _selected_ownership_mask
elif _selected_ownership_mask is not None:
    _combined_mask = _selected_ownership_mask
else:
    _combined_mask = _selected_region_mask
if (lookup_key in lookup_table
        and placement_strategy == 'random'
        and _combined_mask is None):
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
        selected_region_mask=_combined_mask,
    )
    # Region Selection Phase 1 — caller stamps the descriptive fields onto
    # results['region_selection']. The mask was built upstream from
    # (layer_key, selected_labels); we have both here. selected_ids carries
    # LABEL VALUES (e.g. ['5']), not positional indices. See the contract
    # in evaluate_scenario's return block.
    if _selected_region_mask is not None:
        results['region_selection']['layer'] = st.session_state.get('selected_region_layer')
        results['region_selection']['selected_ids'] = st.session_state.get('selected_region_ids') or []

# Ownership Integration Commit 1 — caller stamps ownership_filter onto
# results. None for citywide / ownership-inactive scenarios; mode string
# (e.g. 'vacant_public') when active.
results['ownership_filter'] = st.session_state.get('selected_ownership_mode')


# ── Sidebar: Export for InVEST (Brief D1) ─────────────────────────────────────
# Placed AFTER `results` is built so the bundle helper can read it. Streamlit
# renders sidebar elements in code order; this lands at the bottom of the
# sidebar regardless of where in the script body it's defined.
def _build_invest_bundle_for_current_scenario():
    """Gather the current SA scenario state and build a D1 InVEST export
    bundle. Returns (zip_bytes, filename). SA-only (the bundle is built around
    NatCap's compound LULC framework)."""
    import subprocess

    if results['pct_converted'] == 0:
        provenance = eib.PROVENANCE_BASELINE
        generator = {"type": "baseline",
                     "note": "unmodified prototype LULC"}
        scen_label = f"Baseline — {selected_city}"
        scen_id = "baseline"
    elif st.session_state.get("applied_from_region_optimizer"):
        # Region-constrained optimizer (variant B). Surrogate-shortlisted
        # candidate, evaluated by the full engine on the active region∩ownership
        # mask. Distinct from PROVENANCE_OPTIMIZER because the displayed
        # values are engine-true region-local — the surrogate's role stopped
        # at shortlisting. See docs/internal/REGION_OPTIMIZER_SPEC.md §4.
        provenance = eib.PROVENANCE_REGION_OPTIMIZED
        generator = {
            "type": "region_optimizer_engine_verified",
            "pct_converted":            int(results['pct_converted']),
            "green_infrastructure_pct": int(results['green_infrastructure_pct']),
            "food_forest_pct":          int(results['food_forest_pct']),
            "high_density_pct":         int(results['pct_highdensity']),
            "placement_strategy":       placement_strategy,
            "random_seed":              42,
            "note": ("Applied from Region-Optimizer suggestion; a machine-learning "
                     "prefilter ranked candidates, full engine evaluated the "
                     "shortlist on the active region∩ownership mask before "
                     "export."),
        }
        scen_label = f"Region-optimized · {results['scenario_name']}"
        scen_id = (f"region_optimizer_pct{int(results['pct_converted'])}"
                   f"_gi{int(results['green_infrastructure_pct'])}"
                   f"_ff{int(results['food_forest_pct'])}_{placement_strategy}")
    elif st.session_state.get("applied_from_optimizer"):
        # Brief #4 — the slider state matches the optimizer's just-Applied
        # values, so this scenario came from the surrogate's discovery loop and
        # was then evaluated by the InVEST-aligned evaluator. Record OPTIMIZER provenance + an
        # honest generator note so downstream users (and the bundle metadata)
        # don't misread an optimizer-derived design as a manual Explorer one.
        provenance = eib.PROVENANCE_OPTIMIZER
        generator = {
            "type": "optimizer_suggested",
            "pct_converted":            int(results['pct_converted']),
            "green_infrastructure_pct": int(results['green_infrastructure_pct']),
            "food_forest_pct":          int(results['food_forest_pct']),
            "high_density_pct":         int(results['pct_highdensity']),
            "placement_strategy":       placement_strategy,
            "random_seed":              42,
            "note": ("Applied from Optimizer suggestion; evaluated by the "
                     "InVEST-aligned evaluator before export."),
        }
        scen_label = f"Optimizer suggestion · {results['scenario_name']}"
        scen_id = (f"optimizer_pct{int(results['pct_converted'])}"
                   f"_gi{int(results['green_infrastructure_pct'])}"
                   f"_ff{int(results['food_forest_pct'])}_{placement_strategy}")
    else:
        provenance = eib.PROVENANCE_EXPLORER
        generator = {
            "type": "explorer_generated",
            "pct_converted":            int(results['pct_converted']),
            "green_infrastructure_pct": int(results['green_infrastructure_pct']),
            "food_forest_pct":          int(results['food_forest_pct']),
            "high_density_pct":         int(results['pct_highdensity']),
            "placement_strategy":       placement_strategy,
            "random_seed":              42,
        }
        scen_label = results['scenario_name']
        scen_id = (f"explorer_pct{generator['pct_converted']}"
                   f"_gi{generator['green_infrastructure_pct']}"
                   f"_ff{generator['food_forest_pct']}_{placement_strategy}")

    scen_compound = results['scenario_lulc_ucm']         # compound view (SA)
    base_compound = cooling_lulc_compound                # baseline compound
    scen_nlcdtree = reduce_compound_to_nlcd_tree(
        scen_compound, COMPOUND_TO_NLCD_TREE).astype(np.int16)
    base_nlcdtree = reduce_compound_to_nlcd_tree(
        base_compound, COMPOUND_TO_NLCD_TREE).astype(np.int16)
    scen_ndvi = _lulc_to_ndvi_raster(results['scenario_lulc']).astype(np.float32)
    base_ndvi = _lulc_to_ndvi_raster(
        reduce_compound_to_nlcd(base_compound, COMPOUND_TO_NLCD)).astype(np.float32)

    ref_path = os.path.join(city_cfg['data_dir_flood'],
                            city_cfg['compound_lulc_file'])
    with rasterio.open(ref_path) as src:
        profile = dict(height=src.height, width=src.width,
                       crs=src.crs, transform=src.transform)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5).stdout.strip()
    except Exception:
        git_commit = "unknown"

    # Region Selection Phase 1 (Commit 5) + Ownership Integration Commit 3 —
    # propagate the structured region_selection / ownership_filter from
    # results into the bundle metadata, and build the augmented Source-line
    # string for metadata.json. Same augmentation as the main panel header
    # so downstream readers see the identical string. Use the layer-present
    # signal (not mode=='selected_regions') because the combined mask makes
    # mode='selected_regions' even when only ownership is active.
    _bundle_region_selection = (
        results.get('region_selection')
        if (results.get('region_selection') or {}).get('layer') is not None
        else None
    )
    # Scenario Record Pass — enrich the export's ownership block with
    # source / data_date / allowed_classes composed from OWNERSHIP_MODES +
    # CITIES[city]['ownership_layer']. In-memory results['ownership_filter']
    # stays a bare mode string (all existing consumers read it as such);
    # the rich shape lives only in the export bundle, so metadata.json is
    # self-describing without forcing an all-consumer audit.
    # Finer Ownership Classes Pass — `allowed_classes` is the list of
    # band-1 class-enum values that satisfy the mode (e.g. [1] for
    # 'city', [1,2,3,4] for the 'public' rollup, or the union for a
    # multi-class composite). Empty list means the mode keys only on
    # band 2 (vacant); the consumer reads `vacant_required` to detect
    # that. The normalizer handles all three storage shapes (None / str
    # mode key / composite dict from Batch 4 v2's checkbox panel).
    _of_norm = _normalize_ownership_filter(results.get('ownership_filter'))
    if _of_norm is not None:
        _of_layer_meta = (city_cfg.get('ownership_layer') or {})
        # Allowed band-1 class enum values, derived from the normalized
        # class list. Use each class's `band1_eq` from OWNERSHIP_MODES.
        _of_allowed = sorted({
            OWNERSHIP_MODES[c]['band1_eq']
            for c in _of_norm['classes']
            if c in OWNERSHIP_MODES and 'band1_eq' in OWNERSHIP_MODES[c]
        })
        # `mode` field carries the original storage shape — either a
        # string mode key or a small composite dict. Downstream readers
        # can either use it directly or fall back to label + allowed_classes.
        _bundle_ownership_filter = {
            'mode':              results.get('ownership_filter'),
            'label':             _of_norm['label'],
            'allowed_classes':   _of_allowed,
            'vacant_required':   _of_norm['vacant_only'],
            'source':            _of_layer_meta.get('source'),
            'data_date':         _of_layer_meta.get('data_date'),
        }
    else:
        _bundle_ownership_filter = None
    _bundle_source = _PROVENANCE_HEADER_INFO.get(provenance, ("Unknown",))[0]
    if provenance == eib.PROVENANCE_EXPLORER:
        if _bundle_region_selection is not None:
            _bundle_source = f"{_bundle_source} · selected region"
        _bundle_source = f"{_bundle_source}{_ownership_source_suffix(results)}"

    spec = eib.BundleSpec(
        city_name=selected_city, city_slug="san_antonio_tx",
        crs=city_cfg['crs'], pixel_size_m=30,
        scenario_id=scen_id, scenario_label=scen_label,
        scenario_description=f"{provenance} scenario from Ecosystem Explorer.",
        provenance=provenance, generator=generator,
        git_commit=git_commit, scenario_schema_version=SCENARIO_SCHEMA_VERSION,
        is_sa=True, raster_profile=profile,
        region_selection=_bundle_region_selection,
        ownership_filter=_bundle_ownership_filter,
        region_local=results.get('region_local'),
        region_local_treatment=(_REGION_LOCAL_METRICS if results.get('region_local') else None),
        generator_params={
            'pct_converted':            pct_converted,
            'green_infrastructure_pct': green_infrastructure_pct,
            'food_forest_pct':          food_forest_pct,
            'placement_strategy':       placement_strategy,
            'cost_gi':                  cost_gi,
            'cost_ff':                  cost_ff,
            'cost_hd':                  cost_hd,
            'carbon_rate_ff':           st.session_state.get('carbon_rate_ff'),
            'carbon_rate_gi':           st.session_state.get('carbon_rate_gi'),
        },
        source_label=_bundle_source,
        scenario_lulc_compound=scen_compound,
        baseline_lulc_compound=base_compound,
        scenario_lulc_nlcdtree=scen_nlcdtree,
        baseline_lulc_nlcdtree=base_nlcdtree,
        scenario_ndvi=scen_ndvi, baseline_ndvi=base_ndvi,
        pop_path=city_cfg['pop_file'],
        et_path=city_cfg['et_file'],
        soil_path=os.path.join(city_cfg['data_dir_flood'], city_cfg['soil_file']),
        block_groups_path=city_cfg['tracts_file'],
        ucm_table_path='data/sa/natcap_2024/ucm__nlcd_nlud_tree.csv',
        una_table_path=city_cfg['una_table_file'],
        carbon_table_path=city_cfg['carbon_table_file'],
        cn_table_path=os.path.join(city_cfg['data_dir_flood'],
                                   city_cfg['cn_table_file']),
        uhi_max_c=UHI_MAX_C, t_ref_c=35.0, t_air_average_radius_m=600,
        green_area_cooling_distance_m=GREEN_AREA_COOLING_DISTANCE_M,
        una_demand_m2=UNA_DEMAND_M2_PER_CAPITA,
        una_radius_m=int(UNA_SEARCH_RADIUS_M),
        una_decay=UNA_DECAY_FUNCTION,
        design_storm_mm=round(DESIGN_STORM_MM, 1),
        umh_search_radius_m=UMH_SEARCH_RADIUS_M,
        umh_rr_depression=RR_0_1_NDVI_DEPRESSION,
        umh_rr_anxiety=RR_0_1_NDVI_ANXIETY,
        umh_bir_depression=BIR_DEPRESSION, umh_bir_anxiety=BIR_ANXIETY,
        umh_cost_depression=float(COST_PER_DEPRESSION_CASE_USD),
        umh_cost_anxiety=float(COST_PER_ANXIETY_CASE_USD),
    )
    return eib.build_invest_bundle(spec), eib.bundle_filename(spec)


# ── Sidebar section: Export (Sidebar Reorg) ──────────────────────────────────
with _sec_export:
    if not selected_city.startswith("San Antonio"):
        st.caption(
            "InVEST export is currently SA-only (the bundle is built around NatCap's "
            "compound LULC framework). MN export is future work."
        )
    else:
        st.caption(
            "Download the current scenario as a runnable canonical-InVEST 3.19.0 "
            "input bundle — rasters + AOIs + biophysical tables + per-model "
            "`args.json` (UCM / UNA / UFR / Carbon / UMH). ~20 MB; for technical "
            "users with InVEST installed."
        )
        if st.button("Prepare InVEST bundle"):
            with st.spinner("Building InVEST bundle (10–30 s)…"):
                _data, _fname = _build_invest_bundle_for_current_scenario()
                st.session_state["_invest_bundle"] = (_data, _fname)
        if "_invest_bundle" in st.session_state:
            _data, _fname = st.session_state["_invest_bundle"]
            st.download_button(
                f"⬇ Download bundle ({len(_data) / 1e6:.1f} MB)",
                data=_data, file_name=_fname, mime="application/zip",
            )
            st.caption(f"`{_fname}`")
            if st.button("Clear prepared bundle"):
                del st.session_state["_invest_bundle"]
                st.rerun()


# ── Top metric cards ───────────────────────────────────────────────────────────
def _fmt_runoff(af):
    if af >= 1_000:
        return f"{af / 1_000:.1f}K ac-ft"
    return f"{af:.0f} ac-ft"

def _fmt_runoff_value(af):
    """Units-less runoff magnitude for the card value (unit lives in the label).
    Single-sourced on _fmt_sig — 100,800 → '101k'."""
    return _fmt_sig(af)

def _fmt_food(mln_lbs):
    """Food yield — single-sourced on _fmt_sig (pass raw lbs so the k/M unit
    floats): 79.3 → '79.3M lbs/yr', 0.5 → '500k lbs/yr'."""
    return f"{_fmt_sig(mln_lbs * 1e6)} lbs/yr"

def _fmt_people(n):
    if n >= 1_000:
        return f"~{n // 1_000}K people"
    return f"~{n} people"

def _fmt_temp_change(dt, *, threshold=0.1):
    """Natural-language temperature change from a signed ΔT (positive = warmer,
    negative = cooler). Display layer for the internal `temp_change_f`
    convention — keeps the sign out of the user's view (see
    `hm_to_temp_change_f`)."""
    if abs(dt) < threshold:
        return "No change"
    return f"{abs(dt):.1f}°F warmer" if dt > 0 else f"{abs(dt):.1f}°F cooler"

def _delta_pill(value_delta, *, fmt="", suffix="vs baseline", epsilon=0.05,
                inverse=False):
    """Consistent delta string + color for st.metric cards.

    Returns (delta_str, delta_color).
    - Zero-delta  →  (None, "off")  — pill suppressed entirely
    - Otherwise   →  ("±{value} {suffix}", color), where color is "normal" for
      higher-is-better metrics (green ↑ / red ↓) or "inverse" for
      lower-is-better metrics (`inverse=True` → red ↑ / green ↓).

    Pass the RAW signed delta (scenario − baseline) and set `inverse` to match
    the metric's direction-of-goodness, so the arrow tracks the real change and
    the colour tracks whether that change is good. Lower-is-better cards (e.g.
    Runoff Volume) MUST pass `inverse=True` — an increase is a regression, not a
    green-up improvement.

    For zero-baselined metrics (currently Carbon), callers may pass the raw
    value rather than a computed delta — the helper treats them equivalently.
    If a non-zero baseline is ever introduced, call sites must switch to a
    true delta.
    """
    if abs(value_delta) < epsilon:
        return None, "off"
    color = "inverse" if inverse else "normal"
    if value_delta > 0:
        return f"+{value_delta:{fmt}} {suffix}", color
    return f"-{abs(value_delta):{fmt}} {suffix}", color

# The active scenario's context. Drives the badge's anchored-vs-aligned color
# rule. Baseline = pct_converted == 0; any slider-derived non-zero scenario is
# Explorer for B2 v1 (Optimizer scenarios don't have a separate render path —
# the user manually copies the slider values). When the fixed-scenario
# reference view lands (Phase 3), it sets its own NATCAP_FIXED context locally.
_validation_scenario_context = (
    nv.SCENARIO_CONTEXT_BASELINE if results['pct_converted'] == 0
    else nv.SCENARIO_CONTEXT_EXPLORER
)

# read from state to avoid silent-staleness if city switches
_flood_delta = results['flood_reduction'] - (100 - _CURRENT_CITY_STATE.baseline_cn)
_flood_delta_str, _flood_delta_color = _delta_pill(_flood_delta, fmt=".1f", epsilon=0.1)
_temp_change_f = results['temp_change_f']
_temp_change_label = _fmt_temp_change(_temp_change_f)
_hm_delta = results['mean_hm'] - _CURRENT_CITY_STATE.baseline_hm
# Lower Runoff Volume is better, so pass the RAW change (scenario − baseline)
# with inverse=True: a decrease renders green ↓ (improvement), an increase
# renders red ↑ (regression) — matching the "lower Runoff Volume is better"
# caption. (Was pre-flipped `prevented` + "normal", which showed a reduction as
# a green ↑ "+803 vs baseline" that read as runoff going up.)
_runoff_change = results['runoff_acre_feet'] - BASELINE_RUNOFF_ACRE_FEET
_runoff_delta_str, _runoff_delta_color = _delta_pill(
    _runoff_change, fmt=",.0f",
    suffix="ac-ft vs baseline",
    epsilon=1.0,
    inverse=True,
)
_people_fed = results['people_fed']
_food_delta_str = f"feeds ~{_people_fed:,} people" if _people_fed > 0 else None

_carbon_value = results['carbon_tons_co2']

# Brief 30: `_CARBON_IS_STOCK` is set once after the city-state aliasing
# above; here we derive the dependent display strings.
_carbon_unit_suffix = "t CO2e" if _CARBON_IS_STOCK else "t CO2e/yr"

# Optimizer candidate tables + the Fast-estimate range table all render the SAME
# field (carbon_tons_co2 — a one-time stock CHANGE vs baseline for SA, an annual
# sequestration flow for MN), so they share ONE column label and can never
# diverge. Wording matches the card ("Carbon Storage Change" / "Carbon
# Sequestration"); unit is the normalized suffix (t CO2e — never "tons CO2e").
_carbon_table_col_label = (
    f"Carbon storage change ({_carbon_unit_suffix})" if _CARBON_IS_STOCK
    else f"Carbon sequestration ({_carbon_unit_suffix})"
)

def _fmt_carbon(tons):
    """Units-less carbon magnitude for the card value (unit lives in the label).
    Single-sourced on _fmt_sig — 3,095,697 → '3.10M' (screening precision)."""
    return _fmt_sig(tons)

# Brief 2 (Approach Y): the SA four-pool stock card is bespoke, mirroring
# Brief 1's signed-card pattern — flip to a "Loss" label with a positive
# magnitude and a red ↑ delta when conversions reduce stored carbon. MN's
# annual sequestration flow is always ≥ 0, so it keeps the shared `_delta_pill`
# path and the "Carbon Sequestration" label. Lifting only the SA branch out of
# `_delta_pill` leaves the other three callers (flood, runoff, NDVI) untouched.
# Units live in the card LABEL ("… (t CO2e)") and the value/delta carry just the
# abbreviated magnitude, so nothing ellipsizes at 1/3 width.
_CARBON_PILL_EPSILON = 1.0
_carbon_unit_label = f"({_carbon_unit_suffix})"
if _CARBON_IS_STOCK:
    if _carbon_value < -_CARBON_PILL_EPSILON:
        _carbon_card_label = f"Carbon Storage Loss {_carbon_unit_label}"
        _carbon_value_str = _fmt_carbon(abs(_carbon_value))
        _carbon_delta_str = f"+{_fmt_carbon(abs(_carbon_value))} lost"
        _carbon_delta_color = "inverse"
    elif _carbon_value > _CARBON_PILL_EPSILON:
        _carbon_card_label = f"Carbon Storage Change {_carbon_unit_label}"
        _carbon_value_str = _fmt_carbon(_carbon_value)
        _carbon_delta_str = f"+{_fmt_carbon(_carbon_value)} stock change"
        _carbon_delta_color = "normal"
    else:
        _carbon_card_label = f"Carbon Storage Change {_carbon_unit_label}"
        _carbon_value_str = _fmt_carbon(_carbon_value)
        _carbon_delta_str = None
        _carbon_delta_color = "off"
else:
    _carbon_card_label = f"Carbon Sequestration {_carbon_unit_label}"
    _carbon_value_str = _fmt_carbon(_carbon_value)
    if _carbon_value > _CARBON_PILL_EPSILON:
        _carbon_delta_str = f"+{_fmt_carbon(_carbon_value)} from conversions"
        _carbon_delta_color = "normal"
    else:
        _carbon_delta_str = None
        _carbon_delta_color = "off"

# ── Scenario header (Brief #3 — unified Source + Validation) ─────────────────
# The fixed-scenario reference view has its own header above (rendered inside
# _render_natcap_fixed_scenario_view then st.stop()s); this path renders the
# Explorer / baseline / optimizer cases. Brief #4 plumbed the Applied-from-
# Optimizer flag so a just-applied optimizer scenario flips to Surrogate-
# suggested provenance; the clearing logic at the top of the script resets
# it whenever sliders drift, so the OPTIMIZER tag never stays stale.
if results['pct_converted'] == 0:
    _scen_provenance = eib.PROVENANCE_BASELINE
    # Relay A — pct=0 is the user's "no conversion" Explorer choice. Frame
    # honestly via the unified helper rather than the previous
    # "Baseline — {city}" string, which buried the fact that the no-conversion
    # state came from dragging Explorer sliders to 0. Provenance stays
    # BASELINE (the engine output IS baseline-equivalent at pct=0); only
    # the displayed banner label changes.
    _scen_label = _explorer_scenario_label(_resolved_scenario)
elif st.session_state.get("applied_from_region_optimizer"):
    _scen_provenance = eib.PROVENANCE_REGION_OPTIMIZED
    _scen_label = f"Region-optimized · {results['scenario_name']}"
elif st.session_state.get("applied_from_optimizer"):
    _scen_provenance = eib.PROVENANCE_OPTIMIZER
    _scen_label = f"Optimizer suggestion · {results['scenario_name']}"
else:
    _scen_provenance = eib.PROVENANCE_EXPLORER
    # Relay A — use the unified display helper so the banner title shares
    # its source with the main-panel + audit sentences. The helper branches
    # on pct=0 ("no conversion" label) so a no-conversion state reads
    # honestly instead of advertising 0%/0%/100% allocation knobs that
    # don't fire.
    _scen_label = _explorer_scenario_label(_resolved_scenario)

# ── Active scenario (page-root summary) ─────────────────────────────────────
# Provenance-led compact block composed from the single-source `_scen_provenance`
# + `_resolved_scenario` (line 1) and the scope helpers the prior "Current setup"
# line used (line 2). Rendered HIGH — right after provenance resolves and just
# above the Source+Validation header + Discover centerpiece — so "what am I
# looking at" sits above the trust/validation layer. Replaces the verbose "This
# scenario converts…" sentence + the "Current setup:" line (both deleted from the
# old page-root location); onboarding prose lives in the "How to read this scenario"
# expander. Scope pieces stay in lockstep with the Scenario audit expander (same
# _cs_* helpers, "regional extent" vocab).
_scope_area = _cs_area_for_row(results)
_scope_own  = _cs_ownership_for_row(results)
# Lead the scope with the area itself — citywide reuses the locked
# "<city> regional extent" descriptor (state suffix dropped; the city
# subheader directly above already carries it), region active leads with the
# region/district label alone. No standalone "{city}, {state}" token.
_city_short = selected_city.split(',')[0]
_setup_region = (f"{_city_short} regional extent" if _scope_area == "Citywide"
                 else _scope_area)
_setup_own = ("no ownership filter" if _scope_own == "None"
              else _scope_own.lower())
# Short strategy key (not the descriptive label, whose "Random placement" doubled
# the trailing " placement"). Dedup guard keeps it reading "<strategy> placement".
_setup_place = placement_strategy
if not _setup_place.endswith("placement"):
    _setup_place += " placement"
st.markdown(
    f"**Active scenario**  \n"
    f"{_active_scenario_line1(_resolved_scenario, _scen_provenance)}"
)
if results['pct_converted'] > 0:
    st.caption(
        f"Scope: {_setup_region} · {_setup_own} · {_setup_place}"
    )
else:
    # No conversion → ownership/placement are moot; show only the area phrase.
    st.caption(f"Scope: {_setup_region}")
with st.expander("How to read this scenario"):
    st.write(
        "The scenario converts a share of eligible developed land into green "
        "infrastructure, food forest, or higher-density development. Roads, "
        "buildings, existing natural land, and active filters determine the "
        "eligible pool."
    )

# Region Selection Phase 1 (Commit 5) + Ownership Integration Commit 3 —
# augment the Source line text when an Explorer scenario is region- and/or
# ownership-constrained. Baseline (pct=0) reads just 'Baseline' — don't
# augment. PROVENANCE_OPTIMIZER (citywide) and PROVENANCE_REGION_OPTIMIZED
# carry their own scope semantics in the label and don't get the suffix.
_region_active = (
    _scen_provenance == eib.PROVENANCE_EXPLORER
    and st.session_state.get('selected_region_mask') is not None
)
_source_suffix = (
    (" · selected region" if _region_active else "")
    + (_ownership_source_suffix(results) if _scen_provenance == eib.PROVENANCE_EXPLORER else "")
)
# show_scenario_label=False — the Active scenario block (line 1) directly above
# already names the recipe, and the Source line below restates it; the big ##
# heading here was a third copy. Drop it; the Source/Validation box is
# self-labelled, and the actual metrics live two sections down (Discover +
# tabs), so a "results" header here would be misplaced.
_render_scenario_provenance_header(_scen_provenance, scenario_label=_scen_label,
                                    source_suffix=_source_suffix,
                                    show_scenario_label=False)

# ── Optimizer Promotion — main-panel CTA (centerpiece) ──────────────────────
# Two-RELAY lock structure: card title "Discover scenarios" (constant) +
# promoted mode label (visible markdown, NOT a faint caption) + honesty
# caption + short "Optimize" button at full container width. The mode
# label + caption + button co-render in the same st.container — Assertion
# B in verify_baselines machine-locks this pairing so the big button can
# never detach from its honesty framing. Both the sidebar Discover button
# and this CTA route through the same _fire_*_optimize helpers; the
# shared-fire assertion locks that contract. NatCap mode never reaches
# here (st.stop() short-circuits the scenario-source selector).
with st.container(border=True):
    st.markdown("### Discover scenarios")
    if _filter_active:
        st.markdown("**Selected-area search**")
        st.caption(
            "Searches candidate mixes under the current area and filters. Displayed values are computed by the InVEST-aligned evaluator, not model predictions."
        )
    else:
        st.markdown("**Citywide machine-learning search**")
        st.caption("Fast estimates suggest promising mixes; apply one to recompute with the InVEST-aligned evaluator.")
    # No help= on the main CTA button: the card already explains itself (mode
    # label + caption above), and a help tooltip here floats over the card. The
    # sidebar Optimize buttons keep their help (_OPTIMIZE_HELP_*), where there's
    # less overlap.
    if st.button("Optimize", type="primary",
                  key="main_cta_optimize_button",
                  width="stretch"):
        if _filter_active:
            _fire_region_optimize(
                _CURRENT_CITY_STATE, selected_city,
                DATA_DIR_FLOOD, DATA_DIR_COOLING,
                st.session_state.get('selected_region_mask'),
                st.session_state.get('selected_ownership_mask'),
                cost_gi, cost_ff, cost_hd,
                _region_opt_weights,
            )
        else:
            _fire_citywide_optimize(
                surrogate, min_flood, min_cool, min_food,
                max_runoff, min_carbon,
                MAX_FOOD, MAX_FLOOD, MAX_COOL,
            )

# ── Scenario audit expander ───────────────────────────────────────────────────
# Single-place view of the current scenario's record. Every field reads the
# record directly — no recomputation, no parallel truth. Inapplicable fields
# render the uniform value ("Citywide" / "None") so the field list is
# consistent across all scenario types. Module-level helpers
# `_cs_area_for_row` / `_cs_ownership_for_row` compose the Area / Ownership
# cells (same rule the comparison-table columns use).
with st.expander("Scenario audit", expanded=False):
    # UI feedback #3 — open with a prose sentence (mirroring the main
    # scenario caption rendered below the tabs) so the audit reads as
    # human description first, then the structured fields. None of the
    # interpolated values carry `$`, so no escape needed; if that ever
    # changes (e.g. cost is interpolated into the sentence), escape.
    _audit_area_inline = _cs_area_for_row(results)
    _audit_own_inline  = _cs_ownership_for_row(results)
    _audit_strategy    = PLACEMENT_STRATEGY_LABELS.get(
        placement_strategy, placement_strategy)
    _audit_own_clause  = (
        "" if _audit_own_inline == "None"
        else f" restricted to {_audit_own_inline.lower()}"
    )
    # Relay A — audit sentence reads from the same _resolved_scenario the
    # banner title + main-panel sentence use, branching on pct=0.
    st.write(_explorer_audit_sentence(
        _resolved_scenario, _audit_area_inline, _audit_own_clause,
        _audit_strategy,
    ))
    _audit_rs = results.get('region_selection') or {}
    _audit_eligible_acres = (
        (_audit_rs.get('eligible_pixels_in_region') or 0) * PIXEL_AREA_ACRES
    )
    _audit_converted_acres = _audit_rs.get('converted_acres') or 0.0
    # Validation label = locked badge vocab from _PROVENANCE_HEADER_INFO
    # (the same mapping the provenance header renders). Tuple shape:
    # (Source, Validation, color).
    _audit_validation = _PROVENANCE_HEADER_INFO.get(
        _scen_provenance, ("Unknown", "provenance not recorded", "gray")
    )[1]
    # Source = the augmented Source-line text the header just rendered
    # (provenance label + selected-region / ownership suffixes when active).
    _audit_source = _PROVENANCE_HEADER_INFO.get(
        _scen_provenance, ("Unknown",))[0] + _source_suffix
    _audit_rows = [
        ("Source",          _audit_source),
        ("Area",            _cs_area_for_row(results)),
        ("Ownership",       _cs_ownership_for_row(results)),
        ("Placement",       PLACEMENT_STRATEGY_LABELS.get(
                                placement_strategy, placement_strategy)),
        ("Seed",            "42"),
        ("Eligible acres",  f"{_audit_eligible_acres:,.0f} acres"),
        ("Converted acres", f"{_audit_converted_acres:,.0f} acres"),
        ("Validation",      _audit_validation),
        ("Export schema",   str(SCENARIO_SCHEMA_VERSION)),
    ]
    _audit_df = pd.DataFrame(_audit_rows, columns=["Field", "Value"])
    st.dataframe(
        _audit_df, hide_index=True, width="stretch",
        column_config={
            "Field": st.column_config.TextColumn("Field", width="small"),
            "Value": st.column_config.TextColumn("Value", width="large"),
        },
    )

if placement_strategy != 'random':
    st.caption(f"Placement: {PLACEMENT_STRATEGY_LABELS[placement_strategy]}")

# Relay 46/48 — interpretation header above the metric cards. Behavior unchanged:
# the cards are citywide. The heading reflects whether the scenario is
# constrained; the constrained scope line and the higher/lower direction line are
# each stated ONCE here (the direction line is removed from the region-local
# header below). Badge meaning lives in the color legend + signposting line just
# below — not duplicated here, and no "provenance" (Relay 43).
_scope_constrained = (
    (results.get('region_selection') or {}).get('mode') == 'selected_regions'
    or st.session_state.get('selected_ownership_mask') is not None
)
st.markdown(
    "### Citywide impact from selected-area placement" if _scope_constrained
    else "### Citywide impact"
)
if _scope_constrained:
    st.caption(
        "Cards report impact across the full modeled region, even when changes "
        "are constrained to selected areas. Local effects are in the Scenario "
        "tab and selected-area tradeoff plot."
    )
st.markdown(
    "Higher is generally better for benefit metrics; lower is better for "
    "Runoff Volume and Implementation Cost."
)
st.caption(
    "Badges show validation/provenance: ◆ NatCap published value = reference; "
    "■ InVEST-validated = checked against canonical InVEST; ○ InVEST-aligned = "
    "same method, not directly checked here; △ Prototype = exploratory. "
    "Full definitions in 'How this prototype works'."
)
st.caption(
    "Details are available through metric badges, tooltips, and the Scenario audit."
)

st.markdown("#### Ecological")
# Flood trio forced into their own row of three (Option B / Relay) so the
# under-row "Flood metrics" caption sits beneath exactly these three cards
# and not Temp change. Order: Flood Index → Runoff Retention → Runoff Volume.
eco1, eco2, eco3 = st.columns(3)
eco1.metric(
    "Flood Index",
    _fmt_sig(results['flood_reduction']),
    delta=_flood_delta_str,
    delta_color=_flood_delta_color,
    help=(
        "Higher is better. A unitless index derived from Curve Number: "
        "100 − mean CN. Higher values generally indicate lower runoff potential. "
        "Useful for comparing scenarios, not a direct flood-volume estimate. "
        "See 'How this prototype works'."
    )
)
_render_validation_caption(eco1, "flood_reduction", _validation_scenario_context)
eco2.metric(
    "Runoff Retention",
    f"{_fmt_sig(results['runoff_retention_idx'] * 100)}%",
    delta=None,
    delta_color="off",
    help=(
        "Higher is better. Estimated share of design-storm rainfall retained "
        "rather than becoming runoff, averaged over modeled pixels. This is the "
        "InVEST-aligned runoff-retention metric; use it alongside Flood Index and "
        "Runoff Volume. Validated against canonical InVEST where comparable. "
        "See 'How this prototype works'."
    )
)
_render_validation_caption(eco2, "runoff_retention_idx", _validation_scenario_context, explicit_status="aligned_method")
eco3.metric(
    "Runoff Volume (ac-ft)",
    _fmt_runoff_value(results['runoff_acre_feet']),
    delta=_runoff_delta_str,
    delta_color=_runoff_delta_color,
    help=(
        "Lower is better. Modeled runoff volume for the city-specific design "
        "storm. Values are shown in acre-feet; the delta shows reduction versus "
        "baseline. See 'How this prototype works'."
    )
)
_render_validation_caption(eco3, "runoff_acre_feet", _validation_scenario_context, explicit_status="aligned_method")
st.caption(
    "Flood metrics: higher Flood Index and Runoff Retention are better; "
    "lower Runoff Volume is better."
)

_ndvi_delta = results['mean_ndvi'] - BASELINE_NDVI
_ndvi_delta_str, _ndvi_delta_color = _delta_pill(_ndvi_delta, fmt=".3f", suffix="vs baseline", epsilon=0.001)

# Second eco row: Temp change → Carbon Storage Change → NDVI. Carbon gets extra
# width (weighted row, per follow-up relay) so "Carbon Storage Change (t CO2e)"
# doesn't truncate; trades row-1/row-2 column alignment for a full label+unit.
eco4, eco5, eco6 = st.columns([2, 3, 2])
# Temp card: magnitude in the value; the cooler/warmer direction goes in a plain
# caption beneath — NOT st.metric's delta slot, which renders a meaningless arrow
# (identical for cooler and warmer under delta_color="off", so it carries no info).
if abs(_temp_change_f) < 0.1:
    _temp_card_value, _temp_card_dir = "No change", None
else:
    _temp_card_value = f"{abs(_temp_change_f):.1f}°F"
    _temp_card_dir = "warmer" if _temp_change_f > 0 else "cooler"
eco4.metric(
    "Temp change",
    _temp_card_value,
    delta=None,
    delta_color="off",
    help="Cooler is better. Approximate temperature change from the Heat Mitigation Index. Useful for comparing scenarios; treat °F changes as approximate, about ±2°F at best, because HMI-to-temperature calibration is uncertain. See 'How this prototype works'."
)
# Badge BEFORE the cooler/warmer caption so Temp's badge baseline aligns with
# Carbon / NDVI in this row (which go value → badge directly). The other rows'
# sub-captions already sit after their badges. POST-COMMIT EYEBALL: confirm
# "cooler"/"warmer" still reads right sitting BELOW the badge; if it reads as
# attached to the badge instead of the value, fall back to folding the direction
# back into the value string (e.g. a compact "0.3°F cooler" form).
_render_validation_caption(eco4, "temp_change_f", _validation_scenario_context)
if _temp_card_dir:
    eco4.caption(_temp_card_dir)
# `_carbon_card_label` is set above alongside the value/delta (Brief 2,
# Approach Y) so the SA loss-flip label survives.
_carbon_card_help = (
    (
        "One-time change in landscape carbon storage from land-cover "
        "conversion. Includes aboveground biomass, belowground biomass, soil, "
        "and dead organic matter where available. "
        "This is a stock change, not an annual flow. See 'How this prototype works'."
    )
    if _CARBON_IS_STOCK else
    (
        "Prototype annual carbon-sequestration estimate for newly converted "
        "pixels, using provisional regional rates. An annual flow, not a stock "
        "change. Treat as directional. See 'How this prototype works'."
    )
)
eco5.metric(
    _carbon_card_label,
    _carbon_value_str,
    delta=_carbon_delta_str,
    delta_color=_carbon_delta_color,
    help=_carbon_card_help,
)
_render_validation_caption(eco5, "carbon_tons_co2", _validation_scenario_context, explicit_status=None if _CARBON_IS_STOCK else "prototype", validated_path=_CARBON_IS_STOCK)
eco6.metric(
    "NDVI",
    f"{results['mean_ndvi']:.3f}",
    delta=_ndvi_delta_str,
    delta_color=_ndvi_delta_color,
    help=(
        "Prototype greenness indicator derived from land-cover class, not "
        "measured satellite NDVI. Higher means more vegetation. Treat as "
        "directional. See 'How this prototype works'."
    )
)
_render_validation_caption(eco6, "ndvi", _validation_scenario_context, explicit_status="prototype")

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
# Children's Nature Access — hide-when-near-equal. When child data exists but
# children's access is within EPSILON of overall Nature Access, the card carries
# no new signal (e.g. SA ~99.7% vs ~99.7%) — suppress it entirely so its
# PRESENCE means "kids are differently served here." Keep it when child data is
# absent (renders "—") or when it meaningfully diverges (>= EPSILON).
_CHILD_NAT_DIVERGENCE_EPSILON_PP = 0.5


def _should_show_child_card(child_nat, nature_access, eps):
    """Children's Nature Access card visibility (Relay 65 — machine-locked).
    Show only when child data is ABSENT (the card renders '—') OR children's
    access diverges from overall by at least `eps` pp. Suppressing the
    near-equal case keeps the card from implying children are a distinct
    beneficiary group when the measurement says they track the overall metric
    (SA ~0.3pp). Tested non-vacuously in verify_baselines."""
    return (child_nat is None) or (abs(child_nat - nature_access) >= eps)


_nature_access = results.get('nature_access_pct', 0.0)
_child_nat = results.get('children_nature_access_pct')
_show_child_card = _should_show_child_card(
    _child_nat, _nature_access, _CHILD_NAT_DIVERGENCE_EPSILON_PP)

# Two-row layout — Row 1 (Nature Access cluster) gets three columns (two when
# the Children's card is suppressed); Row 2 (MH outcomes) gets two.
# Five-in-one-row was truncating the longer card labels.
if _show_child_card:
    hs_na, hs_cna, hs_sch = st.columns(3)
else:
    hs_na, hs_sch = st.columns(2)
hs3, hs4 = st.columns(2)

# Nature Access — canonical InVEST Urban Nature Access (2SFCA), re-implemented
# in numpy by `calculate_nature_access`. See docs/internal/DESIGN_NOTES.md.
_nature_aoi = (
    "City of San Antonio (ACS block groups)"
    if selected_city.startswith("San Antonio")
    else "the downtown Minneapolis modelable extent (Census tracts)"
)
hs_na.metric(
    "Nature Access",
    f'{_nature_access:.1f}%',
    help=(
        "Share of the modeled population whose nearby nature supply meets the "
        "per-capita access threshold. Computed using the InVEST Urban Nature "
        "Access 2SFCA method. Reports only the modelable population extent. "
        "See 'How this prototype works'."
        + ("" if _show_child_card else
           " Children's nature access matches overall access here "
           f"(within {_CHILD_NAT_DIVERGENCE_EPSILON_PP:g} pp), so the "
           "separate Children's Nature Access card is hidden.")
    ),
)
_render_validation_caption(hs_na, "nature_access_pct", _validation_scenario_context)

# Children's nature access (RELAY) — same adequate mask as Nature Access,
# weighted by Census 2020 block-level under-18 population (P1 - P3 from
# PL 94-171, uniform-block-spread). Renders "—" when no child raster is
# configured for the active city; suppressed entirely (see _show_child_card
# above) when it tracks overall Nature Access within EPSILON. Strongest paired
# with the school-land ownership filter — that combination targets conversions
# to school-land parcels and reports how many children gain access as a result.
if _show_child_card:
    if _child_nat is None:
        _child_nat_value = "—"
        _child_nat_help_tail = (
            " (no Census child-population raster configured for this city; "
            "card hides the value rather than showing zero.)"
        )
    else:
        _child_nat_value = f'{_child_nat:.1f}%'
        _child_nat_help_tail = (
            " This card appears only when children's access diverges from "
            f"overall access by at least {_CHILD_NAT_DIVERGENCE_EPSILON_PP:g} pp "
            "— its presence itself signals kids are differently served here."
        )
    hs_cna.metric(
        "Children's Nature Access",
        _child_nat_value,
        help=(
            "Share of the modeled under-18 population whose nearby nature supply "
            "meets the per-capita access threshold. Same method as Nature Access, "
            "reweighted by child population, so it closely tracks overall access. "
            "See 'How this prototype works'." + _child_nat_help_tail
        ),
    )
    _render_validation_caption(
        hs_cna, "children_nature_access_pct", _validation_scenario_context,
        explicit_status="aligned_method",
    )

# Nature Access at Schools (RELAY) — destination-based metric. Samples the
# same 2SFCA `adequate` mask at school-point locations. Strongest paired
# with the residential metrics above for the "where children live vs where
# they spend the day" story. Source: NCES CCD/PSS/EDGE 2021-22 (K-12
# public + charter + private; clipped to modelable extent).
_sch = results.get('schools_nature_access') or {}
_sch_pct = _sch.get('pct')
if _sch_pct is None:
    _sch_value = "—"
    _sch_help_tail = (
        " (no NCES schools file configured for this city; card hides the "
        "value rather than showing zero.)"
    )
    _sch_bd = ""
else:
    _sch_value = f"{_sch_pct:.1f}%"
    _bd = _sch.get('by_sector') or {}
    _sch_bd = (
        f"public {_bd.get('public', {}).get('pct', '—')}% · "
        f"charter {_bd.get('charter', {}).get('pct', '—')}% · "
        f"private {_bd.get('private', {}).get('pct', '—')}%"
    )
    _sch_help_tail = ""
hs_sch.metric(
    "Schools with Nature Access",
    _sch_value,
    help=(
        "Share of mapped K–12 school locations meeting the same nature-access "
        "threshold. This is sampled at school points, "
        "not attendance boundaries, so it does not estimate which students "
        "attend which schools. See 'How this prototype works'." + _sch_help_tail
    ),
)
_render_validation_caption(
    hs_sch, "schools_nature_access_pct", _validation_scenario_context,
    explicit_status="aligned_method",
)
if _sch_bd:
    hs_sch.caption(_sch_bd)

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
        "Estimated preventable depression and anxiety cases from the scenario's "
        "greenness exposure change, using the InVEST Urban Mental Health model. "
        "Scenario greenness is derived from land-cover classes in this prototype, "
        "so the response is more meaningful than the absolute case count. "
        "See 'How this prototype works'."
    ),
)
_render_validation_caption(hs3, "preventable_mh_cases", _validation_scenario_context)
hs3.caption("cases prevented" if _mh_cases >= 0 else "cases induced")
if _mh_cost >= _MH_COST_PILL_EPSILON:
    _mh_cost_label = "Avoided MH Costs"
    _mh_cost_value = f'{_fmt_usd(_mh_cost)}/yr'
    _mh_cost_delta = f"+{_fmt_usd(_mh_cost)}/yr avoided"
    _mh_cost_color = "normal"
elif _mh_cost <= -_MH_COST_PILL_EPSILON:
    _mh_cost_label = "Added MH Costs"
    _mh_cost_value = f'{_fmt_usd(abs(_mh_cost))}/yr'
    _mh_cost_delta = f"+{_fmt_usd(abs(_mh_cost))}/yr added in costs"
    _mh_cost_color = "inverse"
else:
    _mh_cost_label = "Avoided MH Costs"
    _mh_cost_value = f'{_fmt_usd(_mh_cost)}/yr'
    _mh_cost_delta = None
    _mh_cost_color = "off"
hs4.metric(
    _mh_cost_label,
    _mh_cost_value,
    delta=_mh_cost_delta,
    delta_color=_mh_cost_color,
    help=(
        "Estimated avoided healthcare costs from preventable mental-health "
        "cases. Uses per-case cost assumptions for depression and anxiety. "
        "Treat as order-of-magnitude, not a budget estimate. "
        "See 'How this prototype works'."
    ),
)
_render_validation_caption(hs4, "avoided_mh_cost_usd", _validation_scenario_context, explicit_status="aligned_method")
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
        "Prototype food-yield estimate for newly converted food-forest pixels "
        "only. Uses a yield benchmark rather than site-specific agronomic "
        "modeling. Treat as directional. See 'How this prototype works'."
    )
)
_render_validation_caption(econ1, "food_mln_lbs", _validation_scenario_context)
if results['food_mln_lbs'] == 0:
    econ1.caption(
        "No food forest in this scenario — add Food Forest % to see production estimates."
    )
econ2.metric(
    "Est. Implementation Cost",
    _fmt_usd(results['total_cost_mln'] * 1e6),
    delta=None,
    help="Order-of-magnitude implementation cost: converted acres × user-set $/acre assumptions. Adjust costs in the sidebar. Not a site-specific bid or budget estimate. See 'How this prototype works'."
)
_render_validation_caption(econ2, "total_cost_mln", _validation_scenario_context, explicit_status="prototype")

# Row 2: the three model-derived dollar metrics (each computed downstream
# from the scenario, not directly from the user's sliders).
_flood_damage_avoided = results.get('flood_damage_avoided_usd', 0.0)
# Flood Damage Avoided is a CONDITIONAL capability. The gate is TABLE-PRESENCE,
# never the computed value: TOTAL_POTENTIAL_DAMAGE_USD > 0 is the single signal
# for "this city has per-building types AND a damage-valuation table". A city
# WITH a table that genuinely produced $0 avoided damage still renders the card
# (with its "no avoided damage" caveat). When NO table is loaded, the card is
# hidden entirely — no empty "—" slot — and a compact, visible unavailable note
# takes its place (the dash on an absent table read as a broken card, not a
# scoped capability).
_show_flood_damage = (BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES
                      and TOTAL_POTENTIAL_DAMAGE_USD > 0)
if not _show_flood_damage:
    # Card is hidden (not blanked) for cities without a damage-valuation table;
    # the explanation lives in the "How this prototype works" methodology
    # expander (Conditional outputs), not as an in-dashboard note drawing the
    # eye to the absence. Layout-only branch here.
    econ4, econ5 = st.columns(2)
else:
    econ3, econ4, econ5 = st.columns(3)
    _n_typed_buildings = int(np.sum(BUILDINGS_TYPE_RASTER > 0))
    econ3.metric(
        "Flood Damage Avoided",
        _fmt_usd(_flood_damage_avoided),
        delta=(
            f"+{_fmt_usd(_flood_damage_avoided)} vs baseline"
            if _flood_damage_avoided >= 1e4 else "no avoided damage"
        ),
        delta_color="normal" if _flood_damage_avoided >= 1e4 else "off",
        help=(
            "Shown only where a city-specific flood damage valuation table is "
            "available. If unavailable, use Flood Index and Runoff Volume for "
            "the hydrologic signal. See 'How this prototype works'."
        ),
    )
    _render_validation_caption(econ3, "flood_damage_avoided_usd", _validation_scenario_context, explicit_status="aligned_method")

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
    _help_text = (
        "Estimated avoided air-conditioning costs from modeled cooling effects "
        "over typed building pixels. Conservative lower-bound estimate; "
        "sensitive to building-type coverage. See 'How this prototype works'."
    )
    econ4.metric(
        "Cooling Energy Savings",
        f"{_fmt_usd(_energy_savings)}/yr",
        delta=(
            f"+{_fmt_usd(_energy_savings)}/yr vs baseline"
            if _energy_savings >= 1e3 else "no avoided energy cost"
        ),
        delta_color="normal" if _energy_savings >= 1e3 else "off",
        help=_help_text,
    )
    _render_validation_caption(econ4, "cooling_energy_savings_usd", _validation_scenario_context, explicit_status="aligned_method")
    # Per-pixel rate as a small secondary caption — only when the city
    # total is meaningful. Suppresses at HD-only scenarios where there's
    # no cooling delta to amortize.
    _rate_str = _fmt_per_pixel_rate(_per_pixel_cooling_usd)
    if _rate_str is not None and _energy_savings >= 1e3:
        econ4.caption(_rate_str)
else:
    if BUILDINGS_DATA_AVAILABLE and not BUILDINGS_HAVE_TYPES:
        _help_text = (
            "Not available for this extent — building-type data is required to "
            "estimate cooling energy savings. See 'How this prototype works'."
        )
    else:
        _help_text = (
            "Not available — cooling inputs (ET raster, energy table, or "
            "buildings) are not loaded for this city. "
            "See 'How this prototype works'."
        )
    econ4.metric("Cooling Energy Savings", "—", help=_help_text)
    _render_validation_caption(econ4, "cooling_energy_savings_usd", _validation_scenario_context, explicit_status="aligned_method")

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
        "One-time dollar value of carbon stock change using a "
        "social-cost-of-carbon assumption. Not an annual benefit flow. "
        "See 'How this prototype works'."
    )
    if _CARBON_IS_STOCK else
    (
        "Annual dollar value of carbon sequestration using a "
        "social-cost-of-carbon assumption. See 'How this prototype works'."
    )
)
econ5.metric(
    _carbon_dollar_label,
    _carbon_dollar_value,
    delta=_carbon_dollar_delta,
    delta_color=_carbon_dollar_color,
    help=_carbon_dollar_help,
)
_render_validation_caption(econ5, "carbon_value_usd", _validation_scenario_context, explicit_status="natcap_published" if _CARBON_IS_STOCK else "prototype")

# ── Region-Local View (Region-Local Metrics Commit 2) ─────────────────────────
# Region-clipped readings paired with the existing citywide cards above. Only
# renders for region scenarios — the citywide cards are sufficient on their own
# when no region is selected. Honesty contract from REGION_LOCAL_METRICS_SPEC.md:
# never show a region-local number without its citywide companion — that's
# what the side-by-side rows enforce.
#
# Map-click stability fix (post-multi-select RELAY): this section is gated on
# the active main_tab being 'Scenario'. Page-root content like this used to
# emit unconditionally and appear above the tabs UI on every rerun. Clicking
# a district on the Map View tab triggers a rerun, and when the first
# clicked district populated _region_local for the first time, this section
# appeared at page-root and grew the page above the tabs — the browser
# preserves scroll position by pixel offset, so the map shifted out of the
# viewport while the new impact section landed where the user was looking.
# Confining the section to the Scenario tab keeps the Map View page-root
# unchanged across map-click reruns: scroll position stays on the map so
# the user can click another district without scrolling back.
_region_local = results.get('region_local')
# Read the persisted main_tab from session_state (the segmented_control at
# the bottom of the page hasn't been defined yet at this point in top-to-
# bottom execution — the widget owns its key, so the keyed value is what
# the next render will use). Default 'Scenario' matches the control's
# default so the first-ever run also lands correctly.
if st.session_state.get('main_tab', 'Scenario') == 'Scenario' and _region_local:
    st.divider()
    _rs = results.get('region_selection') or {}
    _rs_layer = _rs.get('layer')
    _rs_ids = _rs.get('selected_ids') or []
    _rs_display = (
        _CURRENT_CITY_STATE.region_layer_display_names.get(_rs_layer, "region")
        if _rs_layer else "selected region"
    )
    _rs_plural = "s" if len(_rs_ids) != 1 else ""
    _rs_label = f"{_rs_display}{_rs_plural} {', '.join(_rs_ids)}" if _rs_ids else _rs_display
    st.markdown(f"#### Selected-region impact — {_rs_label}")
    # Relay 2 #4 — short locked visible caption. Region-local summary +
    # reach-effect honesty (cooling / nature access / mental-health
    # exposure may extend beyond the selected boundary). The
    # flood-routing and food/cost/carbon-matching detail moves below
    # into expanders.
    st.caption(
        "Region-local values summarize the selected area; citywide "
        "values show the system-level result. Reach effects for "
        "cooling, nature access, and mental-health exposure may "
        "extend beyond the selected boundary."
    )

    # Locked per-metric display rows. Each row pulls citywide from `results`,
    # region from `_region_local`, and formats both with the same helpers the
    # citywide cards use so the numbers are apples-to-apples. Order matches the
    # Ecological → Human & Social → Economic flow above.
    def _fmt_co2(t):  return f"{t:+,.0f} t CO2e" if t is not None else "—"
    def _fmt_money(d): return _fmt_usd(d) if d is not None else "—"
    def _fmt_pct(p):   return f"{p:.1f}%" if p is not None else "—"
    def _fmt_pp(n):    return f"{int(n):,} people" if n is not None else "—"
    def _fmt_cases(n): return f"{n:.0f} cases" if n is not None else "—"
    def _fmt_cost(m):  return _fmt_usd(m * 1e6) if m is not None else "—"
    # UI feedback #6 — when the active city has no flood-damage valuation
    # method (SA's case: no damage_table_file → TOTAL_POTENTIAL_DAMAGE_USD
    # == 0), don't render "$0" — render the n/a sentinel so the dashboard
    # doesn't claim a precise dollar figure where none can be computed.
    # MN has the InVEST UFR damage table and renders the real dollar value.
    def _fmt_flood_dmg(d):
        if TOTAL_POTENTIAL_DAMAGE_USD <= 0:
            return "n/a — no damage valuation available"
        return _fmt_money(d)

    # UI feedback #5 — carbon row label matches per-city method:
    # SA = four-pool stock change (one-time, t CO2e), MN = annual
    # sequestration flow (t CO2e/yr). Same `_CARBON_IS_STOCK` switch
    # the main carbon card uses.
    _rl_carbon_label = ('Carbon Storage Change' if _CARBON_IS_STOCK
                        else 'Carbon Sequestration')
    _rl_rows = [
        ("Flood Index",              f"{_region_local['flood_reduction']:.1f}",                       f"{results['flood_reduction']:.1f}"),
        ("Temp change",              _fmt_temp_change(_region_local['temp_change_f']),                _fmt_temp_change(results['temp_change_f'])),
        ("Runoff Volume",            _fmt_runoff(_region_local['runoff_acre_feet']),                  _fmt_runoff(results['runoff_acre_feet'])),
        ("Runoff Retention",         f"{_region_local['runoff_retention_idx'] * 100:.1f}%",           f"{results['runoff_retention_idx'] * 100:.1f}%"),
        ("Mean NDVI",                f"{_region_local['mean_ndvi']:.3f}",                             f"{results['mean_ndvi']:.3f}"),
        (_rl_carbon_label,           _fmt_co2(_region_local['carbon_tons_co2']),                      _fmt_co2(results['carbon_tons_co2'])),
        ("Food Production",          _fmt_food(_region_local['food_mln_lbs']),                       _fmt_food(results['food_mln_lbs'])),
        ("Cost",                     _fmt_cost(_region_local['total_cost_mln']),                      _fmt_cost(results['total_cost_mln'])),
        ("Cooling Energy Savings",   _fmt_money(_region_local['cooling_energy_savings_usd']),        _fmt_money(results['cooling_energy_savings_usd'])),
        ("Flood Damage Avoided",     _fmt_flood_dmg(_region_local['flood_damage_avoided_usd']),       _fmt_flood_dmg(results['flood_damage_avoided_usd'])),
        ("Nature Access",            _fmt_pct(_region_local['nature_access_pct']),                   _fmt_pct(results['nature_access_pct'])),
        ("People with Nature Access", _fmt_pp(_region_local['people_with_nature_access']),           _fmt_pp(results['people_with_nature_access'])),
        ("Children's Nature Access",
            _fmt_pct(_region_local['children_nature_access_pct'])
                if _region_local.get('children_nature_access_pct') is not None else "—",
            _fmt_pct(results['children_nature_access_pct'])
                if results.get('children_nature_access_pct') is not None else "—"),
        ("Preventable MH Cases",     _fmt_cases(_region_local['preventable_mh_cases']),              _fmt_cases(results['preventable_mh_cases'])),
        ("Avoided MH Cost",          _fmt_money(_region_local['avoided_mh_cost_usd']),               _fmt_money(results['avoided_mh_cost_usd'])),
    ]
    # No damage-valuation table (SA): curate the Flood Damage Avoided row out of
    # the comparison table, the same way the Economic cards and the
    # Baseline-vs-Scenario audit expander already omit it. Damage-table cities
    # (MN) keep the row with its real value. The methodology expander explains
    # the absence; the engine still computes flood_damage_avoided_usd in
    # `results`, so the underlying record stays complete.
    if TOTAL_POTENTIAL_DAMAGE_USD <= 0:
        _rl_rows = [r for r in _rl_rows if r[0] != "Flood Damage Avoided"]
    _rl_df = pd.DataFrame(_rl_rows, columns=["Metric", f"Region ({_rs_label})", "Citywide"])
    st.dataframe(_rl_df, width="stretch", hide_index=True)

    # Population-allocation honesty caveat (mechanics layer). The population
    # rows above (People with Nature Access, Children's Nature Access) count
    # residents allocated from Census 2020 blocks across each block's AREA, so
    # a small count on institutional land (e.g. school parcels) reflects that
    # area-spread allocation, not on-site residents. Shown only when an
    # ownership filter narrows the selection to a class — that's the count this
    # caveat is about. Expander, not headline; no badge vocab.
    if _normalize_ownership_filter(results.get('ownership_filter')) is not None:
        with st.expander("ⓘ How these population counts are allocated", expanded=False):
            st.caption(
                "Population is allocated from census blocks across each block's "
                "area, so small counts on institutional land (e.g. school "
                "parcels) reflect that allocation, not on-site residents."
            )

    # Honesty-Surface Pass Commit 1 — make the validation-state inheritance
    # explicit. The Region-Local table has no per-row validation badge
    # because region-local is a second aggregation of the same per-pixel
    # outputs; each row's validation state is identical to the per-metric
    # badge on the citywide card above. State the inheritance so users
    # don't read the absence as "no validation state set".
    st.caption(
        "Validation states for these rows inherit from the per-metric badges "
        "on the citywide cards above — region-local doesn't change the InVEST-aligned evaluator, "
        "only the aggregation scope. The pairing keeps it honest: never read a "
        "region-local number without its citywide companion."
    )
    # UI-Text Pass — food/cost/carbon equality note. Direct conversion metrics
    # sum identical per-pixel quantities whether tallied citywide or only over
    # the region, so they read equal under the locked clip-clean treatment.
    # Framed as an explanation, not a caveat.
    with st.expander(
        "ⓘ Why region and citywide totals can match for food / cost / carbon",
        expanded=False,
    ):
        st.caption(
            "For direct conversion metrics (food production, cost, carbon), "
            "region totals equal citywide totals when all converted pixels are "
            "inside the selected region — this matches the locked clip-clean "
            "treatment. The equality is correct, not a rounding artifact."
        )

    # Relay 2 #4 — flood-routing detail moved into an expander; the
    # reach-effect caveat moved into the short visible caption at the top
    # of the Selected-region impact section (above the table).
    with st.expander(
        "Flood routing — why region and citywide flood values differ",
        expanded=False,
    ):
        st.caption(
            "The flood metrics are a closed-form SCS-CN volume derived "
            "from the region's mean curve number scaled to its developed "
            "area — a regional rate, NOT a per-pixel sum, and NOT routed "
            "hydrology. Because the metric is mean-CN, the score depends "
            "only on the MIX of land covers in scope, not their placement: "
            "two equal-area interventions yield the same flood score whether "
            "sited at the top of a watershed or the bottom of a concrete "
            "basin — elevation and flow position are not modeled. "
            "Region values legitimately differ from citywide "
            "because the mean CN and developed area are computed over a "
            "smaller pixel set; they don't measure flood protection "
            "*delivered to* the region."
        )

st.divider()

ce = compute_cost_effectiveness(results, BASELINE_RUNOFF_ACRE_FEET)
st.markdown("#### Cost Effectiveness")
st.caption(
    "Screening ratios: implementation cost per unit of positive benefit, using "
    "the \\$/acre assumptions in the sidebar — for comparing scenarios, not "
    "budgeting (lower is better). Ratios appear only for metrics the scenario "
    "meaningfully improves. Metrics that don't improve — including negative "
    "impacts — stay visible in the outcome cards above."
)
# Render a ratio card only where the ratio is meaningful. compute_cost_effectiveness
# returns None for zero/negative/below-epsilon denominators (incl. warming, which
# yields no cooling ratio); an "N/A" card on the dashboard reads as broken, so we
# hide it instead. The outcome cards above are a separate, unconditional section
# that still shows good/zero/negative impacts — so hiding a ratio never hides bad
# news. Tables/CSV/audit/optimizer keep _fmt_ce → "N/A" for stable columns.
_CE_CARD_SPECS = [
    ("Cost / ac-ft runoff", "cost_per_acft",
     "Implementation cost divided by acre-feet of runoff reduced versus "
     "baseline. Lower is better. See 'How this prototype works'."),
    ("Cost / °F cooling", "cost_per_degf",
     "Implementation cost divided by city-average °F cooling versus baseline. "
     "Lower is better. See 'How this prototype works'."),
    ("Cost / 1k people fed", "cost_per_1k_people",
     "Implementation cost divided by estimated people fed, in thousands. "
     "Lower is better. See 'How this prototype works'."),
]
_ce_cards = [(lbl, ce[key], hlp, key) for (lbl, key, hlp) in _CE_CARD_SPECS
             if ce[key] is not None]
if _ce_cards:
    _ce_cols = st.columns(len(_ce_cards))
    for _col, (_lbl, _val, _hlp, _key) in zip(_ce_cols, _ce_cards):
        _col.metric(_lbl, _fmt_ce(_val), delta=None, help=_hlp)
        _render_validation_caption(_col, _key, _validation_scenario_context,
                                   explicit_status="prototype")
else:
    st.caption(
        "No cost-effectiveness ratios for this scenario — it doesn't meaningfully "
        "improve runoff, cooling, or food production. Its impacts are in the "
        "outcome cards above."
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
    _carbon_dollar_period = '' if _CARBON_IS_STOCK else '/yr'
    # Per-city flood-damage rendering. Only cities with a damage table (MN)
    # get a Flood Damage Avoided row; cities without (SA) omit it — the CN
    # index is NOT a flood-volume percent, and Runoff Volume already carries
    # the physical signal. (Old behavior relabeled the row "Flood Volume
    # Reduction" and showed the unitless index as a %, which is removed.) The
    # row is gated into all four parallel lists below via `_flood_damage_monetized`.
    _flood_damage_monetized = (
        BUILDINGS_DATA_AVAILABLE and BUILDINGS_HAVE_TYPES and TOTAL_POTENTIAL_DAMAGE_USD > 0
    )
    if _flood_damage_monetized:
        _flood_label_table = 'Flood Damage Avoided'
        _flood_baseline_table = '$0'
        _flood_scenario_table = _fmt_usd(_flood_damage_avoided)
        _flood_change_table = (
            f'+{_fmt_usd(_flood_damage_avoided)}'
            if _flood_damage_avoided >= 1e4 else '$0'
        )
    comparison_data = {
        'Metric': [
            'Flood Index', 'Runoff Volume', 'Runoff Retention', 'Temperature Change',
            'Food Production', _carbon_metric_label, 'NDVI',
            *([_flood_label_table] if _flood_damage_monetized else []), 'Cooling Energy Savings', _carbon_dollar_label_table,
        ],
        'Baseline': [
            f'{_baseline_flood:.1f}',
            f'{BASELINE_RUNOFF_ACRE_FEET:,.0f} ac-ft',
            (f'{BASELINE_RUNOFF_RETENTION_IDX * 100:.1f}%' if BASELINE_RUNOFF_RETENTION_IDX is not None else '—'),
            'Reference',
            '0 lbs',
            f'0 {_carbon_unit_suffix}',
            f'{BASELINE_NDVI:.3f}',
            *([_flood_baseline_table] if _flood_damage_monetized else []),
            '$0/yr',
            f'$0{_carbon_dollar_period}',
        ],
        'This Scenario': [
            f'{results["flood_reduction"]:.1f}',
            f'{results["runoff_acre_feet"]:,.0f} ac-ft',
            f'{results["runoff_retention_idx"] * 100:.1f}%',
            _temp_change_label,
            f'{results["food_mln_lbs"] * 1e6:,.0f} lbs/yr',
            f'{_carbon_tons_table:,.0f} {_carbon_unit_suffix}',
            f'{results["mean_ndvi"]:.3f}',
            *([_flood_scenario_table] if _flood_damage_monetized else []),
            f'{_fmt_usd(_energy_savings_table)}/yr',
            f'${_carbon_value_table / 1e6:.2f}M{_carbon_dollar_period}' if abs(_carbon_value_table) >= 1e4 else f'${_carbon_value_table:,.0f}{_carbon_dollar_period}',
        ],
        'Change': [
            f'{_flood_diff:+.1f}',
            (
                f'+{_runoff_diff:,.0f} ac-ft'         if _runoff_diff > 0
                else f'{abs(_runoff_diff):,.0f} ac-ft prevented' if _runoff_diff < 0
                else '0 ac-ft'
            ),
            (f'{(results["runoff_retention_idx"] - BASELINE_RUNOFF_RETENTION_IDX) * 100:+.1f} pp'
             if BASELINE_RUNOFF_RETENTION_IDX is not None else '—'),
            _temp_change_label,
            f'+{results["food_mln_lbs"] * 1e6:,.0f} lbs/yr',
            f'{_carbon_tons_table:+,.0f} {_carbon_unit_suffix}',
            f'{results["mean_ndvi"] - BASELINE_NDVI:+.3f}',
            *([_flood_change_table] if _flood_damage_monetized else []),
            f'+{_fmt_usd(_energy_savings_table)}/yr' if _energy_savings_table >= 1e3 else '$0/yr',
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
    st.caption("Detailed modeling assumptions, caveats, and method notes.")
    if selected_city.startswith("San Antonio"):
        st.info(
            "**SA Land Cover:** Using NatCap's compound NLCD×NLUD×tree-canopy "
            "LULC framework (1,984 compound lucodes; foundational adoption "
            "landed Brief 27). UCM, UNA, and Carbon all consume the "
            "compound-keyed biophysical tables directly (Briefs 28b, 29, 30). "
            "See `docs/archive/SA_INTEGRATION_PLAN_2026-05.md` for the brief sequence."
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
            "REFERENCE.md's Flood Index section for the relationship."
        )
    with _assumption_tabs[1]:
        _temp_calibration = (
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per HMI unit. "
            f"Values come from the InVEST UCM args JSON for the Minneapolis AOI "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`, humid continental Köppen Dfa). "
            "Treat the °F output as ±2 °F at best. This reflects HMI-to-temperature "
            "calibration accuracy, not the machine-learning estimate range.\n"
            if selected_city.startswith("Minneapolis") else
            f"- **Calibration:** {HM_TO_FAHRENHEIT:.2f} °F per HMI unit. "
            f"No published InVEST args exist for hot semi-arid Köppen BSh; "
            f"values are an estimate from regional UHI literature "
            f"(`uhi_max = {UHI_MAX_C:.2f} °C`). "
            "Treat the °F output as ±2 °F at best — calibration uncertainty is "
            "larger here than for MN. This reflects HMI-to-temperature calibration "
            "accuracy, not the machine-learning estimate range.\n"
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
            "- **Not in the machine-learning model.** UMH outputs are computed "
            "deterministically inside `evaluate_scenario` from the scenario's "
            "NDVI exposure — the model doesn't need to predict them. They "
            "appear in the precomputed grid columns alongside the other model "
            "targets, but are recomputed live for any scenario the optimizer surfaces."
        )
    with _assumption_tabs[5]:
        st.markdown(
            "- **Order-of-magnitude only:** total cost = "
            "`\\$/acre slider × converted acres`, summed across green "
            "infrastructure, food forest, and high-density development. "
            "Default \\$/acre ranges come from broad planning literature, not "
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
            "- **Suggested scenarios** come from the fast machine-learning model. "
            "Verify any suggestion by manually applying it to the main "
            "sliders so the InVEST-aligned evaluator runs."
        )

st.divider()

_MAIN_TAB_NAMES = ["Scenario", "Tradeoffs", "Map View", "NatCap Reference"]
# Seed the widget's session_state key once instead of passing default= — the
# optimize branches write st.session_state['main_tab'] = "Tradeoffs" before this
# widget instantiates, and a default= alongside a pre-existing key value makes
# Streamlit emit a "default will be ignored" warning. Seeding first-load only
# (Scenario) leaves the post-optimize Tradeoffs write authoritative.
if "main_tab" not in st.session_state:
    st.session_state["main_tab"] = _MAIN_TAB_NAMES[0]
_main_tab = st.segmented_control(
    "Main view",
    options=_MAIN_TAB_NAMES,
    label_visibility="collapsed",
    key="main_tab",
)
# Persisted across reruns via the `main_tab` widget key — region/map-click
# reruns no longer snap back to Scenario. Tab containers always rendered;
# only the active tab's `with` block runs, so inactive tabs are zero-cost.
tab1 = st.container()
tab2 = st.container()
tab3 = st.container()
tab4 = st.container()

if _main_tab == 'Scenario':
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
                f'{_carbon_title} ({_carbon_unit_suffix})\n(higher = more carbon stored)'
                if _CARBON_IS_STOCK
                else f'{_carbon_title} ({_carbon_unit_suffix})\n(higher = more sequestration)'
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

if _main_tab == 'Tradeoffs':
    with tab2:
        # Brief A.3: filter saved scenarios to the active city. The .get("city",
        # selected_city) default is backward-compatible — in-memory saves from
        # before A.3 lacked the `city` key; treat them as belonging to the
        # current city rather than orphaning them.
        _saved_for_city = [
            s for s in st.session_state.saved_scenarios
            if s.get("city", selected_city) == selected_city
        ]

        # ── Tradeoff Space (mode-aware: SELECTED-AREA vs CITYWIDE) ──
        # Mode switch driven by selection state:
        #   (1) region selected (selected_region_mask not None)
        #       → SELECTED-AREA primary (region-local axes, engine-verified
        #         tested mixes) + CITYWIDE in an expander with clustering caveat.
        #   (2) ownership-only (ownership_mask but no region_mask)
        #       → CITYWIDE primary with clustering caveat above.
        #   (3) neither → CITYWIDE primary, no caveat.
        # In all modes the citywide scatter still uses plot_tradeoff
        # unchanged (refs + surrogate diamonds + saves + current star).
        _region_mask_active = st.session_state.get('selected_region_mask') is not None
        _ownership_only = (not _region_mask_active) and (
            st.session_state.get('selected_ownership_mask') is not None
        )
        # Citywide optimizer is stale when ANY filter is active (citywide-
        # scoped predicted values don't reflect the active mask). Skip
        # surrogate diamonds in those modes.
        if _filter_active:
            _opt_for_chart = None
        else:
            _opt_for_chart = st.session_state.optimized_results
            if not isinstance(_opt_for_chart, pd.DataFrame):
                _opt_for_chart = None

        if _region_mask_active:
            # ── SELECTED-AREA mode (region-local primary) ───────────────
            st.subheader("Selected-area tradeoff space")
            # State-aware caption: before a search the plot holds only the
            # current scenario star + the region baseline (no searched points),
            # so the pre-search copy describes those rather than the searched
            # mixes. Both branches keep the not-comparable-to-citywide-NatCap
            # caveat visible.
            _region_opt_for_caption = st.session_state.get('region_optimized_results')
            _has_searched_mixes = (
                isinstance(_region_opt_for_caption, pd.DataFrame)
                and not _region_opt_for_caption.empty
            )
            if _has_searched_mixes:
                st.caption(
                    "Each point is a tested mix evaluated **inside the current "
                    "selected region and conversion filters**. Axes use "
                    "selected-area outcomes, so points are not directly "
                    "comparable to citywide NatCap reference scenarios."
                )
            else:
                st.caption(
                    "Your current scenario (★) and the region baseline. Run a "
                    "search to populate candidate mixes. Axes use selected-area "
                    "outcomes, so points are not directly comparable to citywide "
                    "NatCap reference scenarios."
                )
            # Compute region-local baseline HMI (flood baseline = 0 by
            # definition — flood_reduction is delta vs baseline). Mean
            # of the per-pixel baseline HMI raster restricted to the
            # active region∩ownership mask. Skip NaN cells.
            _rmask = st.session_state.get('selected_region_mask')
            _omask = st.session_state.get('selected_ownership_mask')
            _eff_mask = _rmask if _omask is None else (_rmask & _omask)
            try:
                _base_hm = _BASELINE_HM_RASTER
                _valid = (~np.isnan(_base_hm)) & _eff_mask
                _baseline_hm_region = (
                    float(_base_hm[_valid].mean()) if _valid.any() else None
                )
            except Exception:
                _baseline_hm_region = None
            # Region-optimizer tested mixes (engine-verified region-local).
            _region_opt = st.session_state.get('region_optimized_results')
            if not isinstance(_region_opt, pd.DataFrame):
                _region_opt = None
            _region_fig = plot_tradeoff_region(
                results, _region_opt, _baseline_hm_region,
            )
            # Render with on_select='rerun' so a click on a tested-mix
            # marker fires a rerun and the click handler below applies it.
            _region_chart_event = st.plotly_chart(
                _region_fig, use_container_width=True,
                on_select='rerun', selection_mode='points',
                key='region_tradeoff_picker',
            )
            # ── Click-to-apply handler ──
            # Signature de-dup: forward only when the event's clicked-
            # customdata signature differs from the last-applied signature.
            # Without this guard, any unrelated rerun (slider change, etc.)
            # would re-apply the last-clicked mix and silently clobber
            # hand-tuned slider values. Mirrors the district-selector
            # pattern (region_map_picker_last_sig).
            if _region_chart_event:
                _ev = (_region_chart_event if isinstance(_region_chart_event, dict)
                       else dict(_region_chart_event))
                _pts = (_ev.get('selection') or {}).get('points') or []
                _clicked_indices = [
                    p.get('customdata') for p in _pts
                    if isinstance(p.get('customdata'), (int, np.integer))
                ]
                if _clicked_indices and _region_opt is not None:
                    _evt_sig = tuple(sorted(int(i) for i in _clicked_indices))
                    _last_sig = tuple(
                        st.session_state.get('region_tradeoff_last_apply_sig') or ()
                    )
                    if _evt_sig != _last_sig:
                        _apply_idx = int(_clicked_indices[0])
                        if 0 <= _apply_idx < len(_region_opt):
                            _apply_region_optimizer_mix(
                                _region_opt.iloc[_apply_idx], _apply_idx,
                            )
                            st.session_state['region_tradeoff_last_apply_sig'] = list(_evt_sig)
                            st.rerun()
            # Mis-click guard — revert button if a previous scenario was
            # stashed by _apply_region_optimizer_mix.
            _prev = st.session_state.get('_region_apply_prev')
            if _prev and any(_prev.get(k) is not None for k in ('pct', 'gi', 'ff')):
                _rev_col1, _rev_col2 = st.columns([1, 5])
                with _rev_col1:
                    if st.button("↶ Revert to previous scenario",
                                 key="region_tradeoff_revert"):
                        # Restore via _pending_* so the slider session_state
                        # is overwritten on the next rerun (same mechanism
                        # the Quick Start presets use).
                        if _prev.get('pct') is not None:
                            st.session_state._pending_pct = int(_prev['pct'])
                        if _prev.get('gi') is not None:
                            st.session_state._pending_gi = int(_prev['gi'])
                        if _prev.get('ff') is not None:
                            st.session_state._pending_ff = int(_prev['ff'])
                        # Clear apply state and the stash so the button
                        # doesn't re-render on subsequent reruns.
                        st.session_state.applied_from_region_optimizer = False
                        st.session_state.applied_suggestion = None
                        st.session_state['_applied_region_optimizer_values'] = None
                        st.session_state['_region_apply_prev'] = None
                        st.session_state['region_tradeoff_last_apply_sig'] = None
                        st.rerun()

            # ── Citywide context expander ──
            with st.expander("Citywide context", expanded=False):
                st.caption(
                    "Shows whole-area impacts. Region-constrained scenarios "
                    "may cluster because only a small share of the city changes. "
                    "Full-evaluator region-optimizer mixes appear on the "
                    "Selected-area scatter above this expander; they are not "
                    "overlaid on this citywide view, so the machine-learning citywide "
                    "diamonds (fast estimates) and the region InVEST-aligned evaluator "
                    "squares stay visually distinct."
                )
                st.plotly_chart(plot_tradeoff(
                    results, scenario_df,
                    lookup_table=lookup_table,
                    saved=_saved_for_city,
                    optimized=_opt_for_chart,
                ), use_container_width=True)
        else:
            # ── CITYWIDE mode (unchanged scatter; conditional caveat) ────
            st.subheader("Citywide tradeoff space")
            if _ownership_only:
                # Caveat: ownership-only constraints still shrink the
                # placement pool, so scenarios cluster on the chart.
                st.caption(
                    "Shows whole-area impacts. Region-constrained scenarios "
                    "may cluster because only a small share of the city changes."
                )
            else:
                st.caption(
                    "Each point is a scenario. Better outcomes are toward the "
                    "**top-right** — both axes are higher-is-better (Flood Index "
                    "on x, Heat Mitigation Index on y). The **purple star** is your "
                    "current scenario; **orange diamonds** are citywide machine-learning "
                    "suggestions, shown as fast estimates with calibrated estimate "
                    "ranges. Bubble size shows food production for "
                    "saved and optimizer points. Applied scenarios and selected-area "
                    "results are evaluator-computed, so they carry no range."
                )
            st.plotly_chart(plot_tradeoff(
                results, scenario_df,
                lookup_table=lookup_table,
                saved=_saved_for_city,
                optimized=_opt_for_chart,
            ), use_container_width=True)
            if _opt_for_chart is not None and len(_opt_for_chart):
                st.caption(
                    "Orange diamonds are citywide machine-learning suggestions; "
                    "hover one for its calibrated estimate range. Applied scenarios "
                    "and selected-area results are evaluator-computed, so they carry "
                    "no range."
                )

        st.divider()

        # ── Cross-source comparison table (Brief #5) ──
        # Always shows the active scenario as a row (marked ▶ Current), plus
        # NatCap fixed scenarios as anchor rows on SA, plus any saved scenarios
        # for the active city. Source / Validation columns drive off per-row
        # provenance (Brief #3 wording). MN currently has no NatCap anchors;
        # the rest of the table works unchanged. Flood is intentionally excluded
        # (different derivations between baseline and NatCap alternatives —
        # the per-scenario flood card is the right place for that). Carbon $
        # column is labeled (derived) on every row because it's the prototype's
        # own NatCap-carbon × EPA SC-CO2 multiplication, not itself a NatCap-
        # published dollar value.
        # UI-Text Pass — adaptive title. SA always has NatCap anchor rows so the
        # table is genuinely a comparison; MN has anchors only when the user has
        # saved scenarios. With no anchors and no saves the table is a single-row
        # summary, so "Compare scenarios" overpromises.
        _has_comparison_rows = (
            selected_city.startswith("San Antonio") or bool(_saved_for_city)
        )
        # Adaptive title: under an active region/ownership filter the current-row
        # values reflect the filter scope, but NatCap fixed scenarios and saved
        # scenarios were computed under their own scopes (filter-time for saves;
        # no-filter for NatCap anchors). Surface that in the title so the user
        # doesn't read across rows as same-scope numbers.
        if _has_comparison_rows:
            st.markdown("#### Compare scenarios")
            st.caption(
                ("NatCap-published reference scenarios, t"
                 if selected_city.startswith("San Antonio") else "T")
                + "he current scenario, and any you've saved — side by side. "
                "**Source** says where the value comes from; **Validation** says how "
                "it's grounded. Different sources are not directly comparable as "
                "precision numbers; the columns make the difference visible."
                + (" The current row updates with your active region and "
                   "ownership filters; NatCap reference and saved rows stay "
                   "anchored to their original scope." if _filter_active else "")
            )
        else:
            st.markdown("#### Current scenario summary")
            st.caption(
                "Just the current scenario for now — save scenarios from below to "
                "build up a side-by-side comparison. **Source** and **Validation** "
                "columns describe where each value comes from and how it's grounded."
            )

        def _cs_source_validation(prov):
            info = _PROVENANCE_HEADER_INFO.get(
                prov, ("Unknown", "provenance not recorded", "gray"))
            return info[0], info[1]

        # Short Validation cell labels. The full Brief #3 wording (kept in
        # `_PROVENANCE_HEADER_INFO`) is moved to the column-header tooltip via
        # column_config to keep the table from getting cramped.
        _CS_SHORT_VAL = {
            eib.PROVENANCE_BASELINE:     "evaluator-verified",
            eib.PROVENANCE_NATCAP_FIXED: "displayed (NatCap)",
            eib.PROVENANCE_EXPLORER:     "evaluator-verified",
            eib.PROVENANCE_OPTIMIZER:    "evaluator-verified",
            # Region-constrained optimizer (variant B). The displayed values are
            # engine-true region-local; the surrogate's role stopped at
            # shortlisting. Distinct from PROVENANCE_OPTIMIZER's
            # "evaluator-verified" (citywide).
            eib.PROVENANCE_REGION_OPTIMIZED: "evaluator-verified (region)",
        }
        def _cs_short_validation(prov):
            return _CS_SHORT_VAL.get(prov, "—")

        # UI feedback #5 — carbon column labels match per-city method AND
        # the canonical phrasing used by the main Economic metric card +
        # the Selected-region impact row, so the same quantity reads under
        # one label everywhere in the app (Batch 4 v2 #8 harmonization).
        # SA = four-pool stock change (one-time, t CO2e): "Carbon Storage
        # Change" + "Carbon Storage Value $". MN = annual sequestration
        # flow (t CO2e/yr): "Carbon Sequestration" + "Avoided Carbon Cost
        # $/yr".
        _CS_CARBON_TONS_LABEL    = ('Carbon Storage Change'
                                     if _CARBON_IS_STOCK else 'Carbon Sequestration')
        _CS_CARBON_TONS_UNIT     = ('t CO2e' if _CARBON_IS_STOCK else 't CO2e/yr')
        _CS_CARBON_DOLLAR_LABEL  = ('Carbon Storage Value $ (derived)'
                                     if _CARBON_IS_STOCK
                                     else 'Avoided Carbon Cost $/yr (derived)')
        _CS_CARBON_DOLLAR_PERIOD = '' if _CARBON_IS_STOCK else '/yr'

        def _cs_row_metrics(r):
            """Metric cells for a row drawn from a results-shaped dict (current
            or saved). Each cell returns "—" when the value is missing; 0 is
            a legitimate value and renders normally."""
            v_temp     = r.get('temp_change_f')
            v_carbon   = r.get('carbon_tons_co2')
            v_carbon_d = r.get('carbon_value_usd')
            v_cool     = r.get('cooling_energy_savings_usd')
            v_una      = r.get('nature_access_pct')
            v_food     = r.get('food_mln_lbs')
            v_mh       = r.get('preventable_mh_cases')
            v_cost     = r.get('total_cost_mln')
            # UI feedback #5 — carbon column labels match per-city method:
            # SA = four-pool stock change (one-time, t CO2e), MN = annual
            # sequestration flow (t CO2e/yr). The dollar column is
            # correspondingly "Carbon Storage Value $" (SA) vs "Avoided
            # Carbon Cost $/yr" (MN) — same vocabulary the Economic metric
            # card uses (`_carbon_dollar_label` at app.py:5969).
            return {
                "Temperature":              _fmt_temp_change(v_temp) if v_temp is not None else "—",
                _CS_CARBON_TONS_LABEL:      f"{v_carbon/1e6:+.2f}M {_CS_CARBON_TONS_UNIT}" if v_carbon is not None else "—",
                _CS_CARBON_DOLLAR_LABEL:    f"${v_carbon_d/1e6:+.0f}M{_CS_CARBON_DOLLAR_PERIOD}" if v_carbon_d is not None else "—",
                "Cooling Energy $":         f"${v_cool/1e6:.2f}M/yr"       if v_cool is not None else "—",
                "Nature Access %":          f"{v_una:.1f}%"                if v_una is not None else "—",
                "Food (M lbs)":             f"{v_food:.2f}"                if v_food is not None else "—",
                "MH cases":                 f"{int(v_mh):,}"               if v_mh is not None else "—",
                "Cost $M":                  f"${v_cost:.1f}M"              if v_cost is not None else "—",
            }

        # Scenario Record Pass — Area + Ownership columns compose at render via
        # the module-level _cs_area_for_row / _cs_ownership_for_row helpers
        # (extracted so the Scenario audit expander on tab1 and the CSV export
        # below can reuse the same composition rule).
        _cs_rows = []

        # ── 1. NatCap anchor rows (SA only) ──
        if selected_city.startswith("San Antonio"):
            _src_natcap = _cs_source_validation(eib.PROVENANCE_NATCAP_FIXED)[0]
            _val_natcap = _cs_short_validation(eib.PROVENANCE_NATCAP_FIXED)
            for _sid in ns.SA_NATCAP_FIXED_SCENARIOS.keys():
                _spec = ns.SA_NATCAP_FIXED_SCENARIOS[_sid]
                _, _bv_t_s, _dT_s = nv.published_delta(selected_city, _sid, "temp_change_f")
                _, _bv_c_s, _dC_s = nv.published_delta(selected_city, _sid, "carbon_tons_co2")
                if _sid == "baseline":
                    # Every other row in these three columns is Δ-vs-baseline.
                    # Show "baseline" here rather than absolutes so each column
                    # is on a single basis. NatCap's absolute citywide anchors
                    # (90.08 °F, 107.32M t CO2e, $20.39B) are surfaced in the
                    # Tab 4 reference view, not mixed into this Δ-basis table.
                    _t_str  = "baseline"
                    _c_str  = "baseline"
                    _cv_str = "baseline"
                else:
                    _t_str  = (f"{_fmt_temp_change(_dT_s)} ({_dT_s:+.3f} °F)"
                               if _dT_s is not None else "—")
                    _c_str  = (f"{_dC_s / 1e6:+.2f}M t CO2e"
                               if _dC_s is not None else "—")
                    _cv_str = (f"${_dC_s * EPA_SOCIAL_COST_CARBON / 1e6:+.0f}M"
                               if _dC_s is not None else "—")
                _cs_rows.append({
                    "Scenario":                 _spec["label"],
                    "Source":                   _src_natcap,
                    "Validation":               _val_natcap,
                    "Area":                     "Citywide",
                    "Ownership":                "None",
                    "Temperature":              _t_str,
                    _CS_CARBON_TONS_LABEL:      _c_str,
                    _CS_CARBON_DOLLAR_LABEL:    _cv_str,
                    "Cooling Energy $":         "—",
                    "Nature Access %":          "—",
                    "Food (M lbs)":             "—",
                    "MH cases":                 "—",
                    "Cost $M":                  "—",
                })

        # ── 2. Current scenario row ──
        # Provenance detection mirrors the Brief #3 main-panel header at
        # _scen_provenance below. (Re-derived locally here because tab2 runs on
        # every rerun regardless of which tab is visible — we need a fresh read
        # from results / session_state each time.)
        if results['pct_converted'] == 0:
            _cur_prov = eib.PROVENANCE_BASELINE
            # Relay A — match the banner title's no-conversion framing so the
            # comparison-table "▶ Current" cell stays in sync with the H2 above
            # the metric grid. Provenance stays BASELINE.
            _cur_label = f"▶ Current — {_explorer_scenario_label(_resolved_scenario)}"
        elif st.session_state.get("applied_from_region_optimizer"):
            _cur_prov = eib.PROVENANCE_REGION_OPTIMIZED
            _cur_label = (f"▶ Current — Region-optimized · "
                          f"{results['scenario_name']}")
        elif st.session_state.get("applied_from_optimizer"):
            _cur_prov = eib.PROVENANCE_OPTIMIZER
            _cur_label = f"▶ Current — Optimizer suggestion · {results['scenario_name']}"
        else:
            _cur_prov = eib.PROVENANCE_EXPLORER
            _cur_label = f"▶ Current — {results['scenario_name']}"
        _cs_cur_src = _cs_source_validation(_cur_prov)[0]
        # Region Selection Phase 1 (Commit 5) + Ownership Integration Commit 3 —
        # augment the Source column when an Explorer scenario is region- and/or
        # ownership-constrained. Same suffixes the main panel header + export
        # bundle metadata use. Baseline (pct=0) reads just 'Baseline' — don't
        # augment. Optimizer can't be placement-active (Optimize is disabled
        # when either constraint is set). Use the layer-present signal instead
        # of mode=='selected_regions' so ownership-only doesn't false-trigger
        # the region suffix.
        if _cur_prov == eib.PROVENANCE_EXPLORER:
            if (results.get('region_selection') or {}).get('layer') is not None:
                _cs_cur_src = f"{_cs_cur_src} · selected region"
            _cs_cur_src = f"{_cs_cur_src}{_ownership_source_suffix(results)}"
        _cs_rows.append({
            "Scenario":   _cur_label,
            "Source":     _cs_cur_src,
            "Validation": _cs_short_validation(_cur_prov),
            "Area":       _cs_area_for_row(results),
            "Ownership":  _cs_ownership_for_row(results),
            **_cs_row_metrics(results),
        })

        # ── 3. Saved scenarios for this city ──
        for _saved in _saved_for_city:
            _prov = _saved.get("provenance")
            if _prov is None:
                # Backfill for older in-memory saves predating Brief #5: best-
                # effort guess from the scenario fields. The applied-from-
                # optimizer flag was an in-memory state at save time, so we can't
                # recover OPTIMIZER for older saves — they read as EXPLORER /
                # BASELINE, which is the safer underclaim.
                _prov = (eib.PROVENANCE_BASELINE if _saved.get("pct_converted", 0) == 0
                         else eib.PROVENANCE_EXPLORER)
            _src = _cs_source_validation(_prov)[0]
            # Region Selection Phase 1 (Commit 5) + Ownership Integration Commit 3
            # — augment Source for saved Explorer scenarios that carry a region
            # and/or ownership selection. The save handler preserves the full
            # results dict (sans scenario_lulc), so both blocks flow through
            # automatically. Pre-29 saves return None safely via .get().
            if _prov == eib.PROVENANCE_EXPLORER:
                if (_saved.get('region_selection') or {}).get('layer') is not None:
                    _src = f"{_src} · selected region"
                _src = f"{_src}{_ownership_source_suffix(_saved)}"
            _label = _saved.get("display_name") or _saved.get("scenario_name") or "(unnamed save)"
            _cs_rows.append({
                "Scenario":   _label,
                "Source":     _src,
                "Validation": _cs_short_validation(_prov),
                "Area":       _cs_area_for_row(_saved),
                "Ownership":  _cs_ownership_for_row(_saved),
                **_cs_row_metrics(_saved),
            })

        # Full Brief #3 wording lives in a column-header tooltip so the cells
        # can stay compact. Source/Validation cell labels are the short form;
        # hover the column header for the source-to-validation mapping.
        _validation_help = (
            "Each source has a different validation context:\n\n"
            "• **NatCap reference** — displayed from NatCap published output; exact scenario raster / aggregation not available.\n\n"
            "• **Baseline** — prototype evaluator, verified against canonical InVEST where comparable; absolute NatCap citywide figures not reproduced.\n\n"
            "• **Explorer-generated** — InVEST-aligned evaluator; verified where comparable; scenario itself not NatCap-published.\n\n"
            "• **machine-learning suggestion** — evaluated with the InVEST-aligned evaluator on apply — exploratory candidate for further validation."
        )
        st.dataframe(
            pd.DataFrame(_cs_rows),
            width='stretch',
            hide_index=True,
            column_config={
                "Scenario":   st.column_config.TextColumn("Scenario", width="medium"),
                "Source":     st.column_config.TextColumn(
                    "Source",
                    width="medium",
                    help="What kind of scenario this row represents. See the Validation column for how that source is grounded.",
                ),
                "Validation": st.column_config.TextColumn(
                    "Validation",
                    width="medium",
                    help=_validation_help,
                ),
                "Area":       st.column_config.TextColumn(
                    "Area",
                    width="small",
                    help="Where conversions were placed. 'Citywide' = no region constraint; otherwise the selected region(s).",
                ),
                "Ownership":  st.column_config.TextColumn(
                    "Ownership",
                    width="small",
                    help="Ownership / vacancy screen applied to the placement pool. 'None' = no screen. SA-only today.",
                ),
            },
        )
        st.caption(
            "Notes: Temperature, carbon stock, and carbon value are shown as changes "
            "from each row's own baseline. NatCap-published rows use NatCap's published "
            "baseline; Explorer-generated rows use the prototype baseline. Flood is "
            "shown separately because NatCap reference and Explorer scenarios use "
            "different derivations."
        )

        # ── Scenario CSV export ───────────────────────────────────────────────
        # Data-complete download of the comparison set: one row per scenario
        # (current + saved-for-city). Full record + computed metrics; every
        # value reads results / _saved directly — no recomputation. NatCap
        # anchors intentionally excluded (no full record; would force "—" on
        # most columns and dilute the round-trip guarantee).
        import io as _csv_io
        from datetime import datetime as _csv_dt, timezone as _csv_tz

        def _csv_row_from_scenario(d, label, provenance, source_label, validation_label):
            rs = d.get('region_selection') or {}
            # Batch 4 v2 — ownership_filter is now str / composite-dict / None.
            # The normalizer collapses all three shapes; the CSV serializes
            # `ownership_classes` as a pipe-joined list (e.g. "city|school"
            # for a multi-class composite). The `ownership_mode` column
            # carries the storage shape for round-trip — string mode key
            # when single-class, JSON-encoded for the composite.
            own_raw  = d.get('ownership_filter')
            own_norm = _normalize_ownership_filter(own_raw)
            city_for_row = d.get('city', selected_city)
            own_layer_meta = (CITIES.get(city_for_row, {}).get('ownership_layer') or {})
            rl = d.get('region_local') or {}
            if own_norm is None:
                _csv_own_mode    = ''
                _csv_own_label   = ''
                _csv_own_classes = ''
                _csv_own_vacant  = ''
                _csv_own_src     = ''
                _csv_own_date    = ''
            else:
                # Round-tripable mode column: string when storage is str,
                # else a JSON-style dict literal so the consumer can ast.literal_eval.
                if isinstance(own_raw, str):
                    _csv_own_mode = own_raw
                else:
                    _csv_own_mode = repr(own_raw)
                _csv_own_label   = own_norm['label']
                _csv_own_classes = '|'.join(own_norm['classes'])
                _csv_own_vacant  = 'true' if own_norm['vacant_only'] else 'false'
                _csv_own_src     = own_layer_meta.get('source') or ''
                _csv_own_date    = own_layer_meta.get('data_date') or ''
            row = {
                'scenario_label':             label,
                'city':                       city_for_row,
                'provenance':                 provenance,
                'source_label':               source_label,
                'validation':                 validation_label,
                'region_layer':               rs.get('layer') or '',
                'region_selected_ids':        '|'.join(rs.get('selected_ids') or []),
                'region_selected_area_acres': (rs.get('selected_area_acres')
                                                if rs.get('mode') == 'selected_regions' else ''),
                'region_eligible_acres':      (rs.get('eligible_pixels_in_region') or 0) * PIXEL_AREA_ACRES,
                'region_converted_acres':     rs.get('converted_acres', 0.0),
                'ownership_mode':             _csv_own_mode,
                # CSV-round-trip safety: empty cell (not "None" sentinel) when no
                # filter is active — pandas read_csv treats "None" as NaN.
                'ownership_label':            _csv_own_label,
                'ownership_classes':          _csv_own_classes,  # NEW: pipe-joined for multi-class
                'ownership_vacant_only':      _csv_own_vacant,   # NEW: 'true'/'false'/'' tri-state
                'ownership_source':           _csv_own_src,
                'ownership_data_date':        _csv_own_date,
                'pct_converted':              d.get('pct_converted'),
                'green_infrastructure_pct':   d.get('green_infrastructure_pct'),
                'food_forest_pct':            d.get('food_forest_pct'),
                'pct_highdensity':            100 - (d.get('green_infrastructure_pct') or 0)
                                                  - (d.get('food_forest_pct') or 0),
                'placement_strategy':         d.get('placement_strategy', placement_strategy),
                'random_seed':                d.get('random_seed', 42),
                'scenario_schema_version':    SCENARIO_SCHEMA_VERSION,
            }
            for k in ('flood_reduction', 'temp_change_f', 'mean_hm', 'mean_ndvi',
                      'food_mln_lbs', 'carbon_tons_co2', 'carbon_value_usd',
                      'cooling_energy_savings_usd', 'nature_access_pct',
                      'people_with_nature_access', 'preventable_mh_cases',
                      'avoided_mh_cost_usd', 'total_cost_mln', 'runoff_acre_feet'):
                row[k] = d.get(k)
            # Flood Damage Avoided — complete-record column for the CSV handoff
            # (the dashboard family curates it out; the export stays complete).
            # Damage-table cities (MN) emit the raw value; no-damage-table cities
            # (SA) emit an EMPTY cell — the engine returns 0.0 there, which as a
            # raw "0" would misread as "avoided zero damage", and a string
            # sentinel would break the numeric column. Same single-source
            # availability gate (TOTAL_POTENTIAL_DAMAGE_USD > 0) the dashboard
            # and comparison surfaces use. `d.get` keeps it null-safe for older
            # saves missing the field.
            row['flood_damage_avoided_usd'] = (
                d.get('flood_damage_avoided_usd')
                if TOTAL_POTENTIAL_DAMAGE_USD > 0 else ''
            )
            for k, cfg in _REGION_LOCAL_METRICS.items():
                if cfg.get('decomposable'):
                    row[f'region_local__{k}'] = rl.get(k) if rl else ''
            return row

        _csv_cur_val = _PROVENANCE_HEADER_INFO.get(
            _cur_prov, ("Unknown", "provenance not recorded", "gray")
        )[1]
        _csv_rows = [
            _csv_row_from_scenario(
                results, _cur_label, _cur_prov, _cs_cur_src, _csv_cur_val,
            )
        ]
        for _saved in _saved_for_city:
            _prov_save = _saved.get("provenance")
            if _prov_save is None:
                _prov_save = (eib.PROVENANCE_BASELINE
                              if _saved.get("pct_converted", 0) == 0
                              else eib.PROVENANCE_EXPLORER)
            _src_save = _cs_source_validation(_prov_save)[0]
            if _prov_save == eib.PROVENANCE_EXPLORER:
                if (_saved.get('region_selection') or {}).get('layer') is not None:
                    _src_save = f"{_src_save} · selected region"
                _src_save = f"{_src_save}{_ownership_source_suffix(_saved)}"
            _label_save = (_saved.get("display_name")
                           or _saved.get("scenario_name") or "(unnamed save)")
            _val_save = _PROVENANCE_HEADER_INFO.get(
                _prov_save, ("Unknown", "provenance not recorded", "gray")
            )[1]
            _csv_rows.append(
                _csv_row_from_scenario(_saved, _label_save, _prov_save,
                                        _src_save, _val_save)
            )

        _csv_buf = _csv_io.StringIO()
        pd.DataFrame(_csv_rows).to_csv(_csv_buf, index=False)
        _csv_filename = (
            f"scenario_summary_"
            f"{selected_city.split(',')[0].lower().replace(' ', '_')}"
            f"_{_csv_dt.now(_csv_tz.utc).strftime('%Y-%m-%d')}.csv"
        )
        st.download_button(
            label="Download scenario summary (CSV)",
            data=_csv_buf.getvalue(),
            file_name=_csv_filename,
            mime="text/csv",
            help=("Current scenario plus every saved scenario for this city as a "
                  "CSV. Full record (region, ownership, placement, seed) plus "
                  "citywide and region-local metrics. NatCap reference rows are "
                  "not included — they don't carry a complete record."),
        )

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
                f"cooling. The 'vs city avg' columns show each {_polygon_unit_singular}'s "
                f"temperature relative to the city-wide average (positive = warmer); the "
                f"Temperature change column shows the scenario's effect. Population-weighted "
                f"within each {_polygon_unit_singular}."
            )
            _tracts_summary = compute_per_tract_summary(results['scenario_lulc_ucm'])
            if not _tracts_summary.empty:
                # Most cooling first: _change_f is negative for cooling under the
                # ΔT convention (positive = warmer), so sort ascending.
                _top5 = (
                    _tracts_summary
                    .sort_values("_change_f", ascending=True)
                    .head(5)
                    .reset_index(drop=True)
                )
                # Display layer: render the signed ΔT as natural language (no bare
                # signed numbers for the user) and color the change column —
                # cooler = green, warmer = red.
                _top5["Temperature change"] = _top5["_change_f"].apply(_fmt_temp_change)
                _top5 = _top5.drop(columns=["_change_f"])

                def _color_temp_change(val):
                    if isinstance(val, str) and "cooler" in val:
                        return "color: #1a7f37; font-weight: 600"   # green
                    if isinstance(val, str) and "warmer" in val:
                        return "color: #cf222e; font-weight: 600"   # red
                    return ""

                _styled = _top5.style.map(_color_temp_change, subset=["Temperature change"])
                st.dataframe(_styled, width='stretch', hide_index=True)
            else:
                st.caption(f"No {_polygon_unit_singular}-level data could be computed for this scenario.")

        # UI feedback #4 — the "Best scenarios by goal" library is the
        # citywide precomputed lookup; its rankings are computed at citywide
        # scope and don't reflect any active region or ownership filter.
        # Showing it under a region/ownership scenario would imply rankings
        # that account for the filter when they don't. Hide it in that
        # case; show it only in citywide-no-filter mode.
        _best_by_goal_filter_active = (
            st.session_state.get('selected_region_mask') is not None
            or st.session_state.get('selected_ownership_mask') is not None
        )
        if _best_by_goal_filter_active:
            # Single-line note so the user knows why the section is missing,
            # plus the action that brings it back.
            st.divider()
            st.caption(
                "_'Best citywide scenarios by goal' is hidden under a region or "
                "ownership filter — the precomputed library is citywide "
                "and its rankings don't reflect your filter. Clear the "
                "region selection and ownership filter to see it._"
            )
        else:
            st.divider()
            st.markdown("#### Best citywide scenarios by goal")
            st.caption("From the pre-computed scenario library — InVEST-aligned evaluator results, not fast estimates. Citywide library results; not filtered by selected region or ownership.")

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
                        # Brief #4: Best-by-Goal comes from the precomputed scenario
                        # grid, not the surrogate optimizer — make sure a previously-
                        # set Applied-from-Optimizer flag is cleared, so a best-goal
                        # scenario that happens to share pct/gi/ff with a prior
                        # optimizer Apply doesn't inherit OPTIMIZER provenance via the
                        # auto-clear's "values match" path. Same defense for the
                        # region-optimizer flag.
                        st.session_state.applied_from_optimizer = False
                        st.session_state._applied_optimizer_values = None
                        st.session_state.applied_from_region_optimizer = False
                        st.session_state._applied_region_optimizer_values = None
                        st.session_state._show_apply_toast = True
                        st.rerun()

            _render_apply_toast()

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
                # Scenario Record Pass — capture the seed even though it's
                # hardcoded to 42 today. All five placement strategies route
                # through rng (the ranking strategies sample stochastically with
                # weights), so capturing the seed forward-compats any future
                # seed-variation work and makes the record self-reproducing.
                saved["random_seed"] = 42
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
                # Brief #5 — record the scenario's provenance so the cross-source
                # comparison table can read it back later. Uses the same detection
                # as the Brief #3 main-panel header / Brief #4 D1 export branch.
                # Older in-memory saves predating this brief get an explicit None
                # backfill in the table itself, so this is safe to add without a
                # schema bump.
                if results['pct_converted'] == 0:
                    saved["provenance"] = eib.PROVENANCE_BASELINE
                elif st.session_state.get("applied_from_region_optimizer"):
                    saved["provenance"] = eib.PROVENANCE_REGION_OPTIMIZED
                elif st.session_state.get("applied_from_optimizer"):
                    saved["provenance"] = eib.PROVENANCE_OPTIMIZER
                else:
                    saved["provenance"] = eib.PROVENANCE_EXPLORER
                st.session_state.saved_scenarios.append(saved)
                st.session_state.show_save_input = False
                st.success(f"Saved: {scenario_name_input}")
                st.rerun()
            elif confirm_clicked and not scenario_name_input:
                st.warning("Please enter a name before saving.")

        # Mode-switch render. Filter-active → render the region-optimized
        # results (engine-true region-local values, no surrogate bands). Otherwise
        # → render the existing citywide surrogate suggestions. The two paths
        # write to distinct session-state slots (`optimized_results` /
        # `region_optimized_results`) so each is mode-safe. See
        # docs/internal/REGION_OPTIMIZER_SPEC.md §6.
        if (_filter_active
                and st.session_state.region_optimized_results is not None
                and not st.session_state.region_optimized_results.empty):
            st.divider()
            # Relay B: header + caption + column set. "Best tested mixes for
            # selected area" framing keeps the user honest that this is "best
            # among what we tested," not "the optimum." The caption is the
            # short engine-vs-prediction reminder; the "coarse search" caveat
            # is owned by the sidebar Discover copy above. Columns: Rank / Mix
            # / Score / Converted acres / Cooling / Flood Index / Carbon /
            # Food / Cost / Apply.
            st.subheader("Best tested mixes for selected area")
            st.caption(
                "Evaluated with the InVEST-aligned evaluator under the current "
                "selected area and conversion filters."
            )
            _ropt = st.session_state.region_optimized_results.copy()
            # Synthesize Rank + Mix columns for display. Mix folds the three
            # knob percentages into one cell so the table is readable at
            # sidebar widths.
            _ropt = _ropt.reset_index(drop=True)
            _ropt.insert(0, 'Rank', _ropt.index + 1)
            _ropt['Mix'] = _ropt.apply(
                lambda r: (
                    f"{int(r.pct_converted)}% conv — "
                    f"GI {int(r.green_infrastructure_pct)}% / "
                    f"FF {int(r.food_forest_pct)}%"
                ),
                axis=1,
            )
            _opt_carbon_col_label_r = _carbon_table_col_label
            _r_display_cols = [
                'Rank', 'Mix', 'weighted_score', 'converted_acres',
                'mean_hm', 'flood_reduction', 'carbon_tons_co2',
                'food_mln_lbs', 'total_cost_mln',
            ]
            _r_col_rename = {
                'weighted_score':           'Score',
                'converted_acres':          'Converted acres',
                'mean_hm':                  'Cooling',
                'flood_reduction':          'Flood Index',
                'carbon_tons_co2':          _opt_carbon_col_label_r,
                'food_mln_lbs':             'Food (M lbs)',
                'total_cost_mln':           'Cost ($M)',
            }
            _r_present = [c for c in _r_display_cols if c in _ropt.columns]
            st.dataframe(_ropt[_r_present].rename(columns=_r_col_rename),
                         width='stretch', hide_index=True)

            st.markdown("#### Apply a suggestion")
            st.caption(
                "Applying a suggestion re-runs the engine on your selected "
                "area; provenance becomes \"Engine-verified — region-optimized\"."
            )
            _r_btn_cols = st.columns(len(_ropt))
            for i, (_, row) in enumerate(_ropt.iterrows()):
                with _r_btn_cols[i]:
                    _prefix = ("✓ " if st.session_state.get("applied_suggestion") == i
                               else "")
                    _label = f"{_prefix}#{i+1}: {int(row.pct_converted)}% conv"
                    if st.button(_label, key=f"apply_region_opt_{i}"):
                        # Single source of truth — same helper the chart
                        # click-to-apply path calls.
                        _apply_region_optimizer_mix(row, i)
                        st.rerun()

            _render_apply_toast()

            st.divider()

        if (not _filter_active) and st.session_state.optimized_results is not None:
            st.divider()
            # Optimizer Promotion — de-optimize the heading: "Suggested
            # scenarios" framed as predicted (not "Optimized"). Caption below
            # still flags the surrogate-prediction nature.
            st.subheader("Suggested scenarios")
            st.caption("Scroll down to see suggestions and apply them to the sliders.")
            opt = st.session_state.optimized_results
            # Brief 30: SA optimizer reports stock-change; MN reports annual flow.
            _opt_carbon_col_label = _carbon_table_col_label
            if isinstance(opt, dict) and not opt.get('found'):
                st.warning(
                    f"No scenarios found meeting all targets simultaneously.  \n"
                    f"Maximum achievable values across all candidates:  \n"
                    f"- Flood Index: up to **{opt['max_flood']}** (your target: {min_flood})  \n"
                    f"- Cooling: up to **{opt['max_cool']:.4f} HMI** (your target: {min_cool:.4f})  \n"
                    f"- Food: up to **{opt['max_food']:.3f}M lbs** (your target: {min_food:.3f})  \n"
                    f"- Carbon: up to **{opt['max_carbon']:,.0f} {_carbon_unit_suffix}** (your target: {min_carbon:,})  \n"
                    f"Try lowering the target for whichever metric is furthest from its maximum."
                )
            else:
                st.caption(
                    "Top scenarios meeting the minimum Flood Index, cooling, food, and carbon "
                    "thresholds set by the sliders — ranked by balanced score. "
                    "Numbers are fast estimates from the machine-learning model, with calibrated estimate ranges."
                )

                # Display table with estimate-range columns
                display_cols = ['scenario_name', 'pct_converted', 'green_infrastructure_pct',
                                'food_forest_pct', 'flood_reduction', 'mean_hm', 'food_mln_lbs',
                                'carbon_tons_co2']
                # Add estimate-range columns if present (dropped when the active
                # mode has no calibration artifact — then no range is shown).
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
                    "These are fast estimates from the machine-learning model. Click Apply to compute it "
                    "with the InVEST-aligned evaluator and verify the result."
                )
                if unc_cols:
                    # Long table for the TOP candidate (#1): Metric | Fast estimate
                    # | Estimate range, lo–hi combined into one cell, no raw column
                    # names surfaced. The plot hover covers the other candidates
                    # per-point, so no selectbox. Nature access is excluded (no band).
                    _exp_label = ("Estimate ranges" if len(opt) == 1
                                  else "Estimate ranges (top candidate)")
                    with st.expander(_exp_label, expanded=False):
                        st.caption(
                            "Ranges are calibrated from the fast model's "
                            "cross-validation errors against evaluator-computed "
                            "results. They apply only to citywide machine-learning "
                            "suggestions and are not guarantees."
                        )
                        _t = opt.iloc[0]

                        def _carb_cell(*vals):
                            # Consistent unit across the cell (k vs M), not _fmt_sig's
                            # auto-float — so a range reads '0.76M–1.06M', not
                            # '760k–1.06M'. Suffix added by the caller.
                            mx = max(abs(v) for v in vals)
                            if mx >= 1e6:
                                return [f"{v / 1e6:.2f}M" for v in vals]
                            if mx >= 1e3:
                                return [f"{v / 1e3:.0f}k" for v in vals]
                            return [f"{v:,.0f}" for v in vals]

                        _cf, _cl, _ch = _carb_cell(
                            _t['carbon_tons_co2'], _t['carbon_lower'], _t['carbon_upper'])
                        _range_rows = [
                            ("Flood Index", _fmt_sig(_t['flood_reduction']),
                             f"{_fmt_sig(_t['flood_lower'])}–{_fmt_sig(_t['flood_upper'])}"),
                            ("Cooling HM", _fmt_sig(_t['mean_hm']),
                             f"{_fmt_sig(_t['hm_lower'])}–{_fmt_sig(_t['hm_upper'])}"),
                            ("Food production", f"{_t['food_mln_lbs']:.1f}M lbs",
                             f"{_t['food_lower']:.1f}–{_t['food_upper']:.1f}M lbs"),
                            (_carbon_table_col_label, _cf,
                             f"{_cl}–{_ch}"),
                        ]
                        st.dataframe(
                            pd.DataFrame(_range_rows,
                                         columns=["Metric", "Fast estimate", "Estimate range"]),
                            width='stretch', hide_index=True)
                # Compact candidate table — one row per scenario, screening
                # precision. The conversion / GI / FF % columns are dropped: the
                # Scenario name already carries "N% converted — GI x / FF y".
                _cand_display = pd.DataFrame({
                    'Scenario':              opt['scenario_name'].values,
                    'Flood Index':           [_fmt_sig(v) for v in opt['flood_reduction']],
                    'Cooling HM':            [_fmt_sig(v) for v in opt['mean_hm']],
                    'Food':                  [_fmt_food(v) for v in opt['food_mln_lbs']],
                    _opt_carbon_col_label:   [_fmt_sig(v) for v in opt['carbon_tons_co2']],
                })
                st.dataframe(_cand_display, width='stretch', hide_index=True)
                st.caption(
                    "Note: suggestions with small amounts of High Density (2–10%) may "
                    "reflect the machine-learning model's approximation — consider setting HD to 0% when applying."
                )

                st.markdown("#### Input Influence")
                st.caption("**Influence Map** — which inputs the fast machine-learning model relies on most. Not a causal ranking.")
                st.plotly_chart(plot_feature_importance(surrogate), use_container_width=True)

                st.markdown("#### Apply a suggestion")
                st.caption(
                    "Suggestions are ranked by balanced score across flood, cooling, "
                    "and food metrics. #1 is the top-ranked scenario."
                )
                st.caption(
                    "Apply a suggestion to make it the active scenario — the map, "
                    "metrics, and comparison table update."
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
                            # Brief #4: tag this scenario as Applied-from-Optimizer
                            # so the main panel header reads "machine-learning suggestion"
                            # and the D1 export records PROVENANCE_OPTIMIZER, not
                            # Explorer. The clearing logic at the top of the script
                            # resets the flag when slider values drift away. Clear
                            # the region-optimizer flag so the two states can't
                            # co-fire after a citywide Apply on a previous run.
                            st.session_state.applied_from_optimizer = True
                            st.session_state._applied_optimizer_values = (
                                st.session_state._pending_pct,
                                st.session_state._pending_gi,
                                st.session_state._pending_ff,
                            )
                            st.session_state.applied_from_region_optimizer = False
                            st.session_state._applied_region_optimizer_values = None
                            st.session_state._show_apply_toast = True
                            st.rerun()

                # One-shot confirmation toast: rendered on the rerun immediately
                # following an Apply click, then cleared so it doesn't persist
                # through unrelated reruns.
                _render_apply_toast()

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
                    'temp_change_f',
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

if _main_tab == 'Map View':
    with tab3:
        st.subheader("Where land-cover changes happen")

        # ── Interactive Region Selector (Interactive Region Map Spec, Path C) ──
        # Plain-cartesian plotly polygon traces over EPSG:5070 coords — no basemap,
        # no new deps. Click a district to set the selection; shift- or ctrl-click
        # to add; Clear button to wipe. Syncs with the sidebar dropdown via the
        # top-of-script handler (both write to `region_labels_<layer>` in
        # session_state). Honors whichever region_layer the dropdown currently
        # has active. Hidden when no region layers are configured (citywide-only
        # cities); also hidden when the dropdown is set to "Entire analysis area"
        # since the selector wouldn't compose with it cleanly.
        _t3_layer = st.session_state.get('selected_region_layer')
        _t3_layer_cfg = (
            (city_cfg.get('region_layers') or {}).get(_t3_layer)
            if _t3_layer else None
        )
        if _t3_layer_cfg is not None:
            _t3_polys = _load_region_polygons_for_plotly(
                _t3_layer_cfg['path'], _t3_layer_cfg['label_field']
            )
            _t3_selected_ids = st.session_state.get('selected_region_ids') or []
            _t3_display = _CURRENT_CITY_STATE.region_layer_display_names.get(_t3_layer, "region")
            _t3_fig = go.Figure()
            # Collect centroids so we can add a single markers+text trace
            # at the end. Selection snaps to data POINTS, not polygon
            # interiors — the previous add_annotation centroid labels were
            # static and not clickable, which is why click-to-select didn't
            # fire even with on_select='rerun'. The markers trace renders
            # the same visible labels AND captures click selections via
            # customdata=district_id.
            _t3_cx_list, _t3_cy_list, _t3_label_list = [], [], []
            for _t3_label, _t3_rings in _t3_polys:
                _is_sel = _t3_label in _t3_selected_ids
                _fill = 'rgba(31, 119, 180, 0.55)' if _is_sel else 'rgba(170, 195, 220, 0.18)'
                _line_w = 4.0 if _is_sel else 1.2
                for _xs, _ys in _t3_rings:
                    _t3_fig.add_trace(go.Scatter(
                        x=_xs, y=_ys,
                        fill='toself',
                        fillcolor=_fill,
                        mode='lines',
                        line=dict(color='#1f77b4', width=_line_w),
                        customdata=[_t3_label] * len(_xs),
                        hovertemplate=f"<b>{_t3_display} {_t3_label}</b><extra></extra>",
                        showlegend=False,
                        name=_t3_label,
                    ))
                if _t3_rings:
                    _xs0, _ys0 = _t3_rings[0]
                    _t3_cx_list.append(sum(_xs0) / len(_xs0))
                    _t3_cy_list.append(sum(_ys0) / len(_ys0))
                    _t3_label_list.append(_t3_label)
            # Centroid labels rendered as a clickable markers+text trace.
            # The markers are kept large (size=28) but fully transparent —
            # they act as generous click targets centered on each district's
            # visible label. clickmode='event+select' on the layout makes
            # single clicks fire the selection.
            if _t3_label_list:
                _t3_fig.add_trace(go.Scatter(
                    x=_t3_cx_list, y=_t3_cy_list,
                    text=_t3_label_list,
                    mode='markers+text',
                    marker=dict(size=28, color='rgba(0,0,0,0)'),
                    textfont=dict(size=12, color='#1f3a5c', family='sans-serif'),
                    textposition='middle center',
                    customdata=_t3_label_list,
                    hovertemplate=f"<b>{_t3_display} %{{customdata}}</b><extra></extra>",
                    showlegend=False,
                    name='district-labels',
                ))
            _t3_fig.update_layout(
                xaxis=dict(visible=False, scaleanchor='y', scaleratio=1),
                yaxis=dict(visible=False),
                plot_bgcolor='white', paper_bgcolor='white',
                height=360,
                margin=dict(l=0, r=0, t=10, b=10),
                # clickmode='event+select' makes a single click fire the
                # point-select event. Without it, even with selection_mode=
                # 'points' on the chart, clicks would pan instead — exactly
                # the symptom the brief flagged. dragmode left at Plotly's
                # default (zoom) — dragmode='pan' previously swallowed
                # clicks in this Plotly version, which compounded the bug.
                clickmode='event+select',
            )
            _t3_picker_col, _t3_clear_col = st.columns([6, 1])
            with _t3_picker_col:
                # Stash the current layer key so the top-of-script handler knows
                # which multiselect key to sync the event into.
                st.session_state['region_map_picker_layer'] = _t3_layer
                _t3_event = st.plotly_chart(
                    _t3_fig,
                    use_container_width=True,
                    on_select='rerun',
                    selection_mode='points',
                    key='region_map_picker',
                )
                # Stash the event for the top-of-next-rerun handler, then force
                # a rerun so the handler fires before the sidebar reads
                # region_labels_<layer> on the click frame itself (tab3 runs
                # AFTER the sidebar, so without this the click would land one
                # rerun late).
                #
                # Multi-select RELAY: new-click detection compares the event's
                # selection signature against the LAST-FORWARDED signature in
                # session_state. The prior compare-against-current_ids
                # approach broke toggle-off (after toggling A off, current=[B]
                # but Plotly=[A], so the producer kept re-firing). Pure
                # signature de-dup ALONE broke the deploy-time auto-rerun
                # cascade: on first run with no last_sig recorded, Plotly's
                # sticky selection state from the prior session looks
                # identical to a fresh click and forwards a phantom event
                # that toggles the user's pre-existing selection OFF.
                # Fix: when last_sig is uninitialized (None), check whether
                # the event's signature already matches the user's existing
                # selection state. If yes, this is a re-render of an
                # already-aligned state — sync last_sig without forwarding.
                # Genuine first-click from an empty state still forwards
                # because evt_sig differs from the empty current selection.
                if _t3_event:
                    _new_event = _t3_event if isinstance(_t3_event, dict) else dict(_t3_event)
                    _evt_points = _new_event.get('selection', {}).get('points') or []
                    if _evt_points:
                        _evt_sig = tuple(sorted(
                            p.get("customdata") for p in _evt_points
                            if p.get("customdata") is not None
                        ))
                        _ms_key = f"region_labels_{_t3_layer}"
                        _last_sig_raw = st.session_state.get("region_map_picker_last_sig")
                        if _last_sig_raw is None:
                            # First time this session: distinguish stale
                            # re-render (Plotly's sticky state already matches
                            # the user's existing selection) from a genuine
                            # first click (Plotly's state diverges from
                            # session_state because the user just clicked
                            # something into an empty/different selection).
                            _current_sig = tuple(sorted(
                                st.session_state.get(_ms_key, []) or []
                            ))
                            if _evt_sig == _current_sig:
                                # Aligned — sync silently, don't toggle.
                                st.session_state['region_map_picker_last_sig'] = list(_evt_sig)
                            else:
                                # Genuine click — forward + rerun.
                                st.session_state['region_map_picker_event'] = _new_event
                                st.session_state['region_map_picker_last_sig'] = list(_evt_sig)
                                st.rerun()
                        elif _evt_sig != tuple(_last_sig_raw):
                            # Steady-state new click: signature differs from
                            # what we last forwarded → forward + rerun.
                            st.session_state['region_map_picker_event'] = _new_event
                            st.session_state['region_map_picker_last_sig'] = list(_evt_sig)
                            st.rerun()
            with _t3_clear_col:
                st.write("")
                st.write("")
                if st.button("Clear", key='region_map_clear_btn',
                             help="Deselect all selected areas."):
                    st.session_state[f"region_labels_{_t3_layer}"] = []
                    st.session_state['region_map_picker_event'] = None
                    # Reset the new-click signature too so the next genuine
                    # click fires (without this, a click matching the
                    # last-forwarded signature would be filtered out).
                    st.session_state['region_map_picker_last_sig'] = None
                    st.rerun()
            st.caption(
                f"Click a {_t3_display.lower()} number to toggle its selection — "
                f"click another to add, click again (after picking a different "
                f"{_t3_display.lower()}) to remove. The sidebar dropdown is the "
                "same source of truth. Land-use changes will be placed only "
                "inside the selected area; the Scenario tab shows both "
                "citywide and region-local results."
            )
            # Eligibility Funnel (Interactive Region Map Spec #3 — extended).
            # Shows where pixels drop out at each placement-pool step:
            # selected → developed → after roads/buildings/existing nature →
            # after ownership → converted. Sources every cell from
            # results['region_selection'] (selected_area_acres, converted_acres,
            # eligible_pixels_in_region) or a one-line intersection of masks
            # already on hand (developed/convertible ∩ region). Monotonicity is
            # guaranteed by the subset invariants (verify_baselines.py — every
            # step ⊆ the prior is a standing assertion).
            _rs_t3 = results.get('region_selection') or {}
            if _rs_t3.get('mode') == 'selected_regions':
                _region_mask = st.session_state.get('selected_region_mask')
                _ownership_active = (
                    st.session_state.get('selected_ownership_mask') is not None
                )
                _t3_sel_area = _rs_t3.get('selected_area_acres') or 0.0
                _t3_final_elig_px = _rs_t3.get('eligible_pixels_in_region') or 0
                _t3_final_elig_acres = _t3_final_elig_px * PIXEL_AREA_ACRES
                _t3_conv_acres = _rs_t3.get('converted_acres') or 0.0
                # The one new computation — developed ∩ region (and, when
                # ownership is also active, convertible ∩ region pre-ownership
                # for the intermediate "After roads/buildings/existing nature"
                # step). Both single-line index ops against arrays already in
                # _CURRENT_CITY_STATE.
                if _region_mask is not None:
                    _dp = _CURRENT_CITY_STATE.developed_pixels
                    _cp = _CURRENT_CITY_STATE.convertible_pixels
                    _t3_dev_in_region_px = int(
                        _region_mask[_dp[:, 0], _dp[:, 1]].sum()
                    )
                    _t3_conv_in_region_px = int(
                        _region_mask[_cp[:, 0], _cp[:, 1]].sum()
                    )
                else:
                    _t3_dev_in_region_px = 0
                    _t3_conv_in_region_px = 0
                _t3_dev_in_region_acres = _t3_dev_in_region_px * PIXEL_AREA_ACRES
                _t3_conv_in_region_acres = _t3_conv_in_region_px * PIXEL_AREA_ACRES

                # Build the chain. Region+ownership splits the convertible step
                # from the final eligible step; region-only collapses them
                # (eligible_pixels_in_region already equals convertible ∩ region
                # when ownership is inactive).
                _funnel_rows = [
                    ("Selected area",                          f"{_t3_sel_area:,.0f} acres"),
                    ("Developed land",                         f"{_t3_dev_in_region_acres:,.0f} acres"),
                ]
                if _ownership_active:
                    _funnel_rows.append(
                        ("After roads / buildings / existing nature",
                         f"{_t3_conv_in_region_acres:,.0f} acres"),
                    )
                    _funnel_rows.append(
                        ("After ownership filter",
                         f"{_t3_final_elig_acres:,.0f} acres"),
                    )
                else:
                    _funnel_rows.append(
                        ("After roads / buildings / existing nature",
                         f"{_t3_final_elig_acres:,.0f} acres"),
                    )
                _funnel_rows.append(
                    ("Converted", f"{_t3_conv_acres:,.0f} acres"),
                )
                _funnel_df = pd.DataFrame(_funnel_rows, columns=["Step", "Acres"])
                st.dataframe(
                    _funnel_df, hide_index=True, width="stretch",
                    column_config={
                        "Step":  st.column_config.TextColumn("Step", width="large"),
                        "Acres": st.column_config.TextColumn("Acres", width="small"),
                    },
                )
                # UI-Text Pass — region-id caption beneath the panel, derived from
                # the active layer's display name; replaces the layer-specific
                # label ("Selected tract" / "Selected district") on the metric.
                _t3_n_sel = len(_t3_selected_ids)
                if _t3_n_sel == 1:
                    _t3_id_caption = f"{_t3_display} {_t3_selected_ids[0]}"
                elif 1 < _t3_n_sel <= 3:
                    _t3_id_caption = (
                        f"{_t3_n_sel} selected {_t3_display.lower()}s: "
                        f"{', '.join(_t3_selected_ids)}"
                    )
                else:
                    _t3_id_caption = f"{_t3_n_sel} selected {_t3_display.lower()}s"
                st.caption(_t3_id_caption)
                st.caption(
                    "Land-use changes are placed only inside the selected area."
                )
            st.divider()

        if placement_strategy != 'random':
            st.info(
            f"**{PLACEMENT_STRATEGY_LABELS[placement_strategy]}** — conversions weighted "
            "toward higher-suitability pixels. Notice the spatial pattern shift vs. random allocation."
            )

        with st.expander("Map display options", expanded=False):
            overlay_opacity = st.slider(
                "Overlay opacity",
                0.0, 0.5, 0.15, 0.05,
                help=(
                    "Developed-area intensity from land cover, used as a proxy "
                    "for urban heat vulnerability. Visual context only — it does "
                    "not change the scenario."
                ),
            )

        # Normalize an all-False selected_region_mask to None — semantically
        # the same as no region selected, and protects downstream consumers
        # from treating a bare zero-mask as a real selection.
        _spatial_mask = st.session_state.get('selected_region_mask')
        if _spatial_mask is not None and not bool(_spatial_mask.any()):
            _spatial_mask = None
        # Render via components.html + base64 data-URI img instead of
        # st.image / st.pyplot. Both Streamlit display surfaces routed
        # through MediaFileManager and BOTH dropped the cold-citywide
        # figure (st.image switch in 16500d8 didn't fix it; the
        # st.image-as-second-element render that worked in f824605's
        # side-by-side was a context that no longer applies when
        # st.image is the sole render). A base64 data URI in a
        # components.html iframe sidesteps the media-file transport
        # entirely. Cleanup recipe stays applied — mask normalization
        # above and _PLOT_MAX_DIM cap inside plot_spatial_map are the
        # structural pieces; this is a transport swap.
        import io
        import base64
        from streamlit.components.v1 import html as _components_html
        _spatial_fig = plot_spatial_map(
            results['scenario_lulc'], cooling_lulc,
            heat_overlay=nlcd_intensity_weights, overlay_alpha=overlay_opacity,
            selected_region_mask=_spatial_mask,
        )
        _png_buf = io.BytesIO()
        _spatial_fig.savefig(_png_buf, format='png')
        plt.close(_spatial_fig)
        _png_b64 = base64.b64encode(_png_buf.getvalue()).decode()
        _components_html(
            f'<img src="data:image/png;base64,{_png_b64}" '
            f'style="width:100%">',
            height=820,
        )
        st.caption(
            "Gray = unchanged developed land. Scenario colors show conversions. "
            "White = outside city boundary. Orange shading shows developed urban "
            "intensity for context; darker orange = more intense development."
        )

        with st.expander("Assumptions and limitations", expanded=False):
            st.caption("Detailed modeling assumptions, caveats, and method notes.")
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

if _main_tab == 'NatCap Reference':
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
