"""export_invest_bundle.py — assemble a runnable canonical-InVEST input bundle
for the currently-displayed scenario (Brief D1).

The prototype reimplements the InVEST urban models in numpy; this module bridges
forward: it packages the active scenario as a zip of rasters + shared inputs +
biophysical tables + per-model `args.json` files + `metadata.json` + `README.md`,
so a technical user can run canonical InVEST 3.19.0 on it directly.

Streamlit-agnostic (mirrors `surrogate.py` / `natcap_scenarios.py`). The caller
(app.py) gathers runtime state into a `BundleSpec` and calls
`build_invest_bundle(spec)`, which returns the zip as bytes (in-memory — no temp
dir, Streamlit-Cloud-safe). A standalone builder (`build_baseline_bundle_for_sa`)
reuses app.py via the streamlit-stub pattern for offline verification.

ENCODING / GATING (Brief B1 + D1 amendments):
- Baseline / Explorer / Optimizer scenarios have a prototype-built **compound**
  (NLCD×NLUD×tree) LULC, so all five models export full args.
- **NatCap fixed *alternative* scenarios** (FF_20ac … UA_MAX) exist only in flood
  encoding (NLCD×tree); their compound inputs are unavailable (see
  OPEN_QUESTIONS.md). For those, `compound_models_available=False` → export the
  flood source raster + UFR args only; `metadata.json` flags the compound-model
  args as unavailable. Never fabricate compound args from the flood raster.
- UCM export is biophysical-cooling-only: `do_energy_valuation=False` (no
  building-vector dependency, no cooling-$ card reproduction) — noted in
  metadata + README.
- UMH canonical execution is verified on the baseline (D1 Phase 3 — all five
  models execute cleanly). The two emitted UMH args files (depression, anxiety)
  use a synthetic uniform prevalence vector and a synthetic NDVI proxy — these
  are input-quality caveats, separate from the (verified) algorithmic parity.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# Provenance taxonomy (Brief B1).
from natcap_scenarios import (
    PROVENANCE_BASELINE,
    PROVENANCE_NATCAP_FIXED,
    PROVENANCE_EXPLORER,
    PROVENANCE_OPTIMIZER,
)

EXPORT_SCHEMA_VERSION = "1.0"

# Per-model validation state (from REFERENCE.md "Official InVEST alignment" +
# the compare_*_invest.py harnesses). Frozen text — surfaced in metadata.json.
_VALIDATION = {
    "ucm": {"status": "validated", "mae": 0.0, "pearson_r": 1.0,
            "reference": "natcap.invest.urban_cooling_model 3.19.0",
            "notes": "HMI = max(CC_local, CC_park) validated to per-pixel parity "
                     "(compare_ucm_invest.py). Export is biophysical-cooling-only "
                     "(do_energy_valuation=False)."},
    "una": {"status": "validated", "mae": 0.0, "pearson_r": 1.0,
            "reference": "natcap.invest.urban_nature_access 3.19.0",
            "notes": "2SFCA re-implementation validated per-pixel. Per-block-group "
                     "aggregation differs from a citywide mean — see "
                     "NATCAP_ALIGNMENT.md."},
    "ufr": {"status": "methodology_aligned",
            "reference": "natcap.invest.urban_flood_risk_mitigation 3.19.0",
            "notes": "Canonical SCS-CN method; the prototype's flood-retention "
                     "index reports 100 − mean_CN rather than InVEST's runoff "
                     "retention index. SA uses NatCap's NLCD×tree CN table; "
                     "damage valuation disabled (no SA damage table, Path C)."},
    "carbon": {"status": "methodology_aligned",
               "reference": "natcap.invest.carbon 3.19.0",
               "notes": "SA: four-pool stock framework per NatCap Vibrant Land "
                        "(Guerry et al. 2023). do_valuation=False in the export "
                        "(stock change only)."},
    "umh": {"status": "validated",
            "reference": "natcap.invest.urban_mental_health 3.19.0",
            "notes": "Per-pixel kernel parity validated (Brief B); canonical "
                     "execution on the baseline bundle verified (D1 Phase 3). "
                     "Two args files emitted (depression, anxiety). Inputs use a "
                     "synthetic uniform baseline-prevalence vector "
                     "(risk_rate = CDC ever-diagnosed BIR) and a synthetic NDVI "
                     "proxy, not satellite NDVI — input-quality caveats, "
                     "separate from algorithmic parity."},
}


# Honesty-Surface Pass Commit 2 — locked seed list of known divergences from
# canonical / published values. Every exported bundle's metadata.json carries
# the full list verbatim under scenario.known_divergences. Commit 4's
# assertion enforces completeness: an exported bundle missing any of these
# IDs fails the gate, so the pre-vetted disclosures can't be silently dropped
# in a later refactor. New divergences added to this list become asserted
# automatically — that's the mechanism, rather than relying on anyone to
# re-notice them. Source: docs/internal/HONESTY_SURFACE_PASS_SPEC.md.
KNOWN_DIVERGENCES = [
    {
        "id": "sa_citywide_not_reproduced",
        "summary": "Citywide San Antonio NatCap figures not reproduced as absolute values",
        "detail": (
            "NatCap's published San Antonio citywide outputs cannot be matched "
            "exactly. The per-scenario LULC rasters NatCap used were unsaved "
            "intermediates; the prototype's engine is validated against canonical "
            "InVEST 3.19.0 at MAE near 0 for UCM / UNA / UMH, but reproducing "
            "NatCap's specific scenario aggregations requires their LULC inputs. "
            "Surfaced as 'absolute NatCap citywide figures not reproduced' on "
            "the Source line + 'NatCap published reference' badge on values "
            "that come directly from NatCap output."
        ),
    },
    {
        "id": "ownership_coarseness_30m",
        "summary": "Ownership filter is approximate at 30 m resolution",
        "detail": (
            "Empirically 69.6 percent of Bexar County parcels are sub-pixel at "
            "30 m (median 0.172 acres vs the 0.222-acre pixel; see "
            "docs/research/ownership/PHASE_0_INVESTIGATION.md). The ownership "
            "mask is reliable for large tracts (parks, civic campuses, school "
            "districts, military installations) and pixelated for residential "
            "subdivisions. Ownership-constrained placement is exploratory, not "
            "parcel-perfect; the sidebar caption when the filter is active "
            "names this caveat."
        ),
    },
    {
        "id": "ownership_vacancy_exempt_keyed",
        "summary": "Vacancy keys on the EX-X total-exemption flag, not a canonical 'vacant land' definition",
        "detail": (
            "The naive 'State_cd C* OR ImprVal == 0' union over-catches "
            "tax-exempt civic / church / university built parcels because "
            "EX-X*-flagged improvements are unassessed (not absent). The "
            "locked rule is 'State_cd C* OR (NOT EX-X*-exempt AND "
            "ImprVal == 0)' — a deliberate methodology choice empirically "
            "supported by the per-token cluster analysis in "
            "docs/research/ownership/PHASE_0_INVESTIGATION.md (EX-X* tokens "
            "land at 15-30 percent built-but-zero; partial-exemption tokens "
            "below 0.5 percent), NOT a canonical 'vacant land' definition "
            "from BCAD."
        ),
    },
    {
        "id": "region_local_spillover_reach_models",
        "summary": "Region-local readings undercount reach-model spillover",
        "detail": (
            "For region-constrained scenarios, the region-local column clips "
            "three reach models — UCM cooling (~600 m), UNA nature access "
            "(~800 m), UMH mental-health exposure (~300 m) — to pixels or "
            "population inside the selected boundary. Effects produced by "
            "in-region conversions that propagate just outside the boundary "
            "are reflected in the citywide column but NOT in the region-local "
            "column. Boundary treatment 'option (b)' per "
            "docs/internal/REGION_LOCAL_METRICS_SPEC.md."
        ),
    },
    {
        "id": "compound_uncertainty_region_ownership",
        "summary": "Region-local + ownership-filtered scenarios stack reach + coarseness uncertainties",
        "detail": (
            "When a scenario is BOTH region-constrained and ownership-filtered, "
            "the region-local readings inherit the reach-model spillover "
            "(UCM / UNA / UMH) AND the 30 m ownership coarseness. Neither "
            "feature's own caption captures the stack — the compound is "
            "surfaced as a divergence to keep the disclosure honest. "
            "Read region-local numbers under this compound as exploratory "
            "lower bounds; the parcel-level ownership mask is only reliable "
            "for large tracts inside the region."
        ),
    },
    {
        "id": "displayed_validated_exploratory_taxonomy",
        "summary": "The displayed / validated / exploratory taxonomy itself is a divergence",
        "detail": (
            "Each metric carries one of four validation states — 'NatCap "
            "published value', 'approx NatCap method', 'approx Aligned "
            "method', 'Prototype' — drawn from the locked 4-state vocabulary "
            "in docs/internal/NATCAP_ALIGNMENT.md. The taxonomy is the "
            "prototype's framing for honest re-use of NatCap and InVEST "
            "outputs; canonical InVEST does NOT categorize its outputs this "
            "way. The per-metric mapping lives in NATCAP_ALIGNMENT.md."
        ),
    },
    {
        "id": "ownership_rule_derived",
        "summary": "Ownership classes are rule-derived, not an authoritative title registry",
        "detail": (
            "Ownership classes (City / County / State-federal / School / "
            "University / Private / Unknown) are derived by parsing the BCAD "
            "parcel attributes (Owner free-text + Exempts codes) with regex "
            "rules locked in docs/internal/OWNERSHIP_FEASIBILITY_PROFILING.md. "
            "The classifier was area-weighted-validated at 99.9% of public "
            "acreage classifying cleanly under the original 6-way split (see "
            "the feasibility doc); the School / University Split addendum "
            "(2026-06-01) tightened the school rule to ISD-only and put "
            "private campuses (Trinity, St. Mary's, OLLU) into a separate "
            "'University' class kept OUT of the 'Publicly-owned land' rollup. "
            "Classes are NOT validated against an authoritative title "
            "registry. The filter is a planning screen — useful for "
            "narrowing where a hypothetical conversion could land — not a "
            "substitute for verified ownership data."
        ),
    },
]


@dataclass
class BundleSpec:
    """Everything the bundle assembler needs. The app.py caller fills this from
    runtime state; the module never imports app.py."""
    # identity / provenance
    city_name: str
    city_slug: str
    crs: str                 # e.g. "EPSG:5070"
    pixel_size_m: int
    scenario_id: str
    scenario_label: str
    scenario_description: str
    provenance: str          # one of PROVENANCE_*
    generator: dict          # polymorphic generator block (caller builds)
    git_commit: str
    scenario_schema_version: int
    is_sa: bool

    # rasterio profile (from the reference compound grid) for writing arrays
    raster_profile: dict

    # prototype-grid rasters (numpy). None where not applicable.
    scenario_lulc_compound: Optional[np.ndarray] = None
    baseline_lulc_compound: Optional[np.ndarray] = None
    scenario_lulc_nlcdtree: Optional[np.ndarray] = None
    baseline_lulc_nlcdtree: Optional[np.ndarray] = None
    scenario_ndvi: Optional[np.ndarray] = None
    baseline_ndvi: Optional[np.ndarray] = None

    # shared input file paths on disk (copied verbatim into the bundle)
    pop_path: Optional[str] = None
    et_path: Optional[str] = None
    soil_path: Optional[str] = None
    block_groups_path: Optional[str] = None
    ucm_table_path: Optional[str] = None
    una_table_path: Optional[str] = None
    carbon_table_path: Optional[str] = None
    cn_table_path: Optional[str] = None

    # fixed-alternative (deferred) case
    compound_models_available: bool = True
    fixed_alt_source_raster_path: Optional[str] = None

    # Region Selection Phase 1 — the structured region_selection block carried
    # in evaluate_scenario's result dict. None for citywide scenarios; for
    # region-selected Explorer scenarios it carries {mode, layer, selected_ids
    # (label values, not positional), selected_area_acres,
    # eligible_pixels_in_region}. Flows into metadata.json's scenario block
    # verbatim. Per-model validation states are NOT affected — region narrows
    # placement only; the engine is unchanged.
    region_selection: Optional[dict] = None
    # Ownership Integration Commit 3 + Scenario Record Pass — structured
    # ownership block composed by the caller from the bare mode string
    # (results['ownership_filter']) + OWNERSHIP_MODES + CITIES config. Shape:
    # {mode, label, allowed_classes (list[int] of raster codes), source,
    # data_date}, or None for citywide / filter-inactive. The in-memory
    # results dict still carries the bare mode string for all existing
    # consumers; the rich shape lives only in the export bundle so
    # metadata.json is self-describing. Per-model validation states unchanged.
    ownership_filter: Optional[dict] = None
    # Region-Local Metrics Commit 3 — region-clipped per-metric values for
    # region scenarios; None for citywide. Carries the locked boundary
    # treatment (option (b) from the spec) and the per-model treatment table
    # (clip / caveat / reach_m) so downstream readers see how each metric
    # decomposes. None for non-region scenarios.
    region_local: Optional[dict] = None
    region_local_treatment: Optional[dict] = None
    # Honesty-Surface Pass Commit 3 — generator params: the slider values
    # and other inputs that produced this scenario, so a downstream consumer
    # can reconstruct it. Caller fills from results + sidebar state.
    generator_params: Optional[dict] = None
    # The complete Source-line string the dashboard renders (e.g.
    # 'Explorer-generated · selected-region placement · vacant publicly-owned
    # land'). Caller-computed from app.py's `_PROVENANCE_HEADER_INFO` so this
    # module stays Streamlit-agnostic.
    source_label: Optional[str] = None

    # args constants (per-city)
    uhi_max_c: float = 11.0
    t_ref_c: float = 35.0
    t_air_average_radius_m: int = 600
    green_area_cooling_distance_m: int = 450
    cc_weight_shade: float = 0.6
    cc_weight_albedo: float = 0.2
    cc_weight_eti: float = 0.2
    una_demand_m2: float = 16.7
    una_radius_m: int = 800
    una_decay: str = "dichotomy"
    design_storm_mm: float = 157.0
    umh_search_radius_m: int = 300
    umh_rr_depression: float = 0.96
    umh_rr_anxiety: float = 0.97
    umh_bir_depression: float = 0.21
    umh_bir_anxiety: float = 0.19
    umh_cost_depression: float = 8467.0
    umh_cost_anxiety: float = 5765.0


# ── raster / vector synthesis helpers (all in-memory) ────────────────────────

def _raster_bytes(arr: np.ndarray, profile: dict, dtype: str, nodata) -> bytes:
    """Serialize a numpy array as a single-band GeoTIFF (bytes) using the
    reference grid's transform/CRS."""
    from rasterio.io import MemoryFile

    prof = dict(
        driver="GTiff",
        height=profile["height"], width=profile["width"],
        count=1, dtype=dtype, crs=profile["crs"],
        transform=profile["transform"], compress="deflate",
    )
    if nodata is not None:
        prof["nodata"] = nodata
    with MemoryFile() as mf:
        with mf.open(**prof) as ds:
            ds.write(arr.astype(dtype), 1)
        return mf.read()


def _gdf_to_gpkg_bytes(gdf) -> bytes:
    """Write a GeoDataFrame to GPKG (SQLite needs a real path) and return bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    path = tmp.name
    tmp.close()
    try:
        gdf.to_file(path, driver="GPKG")
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def _bbox_aoi_gdf(profile: dict, crs: str):
    """A single rectangle polygon covering the reference grid extent (the
    'prototype extent' AOI)."""
    import geopandas as gpd
    from shapely.geometry import box

    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    minx, maxy = t.c, t.f
    maxx = minx + w * t.a
    miny = maxy + h * t.e        # t.e is negative
    return gpd.GeoDataFrame(
        {"id": [1], "name": ["prototype_extent"]},
        geometry=[box(minx, miny, maxx, maxy)], crs=crs,
    )


def _prevalence_gdf(profile: dict, crs: str, risk_rate: float):
    """A single-polygon baseline-prevalence vector for UMH with the required
    `risk_rate` field (uniform — the prototype uses a scalar BIR)."""
    import geopandas as gpd
    from shapely.geometry import box

    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    minx, maxy = t.c, t.f
    maxx = minx + w * t.a
    miny = maxy + h * t.e
    return gpd.GeoDataFrame(
        {"risk_rate": [risk_rate]},
        geometry=[box(minx, miny, maxx, maxy)], crs=crs,
    )


# ── per-model args builders (paths are bundle-root-relative) ─────────────────
_P_SCEN_COMPOUND = "inputs/prototype/scenario_lulc_evaluated_30m_5070.tif"
_P_BASE_COMPOUND = "inputs/prototype/baseline_lulc_evaluated_30m_5070.tif"
_P_SCEN_NLCDTREE = "inputs/prototype/scenario_lulc_nlcdtree_30m_5070.tif"
_P_BASE_NLCDTREE = "inputs/prototype/baseline_lulc_nlcdtree_30m_5070.tif"
_P_SCEN_NDVI = "inputs/prototype/scenario_ndvi_30m_5070.tif"
_P_BASE_NDVI = "inputs/prototype/baseline_ndvi_30m_5070.tif"
_P_POP = "inputs/shared/population.tif"
_P_ETO = "inputs/shared/ref_eto.tif"
_P_SOIL = "inputs/shared/soil_hydrologic_group.tif"
_P_AOI_EXTENT = "inputs/shared/aoi_prototype_extent.gpkg"
_P_AOI_BLOCKGROUPS = "inputs/shared/aoi_natcap_block_groups.gpkg"
_P_PREV_DEP = "inputs/shared/baseline_prevalence_depression.gpkg"
_P_PREV_ANX = "inputs/shared/baseline_prevalence_anxiety.gpkg"
_P_UCM_TBL = "inputs/biophysical/ucm__nlcd_nlud_tree.csv"
_P_UNA_TBL = "inputs/biophysical/una__nlcd_nlud_tree.csv"
_P_CARBON_TBL = "inputs/biophysical/carbon__nlcd_nlud_tree.csv"
_P_CN_TBL = "inputs/biophysical/biophys_floodmitig_sa.csv"


def _ucm_args(s: BundleSpec) -> dict:
    return {
        "workspace_dir": "./workspace_ucm",
        "results_suffix": "ee_export",
        "n_workers": "-1",
        "lulc_raster_path": _P_SCEN_COMPOUND,
        "ref_eto_raster_path": _P_ETO,
        "aoi_vector_path": _P_AOI_EXTENT,
        "biophysical_table_path": _P_UCM_TBL,
        "green_area_cooling_distance": str(s.green_area_cooling_distance_m),
        "t_air_average_radius": str(s.t_air_average_radius_m),
        "t_ref": str(s.t_ref_c),
        "uhi_max": str(s.uhi_max_c),
        "do_energy_valuation": False,
        "do_productivity_valuation": False,
        "cc_method": "factors",
        "cc_weight_shade": str(s.cc_weight_shade),
        "cc_weight_albedo": str(s.cc_weight_albedo),
        "cc_weight_eti": str(s.cc_weight_eti),
    }


def _una_args(s: BundleSpec) -> dict:
    return {
        "workspace_dir": "./workspace_una",
        "results_suffix": "ee_export",
        "n_workers": "-1",
        "lulc_raster_path": _P_SCEN_COMPOUND,
        "lulc_attribute_table": _P_UNA_TBL,
        "population_raster_path": _P_POP,
        "admin_boundaries_vector_path": _P_AOI_BLOCKGROUPS,
        "urban_nature_demand": str(s.una_demand_m2),
        "decay_function": s.una_decay,
        "search_radius_mode": "uniform radius",
        "search_radius": str(s.una_radius_m),
        "aggregate_by_pop_group": False,
    }


def _ufr_args(s: BundleSpec) -> dict:
    # built_infrastructure omitted: SA has no damage table (Path C).
    return {
        "workspace_dir": "./workspace_ufr",
        "results_suffix": "ee_export",
        "n_workers": "-1",
        "aoi_watersheds_path": _P_AOI_EXTENT,
        "rainfall_depth": str(s.design_storm_mm),
        "lulc_path": _P_SCEN_NLCDTREE,
        "soils_hydrological_group_raster_path": _P_SOIL,
        "curve_number_table_path": _P_CN_TBL,
    }


def _carbon_args(s: BundleSpec) -> dict:
    # bas + alt in one run; valuation off (stock change only).
    return {
        "workspace_dir": "./workspace_carbon",
        "results_suffix": "ee_export",
        "n_workers": "-1",
        "lulc_bas_path": _P_BASE_COMPOUND,
        "lulc_alt_path": _P_SCEN_COMPOUND,
        "carbon_pools_path": _P_CARBON_TBL,
        "calc_sequestration": True,
        "do_valuation": False,
    }


def _umh_args(s: BundleSpec, condition: str) -> dict:
    rr = s.umh_rr_depression if condition == "depression" else s.umh_rr_anxiety
    cost = s.umh_cost_depression if condition == "depression" else s.umh_cost_anxiety
    prev = _P_PREV_DEP if condition == "depression" else _P_PREV_ANX
    return {
        "workspace_dir": f"./workspace_umh_{condition}",
        "results_suffix": f"ee_export_{condition}",
        "n_workers": "-1",
        "aoi_path": _P_AOI_EXTENT,
        "population_raster": _P_POP,
        "search_radius": str(s.umh_search_radius_m),
        "effect_size": str(rr),
        "baseline_prevalence_vector": prev,
        "health_cost_rate": str(cost),
        "model_option": "ndvi",
        "ndvi_base": _P_BASE_NDVI,
        "ndvi_alt": _P_SCEN_NDVI,
    }


def _raster_lineage_for_city(s: BundleSpec) -> dict:
    """Per-raster lineage (source / vintage or pull-date / methodology) for the
    inputs the bundle was built on. Not a catalog — that lives in
    `docs/internal/DATA_INVENTORY.md`. This surfaces the rasters whose
    provenance carries honest-disclosure weight at export time (Honesty-
    Surface Pass Commit 3).
    """
    if s.is_sa:
        return {
            "lulc_compound": {
                "source": "NatCap (Stanford Natural Capital Project), Vibrant Land for SA project",
                "vintage": "2024-08",
                "methodology": (
                    "Compound NLCD x NLUD x tree-canopy 4-digit code (1,984 "
                    "classes); see docs/internal/CITY_PARITY.md SA LULC."
                ),
            },
            "nlcd_2021": {
                "source": "MRLC NLCD 2021 (legacy product)",
                "vintage": "2021",
                "methodology": (
                    "21-class national land-cover dataset; used for the "
                    "NLCD-reduced views consumed by UFR / food / NDVI."
                ),
            },
            "ownership_public_vacant_30m": {
                "source": "Bexar County GIS / BCAD — https://maps.bexar.org/arcgis/rest/services/Parcels/MapServer/0",
                "pull_date": "2026-05-31",
                "methodology": (
                    "Full-county pull (710,772 parcels, EOD-confirmed). "
                    "is_public = government-owned (city/county/state/federal/"
                    "ISD/river_auth). is_vacant = State_cd C* OR (NOT EX-X* "
                    "exempt AND ImprVal == 0) — empirically grounded in the "
                    "per-token cluster analysis in "
                    "docs/research/ownership/PHASE_0_INVESTIGATION.md."
                ),
                "license": "not explicitly stated; Bexar County GIS / BCAD; attribution cited",
            },
            "council_districts": {
                "source": "City of San Antonio Open Data Portal — Redistricted Council Districts 2022",
                "vintage": "2022 (boundaries effective for May 2023 municipal election)",
                "license": "not explicitly stated; City of San Antonio Open Data Portal; attribution cited",
            },
            "acs_block_groups": {
                "source": "NatCap-curated ACS block groups for SA (Vibrant Land equity framing)",
                "vintage": "2024-08",
                "methodology": (
                    "1,124 polygons covering the City of San Antonio; finer "
                    "granularity than TIGER 2020 Bexar tracts."
                ),
            },
        }
    return {
        "lulc": {
            "source": "InVEST UFR / UNA sample bundle for MN downtown",
            "vintage": "NLCD 2021 (legacy product)",
            "methodology": (
                "21-class national land-cover; identical raster for "
                "cooling/UNA and a separate-but-same-content raster for UFR."
            ),
        },
        "buildings_typed": {
            "source": "InVEST UFR sample shapefile (MN)",
            "methodology": (
                "Pre-typed building footprints (commercial/residential/"
                "industrial); 447 building pixels at 96% typing coverage."
            ),
        },
        "tracts": {
            "source": "TIGER 2020 census tracts (downtown subset)",
            "vintage": "2020",
            "methodology": "27 tracts intersecting the InVEST sample AOI.",
        },
    }


def _build_metadata(s: BundleSpec, args_files: dict) -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "prototype": {
            "name": "Ecosystem Explorer",
            "git_commit": s.git_commit,
            "scenario_schema_version": s.scenario_schema_version,
        },
        "city": {
            "name": s.city_name,
            "crs_used_by_prototype": s.crs,
            "pixel_size_m": s.pixel_size_m,
        },
        "scenario": {
            "scenario_id": s.scenario_id,
            "label": s.scenario_label,
            "provenance": s.provenance,
            "description": s.scenario_description,
            # Region Selection Phase 1 — `source_label` is the augmented Source
            # line as the dashboard renders it (e.g. 'Explorer-generated ·
            # selected-region placement'), passed in pre-composed by the
            # caller. `region_selection` is the structured carrier (mode /
            # layer / selected_ids / selected_area_acres /
            # eligible_pixels_in_region). Both are None for citywide scenarios.
            "source_label": s.source_label,
            "region_selection": s.region_selection,
            "ownership_filter": s.ownership_filter,
            "region_local": s.region_local,
            "region_local_treatment": s.region_local_treatment,
            "boundary_treatment": (
                "option_b_clip_with_spillover_caveat"
                if s.region_local is not None else None
            ),
            # Honesty-Surface Pass Commit 2 — the locked seed list of
            # known divergences travels with every bundle. Commit 4's
            # completeness assertion guarantees no entry is silently dropped.
            "known_divergences": KNOWN_DIVERGENCES,
        },
        "scenario_rasters": _scenario_rasters_block(s),
        "generator": dict(
            s.generator,
            type=s.generator.get("type", s.provenance),
            # Honesty-Surface Pass Commit 3 — generator params that produced
            # this scenario, so a downstream consumer can reconstruct it.
            params=(s.generator_params or {}),
        ),
        # Honesty-Surface Pass Commit 3 — per-raster lineage (source / pull
        # date / methodology) for the inputs this bundle was built on. Lives
        # at the top of metadata.json so a downstream reviewer can audit
        # provenance without scanning the scenario block.
        "raster_lineage": _raster_lineage_for_city(s),
        "aoi": {
            "prototype_extent": {
                "path": _P_AOI_EXTENT,
                "role": "AOI for UCM / UFR (watersheds) / UMH",
                "reason": "Bounding-box polygon of the prototype LULC grid — a "
                          "citywide aggregation footprint; the prototype reports "
                          "citywide means.",
            },
            "natcap_block_groups": ({
                "path": _P_AOI_BLOCKGROUPS,
                "role": "admin_boundaries for UNA",
                "reason": "NatCap's ACS block-group polygons — the framing NatCap "
                          "uses for SA equity analysis (Vibrant Land).",
            } if s.is_sa and s.block_groups_path else None),
        },
        "compound_models_available": s.compound_models_available,
        "compound_models_unavailable_reason": (
            None if s.compound_models_available else
            "NatCap fixed alternative scenario: compound (NLCD×NLUD×tree) LULC "
            "not provided by NatCap — only flood-encoded (NLCD×tree) source "
            "exists. Carbon/UCM/UNA cannot be generated. See OPEN_QUESTIONS.md."
        ),
        "validation": _VALIDATION,
        "args_files": {"prototype_grid": args_files},
        "notes": [
            "UCM export is biophysical-cooling only (do_energy_valuation=False); "
            "the prototype's Cooling Energy Savings $-card is NOT reproduced here.",
            "Carbon args run baseline (lulc_bas) vs scenario (lulc_alt) in one "
            "execution; do_valuation=False (stock change only).",
            "UCM/UNA produce one result per LULC; to reproduce the scenario-vs-"
            "baseline delta, re-run each with lulc pointed at the baseline raster "
            "(see README).",
            "UMH: two args files emitted (depression, anxiety) using a synthetic "
            "uniform prevalence vector + synthetic NDVI proxy; canonical "
            "execution verified on baseline (D1 Phase 3).",
        ],
    }


def _scenario_rasters_block(s: BundleSpec) -> dict:
    if not s.compound_models_available:
        return {
            "source_original": {
                "path": "inputs/source/scenario_lulc_original_10m.tif",
                "crs": "WGS84 Albers (NatCap flood input)",
                "resolution_m": 10,
                "encoding": "nlcd_tree_3tier",
                "role": "NatCap flood-model source raster (only encoding provided)",
            },
            "prototype_evaluated": None,
            "resampling": {"method": "none",
                           "reason": "compound LULC unavailable — see metadata."},
        }
    return {
        "source_original": None,
        "prototype_evaluated": {
            "path": _P_SCEN_COMPOUND,
            "crs": s.crs,
            "resolution_m": s.pixel_size_m,
            "encoding": "compound_nlcd_nlud_tree",
            "role": "Prototype-evaluated scenario LULC (UCM/UNA/Carbon); the "
                    "NLCD×tree reduction is exported separately for UFR.",
        },
        "resampling": {
            "method": "n/a (prototype-built)",
            "reason": "Explorer/Optimizer/baseline scenarios are built on the "
                      "prototype's 30 m EPSG:5070 compound grid directly.",
        },
    }


def _readme(s: BundleSpec, args_files: dict) -> str:
    lines = [
        f"# Ecosystem Explorer → InVEST export bundle",
        "",
        f"- **City:** {s.city_name}  ",
        f"- **Scenario:** {s.scenario_label} (`{s.scenario_id}`, {s.provenance})  ",
        f"- **Prototype grid:** {s.crs}, {s.pixel_size_m} m  ",
        f"- **Exported:** {datetime.now(timezone.utc).isoformat()}  ",
        f"- **Prototype commit:** `{s.git_commit}`",
        "",
        "This bundle lets you run canonical **InVEST 3.19.0** on a scenario "
        "discovered in Ecosystem Explorer. The prototype is the scenario "
        "discovery engine; canonical InVEST is the validation endpoint.",
        "",
        "## Structure",
        "```",
        "inputs/prototype/   prototype-evaluated rasters (LULC compound + NLCD×tree + NDVI)",
        "inputs/shared/      population, reference ET, soil group, AOIs, prevalence vectors",
        "inputs/biophysical/ per-model biophysical / curve-number / carbon-pool tables",
        "args/prototype_grid/ one args.json per InVEST model",
        "metadata.json       provenance, generator params, per-model validation state",
        "```",
        "",
        "## Running a model",
        "From the **bundle root** (paths in the args files are bundle-root-relative):",
        "```bash",
        "python -c \"import json; from natcap.invest import urban_cooling_model as m; "
        "m.execute(json.load(open('args/prototype_grid/urban_cooling_args.json')))\"",
        "```",
        "Replace the module + args path for each model:",
    ]
    for k, v in args_files.items():
        lines.append(f"- **{k.upper()}** → `{v}`")
    lines += [
        "",
        "## Scenario-vs-baseline deltas",
        "Carbon runs baseline-vs-scenario in one execution (`lulc_bas` / "
        "`lulc_alt`). UCM and UNA produce one result per LULC — to get the delta, "
        "run each twice, pointing `lulc_raster_path` at "
        "`inputs/prototype/scenario_lulc_evaluated_30m_5070.tif` and then at "
        "`inputs/prototype/baseline_lulc_evaluated_30m_5070.tif`.",
        "",
        "## Known limitations",
        "- **UCM:** biophysical-cooling only (`do_energy_valuation=False`); the "
        "prototype's Cooling Energy Savings $-metric is not reproduced.",
        "- **UFR:** SA uses NatCap's NLCD×tree curve-number table; damage "
        "valuation is disabled (no SA damage table). Uses a bounding-box AOI as "
        "the watershed footprint.",
        "- **UMH:** two args files (depression, anxiety) emitted using a "
        "synthetic uniform baseline-prevalence vector "
        "(`risk_rate` = CDC ever-diagnosed prevalence) and a **synthetic NDVI "
        "proxy** (per-land-cover, not satellite). Algorithmic parity is "
        "validated and canonical execution on the baseline is verified — these "
        "are input-quality caveats, not parity gaps.",
        "- **AOI:** the prototype-extent AOI is a bounding-box polygon, not a "
        "hydrologic watershed or administrative boundary.",
        "- See `metadata.json` → `validation` for per-model parity status.",
    ]
    if not s.compound_models_available:
        lines += [
            "",
            "## NatCap fixed alternative scenario — partial export",
            "This is a NatCap fixed *alternative* scenario, shipped only in flood "
            "encoding (NLCD×tree). The compound (NLCD×NLUD×tree) LULC needed for "
            "Carbon / UCM / UNA was not provided by NatCap, so **only UFR args are "
            "included**. See `metadata.json` and OPEN_QUESTIONS.md.",
        ]
    return "\n".join(lines) + "\n"


def build_invest_bundle(spec: BundleSpec) -> bytes:
    """Assemble the bundle and return it as zip bytes."""
    import rasterio  # noqa: F401  (ensures rasterio present before raster writes)

    prof = spec.raster_profile
    crs = spec.crs

    args_files: dict[str, str] = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        if spec.compound_models_available:
            # ── prototype rasters ──
            zf.writestr(_P_SCEN_COMPOUND,
                        _raster_bytes(spec.scenario_lulc_compound, prof, "int16", -1))
            zf.writestr(_P_BASE_COMPOUND,
                        _raster_bytes(spec.baseline_lulc_compound, prof, "int16", -1))
            # NODATA -128: reduce_compound_to_nlcd_tree emits the prototype's
            # NODATA sentinel (-128) for outside-boundary pixels. The prototype
            # tolerates unmapped lucodes by filtering CN>0; canonical InVEST UFR
            # does NOT — it raises if any non-nodata lulc value isn't in the CN
            # table. So mark -128 as the raster nodata so InVEST masks it.
            zf.writestr(_P_SCEN_NLCDTREE,
                        _raster_bytes(spec.scenario_lulc_nlcdtree, prof, "int16", -128))
            zf.writestr(_P_BASE_NLCDTREE,
                        _raster_bytes(spec.baseline_lulc_nlcdtree, prof, "int16", -128))
            zf.writestr(_P_SCEN_NDVI,
                        _raster_bytes(spec.scenario_ndvi, prof, "float32", -1.0))
            zf.writestr(_P_BASE_NDVI,
                        _raster_bytes(spec.baseline_ndvi, prof, "float32", -1.0))

            # ── shared inputs (copied from disk) ──
            for src, dst in [(spec.pop_path, _P_POP), (spec.et_path, _P_ETO),
                             (spec.soil_path, _P_SOIL)]:
                if src and os.path.exists(src):
                    with open(src, "rb") as fh:
                        zf.writestr(dst, fh.read())
            # synthesized AOIs + prevalence vectors
            zf.writestr(_P_AOI_EXTENT, _gdf_to_gpkg_bytes(_bbox_aoi_gdf(prof, crs)))
            if spec.is_sa and spec.block_groups_path and os.path.exists(spec.block_groups_path):
                with open(spec.block_groups_path, "rb") as fh:
                    zf.writestr(_P_AOI_BLOCKGROUPS, fh.read())
            zf.writestr(_P_PREV_DEP, _gdf_to_gpkg_bytes(
                _prevalence_gdf(prof, crs, spec.umh_bir_depression)))
            zf.writestr(_P_PREV_ANX, _gdf_to_gpkg_bytes(
                _prevalence_gdf(prof, crs, spec.umh_bir_anxiety)))

            # ── biophysical tables (copied from disk) ──
            for src, dst in [(spec.ucm_table_path, _P_UCM_TBL),
                             (spec.una_table_path, _P_UNA_TBL),
                             (spec.carbon_table_path, _P_CARBON_TBL),
                             (spec.cn_table_path, _P_CN_TBL)]:
                if src and os.path.exists(src):
                    with open(src, "rb") as fh:
                        zf.writestr(dst, fh.read())

            # ── args files ──
            _w = lambda name, d: (zf.writestr(f"args/prototype_grid/{name}",
                                              json.dumps(d, indent=2)),
                                  f"args/prototype_grid/{name}")[1]
            args_files["ucm"] = _w("urban_cooling_args.json", _ucm_args(spec))
            args_files["una"] = _w("urban_nature_access_args.json", _una_args(spec))
            args_files["ufr"] = _w("urban_flood_risk_mitigation_args.json", _ufr_args(spec))
            args_files["carbon"] = _w("carbon_args.json", _carbon_args(spec))
            args_files["umh_depression"] = _w("urban_mental_health_depression_args.json",
                                              _umh_args(spec, "depression"))
            args_files["umh_anxiety"] = _w("urban_mental_health_anxiety_args.json",
                                           _umh_args(spec, "anxiety"))
        else:
            # ── NatCap fixed alternative: flood source + UFR args only ──
            if spec.fixed_alt_source_raster_path and os.path.exists(spec.fixed_alt_source_raster_path):
                with open(spec.fixed_alt_source_raster_path, "rb") as fh:
                    zf.writestr("inputs/source/scenario_lulc_original_10m.tif", fh.read())
            if spec.soil_path and os.path.exists(spec.soil_path):
                with open(spec.soil_path, "rb") as fh:
                    zf.writestr(_P_SOIL, fh.read())
            if spec.cn_table_path and os.path.exists(spec.cn_table_path):
                with open(spec.cn_table_path, "rb") as fh:
                    zf.writestr(_P_CN_TBL, fh.read())
            zf.writestr(_P_AOI_EXTENT, _gdf_to_gpkg_bytes(_bbox_aoi_gdf(prof, crs)))
            # UFR args still reference the flood-encoded source (note in metadata).
            ufr = _ufr_args(spec)
            ufr["lulc_path"] = "inputs/source/scenario_lulc_original_10m.tif"
            args_files["ufr"] = "args/prototype_grid/urban_flood_risk_mitigation_args.json"
            zf.writestr(args_files["ufr"], json.dumps(ufr, indent=2))

        # ── metadata + README (always) ──
        meta = _build_metadata(spec, args_files)
        zf.writestr("metadata.json", json.dumps(meta, indent=2))
        zf.writestr("README.md", _readme(spec, args_files))

    return buf.getvalue()


def bundle_filename(spec: BundleSpec) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = spec.scenario_id.replace("/", "_").replace(" ", "_")
    return f"ecosystem_explorer_export_{spec.city_slug}_{sid}_{ts}.zip"
