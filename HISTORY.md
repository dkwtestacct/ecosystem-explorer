# HISTORY.md — Durable historical context

Content extracted from CLAUDE.md during the post-Brief-31 trim
(2026-05-25) when CLAUDE.md exceeded its 40k-char recommended size.

This file holds content that's durably useful for future sessions but
not on the critical-path orientation reading. Categories:

1. **Schema version history** — every `SCENARIO_SCHEMA_VERSION` bump's
   rationale (current value + one-line summary stays in CLAUDE.md).

2. **Retired infrastructure** — components, conventions, or model
   pieces that were removed or replaced. The "what was retired and
   why" context that explains current decisions.

3. **Completed-workstream specifics** — per-brief magnitude evidence
   and implementation detail extracted from CLAUDE.md. The canonical
   per-brief reasoning lives in DESIGN_NOTES.md; this section preserves
   anything from CLAUDE.md that wasn't already duplicated there.

For current-state coding conventions, architecture, data files, and
constants, see CLAUDE.md. For per-brief reasoning, see DESIGN_NOTES.md.
For commit-level changes, see git log.

---

## Schema version log

Full per-bump rationale for every `SCENARIO_SCHEMA_VERSION` increase.
The current value and a one-line summary live in CLAUDE.md.

- **7→8** — UCM rework: ET fix, Gaussian convolution, canonical energy formula.
- **8→9** — ET nodata sentinel masked.
- **9→10** — Full Geofabrik OSM road network (62 % AOI).
- **10→11** — Option B road filter (~29 % AOI).
- **11→12** — `NATURE_RADIUS_CAP_M = 1000 m` fixes nature-access saturation; `BASELINE_CN` now dynamically computed at module load.
- **12→13** — `load_data` parameterized via `city_cfg` path keys; Minneapolis Full activated.
- **13→14** — InVEST Urban Mental Health v3.19.0 added (`preventable_mh_cases` + `avoided_mh_cost_usd` as new surrogate targets, replaces Urban Wellbeing Score metric card).
- **14→15** — San Antonio activated with full pipeline: SSURGO TX029 + Census Bexar + CGIAR ET0 + TIGER 48 + Geofabrik TX OSM; new EPA Social Cost of Carbon dollar metric in Economic row; pre-flight data-check function added; `PIXEL_AREA_ACRES` harmonized to 0.2224 globally.
- **15→16** — SA cooling biophysical table tuned for Köppen BSh — initially landed with classes 21, 41, 42, 52, 81 adjusted from prior MN-copy placeholder, anchored on eddy-covariance Kc measurements per Pôças et al. 2017 + FAO-56 + Stewart & Oke 2012.
- **16→17** — Revert SA class 21 Kc to MN's 0.516 — class 21 was incorrectly tuned in `23328b5` despite the user's explicit Stage-3 instruction to leave it alone. Authorized scope was 4 classes [41, 42, 52, 81]. Restores bug-discipline correctness; SA cooling value drops slightly from the $39.44M measurement on the 16-baseline. See `data/sa/cooling/biophysical_table_sources.md` for the class-21 semantic-divergence rationale.
- **17→18, 18→19, 19→20** — Brief sequence bumps not separately documented here.
- **20→21** — Brief 23 per-city UFR rainfall depth: MN 100 mm canonical, SA 157 mm canonical — every flood metric shifts in both cities.
- **21→22** — Brief 27 foundational SA compound LULC adoption — NatCap `lulc_overlay_3857.tif` reprojected to EPSG:5070 + nearest-neighbor at 30 m produces `data/sa/flood/land_use_compound_sa.tif`; reduced to NLCD via `lulc_crosswalk.csv` for the existing per-NLCD biophysical tables. SA baseline drift <0.5 % on every headline; MN untouched. `DEFAULT_FF_LUCODE=1310`, `DEFAULT_GI_LUCODE=122`, `DEFAULT_HD_LUCODE=341` are the configured fallback compound codes for conversion targets when the source pixel's (NLUD, tree) tuple has no row for the target NLCD; consumed by the load-time `COMPOUND_AFTER_*` lookup arrays. See `DESIGN_NOTES.md` "SA compound LULC integration — foundational decisions".
- **22→23** — Brief 28b SA UCM compound biophysical table (`ucm__nlcd_nlud_tree.csv`) replaces the per-NLCD Köppen-BSh tuning; SA `baseline_hm` 0.2866 → 0.3937 (+37 %) reflecting tree-canopy variation on developed land that per-NLCD couldn't capture; SA `cooling_energy_savings_usd` -77 to -86 % as downstream amplification; MN untouched. `scenario_lulc_ucm` field added to `evaluate_scenario`'s return dict — compound view for SA, same as `scenario_lulc` for MN — so UCM consumers index the right lucode space.
- **23→24** — Brief 29 SA UNA compound biophysical table (`una__nlcd_nlud_tree.csv`) replaces the borrowed-from-MN per-NLCD `LULC_attribute_table_UNA.csv` for SA; SA baseline `nature_access_pct` 89.7 → 94.2 (+5.0 %, +4.5 pp), baseline `people_with_nature_access` +84,486; MN untouched. `scenario_lulc_una` field added to `evaluate_scenario`'s return dict — compound view for SA, same as `scenario_lulc` for MN — mirroring the Brief 28b `scenario_lulc_ucm` pattern. The `URBAN_NATURE_PROPORTION` Python-dict + per-class boolean-mask loop in `_una_supply_percapita` was replaced with a vectorized `urban_nature_arr[scenario_lulc_una]` indexed lookup because the dict pattern would have done 1,984 raster-wide comparisons per call at SA's cardinality. `urban_nature_arr` joins `shade_arr` / `kc_arr` / `albedo_arr` / `green_area_arr` on `CityState`. Three CSV strip sites updated: `compute_scenario_grid`, `compute_lookup_table`, `precompute_scenarios.py`. See `DESIGN_NOTES.md` "SA UNA compound biophysical table adoption".
- **24→25** — Brief 30 SA Carbon four-pool stock framework (`carbon__nlcd_nlud_tree.csv`, 1,984 rows × 27 cols; four pools `c_above` / `c_below` / `c_soil` / `c_dead` in t C/ha) replaces SA's per-conversion-type `CARBON_SEQ_RATES` annual-flow proxy. SA Carbon consumers index `cooling_lulc_compound` directly via a new `scenario_lulc_carbon` field; the `_compute_carbon_four_pool` wrapper computes one-time t CO2 stock change from the LULC delta per the InVEST four-pool framework, matching NatCap's Vibrant Land (Guerry et al. 2023) methodology. **Field rename**: `carbon_tons_co2_yr` → `carbon_tons_co2` (unified key; semantics differ per city — annual flow MN, one-time stock SA). **Dollar metric reframe**: `avoided_carbon_cost_usd` → `carbon_value_usd` with city-conditional dashboard label ("Avoided Carbon Cost"/yr for MN, "Carbon Storage Value" one-time for SA). `EPA_SOCIAL_COST_CARBON=$190/t` (EPA 2023, 2 % discount) is kept untouched; methodology matches Vibrant Land but the SC-CO2 vintage differs from theirs (IWG 2021, $53/t @ 3 %) — same US-government lineage, different vintage, intentional. SA Carbon stock numerically ~30× the prior annual proxy (category-error correction, not a value shift); MN baselines unchanged (zero value divergence across 20 baselines). Three CSV strip sites updated (same as Brief 29): `compute_scenario_grid`, `compute_lookup_table`, `precompute_scenarios.py`. `c_above_arr` / `c_below_arr` / `c_soil_arr` / `c_dead_arr` join the existing per-city arrays on `CityState`. See `DESIGN_NOTES.md` "SA Carbon four-pool framework adoption".


---

## Retired infrastructure

Components, conventions, or model pieces that were removed or replaced.
"What was retired and why" context that helps explain current decisions.
Each entry has a one-line stub in CLAUDE.md pointing here.

### Wellbeing Score (retired; replaced by InVEST UMH preventable cases)

The previous `compute_wellbeing_score` composite metric — plus its
`wgt_ndvi` / `wgt_cooling` / `wgt_nature` sliders and the
`DEFAULT_WGT_*` constants — was removed entirely when InVEST Urban
Mental Health v3.19.0 was integrated (Brief sequence around schema
14). UMH outputs are derived from peer-reviewed effect sizes (Liu et
al. 2023 NDVI exposure RR for depression / anxiety; Li et al. 2025
search radius) rather than user-tunable weights, so there is nothing
to expose in the sidebar. The "Wellbeing Score" UI card is gone; the
"Preventable MH Cases" + "Avoided MH Costs" cards replace it. See
REFERENCE.md "Official InVEST alignment — UMH" for parity status and
divergences (uniform BIR vs. per-admin, Gaussian kernel vs. uniform
buffer, synthetic vs. satellite NDVI).

### Nature Quality Score card (retired)

Previously a population-weighted mean of the 0–1 proxy access score,
computed alongside the old Nature Access proxy as a continuous
companion metric. Removed when Nature Access was reimplemented as
canonical InVEST UNA 2SFCA (2026-05-22) — Quality Score had no
canonical InVEST analog and sensitivity testing
(`UNA_QUALITY_SCORE_SENSITIVITY.md`) showed it behaved as a two-state
"greening vs none" indicator rather than a continuous quality
gradient. The function signature in `calculate_nature_access` still
returns a three-tuple where the middle slot is `0.0` (legacy
placeholder), so call sites are unaffected.

### Full Minneapolis extent — activated 2026-05-09, hidden from UI 2026-05-11

`'Minneapolis Full, MN'` is a live city in `CITIES` but
`available=False`, so it does NOT appear in the sidebar selector.
Reason: per-building-type dollar metrics (Flood Damage Avoided,
Cooling Energy Savings) require InVEST sample buildings with
`type` ∈ {0,1,2,3}, which only cover the downtown extent — Mpls
Full uses OSM polygons with no type codes (Option A), so those
cards degrade to "—". Showing only the downtown city in the UI
keeps the metric coverage complete. All pipeline + rasters +
verified baselines remain in the repo; flip back to `True` once
a typed building dataset exists for the expanded area. Pipeline:
SSURGO via SDA REST API → `process_ssurgo.py` →
`soil_group_mpls_full.tif`; Census 2020 → `process_pop_expanded.py`
→ `pop_mpls_full.tif`; Geofabrik state OSM →
`process_osm_expanded.py` → `roads_mpls_full.geojson` +
`buildings_mpls_full.gpkg`; TIGER 2020 → `tracts_hennepin.shp`.
Schema bumped 12 → 13.

### `load_data` parameterization (2026-05-09)

Historical record of the parameterization transition. Pre-2026-05-09,
`load_data` hardcoded MN file paths and only the MN city was
representable. After: `load_data()` takes `lulc_file`, `soil_file`,
`cooling_lulc_file` from `city_cfg`. Module-level loaders for ET,
energy table, UNA table, buildings, roads, and tracts also read
from `city_cfg`. Biophysical tables (CN + cooling) use a fallback
path via `_resolve_table()` so cities with custom `data_dir`s
(Mpls Full pointing at `data/minneapolis_expanded/`) can still
reference the project-shared tables in `data/flood/` and
`data/cooling/`. EPSG:26915 hardcodes replaced with
`city_cfg['crs']`. This transition is the foundation that made
multi-city support possible (Mpls Full, then SA); the parameterized
signature is now the steady state.

---

## Completed-workstream specifics

Per-brief implementation detail extracted from CLAUDE.md. Canonical
per-brief reasoning lives in DESIGN_NOTES.md; this section preserves
anything from CLAUDE.md that wasn't already duplicated there.

### Streamlit Cloud memory-fit workstream (2026-05-11)

The 1011 keepalive loop OOM on slider interaction was resolved by a
stack of changes:

- float32 downcast of module-level geospatial arrays (population, ET,
  consumption-rate, baseline rasters, precomputed distance fields)
- disk-cached static nature-distance `.npy` artifacts
  (`<city>/precomputed/nature_distance_<lucode>.npy`) under
  `<city_cfg['precomputed_dir']>`
- `@st.cache_resource`-backed `_load_city_runtime_state` so heavy
  per-city work runs at most once per session per city instead of
  every Streamlit rerun
- in-place ops in the `_compute_cc_raw_pure` chain (single scratch
  buffer reused through the entire pipeline)
- uint8 RGB layers + 1024 px-cap downsample in `plot_spatial_map`
  (was allocating ~378 MB transient per rerun on SA's 1713 × 1984
  AOI before the fix)

Together these brought peak memory under Streamlit Cloud's 1 GB
ceiling. SA is the default test bed for any future memory-sensitive
change — if SA fits, MN/Mpls-Full fit by definition (smaller grids).
