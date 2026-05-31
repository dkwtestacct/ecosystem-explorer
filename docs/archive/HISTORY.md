# HISTORY.md — Durable historical context

**Audience:** Archive
**Status:** Historical (durable — still appended)
**Use this for:** Schema-version history, retired infrastructure, completed-workstream specifics
**Do not use this for:** Current conventions (→ ../../CLAUDE.md), per-brief reasoning (→ ../internal/DESIGN_NOTES.md), commit detail (→ git log)
**Source of truth for:** What changed and what was retired, over time

---

Content extracted from ../../CLAUDE.md during the post-Brief-31 trim
(2026-05-25) when ../../CLAUDE.md exceeded its 40k-char recommended size.

This file holds content that's durably useful for future sessions but
not on the critical-path orientation reading. Categories:

1. **Schema version history** — every `SCENARIO_SCHEMA_VERSION` bump's
   rationale (current value + one-line summary stays in ../../CLAUDE.md).

2. **Retired infrastructure** — components, conventions, or model
   pieces that were removed or replaced. The "what was retired and
   why" context that explains current decisions.

3. **Completed-workstream specifics** — per-brief magnitude evidence
   and implementation detail extracted from ../../CLAUDE.md. The canonical
   per-brief reasoning lives in ../internal/DESIGN_NOTES.md; this section preserves
   anything from ../../CLAUDE.md that wasn't already duplicated there.

For current-state coding conventions, architecture, data files, and
constants, see ../../CLAUDE.md. For per-brief reasoning, see ../internal/DESIGN_NOTES.md.
For commit-level changes, see git log.

---

## Schema version log

Full per-bump rationale for every `SCENARIO_SCHEMA_VERSION` increase.
The current value and a one-line summary live in ../../CLAUDE.md.

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
- **17→18** — Placement strategies reformulated to use canonical InVEST quantities (suitability weights derived from the CN table, baseline CC, population, and access score). Commit `14a22e2`.
- **18→19** — SA Urban Cooling args aligned to NatCap canonical: `uhi_max` 3.5 → 11 °C (heat-wave-day scenario per NatCap's curated SA InVEST inputs — 35 °C reference air temp, 11 °C UHI). Commit `f97b693`.
- **19→20** — MN UNA params aligned to NatCap MN-project canonical: demand 250 m²/capita, search radius 1000 m, exponential decay. Commit `12fb92a`.
- **20→21** — Brief 23 per-city UFR rainfall depth: MN 100 mm canonical, SA 157 mm canonical — every flood metric shifts in both cities.
- **21→22** — Brief 27 foundational SA compound LULC adoption — NatCap `lulc_overlay_3857.tif` reprojected to EPSG:5070 + nearest-neighbor at 30 m produces `data/sa/flood/land_use_compound_sa.tif`; reduced to NLCD via `lulc_crosswalk.csv` for the existing per-NLCD biophysical tables. SA baseline drift <0.5 % on every headline; MN untouched. `DEFAULT_FF_LUCODE=1310`, `DEFAULT_GI_LUCODE=122`, `DEFAULT_HD_LUCODE=341` are the configured fallback compound codes for conversion targets when the source pixel's (NLUD, tree) tuple has no row for the target NLCD; consumed by the load-time `COMPOUND_AFTER_*` lookup arrays. See `../internal/DESIGN_NOTES.md` "SA compound LULC integration — foundational decisions".
- **22→23** — Brief 28b SA UCM compound biophysical table (`ucm__nlcd_nlud_tree.csv`) replaces the per-NLCD Köppen-BSh tuning; SA `baseline_hm` 0.2866 → 0.3937 (+37 %) reflecting tree-canopy variation on developed land that per-NLCD couldn't capture; SA `cooling_energy_savings_usd` -77 to -86 % as downstream amplification; MN untouched. `scenario_lulc_ucm` field added to `evaluate_scenario`'s return dict — compound view for SA, same as `scenario_lulc` for MN — so UCM consumers index the right lucode space.
- **23→24** — Brief 29 SA UNA compound biophysical table (`una__nlcd_nlud_tree.csv`) replaces the borrowed-from-MN per-NLCD `LULC_attribute_table_UNA.csv` for SA; SA baseline `nature_access_pct` 89.7 → 94.2 (+5.0 %, +4.5 pp), baseline `people_with_nature_access` +84,486; MN untouched. `scenario_lulc_una` field added to `evaluate_scenario`'s return dict — compound view for SA, same as `scenario_lulc` for MN — mirroring the Brief 28b `scenario_lulc_ucm` pattern. The `URBAN_NATURE_PROPORTION` Python-dict + per-class boolean-mask loop in `_una_supply_percapita` was replaced with a vectorized `urban_nature_arr[scenario_lulc_una]` indexed lookup because the dict pattern would have done 1,984 raster-wide comparisons per call at SA's cardinality. `urban_nature_arr` joins `shade_arr` / `kc_arr` / `albedo_arr` / `green_area_arr` on `CityState`. Three CSV strip sites updated: `compute_scenario_grid`, `compute_lookup_table`, `precompute_scenarios.py`. See `../internal/DESIGN_NOTES.md` "SA UNA compound biophysical table adoption".
- **24→25** — Brief 30 SA Carbon four-pool stock framework (`carbon__nlcd_nlud_tree.csv`, 1,984 rows × 27 cols; four pools `c_above` / `c_below` / `c_soil` / `c_dead` in t C/ha) replaces SA's per-conversion-type `CARBON_SEQ_RATES` annual-flow proxy. SA Carbon consumers index `cooling_lulc_compound` directly via a new `scenario_lulc_carbon` field; the `_compute_carbon_four_pool` wrapper computes one-time t CO2 stock change from the LULC delta per the InVEST four-pool framework, matching NatCap's Vibrant Land (Guerry et al. 2023) methodology. **Field rename**: `carbon_tons_co2_yr` → `carbon_tons_co2` (unified key; semantics differ per city — annual flow MN, one-time stock SA). **Dollar metric reframe**: `avoided_carbon_cost_usd` → `carbon_value_usd` with city-conditional dashboard label ("Avoided Carbon Cost"/yr for MN, "Carbon Storage Value" one-time for SA). `EPA_SOCIAL_COST_CARBON=$190/t` (EPA 2023, 2 % discount) is kept untouched; methodology matches Vibrant Land but the SC-CO2 vintage differs from theirs (IWG 2021, $53/t @ 3 %) — same US-government lineage, different vintage, intentional. SA Carbon stock numerically ~30× the prior annual proxy (category-error correction, not a value shift); MN baselines unchanged (zero value divergence across 20 baselines). Three CSV strip sites updated (same as Brief 29): `compute_scenario_grid`, `compute_lookup_table`, `precompute_scenarios.py`. `c_above_arr` / `c_below_arr` / `c_soil_arr` / `c_dead_arr` join the existing per-city arrays on `CityState`. See `../internal/DESIGN_NOTES.md` "SA Carbon four-pool framework adoption".
- **25→26** — Brief B adds three per-target fallback-pixel diagnostic keys to `evaluate_scenario`'s return dict — `ff_fellback_pixels`, `gi_fellback_pixels`, `hd_fellback_pixels`. For SA these count converted pixels whose source (NLUD, tree-canopy) tuple had no matching crosswalk row and fell back to `DEFAULT_<target>_LUCODE` (1310 / 122 / 341); for MN they're always 0 (no compound conversion path). Not surrogate targets — pure conversion metadata, surfaced in the SA dashboard's Conversion fidelity panel.
- **26→27** — UMH neighborhood-exposure (NE) kernel changed from a Gaussian (σ = search_radius/pixel) to the canonical InVEST UMH 3.19.0 **buffer-mean** (an edge-corrected flat disk of radius search_radius/pixel, via `_convolve_edge_corrected`). Validated to per-pixel parity against `natcap.invest.urban_mental_health.execute()` (compare_umh_invest.py): **MN MAE ≈ 0, r = 1.0** directly; **SA MAE ≈ 0** when the app kernel is fed canonical's own aligned input (the harness's 0.14 % residual on SA's 1713×1984 grid was empirically shown to be large-grid feeding-alignment noise, not a kernel divergence). `preventable_mh_cases` / `avoided_mh_cost_usd` shift ~1.5–3 % for every conversion scenario (the 10 zero-conversion baselines are unchanged); MH card confidence Medium → High. Follows the UMH-validation commit `db94098`. See `../internal/DESIGN_NOTES.md` §6.3 "UMH validation against canonical InVEST 3.19.0".


---

## Retired infrastructure

Components, conventions, or model pieces that were removed or replaced.
"What was retired and why" context that helps explain current decisions.
Each entry has a one-line stub in ../../CLAUDE.md pointing here.

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
../../REFERENCE.md "Official InVEST alignment — UMH" for parity status and
divergences (uniform BIR vs. per-admin, Gaussian kernel vs. uniform
buffer, synthetic vs. satellite NDVI).

### Nature Quality Score card (retired)

Previously a population-weighted mean of the 0–1 proxy access score,
computed alongside the old Nature Access proxy as a continuous
companion metric. Removed when Nature Access was reimplemented as
canonical InVEST UNA 2SFCA (2026-05-22) — Quality Score had no
canonical InVEST analog and sensitivity testing
(`../research/una/UNA_QUALITY_SCORE_SENSITIVITY.md`) showed it behaved as a two-state
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

### Global 2.0″ design storm default (retired 2026-05-24, Brief 23)

The earlier `DESIGN_STORM_INCHES = 2.0` global default (introduced
April 2026 as "typical minor storm") was replaced with per-city
NatCap-canonical values: MN gets 3.94″ (100 mm per NatCap MN
args.json), SA gets 6.18″ (157 mm per NatCap SA README). The 2-inch
default wasn't anchored in any NatCap or InVEST canonical source —
it was a plausibility-level prototype default. Per-city values
better reflect each city's climate (SA's heavier convective storms
vs MN's lighter regional events).

Two non-obvious cascades fell out at retirement:
- SCS-CN nonlinearity in P (`Q = (P − 0.2S)² / (P + 0.8S)`) is not
  linear; doubling P more than doubles Q. The regeneration ratios
  in runoff metrics were ~4–5×, not the 2× MN / 3× SA rainfall
  ratio.
- The `flood-focused` placement weight reads per-pixel Q at the
  design storm; when rainfall changed the weights shifted, and
  flood-focused / balanced scenarios show <5 % cascades on
  downstream UNA / UMH / cooling / NDVI metrics. Intended behavior
  per the §5.2 placement formulas, not a bug. Random /
  cooling-focused / undersupply-focused cells are unaffected
  because their weights don't depend on rainfall.

See `../internal/DESIGN_NOTES.md` §2.4 for the per-city
design-storm decision in current-state form.

### `validate_scenarios.py` diagnostic (retired 2026-05-30)

A standalone diagnostic at `diagnostics/validate_scenarios.py` that ran
five canonical MN scenarios and verified ten directional expectations
(FF carbon > baseline, GI flood < baseline, HD nature ≤ baseline, etc.).
Surfaced during the `docs/` + `validation/diagnostics/scripts.data/`
migration as silently broken: its hand-rolled `_SessionStateStub` had a
stale attribute surface that pre-dated app's adoption of
`session_state.pop()` (Brief A.2) and `session_state.saved_scenarios`
(Brief #5), so module-level `import app` crashed on the first relevant
attribute access. Restored to working state in `ddd60e3` by reusing
`compare_una_invest._StubSt`, then retired immediately after.

Reasoning: the script had been dormant unnoticed since at least
Brief A.2 — the multiple accumulated rot points proved the
directional-sanity coverage wasn't load-bearing. The same
directional invariants are implicitly enforced by
`verify_baselines.py` (40 scenario × strategy snapshots, exact-value
regression). Removing the duplicate coverage simplifies `diagnostics/`
and removes one of the standalone stubs that drift out of sync (a
separate refactor will consolidate the remaining stub copies into a
single shared module).

---

## Completed-workstream specifics

Per-brief implementation detail extracted from ../../CLAUDE.md. Canonical
per-brief reasoning lives in ../internal/DESIGN_NOTES.md; this section preserves
anything from ../../CLAUDE.md that wasn't already duplicated there.

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

**Follow-up 2026-05-26: `max_entries=1` on `_load_city_runtime_state`.**
The original workstream brought the single-city steady state under the
ceiling but didn't address the dual-city case — `@st.cache_resource`
with no `max_entries` cap would keep both cities' ~1.5 GB transient
pipelines cached simultaneously after a city switch, risking OOM on
rapid switching. `max_entries=1` forces eviction of the previously-
cached city on switch. Trade-off: every city switch becomes a cold
load (~minute wait) rather than an instant cache hit; reliability
preferred over speed for the second-switch case.

---

### Brief narrative chronology (2026-05-28 — 2026-05-30)

Per-brief chronological narrative for the Brief 1 / 2 / 4 / 5 / B / A2
/ B1 / D1 / B2-revised / #3 / #4 / #5 work. The durable methodology
decisions from each brief live in `../internal/DESIGN_NOTES.md` in
template form; this section is the chronological narrative — what
shipped, when, what magnitudes — for context.

**Brief 1 (2026-05-28) — Signed metric cards label-flip rule.** Three
"dollar/count" metric cards (Preventable MH Cases, Avoided MH Costs,
Carbon Storage Value) can render negative values for scenarios that
make things worse. Each was hand-rolled inline (no shared renderer):
positive → benefit label + green delta; negative → harm/loss label
with positive magnitude + red `delta_color="inverse"`. Negative-case
labels: "Preventable MH Cases" → "Additional MH Cases"; "Avoided MH
Costs" → "Added MH Costs"; "Carbon Storage Value" → "Carbon Storage
Loss" (SA) / "Avoided Carbon Cost" → "Added Carbon Cost" (MN). Brief
1 also aligned Carbon Storage Value's negative color with the MH
cards — it previously used neutral `"off"` for negatives while MH
used red `"inverse"`. Decision rationale → `../internal/DESIGN_NOTES.md` §10.2.

**Brief 2 (2026-05-28) — Naming and labels for correctness.**
Display-only label cleanup; no metric computation changed, baselines
unaffected. "Flood Risk Reduction" → "Flood Retention" on the lead
Ecological card / tradeoff-plot x-axis / optimizer slider help /
comparison table; the SA Economic card's "Flood Damage Avoided" later
became "Flood Volume Reduction" (Brief 7's Approach B differentiate).
SA carbon confidence badge "Prototype" → "Four-pool stock (NatCap
framework)" (city-conditional via a new `_CONFIDENCE_BADGES` key —
methodology descriptor, not confidence tier). Cost-effectiveness
labels: "Cost / °F Cooling" → "Cost / Citywide °F Cooling"; "Cost /
Acre-Foot Prevented" → "Cost / Acre-Foot Runoff Prevented". Map
legend/caption ↔ slider sync — all three now read "Development-
intensity heat proxy." Brief 2 also landed the Carbon Storage Change
*quantity* card's negative-case bespoke render (Approach Y) — the SA
four-pool stock branch flips to "Carbon Storage Loss" with a positive
magnitude and a red ↑ delta when conversions reduce stored carbon.
Decision rationale → `../internal/DESIGN_NOTES.md` §10.3, §10.2.

**Brief 4 (2026-05-28) — `cooling_f` → `temp_change_f` sign-convention
refactor.** Cooling metric renamed and sign-flipped to the universal
physical ΔT convention (`positive = WARMER`, `negative = cooler`,
`= −old cooling_f`). Producer renamed `hm_to_fahrenheit_cooling` →
`hm_to_temp_change_f`. Display layer (`_fmt_temp_change`) always
renders natural language so users never see the bare signed number.
The optimizer was untouched — `surrogate.py` searches and ranks on
`mean_hm`, never on `cooling_f`. Per-tract breakdown flipped to match.
`cost_per_degf` divides cost by `−temp_change_f` and is defined only
when the scenario cools. Serialized artifacts regenerated: all 40
baseline snapshots, both dense CSVs (`scenarios_dense_mpls.csv`,
`scenarios_dense_sa.csv`). No `SCENARIO_SCHEMA_VERSION` bump (field
set unchanged in count). **Known-stale artifact:**
`data/scenarios_dense_mpls_full.csv` (hidden city) still carries the
old `cooling_f` column; regenerate when Mpls Full is activated.
Decision rationale → `../internal/DESIGN_NOTES.md` §10.1.

**Brief 5 (2026-05-28) — Sidebar reorganization + tooltips.** Sidebar
order: City → Land Use Scenario → Conversion Mix → **Placement
Strategy** → **Find Best Scenario** → Implementation Costs → Advanced
Settings. Placement Strategy previously sat *below* Find Best
Scenario, which inverted the intuitive flow. Find Best Scenario text
trimmed — one-line inline caption + "How this works" expander, with
prose corrected for accuracy (the surrogate explores conversion
percentage and conversion mix only; placement strategy and cost are
NOT part of the surrogate — the earlier draft text was wrong). Nature
Access tooltip gained an explicit AOI line (city-conditional) and a
saturation explanation. Decision rationale → `../internal/DESIGN_NOTES.md` §10.4.

**Brief B (2026-05-29) — UMH NE kernel Gaussian → buffer-mean.** The
UMH neighborhood-exposure kernel was switched from a Gaussian (σ =
search_radius / pixel) to the canonical InVEST UMH 3.19.0
buffer-mean (edge-corrected flat disk, 317-pixel disk at 30 m /
300 m), via the existing `_convolve_edge_corrected` helper.
`scipy.ndimage.gaussian_filter` and `_UMH_SIGMA_PX` removed. Probed
canonical's emitted `kernel.tif` (21×21 binary disk, sum 317) and
confirmed edge-correction via an all-ones NDVI test; an independent
random-NDVI probe matched the candidate to MAE 1e-8. Validation
result: **MN exact** (MAE ≈ 1e-9, r = 1.000000); **SA MAE ≈ 0** on
aligned input (the harness's 0.14 % residual on SA's 1713 × 1984
grid was empirically shown to be large-grid feeding-alignment + FFT
noise in the comparison, not a kernel divergence). Read-only
baseline diff confirmed drift isolated to `preventable_mh_cases` +
`avoided_mh_cost_usd` (60 divergences across 30 conversion
scenarios; ~1.5–3 % shifts). All 40 baselines + both dense CSVs
regenerated. **Schema bump 26 → 27.** MH card confidence Medium →
High. Harness latent bug fixed: `compare_umh_invest.py`'s export
step had an inline Gaussian copy from Brief A that didn't track the
kernel change — now calls `app._umh_neighborhood_exposure`.
Decision rationale → `../internal/DESIGN_NOTES.md` §6.3.

**Brief A2 (2026-05-29) — SA UNA AOI investigation (document-only).**
Question: Yingjie's roadmap said NatCap's SA UNA uses
`acs_block_group.gpkg` as the AOI; the prototype was thought to use
a City-of-SA clipped extent. Finding: prototype's UNA extent is the
Bexar County bbox (3,059 km², 1,906,325 people); NatCap's block
groups are a strict subset (2,519 km², 1,878,866 people). **Area
IoU = 0.824, population overlap = 98.6 %** — only 27,457 people
(1.4 %) are in the bbox but outside the block groups (sparse exurban
Bexar County). Architectural insight (the reason this isn't a config
swap): the prototype's UNA path is raster-only —
`calculate_nature_access(scenario_lulc, pop_count_raster)` takes no
AOI vector; the modelable extent is wherever the LULC/population
rasters have valid data. The `acs_block_groups_3857.gpkg` in the
repo feeds **only** `compute_per_tract_summary`, not any biophysical
model. **Decision: document, don't change.** Per-pixel
`urban_nature_supply_percapita` is computed identically regardless
of aggregation extent; the real validation need (matching NatCap's
per-block-group `ntr_bal_avg`) is met by aggregating the prototype's
supply raster per block group, which is a Track C concern. No code,
no baseline, no schema change. Investigation note also lives in
`../research/una/SA_UNA_BIOPHYSICAL_EXTENT.md` (the durable Brief A2
single home). Parity-claim implication summarised at
`../internal/NATCAP_ALIGNMENT.md` §4 "Computed vs displayed" → SA UNA /
biophysical extent reference + `../internal/CITY_PARITY.md` SA section
"SA biophysical extent vs ACS block-group polygons" callout.

**Brief B1 (2026-05-29) — NatCap fixed scenarios as first-class inputs
(partial).** Goal: make NatCap's seven SA project scenarios
(baseline, FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX)
loadable as first-class scenarios. **Investigate-first finding —
scenario rasters are flood-encoded, not compound.** NatCap shipped
the SA scenario LULCs only in the NLCD × tree-canopy 3-tier
encoding (211/212/213 = NLCD 21 × low/med/high; 998 = food forest;
999 = garden), at 10 m in a WGS84-datum Albers CRS. That matches
the prototype's SA flood CN table directly but does NOT match the
compound NLCD × NLUD × tree (0–1983) encoding the Carbon / UCM /
UNA tables are keyed on. The reference CSV's only two
`natcap_published` metrics (`temp_change_f`, `carbon_tons_co2`) are
both compound-keyed, so carbon/temperature reproduction is gated
pending NatCap's compound scenario inputs. **Local hunt for
compound scenario rasters — empty.** A content-signature search
across `~/Desktop`, `~/Downloads`, `~/Documents`, `/Volumes`, the
Google Drive sync root, and the `_zip_archive` zips found every
compound-signature raster on disk to be a baseline (five distinct
contents, duplicated 17× across pulls + GDrive mirror); there are
no scenario-suffixed inputs. NatCap built the per-scenario compound
LULCs as unsaved pipeline intermediates. NatCap data request to
unblock the gate is **parked (not sent)** in OPEN_QUESTIONS →
"Per-scenario compound LULC inputs". **What landed (scope A —
standalone scaffolding):** `natcap_scenarios.py` (Streamlit-
agnostic) with the four `PROVENANCE_*` constants,
`SA_NATCAP_FIXED_SCENARIOS` metadata, `load_natcap_fixed_scenario()`
(reproject + `Resampling.mode` majority-rule onto 30 m EPSG:5070,
`lru_cache`d), and `flood_reduction_from_nlcd_tree()` pure helper.
Dashboard wiring deferred to B2 (Phase 0 D invasive-change gate).
Smoke-test: `mean_cn ≈ 81.4`, `flood_reduction ≈ 18.5–18.6` across
the six alternative scenarios — matches NatCap's documented SA
flood-invariance under design-storm saturation. Flagged divergence
for B2: the prototype's own SA baseline is `mean_cn 76.54` / flood
23.5 (~5-pt CN gap from the scenarios), pointing to the
compound→NLCD×tree reduction path producing a different canopy-tier
mix than NatCap's native raster. Tracked as OPEN_QUESTIONS → "Native
NLCD×tree baseline flood raster".

**Brief D1 (2026-05-29) — Export for InVEST workflow.** Goal: a
single sidebar button that packages the currently-displayed
scenario as a runnable canonical InVEST 3.19.0 input zip — rasters
+ AOIs + biophysical tables + per-model `args.json` for UCM / UNA /
UFR / Carbon / UMH + a `metadata.json` recording provenance,
generator parameters, and per-model validation state. SA-only for
v1. Architecture: new `export_invest_bundle.py` Streamlit-agnostic
(`BundleSpec` dataclass + `build_invest_bundle(spec) → bytes`); the
zip is built in memory (Streamlit-Cloud-safe). Two-step sidebar
**Prepare → Download** flow avoids rebuilding the ~20 MB bundle on
every rerun. AOIs: synthesized bounding-box polygon for UCM / UFR
(watersheds) / UMH; NatCap's ACS block-group polygons for UNA
(framing NatCap uses for SA equity analysis per Vibrant Land).
Per-model args choices (Phase 0 introspection of InVEST 3.19.0
`ModelSpec`): UCM `cc_method="factors"` with 0.6 / 0.2 / 0.2 weights;
`do_energy_valuation=False` (biophysical-cooling only). UNA
`decay_function="dichotomy"`, `search_radius_mode="uniform radius"`,
`search_radius=800`, `urban_nature_demand=16.7`. UFR uses NatCap's
NLCD × tree CN table directly; damage valuation omitted (no SA
damage table — Path C). Carbon `do_valuation=False`. UMH
`model_option="ndvi"` + synthesized polygon
`baseline_prevalence_vector` carrying CDC ever-diagnosed BIRs; two
args files (depression, anxiety) since UMH's `effect_size` is
per-condition. **Phase 3 verification — all 5 models PASS on the
baseline bundle** (UCM ✓, UNA ✓, UFR ✓, Carbon ✓, UMH-depression ✓,
UMH-anxiety ✓). Brief amendment had marked UMH best-effort /
unverified; it ran cleanly, so the hedge dropped and UMH recorded
`validated`. **Nodata sentinel rule surfaced by Phase 3 UFR
failure-then-fix:** initial export wrote the NLCD × tree-reduced
raster with `nodata=0`, leaving 35,973 −128 pixels unmasked; InVEST
UFR `_lu_to_cn_op` raised `ValueError` (with a misleading empty `[]`
lucode list — known display bug). Fixed by writing `nodata=-128` for
both NLCD × tree rasters; all five models then pass. General truth,
not specific to UFR. Decision rationale → `../internal/DESIGN_NOTES.md` §9.

**Brief B2-revised (2026-05-29) — Validation badges + NatCap fixed-
scenario reference view.** Context: the original B2 (per-metric
Match/Diverged badges) was deferred earlier this session — Match /
Diverged requires prototype reproduction for the six NatCap fixed
alternatives, which is gated on the unavailable compound scenario
inputs. The revised brief expanded scope: keep Match/Diverged out,
deliver the ungated symposium core (three-state taxonomy as badges,
a dedicated fixed-scenario reference view routing around the
monolithic `evaluate_scenario`, side-by-side comparison surface,
baseline reproduction posture, flood reconcile). **Conservative-
floor decision (this session).** The investigation pass under the
"no parameter fitting" guardrail established that NatCap's published
citywide absolute baselines aren't reproducible from disk:
- **Temperature.** No SA UCM `args.json` ships in the drive pull;
  the `T_ref` / `uhi_max` NatCap used for `avg_temp_f = 90.08 °F`
  are not documented or recoverable. `T_air_nomix.tif` exists;
  back-solving from it crosses the no-fit guardrail.
- **Carbon.** NatCap's `tot_c_cur.tif` (EPSG:3857, ~34.5 m nominal
  pixel, 5,283 km² extent, mean 17.2) does not aggregate to the
  published 107.32M t CO2e by any standard interpretation (per-ha →
  25–33M depending on the cos²(lat) area choice; per-pixel-total →
  280M). The published number is a separate aggregation script that
  wasn't shipped. The prototype's own Bexar-bbox four-pool sum is
  147.96M.
- **A3 status.** `natcap_published` is "comparison-READY, never
  executed." The only callers of `compare_to_reference` are the
  four-line `__main__` smoke test with hardcoded values. No
  `evaluate_scenario → compare_to_reference` pipeline has ever run,
  because the only `natcap_published` metrics are exactly those gated
  by the unavailable compound inputs.

Net: a clean citywide-absolute reproduction match is not achievable
from what's on disk. The demonstrable reproduction claim, in order
of strength, is per-pixel parity vs canonical InVEST (HMI MAE 0,
Brief 28b; UMH MAE ≈ 0, Brief B), then four-pool methodology
adoption (Brief 30 — a methodology choice, not a parity
measurement). The badge taxonomy was tightened to match. The four-
state taxonomy locked here (`NatCap published value` / `≈ NatCap
method` / `≈ Aligned method` / `Prototype`) survives as the standing
contract. Decision rationale → `../internal/DESIGN_NOTES.md` §8.

**Brief #3 (2026-05-29) — Scenario provenance + validation header
badge.** Single helper `_render_scenario_provenance_header` driving
off the `PROVENANCE_*` constants re-exported from
`natcap_scenarios.py`. Five-row `_PROVENANCE_HEADER_INFO` mapping
(Baseline / NatCap published reference / Explorer-generated /
Surrogate-suggested) — the Validation strings come straight from
STRATEGY.md §4's honest-claims language. Two render paths wired:
fixed-scenario reference view (folds the previous standalone `##
label` + scenario_id/provenance caption into the unified header)
and main Explorer dashboard (above `#### Ecological`).
OPTIMIZER provenance not yet detected at runtime — the helper
accepts `PROVENANCE_OPTIMIZER` but the main panel doesn't
distinguish "applied from optimizer" from generic Explorer state;
Brief #4 plumbs the flag. Visual treatment: bordered colored block
with `Source:` / `Validation:` labels, using
`_VALIDATION_BADGE_COLOR_HEX` (same green / blue / gray palette as
per-card badges). Decision rationale → `../internal/DESIGN_NOTES.md` §8.2.

**Brief #4 (2026-05-29) — Optimizer as trustworthy scenario
discovery.** The Apply seam pre-#4: the optimizer's surrogate-
prediction table labeled itself "These are surrogate model
predictions. Click Apply to run a full pixel-level simulation."
Apply set `_pending_pct/gi/ff` + `applied_suggestion` +
`_show_apply_toast`, then `st.rerun()`. On rerun, pending became
slider values, full-raster `evaluate_scenario` ran, but nothing
signaled "this just came from the optimizer." Cards looked
identical to any Explorer rendering; the D1 export silently
recorded the scenario as `PROVENANCE_EXPLORER`. Changes (single
commit): (1) `applied_from_optimizer` flag + `_applied_optimizer_values
= (pct, gi, ff)` recording the just-Applied scenario's slider state;
auto-cleared at the top of every rerun if the current slider state
diverges (manual edit, preset, Best-by-Goal Apply, city change). (2)
OPTIMIZER provenance detection plumbed into both the main-panel
scenario header and the D1 export helper; detection order:
`pct_converted == 0` → BASELINE; else `applied_from_optimizer` →
OPTIMIZER; else EXPLORER. The header label flips to "Optimizer
suggestion · {scenario_name}"; the export records
`PROVENANCE_OPTIMIZER` with an `optimizer_suggested` generator block
carrying slider params + a note "Applied from Optimizer suggestion;
full-raster evaluated by the prototype engine before export." (3)
OPTIMIZER validation line updated from #3's placeholder
"engine-validated; exploratory" to "engine-validated; full-raster
evaluated — exploratory candidate for further validation." (4)
Entry-text reframe: sidebar subheader changed from "Find Best
Scenario" (verdict framing) to "Discover scenarios to validate"
(discovery framing). Best-Scenarios-by-Goal Apply is NOT classified
as OPTIMIZER — those come from the precomputed scenario grid.
Decision rationale → `../internal/DESIGN_NOTES.md` §7.2.

**Brief #5 (2026-05-29 → 2026-05-30 visual re-verify) — Cross-
source comparison table.** A single comparison surface that puts
NatCap's published references, the active Explorer/baseline/optimizer
scenario, and any saved scenarios side by side. Architecture: Option
A (NatCap anchors always-available + `saved_scenarios` for the rest)
chosen over Option B (new `compare_set` state) and Option C
(repurpose `saved_scenarios` as "pinned for comparison" + auto-
include NatCap) — reuses the existing Save mechanism, requires only
a single new `provenance` field on each saved dict. Placement: top
of Tradeoff Analysis tab, before `#### Tradeoff Space`. Row
composition: NatCap fixed anchors (SA only) → current scenario
(always, marked `▶ Current — …`) → saved scenarios for the active
city. Columns: Scenario · Source · Validation · Temperature · Carbon
stock · Carbon Value $ (derived) · Cooling Energy $ · Nature Access
% · Food (M lbs) · MH cases · Cost $M. **Honest-display invariants:**
unified Δ-basis across the three NatCap-shared columns (Temperature,
Carbon stock, Carbon Value $) — every row is a delta; NatCap baseline
row reads `"baseline"` rather than its absolute citywide value to
prevent misleading row-to-row comparisons like `+148M next to +0.5M`
(visual re-verify on 2026-05-30 caught the absolute-value drift
post-rebuild; fixed at the cell level). Validation column tooltip via
`column_config`. `—` for unavailable cells driven by row provenance,
not hardcoded. Flood intentionally excluded — different derivations
between baseline (compound→NLCD×tree reduction) and alternatives
(native NLCD×tree raster); the per-scenario flood card handles that.
Source / Validation columns are mandatory — the load-bearing piece
of the honesty story; enforced structurally. Decision rationale →
`../internal/DESIGN_NOTES.md` §8.3.

### SA flood-CN investigation (Q12, resolved 2026-05-29)

After integrating NatCap's SA-specific flood biophysical table
(`biophys_floodmitig_sa.csv`) into the prototype's flood CN lookup,
baseline regen showed unexpected behavior: Green Infrastructure
scenarios (which convert pixels to NLCD 90 / Woody Wetlands) slightly
*increased* modeled runoff rather than decreasing it. Investigation
revealed that NatCap's SA biophysical table uses CN values that diverge
systematically and substantially from standard NRCS TR-55 reference
values.

**Per-class comparison (tier 1 / "no canopy" baseline, HSG A):**

| NLCD | Class | NatCap CN_A | NRCS TR-55 CN_A | Δ |
|---|---|---|---|---|
| 11 | Open Water | 100 | 100 | 0 |
| 21 | Developed Open Space | 49 | 49 | 0 |
| 22 | Developed Low Intensity | 77 | 51 | **+26** |
| 23 | Developed Med Intensity | 89 | 61 | **+28** |
| 24 | Developed High Intensity | 98 | 89 | +9 |
| 31 | Barren | 77 | 77 | 0 |
| 41 | Deciduous Forest | 32 | 36 | −4 |
| 42 | Evergreen Forest | 39 | 36 | +3 |
| 43 | Mixed Forest | 46 | 36 | +10 |
| 52 | Shrub/Scrub | 49 | 35 | +14 |
| 71 | Grassland | 64 | 39 | **+25** |
| 81 | Pasture | 44 | 49 | −5 |
| 82 | Cultivated Crops | 68 | 67 | +2 |
| 90 | Woody Wetlands | 88 | 30 | **+58** |
| 95 | Emergent Herbaceous Wetlands | 89 | 30 | **+59** |

NRCS reference: TR-55, Second Edition (1986); WikiWatershed/tr-55
canonical Python implementation.

**Anomalous vs NRCS-consistent.** Anomalous (large positive Δ):
wetlands (+58/+59), low/med-developed (+26/+28), grassland (+25),
shrub/scrub (+14). NRCS-consistent (Δ ≈ 0): water, developed-open,
developed-high, barren, forests, pasture, cultivated crops.

**Internally coherent under a "wet OR impervious → high runoff" logic
— but not NRCS TR-55.** Water (100), wetlands (88-92), developed-high
(98), and developed-med (89) all rank as high-runoff surfaces; forests,
pasture, and developed-open as low-runoff. This is *a* defensible
hydrologic framework (treating saturated wetland soils similarly to
impervious surfaces), but it directly contradicts the InVEST UFR
documentation's stated intent that "the ranking between different land
uses is generally well captured" with natural infrastructure ranking as
lower-runoff.

**Mechanical consequence pre-resolution.** Under the staged table, the
prototype's Green Infrastructure scenarios — which convert developed
pixels to NLCD 90 (Woody Wetlands) — slightly increased modeled runoff
for SA (e.g., +43 % in some scenarios). The MN-placeholder it replaced
had wetlands at CN=1 (unphysically low), which is also wrong but in the
opposite direction.

**What we couldn't verify locally at the time of the deferral.**
- **NatCap's own SA UFR run outputs** — not present in the `InVEST
  Results/` staging tree (only UCM, UNA, Carbon were delivered). Without
  these we couldn't see whether NatCap's published SA flood scenario
  comparisons exhibited the same GI-increases-runoff behavior.
- **Documentation of the CN framework choice** — the `Notes on NASA
  Urban parameterization QA.docx` contained zero flood/CN/runoff
  content. The README pointed to `Ben NDR and Flood Mar_2023.pptx` for
  flood methodology; that pptx was not in the delivery.

**Resolution (2026-05-29) via `Ben NDR and Flood Mar_2023.pptx`.** The
pptx referenced in the README was located after the deferral commit
(`27d7be3`). Slide 7 explicitly addresses the flood-mitigation finding:

> "From a flooding standpoint, there is essentially no difference
> between garden, food forest, park, or vacant vegetated space. During
> large storms, rainfall rates greatly exceed infiltration capacity of
> soils and interception by trees, so topography and blue-gray
> infrastructure (e.g., pipe size, reservoir placement) tend to be
> very important. Urban ag scenarios investigated here are likely
> mostly swapping one greenspace for another, without changes to
> underlying soil or water storage capacity."

NatCap's CN values for SA reflect a **design-storm-saturation
framework**: under the 24-hour 100-year storm (6.98″ adjusted for SA),
soil infiltration capacity is exceeded across most vegetated surfaces
on SA's clay-rich D-soils, so even wetlands and forests rank as
runoff-generating. Wetland CN of 88-92 is internally consistent with
this framework.

NatCap's own modeled food-forest scenarios show +0.1 % to +1.1 %
increase in flood volume vs baseline — matching the prototype's
behavior when wired to the staged biophysical table. The integration
was correct; deferral was the conservative move while uncertainty
existed, but the uncertainty is now resolved.

Re-activated 2026-05-29 (the follow-up commit reverses the deferral in
`27d7be3`). Summary + the canopy-tier-mapping resolution + the
canopy-weighted-parameter methodology note all live in
`../internal/NATCAP_COLLABORATION.md` "Closed / resolved"; this entry
preserves the per-class CN comparison detail.

### Vocabulary alignment audit (2026-05-23, Briefs 8 + 9)

The 2026-05-23 vocabulary audit aligned the prototype's user-facing
labels with InVEST canonical terminology. The durable per-metric
alignment lives in `../internal/NATCAP_ALIGNMENT.md` §3 (folded in
during the trim commit); this entry preserves the rename chronology:

- **Temperature Change underlying quantity** — renamed `Cooling
  Capacity / CC` → `Heat Mitigation Index / HMI` (Brief 8). Reported
  value was already canonical HMI; the label was stale.
- **Tradeoff plot Y-axis** — renamed `Cooling Capacity` → `Heat
  Mitigation Index` (Brief 8).
- **Temperature-assumption tab kernel description** — corrected from
  `Gaussian` → `exponential decay at d_cool, eq. 118` (Brief 8). The
  decay kernel was always exponential per the canonical InVEST UCM
  formula; the description was wrong.
- **`equity-focused` placement strategy** — renamed to
  `undersupply-focused` (Brief 9). InVEST UNA reserves "equity" for
  demographic-group stratification (age, income, race); using it for
  generic per-capita undersupply crossed NatCap vocabulary. Saved
  scenarios with the legacy `equity-focused` key route via shim.
- **`flood-focused` placement** — formula switched from raw CN to
  per-pixel runoff `Q_{p,i}` from the SCS-CN equation at the design
  storm, matching InVEST UFR `Q_mm.tif` (Brief 9).
- **`cooling-focused` placement** — formula switched from `(1 −
  baseline_CC) × NLCD_intensity_proxy` to `(1 − baseline_HMI) × 1 / (1 +
  distance_to_buildings_px)` with `BUILDINGS_RASTER` distance transform
  (Brief 9). Bare CC sub-component → canonical HMI; NLCD-intensity
  three-value proxy → real building-proximity raster.
- **Avoided MH Costs card** — cross-reference to canonical InVEST UMH
  `preventable_cost.tif` naming added (Brief 8).
- **Cost-Effectiveness, Balanced placement, Smart Scenario Search** —
  flagged as app-specific (no InVEST analog) with explicit pointer to
  ROOT for LP-based multi-objective optimization (Brief 8).

These renames are display-only; no metric computation changed,
baselines unaffected.

---

## WHATS_NEW entries pruned 2026-05-29

Entries removed from `WHATS_NEW_ENTRIES` (app.py) during the changelog
trim, preserved verbatim here. The in-app "What's new" list was reduced
to three San Antonio entries (flood Curve Numbers, land cover, carbon
four-pool framework).

- San Antonio scenarios now show a conversion-fidelity panel reporting how often each conversion used the default target lucode vs. found a matching context-preserving compound row.
- San Antonio cooling estimates updated to NatCap's calibration.
- Flood metrics use per-city design storm depths.
- San Antonio temperature estimates updated to NatCap's calibration.
- Minneapolis nature access updated to NatCap's calibration.
- Placement strategy picker in the sidebar.
- Interactive Input Influence chart on the Tradeoff Analysis tab.
