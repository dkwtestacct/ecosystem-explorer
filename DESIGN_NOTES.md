# Design Notes

**Purpose:** Internal design decisions for the Urban Ecosystem Tradeoff Explorer. Records what options were considered, what was chosen, and why — for decisions that don't surface to users or to NatCap collaborators.

**Audience:** Future Claude sessions and Daniel-six-months-from-now. Not user-facing; not part of the NatCap collaboration log.

**Related docs:**

- `REFERENCE.md` — user-facing methodology (what each metric means).
- `ARCHITECTURE.md` — three-layer system overview.
- `NATCAP_COLLABORATION.md` — running collaboration log with NatCap (asks, gaps, decisions made without confirmation, open questions).
- `NATCAP_ALIGNMENT.md` — per-surface alignment status.
- `CITY_PARITY.md` — per-city alignment matrix.
- `DATA_INVENTORY.md` — every external data source the prototype consumes.

---

## City-specific copy convention

User-visible strings that reference city-specific values (baseline
numbers, data sources, climate framing, yield benchmarks) should
interpolate from one of:

- `_CURRENT_CITY_STATE.*` — for live-computed baselines (`baseline_cn`,
  `baseline_hm`, `baseline_ndvi`)
- Module-level constants set from `city_cfg` — `UHI_MAX_C`,
  `HM_TO_FAHRENHEIT`, `FOOD_FOREST_LBS_ACRE`
- A per-city dict (`_CITY_CAPTIONS`-style) — for prose that varies in
  structure, not just numbers
- A `selected_city.startswith("Minneapolis")` branch — for paragraph-
  level prose that has fundamentally different framing per city

The Temperature assumption tab (app.py around line 3320–3333) is the
reference example for the branch pattern.

When adding a new city, run `grep -n "Minneapolis\|\bMN\b" app.py`
and confirm no hardcoded city names remain in user-facing strings.

## UNA parameters

Working log of InVEST UNA parameter choices for the Minneapolis prototype. For
each parameter: the options considered, the value chosen, and why. To revisit
as the implementation evolves and as collaborators provide feedback.

### `urban_nature_demand` (m²/capita)

The per-capita supply standard. Pixels where supply ≥ demand are "adequately
supplied"; the headline metric reports the share of population that meets this.

**Options considered:**
- 250 m²/capita — InVEST default (used in Phase 1, gave 9.5% adequately supplied)
- 16.7 m²/capita — NatCap SA-study value
- Other published thresholds (e.g., WHO 9 m²/capita) — not investigated

**Chosen: 16.7 m²/capita.**

NatCap-validated for the SA Urban Agriculture project. No formal MN benchmark
exists, so adopting a NatCap-applied value is more defensible than the InVEST
generic default. Applicability of an SA value to MN's different urban context
is an open question.

### `search_radius_mode`

How the search radius is configured.

**Options considered:**
- `'uniform radius'` — single value for all nature classes
- `'urban_nature_table'` — per-class radii from the biophysical table
- `'population_group_radii_table'` — per-population-group radii

**Chosen: `'uniform radius'`.**

Matches NatCap SA-study practice and simplifies parameter exposition. Per-class
radii would be appropriate if the biophysical table had well-justified per-class
values for the MN context; it doesn't.

### `search_radius` (m)

The radius defining what nature is "reachable."

**Options considered:**
- 1000 m — used in Phase 1 (roughly 12-min walk)
- 800 m — NatCap SA-study value (roughly 10-min walk)
- 500 m — common in walkability literature (5-min walk)

**Chosen: 800 m.**

NatCap SA-study practice. Reasonable urban-planning walking distance.

### `decay_function`

How reachability falls off with distance from nature.

**Options considered:**
- `'dichotomy'` — binary in/out within the radius
- `'exponential'` — exponential decay (InVEST default)
- `'gaussian'` — gaussian decay

**Chosen: `'dichotomy'`.**

Matches NatCap SA-study practice. Simpler to explain than exponential.
Exponential is more theoretically grounded; revisit if dichotomy produces
output that's too coarse for the prototype's use case.

### `aggregate_by_pop_group`

Whether to compute aggregates separately for population subgroups.

**Options considered:**
- `False` — single aggregate over all population
- `True` — per-subgroup aggregates (requires `population_group_radii_table`)

**Chosen: `False`.**

The prototype's population raster doesn't have subgroup breakdowns. Subgroup
analysis isn't a current prototype goal.

### `urban_nature_lulc_table`

The biophysical table mapping LULC classes to `urban_nature` values and
per-class search radii.

**Options considered:**
- InVEST MN sample table (`LULC_attribute_table_UNA.csv`, already in repo)
- A NatCap-curated MN-specific table (existence unknown)
- A custom table designed for the prototype

**Chosen: InVEST MN sample table.**

Already in repo, already validated in Phase 1 comparison work. If a
NatCap-curated MN-specific table exists, would warrant adoption.

### `population_raster_path`

The population raster used as demand input.

**Options considered:**
- Existing Census 2020 raster (`pop_count_raster`, already in repo)
- A NatCap-provided alternative (existence unknown)

**Chosen: Existing Census 2020 raster.**

Already in repo. Standard data source.

### Population denominator (in headline reporting)

How to handle the 56.6% of MN population sitting on cooling-LULC nodata pixels
that InVEST cannot model.

**Options considered:**
- Report % of modelable-extent population (InVEST native output)
- Report % of total city population (count off-LULC residents as inadequate)
- Report both numbers separately

**Chosen: % of modelable-extent population, with tooltip clearly framing the
denominator.**

Matches InVEST's native output. Tooltip honesty addresses the denominator
ambiguity. Whether this is the right framing for a planner-facing dashboard
is open.

## Placement strategy

### The question

The prototype models scenarios by reallocating a percentage of developed
land among green infrastructure (wetlands), food forest, and high-density
development. A core question is: *where in the AOI should those
conversions be placed?* This determines the spatial pattern of which
pixels change LULC, which in turn affects the resulting metrics.

### Options considered

NatCap's project document identifies a spectrum of approaches:

**Simpler approaches:**
- **Three-layer non-convertible mask** — buildings + roads + existing nature as constraints. Conversions only go where land isn't already one of these three.
- **Wallpaper approach** — uniform tiling of conversion across the candidate area, rather than spatial clustering.

**Land-use simulation models:**
- **CLUE** (Conversion of Land Use and its Effects) — biophysical land-change modeler, established early 2000s.
- **PLUS** (Patch-generating Land Use Simulation) — ML-based, recent, open-source, from HPSCIL at China University of Geosciences. Built in C++ with Qt UI.
- **LCM** (Land Change Modeler) — proprietary, part of TerrSet.

### The implemented approach: three-layer non-convertible mask

This is the prototype's chosen — and currently implemented — approach.
Three categories of land are excluded from conversion, leaving a
candidate area the placement strategies draw from:

1. **Buildings** — building footprints are rasterized into the
   non-convertible mask (`buildings_raster`). Minneapolis uses the InVEST
   UFR sample building shapefile; San Antonio and Minneapolis Full use
   comprehensive OpenStreetMap footprints.
2. **Roads** — OpenStreetMap road footprints (Geofabrik extracts) are
   rasterized and unioned into the same mask, so impassable surfaces are
   excluded alongside buildings.
3. **Existing nature** — the conversion candidate pool is built only from
   developed LULC classes (NLCD 21–24); nature pixels (forest, water,
   wetland, etc.) are never conversion candidates, by construction.

`convertible_pixels` is therefore `developed_pixels` minus the
building-plus-road mask. The five placement strategies (random,
flood-focused, cooling-focused, equity-focused, balanced) operate only on
that remaining candidate area. Implementation: `_load_city_runtime_state`
in `app.py` (Phase 9 rasterizes roads and unions them into
`buildings_raster`; Phase 11 builds `convertible_pixels`); see also
CLAUDE.md "OSM road exclusion".

**Bounded by design.** This is a deliberately modest spatial-fidelity
improvement: it constrains *where* conversions can physically land using
grounded data, without attempting to predict *where they would* land
(the domain of the land-use simulation models below).

**Phase 2 update.** Comprehensive OSM building footprints
(`data/osm/minneapolis_buildings.geojson` — ~113k city-wide footprints
from Geofabrik) are now integrated for Minneapolis via a **split-config
architecture**. Two config keys: `buildings_file` (the InVEST UFR sample
shapefile — typed, a model input) drives the `buildings_type_raster`
behind the Cooling Energy Savings and Flood Damage Avoided dollar
metrics; `mask_buildings_file` (the OSM footprints — untyped, a placement
constraint) is unioned into the non-convertible mask only. This closes
the earlier downtown-core-only coverage gap without regressing the typed
$ metrics, and reflects NatCap's explicit separation of
placement-constraint inputs from model inputs. The MN non-convertible
mask grew 37,812 → 54,268 pixels; the convertible pool shrank ~21%
(33,357 → 26,372). Roads were already comprehensive OSM for every city.

### Considered but deferred — PLUS / CLUE / LCM

Considered based on literature review (PLUS GitHub README, recent
applications, NatCap's framing) but deferred — they are not part of the
prototype's current implementation.

These models answer a different question than the prototype is set up
for. They project *what will happen* given historical drivers and trends
— useful for asking "what does the AOI look like in 30 years if current
trends continue?" The prototype instead asks *what should happen if
planners intervene* — a different question that doesn't map cleanly onto
status-quo projections.

That said, they're in NatCap's recommendation list because they're
expected to add value. Future phases may incorporate one or more if they
prove useful for:
- Baseline-without-intervention projections (status-quo scenarios)
- Learning placement patterns from historical land-use change
- Comparing planner interventions against business-as-usual

Specific operational concerns: PLUS is a standalone C++ Qt application,
not a Python library — integration would require subprocess execution
or substantial reimplementation. CLUE is Java-based with similar
deployment issues. LCM is proprietary, can't ship in an open-source
prototype.

### Wallpaper approach — interpretation uncertain

NatCap's document lists this alongside the three-layer mask as a
"simpler approach" to placement. The term doesn't have a standard
land-use literature definition we could verify, so the distinction
from random selection is unclear.

**Working interpretation:** Wallpaper applies a uniform tiled pattern
across the AOI (every Nth pixel, repeating motif, etc.) rather than
independent random selection of pixels. If this distinction matters,
the prototype currently does the latter (random + strategy-weighted),
not wallpaper.

**To clarify with NatCap.** This is a real question to ask: what does
NatCap mean by "wallpaper approach" specifically? Whether the prototype
should pursue this as an option depends on the answer.

### Suitability formulas (2026-05-23 reformulation)

For each of the four weighted placement strategies, the suitability
formula determines per-pixel weights used to bias conversion away from
uniform random selection. The 2026-05-23 reformulation aligned each
formula to a canonical InVEST quantity where one exists, replacing
earlier homegrown proxies.

#### `undersupply-focused` (formerly `equity-focused`)

**Options considered:**
- `population × (1 − access_score + 0.01)` — aggregate need with homegrown reachability proxy (the prior implementation)
- `deficit / population` — per-capita inequity weighting
- `max(0, urban_nature_demand − urban_nature_supply_percapita)` — per-capita supply deficit per InVEST UNA canonical `urban_nature_balance_percapita.tif` framing

**Chosen: per-capita supply deficit** (the third option).

The prior formula's population multiplier made it an aggregate-need
metric: a pixel with 1000 undersupplied residents got 10× the weight
of a pixel with 100 equally-undersupplied residents. InVEST UNA's
canonical `urban_nature_balance_percapita.tif` output is per-capita
— a pixel where residents have 5 m²/capita supply is equally undersupplied
regardless of how many residents are there. Adopting the per-capita
form aligns with InVEST UNA's framing exactly. It also has a real
ethical character (every resident's access deficit counts equally,
rather than dense areas dominating by aggregate weight) — both readings
are defensible; the alignment-to-NatCap argument was decisive.

The strategy was also renamed from `equity-focused` to
`undersupply-focused`. InVEST UNA reserves "equity" for demographic-group
stratification (age, income, race); using it for generic undersupply
crosses NatCap vocabulary. `Pund_adm` (the count of undersupplied
population) is the InVEST canonical name for this concept.

The `+ 0.01` floor on the old formula is gone. Pixels with no per-capita
deficit get true zero weight. The saturation fallback in
`_select_pixels_for_conversion` (added in Brief 7) handles cases where
the strategy doesn't have enough non-zero pixels for the requested
conversion count.

#### `flood-focused`

**Options considered:**
- Per-pixel CN as the weight (the prior implementation)
- Per-pixel runoff `Q_{p,i}` from the SCS-CN equation at the design storm — matches InVEST UFR's canonical `Q_mm.tif` output
- Per-pixel `1 − R_i` (non-retention fraction) — equivalent to `Q/P`, also canonical

**Chosen: per-pixel runoff `Q_{p,i}`** (mm at the 2-inch design storm).

The prior `weights = CN` form is monotone with runoff but has the wrong
shape. At low CN (high retention), `Q ≈ 0` regardless of CN — but the
old formula assigned non-zero weight to those pixels. The canonical
SCS-CN runoff equation `Q = (P − 0.2·S)² / (P + 0.8·S)` produces a
sharper distribution that more aggressively concentrates on
high-runoff pixels — closer to what "prioritize flood-prone areas"
should mean.

The placement-strategy diagnostic (Brief 6) found `flood-focused`
under the old formula was the weakest mover on the flood metric.
The reformulation's sharper concentration is expected to address this.
Brief 9's diagnostic re-run measures whether it does.

#### `cooling-focused`

**Options considered:**
- `(1 − baseline_CC) × (NLCD_intensity_proxy + 0.1)` — bare CC + NLCD-class proxy for building proximity (the prior implementation)
- `(1 − baseline_HMI) × (NLCD_intensity_proxy + 0.1)` — same form, canonical HMI substituted
- `(1 − baseline_HMI) × distance_to_buildings_weight` — canonical HMI + real distance-to-building from `BUILDINGS_RASTER`

**Chosen: canonical HMI + real distance-to-buildings** (the third option).

The prior formula's first term used the bare CC sub-component when
the canonical HMI raster (validated against InVEST at MAE=0) was
available. The HMI is what the Temperature Change metric card already
reports; using it here aligns the strategy with the metric it's
trying to improve. The prior formula's second term used the NLCD
intensity raster (NLCD 23→1.0, 22→0.6, 21→0.3) as a proxy for
"near buildings" — a three-value approximation when the actual
buildings raster was available. The reformulation uses
`scipy.ndimage.distance_transform_edt` on `BUILDINGS_RASTER` to
produce a real distance-to-buildings raster (pixel units), then
weights via `1 / (1 + distance)` so a pixel on a building gets weight
1.0 and a pixel 300 m away gets ~0.1.

The `+ 0.1` floor is gone. Pixels truly distant from buildings get
near-zero weight rather than an artificial floor. The saturation
fallback handles edge cases.

#### `balanced`

Unchanged in structure: equal-weighted normalized combination of the
three reformulated focused strategies. Implicitly absorbs the
reformulations through its component strategies. Still an app-specific
heuristic — no InVEST analog.

#### Decision principle

Across all three reformulations: where a canonical InVEST quantity
exists, use it. The user's explicit principle (2026-05-23):
*"I want to be as closely aligned to natcap as possible. even if it
takes more time and results in undoing previous work."* This document
records the chosen formulas; the rationale per strategy is above; the
methodology shift is real, not cosmetic — the new formulas can produce
materially different scenario outputs from the old ones. Brief 9's
verify_baselines regeneration and PLACEMENT_STRATEGY_DIAGNOSTIC.md
re-run capture the empirical impact.

## Land use and land cover sources

The prototype uses **NLCD 2021** (legacy MRLC product) across all cities.
NLCD 2021 is also the LULC vintage shipped with the InVEST UFR, UCM, and
UNA sample data the prototype builds on.

### Rasters in use

Four LULC rasters across the three cities. Minneapolis downtown uses two
distinct rasters — one for the flood / Curve Number calculation, one for
cooling and as the canonical scenario LULC — inherited from separate
InVEST sample bundles (UFR vs UCM/UNA). San Antonio and Minneapolis Full
each use a single raster for both flood and cooling.

| City (role) | Path | CRS | Dimensions | Notes |
|---|---|---|---|---|
| Minneapolis — cooling & scenario LULC | `data/cooling/land_use_2021.tif` | EPSG:26915 | 356 × 360, int16 | The canonical scenario raster. **Byte-identical to the InVEST UNA sample** LULC — see `UNA_LULC_INVESTIGATION.md` (MD5 `56d1080fa70576cad15896642a107a3d`). |
| Minneapolis — flood / CN LULC | `data/flood/LULC_NLCD_2021_MN.tif` | EPSG:26915 | 356 × 360, int16 | Drives the Curve Number / flood calculation. Same AOI and grid as the cooling LULC but a distinct file (MD5 `a8687db9f76394aa1333b8a3d35ec57e`) — from the InVEST UFR sample bundle rather than UCM/UNA. |
| Minneapolis Full (dormant) | `data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif` | EPSG:5070 | 607 × 374, uint8 | `available=False` — hidden from the city selector. One raster for both flood and cooling. |
| San Antonio | `data/sa/flood/land_use_2021_sa.tif` | EPSG:5070 | 1713 × 1984, uint8 | One raster for both flood and cooling (the `cooling_lulc_file` is a `../flood/` relative reference to the same file). Independently sourced via `download_sa_data.py`. |

Minneapolis downtown carries the InVEST sample's native EPSG:26915
(UTM 15N) projection; Minneapolis Full and San Antonio use NLCD's native
EPSG:5070 (CONUS Albers). All four rasters are tracked in git.

### Planned future source for San Antonio

NatCap maintains a curated SA Urban Agriculture data folder (access
pending). When access is granted, the planned migration is to adopt the
curated SA LULC and parameters, replacing the current
independently-sourced raster. Tracked in `NATCAP_ALIGNMENT.md` Tables 2
and 3 as ⏸️ Pending data access.

The SA project covers: Crop yield, Urban Cooling, Carbon Storage, Urban
Nature Access, Flood Mitigation, and Nutrient Delivery Ratio (pending
complete data).

### NLCD legacy vs Annual NLCD (May 2026)

USGS replaced legacy NLCD with **Annual NLCD** in 2024 — a new ensemble
deep-learning methodology with annual coverage 1985–2024 and a revised
class system (21 → 16 classes). MRLC states that "Legacy NLCD data are
not directly comparable to the newer Annual NLCD data due to differences
in methodologies, inputs, and ancillary data."

The prototype stays on **legacy NLCD 2021** because the InVEST sample
data and the biophysical tables (CN, cooling, UNA) are all calibrated
against the legacy 21-class schema. Migrating to Annual NLCD would
require re-validating every lucode mapping and regenerating all
baselines.

**Open question for NatCap:** has the canonical SA Urban Agriculture
data folder migrated to Annual NLCD, or does it remain on legacy
NLCD 2021? The answer determines whether the prototype's continued use
of legacy NLCD aligns with current NatCap practice or diverges from it.

Reference: <https://www.mrlc.gov/faq>

## UCM args alignment (2026-05-24)

The SA Urban Cooling Model parameters were aligned with the
NatCap-curated values documented in
`data/sa/natcap_2024/README_San_Antonio_InVEST_model_inputs.docx`:

- `uhi_max = 11 °C` (was 3.5 °C)
- `reference_air_temperature = 35 °C` — informational; no analog to
  update (the prototype reports pure deltas, never absolute T_air,
  so `reference_air_temperature` doesn't appear in the codebase)

**Interpretation.** NatCap models SA on a heat-wave-day scenario
(35 °C rural reference + 11 °C peak UHI → ~46 °C / ~115 °F peak
urban). The prototype's previous `UHI_MAX_C = 3.5` modeled a
closer-to-average summer day. Both are defensible methodologically;
the project's working principle is alignment with NatCap canonical,
so we adopt their values.

**Consequence.** SA temperature deltas are now ~3× larger than under
the previous parameterization (e.g. an `all_gi` scenario at
`pct_converted = 10` shifts from ~0.13 °F cooling to ~0.40 °F
cooling). The HMI calculation is unchanged; only the
ΔHMI → ΔT_°C → ΔT_°F output scaling differs. `HM_TO_FAHRENHEIT`
auto-derives as `UHI_MAX_C × 1.8 = 19.8 °F/HMI` for SA. MN
parameters (`UHI_MAX_C = 2.05` from the InVEST UCM sample args.json)
are unchanged.

**Other UCM args.** The remaining values from NatCap's README —
`air_blending_distance = 600 m`, `maximum_cooling_distance = 450 m`,
`cc_method = factors`, weights `shade = 0.6 / albedo = 0.2 / et = 0.2` —
already match the prototype's existing values. Energy-savings and
work-productivity valuations are disabled for SA in both NatCap's
setup (`do_energy_valuation: False`, `do_productivity_valuation:
False`) and the prototype (Cooling Energy Savings card gates on
`BUILDINGS_HAVE_TYPES`; SA has no typed buildings, so the card
already degrades to `$0` with explanatory tooltip).

The prototype's one remaining UCM divergence — per-pixel rather
than per-building T_air aggregation over the 600 m blending radius
— is unchanged and affects only the dollar-valued Cooling Energy
Savings metric, not Temperature Change.

## Per-city NatCap parameter framing (2026-05-24, Briefs 22 + 23)

**The meta-decision.** NatCap parameters are project-specific by design.
Each city's project is tuned to its own policy framing — SA project for
the SA Urban Agriculture work uses WHO-minimum-green-space demand and
heat-wave-day climate parameters; MN project uses aspirational green-
space targets and moderate-summer climate parameters. There is no
single "NatCap canonical" value per parameter — alignment is per-city.

This isn't NatCap inconsistency; it's appropriate project-by-project
parameter variation. Different teams, different time periods, different
policy goals per city. Working principle: align MN-side parameters with
the MN project's args.json; align SA-side parameters with the SA
project's README values.

**Why we're acting on this without waiting for NatCap confirmation.**

- The MN sample data bundles (received 2026-05-24) and the SA README
  (received 2026-05-23) are both internally consistent and represent
  NatCap's documented parameter choices for each city. Adopting them
  is alignment, not improvisation.
- The questions worth asking NatCap are confirmatory ("we assumed
  your two framings are intentional — is that right?"), not blocking
  ("we can't proceed without knowing which is canonical").
- The Natural Capital Symposium (June 29 – July 1) is the natural
  place to confirm. Briefs 22/23 changes are the foundation we'll
  discuss against, not a tentative position waiting for confirmation.

**The UNA values (Brief 22).** Minneapolis UNA switched from SA-project
values (`demand=16.7`, `radius=800`, `decay=dichotomy`) to MN-project
canonical (`demand=250`, `radius=1000`, `decay=exponential`). The 15×
demand increase reflects the MN project's higher "adequate green space"
target. Source:
`data/invest/mn_sample_data_natcap_2026/UrbanNatureAccess_sample_data_MN/invest_urban_nature_access_args_MN.json`.
SA UNA stays at SA-project values; per-city framing in action.

The exponential-decay kernel is built canonically per
`pygeoprocessing.kernels.exponential_decay_kernel` as
`natcap.invest.urban_nature_access` calls it: `k(d) = exp(-d /
expected_distance)` for `d ≤ max_distance` else 0, where
`expected_distance = search_radius_in_pixels` and
`max_distance = ceil(search_radius_in_pixels) * 2 + 1`. For MN at
1000 m / 30 m pixels, the kernel is 139×139 (~19k elements). The
dichotomy branch (SA) keeps the existing binary-disk form.

**The rainfall values (Brief 23).** `DESIGN_STORM_INCHES` migrated
from a 2.0-inch global default (introduced April 2026 as "typical
minor storm") to per-city values: MN gets 3.94" (100 mm per NatCap MN
args.json), SA gets 6.18" (157 mm per NatCap SA README). The 2-inch
default wasn't anchored in NatCap or InVEST canonical; it was a
plausibility-level prototype default. Per-city NatCap-aligned values
better reflect each city's climate (SA's heavier convective storms
vs MN's lighter regional events).

Two non-obvious effects of the rainfall change:

1. **SCS-CN nonlinearity in P.** `Q = (P − 0.2S)² / (P + 0.8S)` is not
   linear in P. Doubling P more than doubles Q. The observed
   regeneration ratios (~4-5×) reflect this — *not* a linear scaling
   from the 2× MN / 3× SA rainfall ratio. Correct behavior.
2. **Flood-focused placement cascade.** Brief 9's `flood-focused`
   weighting computes per-pixel Q at the design storm. When rainfall
   changes, those weights shift, and the strategy selects slightly
   different pixels. Downstream metrics (UNA, UMH, cooling, NDVI) then
   show small (<5%) cascades on flood-focused and balanced cells. Not
   a bug — it's the intended Brief 9 design that placement reads the
   per-city rainfall. Random / cooling-focused / undersupply-focused
   cells are unaffected because their weights don't depend on rainfall.

**What this does NOT decide.** Some open questions remain even with
this framing accepted:

- Whether NatCap considers the MN-project framing still current or
  has since shifted toward SA-style values. Worth confirming at the
  symposium. The MN sample data is from March 2026 (recent), so
  likely still current.
- Whether UFR rainfall depth should reflect a 100-year design storm
  vs a typical-rainfall scenario. NatCap's choice of 100mm/157mm is
  their methodology call; we're inheriting it.
- For SA-side parameters where NatCap hasn't published per-city
  values (e.g., Carbon rates, food forest yield), the prototype
  continues using plausibility-level defaults pending more data.

**The pattern this establishes.** Future per-city parameter decisions
follow the same logic: identify the NatCap project's canonical value
for each city; use it; document the decision here rather than holding
for explicit NatCap confirmation. The discipline applies for parameters
where NatCap has clearly documented per-city values. For parameters
without clear NatCap guidance, the prototype's default stays and is
noted in `NATCAP_COLLABORATION.md` as an open question.

## SA compound LULC integration — foundational decisions (2026-05-24, Brief 27)

Brief 27 adopted NatCap's compound NLCD×NLUD×tree-canopy LULC framework
for SA, with three judgment calls worth recording durably (the
planning artifact `SA_INTEGRATION_PLAN.md` walks through the same
decisions ahead of execution; this section is the long-lived record
once that planning artifact ages out of relevance).

**CRS: reprojected to EPSG:5070, accepting coverage loss at extent
edges.** NatCap's SA data ships in EPSG:3857 (Web Mercator). The
prototype's SA stack is EPSG:5070 (Conus Albers, equal-area).
NatCap's choice of 3857 was likely operational (web display); their
MN sample data uses an equal-area projection. The prototype's metrics
are area-based (acres converted, runoff volume, etc.), so equal-area
preservation matters. Reprojecting NatCap → 5070 with nearest-neighbor
resampling at 30 m preserves the prototype's existing grid and avoids
regenerating 5 other SA rasters (soil, ET, population, buildings,
roads). Tradeoff: NatCap's raster extends ~6 minutes farther north,
~3 farther south, ~2 farther west; clipping to the prototype's extent
loses those edge pixels (~1% nodata coverage on the reprojected output;
~15% raw-extent loss). Accepted because the analysis is constrained to
roughly Bexar County regardless.

**Conversion-target mapping: preserve NLUD + tree-canopy, change NLCD
only.** When a user converts a developed pixel "to food forest," the
new lucode in compound LULC must encode the post-conversion (NLCD,
NLUD, tree-canopy) state. The chosen rule preserves the source pixel's
NLUD and tree-canopy bins and changes only the NLCD signal. This is
least presumptuous — the conversion models the land cover change
without claiming knowledge of how the land use or canopy state changes.
The compound code is looked up in `lulc_crosswalk.csv` via (NLCD=target,
NLUD=source_NLUD, tree=source_tree), with `is_realistic_to_create=yes`
rows preferred and ascending `lucode` as a deterministic tiebreaker.
When the (NLUD, tree) tuple doesn't appear with the target NLCD, fall
back to `DEFAULT_FF_LUCODE`, `DEFAULT_GI_LUCODE`, or `DEFAULT_HD_LUCODE`
(see below).

**`DEFAULT_*_LUCODE` choices and rationale.** Fallback compound lucodes
for conversion targets when the (NLUD, tree-canopy) combination from the
source pixel doesn't have a matching row for the target NLCD in the
crosswalk. Picked by filtering for `is_realistic_to_create=yes` in the
crosswalk, then preferring the highest-`frequency` row (the "typical"
representative of the target land cover as seen in NatCap's SA raster).
The `is_realistic_to_paint` column is empty across the entire crosswalk
(all NaN), so `is_realistic_to_create` is the only available flag.

- `DEFAULT_FF_LUCODE = 1310` — Deciduous Forest, Timber NLUD, medium
  tree canopy. Frequency 36,939 (highest among NLCD-41 create-OK rows).
- `DEFAULT_GI_LUCODE = 122` — Woody Wetlands, Wetland NLUD (under
  Waterbody class), medium tree canopy. Frequency 50,384 (highest among
  NLCD-90 create-OK rows).
- `DEFAULT_HD_LUCODE = 341` — Developed High Intensity, Residential
  Urban NLUD, low tree canopy. Frequency 53,389 (highest among NLCD-24
  create-OK rows).

Edge case: when the fallback fires for a substantial fraction of
conversions (>5% of converted pixels), surface it — the rule may need
refinement. Logging the fallback fraction inside `evaluate_scenario`
would help; deferred until Brief 28+ when the compound conversion logic
is actually exercised in metrics.

**Compound→NLCD reduction routing (transitional).** Brief 27 adopts the
compound LULC raster but keeps the prototype's existing per-NLCD
biophysical tables (UCM, UFR, UNA). Compound lucodes are reduced to
NLCD codes via the crosswalk's `nlcd` column once at load time — the
reduced view is assigned to the existing `cooling_lulc` name that every
downstream consumer reads. The `reduce_compound_to_nlcd()` helper builds
a NumPy lookup array (`COMPOUND_TO_NLCD`) and applies it vectorized; the
compound-nodata sentinel (-1 in NatCap's raster) is rewritten to the
prototype's module-wide `NODATA` (-128) so `(scenario_lulc != NODATA)`
masks downstream continue to work. This is a *transitional* state —
Briefs 28-30 will swap individual model tables to compound-keyed
versions, removing the reduction step for each one in turn. Until then,
the SA pipeline runs on compound LULC but reports metrics calibrated to
per-NLCD biophysical tables. Result: 97.91% pixel-wise agreement between
the compound-reduced NLCD view and the prior `land_use_2021_sa.tif`;
SA baselines drift <0.5% on every headline metric.

**Forward-looking infrastructure — `COMPOUND_AFTER_*` lookups.**
Three `int16` lookup arrays (one per conversion target) are built at
load time alongside `COMPOUND_TO_NLCD`: `COMPOUND_AFTER_FF`,
`COMPOUND_AFTER_GI`, `COMPOUND_AFTER_HD`. Each maps source compound
lucode → target compound lucode that preserves NLUD+tree-canopy while
swapping NLCD to the conversion target, falling back to
`DEFAULT_*_LUCODE` when the (NLUD, tree) tuple has no row for the
target NLCD. These are aliased to module-level on city load (None for
cities without `compound_lulc_file`) but are not yet consumed in
`evaluate_scenario` — Brief 28+ will wire them when per-model tables go
compound-keyed and the downstream lookups need a compound view of the
converted scenario rather than the NLCD-only `scenario_lulc`.

**What this does NOT decide.**
- Whether MN should also migrate to compound LULC. Out of scope — MN's
  NLCD-only framework works and NatCap hasn't shipped MN compound data.
- When to swap each per-model biophysical table to compound-keyed.
  Brief sequence in `SA_INTEGRATION_PLAN.md`.
- Whether to switch SA AOI to NatCap's `acs_block_groups_3857.gpkg`.
  Optional (Brief 31).
- The compound `code` column encoding scheme — not positional, no
  documented logic. The serial `lucode` (0–1983) is the join key.
  Worth raising with NatCap as a clarifying question.

## SA UCM compound biophysical table adoption (2026-05-24, Brief 28b)

Brief 28b swapped SA's per-NLCD Köppen-BSh-tuned UCM biophysical table
for NatCap's compound NLCD×NLUD×tree-canopy table
(`data/sa/natcap_2024/ucm__nlcd_nlud_tree.csv`, 1,984 rows), keyed on
the compound LULC raster adopted in Brief 27. SA UCM is now the first
prototype model to consume the compound view directly without going
through `reduce_compound_to_nlcd()`.

**Köppen-BSh tuning retirement.** The prototype's previous SA UCM
table (`data/sa/cooling/biophysical_table_urban_cooling_SA.csv`)
tuned four NLCD classes (41, 42, 52, 81) for hot semi-arid climate,
leaving the rest at the MN-copy defaults. This was a workaround for
not having SA-specific compound LULC. NatCap's compound table
captures the climate-relevant variation through the tree-canopy bin
(none/low/medium/high) and the NLUD code (residential/commercial/
managed-natural/etc.) rather than through climate-tuned per-NLCD
values. The retired table file is kept on disk for historical
reference; it is no longer loaded.

**What changes in metric calculation.** SA's Heat Mitigation Index
(HMI), Cooling Energy Savings ($), and Temperature Change (°F) all
shift. Measured at baseline regeneration (`verify_baselines.py`):

- `baseline_hm`: **0.2866 → 0.3937 (+37.4 %)**
- `mean_hm` on the food-forest scenario at 10 % conversion:
  0.325 → 0.409 (+25.8 %)
- `cooling_energy_savings_usd`: **−77 % to −86 %** across SA
  scenarios

**Why the $ shift exceeds the HMI shift.** The energy-savings formula
is `clip(ΔHMI × UHI_max, 0, ∞) × consumption × area × $/kWh`. With
baseline_hm now ~37 % higher, the marginal scenario ΔHMI shrinks
proportionally — AND the clip-at-zero asymmetry punishes pixels
whose scenario HMI fell below baseline (no negative credit). The
compounded effect of a higher baseline plus the asymmetric clip is
the 77–86 % drop.

**Why the baseline rises.** The compound table gives systematically
different per-pixel inputs for SA's lucode distribution. Aggregating
the compound-keyed values back to NLCD buckets at the SA AOI:

| NLCD | OLD shade | NEW shade | OLD green_area | NEW green_area |
| ---: | --------: | --------: | -------------: | -------------: |
|   21 |    0.300  |    0.231  |          1.000 |          0.030 |
|   22 |    0.100  |    0.215  |          0.000 |          0.009 |
|   23 |    0.000  |    0.193  |          0.000 |          0.005 |
|   24 |    0.000  |    0.164  |          0.000 |          0.002 |
|   42 |    0.850  |    1.000  |          1.000 |          1.000 |
|   81 |    0.000  |    0.158  |          1.000 |          1.000 |

The mechanism is real, not a lookup error. Two specific corrections
the compound table makes that the per-NLCD table couldn't:

- **NLCD-21 (Developed, Open Space) green_area: 1.000 → 0.030.**
  The per-NLCD table flagged every "Developed, Open Space" pixel
  as 2-ha-eligible park green — i.e., as a source of `CC_park`
  cooling for surrounding pixels. The compound table only flags the
  ~3 % with high tree canopy as such; the rest are mowed lawns,
  road shoulders, and managed open lots that aren't parks. The old
  framing was inflating the area of SA "effective park" by ~30×.
- **NLCD-23 (Developed, Medium-Density) shade: 0.000 → 0.193.**
  The per-NLCD table credited zero canopy shade across all
  medium-density developed pixels. The compound table reflects that
  SA's medium-density residential carries non-trivial existing tree
  canopy (back yards, street trees, lot-line trees), and the
  shade-eligible fraction of these pixels is genuinely ~0.2.

Net effect across the developed mask (NLCD 21–24): more shade, more
Kc, less park-credit. The Köppen-BSh tuning was overstating cooling
leverage by understating baseline canopy on the very pixels where
interventions would land. The new baseline is more accurate; the
smaller marginal scenario improvements are too.

**Worth surfacing to NatCap.** The +37 % baseline_hm shift on SA
from the per-NLCD → compound migration is a meaningful finding for
any other group running an InVEST UCM prototype on a per-NLCD table
in hot-canopied cities. Suggests per-NLCD tuning systematically
overstates cooling-intervention $-value in places where developed
land carries non-trivial tree cover.

**MN UCM unchanged.** Brief 28b's table swap is SA-only; MN keeps
`biophysical_table_urban_cooling_MN.csv` and shows zero divergences
across the 20 MN baseline cells.

**Reduction routing for UCM removed.** Brief 27's
`reduce_compound_to_nlcd()` is no longer used by UCM — the compound
table is keyed directly on compound lucodes. UFR and UNA still use
the reduction routing pending Briefs 29 and 30. `scenario_lulc_ucm`
is the new return-dict field that carries the right LULC view per
city (compound for SA, NLCD for MN); the previously-singular
`scenario_lulc` continues to carry the NLCD view for every consumer
that wants it.

## SA UNA compound biophysical table adoption (2026-05-24, Brief 29)

Brief 29 swapped SA's UNA attribute table from the prototype's
per-NLCD `LULC_attribute_table_UNA.csv` (borrowed from MN's UNA
sample bundle) to NatCap's compound NLCD×NLUD×tree-canopy table
(`data/sa/natcap_2024/una__nlcd_nlud_tree.csv`, 1,984 rows). SA UNA
consumers now index the compound LULC raster directly; only Carbon
still routes through compound→NLCD reduction (pending Brief 30).

**What changes in metric calculation.** SA's `nature_access_pct`,
`people_with_nature_access`, and the baseline
`urban_nature_supply_percapita` raster that feeds undersupply-focused
suitability weights all shift. Measured at baseline regeneration:

| Metric | Pre-Brief-29 | Post-Brief-29 | Shift |
|---|---|---|---|
| SA baseline `nature_access_pct` | 89.7 | 94.2 | +4.5 pp / +5.0% |
| SA baseline `people_with_nature_access` | 1,710,167 | 1,794,653 | +84,486 (+4.9%) |
| SA random food_forest `nature_access_pct` | 99.4 | 99.7 | +0.3 pp (saturated) |
| SA random high_density `nature_access_pct` | 88.3 | 94.2 | +5.9 pp |

Scenarios already saturated near 100% (food_forest, green_infrastructure)
shift modestly because there is no headroom. High-density scenarios
that *remove* urban-nature pixels shift more visibly because the
prior per-NLCD table understated the baseline accessibility they
were removing from. Undersupply-focused and balanced placement
strategies also shift downstream metrics (cooling, MH, NDVI, CN)
because the baseline UNA raster feeds suitability weights, which
changes the chosen pixels, which propagates.

**Mechanism.** Brief 24's inventory recorded the compound UNA table's
`urban_nature` distribution as 976 / 48 / 960 rows at 1.0 / 0.5 / 0.0
respectively — ~52% of compound lucodes carry nonzero
`urban_nature`. The prior per-NLCD table treated all developed-class
pixels (NLCD 21-24) uniformly at `urban_nature=0`. The compound
table credits the ~10-15% of developed SA pixels whose NLUD +
tree-canopy bin maps to a compound lucode with `urban_nature` ∈
{0.5, 1.0} — chiefly developed pixels with high tree-canopy bins or
managed-natural NLUD context. Lifting `urban_nature` from 0 → 0.5
or 1.0 on a fraction of developed pixels lifts the baseline 2SFCA
supply per capita across the AOI, which in turn lifts
`pct_pop_supply_ge_demand`.

Same direction as Brief 28b's UCM finding (per-NLCD biases against
existing canopy on developed land); both reflect the compound
framework recognizing per-pixel signals the per-NLCD framework
flattens.

**Vectorized lookup replaces dict-iteration pattern.** The previous
`_una_supply_percapita` used a Python `for lucode, proportion in
URBAN_NATURE_PROPORTION.items(): nature_area[scenario_lulc ==
lucode] = ...` loop with per-class boolean-mask writes. Fine at
MN's ~14 NLCD codes; untenable at SA's 1,984 compound lucodes
(would do 1,984 raster-wide boolean comparisons per call). Brief
29 replaced this with a vectorized `urban_nature_arr[safe] *
pixel_area_m2` indexed read, sized per-city to the table's max
lucode (1,984 for SA, ~96 for MN). The `urban_nature_arr` array
joins `shade_arr`/`kc_arr`/`albedo_arr`/`green_area_arr` on
`CityState`; the wrapper / pure-variant split (zero-deps wrapper
reads module alias; `_pure` variant takes the array explicitly) is
the same shape Brief 28b established for `_compute_hmi_raster`.

**Search radius unchanged.** The compound UNA table's
`search_radius_m` column is all zeros (Brief 24 finding). The
radius comes from `city_cfg['una_search_radius_m']` at args level,
not from the per-row table value. No change to that mechanism.

**MN UNA unchanged.** Brief 29's table swap is SA-only. MN
continues to read its NLCD-keyed sample bundle (`LULC_attribute_table_UNA.csv`).
All 20 MN baselines: zero value divergence; the only diff is the
new `scenario_lulc_una__md5` field, whose hash equals
`scenario_lulc__md5` because for MN the UNA / UCM / NLCD views are
the same array object.

**Reduction routing for UNA removed.** Brief 27's
`reduce_compound_to_nlcd()` is no longer used by UNA — the
compound table is keyed directly on compound lucodes via the new
`scenario_lulc_una` field. Only Carbon still uses the reduction
routing pending Brief 30. After Brief 30 the
`reduce_compound_to_nlcd` helper becomes effectively unused for
live metric computation (still used for `scenario_lulc`, the NLCD
view exposed to spatial-map rendering + non-UCM/UNA/Carbon
consumers).

## SA Carbon four-pool framework adoption (2026-05-25, Brief 30)

Brief 30 swapped SA's Carbon model from the prototype's per-conversion-
type single-rate annual proxy (`n_pixels × CARBON_SEQ_RATES[target] ×
acres-per-pixel`) to NatCap's canonical InVEST four-pool stock
framework via `data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv` (1,984
rows × 27 cols). Four pools per compound LULC class — above-ground
biomass, below-ground biomass, soil organic matter, dead organic
matter — each in tons C/ha. SA Carbon consumers now index the compound
LULC raster directly via a new `scenario_lulc_carbon` return field;
none of the three SA biophysical models (UCM, UNA, Carbon) still route
through the compound→NLCD reduction.

**What the prototype's single-rate proxy was approximating.** A
landscape-level aggregate of carbon flux per converted hectare,
lumping the four pools into one annual-equivalent number. It produced
plausible directional results but conflated stock and flow, didn't
distinguish above-ground from soil carbon, and applied the same rate
regardless of baseline (so converting forest → forest gave the same
nonzero number as converting parking lot → forest, despite the actual
carbon delta being very different).

**The four-pool stock framework.** Per pixel, total carbon stored =
`c_above + c_below + c_soil + c_dead` (t C/ha). For a scenario LULC,
stock delta vs baseline = `sum_pixels((scenario_total - baseline_total) ×
pixel_area_ha) × (44/12)` to convert tons C → tons CO2. This is a
one-time stock change when the land use changes, NOT an annual
sequestration rate. Captures direction (positive when gaining nature,
negative when losing it), magnitude (proportional to the per-pool
deltas, not a flat per-acre rate), and per-pixel-baseline sensitivity
(the actual baseline LULC of converted pixels matters).

**Methodology decision — match NatCap's published SA work.** NatCap's
2023 "Vibrant Land" report (Guerry et al.) describes their SA Carbon
methodology in Appendix 2:

> "We analyzed landscape carbon storage using the InVEST Carbon
> model... We converted carbon storage into monetary value using a
> $53 Social Cost of Carbon based on US government guidance using a
> 3% discount rate (Interagency Working Group on Social Cost of
> Greenhouse Gases, 2021)."

And in the Results section, they report carbon as stock × value
(citywide full-conversion food-forest scenario: 340,000 t C / $17.6M
total value; smaller-AOI food forests in proportion). Direct stock ×
$/t, no amortization, no InVEST NPV valuation.

Brief 30 adopts that framing exactly: four-pool stock × SC-CO2 for the
dollar metric. **No amortization** to annual flow, **no rename-only
fix** (the methodology shift is also intended), **no InVEST NPV**
(Vibrant Land doesn't use it).

**Methodology matches, SC-CO2 vintage differs.** The Vibrant Land
report uses **IWG 2021's $53/ton CO2 @ 3% discount rate**. The
prototype's `EPA_SOCIAL_COST_CARBON = $190/ton CO2 @ 2% discount rate`
is **EPA 2023 final rule** ("Methodology for Estimating the Social
Cost of Greenhouse Gases", Nov 2023) — same US-government standard
lineage, a different and more current vintage. The prototype keeps
$190/t. As a result, SA's dollar carbon value comes out ~3.6× Vibrant
Land's reported figure on equivalent stock magnitudes — methodology
aligns, dollar magnitudes don't, because the price/ton differs by
vintage. NATCAP_COLLABORATION.md flags this as a confirmatory question
for future NatCap conversations ("are you planning to update Vibrant
Land's figures to EPA 2023?").

**Field rename: `carbon_tons_co2_yr` → `carbon_tons_co2`** (unified
across cities; Option D.1 in the brief). The semantics differ per
city — annual flow for MN, one-time stock for SA — but the return-dict
key is the same, with the temporal framing surfaced via metric labels.
Dollar metric: `avoided_carbon_cost_usd` → `carbon_value_usd`. Card
labels branch on `_CARBON_IS_STOCK = c_above_arr is not None`:
- MN card: "Carbon Sequestration" + "/yr" suffix; dollar card "Avoided
  Carbon Cost (per year)".
- SA card: "Carbon Storage Change" (no /yr); dollar card "Carbon
  Storage Value" (one-time).

**Conversion-target lucode evidence (DEFAULT_FF=1310, DEFAULT_GI=122,
DEFAULT_HD=341).** Sanity-checked against the compound carbon table at
investigation time:

| Lucode | Conversion target | LULC × NLUD × tree-canopy | Total t C/ha |
|---|---|---|---|
| 1310 | DEFAULT_FF | Deciduous Forest × Timber × medium canopy (40%) | 190.77 |
| 122  | DEFAULT_GI | Woody Wetlands × Wetland × medium canopy (40%) | 259.06 |
| 341  | DEFAULT_HD | Developed High Intensity × Residential × low canopy (15%) | 80.44 |

Ordering HD < FF < GI is the land-cover-plausible direction (wetlands
have very high soil carbon, forests have high above-ground biomass,
developed has the least of both). Brief 27's defaults work cleanly
for Carbon without needing carbon-specific overrides.

**Magnitudes from baseline regeneration.** MN: zero value divergence
across all 20 baselines — the field is renamed, the value is
identical (e.g., MN food_forest_random `carbon_tons_co2` = 2052.6,
exactly as the schema-24 `carbon_tons_co2_yr`). SA: stock framing
produces values ~30× larger than the prior annual framing, matching
the category-error correction the brief predicted:

| SA scenario | Pre-Brief-30 (annual) | Post-Brief-30 (stock) | Ratio |
|---|---|---|---|
| food_forest_random        |  65,264.9 (t CO2/yr) |  1,936,072 (t CO2 stock) | 29.7× |
| food_forest_balanced      |  65,264.9 (t CO2/yr) |  2,019,502 (t CO2 stock) | 30.9× |
| green_infrastructure_balanced |  37,294.3 (t CO2/yr) |  4,375,912 (t CO2 stock) | 117.3× |
| high_density_random       |       0.0 (t CO2/yr) |   -849,262 (t CO2 stock) | — (sign flip: nature loss) |

Green Infrastructure's higher ratio reflects the woody-wetland's very
high soil-carbon pool (~197 t C/ha at compound lucode 122) versus the
low single-rate annual proxy. High-density now shows negative stock
change — converting baseline (some developed pixels carry meaningful
tree canopy) to compound HD (lowest canopy) loses carbon. The
single-rate proxy bottomed at $0 here, hiding the loss.

**Order-of-magnitude check vs Vibrant Land.** Vibrant Land reports
~340,000 t for a citywide full food-forest conversion. Brief 30's
prototype reports ~1.9M t for converting 10% of developed pixels in a
larger AOI (Bexar-area bbox, not SA city proper). The ~5× difference
is within plausible bounds — different AOI extent, different "full
conversion" definition. Critically, the stock direction and per-pool
balance are correct: positive for nature gain, negative for nature
loss, baseline-zero when no LULC delta.

**Cross-metric temporal-framing comparability.** Cooling, flood, and
mental-health dollar metrics are annual flows. SA Carbon (post-Brief-
30) is a one-time stock value. The two appear side-by-side on the
dashboard; the temporal-framing divergence is **surfaced via metric
labels rather than hidden via amortization**. This matches the
Vibrant Land presentation directly — their report juxtaposes annual
cooling savings ($3.5M/year) with total carbon value ($17.6M) without
forcing them into a shared frame.

**MN Carbon unchanged.** Brief 30's table swap is SA-only. MN
continues to use `CARBON_SEQ_RATES` (FF 3.5, GI 2.0, HD 0.0 t
CO2e/acre/yr) via `_compute_carbon(n_wet, n_for, n_hd)` — the
per-conversion-type annual flow. The cross-city temporal-framing
divergence is per the project's "align with NatCap canonical, per
city" working principle (CLAUDE.md). Replacing MN with a four-pool
framework would require sourcing NatCap MN data of the same shape, out
of scope for this brief.

**Vectorized four-pool lookup.** `_compute_carbon_four_pool_pure(scenario,
baseline, c_above, c_below, c_soil, c_dead)` reads the four arrays
explicitly so the loader can call it before module aliases are
rebound; `_compute_carbon_four_pool(scenario, baseline)` is the zero-
deps wrapper. Single fancy-index per pool + a couple of additions and
a single sum — much cheaper than 1,984 raster-wide boolean
comparisons (the dict-iteration pattern that Brief 29 retired for
UNA). Same shape as Brief 29's `_una_supply_percapita` /
`_una_supply_percapita_pure` pattern.

**The brief had a factual error about EPA_SOCIAL_COST_CARBON.** The
brief assumed the prototype already used $53/t (Vibrant Land's IWG
2021 value). It actually uses $190/t (EPA 2023). User confirmed:
keep $190 untouched, document the methodology-matches-but-constant-
differs distinction, anchor stop-and-report on stock quantity rather
than dollar value. The brief's "Modify EPA_SOCIAL_COST_CARBON ($53/t
stays)" instruction is preserved in its letter (don't modify) — the
mistaken premise about the current value is what shifted.

## SA AOI switch to ACS block-group polygons (2026-05-25, Brief 31)

Brief 31 swapped SA's `tracts_file` config pointer from
`data/sa/tracts_bexar.shp` (375 Bexar County TIGER 2020 tracts) to
NatCap's `data/sa/natcap_2024/acs_block_groups_3857.gpkg` (1,124 ACS
block-group polygons covering the City of San Antonio). Final brief
in the SA NatCap data integration workstream per
SA_INTEGRATION_PLAN.md.

**Shape determination (per the brief's investigate-first
classification).** Shape B (mild form) — unit-of-aggregation change.
The AOI polygon file is consumed *only* by
`compute_per_tract_summary`'s Neighborhood breakdown table (a top-5
most-cooled polygons display in tab2). It is rasterized once at load
time into `tract_id_raster` for spatial-grouping purposes and never
read from `evaluate_scenario`. **No biophysical metric depends on the
polygon file.** Switching the file changes the row count of the
breakdown table (375 → 1,124 aggregation units) and the dashboard
caption ("Census tracts" → "Census block groups" for SA), but no
metric value shifts.

**The LULC raster's valid-pixel mask defines the actual modelable
extent** for every model (UCM/UNA/Carbon/UFR/UMH). The tracts/block-
group polygons are a separate axis: they slice the modelable extent
into aggregation units for reporting, but they don't bound the
biophysical computation. This is why the brief's "doesn't change
global metrics" framing is correct — and why no baseline regen was
needed.

**Why block groups parallel NatCap's framing.** Vibrant Land (Guerry
et al. 2023) reports equity analysis at block-group resolution in
Figure 5 (SNAP recipients / poverty / people of color by tract) and
Figure 10 (correlation between SNAP usage and temperature). NatCap's
provision of `acs_block_groups_3857.gpkg` alongside their compound
LULC + biophysical tables signals that block-group resolution is the
intended SA reporting unit. The prototype's Bexar County tracts were
TIGER 2020 polygons (375), broader than ACS block groups (1,124) and
extending across the full county rather than tightening to the City
of San Antonio extent.

**Implementation details.** Single config-pointer change in
`CITIES['San Antonio, TX']['tracts_file']`. The existing rasterization
+ aggregation code in `_load_city_runtime_state` and
`compute_per_tract_summary` generalizes cleanly — both functions
iterate a generic polygon DataFrame and don't care whether each row
is a tract or block group. The dashboard caption at tab2 was made
city-conditional via `selected_city.startswith("San Antonio")` so
the SA framing reads "Top 5 most-improved Census block groups" while
MN continues to read "Census tracts". The `compute_per_tract_summary`
function name and the `tract_id_raster` / `TRACTS_DATA_AVAILABLE`
state fields were intentionally left unrenamed — the aggregation
code is polygon-name-agnostic, and renaming would have been refactor
scope-creep (the brief explicitly scoped this away).

**The retained `tracts_bexar.shp` file remains on disk** for
reference. Future analyses comparing block-group vs tract aggregation
can re-point the config or load both for side-by-side comparison.

**MN unchanged.** Brief 31's pointer swap is SA-only. MN
(`tracts_file`: `admin_boundaries_census_tracts.shp`, InVEST UFR
sample) and MN Full (`tracts_hennepin.shp`, TIGER 2020) are
untouched. All 40 baselines pass without regen.

**Schema version not bumped** (24 → 25 in Brief 30 was the last
schema change). Brief 31 doesn't change `evaluate_scenario`'s return
dict — it only swaps a config pointer + adjusts a dashboard caption.

## SA flood damage table — resolved (Path C, Brief 33)

The prototype's `avoided_flood_damage_usd` field remains $0 for every
SA scenario because SA's per-NLCD-class damage value table is empty
(or absent), while MN's is populated. The metric exists in the schema
and the calculation pipeline runs; it multiplies through zero.
Known issue since Brief 22. SA's `city_cfg['damage_table_file'] =
None` (`config.py:197`); MN points at the InVEST UFR sample
`Damage_loss_table_MN.csv` with per-NLCD-class values for Roads /
Commercial / Residential / Industrial.

NatCap's own SA README "leaves blank" on this (per CITY_PARITY.md
row 152). They explicitly didn't curate per-NLCD-class damage
values for SA — the only piece of the SA NatCap data integration
workstream (Briefs 27-30) where they didn't provide curated inputs.

**Why this matters.** Prior to Brief 33, the SA dashboard's Flood
Damage Avoided card rendered "—" (em-dash) with help text saying the
damage table was "not sourced yet" and "would light this card up" if
added. The comparison table rendered "$0" / "+$0" in the Flood Damage
Avoided row. Both implied the absence was a data-sourcing gap to fill,
when in fact NatCap *deliberately* didn't monetize SA flood damage —
they reported flood mitigation as percent volume reduction instead,
matching InVEST UFRM's own caveat that the model doesn't produce
inundation maps and therefore can't confirm built-infrastructure
exposure.

**Four resolution paths under consideration:**

### Path A. Source SA-specific damage values independently

Pull from FEMA flood damage data (NFHL, Hazus depth-damage curves),
USACE economic studies, insurance industry per-property damage
estimates, or peer-reviewed flood-risk literature with SA-specific
calibration. Produces an SA-tuned damage table that matches the
rest of the SA workstream's data fidelity.

**Pros.** Most rigorous. SA-specific. Matches the Brief 27-30
standard of using NatCap-aligned or peer-reviewed sources.

**Cons.** Makes parameter choices NatCap didn't endorse.
Potentially diverges from any future NatCap SA work. Significant
effort — weeks of data-sourcing work, not a single brief.

### Path B. Borrow MN's damage table for SA, label as placeholder

Copy MN's per-NLCD damage values into an SA-keyed table. Update
CITY_PARITY.md to note borrowed status. Add dashboard tooltip
explaining the placeholder.

**Pros.** Lowest-effort technical change. SA dashboard stops
showing $0. Single commit.

**Cons.** Re-introduces the per-NLCD-borrowed-from-MN pattern that
Brief 29 just retired for UNA. Numerically suspect — MN's per-NLCD
damage values may not generalize to SA: different urban form,
floodplain hydrology, property values, precipitation regime (MN's
100 mm storm vs SA's 157 mm storm per Brief 23 already makes runoff
non-comparable; property values + structure types also differ).
Likely substantial over- or under-estimate. Numerically arbitrary;
methodologically uncomfortable.

### Path C. Embrace the $0; explain on the dashboard

Modify the dashboard card to detect "no damage table configured"
and render explanatory text ("Per-NLCD damage values not available
for this city — runoff reduction shown separately on the Flood
card.") instead of `$0`. Keep the underlying field at $0 for
surrogate-training compatibility; suppress its rendering as a
dollar value.

**Pros.** Most epistemically honest. Aligns with NatCap's own
"leaves blank" stance. Defensible against any audience: "we don't
invent values when the canonical source leaves it blank."

**Cons.** Reduces dashboard utility — users can't compare cities
on this metric. Surfaces the limitation rather than masking it,
but some users may read $0/blank as the metric being broken rather
than as honest data absence.

### Path D. Hybrid — per-acre-foot proxy decoupled from NLCD class

Replace per-NLCD damage values with a per-acre-foot-of-avoided-runoff
damage constant sourced from flood-risk literature (e.g., FEMA
Hazus general per-AF damage estimates, typically ~$2k-$50k per
acre-foot depending on land use). Calculation becomes
`avoided_flood_damage_usd = avoided_runoff_acre_feet * per_AF_damage`.
Both cities use the same per-AF constant; the dollar value scales
with the city's actual runoff reduction.

**Pros.** Gives SA a dollar number without inventing per-NLCD
values. Grounded in published flood-risk literature. Decouples
dollar value from NLCD class taxonomy — could simplify the model.
May actually be a *methodology upgrade*, not just a workaround.

**Cons.** Different framing from MN's current per-NLCD approach.
Need to decide whether MN also migrates to this framing
(consistency) or stays per-NLCD (per-city methodology, similar to
Brief 30's stock-vs-flow Carbon framing divergence). The per-AF
damage constant has substantial uncertainty (the $2k-$50k range
above is realistic). Constant choice may itself need NatCap
sign-off.

**Status: pending decision.** None of these paths has been chosen
yet. Each requires the user's judgment about methodology alignment
vs. dashboard utility tradeoffs. When a path is chosen, a follow-up
brief will implement it.

**Related conversations to check before choosing:**

- Any prior email/text/meeting discussion with Gretchen / Yingjie /
  other NatCap collaborators about SA flood damage methodology
- The shared NatCap Google doc, if it covers methodology decisions
- The symposium audience consideration (end of June 2026) — SA
  showing $0 in front of NatCap collaborators may or may not be a
  problem worth resolving by then

### Resolution (Brief 33): Path C chosen

Research into NatCap's Vibrant Land report (Guerry et al. 2023)
confirmed they used InVEST UFRM for SA but explicitly did *not*
enable `infrastructure_damage_loss_table_path`. They reported flood
mitigation as **percent reduction in flood volume** (e.g., "2.7%
reduction in flood volume on-site in a 100-year storm"), not as a
monetized dollar figure. This is a deliberate methodology choice
rooted in InVEST's own caveat that the model doesn't produce
inundation maps and therefore can't confirm built-infrastructure
exposure.

**Path C implementation (Brief 33).** Presentation-layer change
only — no model, schema, or baseline changes:

1. **Main card.** The SA-specific branch (typed buildings present,
   no damage table) now renders as `"Flood Volume Reduction"` with the
   value `f"{flood_reduction:.1f}%"` and help text citing the
   Vibrant Land precedent. The previous "—" with "would light this
   card up" framing is gone. MN's branch (damage table present)
   continues to render `"Flood Damage Avoided"` in dollars,
   unchanged.

2. **Comparison table.** The Flood Damage Avoided row is now
   city-conditional. MN renders dollars (Baseline `$0`, Scenario
   `$X.XM`, Change `+$X.XM`). SA renders volume reduction
   (Baseline `0%`, Scenario `X.X%`, Change `+X.X%`). Row label
   shifts to `"Flood Volume Reduction"` for SA.

3. **Underlying field unchanged.** `avoided_flood_damage_usd` still
   returns $0 for SA and the real dollar value for MN. Surrogate
   training, lookup table schema, CSVs, and the 40/40 baseline
   regression are all unaffected (no `SCENARIO_SCHEMA_VERSION`
   bump, no baseline regen).

4. **Per-city methodology pattern.** Same shape as Brief 30's
   per-city Carbon framing (annual flow MN vs. one-time stock SA)
   and Brief 22-23's per-city UNA/rainfall framings: each city
   gets the framing NatCap canonical material uses for *that* city,
   not a forced cross-city unification.

**Reversibility.** If a future NatCap conversation surfaces that
they want SA-specific damage values (Path A), Brief 33's minimal
changes don't block re-rendering the card in dollar terms — point
`damage_table_file` at a curated CSV and the existing MN-branch
code path activates.

## Lookup-overlay safety contract

The slider branch in `app.py` (the `if lookup_key in lookup_table
and placement_strategy == 'random':` block) loads a row from the
lookup table, then overwrites a specific subset of fields with
freshly-computed values via `_fresh = evaluate_scenario(...)`. This
pattern raised a correctness concern during Brief A pre-share
review: if a method changes for a field that lives in the lookup
row but isn't on the overwrite list, the user would see a mix of
stale lookup values and fresh values for the rest.

**Resolution: the pattern is safe-because-schema-versioned.**

`compute_lookup_table` is `@st.cache_data`-decorated with
`schema_version=SCENARIO_SCHEMA_VERSION` as a cache-key parameter:

```python
@st.cache_data
def compute_lookup_table(_state, city_key, data_dir_flood,
                         data_dir_cooling,
                         schema_version=SCENARIO_SCHEMA_VERSION):
    ...
```

When `SCENARIO_SCHEMA_VERSION` is bumped (the standard discipline
for any change to `evaluate_scenario`'s return-dict shape or
semantics — see CLAUDE.md), all cached lookup entries are
automatically invalidated. The lookup table is rebuilt from
scratch using the current `evaluate_scenario`. Every field loaded
from a lookup row is therefore guaranteed to be schema-current.

**What the overwrites are actually for:**

The defensive overwrites in the slider branch fall into two
categories, neither of which is staleness protection:

1. **Per-rerun state dependencies.** `cost_gi`/`cost_ff`/`cost_hd`,
   `carbon_rate_ff`/`carbon_rate_gi` come from sliders the user
   adjusts between reruns. The lookup table was built with default
   slider values, so cost-derived fields (`total_cost_mln`,
   `flood_damage_avoided_usd`, `cooling_energy_savings_usd`,
   `carbon_tons_co2`, `carbon_value_usd`) must be recomputed live.
2. **Stripped fields.** `scenario_lulc` and `scenario_lulc_ucm` are
   intentionally stripped from lookup entries (rasters are too big
   to cache per slider position; see `compute_lookup_table` body),
   so they must be loaded from `_fresh`. Downstream consumers like
   `compute_per_tract_summary` need the raster.

`food_mln_lbs`/`people_fed`/`mean_ndvi` are also overwritten. The
historical reason was that the lookup table predated an
`n_food_pixels` fix; under the schema-version contract that
defense is now redundant but kept for stability — food is cheap to
recompute and the consistency principle ("same scenario, same
recomputed value") has its own value.

**Contract for future devs:**

- Do NOT add new defensive overwrites for surrogate-target fields
  (`flood_reduction`, `mean_hm`, `runoff_acre_feet`,
  `nature_access_pct`) or for other fields that are pure functions
  of `(pct, gi, ff, seed)`. Bump `SCENARIO_SCHEMA_VERSION` instead.
- DO add overwrites for fields that depend on per-rerun state
  (sliders, user toggles) — those aren't a function of the lookup
  key alone.
- The slider branch's leading comment (`app.py` around the
  `lookup_key = ...` line) summarizes this contract. Keep it in
  sync if the overwrite list changes.

The lookup table is only built in High Resolution mode, which is
itself gated behind an opt-in checkbox (Brief C). So the surface
area where this contract matters is narrow by design.

## SA conversion-fallback instrumentation (Brief B)

The SA compound conversion path (`evaluate_scenario`'s
`if cooling_lulc_compound is not None:` branch, ~app.py:1535) tries
to preserve each converted pixel's (NLUD, tree-canopy) tuple by
looking up a matching compound lucode in NatCap's crosswalk
(`load_lulc_crosswalk`, ~app.py:509). When no matching row exists,
the conversion falls back to documented defaults — `DEFAULT_FF_LUCODE
= 1310`, `DEFAULT_GI_LUCODE = 122`, `DEFAULT_HD_LUCODE = 341`.

**Why instrument:** before Brief B there was no visibility into how
often the fallback fires. That matters because the methodology
question "is the default principled?" is academic if <5 % of
converted pixels fall to defaults, and substantive if >30 % do.
The instrumentation lets the numbers themselves answer the question
rather than leaving it to assumption.

**How instrumented:**

1. `load_lulc_crosswalk` now builds three parallel boolean arrays
   (`compound_after_ff_was_default`, `_gi_`, `_hd_`) alongside the
   existing `compound_after_*` lucode arrays. Same shape, same
   indexing (by source compound lucode). `True` = the source
   pixel's (NLUD, tree-canopy) had no matching row in the crosswalk
   for the target NLCD; conversion fell back to the configured
   `DEFAULT_<target>_LUCODE`.

2. At conversion time inside `evaluate_scenario`, each per-target
   conversion site (`if n_for > 0:` etc.) counts
   `int(COMPOUND_AFTER_*_WAS_DEFAULT[src].sum())` over the source
   compound lucodes of the actually-converted pixels.

3. Three new scalar keys in `evaluate_scenario`'s return dict —
   `ff_fellback_pixels`, `gi_fellback_pixels`, `hd_fellback_pixels`
   — surface the per-scenario counts. Always emitted (0 for MN, no
   compound conversion path) so the schema stays consistent across
   cities.

4. Dashboard surfaces a "Conversion fidelity (SA)" panel inside the
   Assumptions and limitations expander showing
   `fellback_pixels / n_converted` as a percentage per target,
   gated on `_COMPOUND_CONVERSION_ACTIVE = COMPOUND_AFTER_FF is not
   None`. Hidden for MN.

**Why flat scalar keys rather than a nested `conversion_diagnostics`
dict:** `verify_baselines.py:_snapshot_from_results` handles
ints/floats/numpy scalars but `dict` falls through to a
`WARN: skipping field` branch. Flat scalars also serialize trivially
to CSV without flattening logic at write time. The dashboard panel
computes the per-target fraction at display time — no precomputed
fraction key needed.

**Why not a surrogate target:** the diagnostic is pure metadata
about the conversion mechanism, not an outcome metric the surrogate
should predict. `REQUIRED_TARGET_COLUMNS` is unchanged. The
`_scenario_signature` cache key (Brief A.1) also doesn't include
the new columns — they don't affect surrogate cache invalidation.

**Schema bump:** `SCENARIO_SCHEMA_VERSION` 25 → 26. Per-city dense
CSVs regenerated (SA / MN / Mpls Full). 40/40 baselines regenerated
via `verify_baselines.py --update`.

**Empirical finding (2026-05-26).** Across all 15 SA baseline
scenarios (5 placement strategies × 3 conversion types), and across
all 550+ scenarios in the regenerated SA dense CSV, the fellback
fraction is **0.000 / 0.000 / 0.000** (min / median / max) for every
target. NatCap's compound crosswalk has comprehensive coverage of
every (NLUD × tree-canopy) tuple actually present in SA's
developed-land pool, for all three conversion targets (NLCD 41 / 90
/ 24). The default-lucode fallback rule never fires in practice; the
preserve-context rule does 100 % of the work. The "is the default
principled?" methodology question is therefore academic — the
defaults' choice doesn't affect any current scenario output.

The instrumentation is still the right move: it makes the answer
explicit and falsifiable, so any future change to NatCap's crosswalk
or to the SA LULC raster that breaks the coverage assumption would
surface immediately in the dashboard panel rather than silently
shifting outcome values via the default lucodes.

## NatCap ROOT as reference point

NatCap's **ROOT** (Restoration Opportunities Optimization Tool) is a
linear-programming-based multi-objective optimization tool for spatial
decision-making. It maximizes weighted sums of objectives
(`max Σ wᵢ Vᵢₛₐ xₛₐ`) over spatial decision units (SDUs), producing
true Pareto frontiers (production possibility frontiers) and agreement
maps showing which SDUs are robustly selected across weight
configurations.

**What ROOT does that the prototype doesn't:**

- True LP-based Pareto optimization at the SDU level — guarantees
  mathematically optimal solutions on the feasibility frontier, not
  heuristic approximations.
- Agreement maps across weight configurations — visualizes which
  spatial decisions are robust to objective weighting choices.
- Cost-as-factor optimization — costs can enter the optimization as
  constraints or factors, not just as post-hoc ratios.
- Operates on rasterized factor layers without requiring a precomputed
  scenario grid.

**Why the prototype uses a surrogate-based optimizer instead:**

ROOT is a desktop optimization tool designed for analyst workflows —
runs take minutes to hours, results are produced for offline analysis.
The prototype is an interactive dashboard where slider responses need
to be millisecond-fast. A Random Forest surrogate over the four
scenario sliders, trained on a precomputed scenario grid, enables
interactive scenario exploration at the cost of giving up true Pareto
optimization. The surrogate approximates the relationship between
scenario parameters and outcomes; ROOT computes the exact optimum
given the LP formulation.

**"Not pursued" means different tool, not wrong tool.** ROOT is the
right tool for analyst-driven offline optimization with strong
guarantees. The prototype's surrogate is the right tool for
interactive sandbox exploration. Adopting ROOT would mean
re-architecting the prototype around a different user model — not a
correctness fix to the current architecture.

**Where ROOT shows up in the docs:**

- `REFERENCE.md` Balanced placement strategy (~line 255) — pointer to
  ROOT for true multi-objective optimization
- `REFERENCE.md` Cost Effectiveness section (~line 549) — pointer to
  ROOT for cost-as-factor optimization
- `REFERENCE.md` Smart Scenario Search (~line 665) — explicit
  Relationship-to-ROOT paragraph
- `NATCAP_ALIGNMENT.md` Research-direction synthesis (~line 140) —
  ROOT listed under "Main pending work" with framing as not currently
  pursued
- `NATCAP_ALIGNMENT.md` Section 5 Research Direction Status
  (~line 161) — ROOT row with status 🔵 Not pursued
- `NATCAP_ALIGNMENT.md` Section 6 Vocabulary (~lines 177–179) — three
  rows where prototype features point to ROOT (Cost Effectiveness,
  Balanced placement, Smart Scenario Search)

## Signed metric cards — label-flip rule (Brief 1, 2026-05-28)

Three "dollar/count" metric cards can render negative values for
scenarios that make things worse (e.g. converting vegetated land to
high-density development): **Preventable MH Cases**, **Avoided MH
Costs**, and **Carbon Storage Value**. Each is hand-rolled inline
(there is no shared metric-card renderer), so the rule below is applied
per-card:

- **Positive value** → benefit label, positive magnitude, green delta.
- **Negative value** → harm/loss label, magnitude shown as a positive
  number (no leading minus), red (`delta_color="inverse"`) delta.

The negative-case labels are: "Preventable MH Cases" → "Additional MH
Cases"; "Avoided MH Costs" → "Added MH Costs"; "Carbon Storage Value" →
"Carbon Storage Loss" (SA) / "Avoided Carbon Cost" → "Added Carbon Cost"
(MN). Brief 1 also brought Carbon Storage Value's negative color into
alignment with the MH cards — it previously used neutral `"off"` for
negatives while the MH cards used red `"inverse"`. (MN's carbon is
always ≥ 0, so its loss labels are defensive and don't surface in
practice; SA's four-pool stock model is the one that can go negative.)

**Landed in Brief 2 — the Carbon Storage Change *quantity* card.** The
Ecological-section carbon card (showing tonnes of CO2e via
`results['carbon_tons_co2']`) previously rendered raw signed values
through the shared `_delta_pill` helper. Brief 2 took **Approach Y**:
the SA four-pool stock branch is now bespoke (flips to "Carbon Storage
Loss" with a positive magnitude and a red ↑ delta — `delta_color="inverse"`
— when conversions reduce stored carbon), mirroring Brief 1's signed
dollar/MH cards. MN's annual-sequestration flow is always ≥ 0, so it
keeps the `_delta_pill` path and the "Carbon Sequestration" label. After
this, `_delta_pill` serves exactly three always-positive cards (Flood
Retention, Runoff, NDVI), so its uniformity is preserved without growing
a polarity parameter (Approach X was rejected for that reason).

## Brief 2 — naming and labels for correctness

Display-only label cleanup; no metric computation changed, baselines
unaffected.

- **"Flood Risk Reduction" → "Flood Retention"** on the lead Ecological
  card, the tradeoff-plot x-axis, the optimizer-slider help, and the
  Baseline-vs-Scenario comparison table. `flood_reduction = 100 − mean_CN`
  (`app.py`) is computed identically for both cities and is a retention
  index, not a damage-curve risk metric — so the rename applies to MN and
  SA alike (the lead-card label is not city-conditional). **Known
  consequence (resolved in Brief 7):** the rename briefly made SA show the
  same `flood_reduction` value under the "Flood Retention" name twice — the
  Ecological card (index 0–100) and the Economic card (Brief 33's % volume
  framing) — plus two comparison-table rows. Brief 7 took **Approach B
  (differentiate)**: the Economic card and its comparison-table row were
  renamed to **"Flood Volume Reduction"**, so the two now carry distinct
  labels matching their distinct framings. The Ecological card keeps
  "Flood Retention".
- **SA carbon confidence badge.** The carbon *quantity* card's badge
  (`_confidence_caption`) was hard-coded "Prototype" for both cities,
  which undercut SA's NatCap four-pool framework (Brief 30). It's now
  city-conditional: SA → `"Four-pool stock (NatCap framework)"` (a new
  `_CONFIDENCE_BADGES` key — a methodology descriptor, not a confidence
  tier); MN keeps "Prototype" (the single-rate proxy is genuinely
  lower-confidence). The SA card's help-text leading line still reads
  "Confidence: Medium" — left as-is; it's an accurate confidence level
  and complements the methodology badge rather than contradicting it.
  The carbon *dollar* card (econ5) already used "medium" and was left
  untouched.
- **Cost-effectiveness labels.** "Cost / °F Cooling" → "Cost / Citywide °F
  Cooling" (the °F is a city-average, which made the raw ratio look
  absurdly large); "Cost / Acre-Foot Prevented" → "Cost / Acre-Foot Runoff
  Prevented". "Cost / 1,000 People Fed" left unchanged (adding "/yr" to a
  people-count read worse than it clarified).
- **Map legend/caption ↔ slider sync.** The map legend patch and the map
  caption said "Heat vulnerability" / "heat vulnerability proxy" while the
  slider said "Development-intensity heat proxy opacity". All now read
  "Development-intensity heat proxy" — accurate to the underlying data,
  which is the `equity_weights`-derived NLCD-intensity proxy (23 > 22 > 21),
  not a real CDC/ATSDR HVI. The placement-exclusion note (buildings + roads
  excluded via OSM) already lives in the map's "Assumptions and limitations"
  expander, so nothing was added there.
- **Doc scope.** REFERENCE.md, NATCAP_ALIGNMENT.md, SUMMARY.md, and
  `app.py`'s methodology-tab pointer originally kept the old "Flood Risk
  Reduction" wording, deferred because the rename cascades through internal
  cross-references. **Resolved by the Brief 6 doc-sweep**, which renamed
  them all to "Flood Retention" (the REFERENCE.md heading plus every prose,
  table, and cross-reference mention; no `#flood-risk-reduction` anchor
  links existed, so nothing broke). No WHATS_NEW entry was added — these
  are clarity/consistency tweaks that don't clear the "would a returning
  user notice" bar, and the in-app changelog was just trimmed to three
  lean SA entries.

## Brief 4 — `cooling_f` → `temp_change_f` sign-convention refactor

**What changed.** The cooling metric was renamed `cooling_f` →
`temp_change_f` and its sign was flipped to the universal physical ΔT
convention: **`temp_change_f = T_after − T_before`, positive = WARMER,
negative = cooler** (`= −old cooling_f`). The producer is
`hm_to_temp_change_f(mean_hm)` (was `hm_to_fahrenheit_cooling`), which
negates the HM-index delta before scaling to °F (higher HM = more cooling
= lower temperature).

**Why.** The old convention (`positive = cooler`) created a "negative
cooling" oxymoron — most visibly in the neighborhood breakdown, where
deviations from the city average rendered as negative "temperatures" that
read like sub-zero absolute values. Two conventions also coexisted
(the main card's `cooling_f` and the breakdown's inline per-tract temps
both said "positive = cooler," but the framing confused users). Adopting
one physical ΔT convention everywhere removes the ambiguity. Users never
see the signed number: the display layer (`_fmt_temp_change`) always
renders natural language — "X°F cooler" / "X°F warmer" / "No change".

**Audit findings (the reason this was safe):**

- **The optimizer is untouched.** `surrogate.py` searches and ranks on
  `mean_hm` (the Heat Mitigation Index, higher = more cooling), never on
  `cooling_f`. The optimizer's cooling-target slider `min_cool_f` (°F) is
  converted to `mean_hm` units before the surrogate sees it. So the rename
  + sign flip changes no objective or constraint, and `mean_hm` was left
  exactly as-is. The feared "maximize → minimize" inversion does not apply.
- **Per-tract breakdown flipped to match.** `compute_per_tract_summary`
  computed its own per-tract temps with the old "positive = cooler"
  formula (independently of `cooling_f`). It was flipped to the new
  convention so the whole app speaks one language. Its columns were
  renamed to the `vs city avg` framing (each polygon's mean temperature
  relative to the city-wide baseline, positive = warmer); the change
  column renders as natural-language text and is color-coded (cooler =
  green, warmer = red) via a pandas Styler. The dormant map tract overlay
  (`plot_spatial_map(tract_value=…)`) is not wired into the live render,
  so it needed no change.
- **`cost_per_degf` (the "Cost / Citywide °F Cooling" card).** Now divides
  cost by `−temp_change_f` and is defined only when the scenario cools
  (`temp_change_f < 0`). The user-facing label stays as Brief 2 set it —
  it's correct regardless of the internal sign.

**Serialized artifacts regenerated.** Every consumer that captured the
field was updated: all 40 baseline snapshots (`verify_baselines.py
--update`) and both dense CSVs (`scenarios_dense_mpls.csv`,
`scenarios_dense_sa.csv`, regenerated sequentially). The read-only
baseline diff before regen confirmed the change was isolated — exactly
`cooling_f` removed + `temp_change_f` added (sign-flipped) per baseline,
no leakage into any other metric. `verify_cooling.py` was updated to read
the new key. No `SCENARIO_SCHEMA_VERSION` bump (field set unchanged in
count; the surrogate targets are untouched).

**Known-stale artifact: `data/scenarios_dense_mpls_full.csv`.** The hidden
"Minneapolis Full, MN" city (`available=False`) still carries the old
`cooling_f` column in its dense CSV. It was intentionally NOT regenerated:
the city isn't loaded live, `verify_baselines.py` only snapshots available
cities (so Mpls Full has no baselines either), the surrogate never reads
`cooling_f`, and `precompute_scenarios.py` can't cleanly target a hidden
city (its stub selectbox is fed only available cities). Regenerate this CSV
when Mpls Full is activated, alongside its baselines.

## Brief 5 — sidebar reorganization + tooltips

**Sidebar order** now follows the configure-then-optimize workflow:
City → Land Use Scenario → Conversion Mix (with Quick Start buttons) →
**Placement Strategy** → **Find Best Scenario** → Implementation Costs →
Advanced Settings. Placement Strategy previously sat *below* Find Best
Scenario, which inverted the intuitive flow — placement shapes the
*current* scenario, so it belongs alongside the conversion mix, before
the optimizer (an advanced, optional action). The move is purely visual:
`placement_strategy` / `use_heat_priority` are not referenced by the Find
Best block, and both are still defined before the main-panel `results`
computation, so no `st.session_state` init reordering was needed.

**Find Best Scenario text** was trimmed — a one-line inline caption plus a
"How this works" expander, instead of three stacked captions before the
controls. The expander prose was corrected for accuracy: the surrogate
explores conversion percentage and conversion mix only; **placement
strategy and cost are not part of the surrogate** (the earlier draft text
that implied placement was searched was wrong). The "~90 simulations"
figure is now framed as the Fast-mode count (higher-quality modes use
more), fixing a stale headline.

**Nature Access tooltip** gained an explicit AOI line (city-conditional:
"City of San Antonio (ACS block groups)" vs the downtown Minneapolis
extent / Census tracts) and a saturation explanation — high values on
food-forest scenarios are the model's intended behavior (converted
developed pixels become nature-supplying), not a suspicious outlier. The
demand standard (16.7 m²/capita) and 800 m search radius were already
interpolated from the city config, so no value is hardcoded.

## UMH validation against canonical InVEST 3.19.0

**What was done (2026-05-29).** Closed the gap that the Carbon/UMH coverage
investigation flagged: the prototype's numpy Urban Mental Health
reimplementation had no parity check against canonical InVEST (UCM and UNA
both have MAE≈0 results; UMH had none). `compare_umh_invest.py` now validates
it against `natcap.invest.urban_mental_health.execute()` (v3.19.0).

**Why a separate environment.** The app `.venv` is Python 3.9; canonical UMH
arrived in InVEST 3.18/3.19 which require Python ≥3.10 (the installed 3.16.2 in
anaconda base has no `urban_mental_health` module at all). So the validation
runs in a dedicated isolated conda env — **neither the app `.venv` nor anaconda
base was modified.** The two-environment "decoupled" harness pattern is
documented in `CONTRIBUTING.md` "Canonical-InVEST validation environments".

**Results (matched inputs, per outcome, both cities):**

| City | Outcome | Proto total | Canonical total | Aggregate Δ | Per-pixel rel MAE | Pearson r |
|------|---------|------------:|----------------:|------------:|------------------:|----------:|
| MN | depression | 126.94 | 128.57 | +1.3 % | 21.3 % | 0.946 |
| MN | anxiety | 85.79 | 86.90 | +1.3 % | 21.3 % | 0.946 |
| SA | depression | 3,980.2 | 4,038.5 | +1.5 % | 16.4 % | 0.975 |
| SA | anxiety | 2,690.7 | 2,730.4 | +1.5 % | 16.4 % | 0.975 |

**The finding.** Canonical UMH 3.19.0 computes neighborhood exposure (NE) as a
uniform **buffer-mean** within the search radius (it emits `ndvi_*_buffer_mean`
rasters + a `kernel` output, and pads the grid by the radius); the prototype
uses a **Gaussian** (σ = radius/px). These are genuinely different kernels. The
city-wide *aggregate* preventable-cases total — the number the dashboard
reports — matches to ~1.3–1.5 % because both kernels roughly conserve total
NDVI exposure. But *per-pixel* they redistribute exposure differently, hence
r ≈ 0.95–0.98 rather than the MAE≈0 of UCM/UNA. `effect_size` semantics were
confirmed correct (= RR per 0.1 NDVI): if they were wrong the totals would
diverge, and they don't. The prototype's old code comment claiming the Gaussian
"matches the canonical behavior" was inaccurate and has been corrected
(app.py); REFERENCE.md already framed it as "similar but not identical" and now
carries the quantified result.

**Decision: document, don't change the formula (Option A).** The reported
aggregate metric validates well and the Gaussian is a defensible smoothing
choice, so the MH cards stay **Medium** confidence (per-pixel agreement isn't
tight enough to claim "High", and the metric is fundamentally spatial). Two
divergences could not be neutralized in the harness and are documented rather
than hidden: (a) the prototype's uniform national CDC prevalence vs InVEST's
per-admin `risk_rate` vector — we have no per-tract MH-prevalence data for
MN/SA, so the "default-input" MAE coincides with the matched-input number;
(b) the NE kernel itself, which is inherent. Validation is on the prototype's
**synthetic NDVI proxy**, so it validates the algorithm, not the NDVI source.

**Deferred — Brief B (exact-parity kernel switch). → DONE, see below.**
Switching the prototype's NE from a Gaussian to a buffer-mean brings per-pixel
parity to canonical; it shifts `preventable_mh_cases` / `avoided_mh_cost_usd`
for every scenario → requires regenerating the 40 baselines + both dense CSVs
and a `SCENARIO_SCHEMA_VERSION` bump. Landed in Brief B (next section).

## Brief B — UMH NE kernel: Gaussian → buffer-mean

**What changed.** The UMH neighborhood-exposure (NE) kernel was switched from a
Gaussian (`σ = search_radius / pixel_size`) to the canonical InVEST UMH 3.19.0
**buffer-mean**: an edge-corrected mean of the NDVI proxy over a flat binary
disk of radius `search_radius / pixel_size` (apothem 10 px at 30 m / 300 m,
317-pixel disk). Implemented by reusing the existing `_convolve_edge_corrected`
helper (the prototype's port of
`pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True)`) with a disk kernel
and `kernel_sum=1.0` so it returns the local mean. The new
`_umh_neighborhood_exposure` helper feeds both the scenario NE
(`calculate_mental_health_impact`) and the baseline NE
(`_BASELINE_NE_RASTER`). `scipy.ndimage.gaussian_filter` and `_UMH_SIGMA_PX`
are gone.

**How the kernel was pinned down.** Probed canonical's emitted `kernel.tif`
(21×21 binary disk, sum 317) and confirmed edge-correction via an all-ones NDVI
test (buffer-mean = 1.0 everywhere, including edges). An independent random-NDVI
probe matched the candidate to MAE 1e-8.

**Validation result (`compare_umh_invest.py`, post-fix):**
- **MN: exact** — MAE ≈ 1e-9, Pearson r = 1.000000 (both outcomes); proto
  totals now equal canonical (128.6 dep / 86.9 anx).
- **SA: MAE ≈ 0 on aligned input.** Feeding the app's raw 30 m grid, the harness
  showed MAE rel 0.14 %, r = 0.9988 — *above* the brief's 0.1 % gate. The
  alignment diagnostic resolved it: applying the app's NE kernel to canonical's
  **own aligned NDVI** (`ndvi_*_aligned_masked`) matched canonical's
  `*_buffer_mean` to **MAE 9.4e-9** over all 3.47 M pixels. So the kernel is
  mathematically identical; the 0.14 % was large-grid feeding-alignment + FFT
  noise in the *comparison*, not a metric divergence. (Lesson: on the big SA
  grid, integer-pixel cropping of canonical's padded output can't absorb
  sub-pixel alignment — validate against canonical's aligned raster directly.)

**Propagation.** Read-only baseline diff confirmed drift isolated to
`preventable_mh_cases` + `avoided_mh_cost_usd` only (60 divergences across the
30 conversion scenarios; the 10 zero-conversion baselines unchanged; shifts
~1.5–3 %). Regenerated all 40 baselines + both dense CSVs;
`SCENARIO_SCHEMA_VERSION` 26 → 27.

**Confidence.** MH cards upgraded Medium → High — the algorithm now has
per-pixel canonical parity. The synthetic-NDVI-proxy caveat is retained in the
card help as a separate *input-quality* note (real-satellite-NDVI is a distinct
future workstream). The remaining documented divergence is the per-admin
prevalence vector (a), unquantifiable without per-tract MH-prevalence data.

**Harness note.** Fixing this surfaced a latent bug in `compare_umh_invest.py`'s
export step (from Brief A): it recomputed the prototype NE with an *inline copy*
of the Gaussian rather than calling the app's actual UMH path, so it didn't
track the kernel change. The exporter now calls
`app._umh_neighborhood_exposure`, so the harness validates the shipped code.

## Brief A2 — SA UNA AOI investigation (document-only)

**Question.** Yingjie's roadmap says NatCap's SA UNA uses
`acs_block_group.gpkg` as the AOI; the prototype was thought to use a
City-of-SA clipped extent. If they differ materially, per-scenario UNA
comparison against NatCap's published outputs would be over different areas.

**Finding.** Measured (LULC-valid mask vs block groups rasterised to the 30 m
EPSG:5070 grid): the prototype's UNA extent is the **Bexar County bbox**
(3,059 km², 1,906,325 people); NatCap's block groups (1,124 polygons, City of
SA) are a **strict subset** (2,519 km², 1,878,866 people). **Area IoU = 0.824**,
but **population overlap = 98.6 %** — only **27,457 people (1.4 %)** are in the
bbox but outside the block groups (sparse exurban Bexar County). Detail +
numbers in NATCAP_ALIGNMENT.md "SA UNA / biophysical extent".

**Architectural insight (the reason this isn't a config swap).** The
prototype's UNA path is **raster-only**: `calculate_nature_access(scenario_lulc,
pop_count_raster)` takes no AOI vector — the modelable extent is wherever the
LULC/population rasters have valid data. The `acs_block_groups_3857.gpkg` in the
repo feeds **only** `compute_per_tract_summary` (the neighborhood-breakdown
table), not any biophysical model. So "swapping the AOI" would mean *adding* a
block-group mask to the UNA computation — coupling the polygons into the
biophysical path where they currently aren't — not repointing a config path.

**Decision: document, don't change (Option b).** The area IoU tripped the
brief's 0.95 gate, but UNA's headline is **population-weighted** and the
extents are 98.6 % population-aligned; the discrepancy is 1.4 % of population on
sparse exurban land. Per-pixel `urban_nature_supply_percapita` is computed
identically regardless of aggregation extent — so the real validation need
(matching NatCap's per-block-group `ntr_bal_avg` in `nootenboom_results`) is met
by **aggregating the prototype's supply raster per block group**, which is a
Track C concern, not an A2 AOI change. Masking the UNA path would cost a code
change + full SA baseline/dense-CSV regen + schema bump for a sub-1 % effect —
cost exceeds value. No code, no baseline, no schema change in Brief A2.

**Forward note.** Track A3 (CSV population) and Track C (parity validation vs
`nootenboom_results`) should both use **per-block-group aggregation** of the
prototype's supply raster, not the citywide headline, for SA UNA comparison.

## Brief B1 — NatCap fixed scenarios as first-class inputs (2026-05-29, partial)

**Goal.** Make NatCap's seven SA project scenarios (`baseline`, `FF_20ac`,
`FF_40ac`, `FF_MAX`, `UA_20ac`, `UA_40ac`, `UA_MAX`) loadable as first-class
scenarios, identified by the `scenario_id` values in
`data/sa/natcap_reference_outputs.csv`, so B2's per-metric validation badges can
join a loaded scenario's prototype output to NatCap's published reference value.
This is the keystone for the "validated scenario exploration" workflow.

**Investigate-first finding — the scenario rasters are flood-encoded, not
compound.** NatCap shipped the SA scenario LULCs (`sa_lc_w_*_10m.tif`) only in
the **NLCD × tree-canopy 3-tier** encoding (211/212/213 = NLCD 21 × low/med/high
canopy; `998` = food forest, `999` = garden), at 10 m in a WGS84-datum Albers
CRS. That matches the prototype's SA **flood** CN table
(`biophys_floodmitig_sa.csv`) directly — so the flood metric computes correctly
on them. It does **not** match the compound NLCD×NLUD×tree `lucode` (0–1983)
encoding the **Carbon / UCM (temperature) / UNA** tables are keyed on. The
integers collide with unrelated compound classes (scenario code `901` = woody
wetland, but compound `lucode 901` = Perennial Ice/Snow; `999` = garden, but
compound `lucode 999` = Mixed Forest), so feeding these rasters into those
tables produces silent, wrong-but-plausible numbers. The reference CSV's only
two `natcap_published` metrics — `temp_change_f` and `carbon_tons_co2` — are
both compound-keyed, so carbon/temperature reproduction (Phase 3) is **gated**
pending NatCap's compound scenario inputs.

**Local hunt for compound scenario rasters — empty (2026-05-29).** Searched by
content signature (max value + unique-value count), not just filename, across
`~/Desktop`, `~/Downloads`, `~/Documents`, `/Volumes`, and the Google Drive
sync root, plus inside the `_zip_archive` zips. Every compound-signature raster
on disk (max ~1900, ~800 unique values) is a **baseline**: five distinct
contents (`lulc_overlay_3857.tif`, `land_use_compound_sa.tif`, and the per-model
baseline inputs `ucm/intermediate/lulc.tif`, `una/intermediate/aligned_lulc.tif`,
`una/intermediate/masked_lulc.tif`), duplicated 17× across the pulls + GDrive
mirror. The `InVEST Results/` tree is a single baseline run (carbon outputs
suffixed `_cur`); there are no scenario-suffixed inputs or per-scenario result
sets. NatCap's per-scenario carbon/temperature spreadsheet values therefore came
from runs whose compound inputs were never shared in this drive pull. The NatCap
data request that would un-gate Phase 3 is **parked (not sent)** in
`OPEN_QUESTIONS.md` → "Per-scenario compound LULC inputs", with a send-ready
draft; nothing goes out without an explicit decision.

**What landed in this commit (scope A — standalone scaffolding).**
`natcap_scenarios.py` (Streamlit-agnostic, mirrors `surrogate.py`):
- Provenance taxonomy: `PROVENANCE_BASELINE` / `_NATCAP_FIXED` / `_EXPLORER` /
  `_OPTIMIZER`. B2 keys its badges off `(provenance, scenario_id)`; D1's
  generator block reads provenance.
- `SA_NATCAP_FIXED_SCENARIOS` metadata dict, keyed on the CSV's `scenario_id`s.
- `load_natcap_fixed_scenario(scenario_id, reference_grid_path)` — reproject +
  `Resampling.mode` (majority-rule, categorical) onto the prototype's 30 m
  EPSG:5070 SA grid; `lru_cache`d. Returns `(lulc_nlcd_tree, metadata)`.
- `flood_reduction_from_nlcd_tree(...)` — pure, dependency-injected flood helper
  mirroring `evaluate_scenario`'s SA CN path. Runoff stays in app.py's
  `cn_to_runoff_acre_feet` (single source for the SCS-CN formula).

**Why the dashboard wiring was deferred to B2 (Phase 0 D — invasive-change
gate).** `evaluate_scenario` is monolithic and conversion-centric: it builds the
scenario raster internally from slider params and computes all metrics in one
pass; there is no pre-built-raster entry point. Routing the dashboard to a fixed
scenario forces every compound-keyed card + the tradeoff plot + the `_hm_delta`/
carbon delta math to render a value they cannot compute on the flood encoding —
which is precisely B2's deferred per-card display work (published value + badge).
So B1 lands the standalone loader/provenance/flood infrastructure; B2 owns the
selector→dashboard routing and per-card display in one coherent pass.

**Smoke-test result + a flagged divergence.** `python natcap_scenarios.py`
reprojects all six alternative scenarios and computes flood: `mean_cn ≈ 81.4`,
`flood_reduction ≈ 18.5–18.6`, varying only minutely across scenarios — matching
NatCap's documented SA conclusion that flood is essentially scenario-invariant
under design-storm saturation. **B2-investigate item:** the prototype's *own* SA
baseline is `mean_cn 76.54` / flood 23.5, a ~5-point CN gap from these scenarios
(~81.4). Both use the same CN table, so the gap points to the prototype's
**compound→NLCD×tree reduction** (`tier = max(tree, 1)`) producing a different
canopy-tier mix than NatCap's native NLCD×tree raster. Not a NatCap-published
mismatch (no NatCap flood reference exists), so it doesn't gate B1 — but it means
a "food forest" fixed scenario currently reads as *lower* flood retention than
the baseline, which B2 must reconcile (align the baseline's flood derivation to
the native NLCD×tree path, or annotate the card) before surfacing the number.
Tracked in `OPEN_QUESTIONS.md` → "Native NLCD×tree baseline flood raster".

**Interim badge stance for B2 (captured here so it isn't lost).** For a fixed
scenario, display NatCap's published carbon/temperature value from the reference
CSV and badge it "NatCap published"; once Phase 3 reproduction passes (compound
inputs arrive), upgrade to "✓ NatCap match — independently reproduced, Δ X%".
Flood can show the prototype's computed value now (no NatCap reference).

**Deploy-safety note.** The loader resolves scenario rasters from external paths
(`~/Desktop/natcap_drive_pull/...`); the in-repo `data/sa/natcap_scenarios/` dir
is preferred but not yet populated. When B2 wires this into the deployed app,
the chosen scenarios should be vendored + pre-reprojected into the repo
(Streamlit Cloud has no `~/Desktop`). Deferred to avoid committing binaries in
this code-only scaffolding commit.

## Brief D1 — Export for InVEST workflow (2026-05-29)

**Goal.** Bridge the prototype to canonical InVEST: a single sidebar button packages
the currently-displayed scenario as a runnable **InVEST 3.19.0** input zip — rasters
+ AOIs + biophysical tables + per-model `args.json` for **UCM / UNA / UFR / Carbon /
UMH** + a `metadata.json` recording provenance, generator parameters, and per-model
validation state. Delivers value-ladder Level 5 ("workflow layer") as a concrete
capability, not aspirational. SA-only for v1.

**Architecture.** New `export_invest_bundle.py` is Streamlit-agnostic (mirrors
`surrogate.py` / `natcap_scenarios.py`): a `BundleSpec` dataclass + a `build_invest_bundle(spec)
→ bytes` builder that writes the zip in memory (Streamlit-Cloud-safe, no temp dir).
The app.py caller (`_build_invest_bundle_for_current_scenario`) gathers runtime
state into the spec and calls the builder; the sidebar uses a two-step
**Prepare → Download** flow (avoids rebuilding the ~20 MB bundle on every rerun)
plus a "Clear prepared bundle" reset.

**Scenario-vs-baseline shape.** Baseline scenarios export `provenance=baseline`;
Explorer-generated scenarios export `provenance=explorer_generated` with the full
slider parameter set + `placement_strategy` + `random_seed`. The polymorphic
`generator` block keys off the `natcap_scenarios.py` provenance taxonomy. Carbon
runs `lulc_bas` + `lulc_alt` in a single args file; UCM/UNA produce one result per
LULC — the bundle includes both the scenario and baseline rasters so the user
re-runs each for the delta (documented in the bundle README). NatCap fixed
*alternative* scenarios are flagged in the bundle metadata as
`compound_models_available=False` and export the flood source + UFR args only —
not yet wired through the sidebar (compound LULC unavailable per Brief B1;
`OPEN_QUESTIONS.md`).

**AOIs.** Synthesized **bounding-box** polygon (`aoi_prototype_extent.gpkg`) for
UCM / UFR (watersheds) / UMH — citywide aggregation footprint matching the
prototype's citywide-mean reporting. NatCap's ACS block-group polygons
(`aoi_natcap_block_groups.gpkg`) are used for UNA's `admin_boundaries` — the
framing NatCap uses for SA equity analysis (Vibrant Land). The
`metadata.json → aoi` block records each AOI's role + reason.

**Per-model args choices** (Phase 0 introspection of InVEST 3.19.0 `ModelSpec`):
UCM `cc_method="factors"` with weights 0.6 / 0.2 / 0.2 mirroring the prototype's
HMI formula; **`do_energy_valuation=False`** — biophysical-cooling only, no
building-vector dependency, no Cooling Energy Savings $-card reproduction (noted
in README + metadata). UNA `decay_function="dichotomy"`, `search_radius_mode="uniform
radius"`, `search_radius=800`, `urban_nature_demand=16.7`. UFR uses NatCap's
NLCD×tree CN table directly; damage valuation omitted (no SA damage table, Path
C). Carbon `do_valuation=False` (stock change only — keeps the run minimal +
matches the prototype's SC-CO2-applied-separately framing). UMH `model_option="ndvi"`
+ a synthesized polygon `baseline_prevalence_vector` carrying `risk_rate` =
CDC ever-diagnosed BIR; two args files emitted (depression, anxiety) since
InVEST UMH's `effect_size` is per-condition.

**Phase 3 verification — all 5 models PASS on the baseline bundle** (canonical
`natcap.invest` 3.19.0, conda env `natcap_umh_validation`): UCM ✓, UNA ✓, UFR ✓,
Carbon ✓, UMH-depression ✓, UMH-anxiety ✓. Per-model validation states in the
exported `metadata.json` reflect this: UCM / UNA / UMH = `validated`; Carbon / UFR
= `methodology_aligned` (canonical method, framing differences documented in
notes). Brief amendment had marked UMH best-effort/unverified; it ran cleanly,
so the hedge is dropped and UMH is recorded `validated`.

**Nodata sentinel rule (general truth, surfaced by D1 Phase 3 UFR failure-then-fix).**
The prototype tolerates unmapped or sentinel lucodes — its CN aggregation filters
`CN > 0` and silently drops them. **Canonical InVEST does NOT** — InVEST UFR
`_lu_to_cn_op` raises `ValueError` (with a misleading empty `[]` lucode list, a
known display bug) when any non-nodata pixel maps to an all-zero CN row. So
**every exported raster's `nodata` tag must match the sentinel that raster
actually carries**: the NLCD×tree-reduced view emits the prototype's **−128**
sentinel for outside-boundary pixels → write with `nodata=-128`; compound LULC
nodata is **−1** (matches the source); NDVI is float and uses **−1.0**.
Initial D1 UFR run wrote the NLCD×tree raster with `nodata=0`, leaving 35,973
−128 pixels unmasked — InVEST raised on them. Fixed in
`export_invest_bundle.py` (`nodata=-128` for both nlcdtree rasters); all five
models then PASS. Save the next export path — fixed-alternative, a fresh
Explorer-scenario export, or any future raster added to the bundle — from
rediscovering this the hard way.

**WHATS_NEW + Underway discipline.** Added a WHATS_NEW entry — D1 clears the bar
(major user-visible capability: a new sidebar button + downloadable bundle).
Underway stays empty.

## Brief B2 (revised) — Validation badges + NatCap fixed-scenario reference view (2026-05-29)

**Context.** The original B2 (per-metric Match/Diverged badges keyed on a
prototype-computed value vs NatCap's published value) was deferred earlier this
session (see `OPEN_QUESTIONS.md` → "Deferred briefs") because Match/Diverged
needs prototype reproduction for the six NatCap fixed alternative scenarios,
which is gated on the unavailable compound scenario inputs. The revised brief
expanded scope: keep Match/Diverged out, but deliver the ungated symposium core
— the three-state taxonomy as badges, a dedicated **fixed-scenario reference
view** routing around the monolithic `evaluate_scenario`, a side-by-side
comparison surface, baseline reproduction posture, and a flood reconcile.

**Conservative-floor decision (this session).** The investigation pass under
the "no parameter fitting" guardrail established that NatCap's published
**citywide absolute baselines** aren't reproducible from disk:

- **Temperature.** No SA UCM `args.json` ships anywhere in the drive pull. The
  `T_ref` / `uhi_max` NatCap used for `avg_temp_f = 90.08 °F` are not
  documented or recoverable. (`T_air_nomix.tif` exists; back-solving from it
  crosses the no-fit guardrail.)
- **Carbon.** NatCap's `tot_c_cur.tif` (EPSG:3857, ~34.5 m nominal pixel,
  5,283 km² extent, mean 17.2) does NOT aggregate to the published 107.32M
  t CO2e by any standard interpretation (per-ha → 25–33M depending on the
  cos²(lat) area choice; per-pixel-total → 280M). The published number is a
  separate aggregation script that wasn't shipped. The prototype's own Bexar-
  bbox four-pool sum is 147.96M.
- **A3 status.** `natcap_published` is "comparison-READY, never executed."
  The only callers of `compare_to_reference` are the four-line `__main__`
  smoke test with hardcoded values. No `evaluate_scenario → compare_to_reference`
  pipeline has ever run, because the only `natcap_published` metrics
  (`temp_change_f`, `carbon_tons_co2`) are exactly those gated by the
  unavailable compound inputs.

Net: a clean citywide-absolute reproduction match is not achievable from what's
on disk. The demonstrable reproduction claim, in order of strength, is **per-
pixel parity vs canonical InVEST** (HMI MAE 0.0000 / r 1.0000, Brief 28b; UMH
MAE ≈ 0, Brief B), then **four-pool methodology adoption** (Brief 30 — a
methodology choice, not a parity measurement). The badge taxonomy was tightened
to match.

**Badge floor (`natcap_validation.render_validation_badge`).**

| context | metric class | badge |
|---|---|---|
| `NATCAP_FIXED` | `natcap_published` | **green "NatCap published value"** — the ONLY green case; the fixed-scenario reference view surfaces NatCap's number directly from `natcap_reference_outputs.csv`. |
| `BASELINE` / `EXPLORER` / `OPTIMIZER` | `natcap_published` | **blue "≈ NatCap method"** — prototype's own computation, methodology-aligned. |
| any | `aligned_method` | blue "≈ Aligned method" |
| any | `prototype` | gray "Prototype" |

**Metric-aware tooltip** on `≈ NatCap method`: temperature CAN cite measured
per-pixel HMI parity (Brief 28b); carbon MUST NOT — four-pool *methodology
adoption* (Brief 30) is alignment, not a per-pixel parity measurement, and
overclaiming would misframe the reproduction posture.

**Curated non-CSV-card status map** (for `_render_validation_caption(...,
explicit_status=...)` callers): runoff = aligned_method (canonical SCS-CN
derivation); NDVI = prototype (synthetic per-NLCD proxy); implementation cost
= prototype (slider artifact); flood damage avoided / volume reduction =
aligned_method; cooling energy savings = aligned_method (typed-OSM scope
caveat); carbon-$ = natcap_published on SA (derived from natcap_published
carbon × SC-CO2) / prototype on MN (per-cover annual proxy); MH costs =
aligned_method (pairs with MH cases); cost-effectiveness ratios = prototype.

**Reference-view architecture.** Routed by a sidebar "Scenario source" radio
(SA-only) — selecting "NatCap project scenario" + a scenario_id from
`SA_NATCAP_FIXED_SCENARIOS` calls `_render_natcap_fixed_scenario_view(...)`
then `st.stop()`s the script before the Explorer sidebar / sliders / optimizer
/ Export-for-InVEST block. Keeps the Explorer path untouched. The view reads
NatCap's published temp/carbon from `natcap_reference_outputs.csv` and the
flood metric via the B1 helper (`natcap_scenarios.flood_reduction_from_nlcd_tree`)
— it does NOT route through `evaluate_scenario`. Compound-gated cards (Nature
Access, MH cases+costs, Cooling Energy, Food, NDVI, Cost & CE) are listed in an
explicit "not available for this NatCap scenario" section, with a pointer to
`OPEN_QUESTIONS.md`.

**Baseline panel — conservative reframe.** The original Phase-4 ambition was a
prototype-vs-NatCap absolute side-by-side with a ✓/✗ verdict. Given the
investigation result, that would put two ✗ markers on the symposium credibility
panel, for parameter/scope reasons not methodology divergence. Replaced with a
single plain-line claim:

> **Validated:** per-pixel parity vs canonical InVEST (HMI MAE 0.0000, Brief
> 28b; UMH MAE ≈ 0, Brief B). **Not established:** reproduction of NatCap's
> published citywide figures — their UCM args and carbon-aggregation script
> aren't recoverable from disk. See OPEN_QUESTIONS.md.

No numbers (which would invite a false comparison). Honest about what is and
isn't validated. If the audience is NatCap-savvy (in which the scope/parameter
caveats land naturally), a methodology-comparison panel with the figures + the
caveat would be the alternative — that's a one-line edit when needed.

**Cross-scenario comparison table.** A small `st.dataframe` over the seven
NatCap scenarios × (temperature, carbon stock change, carbon $) — all values
from the reference CSV (NatCap-published), with explicit "(NatCap published)"
column labels. Flood is intentionally excluded (different derivation between
baseline and alternatives — baseline uses compound→NLCD×tree reduction,
alternatives use NatCap's native NLCD×tree raster; the ~5-pt CN gap is mostly
artifact). The active scenario is marked with a ▶ prefix.

**Flood reconcile (per-scenario card).** The fixed-scenario flood card
suppresses the literal scenario−baseline delta and renders **"≈ invariant
(design-storm saturation, NatCap finding)"** — matching NatCap's documented SA
result that flood is essentially scenario-invariant. Tooltip explains the
derivation gap.

## Brief #3 — Scenario provenance + validation header badge (2026-05-29)

**Goal.** Make scenario-level provenance impossible to miss with a prominent
**Source + Validation** badge near the active scenario's title, present for
every scenario the main panel shows. Complements the per-metric card badges
(B2-revised) by describing the *scenario as a whole*; the card badges still
carry per-metric nuance.

**Architecture.** Single helper `_render_scenario_provenance_header(provenance,
scenario_label, scenario_id, trailing_caption)` drives off the `PROVENANCE_*`
constants re-exported from `natcap_scenarios.py` via `eib`. Five source-rule
mappings in `_PROVENANCE_HEADER_INFO`:

| Source label | Triggered when | Validation line | Color |
|---|---|---|---|
| `Baseline` | `pct_converted == 0` on Explorer dashboard | engine verified vs canonical InVEST; absolute NatCap citywide figures not reproduced | blue |
| `NatCap published reference` | fixed-scenario reference view | displayed from NatCap output; exact scenario raster / aggregation not available | green |
| `Explorer-generated` | slider scenario, non-baseline | canonical engine verified; scenario not NatCap-published | blue |
| `Surrogate-suggested` | optimizer-applied scenario | engine-validated; exploratory (Brief #4 will append "full-raster evaluated" once Apply has run) | blue |

The Validation strings come straight from STRATEGY.md §4's honest-claims
language — no new validation logic, no new verification claims.

**Two render paths wired.**
- **Fixed-scenario reference view** (`_render_natcap_fixed_scenario_view`) — the
  previous standalone `## label` + scenario_id/provenance caption is folded
  into the unified header. The "Sidebar source = NatCap project scenario.
  Flip to Explorer for custom scenarios." line is preserved as the helper's
  `trailing_caption`. Single canonical statement, not two.
- **Main Explorer dashboard** — new header rendered just before `#### Ecological`,
  with provenance detected from `results['pct_converted']` (0 → BASELINE, else
  EXPLORER). Labels are `"Baseline — {selected_city}"` or `"Explorer scenario ·
  {scenario_name}"`.

**OPTIMIZER provenance not yet detected at runtime.** The helper accepts
`PROVENANCE_OPTIMIZER` (wording in place), but the main panel doesn't yet
distinguish "applied from optimizer" from generic Explorer state — `apply_opt_*`
buttons in the optimizer panel set slider state and rerun, with no flag
persisting through the rerun beyond `applied_suggestion` (used for the ✓ marker
on the apply button only). Brief #4 will plumb a real Applied-from-Optimizer
flag through session_state so Apply → evaluate → header flips to
Surrogate-suggested, and the D1 export stops mis-recording optimizer scenarios
as Explorer.

**Visual treatment.** Left-border-coloured block + `Source:` / `Validation:`
labels, using `_VALIDATION_BADGE_COLOR_HEX` (the same green / blue / gray palette
as the per-metric card badges) so the scenario header reads as the scenario-
level analog. The block is a bordered `div` rather than a Streamlit native
component because Streamlit doesn't offer a colored callout that lets the
border color stay consistent with the per-card badge palette.

**Not in scope for this brief.**
- No new validation *logic* — this surfaces existing provenance + the validation
  boundary; nothing is computed differently.
- Per-metric card badges unchanged.
- Saved-scenarios display unchanged — the brief targets the *active* scenario's
  title; Brief #5 will reuse the same Source/Validation wording in the
  cross-source comparison table when saved/optimizer scenarios mix with NatCap
  rows.

## Brief #4 — Optimizer as trustworthy scenario discovery (2026-05-29)

**Goal.** Make the optimizer read as the bridge from exploration (L4) to
canonical validation (L5): clearer framing as scenario discovery,
predicted-vs-evaluated kept honest and visible, applied optimizer scenarios
properly tagged so they stop being mislabeled as Explorer-generated in the
D1 export bundle, and the Brief #3 "Surrogate-suggested" header firing on
actually-applied optimizer scenarios (was dormant in Brief #3).

**The Apply seam pre-#4.** The optimizer's surrogate-prediction table at the
bottom of the Tradeoff Analysis tab labeled itself "These are surrogate model
predictions. Click Apply to run a full pixel-level simulation and verify the
result." The Apply button (`apply_opt_*`) set `_pending_pct/gi/ff` +
`applied_suggestion` + `_show_apply_toast`, then `st.rerun()`. On rerun, the
pending values became slider values, `evaluate_scenario` ran on those values,
the main cards showed full-raster evaluated values — but nothing on the
dashboard signaled "this just came from the optimizer." The cards looked
identical to any Explorer-scenario rendering, and the D1 export silently
recorded the scenario as `PROVENANCE_EXPLORER`.

**Changes (single commit).**

1. **`applied_from_optimizer` flag through session_state.** A boolean
   session_state value plus `_applied_optimizer_values = (pct, gi, ff)`
   recording the just-Applied scenario's slider state. Set in the optimizer
   Apply button (`apply_opt_*`); auto-cleared at the top of every rerun if
   the current slider state diverges from `_applied_optimizer_values`
   (manual edit, preset button, Best-by-Goal Apply, etc.). Also reset on
   city change so an MN-applied scenario doesn't carry OPTIMIZER provenance
   into SA.

2. **OPTIMIZER provenance detection** plumbed into the **main-panel scenario
   header** (above `#### Ecological`) and the **D1 export helper**
   (`_build_invest_bundle_for_current_scenario`). Detection order:
   `pct_converted == 0` → BASELINE; else `applied_from_optimizer` →
   OPTIMIZER; else EXPLORER. The header label flips to
   `"Optimizer suggestion · {scenario_name}"`; the export records
   `PROVENANCE_OPTIMIZER` with an `optimizer_suggested` generator block
   carrying the slider params + a note *"Applied from Optimizer suggestion;
   full-raster evaluated by the prototype engine before export."*

3. **OPTIMIZER validation line updated** in `_PROVENANCE_HEADER_INFO` from
   Brief #3's placeholder *"engine-validated; exploratory"* to
   *"engine-validated; full-raster evaluated — exploratory candidate for
   further validation."* The OPTIMIZER provenance only fires on a scenario
   that has been Applied (slider state matches `_applied_optimizer_values`),
   so "full-raster evaluated" is always true when this badge appears.

4. **Entry-text reframe** in the sidebar. The subheader changed from
   *"Find Best Scenario"* (verdict framing) to *"Discover scenarios to
   validate"* (discovery framing). The first caption opens with
   *"Searches for promising scenarios to validate further."* and walks the
   discovery → Apply → evaluate → export path. The existing surrogate
   prediction-uncertainty language under "Optimized Scenario Suggestions"
   stays — it's already good.

5. **D1 export path** already worked for optimizer-applied scenarios (it
   operates on `results`, which is engine-evaluated post-Apply) — but the
   bundle's `metadata.json` recorded the wrong provenance. The OPTIMIZER
   branch fixes that so downstream readers of the bundle know the scenario
   came from the surrogate's discovery loop, not a manual Explorer build.

**Not in scope.**
- Best-Scenarios-by-Goal (`apply_best_goal_*`) is NOT classified as OPTIMIZER
  — those come from the precomputed scenario grid, not the surrogate's
  discovery search. They remain Explorer-generated when applied.
- Per-metric card badges unchanged (B2-revised already carries the per-metric
  validation state correctly).
- Surrogate prediction-uncertainty bands and the "These are surrogate model
  predictions" caption pre-existed; #4 only added the matching post-Apply
  provenance + framing story.

## Topics not yet documented

Sections that might land here when the relevant work happens. Listed
so future sessions know this doc is the right home.

- UCM cooling parameters (UHI_MAX_C, energy table, HMI vs energy aggregation)
- NDVI source (synthetic proxy vs satellite-derived AlphaEarth)
- Population data (Census 2020 block vs ACS block-group)
- Surrogate model architecture and hyperparameters
- "Wallpaper approach" — to clarify with NatCap (see Placement strategy section)
- SA data adoption (when access comes through)
- Mental health parameters (RR per 0.1 NDVI, cost-of-illness)
