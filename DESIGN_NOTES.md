# Design Notes

Running record of design decisions and considered-but-deferred directions
for the Ecosystem Explorer prototype. Organized by topic. Purpose:
single-source-of-truth for "what was considered, what was chosen, what's
still open" — useful before any conversation with collaborators about
methodology or future direction.

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

## Topics not yet documented

Sections that might land here when the relevant work happens. Listed
so future sessions know this doc is the right home.

- UCM cooling parameters (UHI_MAX_C, energy table, HMI vs energy aggregation)
- NDVI source (synthetic proxy vs satellite-derived AlphaEarth)
- Population data (Census 2020 block vs ACS block-group)
- Carbon sequestration methodology
- Surrogate model architecture and hyperparameters
- "Wallpaper approach" — to clarify with NatCap (see Placement strategy section)
- SA data adoption (when access comes through)
- Mental health parameters (RR per 0.1 NDVI, cost-of-illness)
