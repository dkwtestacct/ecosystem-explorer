# Design Notes

**Audience:** Internal
**Status:** Current technical decision log
**Use this for:** Why a given implementation choice was made
**Do not use this for:** Metric definitions (→ ../../REFERENCE.md), system structure (→ ARCHITECTURE.md), collaboration history or brief chronology (→ ../archive/HISTORY.md)
**Source of truth for:** Technical rationale and tradeoffs

---

Each entry below uses a single template — **Decision / Why / Alternatives considered / Consequences / Revisit if / Code touchpoints** — so future sessions can scan a section and pick up just the part they need. Code touchpoints cite stable symbol names; the line numbers drift but `grep` finds the symbols instantly. Where a decision overlaps ARCHITECTURE.md (system structure) or NATCAP_ALIGNMENT.md (validation taxonomy), the rationale lives here and the structure / taxonomy lives there, with cross-references rather than restated content.

---

## Decision log (2026-06)

One-screen index of this session's methodological decisions and their revisit triggers. Detail lives in the cited sections — this points, it doesn't duplicate.

1. **Population on non-residential land — NOT dasymetrically redistributed.** Measured effect is <1 % on clean cases (open water 0.04 %, hard NLCD exclusions ~0.77 %), and ownership class is a poor "no-housing" proxy (city land holds housing), so the correction doesn't earn its baseline-affecting blast radius. *Revisit trigger:* the binding constraint is ROI, not data availability — a residential/land-use layer is likely already derivable from the BCAD pull (state-use codes), so revisit only if the ROI justifies it, **not** "when data arrives." (Reframes the older "revisit when a parcel-level residential layer becomes available" wording in §6.8.) → **[§6.8]**

2. **SA ownership 16 % nodata — FIXED (`5751084`).** It was a 30 m grid-tiling artifact, not a coverage hole (the source is a complete 710,772-parcel BCAD pull); fixed by sub-pixel area-majority re-rasterization (16.14 % → 0.97 %), with the 0.97 % residual kept as nodata **by design** (cells with no parcel present at 10 m). *Revisit trigger:* the declined lever to reach ~0.2 % is a capped nearest-fill, which *infers* ownership for parcel-free cells — revisit only if a future need justifies inferred ownership. → **[§6.8, DATA_INVENTORY ownership entry]**

3. **Region-optimizer prefilter recall — VALIDATED; explore-reservation NOT built.** Top-1 regret = 0 across all selections × weights, because the current non-spatial engine has monotone-to-corner optima — the optimum is always a grid corner the candidate set already contains, so there's no interior optimum to drop and the K-cap never engages. *Revisit trigger:* re-validate if flow-routing / spatial dependency is added — the optimum stops being a grid corner, the K-cap engages, and recall would likely bite. → **[§7.3]**

4. **Flood / runoff — honest as a mean-CN lumped index** (100 − mean CN), not per-pixel-summed retention and not flow-routed; **placement-invariant by construction** (the score depends only on the mix of land covers in scope, not their siting). Linked to #3: the same change that breaks recall (engine becomes spatial) is what would make flood placement matter — a DEM + routing upgrade makes the optimization **non-separable and non-convex** (weighted-sum scalarization no longer suffices), i.e. a re-architecture, not an increment. → **[flood-routing expander + card help; §7.3 for the recall linkage]**

---

## 1. Documentation and naming conventions

**Canonical vocabulary.** The locked user-facing term set — and the retired phrasings that must not reappear — lives in `REFERENCE.md` § "Vocabulary (canonical terms)". A grep guard, `scripts/check_vocabulary.py`, is wired into the `verify_baselines.py` gate and fails the run if an unambiguous retired term reappears on a user-facing surface (`app.py` + `REFERENCE.md` / `CAPABILITIES.md` / `README.md`). "surrogate" is intentionally not guarded (legit code/methodology uses) — it stays a review item.

### 1.1 City-specific copy convention

**Decision.** User-visible strings that reference city-specific values (baseline numbers, data sources, climate framing, yield benchmarks) interpolate from one of four sources — never hardcoded.

**Why.** Adding a new city should be a config change, not a sweep through user-facing strings. Hardcoded city names in copy are a latent bug.

**Alternatives considered.**
- Hardcoded per-city strings — what the codebase started with; bug factory at each new-city onboarding.
- Single shared phrasing across cities — breaks when one city's data/framing genuinely differs (e.g. tract vs block-group nomenclature, °C-heat vs °C-cooling framing).

**Consequences.** The four allowed interpolation sources are:

1. `_CURRENT_CITY_STATE.*` for live-computed baselines (`baseline_cn`, `baseline_hm`, `baseline_ndvi`)
2. Module-level constants set from `city_cfg` — `UHI_MAX_C`, `HM_TO_FAHRENHEIT`, `FOOD_FOREST_LBS_ACRE`, etc.
3. A per-city dict (`_CITY_CAPTIONS`-style) — for prose that varies in structure, not just numbers
4. A `selected_city.startswith("Minneapolis")` branch — for paragraph-level prose with fundamentally different framing per city

The Temperature-assumption tab is the reference example for the branch pattern.

**Revisit if.** A third city joins — re-grep for hardcoded city names and audit the branch sites.

**Code touchpoints.** `_PROVENANCE_HEADER_INFO` (reference example for per-city branched prose), `_render_scenario_provenance_header` in `app.py`. Sanity check: `grep -n "Minneapolis\|\bMN\b" app.py` should turn up no hardcoded city names in user-facing strings.

---

## 2. City configuration and per-city parameters

### 2.1 Per-city NatCap parameter framing (the anchor)

**Decision.** NatCap parameters are project-specific by design — each city's project is tuned to its own policy framing. There is no single "NatCap canonical" value per parameter; alignment is per-city. The prototype adopts NatCap's published per-city parameters as documented in each project's canonical materials.

**Why.** SA's Urban Agriculture project uses WHO-minimum-green-space demand and heat-wave-day climate parameters; MN's project uses aspirational green-space targets and moderate-summer climate parameters. Different teams, different time periods, different policy goals. Forcing one set of "canonical" values would override one project's framing in favor of the other's — improvisation, not alignment.

**Alternatives considered.**
- Single set of "global canonical" parameters across cities — overrides one project's framing.
- Wait for explicit NatCap confirmation before adopting either set — defensible but blocking; the two project records are internally consistent and the question becomes confirmatory rather than gating.

**Consequences.** Every per-city parameter decision below (UNA, UCM, design storm, AOI polygons) follows the same logic: adopt the NatCap project's canonical value for that city; document the per-city divergence here rather than holding for blanket NatCap confirmation. For parameters where NatCap hasn't published per-city values, the prototype keeps a plausibility-level default and surfaces the gap in NATCAP_COLLABORATION.md as an open question.

**Revisit if.** NatCap publishes blanket cross-city canonical values that supersede the per-city project framing, OR the Natural Capital Symposium confirms the per-city framing is intentional (the expected outcome).

**Code touchpoints.** `config.py` `CITIES['Minneapolis, MN']` and `CITIES['San Antonio, TX']` — every per-city scalar (`uhi_max_c`, `design_storm_inches`, `una_demand_m2_per_capita`, `una_search_radius_m`, `una_decay_function`) is declared independently per city.

### 2.2 UNA parameters — per-city

**Decision.** Each city carries its own NatCap-project-canonical UNA configuration (demand, search radius, decay function). The per-city values are the source of truth in CITY_PARITY.md's UNA rows (MN section + SA section); this entry holds the *why*, not the values.

**Why.** MN's values come from the InVEST UNA sample bundle's `invest_urban_nature_access_args_MN.json`; SA's come from NatCap's SA README. Both reflect each project's documented framing — MN's higher demand (≈15× SA) targets aspirational green-space provision; SA's lower demand uses WHO-minimum framing. Per the single-home matrix, durable per-city parameter values live in CITY_PARITY; this section keeps the rationale.

**Alternatives considered.**
- A single shared UNA configuration across cities — overrides one project's framing.
- WHO 9 m²/capita as a global default — not investigated against either NatCap project.
- The InVEST generic default (also 250 m²/capita) — matches MN but inappropriate for SA.

**Consequences.** Headlines and per-pixel `urban_nature_supply_percapita` differ per city. The undersupply-focused placement strategy (§5) reads the per-city demand value, so its weights also differ per city.

**Revisit if.** NatCap publishes updated per-city values, or population-group stratification becomes a prototype goal (would require `aggregate_by_pop_group=True` + a population-subgroup raster).

**Code touchpoints.** `config.py` per-city `una_*` keys; `_una_supply_percapita` / `_una_supply_percapita_pure` in `app.py`; the exponential kernel is built canonically per `pygeoprocessing.kernels.exponential_decay_kernel`.

### 2.2a Children's nature access — measured, not built as an optimization target

**Decision.** `children_nature_access_pct` (UNA access share reweighted by under-18 population) stays a **diagnostic card** (hidden when it tracks overall access within 0.5pp — `_should_show_child_card`, gate-locked) and stays in the comparison table + export. It is **not** an optimization target and there is **no children-framed placement strategy**.

**Why (the Relay 55 → 59 thread, measured).** A cross-city sweep settled what the metric can actually move:
- **San Antonio — near-moot.** Baseline children's access 93.9% (6.1pp headroom); it tracks overall access within **≤0.3pp** across the mix grid and **saturates to 100% under any moderate conversion regardless of placement** (placement span 0.0pp at a 30/50/50 mix). A children's target would duplicate the existing nature-access goal.
- **Minneapolis downtown — real headroom but mix-driven.** Baseline 8.0% (92pp headroom); diverges from overall by up to **3.8pp**; **~96% of the downtown under-18 cohort (≈30,000) are underserved**. But the signal is **mix-driven** (26pp mix span vs 4.8pp placement span) — the lever is the *overall* conversion mix, not a child-specific search. The existing `undersupply-focused` strategy already captures most of the placement lift (+4.5pp), framed as "reach underserved residents," never "children."

**Conclusion.** Where children's access tracks overall (SA), a separate target is redundant; where it has headroom (MN), the real lever is the overall mix the nature-access goal already optimizes. So: measure it, surface it honestly via the hide-when-near-equal card, but do not present children as a separately served or separately optimizable group. The card's framing was softened accordingly (Relay 65); the surrogate fork (the citywide Fast model doesn't predict child access) is therefore left unaddressed by design, not oversight.

**Revisit if.** A city appears where children's access both diverges materially from overall AND responds to a lever the overall mix doesn't reach (a genuinely child-specific spatial signal) — then a child-weighted placement strategy (not a mix goal) would be the proportionate build.

### 2.3 UCM args — per-city UHI magnitude

**Decision.** `uhi_max_c` is per-city: **MN 2.05 °C** (from the InVEST UCM sample args.json), **SA 11 °C** (from NatCap's SA README, modeling a heat-wave-day scenario at 35 °C reference + 11 °C peak UHI).

**Why.** Each value is the NatCap project's documented configuration for that city. SA's 11 °C reflects a peak-stress modeling choice; MN's 2.05 °C reflects average-summer-day framing. Adopting both reflects the per-city framing principle (§2.1).

**Alternatives considered.** A single shared `UHI_MAX_C` (e.g. an average) — would simultaneously misrepresent both projects.

**Consequences.** SA temperature deltas under the 11 °C parameter are ~3× larger than under a moderate-summer parameter (an `all_gi` 10 % scenario shifts from ~0.13 °F to ~0.40 °F cooling). The HMI calculation is unchanged; only the ΔHMI → ΔT_°C scaling differs. `HM_TO_FAHRENHEIT = UHI_MAX_C × 1.8` auto-derives per city.

The remaining UCM args (`air_blending_distance = 600 m`, `maximum_cooling_distance = 450 m`, `cc_method = factors`, weights 0.6 / 0.2 / 0.2) already matched the prototype's existing values and are shared.

**Revisit if.** NatCap publishes a moderate-summer SA framing alongside the heat-wave-day one (would warrant exposing a UI toggle).

**Code touchpoints.** `config.py` per-city `uhi_max_c`; `HM_TO_FAHRENHEIT` (derived at module load); `compute_cooling_energy_savings` consumes both.

### 2.4 Design storm — per-city UFR rainfall depth

**Decision.** `DESIGN_STORM_INCHES` is per-city: **MN 3.94″** (100 mm, NatCap MN args.json), **SA 6.18″** (157 mm, NatCap SA README). Derived `DESIGN_STORM_MM = DESIGN_STORM_INCHES × 25.4` is used in tooltip display; the SCS-CN formula consumes inches internally.

**Why.** Per-city framing (§2.1) — each value reflects the NatCap project's design-storm choice. SA's heavier convective storms warrant a deeper design event than MN's lighter regional rainfall.

**Alternatives considered.**
- A 2″ global default — the prototype's previous value; not anchored in any NatCap or InVEST canonical source. **Retired** (see `../archive/HISTORY.md` "Retired infrastructure").
- A 100-yr design storm in both cities at NatCap's documented depths — adopted (current state).
- A typical-rainfall scenario — NatCap's methodology call is for the 100-yr depth; the prototype inherits.

**Consequences.** Two non-obvious cascades:
1. **SCS-CN nonlinearity in P.** `Q = (P − 0.2S)² / (P + 0.8S)` is not linear in P; doubling P more than doubles Q. The ~4–5× regeneration ratios in runoff metrics reflect this — *not* the 2× MN / 3× SA rainfall ratio.
2. **Flood-focused placement cascade.** The `flood-focused` suitability weight is per-pixel runoff `Q_{p,i}` at the design storm. When rainfall changes, weights shift, and the strategy picks slightly different pixels — flood-focused + balanced cells show small (<5 %) cascades on downstream metrics (UNA, UMH, cooling, NDVI). Intended behavior, not a bug.

**Revisit if.** NatCap publishes a typical-rainfall framing alongside the 100-yr framing — would warrant a UI toggle.

**Code touchpoints.** `config.py` per-city `design_storm_inches`; the SCS-CN call site inside `evaluate_scenario`; the `flood-focused` suitability formula in `_compute_suitability_weights`.

### 2.5 SA AOI — ACS block groups

**Decision.** SA uses NatCap's `acs_block_groups_3857.gpkg` (1,124 ACS block-group polygons covering the City of San Antonio) for the per-tract Neighborhood breakdown table. MN keeps `admin_boundaries_census_tracts.shp` (InVEST UFR sample); MN Full keeps `tracts_hennepin.shp` (TIGER 2020).

**Why.** NatCap's Vibrant Land report (Guerry et al. 2023) reports SA equity analysis at block-group resolution; providing the file alongside the compound LULC and biophysical tables signals it as the intended SA reporting unit. Block groups also tighten the SA reporting extent from full Bexar County (TIGER tracts) to the City of San Antonio.

**Alternatives considered.**
- Stay on TIGER Bexar County tracts — broader-than-city extent, not aligned with NatCap's reporting.
- Rasterize block groups at runtime — wasteful; the rasterization is done once at load time.
- Couple the polygon file into the biophysical computation (UNA in particular) — the prototype's UNA path is **raster-only** (`calculate_nature_access(scenario_lulc, pop_count_raster)`), so coupling a polygon mask would require code surgery for a sub-1 % population effect (Brief A2 investigation). Deferred.

**Consequences.** No biophysical metric value changes — the LULC raster's valid-pixel mask defines the modelable extent for every model. Only the Neighborhood breakdown table's row count (375 → 1,124) and the dashboard caption ("Census tracts" → "Census block groups" for SA) shift.

**Revisit if.** A future SA UNA validation needs per-block-group aggregation of the prototype's supply raster (Track C concern) — that's an additive feature on the supply raster, not a config swap.

**Code touchpoints.** `config.py` per-city `tracts_file`; `compute_per_tract_summary` and `_load_city_runtime_state`'s polygon rasterization step (polygon-name-agnostic, generalizes cleanly across the file shape).

---

## 3. Land-cover representation

### 3.1 NLCD legacy as the project-wide LULC vintage

**Decision.** The prototype uses NLCD 2021 (legacy MRLC product) across all cities. Future Annual NLCD migration is deferred.

**Why.** The InVEST sample data and the biophysical tables (CN, cooling, UNA) are all calibrated against the legacy 21-class schema. USGS replaced legacy NLCD with Annual NLCD in 2024 — a new ensemble deep-learning methodology with a revised class system (21 → 16 classes). MRLC states explicitly that legacy NLCD and Annual NLCD are "not directly comparable due to differences in methodologies, inputs, and ancillary data." Migrating would require re-validating every lucode mapping and regenerating all baselines.

**Alternatives considered.** Annual NLCD adoption — deferred until NatCap's published SA / MN project data migrates.

**Consequences.** The prototype stays compatible with InVEST's UFR / UCM / UNA sample bundles. Open question for NatCap: has SA Urban Agriculture data migrated to Annual NLCD or remained on legacy?

**Revisit if.** NatCap migrates their published per-city project data to Annual NLCD, OR the prototype begins ingesting non-NatCap LULC sources that ship in the Annual schema.

**Code touchpoints.** Per-city LULC paths in `config.py` (`lulc_file`, `cooling_lulc_file`); the per-city CN / UCM / UNA biophysical tables.

### 3.2 LULC raster per-city framework

**Decision.** Per-city LULC representation:

| City (role) | Path | CRS | Encoding |
|---|---|---|---|
| Minneapolis — cooling & scenario | `data/cooling/land_use_2021.tif` | EPSG:26915 | NLCD only; byte-identical to the InVEST UNA sample |
| Minneapolis — flood / CN | `data/flood/LULC_NLCD_2021_MN.tif` | EPSG:26915 | NLCD only; same AOI, distinct file (InVEST UFR sample) |
| Minneapolis Full (dormant) | `data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif` | EPSG:5070 | NLCD only |
| San Antonio | `data/sa/flood/land_use_compound_sa.tif` | EPSG:5070 | NatCap compound NLCD × NLUD × tree-canopy |

**Why.** MN inherits the InVEST sample bundle's separate-file structure (UFR vs UCM/UNA shipped distinct LULC files at the same AOI); the prototype reads each role's file directly rather than synthesizing one. SA adopts NatCap's compound LULC framework (see §3.3 below).

**Alternatives considered.** Consolidate MN's two rasters into one — would invalidate the byte-identical-to-InVEST claim used for the UNA validation harness.

**Consequences.** MN downtown carries the InVEST sample's native EPSG:26915 (UTM 15N); MN Full and SA use NLCD's native EPSG:5070 (Conus Albers). Detail (file paths, MD5s) → `DATA_INVENTORY.md`.

**Revisit if.** NatCap supplies a curated MN compound LULC (SA-style).

**Code touchpoints.** `config.py` per-city `lulc_file` / `cooling_lulc_file` / `compound_lulc_file`; `load_data` (path-parameterized).

### 3.3 SA compound LULC adoption — CRS choice + encoding

**Decision.** SA adopts NatCap's compound NLCD × NLUD × tree-canopy LULC framework, reprojected from EPSG:3857 to EPSG:5070 (nearest-neighbor resampling at 30 m). The compound raster is the canonical SA LULC; the per-model compound biophysical tables (UCM / UNA / Carbon) consume it directly.

**Why.** NatCap's compound framework captures climate- and policy-relevant per-pixel variation (tree-canopy bin, NLUD residential vs commercial vs managed-natural) that the per-NLCD framework flattens. Adopting it brings the prototype onto the same per-pixel inputs NatCap's SA work uses. EPSG:5070 (equal-area) is the prototype's canonical SA CRS; NatCap's choice of EPSG:3857 (Web Mercator) was likely operational. Area-based metrics (acres converted, runoff volume) require equal-area preservation.

**Alternatives considered.**
- Keep SA on per-NLCD biophysical tables — flattens canopy- and NLUD-driven per-pixel variation; understates baseline `mean_hm` by ~37 % and overstates cooling intervention dollar leverage (Brief 28b finding).
- Reproject EPSG:3857 → EPSG:5070 via bilinear / cubic — would alter integer lucodes; nearest-neighbor preserves the categorical encoding.
- Adopt 3857 as the runtime CRS — heavily distorts area at non-equatorial latitudes; unsuitable for area-based math.
- Reproject every SA raster to 3857 (soil, ET, population, buildings, roads) — would also distort, and would require regenerating 5 existing rasters.

**Consequences.** SA baseline metric shifts on adoption (one-time, per-model — magnitudes in `../archive/HISTORY.md` schema log 22→23 and 23→24): `baseline_hm` +37 %, `nature_access_pct` +5 pp, carbon stock framing replaces annual-flow proxy. ~1 % nodata at clipped extent edges (NatCap's raster extended ~6′ farther north / ~3′ south / ~2′ west; clipping to the prototype's Bexar bbox loses those edge pixels). The trade is accepted because the analysis is constrained to roughly Bexar County regardless.

**Revisit if.** NatCap publishes a 5070-native compound LULC (would obviate the reprojection step).

**Code touchpoints.** `data/sa/flood/land_use_compound_sa.tif`; `_load_city_runtime_state` reads `cooling_lulc_compound` and exposes it to compound-keyed consumers (see §3.4); `download_sa_data.py` / pipeline scripts for regeneration.

### 3.4 Per-model compound biophysical tables (SA)

**Decision.** SA's UCM / UNA / Carbon biophysical tables are NatCap's compound NLCD × NLUD × tree-canopy tables (1,984 rows each, keyed on compound `lucode` 0–1983), indexed directly off the compound LULC raster. SA's flood CN table uses NatCap's NLCD × tree-canopy 3-tier encoding (`biophys_floodmitig_sa.csv`).

**Why.** Adopting NatCap's per-model tables completes the SA-side per-city framing (§2.1). The compound tables encode tree-canopy and NLUD effects per-pixel — they're the data, not a user-tuned override layer.

**Alternatives considered.**
- Per-NLCD tables tuned for Köppen-BSh climate (the prior SA UCM approach) — a workaround for not having compound LULC; retired (see HISTORY).
- Borrow MN's per-NLCD UNA table for SA — the prior SA UNA approach; retired in favor of the compound table.

**Consequences.** UCM / UNA / Carbon all index `cooling_lulc_compound` directly via per-city NumPy lookup arrays on `CityState` (`shade_arr` / `kc_arr` / `albedo_arr` / `urban_nature_arr` / `c_above_arr` / `c_below_arr` / `c_soil_arr` / `c_dead_arr`). The dict-iteration pattern (`for lucode in URBAN_NATURE_PROPORTION.items(): ...`) was retired because at SA's 1,984 compound lucodes it would run 1,984 raster-wide boolean comparisons per call; the vectorized `arr[scenario_lulc_X]` indexed read is the new normal. Per-model schema fields (`scenario_lulc_ucm`, `scenario_lulc_una`, `scenario_lulc_carbon`) carry the right lucode-space view per city (compound for SA, NLCD for MN).

**Revisit if.** NatCap publishes a refreshed compound table — re-import is a single file swap.

**Code touchpoints.** `config.py` per-city `ucm_table_file` / `una_table_file` / `carbon_table_file`; `_load_city_runtime_state` per-city lookup-array population; the per-model `_pure` variants (e.g. `_compute_carbon_four_pool_pure`) for explicit-dependency injection.

### 3.5 Compound→NLCD reduction routing (transitional, for non-compound-keyed consumers)

**Decision.** SA's compound LULC is reduced to NLCD via `lulc_crosswalk.csv` once at load time, producing the NLCD-view raster every non-compound-keyed consumer reads as `scenario_lulc`. UCM / UNA / Carbon have moved off this reduction (they read the compound view directly per §3.4); the reduction routing survives for the spatial-map render, the flood path (via the NLCD × tree reduction), and any future non-compound-keyed consumer.

**Why.** Adopting compound-keyed tables happened model-by-model (Briefs 28b / 29 / 30). The reduction is the bridge that keeps everything else working during the transition.

**Alternatives considered.** Rewrite every consumer to compound-aware in one pass — large blast radius for a long-tail of tools (spatial map, lookup-table strip sites, surrogate training CSV columns). Incremental migration is safer.

**Consequences.** `COMPOUND_TO_NLCD` (compound → NLCD) and `COMPOUND_TO_NLCD_TREE` (compound → NLCD × tree-tier) are built once at load time as NumPy lookup arrays. The compound-nodata sentinel (-1 in NatCap's raster) is rewritten to the prototype's module-wide `NODATA = -128` so existing `(scenario_lulc != NODATA)` masks downstream continue to work. 97.91 % pixel-wise agreement between the compound-reduced NLCD view and the prior `land_use_2021_sa.tif`.

**Revisit if.** A new consumer arrives that needs to index `scenario_lulc` and would benefit from being compound-keyed.

**Code touchpoints.** `load_lulc_crosswalk` + `reduce_compound_to_nlcd` / `reduce_compound_to_nlcd_tree` in `app.py`; `COMPOUND_TO_NLCD` / `COMPOUND_TO_NLCD_TREE` module-level arrays.

---

## 4. Scenario generation and conversion logic

### 4.1 Conversion-target mapping (SA) — preserve NLUD + tree-canopy, change NLCD only

**Decision.** When a user converts a developed SA pixel "to food forest" / "to green infrastructure" / "to high-density development," the post-conversion compound lucode preserves the source pixel's NLUD and tree-canopy bins and changes only the NLCD signal. The compound code is looked up in `lulc_crosswalk.csv` via `(NLCD=target, NLUD=source_NLUD, tree=source_tree)`, with `is_realistic_to_create=yes` rows preferred and ascending `lucode` as a deterministic tiebreaker.

**Why.** Least presumptuous — the conversion models the land cover change without claiming knowledge of how the land use or canopy state changes alongside it.

**Alternatives considered.**
- Change all three bins (NLCD + NLUD + tree-canopy) — presumes too much; e.g. converting a low-canopy parking lot to "food forest" shouldn't claim high canopy on the same pixel.
- Use the `is_realistic_to_paint` column for the realism filter — empty across the entire crosswalk (all NaN); `is_realistic_to_create` is the only available signal.

**Consequences.** When the source pixel's (NLUD, tree-canopy) tuple has no row for the target NLCD, the conversion falls back to a documented default lucode (§4.2). The instrumentation (§4.3) reports how often this fallback fires.

**Revisit if.** NatCap publishes paint-realism flags, or scenario fidelity requires non-default conversion logic.

**Code touchpoints.** `load_lulc_crosswalk` builds the per-target `COMPOUND_AFTER_FF` / `COMPOUND_AFTER_GI` / `COMPOUND_AFTER_HD` lookup arrays; consumed inside `evaluate_scenario`'s SA branch.

### 4.2 `DEFAULT_*_LUCODE` fallback choices

**Decision.** Fallback compound lucodes when the source pixel's (NLUD, tree-canopy) has no matching row in the crosswalk for the target NLCD:

| Fallback | Lucode | Compound class | Why this lucode |
|---|---:|---|---|
| `DEFAULT_FF_LUCODE` | 1310 | Deciduous Forest × Timber × medium canopy | Highest-frequency `is_realistic_to_create=yes` row for NLCD-41 (36,939) |
| `DEFAULT_GI_LUCODE` | 122 | Woody Wetlands × Wetland × medium canopy | Highest-frequency for NLCD-90 (50,384) |
| `DEFAULT_HD_LUCODE` | 341 | Developed High Intensity × Residential × low canopy | Highest-frequency for NLCD-24 (53,389) |

**Why.** Filter the crosswalk for `is_realistic_to_create=yes` (the only realism flag with non-empty rows), then prefer the highest-`frequency` row — the "typical" representative of the target land cover as it appears in NatCap's SA raster.

**Alternatives considered.**
- Pick by deterministic lucode ordering (smallest valid lucode) — produces less-representative archetypes.
- Pick by tree-canopy maximization (highest-canopy row for the target NLCD) — risks claiming unrealistic post-conversion canopy.

**Consequences.** As of Brief B's instrumentation (§4.3), the fallback fires for **0 % of converted pixels** across every SA scenario and placement strategy — NatCap's crosswalk has comprehensive coverage of every (NLUD × tree) tuple in SA's developed-land pool. The "is the default principled?" methodology question is therefore academic for the current SA pipeline — the defaults' choice doesn't affect any current scenario output. The instrumentation surfaces the answer explicitly and falsifiably so any future change that breaks the coverage assumption surfaces immediately.

**Revisit if.** NatCap ships an updated SA LULC or crosswalk that breaks the 100 %-coverage property.

**Code touchpoints.** Module-level `DEFAULT_FF_LUCODE` / `DEFAULT_GI_LUCODE` / `DEFAULT_HD_LUCODE` in `app.py`; consumed by `load_lulc_crosswalk`.

### 4.3 Conversion-fallback instrumentation (Brief B)

**Decision.** Each per-target conversion site emits a fallback-pixel count: `ff_fellback_pixels`, `gi_fellback_pixels`, `hd_fellback_pixels` (always emitted as scalar return-dict keys — 0 for MN, no compound conversion path). The dashboard's Conversion-fidelity panel inside the Assumptions and limitations expander renders `fellback_pixels / n_converted` as a percentage per target (SA only, gated on `_COMPOUND_CONVERSION_ACTIVE`).

**Why.** Before instrumentation there was no visibility into how often the §4.2 fallback fires. That matters because the methodology question "is the default principled?" is academic at <5 % and substantive at >30 %.

**Alternatives considered.**
- A nested `conversion_diagnostics: dict` — `verify_baselines.py:_snapshot_from_results` handles scalars but `dict` falls to a `WARN: skipping field` branch; would also need flattening for CSV serialization. Flat scalars are simpler.
- Pre-computed fraction keys — recomputed at display time instead; one fewer field in the schema.
- Make the diagnostic a surrogate target — pure metadata about the conversion mechanism, not an outcome metric; `REQUIRED_TARGET_COLUMNS` correctly excludes it.

**Consequences.** Schema bump 25 → 26. Empirically all three counters are 0/0/0 across every SA scenario; the instrumentation is dormant in normal operation but lights up immediately on any future regression.

**Revisit if.** The dashboard panel reports a non-zero fallback rate.

**Code touchpoints.** `load_lulc_crosswalk` builds `COMPOUND_AFTER_*_WAS_DEFAULT` boolean arrays; `evaluate_scenario`'s SA branch sums them per-target; the dashboard panel inside the Assumptions and limitations expander.

### 4.4 Lookup-overlay safety contract (this section owns the rationale)

**Decision.** When the lookup table is in use (High resolution mode, random placement strategy), the lookup hit short-circuits ~17 of 27 return-dict fields; the remaining **10 fields** are recomputed live on every slider interaction via a `_fresh = evaluate_scenario(...)` call. The field list and live-overwrite mechanics are documented in ARCHITECTURE §5; this section owns the *why*.

**Why.** Two distinct correctness concerns drive the pattern, and both are real but for *different* reasons:

1. **Schema-versioning protects fields that are pure functions of the lookup key.** `compute_lookup_table` is cached with `schema_version=SCENARIO_SCHEMA_VERSION` as a cache-key parameter. Any change to `evaluate_scenario`'s return-dict shape or semantics — when properly accompanied by a schema bump — invalidates the entire lookup, which is then rebuilt from scratch using the current `evaluate_scenario`. Every field loaded from a lookup row is therefore guaranteed schema-current. No defensive overwrite needed for surrogate-target fields (`flood_reduction`, `mean_hm`, `runoff_acre_feet`, `nature_access_pct`) or any field that's a pure function of `(pct, gi, ff, seed)`.

2. **Per-rerun state dependencies need explicit overwrites.** Cost-slider state (`cost_gi` / `cost_ff` / `cost_hd`), MN carbon-rate sliders (`carbon_rate_ff` / `carbon_rate_gi`), and rasters that are too expensive to cache per slider position (`scenario_lulc`, `scenario_lulc_ucm`) are not a function of the lookup key alone. They must be recomputed live every rerun.

The two categories sit on opposite sides of a single principle: *if a field can drift from the lookup key, it's overwritten; if it can't, the schema-versioned cache is the safety mechanism.* This is the "schema-vs-slider-sensitivity gap" the live-overlay closes.

**Alternatives considered.**
- Overwrite every field defensively — silently regresses to "Layer 1 only" performance because every overwrite re-runs the per-pixel pipeline. Defeats the lookup's purpose.
- Cache cost / carbon rates by their values too — explodes the cache-key combinatorics and re-introduces invalidation timing risk on slider edits.
- Strip `scenario_lulc` / `scenario_lulc_ucm` and forbid downstream raster reads — the spatial map needs the raster; per-tract aggregation in `compute_per_tract_summary` needs it.

**Consequences.** Contract for future devs (the part this section enforces):
- **Do NOT** add new defensive overwrites for surrogate-target fields or for any field that's a pure function of `(pct, gi, ff, seed)`. Bump `SCENARIO_SCHEMA_VERSION` instead.
- **DO** add overwrites for fields that depend on per-rerun state (sliders, user toggles).
- The slider-branch leading comment in `app.py` summarizes this contract. Keep it in sync if the overwrite list changes.

The lookup table is only built in High resolution mode (opt-in checkbox), so the surface area where this contract matters is narrow by design.

**Revisit if.** A new return-dict field is added that depends on per-rerun state (must be added to the overwrite list), or `SCENARIO_SCHEMA_VERSION` discipline lapses (must be re-enforced).

**Code touchpoints.** `compute_lookup_table` and the lookup-branch in `app.py`'s main `evaluate_scenario` call site; the leading inline comment on the lookup branch; `SCENARIO_SCHEMA_VERSION` constant; ARCHITECTURE §5 holds the per-field list and live-overwrite mechanics.

---

## 5. Placement strategy

### 5.1 Three-layer non-convertible mask

**Decision.** Conversions only land where the candidate area allows — three categories of land are excluded by construction:

1. **Buildings** — building footprints rasterized into `buildings_raster` (InVEST UFR sample shapefile for MN downtown; comprehensive OSM footprints for SA and MN Full).
2. **Roads** — OSM road footprints (Geofabrik extracts) rasterized and unioned into the same mask.
3. **Existing nature** — the conversion candidate pool is built only from developed LULC classes (NLCD 21–24); nature pixels (forest, water, wetland, etc.) are never candidates.

**Why.** Modest spatial-fidelity improvement: constrain *where* conversions can physically land using grounded data, without attempting to predict *where they would* land (the domain of the deferred land-use simulation models — see §11). Bounded by design.

**Alternatives considered.**
- Wallpaper / uniform tiling across the AOI — NatCap's documents name this alongside the three-layer mask as a "simpler approach," but the term has no standard land-use literature definition; interpretation uncertain. One-line pointer in §11 + a NATCAP_COLLABORATION ask.
- No mask at all — allows conversions on buildings/roads, which is physically implausible.
- PLUS / CLUE / LCM — answer a different question (status-quo projection, not planner intervention); §11.

**Consequences.** `convertible_pixels = developed_pixels − (buildings ∪ roads)`. After the comprehensive OSM building footprints landed (MN), the convertible pool shrank ~21 % (33,357 → 26,372 pixels). MN's separate `buildings_file` (the InVEST UFR sample, typed) drives the per-typed-building dollar metrics; `mask_buildings_file` (the OSM footprints, untyped) drives the placement mask. The split-config architecture is documented under DATA_INVENTORY → buildings.

**Revisit if.** A future workstream wants to introduce a probabilistic placement-prediction model (status-quo projection) — at which point the simulation models in §11 become candidates.

**Code touchpoints.** `_load_city_runtime_state` Phase 9 (road rasterization + union into buildings mask), Phase 11 (`convertible_pixels` build); `BUILDINGS_RASTER` / `BUILDINGS_TYPE_RASTER`; `../../CLAUDE.md` "OSM road exclusion" for the Option B class filter rationale.

### 5.2 Suitability formulas — canonical InVEST quantities

**Decision.** Each of the four weighted placement strategies (`undersupply-focused`, `flood-focused`, `cooling-focused`, `balanced`) uses a suitability formula derived from a canonical InVEST quantity where one exists. The decision principle: where a canonical InVEST quantity exists, use it.

| Strategy | Suitability formula |
|---|---|
| `undersupply-focused` | `max(0, urban_nature_demand − urban_nature_supply_percapita)` — per-capita supply deficit per InVEST UNA's `urban_nature_balance_percapita.tif` |
| `flood-focused` | Per-pixel runoff `Q_{p,i}` at the per-city design storm — matches InVEST UFR's `Q_mm.tif` |
| `cooling-focused` | `(1 − baseline_HMI) × distance_to_buildings_weight` — canonical HMI + real Euclidean distance-to-buildings via `scipy.ndimage.distance_transform_edt(BUILDINGS_RASTER)` |
| `balanced` | Equal-weighted normalized combination of the three above |

**Why.** The user's explicit working principle: *"I want to be as closely aligned to natcap as possible. even if it takes more time and results in undoing previous work."* Where InVEST emits a canonical quantity that already captures the priority the strategy is trying to address, use it directly.

**Alternatives considered (and why retired).**
- `undersupply-focused` formerly used `population × (1 − access_score + 0.01)` — aggregate need (a 1,000-undersupplied pixel weighted 10× a 100-undersupplied pixel). InVEST UNA's canonical output is per-capita; adopting the per-capita form aligned with InVEST and has a real ethical character (every resident's deficit counts equally).
- `flood-focused` formerly used per-pixel CN as the weight — monotone with runoff but the wrong shape. At low CN, `Q ≈ 0` regardless of CN, but the old formula assigned non-zero weight there. Canonical `Q` concentrates sharper on high-runoff pixels.
- `cooling-focused` formerly used `(1 − baseline_CC) × NLCD_intensity_proxy` — used the bare CC sub-component when the canonical HMI (MAE 0 vs InVEST) was available, and a three-value NLCD-class proxy for building proximity. Adopting HMI + real distance-to-buildings aligns with the validated metric and grounded data.
- An `equity-focused` name was retired alongside the formula change — InVEST UNA reserves "equity" for demographic-group stratification (age, income, race). Generic undersupply borrowed NatCap's vocabulary; renamed to `undersupply-focused` to free up the word.
- The `+ 0.01` / `+ 0.1` floors on the old formulas are gone. Pixels with no deficit / no proximity get true zero weight. The saturation fallback in `_select_pixels_for_conversion` handles the edge case where the strategy doesn't have enough non-zero pixels for the requested conversion count.

**Consequences.** The methodology shift is real, not cosmetic — the new formulas can produce materially different scenario outputs than the old ones. The `_compute_suitability_weights` symbol is the single home for every focused-strategy formula.

**Revisit if.** A new canonical InVEST output becomes relevant to a placement priority (e.g. an InVEST-published heat-vulnerability score would supersede the current development-intensity proxy in the `cooling-focused` weight).

**Code touchpoints.** `_compute_suitability_weights`, `_select_pixels_for_conversion` in `app.py`; the legacy `use_heat_priority=True` kwarg in `evaluate_scenario` internally translates to `placement_strategy='cooling-focused'`.

---

## 6. Model evaluation design

### 6.1 Why numpy, not canonical `natcap.invest` at runtime

**Decision.** The prototype implements UCM / UFR / UNA / UMH / Carbon as numpy ports rather than calling `natcap.invest.*.execute()` directly.

**Why.**
- **Latency.** Canonical InVEST is built on `taskgraph`, a desktop pipeline framework that reads inputs from disk, executes in a worker process, and writes outputs back to disk. For Bexar County extent at 30 m (~3.4 M pixels), a single `execute()` call takes minutes. Streamlit's rerun-on-interaction model would re-trigger the pipeline on every slider move — incompatible with the prototype's three-layer caching architecture (ARCHITECTURE §5), which serves slider responses in milliseconds.
- **No `execute_from_arrays()` API.** The canonical API takes file paths in its args dict. There is no in-memory variant. Working around this would require writing temporary `.tif` files on every slider move — defeats the latency goal and surfaces disk-write race conditions under fast slider drags.
- **Validation, not replacement.** The numpy implementations are validated against canonical `natcap.invest` runs in the `compare_*_invest.py` harnesses (UCM, UNA, Carbon in anaconda base; UMH in the isolated env per ../dev/CONTRIBUTING.md). The prototype's runtime is fast; its correctness is anchored to canonical InVEST through these offline validation runs. The export bundle (§9) is the bridge for users who want canonical InVEST results — they run `natcap.invest.*.execute()` against the bundle's inputs.

**Alternatives considered.**
- Wrap `natcap.invest.execute()` calls — fails on latency (minutes per call) and disk-write semantics.
- Adopt `pygeoprocessing` primitives but not the model wrappers — viable for some helpers (the prototype uses `pygeoprocessing.convolve_2d`'s edge-correction semantics via `_convolve_edge_corrected`); not viable as a whole-model replacement because the model-level orchestration (CC vs HMI, 2SFCA two-step, UMH RR aggregation) lives inside `execute()`.
- Run canonical InVEST in a background worker and stream results — out of scope for a Streamlit prototype, and would require disk artifact management between the worker and the UI process.

**Consequences.** Every model has both a fast in-app path (numpy port) and a validation path (canonical InVEST). Two infrastructure choices follow:
- The `_pure` / wrapper split for compute helpers (e.g. `_compute_hmi_raster` reads module aliases; `_compute_hmi_raster_pure` takes deps explicitly) — the loader uses the pure variant because the module aliases aren't rebound yet at loader-call time.
- The validation harness pattern (per ../dev/CONTRIBUTING.md "Canonical-InVEST validation environments") — two-env decoupled when UMH 3.19 needs Python ≥ 3.10 and the app `.venv` is 3.9.

**Revisit if.** NatCap adds an `execute_from_arrays` API to `natcap.invest` (would warrant re-evaluating wrap-vs-port for sub-second-friendly cases) — none on their roadmap as of 2026-05.

**Code touchpoints.** `_compute_hmi_raster` / `_compute_hmi_raster_pure` (UCM); `_una_supply_percapita` / `_una_supply_percapita_pure` (UNA); `_compute_carbon_four_pool` / `_compute_carbon_four_pool_pure` (Carbon); `_umh_neighborhood_exposure` (UMH); `compute_lookup_table`. Validation comparators in `validation/compare_*_invest.py`.

### 6.2 UCM canonical HMI implementation

**Decision.** Per-pixel `CC = 0.6·shade + 0.2·albedo + 0.2·ETI`; `HMI = max(CC_local, CC_park)` where `CC_park` is the exponentially distance-weighted CC sourced from green areas, applied only where a pixel has ≥ 2 ha of green within `d_cool = 450 m`. Convolutions use `scipy.signal.fftconvolve` with an InVEST-canonical edge correction (`_convolve_edge_corrected`, reproducing `pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True)`).

**Why.** Matches the canonical InVEST UCM formula exactly. The 2-ha-within-450-m gate distinguishes "parks" from incidental greenery. The edge correction matches InVEST's treatment of valid-pixel boundaries.

**Alternatives considered.**
- A simpler CC-only metric (no park sourcing) — drops the canonical InVEST `max(CC_local, CC_park)` behavior; would miss the park-cooling effect on adjacent developed pixels.
- Per-building T_air aggregation over the 600 m air-blending radius for the dollar metric — would match canonical InVEST UCM's `T_air` output exactly. Deferred (the prototype computes the dollar metric per pixel rather than per-building). The remaining UCM divergence; affects only the dollar Cooling Energy Savings metric, not Temperature Change.

**Consequences.** Validated against `natcap.invest.urban_cooling_model.execute()` at MAE = 0.0000 / r = 1.0000 (`validation/compare_ucm_invest.py`).

**Revisit if.** The per-building Cooling Energy Savings aggregation is requested explicitly — would require wiring the building polygon raster into the dollar metric (it already feeds the per-typed-pixel rate denominator).

**Code touchpoints.** `_compute_hmi_raster` / `_compute_hmi_raster_pure`, `_compute_cc_raw_pure`, `_compute_cc_park_raster`, `_compute_green_area_sum`; `_convolve_edge_corrected`; `compute_cooling_energy_savings`.

### 6.3 UMH validation against canonical InVEST 3.19.0

**Decision.** The Urban Mental Health implementation uses the canonical InVEST UMH 3.19.0 buffer-mean neighborhood-exposure (NE) kernel — an edge-corrected mean of NDVI over a flat binary disk of radius `search_radius / pixel_size` (apothem 10 px at 30 m / 300 m, 317-pixel disk). RR per-pixel = `exp(ln(RR₀.₁) × 10 × ΔNE)`; preventable cases = `(1 − RR) × baseline_prevalence × population`, summed depression + anxiety.

**Why.** The buffer-mean is what InVEST UMH 3.19 emits as `ndvi_*_buffer_mean` and what its `kernel` output documents. Per-pixel parity against canonical is the validation credibility anchor (alongside UCM/UNA's MAE-0 results).

**Alternatives considered.**
- Keep the Gaussian (`σ = search_radius / pixel`) the prototype originally used — defensible smoothing; aggregate preventable-cases total agreed to ~1.3–1.5 %, but per-pixel `r ≈ 0.95–0.98` rather than the MAE-0 of UCM/UNA. Aggregate-only validation undersells the per-pixel claim. Retired in favor of the buffer-mean (Brief B).
- Document the Gaussian-vs-canonical divergence rather than fix it (the earlier Option A) — taken initially; superseded by Brief B's kernel switch when the buffer-mean was confirmed identical.

**Consequences.** Per-pixel parity vs canonical: **MN MAE ≈ 1e-9, r = 1.000000** (both outcomes); **SA MAE ≈ 0** when the app kernel is fed canonical's own aligned input (the harness's 0.14 % residual on SA's 1713 × 1984 grid is large-grid feeding-alignment + FFT noise in the comparison, not a metric divergence). MH card confidence Medium → High. Schema bump 26 → 27.

Two divergences remain documented rather than hidden:
- The prototype uses a uniform national CDC prevalence vs InVEST's per-admin `risk_rate` vector — no per-tract MH-prevalence data available for MN/SA; the matched-input MAE = the default-input MAE.
- Validation uses the prototype's synthetic NDVI proxy — validates the algorithm, not the NDVI source (satellite-derived NDVI is a distinct future workstream).

**Why a land-cover-derived greenness proxy, not satellite NDVI.** The tool's purpose is to compute a scenario *response* — what changes when developed land is reconverted. A static observed-NDVI raster cannot respond to a hypothetical conversion (the satellite never saw the food forest that doesn't exist yet), so greenness must be a function of the scenario's land-cover map for UMH to move at all. **The cost:** the per-class proxy flattens within-class greenness variation and is not a measurement of true baseline greenness, so absolute UMH case counts are indicative, not survey-grade — **the scenario response (Δ vs baseline) is the signal**, not the absolute level. This is the sharpest instance of a general property: UMH greenness, the UCM cooling inputs, and the UNA greenspace definition are all derived from the scenario land-cover map (not live remote sensing) precisely so they respond to scenarios; UMH stands out only because "NDVI" conventionally connotes satellite data. The validated UMH badge therefore means "method matches InVEST UMH where tested," not "output matches observed reality."

**Revisit if.** Per-tract MH-prevalence data becomes available (would unlock the per-admin `risk_rate` path), or AlphaEarth/another satellite NDVI source replaces the synthetic proxy.

**Code touchpoints.** `_umh_neighborhood_exposure` in `app.py`; `_convolve_edge_corrected`; `calculate_mental_health_impact`; `_BASELINE_NE_RASTER`. Validation: `validation/compare_umh_invest.py` (two-env harness — see ../dev/CONTRIBUTING.md). NATCAP_ALIGNMENT.md UMH row holds the parity numbers.

> **Anchor preserved:** `## UMH validation against canonical InVEST 3.19.0` (linked from `../dev/CONTRIBUTING.md` line 70).

### 6.4 SA Carbon four-pool stock framework

**Decision.** SA Carbon uses NatCap's canonical InVEST four-pool stock framework (`c_above + c_below + c_soil + c_dead`, in t C/ha per compound lucode). For a scenario LULC, stock delta vs baseline = `Σ_pixels((scenario_total − baseline_total) × pixel_area_ha) × (44/12)` to convert t C → t CO2. This is a one-time stock change when the land use changes, NOT an annual sequestration rate.

**Why.** NatCap's 2023 Vibrant Land report (Guerry et al.) describes the SA Carbon methodology as InVEST's four-pool stock model multiplied by Social Cost of Carbon — no annual amortization, no InVEST NPV. The prototype adopts the same framing exactly. The four-pool framework captures direction (positive when gaining nature, negative when losing it), magnitude (proportional to per-pool deltas), and per-pixel-baseline sensitivity (the source pixel's actual baseline LULC matters).

**Alternatives considered.**
- Stay on the per-conversion-type annual-flow proxy (`n_pixels × CARBON_SEQ_RATES[target] × acres-per-pixel`) — the prior SA approach; conflated stock and flow, applied the same rate regardless of baseline (converting forest→forest gave the same number as parking lot→forest), and bottomed at $0 on high-density conversions hiding the carbon-loss case.
- Amortize the stock delta to annual flow for cross-metric comparability — Vibrant Land doesn't; it juxtaposes annual cooling savings with one-time carbon value without forcing a shared frame. Adopting amortization would diverge from NatCap's published presentation.
- Use InVEST NPV valuation — Vibrant Land doesn't; the prototype follows.

**Consequences.**
- **Methodology matches, SC-CO2 vintage differs.** Vibrant Land uses IWG 2021's $53/t CO2 @ 3 % discount; the prototype keeps `EPA_SOCIAL_COST_CARBON = $190/t` (EPA 2023 final rule, 2 % discount). Same US-government standard lineage, different vintage. SA's dollar carbon value comes out ~3.6× Vibrant Land's reported figure on equivalent stock magnitudes — methodology aligns, dollar magnitudes don't, by intentional vintage choice. NATCAP_COLLABORATION.md carries this as a confirmatory ask.
- **Cross-metric temporal-framing comparability.** Cooling, flood, and MH dollar metrics are annual flows. SA Carbon is one-time stock. The two appear side-by-side; the divergence is **surfaced via metric labels rather than hidden via amortization** (SA: "Carbon Storage Value"; MN: "Avoided Carbon Cost (per year)"). Matches Vibrant Land's presentation.
- **MN Carbon unchanged.** Per-cover annual rate proxy (`CARBON_SEQ_RATES`) — MN-side four-pool data of the same shape isn't sourced; the per-city framework principle (§2.1) keeps each city on its own published methodology.
- **Field rename.** `carbon_tons_co2_yr` → `carbon_tons_co2` (unified return-dict key; per-city semantics surfaced via labels). `_CARBON_IS_STOCK = c_above_arr is not None` drives label branching.

**Revisit if.** NatCap publishes MN four-pool data (would warrant migrating MN), or refreshes Vibrant Land using EPA 2023 SC-CO2 (would resolve the vintage gap).

**Code touchpoints.** `_compute_carbon_four_pool` / `_compute_carbon_four_pool_pure` (SA); `_compute_carbon` (MN); `EPA_SOCIAL_COST_CARBON` constant; `_CARBON_IS_STOCK` flag; `carbon_value_usd` return field.

### 6.5 SA Flood Damage — embrace $0 (Path C)

**Decision.** SA's `avoided_flood_damage_usd` field remains $0 because NatCap's Vibrant Land report explicitly did not enable `infrastructure_damage_loss_table_path` for SA — they reported flood mitigation as **percent reduction in flood volume**, not as a monetized dollar figure. The dashboard surfaces this directly: SA shows no damage-avoided dollar figure — its Economic flood card renders **"Flood Damage Avoided — n/a"** — while MN renders **"Flood Damage Avoided"** in dollars. (Originally SA showed a **"Flood Volume Reduction (% vs baseline)"** card; the Flood Index honesty pass — Relay 24 — removed it because it presented the unitless CN index `100 − mean_CN` as a percent of flood volume, which it is not. SA's hydrologic signal is the Flood Index + Runoff Volume cards.) Presentation-layer choice only; no model, schema, or baseline changes.

**Why.** Honest with NatCap's own methodology choice. InVEST UFRM's own caveat is that the model doesn't produce inundation maps and therefore can't confirm built-infrastructure exposure; Vibrant Land's flood-volume-reduction framing inherits that limitation rather than monetizing through it.

**Alternatives considered.**
- **Path A: Source SA-specific damage values independently** (FEMA NFHL / Hazus depth-damage curves / USACE economic studies). Most rigorous, weeks of data-sourcing work, and makes parameter choices NatCap didn't endorse — potentially diverges from any future NatCap SA work.
- **Path B: Borrow MN's damage table for SA, label as placeholder.** Lowest-effort, but re-introduces the per-NLCD-borrowed-from-MN pattern Brief 29 just retired for UNA. MN values may not generalize to SA (different urban form, floodplain hydrology, property values, precipitation regime). Numerically arbitrary; methodologically uncomfortable.
- **Path D: Per-acre-foot proxy decoupled from NLCD class.** Replace per-NLCD damage values with a per-AF damage constant from flood-risk literature. Gives SA a dollar number without inventing per-NLCD values, but is a different framing from MN's per-NLCD approach. The per-AF constant has substantial uncertainty (FEMA Hazus range ≈ $2k–$50k/AF). May need NatCap sign-off.

Path C chosen because it aligns with NatCap's documented stance and is the most defensible position to any audience.

**Consequences.** Underlying `avoided_flood_damage_usd` field unchanged ($0 for SA, real value for MN) — surrogate training, lookup-table schema, CSVs, and the 40/40 baseline regression are all unaffected (no schema bump). The dashboard card and the comparison-table row are city-conditional. Same shape as Brief 30's per-city Carbon framing — each city gets the framing NatCap canonical material uses for that city.

**Revisit if.** A future NatCap conversation surfaces a preference for SA-specific damage values (re-pointing `damage_table_file` to a curated CSV reactivates the dollar code path; no schema work). Or Vibrant Land migrates to monetized framing.

**Code touchpoints.** `config.py` SA `damage_table_file = None`; the SA-branch dashboard card; the comparison-table city-conditional row.

### 6.6 Children's nature access — share-only reweight

**Decision.** Reweight only the *access share* of the UNA metric by under-18 population. Do not alter the 2SFCA supply/demand calculation; do not touch UMH.

**Why.** The 2SFCA supply model and UMH's burden-of-illness rate + effect sizes are total-/adult-calibrated. Reweighting the supply convolution or the UMH dose-response by child population would manufacture a number with no calibrated basis. The access *share* is different — it's a clean reweight of an already-valid per-pixel adequacy classification (the same adequate mask the adult metric uses), so "what fraction of *children* live on adequately-served pixels" is well-defined without re-deriving any model. Honest by construction: a real question answered only with quantities the engine already computes validly.

**Source.** Census 2020 PL 94-171 block-level under-18 (`P1_001N − P3_001N`) — chosen over ACS B09001 precisely so the child and total rasters are the same product (no cross-source drift). Per-city extent share anchored in verify_baselines (MN 20.2 %, SA 24.5 %).

**Code touchpoints.** `_invest_una_pct_pop_supply_ge_demand` (optional `child_pop_count_raster` kwarg → 6-tuple return); `calculate_nature_access` (mirrors the extension); `evaluate_scenario` populates `children_nature_access_pct` + `children_with_nature_access` (citywide + region_local); `child_pop_count_raster` on `CityState`; `_load_city_runtime_state` Phase 2b; `scripts/data/download_census_pop*.py`; child-pop staleness cell in `verify_baselines.py` with halve-the-raster meta-test.

### 6.7 Nature Access at Schools — destination-based readout (private included)

**Decision.** Add a destination-based UNA metric that point-samples the existing 2SFCA `adequate` mask at K-12 school locations. Same supply pipeline, same per-city `UNA_DEMAND_M2_PER_CAPITA` threshold, same valid-LULC restriction as the residential Nature Access metric — only the consumer (point sample vs population sum) is different. Include **private** schools alongside public + charter, surfaced explicitly on the metric card and in `REFERENCE §6`.

**Why.** Residential metrics answer "do people live where supply meets demand"; the school readout answers "do children spend the school day where supply meets demand." Real divergence shows up on the second question even when the first looks saturated. Reusing the existing 2SFCA pipeline keeps the validation story consistent: the threshold and the per-pixel adequacy classification are already the validated outputs; sampling at point locations is the only added step.

**Source choice.** **NCES CCD 2022-23** for the public-school directory (LEVEL + CHARTER status) + **NCES EDGE 2021-22 Geocode Public Schools** for public/charter lat/lon + **NCES EDGE 2021-22 Geocode Private Schools (PSS Universe)** for private lat/lon. Accepting the ~1-year vintage offset between CCD 2022-23 and EDGE 2021-22 — school directory data doesn't churn rapidly, and the EDGE 2021-22 geocoded set is the most recent geocoded vintage available at build time. **Private included** per the user decision; the alternative (public + charter only) would understate where children actually attend school in cities with substantial private enrollment. Vintage + private-inclusion are surfaced on the metric card's tooltip + REFERENCE §6 — they aren't a hidden caveat.

**Alternatives considered.**
- *Skip private schools.* Public + charter only. Cleanest single-source story (NCES CCD covers it). Rejected: K-12 private enrollment is non-trivial (~10 % nationally; varies by metro) and ignoring it would systematically understate destination access where private siting differs from public.
- *Use NCES SABS attendance boundaries as a region layer instead of a point readout.* Stronger instrument (each school's actual catchment, not a point sample), but a much bigger lift — schools_file would become a polygon layer, region-local treatment would need attendance-boundary aggregation, and SABS coverage is non-uniform (public-school catchments only — charter and private have no catchments). Deferred as a follow-up; v1 = point readout.
- *Enrollment-weighted readout.* Read as "% of students at schools with access" instead of "% of schools." Requires reliable per-school enrollment counts (CCD has them for public + charter; PSS Universe doesn't always). Skipped for v1 per the user brief.
- *Region-local treatment.* `_sample_schools_access` already accepts a `mask` parameter for region-clip. Not wired into evaluate_scenario's region_local block for v1; follow-up.

**Consequences.**
- Honest by construction: the metric never overclaims relative to the residential pipeline because it shares the same `adequate` mask. Different question, same evidence.
- Schema version bumped 34 → 35 to add six new scalar fields (`schools_nature_access_pct`, `schools_n_total`, `schools_n_with_access`, `schools_public_pct`, `schools_charter_pct`, `schools_private_pct`) plus the full structured `schools_nature_access` dict for the UI card's tooltip + breakdowns.
- MN downtown surfaces the headline use-case: at 10/50/50 conversion, residential Nature Access is 14.1 %, Children's Nature Access is 11.8 %, **Schools at 6.7 %** — schools sit in less-served areas than residents on average. SA at 99.7 % residential is saturated and the schools metric is close to it (99.5 %) — the divergence shows up where it matters.

**Code touchpoints.** `calculate_schools_nature_access` (top-level helper) + `_sample_schools_access` (low-level mask-sampler, region-mask aware for future region-local treatment); `evaluate_scenario` calls inline alongside the residential `calculate_nature_access` call; `CityState.schools_pixels` + `schools_sectors` + `schools_metadata`; `_load_city_runtime_state` Phase 2c (load GeoJSON → project to LULC CRS → convert to pixel coords); `scripts/data/prep_school_points.py` (offline NCES download + filter + per-city write); module-level aliases + `_rebind_city` extension; UI card in `app.py` (Human & Social row extended from 4 to 5 columns); doc-side surfaces in `REFERENCE.md §6 Nature Access at Schools`, `CAPABILITIES.md` (Outcome models), and the WHATS_NEW school-related-scenarios bullet.

### 6.8 Population raster — block-area allocation and its known limitations

**What it is.** Every population-weighted metric (Nature Access, Children's Nature Access, Schools, UMH) reads `pop_count_raster`: Census 2020 block totals spread uniformly across each block's **area**, rasterized to the NLCD grid and **bilinear-resampled** to the working grid. It is **not parcel-aware** and **not dasymetrically refined** — a block's residents are smeared across all of the block's pixels (including any school, water, or right-of-way pixels inside the block), not placed on the residential parcels.

**Consequence on the honesty surface.** A selection restricted to an ownership class (e.g. school land) shows a non-zero population count that reflects this area-spread allocation, not on-site residents. The mechanics-layer caveat lives one click in, on the Selected-region impact table's population rows (`app.py` region-local block, gated on an active ownership filter) — not on the headline. No number changes; the caveat explains the number.

**Measured magnitudes (valid-extent population; measured 2026-06-04).** Share of modelable-extent population sitting on land that can't hold housing is small:

| Bucket | SA | MN |
|---|--:|--:|
| Open water (NLCD 11) | 0.04 % | 0.05 % |
| Hard NLCD exclusions (water + wetlands + barren) | ~0.77 % | ~0.10 % |
| K-12 school-owned (ownership enum 4) | 0.15 % (2,915) | — (no layer) |
| All public/institutional (enums 1–4, +university) | 4.71 % | — |
| University-owned (enum 6 — plausibly real dorm pop) | 0.44 % | — |

The 4.71 % SA public/institutional figure is **dominated by city-owned land (~3 %), not schools** — and city land routinely holds housing, so ownership class is a poor "no-housing" proxy. SA's ownership layer also leaves **16.14 % (~307k) of valid-extent population on unclassified (nodata) pixels**, which are excluded from every ownership selection. MN has **no ownership layer** — ownership filters are SA-only.

**Decision — dasymetric redistribution NOT pursued.** The clean correctable cases (open water, hard NLCD exclusions) total <1 % of valid-extent population, while any change to `pop_count_raster` is baseline-affecting: it forces a `SCENARIO_SCHEMA_VERSION` bump and a 40/40 re-snapshot, and shifts every population-weighted number users may already have cited. With no large *clean* bucket to recover (the biggest institutional bucket is city land, which legitimately houses people), the correction doesn't earn its blast radius. Documented here so the tradeoff is evidence-based, not folklore; revisit if a parcel-level residential layer becomes available (would also retire the in-app allocation caveat).

**Fix — the 16.14 % ownership-nodata was a 30 m grid-tiling artifact, re-rasterized sub-pixel in this commit (2026-06-04).** Not a coverage hole: the source is a complete full-county BCAD pull (710,772 parcels, 2026-05-31, EOD-confirmed; `scripts/data/download_bexar_parcels.py`), so a re-pull recovers nothing — the brief's hypothesized "fuller source" was already the current source. The nodata was a rasterization artifact: 30 m cells whose center landed in the interstitial gap between abutting parcels got no parcel. (Framing trail: commit `a3a8390` first mis-read this as a developed-land *coverage hole*; `48dd9f2` corrected it to *alignment artifact*; this commit fixes it.)
- **Evidence it was alignment, not absent parcels.** 98.6 % of the ~307k nodata pop sat on developed LULC (94 % dense developed 22/23/24), ~0 % on open water; a nearest-classified distance transform put **99.1 % of nodata pop within 1 px (99.8 % within 2 px)** of a classified parcel, and every top nodata component (incl. the 523k-px blob) had a 100 %-classified perimeter — interstitial gaps, not missing parcels.
- **Fix — sub-pixel area-majority.** `_rasterize_two_band` now burns parcels at **3× (10 m)** and assigns each 30 m cell the **area-majority owner class** of the parcels intersecting it; band 2 (vacant) is carried through the same per-class priority burn (packed as `enum*2 + is_vacant`) so a cell inherits the vacant flag of the parcel(s) it was assigned from, not an independent pass. Result: **nodata pop 16.14 % → 0.97 %** (recovered 289k, a 16× cut); **96.4 % of recovered pop → `private`** (residential), the rest across unknown / city / etc. Boundary footprint: **4,514 previously-classified px (0.17 %)** shifted to their area-majority owner — expected, and more accurate than the prior center-point pick.
- **Residual 0.97 % (18,424 pop) is nodata BY DESIGN.** These are 30 m cells with **no parcel present at 10 m** — skipped degenerate slivers, narrow non-parcel ROW, and genuine micro-gaps (water / federal interiors). They get no majority and stay `-1`; **no inferential nearest-fill** is applied (ownership isn't fabricated for cells no parcel covers). The in-app population-allocation caveat still covers this residual.
- **Blast radius (confirmed).** Ownership-dependent baselines moved: the `verify_baselines` raster-integrity lock (`_RASTER_EXPECTED_AC`, ±5 %) was re-snapshotted (every class up; `private` +9 %, county +8 %, school +9 %); the rule-output polygon-Acres (±0.5 %) are unchanged (the classifier didn't move, only the rasterization). The **40/40 citywide population-weighted baselines are byte-identical** — they read `pop_count_raster`, not the ownership raster. No `SCENARIO_SCHEMA_VERSION` bump (no `evaluate_scenario` output-shape change).

**Code touchpoints.** Population side: `load_population_data` (bilinear resample); `pop_count_raster` on `CityState`; consumed by `calculate_nature_access` / `_invest_una_pct_pop_supply_ge_demand` / `calculate_mental_health_impact`; in-app caveat in the `app.py` Selected-region impact block (ownership-filter-gated expander). Ownership-rasterize side: `_rasterize_two_band` (sub-pixel area-majority) in `scripts/data/download_bexar_parcels.py`; the re-snapshotted `_RASTER_EXPECTED_AC` lock in `verify_baselines.py`; `data/sa/sa_ownership_2band_30m.tif`; provenance in `DATA_INVENTORY.md`.

---

## 7. Lookup table and surrogate optimizer

### 7.1 Three model-quality modes — training-scenarios story, hidden tree count

**Decision.** Three model-quality modes (Fast prototype / Balanced / High resolution) selected via the Advanced Settings radio. The mode determines (1) the training-scenario source for the surrogate and (2) the RF tree count. **Only the training-scenario story is shown to users**; the tree count is intentionally hidden.

| Mode | Training scenarios | RF trees |
|---|---|---:|
| Fast prototype | Live `compute_scenario_grid(step_pct=10, step_alloc=25)` — ~90 | 100 |
| Balanced | Per-city dense CSV (`data/scenarios_dense_<city>.csv`, precomputed offline) — ~726 | 200 |
| High resolution | 2,541-entry lookup table reused as training data (free; the rows are already computed for instant slider response) | 300 |

**Why.** The training-scenario count is the conceptually meaningful knob — it controls how densely the slider space is sampled. The tree count is a tuning knob that users would over-interpret without enough RF background to use it well; surfacing it would invite confusion about "more is better" (it isn't, past a point).

**Alternatives considered.**
- Surface the tree count directly — invites over-tuning.
- Single mode with auto-detection — defeats the user's ability to opt into faster vs more-thorough surrogates.

**Consequences.** **Conceptual separation:** training scenarios and tree count are surrogate-side knobs; the optimizer's ~10,000-random-candidate sampling at search time is independent and unchanged across modes. The Balanced default CSV is built offline by `precompute_scenarios.py`, which stubs `streamlit` so it can `import app` and reuse `evaluate_scenario` / `_compute_carbon` / `calculate_nature_access` / `pop_count_raster` without duplicating logic.

**Revisit if.** A new training-scenario source (e.g. Latin Hypercube sampling) emerges that beats grid sampling at the same point count.

**Code touchpoints.** `SURROGATE_TREES` dict in `app.py`; `compute_scenario_grid` / `compute_lookup_table`; `precompute_scenarios.py`; `_cached_train_surrogate` (`@st.cache_resource` wrapper).

### 7.2 Optimizer as trustworthy scenario discovery

**Decision.** The optimizer reads as the bridge from exploration (L4) to canonical validation (L5): a discovery loop that surfaces promising scenarios to validate further. Applied optimizer scenarios are tagged `PROVENANCE_OPTIMIZER` so they're properly distinguished from `PROVENANCE_EXPLORER` in both the scenario header and the D1 export bundle.

**Why.** Pre-#4, the surrogate's prediction table labeled itself "predictions — click Apply to verify," but the Apply path silently re-tagged the scenario as Explorer-generated and the export recorded it as Explorer. The framing said "discovery → validation" while the data path said "discovery → undifferentiated." #4 closes the gap.

**Alternatives considered.**
- Drop the surrogate, run live `evaluate_scenario` per optimizer candidate — too slow (SA's ~0.9 s × 10,000 candidates ≈ 2.5 h per optimization run); kills interactivity.
- Tag optimizer-applied scenarios as Explorer (the pre-#4 state) — silently misrepresents provenance in the export bundle.
- Persist the optimizer flag forever after Apply — wrong; if the user edits any slider after Apply, the scenario is no longer the surrogate-suggested one and should re-tag as Explorer.

**Consequences.**
- **Auto-clear discipline.** The `applied_from_optimizer` session-state flag auto-clears on any rerun where the current slider state diverges from `_applied_optimizer_values`. Manual edits, preset clicks, Best-by-Goal Apply, and city changes all reset to `PROVENANCE_EXPLORER`. The discipline is enforced at the top of every rerun, not at edit time.
- **Detection order** (used by both the main-panel header and the D1 export branch): `pct_converted == 0` → BASELINE; else `applied_from_optimizer` → OPTIMIZER; else EXPLORER.
- **Entry-text reframe** in the sidebar: "Find Best Scenario" (verdict framing) → "Discover scenarios to validate" (discovery framing). The expander caption walks the discovery → Apply → evaluate → export path.
- **Best-Scenarios-by-Goal Apply is NOT classified as OPTIMIZER** — those come from the precomputed scenario grid, not the surrogate's discovery search. They remain Explorer-generated.

**Revisit if.** The surrogate's prediction accuracy reaches a level where Apply could be skipped (i.e. surrogate predictions could be exported directly as validated scenarios) — not in foreseeable scope.

**Code touchpoints.** `applied_from_optimizer` and `_applied_optimizer_values` session-state keys; the `apply_opt_*` button handlers; `_render_scenario_provenance_header`; `_build_invest_bundle_for_current_scenario`; the sidebar Discover-scenarios subheader.

### 7.3 Region-constrained optimizer — prefilter then verify

**Decision.** When a region or ownership filter is active, the optimizer runs a two-stage pipeline: the Fast surrogate (~90 recipes, 100 trees) shortlists ~40 Pareto-efficient candidates citywide, then the full engine evaluates each shortlisted recipe on the active `region ∩ ownership` mask. Displayed values are engine-true region-local — no surrogate predictions surface. Records carry a new `PROVENANCE_REGION_OPTIMIZED` ("Engine-verified — region-optimized"), distinct from the citywide path's `PROVENANCE_OPTIMIZER` ("Citywide surrogate suggestion — engine-evaluated on apply"). Canonical internal spec is `docs/internal/REGION_OPTIMIZER_SPEC.md`.

**Why prefilter-then-verify (vs either alone).** Two empirical findings from the Phase-0 and Phase-0.5 recon scripts under `scripts/`:

- **Phase-0 (`phase0_region_eval.py`)** measured one full-engine eval over a region at ~2.1 s, **flat regardless of region size** (a 25-px synthetic, a 71 k-px small district, and a 252 k-px large district all came in within ±5 % of each other, and all 24–30 % *more* expensive than the citywide reference at 1.73 s). The biophysics is non-local — UCM HMI, UNA 2SFCA convolutions, NDVI, Carbon four-pool stocks, and SCS-CN runoff all integrate over the full AOI raster; a region mask only narrows the convertible sample and triggers a region-local clipping pass that adds the +24 % overhead. **A region cannot be cheaply isolated without leaving the validated tier** (sub-AOI biophysics would require re-validating against canonical InVEST on the sub-AOI). At 2.1 s per eval, the interactive budget is ~57–143 candidate evals at a 2–5 min target — a 5×5×5 = 125 grid fits comfortably but a 10×10×10 = 1,000 grid does not. So **engine-only is out** (too few candidates).
- **Phase-0.5 (`phase0_5_surrogate_ranking.py`)** measured how well the citywide-trained Fast surrogate ranks region-scoped candidates compared to the engine ground truth on SA Council District 5. Across 32 candidates spanning the knob space, the surrogate's per-metric Spearman ρ landed at 0.83–0.98 across cooling / flood / runoff / carbon / food / nature-access, and recall@top-15 hit 1.00 for every metric — the engine's true top-5 winners on every objective sit within the surrogate's top-15 shortlist. So **surrogate-only is also out** (predictions are not engine-true) but **surrogate-as-shortlister is sound**: the ranking is good enough that a generous K = 40 prefilter doesn't drop true winners, and the weighting-agnostic property holds (every constituent metric ranks well → any user-weighted combination also ranks well).

The combined budget — 1 surrogate prefilter (instant) + K × 2.1 s engine-verify — lands at ~85 s for K = 40 on SA. That's the only configuration that fits the interactive target while keeping the displayed values engine-true.

**Why values are engine-verified, not predicted.** The Phase-0.5 ranking quality is strong but not perfect (ρ = 0.83 on food is the weakest axis). If displayed values were the surrogate's predictions, a user weighing food heavily could see a recipe whose ranked position is correct but whose absolute predicted food value diverges from the engine truth by a non-trivial margin. The honest path is to keep the surrogate's role limited to ordering — what it does well — and surface engine-true magnitudes for everything the user *reads*.

**Spec-vs-v1 gap to flag: weight-slider rerank is not instant.** REGION_OPTIMIZER_SPEC.md §3 calls out the K-weight-robust property: moving a weight slider should re-rank the already-engine-evaluated K instantly without another K × 2.1 s pass. The v1 implementation falls short — the click handler runs the full pipeline (prefilter + K=40 engine-verify + rank + dedup) every time "Optimize selected area" is clicked, and a weight-slider change does not trigger anything until the next click. The shipped behavior: change weights → re-click Optimize → wait ~85 s for a fresh pass. Future: cache the engine-evaluated K alongside its mask + recipe set so subsequent clicks at the same `(mask, recipe set)` skip straight to rank+dedup. Not in v1 because it adds session-state surface area without affecting honesty — the displayed values are still engine-true.

**The honest split — what's real, what's caveated.**

- **Engine values are true.** Each returned record is the output of `evaluate_scenario` called with the same recipe + the same mask the user has active. The reconciliation assertion in `verify_baselines.py` locks this: a fresh engine eval on the record's recipe must reproduce the recorded metrics at rtol=1e-9. The meta-test (poisoning a record with the surrogate's citywide `mean_hm` prediction and asserting the reconciliation cell **fails**) proves the guard catches surrogate-vs-engine drift; the reconciliation isn't green-light theatre.
- **The shortlist carries the completeness caveat.** Up to 40 candidates from a ~90-recipe surrogate Pareto. The remaining ~50 citywide-suboptimal recipes might score well on a specific user weighting under the region's specific biophysics — the recon validated *ranking*, not *frontier coverage*. The header reads *"Best tested mixes — selected area"* — best among the candidates the engine actually tested, not "the optimum"; the caption owns both the truth ("results shown are real") and the limit ("shortlist may not be exhaustive").

**Why a distinct provenance constant (`PROVENANCE_REGION_OPTIMIZED`).** Conflating with `PROVENANCE_OPTIMIZER` would let region-optimized records inherit the citywide surrogate's framing — but the region path's displayed values are engine-true, not surrogate-shortlisted suggestions awaiting Apply-time verification. The distinct constant + label keeps the user-facing meaning honest: "Citywide surrogate suggestion — engine-evaluated on apply" frames the citywide path (the table shows surrogate predictions; Apply re-runs the engine); "Engine-verified — region-optimized" frames the region path (the table already shows engine-true region-local values; the search scope was your region filter, the candidate set was shortlisted). The verify_baselines optimizer cell + Two-RELAY lock cell (Assertion C) jointly assert both constants and rendered Source labels remain distinct, and that the citywide `optimize_scenario` DataFrame doesn't emit the `source` / `validation` columns the region path uses — a both-ways collapse-prevention guard.

**Alternatives considered.**
- **Engine-only over a small region (no surrogate).** Phase-0 ruled out: per-eval cost is flat, not region-size-proportional. A brute-force 7×7×7 = 343 engine evals over a small region would take ~12 min — outside an interactive budget.
- **Surrogate-only, no engine-verify.** Phase-0.5's ρ = 0.83 on food (the weakest axis) means absolute predicted values could mislead a user weighting food heavily. Engine-verify costs ~85 s and removes that risk.
- **Reuse `PROVENANCE_OPTIMIZER`.** Conflates engine-true region-local values with the citywide path's surrogate predictions. The user-facing distinction is what the Two-RELAY lock surfaces explicitly — citywide "engine-evaluated on apply" vs region "engine-verified" — not just internal bookkeeping.
- **Apply path running citywide values + flipping to engine-true only on Apply** (the pre-region-optimizer state). The user would pick among 5 records whose ordering is engine-correct but whose absolute values are not what they'll see after Apply — confusing and hides the per-axis-magnitude story.

**Revisit if.** A region-aware surrogate becomes available (training data per-region is the limiting factor) — could move the engine-verify step earlier or replace it. Until then, the prefilter-then-verify shape is what fits the empirical budget envelope.

**Full-recall validation (2026-06-04) — prefilter loses nothing; explore-reservation NOT built.** A denser follow-up to Phase-0.5: a ground-truth engine sweep (pct 5–50 step 5 × gi/ff step 10, gi+ff ≤ 100 = 660 recipes/selection, ~7× the surrogate grid), scored against the actual `optimize_scenario_region` output in one union-normalized frame (dense ∪ K-shortlist, the optimizer's own min-max → weighted-sum scheme). Across **SMALL (D5+D9 ∩ school, ~165 ac) / MEDIUM (D5, ~8k ac) / LARGE (all districts, ~118k ac) × equal + cooling-max weights**, **top-1 regret = 0.0000 in every case** — the prefilter returns the *exact* true #1 (all six per-metric region-local deltas 0), and regret is flat as acreage shrinks (SMALL = MEDIUM = LARGE = 0).
- **Why it holds.** Under the current **non-spatial engine** the metrics are monotone toward simplex corners (more conversion + an extreme allocation maximizes the weighted min-max sum), so the optimum is always a *grid corner the Fast candidate grid already contains* — there is no interior optimum to drop. Recall@K reads a low 1–2/5 only because the "missed" true-top-5 are flat-ridge neighbors (Δscore ≤ ~0.003) the coarse step-25 grid can't represent; they're strictly worse than the recovered corner. The K = 40 cap never engaged — the surrogate's 6-metric Pareto front over the 75 candidates is only **12 points**, so the maximin sampler is dormant. Missed-recipe attribution was **100 % "absent from grid"** (0 % Pareto-dropped) — so the only (zero-regret) lever would be denser/continuous candidates, never uncertainty/maximin explore slots.
- **Decision.** The explore-reservation / continuous-wildcard mechanism is **not built** — it's insurance against a non-problem at current regret.
- **Re-validate if.** This guarantee holds **only while the engine stays non-spatial**. If flow-routing or any spatial/placement dependency is added (so the optimum can move off a simplex corner into the interior), recall must be re-measured — an interior optimum the coarse grid can't represent would then carry real regret, and the reservation/wildcard question reopens.

**Code touchpoints.** `surrogate.optimize_scenario_region`; `app.py` sidebar mode switch keyed on `_filter_active`; `_cached_fast_surrogate_for_region` (Phase-0.5-validated Fast configuration regardless of active model-quality mode); `applied_from_region_optimizer` flag + auto-clear mirror of `applied_from_optimizer`; `PROVENANCE_REGION_OPTIMIZED` constant in `natcap_scenarios.py`; the `_PROVENANCE_HEADER_INFO` entry for the new constant; the region-optimizer cell in `verify_baselines.py` (subset / reconciliation / meta-test / provenance-distinction).

### 7.4 Band semantics: model disagreement vs calibrated error

The optimizer's citywide suggestion bands are the **10th–90th percentile across the RF's trees** (`surrogate.predict_with_uncertainty`) — i.e. **model disagreement**, NOT a calibrated confidence interval. The visible surfaces say "model-disagreement bands" precisely so they can't be misread as truth coverage.

- **What the band is.** Inter-tree spread. It does **not** bound the evaluator-computed value, and inter-tree agreement ≠ accuracy (trees can agree on a wrong answer). It is **blind to placement**: the same `(pct, gi, ff)` yields the same band regardless of siting — the same spatial-blindness as the surrogate itself (linked to §7.3's recall guarantee and #3/#4 in the decision log).
- **Presence/absence is meaningful BY DESIGN.** Bands ⇒ fast estimate (citywide suggestions only). No bands ⇒ evaluator-computed (applied scenarios, selected-area / region-optimized results — values are real, not quantiles). The ±2 °F temperature note is a *separate* axis: it reflects HMI-to-temperature **calibration** accuracy, not model disagreement.
- **Future calibrated path — do NOT ship partial.** (1) Hold out a full-evaluator set; (2) compute AI-vs-evaluator residuals per metric; (3) build residual distributions; (4) **stratify by placement strategy / add spatial features to X**; (5) display empirical ranges. **Step 4 is load-bearing:** residuals over `(pct, gi, ff)` alone are calibrated *on average* but miscalibrated *per scenario* — too narrow for spatially-concentrated placements, too wide for diffuse ones. A band that wears the word "calibrated" while still placement-blind is arguably worse than the honestly-labeled tree-spread, because it hides the same error under a trustworthy name. Ties to the §7.3 recall re-validate trigger: re-check if the evaluator becomes spatial/flow-routed.

**Code touchpoints.** `surrogate.predict_with_uncertainty`; the optimizer overlay error bars + suggestions/candidate captions + "Show model disagreement bands" expander in `app.py`; the ±2 °F calibration note in the temperature assumptions tab.

---

## 8. Validation and provenance design

### 8.1 Two-surface validation vocabulary — locked

**Decision.** The validation taxonomy uses two locked vocabularies, distinct by surface:

- **Per-card badge (4 states):** `NatCap published value` / `≈ NatCap method` / `≈ Aligned method` / `Prototype`.
- **Per-scenario provenance header (5 sources):** `Baseline` / `NatCap published reference` / `Explorer-generated` / `Surrogate-suggested` / `Engine-verified — region-optimized` (the fifth added by §7.3; distinct from `Surrogate-suggested` because the displayed values are engine-true region-local, not surrogate predictions).

**The four-state badge taxonomy is the authoritative version in NATCAP_ALIGNMENT.md §2.** This section owns the *design rationale*; NATCAP_ALIGNMENT.md owns the per-metric assignment; REFERENCE.md owns the user-facing explanation; ARCHITECTURE.md §6 owns the rendering components.

**Why two surfaces.** Card-level validation captures methodology-vs-measurement nuance per metric (temperature and carbon can both cite measured per-pixel parity; an aligned-method dollar metric cannot). Scenario-level provenance captures where the *scenario as a whole* came from (slider-driven exploration vs NatCap-published reference vs optimizer suggestion). Conflating the two would lose precision in either direction.

**Why these four states per surface, not more or fewer.**
- Card surface: a Match/Diverged state (5th) was the original B2 plan but requires prototype reproduction for NatCap fixed scenarios, which is gated on the unavailable compound LULC inputs (OPEN_QUESTIONS) and may never arrive. Adding speculative states would imply validations not actually performed.
- Scenario surface: a Saved state was considered; rejected because saved scenarios carry whatever provenance the scenario had at save time. Saving doesn't create a new validation context.

**Conservative-floor framing.** The conservative floor is *don't overclaim*. Concretely:
- `NatCap published value` (green) fires only in the fixed-scenario reference view — the one case where the prototype literally displays NatCap's number from `natcap_reference_outputs.csv`, no computation.
- Everywhere else (Baseline / Explorer / Optimizer), even where the prototype's methodology is canonical InVEST and parity is measured, the badge stays blue **≈ NatCap method**. Blue is honest about "prototype's own computation, methodology-aligned" without overclaiming.
- Metric-aware tooltip on `≈ NatCap method`: temperature cites measured per-pixel HMI parity (Brief 28b); carbon now cites measured per-pixel parity too — the four-pool framework (Brief 30) is validated vs canonical InVEST 3.19.0 at MAE ≈ 0 / r 1.0 in matched units (Relay 69). The badge stays blue regardless (prototype's own computation, no NatCap scenario anchor) — the tooltip carries the evidence, never overclaiming reproduction of NatCap's citywide absolute.

**Alternatives considered.**
- A unified single-vocabulary taxonomy across surfaces — flattens the methodology-vs-source distinction.
- A confidence-tier vocabulary (high / medium / low) — was the prior framing; replaced because confidence and validation are different axes (MH cards can be "high-confidence in methodology" but "no per-scenario reference value to compare against").
- Match / Diverged states everywhere — overclaims at the per-card level (most metrics have no NatCap published per-scenario value to match against).

**Consequences.**
- One source of truth for the badge vocabulary (NATCAP_ALIGNMENT §2); REFERENCE §4 mirrors the user-facing copy; ARCHITECTURE §6 wires the renderers.
- Curated non-CSV-card status map (for the `explicit_status` path of `_render_validation_caption`): runoff = `aligned_method`; NDVI = `prototype`; implementation cost = `prototype`; flood damage / volume = `aligned_method`; cooling energy = `aligned_method`; carbon-$ on SA = `natcap_published` (derived); carbon-$ on MN = `prototype`; MH costs = `aligned_method`; cost-effectiveness ratios = `prototype`.
- Per-scenario provenance source for the OPTIMIZER state is "engine-validated; full-raster evaluated — exploratory candidate for further validation" — full-raster evaluation is always true when the badge appears (auto-clear on slider divergence per §7.2).

**Revisit if.** Compound LULC inputs for the NatCap fixed alternatives arrive — the deferred Match / Diverged states from the original B2 design (preserved in §11) become re-implementable.

**Code touchpoints.** `_render_validation_caption` (per-card badge); `_render_scenario_provenance_header` (per-scenario header); `_PROVENANCE_HEADER_INFO` (source → label/validation/color table); `_VALIDATION_BADGE_COLOR_HEX` (the green / blue / gray palette); `natcap_validation.render_validation_badge`. NATCAP_ALIGNMENT §2 holds the assignment table.

### 8.2 Provenance header — placement and structure

**Decision.** The per-scenario provenance header renders as a bordered, colored block above the metric grid (just before `#### Ecological`) showing **Source:** {Baseline / NatCap published reference / Explorer-generated / Surrogate-suggested} + **Validation:** {one-line claim}. Same renderer wires both the Explorer dashboard and the fixed-scenario reference view.

**Why.** A scenario-level provenance signal must be impossible to miss but must not visually overwhelm the metric cards. A colored bordered block above the cards parallels the per-card badges' green / blue / gray palette so the two surfaces read as scaled versions of each other.

**Alternatives considered.**
- A small inline caption under the page title — too easy to miss.
- A Streamlit `st.info` / `st.success` callout — color palette doesn't match the per-card badge palette (Streamlit's defaults don't include the specific green / blue / gray needed).
- Per-card provenance instead of a scenario-level header — already covered by per-card badges; would be redundant.

**Consequences.** The bordered `div` (rather than a Streamlit native component) is the cost of color consistency with the per-card badges. `_PROVENANCE_HEADER_INFO` is the single source of truth for the (source label, validation line, color) triple keyed by the four `PROVENANCE_*` constants; both render paths (fixed-scenario view + Explorer dashboard) consume it.

**Revisit if.** Streamlit adds a colored-callout primitive with a configurable border color.

**Code touchpoints.** `_render_scenario_provenance_header`; `_PROVENANCE_HEADER_INFO` keyed off the four `PROVENANCE_*` constants from `natcap_scenarios.py`.

### 8.3 Cross-source comparison table — Δ-basis invariants

**Decision.** The Tradeoff Analysis tab's cross-source comparison table puts NatCap published references, the active scenario, and any saved scenarios side by side with explicit Source and Validation columns. The three NatCap-shared metric columns (Temperature, Carbon stock, Carbon Value $) are rendered as **deltas vs each row's own baseline**, with the NatCap baseline row reading literal `"baseline"` rather than its absolute citywide value.

**Why.** Two invariants drive the design:
- **Unified Δ-basis.** Every row in those three columns is a delta. NatCap rows use NatCap's baseline (via `natcap_validation.published_delta`); prototype rows use the prototype's baseline (via `evaluate_scenario`'s already-delta outputs). Mixing absolute values (NatCap baseline at 90.08 °F / 107.32M t CO2e) into the same columns as deltas (`+0.5°F` cooling, `+0.5M t CO2e`) would produce misleading row-to-row comparisons (`+148M next to +0.5M` reads as "NatCap is 296× better" — meaningless).
- **Source + Validation columns mandatory.** They're the load-bearing piece of the honesty story; the brief's guardrail "never ship the table without them" is enforced structurally (no code path emits a row without them).

**Alternatives considered.**
- Option B: new `compare_set` state separate from `saved_scenarios` — duplicates the Save UX.
- Option C: repurpose `saved_scenarios` as "pinned for comparison" + auto-include NatCap — overloads the Save action.
- Option A (chosen): NatCap anchors always-available + `saved_scenarios` for the rest — reuses the existing Save mechanism, aligns with how the user already thinks about scenario persistence, requires only one new field on each saved dict (`provenance`).
- Include Flood in the comparison — different derivations between baseline (compound→NLCD×tree reduction) and alternatives (native NLCD×tree raster); not directly comparable, and SA flood is ~scenario-invariant under the design storm anyway. The per-scenario Flood card handles that; excluding from the table avoids a spurious-looking gap.
- Render the NatCap baseline row's absolute values in the three Δ columns — mixes bases; produces the `+148M / +0.5M` misreading.

**Consequences.**
- Row composition: NatCap fixed anchors (SA only) → current scenario (always, marked `▶ Current — …`) → saved scenarios for the active city. Empty-state is impossible — the Current row is always present.
- Provenance recording at save time: the Save handler stamps `saved["provenance"]` using the same detection as the main-panel header (BASELINE / OPTIMIZER / EXPLORER). Older saves backfill best-effort from `pct_converted`; OPTIMIZER cannot be backfilled (the flag's in-memory state isn't recoverable) — the safer underclaim.
- `—` for unavailable cells is driven by row provenance, not hardcoded (NatCap rows get `—` because compound inputs are gated; Explorer / Optimizer / Saved rows get the actual computed value including 0).
- Source / Validation cells use short labels (`displayed (NatCap)` / `engine verified` / `engine + full-raster`); the full mapping lives in the column-header `help=` tooltip.

**Revisit if.** NatCap MN scenarios become available (would drop the `startswith("San Antonio")` gate on the NatCap-anchor section). Or compound inputs for SA fixed alternatives arrive (would lift the `—` cells on Nature Access / Cooling Energy / MH / Food / NDVI).

**Code touchpoints.** The Tradeoff Analysis tab's comparison-section render (top of tab2, before `#### Tradeoff Space`); `_PROVENANCE_HEADER_INFO` (source-to-validation mapping); `natcap_validation.published_delta`; `saved["provenance"]` field on each saved-scenario dict.

### 8.4 KNOWN_DIVERGENCES — surfaces and the export-bundle completeness check

**Decision.** Pre-vetted methodology divergences from canonical / published values are stored as a single locked list (`KNOWN_DIVERGENCES` in `export_invest_bundle.py`, ~7 entries today). The list surfaces in two places:

- **Export bundle metadata.json** — every exported zip carries the entire list verbatim under `scenario.known_divergences` so a downstream user opening a bundle reads the disclosures alongside the inputs, not separately.
- **App-side sidebar caveat captions** — the captions next to a relevant control re-state the same caveat in user-facing prose where the user is making the choice. Two examples currently wired:
  - Eligible-land filter panel (sidebar) carries the `ownership_rule_derived` caveat — classes are rule-derived from BCAD owner-name + exemption parsing, NOT validated against a title registry; `school` matches `ISD` / `SCHOOL DISTRICT` only; `university` spans both public (UT / A&M / Alamo CCD) and private (Trinity, St. Mary's, OLLU) campuses.
  - Region-local Selected-region impact table carries the `region_local_spillover_reach_models` caveat — UCM / UNA / UMH reach effects are clipped to the region boundary.

**Export-bundle completeness check.** `verify_baselines.py:680-689` (Honesty-Surface Pass Commit 4) enforces **bidirectional set-equality** between the in-memory `eib.KNOWN_DIVERGENCES` list and the exported metadata.json's `scenario.known_divergences` array — both `expected_ids - emitted_ids` (silently dropped in the serializer) and `emitted_ids - expected_ids` (unexpected extras in the output) fail the gate. The guarantee is "the export bundle reflects the in-memory list exactly," NOT "the list can only grow." Entries can be added or removed freely; the assertion just enforces that whatever's in `KNOWN_DIVERGENCES` matches what gets stamped onto metadata.json. The two app-side caveat captions in the prior paragraph are NOT machine-checked — they're prose, and a refactor that drops one of those captions wouldn't trip any assertion.

**Why disclosure-as-data, not prose-only.** A prose caveat in REFERENCE.md or an app caption can drift silently. A data row in `KNOWN_DIVERGENCES` that's stamped into every export bundle and machine-checked against the serializer cannot — a refactor that breaks the metadata.json serialization fails the gate.

**Three-state taxonomy cross-reference.** The four-badge per-card vocabulary in §8.1 ties to the broader three-state framing the project communicates externally: *validated where possible* (per-pixel parity vs canonical InVEST, MAE ≈ 0) / *displayed where NatCap-published* (the dashboard surfaces NatCap's number; doesn't reproduce it) / *exploratory where the model is sound but no anchor exists* (Explorer-generated and Optimizer-suggested scenarios — engine-validated, no per-scenario reference value to compare against). The full three-state framing lives in STRATEGY.md §3-§4 and the in-app B2a "validated vs displayed vs exploratory" note; this section's four badges are the per-card refinement of that three-state stance.

**Alternatives considered.**
- A separate `disclosures.md` doc — would need cross-link discipline; the metadata.json embedding is more durable.
- Per-metric divergence flags in the card validation badge — the 4-state badge taxonomy (§8.1) is locked, and Match/Diverged states require per-pixel reproductions that aren't available for most metrics.

**Revisit if.** A divergence becomes resolved (e.g. SA's compound LULC arrives → `sa_citywide_not_reproduced` can be re-scoped) — remove the entry from `KNOWN_DIVERGENCES` and the gate stops checking for it.

**Code touchpoints.** `KNOWN_DIVERGENCES` (`export_invest_bundle.py`); the completeness check (`verify_baselines.py:680-689`, Honesty-Surface block); sidebar caveat captions in the Eligible-land-filter panel + the Selected-region impact table.

---

## 9. Export for InVEST

### 9.1 Workflow design — value-ladder Level 5 as a concrete capability

**Decision.** A single sidebar button packages the currently-displayed scenario as a runnable canonical InVEST 3.19.0 input zip — rasters + AOIs + biophysical tables + per-model `args.json` for UCM / UNA / UFR / Carbon / UMH + a `metadata.json` recording provenance, generator parameters, and per-model validation state. SA-only for v1.

**Why.** The value-ladder Level 5 ("workflow layer") needs a concrete capability, not aspirational framing. The export bundle is the bridge for users who want canonical InVEST results — they run `natcap.invest.*.execute()` against the bundle's inputs and compare against the prototype's reported cards. It also doubles as a reproducibility artifact (a frozen scenario state with full provenance).

**Alternatives considered.**
- Inline canonical InVEST runs from the dashboard — slow (minutes per call) and disk-write-heavy (per-rerun temp file management). The §6.1 reasons that justify the numpy port also justify keeping canonical InVEST out of the live path.
- Embed the export as a download-only artifact without the runnable args.json — strips the "run canonical InVEST on this" affordance, which is the whole point.
- Export each model's inputs as separate downloads — fragments the single-scenario picture; a zip with the whole bundle is one download.

**Consequences.**
- **Two-step Prepare → Download UI.** Avoids rebuilding the ~20 MB bundle on every Streamlit rerun. Plus a "Clear prepared bundle" reset.
- **Bundle structure stable.** Documented in ARCHITECTURE §7 (this section owns the design rationale; ARCHITECTURE owns the bundle layout). Carbon runs `lulc_bas` + `lulc_alt` in a single args file; UCM / UNA / UMH run each LULC separately (documented in the bundle README).
- **NatCap fixed alternatives export flood-only.** Compound LULC unavailable per OPEN_QUESTIONS; UCM / UNA / Carbon args are marked `available=False` with `reason: "NatCap did not ship a compound LULC for this fixed scenario; only flood is exported"` in `metadata.json → model_availability`.
- **Validation block ≠ per-card badge.** `metadata.json → validation` records each model with one of **two states**: `validated` (per-pixel parity measured — UCM, UNA, UMH, Carbon) or `methodology_aligned` (canonical method, no per-pixel parity check — UFR). This is the export-bundle's two-state taxonomy, distinct from §8's four-state per-card badge. Justification: downstream users opening a bundle want a yes/no answer per model, not the per-card four-state nuance which carries metric-specific tooltips that don't survive serialization.
- **Phase 3 verification: all five InVEST 3.19.0 urban models execute cleanly** on the SA baseline bundle (UCM ✓, UNA ✓, UFR ✓, Carbon ✓, UMH-depression ✓, UMH-anxiety ✓). Validation is on the *baseline*; scenario bundles share the same input schema, so the verification carries.

**Revisit if.** NatCap publishes a canonical UNA `population_group_radii_table` for SA equity stratification (would extend UNA args), or InVEST 3.20+ changes the args schema for any model.

**Code touchpoints.** `export_invest_bundle.py` (Streamlit-agnostic builder: `BundleSpec` dataclass + `build_invest_bundle(spec) → bytes`); `_build_invest_bundle_for_current_scenario` in `app.py` (caller); the sidebar two-step Prepare/Download flow. ARCHITECTURE §7 holds the bundle layout + `metadata.json` field list.

### 9.2 Exported raster nodata sentinel rule

**Decision.** Every raster the export bundle writes carries a `nodata` tag that matches the sentinel the array actually carries: NLCD × tree-reduced rasters → `nodata=-128`; compound LULC → `nodata=-1`; NDVI → `nodata=-1.0`.

**Why.** The prototype tolerates unmapped or sentinel lucodes — its CN aggregation filters `CN > 0` and silently drops them. **Canonical InVEST does NOT.** InVEST UFR's `_lu_to_cn_op` raises `ValueError` (with a misleading empty `[]` lucode list — a known display bug) when any non-nodata pixel maps to an all-zero CN row. So an exported raster whose declared nodata doesn't match its actual sentinel leaves "outside-boundary" pixels unmasked, which canonical InVEST then treats as live land-cover with no CN mapping, which crashes.

**Alternatives considered.**
- Force every nodata to a single value (e.g. 0) — fights the per-raster source encoding; 0 is a valid NDVI value and a valid compound lucode.
- Skip the explicit `nodata` tag and rely on InVEST's auto-detection — InVEST doesn't auto-detect from array content.
- Translate every export to the prototype's `NODATA = -128` — would require re-encoding the compound LULC (currently -1 per the source) and the NDVI (currently -1.0 per the source). Adds work; loses provenance.

**Consequences.** Surfaced by the D1 Phase 3 UFR failure-then-fix: initial export wrote the NLCD × tree-reduced raster with `nodata=0`, leaving 35,973 −128 pixels unmasked; InVEST raised on them. Fixed by writing `nodata=-128` for both NLCD × tree rasters; all five InVEST models then pass. **General truth surfaced by D1, not specific to UFR**: applies to every future raster added to the bundle.

**Revisit if.** Canonical InVEST ever silently absorbs sentinel-mismatched rasters (it doesn't currently; this rule guards against the failure mode).

**Code touchpoints.** Per-raster write sites in `export_invest_bundle.py`; the bundle's `metadata.json → raster_lineage` block (records each raster's source + sentinel).

---

## 10. UI communication decisions

### 10.1 Sign-convention — `temp_change_f` (positive = warmer, negative = cooler)

**Decision.** The cooling metric uses the universal physical ΔT convention: **`temp_change_f = T_after − T_before`, positive = WARMER, negative = cooler** (`= −old cooling_f`). Producer: `hm_to_temp_change_f(mean_hm)`, which negates the HM-index delta before scaling to °F.

**Why.** The old `positive = cooler` convention created a "negative cooling" oxymoron — most visibly in the neighborhood breakdown, where deviations from the city average rendered as negative "temperatures" that read like sub-zero absolute values. Two conventions also coexisted (the main card and the breakdown both said "positive = cooler" but the framing confused users). Adopting one physical ΔT convention everywhere removes the ambiguity.

**Alternatives considered.**
- Keep `positive = cooler` — the prior convention; produces the "negative cooling" oxymoron.
- Use absolute temperatures — would require a per-pixel `T_air` baseline the prototype doesn't compute (the prototype reports ΔT, not absolute T_air).
- Drop the signed display and only show "cooler" / "warmer" — loses the magnitude information users need to compare scenarios.

**Consequences.**
- **Users never see the signed number** — the display layer (`_fmt_temp_change`) always renders natural language: "X°F cooler" / "X°F warmer" / "No change". The signed convention is internal.
- **The optimizer is untouched.** `surrogate.py` searches and ranks on `mean_hm` (higher = more cooling), never on `cooling_f` / `temp_change_f`. The cooling-target slider `min_cool_f` is converted to `mean_hm` units before the surrogate sees it. The feared "maximize → minimize" inversion does not apply.
- **Per-tract breakdown uses the same convention.** Columns renamed to the `vs city avg` framing (each polygon's mean temperature relative to the city-wide baseline, positive = warmer); change column renders as natural-language text, color-coded green/red via a pandas Styler.
- **`cost_per_degf` divides cost by `−temp_change_f`** and is defined only when the scenario cools (`temp_change_f < 0`).

**Revisit if.** A future climate-impact metric needs a different sign convention (e.g. heatwave-day temperature anomaly — positive could mean "more heatwave"; same direction as current `temp_change_f`, so no conflict expected).

**Code touchpoints.** `hm_to_temp_change_f`; `_fmt_temp_change` (display); `compute_per_tract_summary` (per-tract Styler); the `cost_per_degf` denominator.

### 10.2 Signed metric cards — label-flip rule

**Decision.** Three "dollar/count" metric cards can render negative values for scenarios that make things worse (e.g. converting vegetated land to high-density development): **Preventable MH Cases**, **Avoided MH Costs**, and **Carbon Storage Value** (SA) / **Carbon Storage Change** (Ecological card). Each is hand-rolled inline (no shared metric-card renderer); the rule below is applied per-card:

- **Positive value** → benefit label, positive magnitude, green delta.
- **Negative value** → harm/loss label, magnitude shown as a positive number (no leading minus), red (`delta_color="inverse"`) delta.

Negative-case labels: "Preventable MH Cases" → "Additional MH Cases"; "Avoided MH Costs" → "Added MH Costs"; "Carbon Storage Value" → "Carbon Storage Loss" (SA) / "Avoided Carbon Cost" → "Added Carbon Cost" (MN); "Carbon Storage Change" → "Carbon Storage Loss" with red ↑ delta on the Ecological card.

**Why.** Signed numbers in metric cards read poorly — `-$1.2M` next to `+$3.5M` looks like an arithmetic problem, not a methodologically-correct harm signal. The label flip + sign flip presentation makes the direction explicit at glance.

**Alternatives considered.**
- Approach X: extend the shared `_delta_pill` helper with a polarity parameter — grows the helper's API for a 3-card edge case while the other always-positive cards (Flood Retention, Runoff, NDVI) stay uniform.
- Approach Y (chosen): hand-roll the three negative-capable cards inline; keep `_delta_pill` uniform for the always-positive cards.

**Consequences.** `_delta_pill` serves exactly three always-positive cards (Flood Retention, Runoff, NDVI); its uniformity is preserved without growing a polarity parameter. MN's carbon is always ≥ 0, so its loss labels are defensive and don't surface in practice; SA's four-pool stock model is the one that can go negative.

**Revisit if.** A fourth signed card is added — at which point Approach X (a shared signed-card renderer) might justify the API cost.

**Code touchpoints.** The three inline metric-card sites in `app.py` (Preventable MH Cases, Avoided MH Costs, Carbon Storage Value/Change); `_delta_pill` for the always-positive ones.

### 10.3 Metric labels — clarity / consistency rules

**Decision.** Display-only label cleanup applied uniformly across cards / charts / tooltips:

| Old | New | Why |
|---|---|---|
| "Flood Risk Reduction" (Ecological card) | "Flood Retention" | `flood_reduction = 100 − mean_CN` is a retention index, not a damage-curve risk metric. |
| "Flood Damage Avoided" (SA Economic card) | "Flood Volume Reduction" | NatCap's published SA flood framing is volume reduction, not damage avoidance (§6.5). |
| "Cost / °F Cooling" (cost-effectiveness) | "Cost / Citywide °F Cooling" | The °F is a city-average; the raw ratio looked absurdly large without the qualifier. |
| "Cost / Acre-Foot Prevented" | "Cost / Acre-Foot Runoff Prevented" | Clarity. |
| "Heat vulnerability" / "heat vulnerability proxy" (map legend + caption) | "Development-intensity heat proxy" | Accurate to the underlying NLCD-intensity proxy (23 > 22 > 21), not a real CDC/ATSDR HVI. |
| Carbon badge "Prototype" on SA | "Four-pool stock (NatCap framework)" | The single-rate-proxy framing was retired (§6.4); the badge needs to describe the new methodology, not a confidence tier. MN keeps "Prototype" (single-rate proxy is genuinely lower-confidence). |

**Why.** Each label was inaccurate in a way that would mislead a careful reader. None changes a metric computation; baselines are unaffected.

**Alternatives considered.** Leave the old labels in place — the inaccuracies aren't load-bearing; chose to fix anyway to stay precise.

**Consequences.** Doc-side rename cascades through REFERENCE, NATCAP_ALIGNMENT, and `app.py`'s methodology-tab pointer. No anchor links named `#flood-risk-reduction` existed, so nothing broke. The in-app changelog (`WHATS_NEW_ENTRIES`) does not carry these — they're clarity/consistency tweaks that don't clear the "would a returning user notice" bar.

**Revisit if.** A new metric label is added that needs the same precision audit.

**Code touchpoints.** The card / chart / tooltip render sites in `app.py`; `_CONFIDENCE_BADGES` carries the SA-carbon methodology descriptor.

### 10.3a `$`-escape rule — markdown-rendered elements only

**Decision.** Escape `$` as `\$` ONLY in markdown-rendered Streamlit elements (`st.markdown` / `st.write` / `st.caption` / `st.expander` labels / `st.subheader` / `st.radio` / `st.slider` / `st.button` labels / `help=` tooltips). Do NOT escape `$` in `st.metric` value, label, or delta arguments — `st.metric` renders as plain text, and `\$` in that context prints a literal backslash.

**Why.** Streamlit's markdown renderer supports LaTeX via paired `$…$`. An unescaped paired `$…$` inside a markdown-rendered string can silently flip into LaTeX math (the chars between the `$` get stripped and re-rendered). Escaping with `\$` defuses the pairing. But `st.metric` does NOT pass its arguments through the markdown renderer — the `\` is rendered verbatim, producing the literal `\$` the user sees on screen (caught in post-push eyeball on the NatCap Carbon Value card: "+\$82M" / delta "@ \$190/t").

**Rule (mnemonic).**

- `st.metric(value, …, delta=…)` — plain text. **Bare `$`. Never escape.**
- `st.markdown / write / caption / subheader / help=` — markdown. **`\$` for any literal `$`** (paired or not — paired-`$` is the bug, but a single bare `$` could also surprise a future edit that adds a second one nearby).
- DataFrame cells (`st.dataframe(df)`) — plain text. **Bare `$`. Never escape.**
- DataFrame column headers — typically plain (older Streamlit) but treat as plain. **Bare `$`.**

**Alternatives considered.**
- Always escape everywhere (safe but produces literal `\$` in metric / DataFrame cells — the bug).
- Never escape (paired-`$` LaTeX flip in markdown).
- A per-element lookup table with hard-coded escape rules per call (over-engineered; the renderer behavior is the canonical reference).

**Consequences.**
- The `verify_baselines.py` `$`-discipline static lint enforces this both ways: (a) no `\$` inside any `st.metric` value/label/delta arg; (b) no paired unescaped `$…$` in any `st.markdown` / `write` / `caption` string. Meta-test: a seeded violation MUST trip the check, otherwise the lint is green-light theatre.
- Card-value truncation fixes (NatCap fix #2 — 4-card row → 2×2) are independent of this rule but landed in the same commit because both were caught in the same eyeball pass.

**Revisit if.** Streamlit changes the renderer for `st.metric` or DataFrame cells to support markdown — the rule needs to invert for those surfaces. Until then, lock the current behavior.

**Code touchpoints.** The 4 `st.metric` un-escape sites in `_render_natcap_fixed_scenario_view` (Carbon Value baseline + alt, value + delta); the `$`-discipline lint cell in `verify_baselines.py`.

### 10.4 Sidebar order — configure-then-optimize workflow

**Decision.** Sidebar order: **City → Land Use Scenario → Conversion Mix (with Quick Start buttons) → Placement Strategy → Discover scenarios to validate → Implementation Costs → Advanced Settings.**

**Why.** Placement Strategy shapes the *current* scenario, so it belongs alongside the conversion mix, before the optimizer (an advanced, optional action). The previous order (Placement Strategy below the optimizer) inverted the intuitive workflow.

**Alternatives considered.**
- Keep the prior order — semantically inverted relative to the configure-then-optimize flow.
- Place Implementation Costs above Placement Strategy — costs are downstream of placement choices; they belong after.

**Consequences.** Purely visual move — `placement_strategy` / `use_heat_priority` are not referenced by the optimizer block, and both are still defined before the main-panel `results` computation, so no `st.session_state` init reordering was needed. The optimizer's entry text was reframed (§7.2).

**Revisit if.** A new sidebar control's logical position differs from its visual one.

**Code touchpoints.** Sidebar render order in `app.py`.

### 10.5 Ownership / eligibility filters are feasibility constraints, not engine inputs

**Decision.** Region Selection and the Eligible-land filter (the seven-class ownership taxonomy + vacant overlay + multi-class union via checkboxes) constrain *where* conversions can be placed — they do NOT enter the biophysical model equations. The locked sidebar caption next to the panel says exactly this: *"Ownership filters are feasibility constraints. They limit where conversions may be placed but do not change the biophysical model equations."*

**Why this framing matters.** A planner picking "City-owned land + vacant" is narrowing the candidate pool of pixels eligible to be converted; the per-pixel UCM / UNA / UFR / Carbon / UMH math runs identically on the resulting conversions. A region-clipped scenario produces a region-local NUMBER (a different aggregation scope) — not a region-local MODEL. Conflating the two would suggest the engine treats public-land conversions differently from private-land conversions, which it doesn't. The math is identical; what changes is the pixel set that gets converted.

**Subset-invariant consequence.** The "engine doesn't read the ownership raster" property is what keeps the 40/40 baseline snapshots byte-identical across every batch in the Finer Ownership Classes workstream. The mask is composed by the caller (`_build_ownership_mask` at the sidebar render site) and passed as the `selected_region_mask` arg to `evaluate_scenario`. The engine just consumes a boolean mask — it doesn't know which class the mask came from. See ARCHITECTURE.md §3 for the data flow + the subset-invariant contract.

**Design boundary.** This decision deliberately does NOT extend the engine to:
- Read per-class biophysical parameters (e.g. treat city-owned conversions differently from private-owned). The cooling / flood / carbon / nature-access / MH equations are pixel-physics, not ownership-conditioned; an ownership-conditioned biophysical parameter would invent a per-class effect that has no measured basis.
- Encode the filter as a per-pixel weighting rather than a hard mask. The placement contract is "this pixel is eligible or it isn't"; a soft-preference weighting would muddle the subset-invariant check and the eligibility-funnel arithmetic both of which assume a hard mask.

These are boundaries, not rejected-after-deliberation alternatives — they record where the engine's responsibility stops, not what was considered and discarded.

**Consequences.** Adding a new ownership class (the Batch 4 v2 city ∪ school union path; future region-by-class subset cells) is a UI + mask-helper change — never a math change. The honesty caption is what makes this property legible to the user; the subset-invariant assertion in `verify_baselines.py` (converted ⊆ eligible ∩ region ∩ ownership) is what enforces it.

**Code touchpoints.** Locked caption (`app.py` Eligible-land-filter panel); `_build_ownership_mask` + `_compose_eligible_filter_cfg` (single-source mask helpers); `evaluate_scenario(selected_region_mask=…)` (the engine's only filter input); subset-invariant matrix (`verify_baselines.py`).

### 10.6 Carbon stock vs flow — per-city framing + label choice

**Decision.** SA carbon is reported as a **stock change** (one-time, t CO2e), MN carbon as an **annual flow** (t CO2e/yr). The labels match the underlying methodology:

| | SA (`_CARBON_IS_STOCK = True`) | MN (`_CARBON_IS_STOCK = False`) |
|---|---|---|
| Quantity card label | "Carbon Storage Change" | "Carbon Sequestration" |
| Dollar card label | "Carbon Storage Value" (positive) / "Carbon Storage Loss" (negative) | "Avoided Carbon Cost" (positive) / "Added Carbon Cost" (negative) |
| Selected-region impact row | "Carbon Storage Change" | "Carbon Sequestration" |
| Compare-scenarios column | "Carbon Storage Change" / "Carbon Storage Value $ (derived)" | "Carbon Sequestration" / "Avoided Carbon Cost $/yr (derived)" |
| Unit | t CO2e | t CO2e/yr |
| Methodology | NatCap four-pool stock change (Vibrant Land convention; `c_above_arr` + `c_below_arr` + `c_soil_arr` + `c_dead_arr` — `_compute_carbon_four_pool`) | Per-cover-class annual sequestration rate (USDA NRCS / IPCC midpoints — `CARBON_SEQ_RATES`) |

**Why per-city, not unified.** SA's NatCap stack hands the prototype a four-pool stock framework directly; reframing it as an annual flow would invent a temporal scope NatCap doesn't claim. MN has no four-pool framework available; the per-cover-class annual rate proxy is what the prototype can compute. Forcing the two cities into a single label would either misrepresent SA's stock framing or invent a flow framing for SA that isn't supported by the data. Keeping the per-city distinction in labels is the honesty floor.

**Sign convention symmetry.** Both cities use the locked-positive-magnitude pattern (§10.2): a negative carbon outcome flips the label ("Storage Loss" / "Added Cost") and the delta arrow color so a precise positive number doesn't read as a benefit when the scenario is actually losing carbon (only SA's four-pool stock model can produce a negative outcome at present; MN's annual rates are non-negative).

**Alternatives considered.**
- Single canonical label per quantity ("Carbon" / "Carbon $") with a per-card tooltip explaining the per-city framing — loses the temporal-scope honesty at the headline level; users skim labels more than tooltips.
- Force both cities to a stock framing — requires inventing a stock arithmetic for MN that the per-cover-class annual rate doesn't support.
- Force both to a flow framing — requires inventing a temporal divisor for SA's four-pool stock that NatCap doesn't claim.

**Revisit if.** SA-style four-pool data becomes available for MN (would unify both cities on stock-change semantics), OR an MN-specific per-cover-class table calibrated to local species becomes available (would tighten the flow-side framing without changing the label).

**Code touchpoints.** `_CARBON_IS_STOCK` (set once after city-state aliasing in `app.py`); `_carbon_card_label` / `_carbon_dollar_label` (per-city branches); `_CS_CARBON_TONS_LABEL` / `_CS_CARBON_DOLLAR_LABEL` (compare-scenarios table constants); the SA NatCap fixed-scenario reference view's Carbon Storage Value branch; `_compute_carbon_four_pool` (SA stock math) vs `CARBON_SEQ_RATES` (MN flow math).

---

## 11. Deferred alternatives

This section holds **cross-cutting deferred approaches** not owned by any single decision above. Per-decision alternatives live in each entry's "Alternatives considered" field; this is the place for approaches that span multiple sections or sit outside the prototype's current model.

### 11.1 PLUS / CLUE / LCM — land-use simulation models

**Considered.** NatCap's project document lists three land-use change simulation models alongside the prototype's three-layer placement mask:

- **CLUE** (Conversion of Land Use and its Effects) — biophysical land-change modeler, established early 2000s, Java-based.
- **PLUS** (Patch-generating Land Use Simulation) — ML-based, recent, open-source from HPSCIL at China University of Geosciences; standalone C++ Qt application.
- **LCM** (Land Change Modeler) — proprietary, part of TerrSet.

**Why deferred.** These models answer a different question than the prototype is set up for. They project *what will happen* given historical drivers and trends — useful for "what does the AOI look like in 30 years if current trends continue?" The prototype asks *what should happen if planners intervene* — a different question that doesn't map cleanly onto status-quo projections.

That said, they're in NatCap's recommendation list because they're expected to add value. Future phases may incorporate one or more for: baseline-without-intervention projections (status-quo scenarios); learning placement patterns from historical land-use change; comparing planner interventions against business-as-usual.

**Specific operational concerns.** PLUS is a standalone C++ Qt application, not a Python library — integration would require subprocess execution or substantial reimplementation. CLUE is Java-based with similar deployment issues. LCM is proprietary, can't ship in an open-source prototype.

**Revisit if.** A status-quo projection becomes a prototype goal (e.g. for the "without-intervention" baseline a future climate-impact phase might need).

### 11.2 Wallpaper approach — interpretation uncertain

NatCap's project document lists "wallpaper" alongside the three-layer mask as a "simpler approach" to placement. The term has no standard land-use literature definition we could verify. Working interpretation: uniform tiled pattern across the AOI (every Nth pixel, repeating motif) rather than independent random selection. If that's right, the prototype currently does the latter, not wallpaper.

**To clarify with NatCap.** Tracked in NATCAP_COLLABORATION as a clarifying ask. Whether the prototype should pursue this as an option depends on the answer.

### 11.3 NatCap ROOT — multi-objective optimization

**Considered.** NatCap's ROOT (Restoration Opportunities Optimization Tool) is a linear-programming-based multi-objective optimization tool for spatial decision-making. It maximizes weighted sums of objectives (`max Σ wᵢ Vᵢₛₐ xₛₐ`) over spatial decision units (SDUs), producing true Pareto frontiers (production possibility frontiers) and agreement maps.

**What ROOT does that the prototype doesn't.**
- True LP-based Pareto optimization at the SDU level — guarantees mathematically optimal solutions on the feasibility frontier, not heuristic approximations.
- Agreement maps across weight configurations — visualizes which spatial decisions are robust to objective weighting choices.
- Cost-as-factor optimization — costs can enter the optimization as constraints or factors, not just as post-hoc ratios.
- Operates on rasterized factor layers without requiring a precomputed scenario grid.

**Why the prototype uses a surrogate-based optimizer instead.** ROOT is a desktop optimization tool designed for analyst workflows — runs take minutes to hours, results are produced for offline analysis. The prototype is an interactive dashboard where slider responses need to be millisecond-fast. A Random Forest surrogate over the four scenario sliders, trained on a precomputed scenario grid, enables interactive scenario exploration at the cost of giving up true Pareto optimization.

**"Not pursued" means different tool, not wrong tool.** ROOT is the right tool for analyst-driven offline optimization with strong guarantees. The prototype's surrogate is the right tool for interactive sandbox exploration. Adopting ROOT would mean re-architecting the prototype around a different user model — not a correctness fix to the current architecture. ARCHITECTURE §10's "Why not ROOT" carries the one-paragraph framing; this entry holds the deferred-approach rationale and the cross-document pointer.

**Revisit if.** A future workstream needs true LP-based Pareto frontiers or agreement maps — at which point ROOT becomes a candidate for an offline analyst-mode export, not an in-app replacement.

### 11.4 Stratified Impervious Siting (per-NLCD-intensity placement control)

**Considered.** A sidebar control exposing per-NLCD-class placement targeting — NLCD 21 (≥ 20 % impervious, open-space dominant), NLCD 22/23 (low–medium intensity), NLCD 24 (≥ 80 % impervious, depaving / high-intensity mitigation). The current stochastic placement step samples uniformly from the building/road-filtered NLCD 21–24 pool, treating all impervious-intensity classes as equivalent for siting; stratification would expose the choice to the user.

**Why considered.**
- Depaving-focused scenarios that target high-impervious pixels (NLCD 24) where the cooling / runoff leverage per converted acre is highest.
- "Neighborhood green-space" scenarios that target open-space-dominant pixels (NLCD 21).
- Empirical question — does stratified placement resolve the Nature Access saturation issue noted in REFERENCE.md (validate before claiming)?

**Why deferred.** Three open scoping questions:
1. UI shape — mutually-exclusive radio buttons vs multi-select vs per-tier weight sliders.
2. Whether to dynamically clamp the conversion-percentage slider max based on the selected tier's available acreage.
3. Empirical validation that stratification actually resolves the saturation question — must be measured, not claimed.

Optional micro-siting refinement via `scipy.ndimage.distance_transform_edt` against `BUILDINGS_RASTER` ("open lot" vs "private yard" at 15 m / 30 m distance thresholds) is a secondary refinement to scope separately.

**Framing constraint.** Strictly impervious-intensity stratification, not policy / ownership tiering — NLCD classes correlate with but do not equal ownership. The user-facing label must not imply parcel-level knowledge.

**Revisit if.** Phase-3+ region-selection work makes per-region placement controls the natural surface for stratification; the two features compose. Or a user-facing study question (NA saturation, depaving leverage) explicitly requires per-class targeting.

**Source.** Iterated proposal (Gemini-3 v3 after Claude critique) — v3 is the version to scope from. Originally captured in CLAUDE.md "Blocked / pending work" (pre-trim); absorbed here so the proposal has a durable home.

### 11.5 B2 — per-scenario Match / Diverged validation badges

**Considered.** The original B2 design: per-metric validation badges with ✓ NatCap match / × Diverged X% states, driven by a prototype value *computed for a NatCap fixed scenario* against NatCap's published value. The five-state badge taxonomy was ✓ NatCap match / × Diverged X% / ≈ Aligned method / Prototype / interim "NatCap published".

**Why deferred.** The Match / Diverged states require prototype reproduction of the six NatCap fixed alternative scenarios (FF_20ac / FF_40ac / FF_MAX / UA_20ac / UA_40ac / UA_MAX) for the two `natcap_published` metrics (`temp_change_f`, `carbon_tons_co2`). Both are compound-keyed (UCM and four-pool Carbon); NatCap's scenario rasters are flood-encoded (NLCD × tree). The compound LULC inputs for those alternatives weren't shipped — NatCap built them as unsaved pipeline intermediates (OPEN_QUESTIONS "Per-scenario compound LULC inputs"). The reproduction is gated; may never un-gate.

The B2 *revised* scope (§8.1) delivers the ungated core: the four-state badge taxonomy (no Match / Diverged); the fixed-scenario reference view; the cross-source comparison table; and the conservative-floor baseline reproduction posture.

**What landed (the conservative floor — §8.1).** Three-state badge taxonomy → four states (the "NatCap published value" green state survived; Match / Diverged dropped). Reference-view architecture (`_render_natcap_fixed_scenario_view` routed by a SA-only sidebar "Scenario source" radio, calling `st.stop()` before the Explorer panel renders). Curated non-CSV-card status map. Compound-gated cards listed in an explicit "not available for this NatCap scenario" section with a pointer to OPEN_QUESTIONS.

**Preserved card inventory + fixed-scenario classification** (reusable if Match / Diverged is rebuilt):

| Card | metric key | tier | NatCap CSV status | fixed-scenario classification |
|---|---|---|---|---|
| Flood Retention | `flood_reduction` | high | `aligned_method` | **computed** + reconcile |
| Temperature Change | `temp_change_f` | high | `natcap_published` | **published** (Δ) |
| Runoff Volume | `runoff_acre_feet` | high | — | computed (flood path) |
| Carbon Storage Change | `carbon_tons_co2` | four_pool / proto | `natcap_published` | **published** (Δ) |
| NDVI | `mean_ndvi` | prototype | — | **unavailable** |
| Nature Access | `nature_access_pct` | medium | `aligned_method` | **unavailable** (compound) |
| Preventable MH Cases | `preventable_mh_cases` | high | `aligned_method` | **unavailable** (compound / NDVI) |
| Avoided MH Costs | `avoided_mh_cost_usd` | high | pairs w/ MH `aligned_method` | **unavailable** |
| Food Production | `food_mln_lbs` | prototype | `prototype` | **unavailable** (lucode 998 ≠ 41; no ref) |
| Est. Implementation Cost | `total_cost_mln` | medium | — | **unavailable** (slider / mix artifact) |
| Flood Damage Avoided / Volume Reduction | `flood_damage_avoided_usd` / `flood_reduction` | medium | — | computed (SA → "Volume Reduction") + reconcile |
| Cooling Energy Savings | `cooling_energy_savings_usd` | medium | `aligned_method` | **unavailable** (UCM compound) |
| Carbon Storage Value $ | `carbon_value_usd` | medium | carbon × SC-CO2 → published-derived | **published** (derived) |
| Cost-effectiveness ratios | `ce[...]` | medium | — | **unavailable** (inputs unavailable) |

Net per fixed scenario: ~4–5 cards carry a value; ~11 are "not available." This justifies the dedicated-reference-view architecture (Option b2) over a guarded full 16-card grid: pervasive-guards would touch every card and inline delta-pill math, risking regression-prone code.

**Three open decisions for a rework** (preserved from the Phase 0 design, in case the deferral lifts):
1. Reference-view layout — **(b2) compact dedicated view [recommended]** vs (b1) full 16-card grid with ~11 "not available" cards (lots of dead space).
2. Confidence-badge replacement scope — **(i) validation badges on the fixed-view only, keep confidence badges on Explorer [recommended]** vs (ii) unified hand-mapped taxonomy everywhere. (ii) collapses high/medium and mislabels non-CSV cards (runoff, carbon-$ fall to "Prototype" via the "no row" rule though InVEST-derived), so needs a curated per-card map, not a raw CSV lookup.
3. Flood reconcile (B1 ~5-point CN gap, native 81.4 vs prototype baseline 76.54) — **(i) suppress the fixed-scenario flood delta, show "≈ invariant" [recommended]** (matches NatCap's finding, low-risk) vs (ii) re-derive the SA baseline flood through the native NLCD × tree path (more correct, but a methodology change with wider blast radius). The B2-revised scope adopted (i): the fixed-scenario flood card renders "≈ invariant (design-storm saturation, NatCap finding)" with a tooltip explaining the derivation gap.

**Revisit if.** Compound LULC inputs for the NatCap fixed alternatives arrive (OPEN_QUESTIONS) — the gating constraint lifts and the Match / Diverged states become re-implementable. The likely shape would be a **reworked, smaller surface** (baseline reproduction + a NatCap reference comparison table) rather than the original per-card per-scenario badge design.

### 11.6 Topics not yet documented

Sections that might land here when the relevant work happens. Listed so future sessions know this doc is the right home:

- UCM cooling parameters (UHI_MAX_C, energy table, HMI vs energy aggregation)
- NDVI source — synthetic proxy vs satellite-derived (AlphaEarth Foundations; feasibility research at [ALPHAEARTH_FEASIBILITY.md](https://github.com/dkwtestacct/ecosystem-explorer/blob/main/ALPHAEARTH_FEASIBILITY.md))
- Population data — Census 2020 block vs ACS block-group
- Surrogate model architecture and hyperparameters
- Real CDC / ATSDR Heat Vulnerability Index integration (replacing the development-intensity proxy)
- Mental health parameters (RR per 0.1 NDVI, cost-of-illness)
