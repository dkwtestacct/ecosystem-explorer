# Ecosystem Explorer — Reference

**Audience:** External
**Status:** Current methodology reference
**Use this for:** Understanding dashboard metrics, data sources, validation badges, and limitations
**Do not use this for:** Internal task tracking, collaboration history, or implementation decisions
**Source of truth for:** What the dashboard numbers mean

---

## 1. What this tool is

Ecosystem Explorer validates its modeling engine against canonical InVEST, displays NatCap project reference values where available, and lets users explore additional scenarios beyond the fixed project set — then export promising ones back to canonical InVEST for full validation.

The dashboard simulates how reallocating developed urban land among green infrastructure (wetlands), food forest, and high-density development affects a portfolio of ecological, social, and economic outcomes: flood retention, urban cooling, food production, nature access, mental-health proxy outcomes, and carbon. The biophysical engine reimplements five InVEST urban models (Urban Cooling, Urban Flood Risk, Urban Nature Access, Urban Mental Health, Carbon) in numpy, validated against canonical `natcap.invest.*.execute()` where comparable inputs exist (see §8 for the per-model status).

**This tool is designed for:**

- Comparing alternative land-use allocation strategies
- Exploring tradeoffs across multiple ecosystem services
- Identifying candidate scenarios for deeper analysis
- Supporting early-stage planning conversations and stakeholder discussions

For what it is *not* designed for and what to do before relying on the numbers, see §10.

**Multi-city support.** Two cities are currently selectable in the UI — Minneapolis, MN (downtown, ~123 km², InVEST sample buildings with per-type codes) and San Antonio, TX (Bexar County bbox, ~3,060 km², ~1.9M residents). A third configuration — Minneapolis Full, MN — is implemented in the codebase but hidden from the selector pending per-building-type data for the expanded area. Each city has its own configuration entry declaring input paths, CRS, and biophysical parameters; switching cities is a sidebar interaction with separate cached state. See §7 for per-city detail.

---

## 2. How to read the dashboard

### Metrics run live; the surrogate is a separate, opt-in tool

Metrics on the metric cards run live on every slider change — your sidebar settings feed straight into the biophysical engine each interaction. In *High resolution* mode some supporting raster aggregates are precomputed at startup to keep slider response interactive, but the live re-evaluation still runs. The Random Forest surrogate is a separate optimization tool — only used when you explicitly click **Optimize** under *Discover scenarios*. For the deep mechanics — model-quality modes, lookup-table internals, surrogate training, cache-invalidation discipline — see `docs/internal/ARCHITECTURE.md` (At a glance + Layer 1/2/3).

### Tabs

| Tab | What you see |
|---|---|
| **Scenario** | Sidebar slider state → 13 metric cards in three categories (🌿 Ecological / 👥 Human & Social / 💵 Economic) plus a 📊 Cost Effectiveness sub-section. Each card carries a validation badge (see §4). Below the cards: a plain-language scenario summary, a Baseline-vs-Scenario comparison expander, and small Flood / Cooling / Food bar charts. |
| **Tradeoffs** | A Plotly tradeoff chart with Flood Retention on X and Heat Mitigation Index on Y, populated with current scenario (purple star), saved scenarios (purple circles, sized by food production), optimizer suggestions (orange diamonds with 10th/90th percentile error bars from the surrogate's tree ensemble — see §8), Pareto frontier (gold), and per-city reference benchmarks. Also: the cross-source Compare scenarios table (NatCap-published + current + saved rows side by side with Area + Ownership columns) and a `Download scenario summary (CSV)` button. **Best Scenarios by Goal** (the citywide precomputed library) renders only when no Region Selection and no Eligible-land filter is active — under either filter it hides with a one-line note explaining why (the rankings are citywide and don't reflect your filter). |
| **Map View** | Pixel-level raster showing which developed pixels changed and to what cover (teal = Green Infrastructure, green = Food Forest, red = High Density). When a region is selected, this tab also surfaces the **interactive region-picker** map (click polygons to add or remove from the selection — multi-select via shift/ctrl) and a five-row **eligibility funnel** that traces convertible-pool dropouts (see §2 *Eligibility funnel*). |

### Sidebar controls

| Control | What it does |
|---|---|
| **City** | Selectbox; loads the active city's data, CRS, and biophysical tables. |
| **% of Developed Land to Convert** | 0–50 slider. Determines how many developed pixels (NLCD 21–24) are eligible — minus buildings and roads — get converted. |
| **Green Infrastructure / Food Forest / High Density %** | Three number inputs (0–100, step 5) that must sum to 100. Splits the converted pixels into the three target classes. |
| **Implementation Cost Sliders** | Per-acre cost sliders for GI / FF / HD. Scales the Implementation Cost metric. Does not affect biophysical outcomes. |
| **Spatial Priority: Target High Heat-Exposure Areas** | Toggle. When on, converted pixels are sampled with higher probability in NLCD 23 > 22 > 21 areas (development-intensity proxy). |
| **Placement strategy** | Radio picker over five strategies — see §5. |
| **Example Scenario buttons** | Three one-click presets: Green Infrastructure (100 % GI at 10 % converted), Food Forest (100 % FF), High Density (100 % HD — control case). |
| **Region Selection** | "Apply changes within" radio (Entire analysis area / Selected regions) + a region-layer dropdown + a multiselect of region labels. Picking one or more polygons constrains conversions to inside them. Per-pixel engine is the same validated math; only the WHERE narrows. SA layers: council districts, Bexar tracts. MN layer: downtown tracts. See §2 *Region selection + region-local metrics* below. |
| **Eligible land filter** (SA only) | Always-on exclusions (buildings / roads / existing natural land) plus five finer ownership classes as multi-select checkboxes — `City`, `County`, `State or federal`, `School district (K-12 public)`, `College or university` — plus a `Limit to vacant parcels only` overlay and three quick-set buttons (`Public` / `Vacant + Public` / `Clear`). Mask = union of checked classes (∩ vacant if toggled). The `Public` quick-set = city + county + state-federal + school; `university` is intentionally OUT of the public rollup because that bucket mixes public (UT / A&M / Alamo CCD) and private (Trinity, St. Mary's, OLLU) institutions. **Rule-derived caveat** (also shown as a sidebar caption + as the `ownership_rule_derived` entry in exported metadata.json): classes are parsed from BCAD owner names + exemption codes, not validated against a title registry; `school` matches `ISD` / `SCHOOL DISTRICT` only (charters and private K-12 fall through to `private`); `university` mixes public + private. Planning screen, not verified ownership. |
| **Discover scenarios** *(no filter active)* | Card titled *Discover scenarios* with mode label **Citywide surrogate search**. Sliders for minimum flood / cooling / food / carbon plus a runoff cap, then the **Optimize** button that scores the citywide surrogate over ~10,000 random `(pct, GI%, FF%)` candidates and returns up to 5 suggestions ranked by a balanced score (header: *Suggested scenarios*). Predicted values, not engine outputs — Apply a suggestion to run the full engine. The orange P10/P90 bands on the Tradeoff chart and in the optimizer table reflect the surrogate's per-tree disagreement (see §8). |
| **Discover scenarios** *(region or ownership filter active)* | Same card, mode label flips to **Selected-area full-engine search**. Five **weight sliders** (Cooling / Flood reduction / Carbon / Food / Cost) replace the min-target sliders, and the **Optimize** button runs a two-stage search: a fast surrogate (Phase-0.5-validated Fast config — 90 recipes, 100 trees) shortlists ~40 Pareto-efficient candidates, then the full engine evaluates each shortlisted recipe on the active region∩ownership mask. The 5 returned mixes carry **engine-true region-local values** — not predictions, no P10/P90 bands. Header above the table reads *"Best tested mixes for selected area"*, framing them as the best among the candidates the engine actually tested rather than a global optimum (the shortlist may not be exhaustive). Apply on one of the 5 sets the sliders, reruns the engine on the active mask, and flips provenance to **"Engine-verified — region-optimized"** (distinct from the citywide *"Citywide surrogate suggestion — engine-evaluated on apply"*). See §8. |
| **Export for InVEST** | (San Antonio only) packages the active scenario as a runnable canonical InVEST 3.19.0 input bundle. See §8 for the bundle structure. |
| **⚙️ Advanced Settings** | Carbon-rate sliders (MN only — SA reads NatCap's four-pool table directly) and the model-quality mode radio. |

### Region selection + region-local metrics

When you pick one or more region polygons in the sidebar, conversions are placed only inside that mask (intersected with the convertible pool, defined as developed land minus buildings, roads, and existing natural land). The Scenario tab gains a **Selected-region impact** table beneath the citywide metric cards: each row pairs the per-metric region-clipped value with the citywide value so the two are honest about each other.

Region-local readings inherit per-metric validation badges from the citywide cards — region-local doesn't change the engine, only the aggregation scope. Two locked caveats:

- **Flood routing.** The flood retention metric is a closed-form SCS-CN volume scaled to the region's developed area, not routed hydrology. Regional CN means legitimately differ from citywide ones.
- **Reach effects (UCM / UNA / UMH).** The three reach models (~600 m for cooling, ~800 m for nature access, ~300 m for mental health) have spillover at the region boundary — effects produced by in-region conversions that propagate just outside the boundary are reflected in the citywide column but NOT in the region-local column. Documented in the export bundle as the `region_local_spillover_reach_models` divergence.

Reconciliation contract: for every decomposable metric, computing the region-local value over the entire AOI must equal citywide (machine-checked by `verify_baselines.py`). That's how the region-local code path stays trustworthy.

### Eligibility funnel

On the Map View tab, when a region is selected the panel shows a small five-row funnel that traces where the convertible pool drops out at each step:

```
Selected area:                       e.g. 12,346 acres
Developed land:                       2,876 acres
After roads / buildings / existing nature: 337 acres
After ownership filter:                  34 acres       (only when ownership active)
Converted:                                34 acres
```

When ownership isn't active the "After ownership filter" row is omitted. Acres come from the same record fields the metric cards use; numbers reconcile to the audit expander and the comparison-table Area / Ownership columns by construction.

### Saved scenarios, audit expander, and CSV export

- **Save this scenario** appends the current scenario's record to a per-session list. **Saved scenarios are session-only** (lost on page refresh or server restart) — the in-app caption next to the saved list and at the export site says so explicitly. There is no disk persistence and no "load from file" path.
- **Scenario audit** is a collapsed expander under the provenance header on the Scenario tab. It opens with a prose sentence (what was converted, where, under what ownership, with which strategy) and then a 9-row field list: Source · Area · Ownership · Placement · Seed · Eligible acres · Converted acres · Validation · Export schema. Every field reads the underlying record directly — no recomputation.
- **Download scenario summary (CSV)** is the durable export. One row per scenario (current + every saved scenario for the active city); ~60 columns covering identity + provenance + region + ownership (including a multi-class `ownership_classes` column and a `ownership_vacant_only` boolean) + conversion mix + placement + seed + schema_version + citywide + region-local metrics. NatCap reference rows are intentionally excluded — they don't carry a full record. The CSV is what survives the page refresh.

### Scenario summary

Below the metric cards: *"This scenario converts X% of developed land, allocating Y% to green infrastructure, Z% to food forest, and W% to high-density development, using [placement strategy label]."*

### Baseline-vs-Scenario comparison + Bar charts

The Scenario tab's `📊 Baseline vs Scenario Comparison` expander (collapsed by default) is a single scannable table of all primary metrics with color-coded Change column. The three side-by-side bar charts directly below the comparison plot Flood Risk / Urban Cooling / Food Production against the unmodified baseline.

---

## 3. Scenario sources and provenance

Every scenario on the dashboard carries one of four **sources**, surfaced as a prominent header above the metric cards. The header tells you whose scenario this is and how to interpret its numbers.

| Source (rendered text) | What it means |
|---|---|
| **Baseline** | The unmodified land cover for the active city. The engine has been validated against canonical InVEST per-pixel where comparable inputs exist; absolute NatCap citywide figures are not independently reproduced. |
| **NatCap published reference** | A NatCap fixed-project scenario — the dashboard displays NatCap's own published number from the project's results file. We surface NatCap's value; we do not independently reproduce it (the exact scenario raster / aggregation path is not available). |
| **Explorer-generated** | A scenario you constructed with the sidebar sliders. Computed by the canonical-engine-verified biophysical models; not a NatCap-published scenario. |
| **Surrogate-suggested** | A scenario suggested by the optimizer and then **Applied** to the sliders — at that point the displayed metric cards reflect a full-raster evaluation by the canonical-engine-verified models (not a surrogate prediction). Treat as an exploratory candidate worth deeper validation. |

Per-card validation badges (§4) carry a finer-grained signal about each *individual metric* in the active scenario.

---

## 4. Validation badges

The dashboard renders **two distinct validation surfaces**. They serve different questions; they coexist by design.

- **The scenario provenance header** (§3 above) answers: *whose scenario is this?* — at the scenario level.
- **The per-card validation badge** (this section) answers: *how trustworthy is this specific number on this card, right now?* — at the metric × scenario context level.

### The per-card validation badge — four states

Each metric card displays one of four badges as an inline caption under its value:

| Rendered text | Color | Fires when |
|---|---|---|
| **`NatCap published value`** | green | A `natcap_published`-class metric × the fixed-scenario reference view *only*. The card displays NatCap's own number directly from the reference outputs file. *Not a reproduction claim* — we surface NatCap's figure. |
| **`≈ NatCap method`** | blue | A `natcap_published`-class metric × any other scenario context (Baseline / Explorer-generated / Surrogate-suggested). The displayed value is the prototype's own computation; the methodology is aligned with NatCap's. Tooltip is metric-aware (e.g. temperature cites measured per-pixel parity; carbon cites four-pool methodology adoption). |
| **`≈ Aligned method`** | blue | An `aligned_method` metric (canonical InVEST methodology with no directly-comparable NatCap citywide reference) in any context. |
| **`Prototype`** | gray | A `prototype` metric (no canonical InVEST analog) in any context. |

**Context-switch rule.** `NatCap published value` only fires when the dashboard is in the fixed-scenario reference view *and* a NatCap published number exists for that metric. In every other scenario context (Baseline / Explorer-generated / Surrogate-suggested), a `natcap_published`-class metric reads `≈ NatCap method` because the displayed value is the prototype's own computation, not NatCap's published number.

Where each metric falls on these four states — and the underlying evidence (measured MAE against canonical, methodology adoption without per-pixel parity, etc.) — lives in §6 alongside each metric's mini-template. §4 documents only what the badges *mean*.

---

## 5. Land-use scenarios

This section covers how scenario conversions are constructed: which pixels are eligible, how the target land cover is chosen, and the five placement strategies for picking which eligible pixels get converted.

### Land-use alignment

Per-city detail on the baseline LULC raster, the land-use code system, and how scenario conversions produce valid target lucodes within each system.

#### Minneapolis

- **Baseline LULC:** NLCD 2021 (`data/cooling/land_use_2021.tif`, byte-identical to the InVEST UNA sample LULC).
- **Code system:** NLCD lucodes.
- **Biophysical tables:** NLCD-keyed (UCM, UNA, CN biophysical, Carbon per-class annual rates).
- **Scenario conversions:** Direct NLCD mapping. Food forest → NLCD 41 (deciduous forest proxy). Green infrastructure → NLCD 90 (woody wetlands proxy). High density → NLCD 24 (developed, high intensity). No fallback logic needed — all targets are valid NLCD lucodes.

#### San Antonio

- **Baseline LULC:** NatCap's compound NLCD × NLUD × tree-canopy framework (`data/sa/flood/land_use_compound_sa.tif`). Source is NatCap's 2024 NASA Urban project compound LULC, reprojected EPSG:3857 → EPSG:5070 + nearest-neighbor resampled to 30 m.
- **Code system:** Compound lucodes — the Cartesian product of 16 NLCD codes × 31 NLUD classes × 4 tree-canopy levels = 1,984 distinct lucodes. Joined to NLCD / NLUD / tree-canopy attributes via `lulc_crosswalk.csv`.
- **Biophysical tables:** Compound-keyed for UCM, UNA, and Carbon. CN biophysical for flood is NLCD-keyed and reduces compound → NLCD via the crosswalk.
- **Scenario conversions:** Preserve each pixel's NLUD and tree-canopy context where possible; change only the NLCD component. A "Developed Med Intensity × Commercial × low canopy" pixel being converted to Food Forest looks up "NLCD 41 × Commercial × low canopy" in the crosswalk and gets the matching compound lucode if one exists. Crosswalk lookups prefer rows tagged `is_realistic_to_create=yes`.
- **Fallback:** If no matching compound lucode exists for a given (NLUD, tree-canopy) tuple, the conversion falls back to a documented default target lucode: `DEFAULT_FF_LUCODE = 1310` (Deciduous Forest × Timber × medium canopy), `DEFAULT_GI_LUCODE = 122` (Woody Wetlands × Wetland × medium canopy), `DEFAULT_HD_LUCODE = 341` (Developed High Intensity × Residential × low canopy). The default lucodes are real, well-characterized rows with valid values in every compound biophysical table. The `evaluate_scenario` result carries per-target fallback-pixel counts, surfaced in the SA dashboard's "Conversion fidelity" panel (Assumptions and limitations expander).

#### Closing

InVEST models are placement-agnostic — they evaluate user-provided LULC rasters. Ecosystem Explorer's placement strategies and conversion mechanism produce those rasters; InVEST does not prescribe either. Per-city land-use fidelity therefore reflects per-city data availability rather than any methodology choice imposed by InVEST.

### Placement strategies

The convertible pool defines *which pixels are eligible* for conversion; the placement strategy defines *which of those eligible pixels actually get converted* for a given scenario. The app provides five strategies, exposed in the sidebar as a radio picker. The default is uniform random sampling.

The strategies are an app feature, not an InVEST methodology requirement — InVEST's urban models are placement-agnostic and accept whatever LULC raster the user provides. The strategies offer faster scenario exploration of "what if we targeted high-CN pixels?" or "what if we prioritized underserved areas?" without manually constructing per-pixel LULC alternatives. Each focused strategy's suitability surface is grounded in a canonical InVEST quantity. See `docs/research/INVEST_PLACEMENT.md` for the underlying research and `docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md` for measured per-strategy effects.

| Strategy | Suitability formula | Rationale |
|---|---|---|
| Random placement | Uniform | Default — no spatial bias, useful for average-case scenario exploration. |
| Prioritize flood-prone areas | Per-pixel runoff `Q_{p,i}` from the SCS-CN equation at the design storm (canonical InVEST UFR `Q_mm.tif`) | Pixels with high runoff have the most potential benefit from greening — greening lowers CN and so lowers Q. |
| Prioritize hot areas near buildings | `(1 − baseline_HMI) × (1 / (1 + distance_to_building_px))` | Hot pixels (low canonical HMI) near buildings, with real distance-to-building from the buildings raster. |
| Prioritize areas with unmet nature demand | `max(0, urban_nature_demand − urban_nature_supply_percapita)` per pixel (canonical InVEST UNA per-capita supply deficit) | Pixels where residents lack the per-city demand standard. |
| Balanced approach | Equal-weighted normalized combination of the three above | App-specific heuristic — no InVEST analog. For true multi-objective optimization, NatCap's ROOT tool implements weighted-sum LP optimization with production possibility frontier outputs. |

**Honest caveats.**

- The balanced strategy's equal weighting is a default, not a derivation. Per-component normalization to sum-1 before averaging is mathematically defensible; the equal weights are subjective.
- No InVEST parity claim on the placement strategies themselves — no InVEST model prescribes *where* to site interventions. Each focused strategy is grounded in a canonical InVEST quantity (`Q_{p,i}` for flood, HMI for cooling, `urban_nature_supply_percapita` for undersupply); the strategy's choice of *what to optimize over* is an app heuristic.
- Strategy effect sizes are measured. Empirical per-city, per-metric strategy effects under the current formulas live in `docs/research/PLACEMENT_STRATEGY_DIAGNOSTIC.md`.

---

## 6. Metrics

Each metric card uses the same mini-template:

- **What it shows** — plain English.
- **How it is computed** — short method + data source.
- **Units** — what the number actually is.
- **Validation status** — which §4 badge fires here, plus the per-metric evidence (measured per-pixel parity? methodology adoption? canonical method with no NatCap comparable?).
- **Main caveat** — the single most important honest line.

The dashboard groups 13 cards under three categories. Below are the metrics in the same order the dashboard renders them.

### 🌿 Ecological

#### Flood Retention

- **What it shows** — A unitless index of runoff potential derived from the area-weighted average Curve Number. Higher is better.
- **How it is computed** — `100 − mean_CN`, where `mean_CN` is the area-weighted CN derived from the per-city CN biophysical table (`UFR_biophysical_table_<city>.csv`) by land cover × soil group, over the LULC raster.
- **Units** — Index (0–100).
- **Validation status** — `≈ Aligned method`. Canonical SCS-CN method. The card's metric is `100 − mean_CN`, monotone with InVEST UFR's per-watershed `rnf_rt_idx = 1 − Q/P` but computed differently — the app inverts the CN average rather than the post-storm runoff. Direction is consistent; the scale differs.
- **Main caveat** — Not a direct percentage reduction in runoff volume. For SA specifically, the index is **nearly scenario-invariant** at the Bexar-County dashboard scale — developed pixels are a small share of the bbox, so total-metric movement is small even when per-pixel greening is effective. See §7 (SA).

#### Temperature Change

- **What it shows** — Approximate air-temperature difference vs the unmodified baseline, in degrees Fahrenheit (e.g. *"1.2 °F cooler"* / *"0.8 °F warmer"* / *"No change"* below the 0.1 °F display threshold).
- **How it is computed** — Per-pixel Cooling Capacity (`CC = 0.6·shade + 0.2·albedo + 0.2·ETI`, where `ETI = Kc × ET / max(ET_in_AOI)`), then the canonical Heat Mitigation Index `HMI = max(CC_local, CC_park)` — where `CC_park` is the exponentially distance-weighted CC from green areas ≥ 2 ha within `d_cool = 450 m`. Temperature change vs baseline = `(mean(HMI_scenario) − mean(HMI_baseline)) × UHI_MAX_C × 1.8`. Data sources: per-city `biophysical_table_urban_cooling_<city>.csv` (shade, Kc, albedo per lucode) and per-city annual ET raster. SA's cooling biophysical table is tuned for Köppen BSh (hot semi-arid) — see §7 (SA).
- **Units** — °F (vs baseline).
- **Validation status** — `≈ NatCap method`. **Measured per-pixel parity:** the HMI raster matches `natcap.invest.urban_cooling_model.execute()` at MAE = 0.0000, Pearson r = 1.0000 on the MN baseline (`compare_ucm_invest.py`). One remaining UCM divergence affects only Cooling Energy Savings (see §8 for the gap), not this Temperature Change card, which is a direct function of the canonical HMI raster.
- **Main caveat** — Treat the °F output as ±2 °F at best: wind, humidity, urban geometry, and anthropogenic heat are not modelled. `UHI_MAX_C` is from the InVEST args JSON for MN; SA uses an interim estimate pending a SA-calibrated args run.

##### Official InVEST alignment — UCM

The app implements the canonical UCM HMI algorithm — `HMI = max(CC_local, CC_park)` with the 2-hectare park-area threshold and exponential decay over `d_cool = 450 m`, a faithful port of `urban_cooling_model.execute`'s `mask_cc_green_areas_op` → exponential-decay convolution → `hm_op` chain. The implementation is in `_compute_hmi_raster` (convolution via `scipy.signal.fftconvolve` with InVEST-canonical edge correction). Validated at MAE = 0.0000, Pearson r = 1.0000 vs `natcap.invest.urban_cooling_model.execute()` on the MN baseline.

The remaining divergence is in *Cooling Energy Savings*, not Temperature Change: InVEST samples T_air per *building* over the 600 m `t_air_average_radius` before applying the consumption rate; the app applies the formula per *pixel*. This affects dollar magnitudes but not the HMI-derived Temperature Change.

**Reference:** [InVEST UCM User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html)

#### Runoff Volume

- **What it shows** — Total stormwater runoff volume under the scenario for the design storm, across all developed land. Lower is better.
- **How it is computed** — Standard USDA-SCS Curve Number runoff. Per pixel: `S = 1000/CN − 10`, `Ia = 0.2·S`, runoff depth `R = (P − Ia)² / (P − Ia + S)`. Volume `V_acre-ft = (R/12) × developed_acres`. Design storm `P` is **per-city, NatCap-canonical**: Minneapolis uses 3.94″ (100 mm, NatCap MN-project `args.json`); San Antonio uses 6.18″ (157 mm, NatCap SA README).
- **Units** — acre-feet.
- **Validation status** — `≈ Aligned method`. Canonical SCS-CN. No NatCap citywide reference for direct comparison.
- **Main caveat** — Each city's design storm reflects its NatCap-project canonical depth; results for larger storms will differ. The card delta sub-label uses `±N ac-ft vs baseline` with green ↓ (reduction is better) and red ↑.

#### Carbon Sequestration (MN) / Carbon Storage Change (SA)

- **What it shows** — **MN:** annual CO2e sequestration from newly converted pixels only (t CO2e/yr). **SA:** one-time stock change in landscape carbon storage from the LULC delta (t CO2). The card label, units, and delta string branch on the per-city framing.
- **How it is computed** — **MN:** `n_FF × 0.222 ac × 3.5 + n_GI × 0.222 ac × 2.0 + n_HD × 0.222 ac × 0.0` t CO2e/yr (provisional USDA NRCS / IPCC regional rates). **SA:** four-pool stock change via NatCap's compound `carbon__nlcd_nlud_tree.csv` (1,984 rows × four pools: above-ground biomass, below-ground biomass, soil organic matter, dead organic matter). Per-pixel `(scen_total − base_total) × pixel_area_ha × 44/12` summed and reported in t CO2. The dollar metric multiplies by `EPA_SOCIAL_COST_CARBON = $190/ton CO2e` (EPA 2023; flat per-ton rather than InVEST's time-discounted NPV — matches the methodology in NatCap's 2023 Vibrant Land report, which also doesn't enable NPV).
- **Units** — **MN:** t CO2e/yr. **SA:** t CO2 (one-time stock change).
- **Validation status** — **SA:** `≈ NatCap method`. **Method-adoption without per-pixel parity:** SA Carbon uses InVEST's canonical four-pool stock framework via NatCap's compound table — aligned with the methodology in NatCap's 2023 Vibrant Land report (Guerry et al.) for the same SA project — but there is no measured per-pixel comparison against `natcap.invest.carbon.execute()`. **MN:** `Prototype`. No NatCap MN four-pool bundle is available; the per-cover-class annual rate is provisional.
- **Main caveat** — The InVEST guide notes that the canonical model "assumes that none of the LULC types in the landscape are gaining or losing carbon over time" — both prototype framings honor this assumption for the LULC classes they cover. SA values for moderate-conversion scenarios are within plausible order-of-magnitude bounds vs Vibrant Land's citywide reference; MN values are directional only — provisional regional rates, not locally calibrated.

#### NDVI

- **What it shows** — Mean Normalized Difference Vegetation Index across all valid pixels (0–1, higher = denser vegetation).
- **How it is computed** — NDVI is **assigned per NLCD code as a synthetic proxy** (woody wetlands 0.70, deciduous forest 0.75, high-density 0.10, NLCD 23 0.15, NLCD 22 0.20, NLCD 21 0.30, other developed 0.25, other natural 0.60). The card reports the area-weighted mean.
- **Units** — Index (0–1).
- **Validation status** — `Prototype`. Not satellite-derived. This is a stand-in until a real Sentinel-2 / Landsat NDVI raster is integrated.
- **Main caveat** — Treat as directional only. The same synthetic NDVI is the input to the Mental Health metrics — replacing it with satellite NDVI would change both the NDVI card and the MH cards.

### 👥 Human & Social

#### Nature Access

- **What it shows** — Share of the *modelable-extent* population whose per-capita nature supply meets the per-city demand standard. Rendered as `pct_pop_supply_ge_demand`.
- **How it is computed** — Numpy re-implementation of `natcap.invest.urban_nature_access` (Two-Step Floating Catchment Area). Per-pixel urban-nature area = `urban_nature` proportion × 900 m²; the population raster is convolved with the search kernel; the R_j ratio = nature area ÷ decay-weighted population; a second convolution of R_j yields per-pixel `urban_nature_supply_percapita`; the headline is the population-weighted share of pixels where supply ≥ demand. Parameters are **per-city** (see §7). The `urban_nature` proportions come from the per-city UNA biophysical table.
- **Units** — Percent of modelable-extent population.
- **Validation status** — `≈ NatCap method`. **Measured per-pixel parity:** the numpy implementation matches `natcap.invest.urban_nature_access.execute()` at MAE 0.0234 m²/person, Pearson r = 1.000000 on the MN baseline (`validation/compare_una_invest.py`). The aggregate headline `pct_pop_supply_ge_demand` is identical at 46.86 % on the same baseline.
- **Main caveat** — The headline reports the share of the **modelable-extent population** (residents on pixels with valid land cover), not the whole city — InVEST 2SFCA cannot model per-capita supply for residents on cooling-LULC nodata pixels. The card tooltip frames this denominator. The search kernel is Euclidean — no street network, no barriers, no slope.

##### Official InVEST alignment — UNA

InVEST UNA implements a Two-Step Floating Catchment Area (2SFCA) method: Step 1 computes a nature-to-population ratio for each nature pixel (nature area ÷ decay-weighted population within the search radius), Step 2 sums these ratios for each population pixel to produce per-capita supply (m²/person). Supply is then compared against a per-capita demand standard to flag whether each resident is adequately served. The app implements this canonically as a numpy port — `calculate_nature_access` — matching `natcap.invest.urban_nature_access` at MAE 0.0234 m²/person, Pearson r = 1.000000.

**Reference:** [InVEST UNA User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_nature_access.html)

#### Children's Nature Access

- **What it measures** — Share of the *modelable-extent under-18 population* whose per-capita urban-nature supply meets the per-city demand standard. The child-facing counterpart to Nature Access.
- **How it is computed** — Identical 2SFCA pipeline and *adequate* mask as Nature Access. The supply/demand calculation stays on **total** population (the InVEST UNA convention); only the access **share** is reweighted by under-18 population. A pixel is "adequately served" by the same rule as the adult metric — the headline diverges only when the residential distribution of under-18s differs from the total (e.g. school-zone interventions can lift this without moving the citywide adult metric). Returns `children_nature_access_pct` and `children_with_nature_access`; region-local reports both. Reads "—" where no child-population raster is configured (MN + SA configured; Mpls Full unset).
- **Validation status** — `≈ Aligned method`. Inherits Nature Access's canonical-InVEST 2SFCA validation (MAE 0.0234 m²/person, MN baseline); the child reweight touches only the final share, so no separate per-pixel parity claim. Not a surrogate target — computed deterministically (like UMH), so it carries no P10/P90 band.
- **Source** — US Census 2020 PL 94-171, block-level. Under-18 = `P1_001N − P3_001N` (total − 18-and-over), uniform-spread to the NLCD grid by the same method as the total-population raster — same source, vintage, resolution; **not** ACS. Per-city extent share anchored in `verify_baselines.py` (MN 20.2 %, SA 24.5 %, ±2 pp, halve-the-raster meta-test).
- **Caveats** — Block-level counts (uniform within-block density); 2020 vintage. The per-pixel `child ≤ total` invariant holds by construction. Child population is *not* fed into the supply model or UMH — those stay total-/adult-calibrated by design (rationale in DESIGN_NOTES).

#### Nature Access at Schools

- **What it measures** — Share of K-12 school point locations sitting on pixels where the 2SFCA `adequate` mask is True (per-capita urban-nature supply meets the per-city demand standard). Destination-based readout — sampled at school POINT locations, not residential pixels — so this answers "do schools sit where supply meets demand" alongside the residential Nature Access and Children's Nature Access metrics. Reported as a citywide headline plus per-sector breakdowns (public / charter / private).
- **How it is computed** — Point-sample the canonical Nature Access `adequate` per-pixel mask at each school's (row, col) pixel. Same 2SFCA pipeline, same `UNA_DEMAND_M2_PER_CAPITA` threshold, same valid-LULC restriction as the residential metric — no new threshold introduced. The pipeline runs once; the residential and school metrics consume the same per-pixel adequacy classification. School points are pre-computed at city load by `_load_city_runtime_state` Phase 2c (project to LULC CRS, convert lat/lon → (row, col), drop off-extent rows).
- **Validation status** — `≈ Aligned method`. Reuses Nature Access's canonical-InVEST 2SFCA validation (MAE 0.0234 m²/person, MN baseline). Different *quantity* (count of school points, not population-weighted), same *method*; sampling at point locations is the only addition, not a new model. Not a surrogate target — computed deterministically.
- **Source** — School points pulled from **NCES Common Core of Data (CCD) 2022-23** (school directory: LEVEL + CHARTER status, joined on NCESSCH) + **NCES EDGE 2021-22 Geocode Public Schools** (geocoded lat/lon for public + charter) + **NCES EDGE 2021-22 Geocode Private Schools / PSS** (geocoded lat/lon for private). The ~1-year vintage offset between CCD 2022-23 and EDGE 2021-22 is accepted (school directory data doesn't churn rapidly). K-12 filter keeps levels `Elementary`, `Middle`, `High`, `Secondary`, `Other`, `Ungraded`; drops `Prekindergarten`, `Adult Education`, `Postsecondary`, and unreported levels. **Private schools are INCLUDED** — the source is the PSS Universe, which may include some Pre-K-only centers. Per-city school points clipped to the modelable extent and persisted as small GeoJSONs in the repo; raw NCES bundles (~76 MB) are gitignored. Built by `scripts/data/prep_school_points.py`.
- **Per-city counts** — SA modeled extent: **647 K-12 schools** (404 public + 164 charter + 79 private). MN downtown extent: **60 K-12 schools** (32 public + 21 charter + 7 private).
- **Sector breakdown caption** — Below the metric value, three per-sector percentages (public / charter / private). Differences across sectors surface real signal: e.g. MN downtown at 10/50/50 reads 6.7 % overall but 6.2 % public / 0 % charter / 28.6 % private — a small sample, but the divergence highlights how charter siting differs from public-school siting.
- **Caveats** — Vintage offset (CCD 2022-23 vs EDGE/PSS 2021-22). PSS may include some Pre-K-only centers under the private sector. Destination-based metric does NOT account for student travel time, the school's actual attendance catchment, or students who attend out-of-extent schools. Real student-attendance modeling would need NCES SABS attendance boundaries (on the roadmap as a follow-up); v1 is the destination-only readout.

#### Preventable MH Cases & Avoided MH Costs

- **What it shows** — Two paired metrics from the InVEST Urban Mental Health Model (v3.19.0): per-year preventable depression-and-anxiety cases attributable to the scenario's NDVI exposure change, and the avoided healthcare cost in $. Both are zero at baseline by construction (ΔNE = 0 → PF = 0). Card displays the value alone; conditional caption below distinguishes "cases prevented" / "cases induced" (and avoided / added costs) based on sign.
- **How it is computed** — Per pixel: ① `NE = scipy.ndimage.uniform_filter(NDVI_proxy, size = 2×r+1)` — the canonical **edge-corrected buffer-mean** of NDVI over a flat disk of radius `r = UMH_SEARCH_RADIUS_M / PIXEL_SIZE_M = 300 m / 30 m = 10 px` (the canonical InVEST UMH kernel). ② `ΔNE = NE_scenario − NE_baseline`. ③ `RR = exp(ln(RR_0.1) × 10 × ΔNE)`. ④ `PF = 1 − RR` (preventable fraction). ⑤ `PC = PF × baseline_prevalence × population_count`. ⑥ Sum over valid pixels for depression and anxiety; multiply by per-case cost-of-illness for the dollar metric. Constants: `RR_0_1_NDVI = 0.96` depression / `0.97` anxiety (Liu et al. 2023), `BIR = 0.21` / `0.19` (CDC 2023 ever-diagnosed), cost-of-illness `$8,467` / `$5,765` (US nominal). Population: Census 2020 block-level totals rasterized to the active city's grid.
- **Units** — Cases/yr · USD/yr.
- **Validation status** — `≈ NatCap method`. **Measured per-pixel parity:** matches `natcap.invest.urban_mental_health.execute()` (v3.19.0) on identical NDVI / population inputs at MAE ≈ 0, Pearson r = 1.0 (`validation/compare_umh_invest.py`). The validation used the synthetic NDVI proxy — it confirms the *algorithm*, not the NDVI source.
- **Main caveat** — The **NDVI proxy is synthetic** (assigned per NLCD class). Replacing it with satellite NDVI would meaningfully change UMH outputs. Other caveats: lifetime vs annual prevalence (using ever-diagnosed CDC numbers as the at-risk pool may overstate actual annual incidence), static population (no in-migration following improved amenities), direct exposure pathway only (omits air-quality and social-cohesion mechanisms), nominal cost-of-illness numbers.

##### Official InVEST alignment — UMH

InVEST UMH (added in v3.19.0) computes preventable mental health cases via `PC = (1 − RR) × BIR × POP`, where `RR = exp(ln(RR₀.₁) × 10 × ΔNE)` and NE is a neighborhood-mean NDVI within a user-specified search radius. The app implements this canonically: NE uses the **edge-corrected buffer-mean** kernel InVEST uses (a flat disk of pixel-radius `r = search_radius / pixel_size`); the prefactor and the per-outcome RR / BIR / cost-of-illness constants are surfaced as named module-level constants.

**Documented divergence:** InVEST takes baseline incidence per administrative unit from a vector input (allowing spatial variation in prevalence); the app uses uniform national CDC rates — not quantifiable here because no per-tract MH-prevalence data exists for MN/SA. The validation used the **synthetic NDVI proxy**, so it confirms the algorithm, not the NDVI source.

**Reference:** [InVEST UMH User Guide](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_mental_health.html)

### 💵 Economic

#### Food Production

- **What it shows** — Annual food yield from pixels converted to food-forest land cover, in millions of pounds per year. Delta sub-label: "people fed" (total lbs ÷ 2,000 lbs/person/yr).
- **How it is computed** — `n_food_forest_pixels × 0.222 ac/pixel × FOOD_FOREST_LBS_ACRE`. Per-city benchmarks: **MN 11,500 lbs/acre/yr** (from NatCap/InVEST food forest studies), **SA 8,500 lbs/acre/yr** (placeholder reflecting lower productivity in hot semi-arid climate, pending per-crop yield data from the SA project).
- **Units** — M lbs/year.
- **Validation status** — `Prototype`. There is no InVEST analog. InVEST Crop Production models conventional monoculture crops via climate-binned percentile distributions; the app's metric is a yield benchmark for managed food forests. The systems are fundamentally different — the parity rating is N/A rather than Proxy.
- **Main caveat** — Yield benchmark, not site-specific. Actual yield depends on species mix, management, and soil quality. NLCD 41 (deciduous forest) is the proxy because it best represents canopy structure, shade, and ET characteristics of a managed food forest — no NLCD class exists specifically for food forests or agroforestry.

#### Est. Implementation Cost

- **What it shows** — Rough order-of-magnitude total capital cost to implement all conversions in the scenario.
- **How it is computed** — `(n_GI × 0.222 × cost_GI) + (n_FF × 0.222 × cost_FF) + (n_HD × 0.222 × cost_HD)`. Per-acre costs come from the sidebar sliders.
- **Units** — Dollars (typically $M for display).
- **Validation status** — `Prototype`. App-level computation, not an InVEST metric. Default costs ($50k/ac GI, $10k/ac FF, $5k/ac HD) are placeholder mid-points to show how the math works — plug in actual local budget numbers using the sliders.
- **Main caveat** — Does not include maintenance, land acquisition, or displacement costs.

#### Flood Damage Avoided (MN) / Flood Volume Reduction (SA)

- **What it shows** — **MN:** estimated reduction in expected flood-damage costs ($) from the scenario's runoff reduction vs the unmodified baseline. **SA:** the card relabels to "Flood Volume Reduction" — NatCap's Vibrant Land report (Guerry et al. 2023) used InVEST UFRM for SA but explicitly did not enable damage valuation, so SA does not surface a dollar damage figure.
- **How it is computed** — **MN:** `TOTAL_POTENTIAL_DAMAGE_USD × max(0, runoff_reduction_fraction)`, where `TOTAL_POTENTIAL_DAMAGE_USD = Σ(building_footprint_m² × per_type_damage_rate_$/m²)` from `Damage_loss_table_MN.csv` (Other $40, Commercial $120, Residential $150, Industrial $100 per m²). Capped at $0 when the scenario warms / increases runoff. **SA:** displayed runoff-reduction volume only.
- **Units** — **MN:** USD/yr. **SA:** acre-feet.
- **Validation status** — `≈ Aligned method`. Canonical SCS-CN underneath; MN dollar approach is a proportional scaling rather than InVEST UFR's per-watershed `serv_blt` indicator (which the InVEST docs themselves call "only an indicator of service, not an actual measure of damage or savings").
- **Main caveat** — **Order-of-magnitude.** Direction tracks well; absolute magnitudes do not. For **Minneapolis Full** the dollar card returns "—" with an explanatory tooltip — OSM-only buildings don't carry the per-type codes the formula requires (Option A — see §7). On the **Selected-region impact** table (region scenarios), the flood-damage row reads **"n/a — no damage valuation available"** for SA and the real $ figure for MN — the trigger keys on the city's `damage_table_file` config, not on the computed value, so an MN scenario that legitimately produces $0 avoided still shows $0 (real number), not n/a.

#### Cooling Energy Savings

- **What it shows** — Annual avoided air-conditioning energy cost ($/yr) attributable to the scenario's cooling gain over baseline. Reported as `$X.XXM/yr`. A small caption beneath the headline reports `~$N/yr per typed building` — the city-agnostic comparable number, since the headline total depends on each city's building-extent scope.
- **How it is computed** — Per pixel: `ΔHMI = HMI_scenario − HMI_baseline`, `ΔT_°C = max(ΔHMI × UHI_MAX_C, 0)` (clamped non-negative — we credit cooling, not penalise warming). Per-pixel kWh saved = `consumption_rate_kWh_per_m²_per_°C × ΔT_°C × pixel_area_m²`. Per-pixel $ = `kWh × $0.13/kWh`. Per-type consumption rate from `energy_consumption.csv` (other 10, commercial 30, residential 20, industrial 25 kWh/(m²·°C)/yr). Returns 0 if buildings, the energy table, or the ET raster are unavailable.
- **Units** — USD/yr.
- **Validation status** — `≈ Aligned method`. Canonical InVEST UCM energy-valuation formula. The HMI input is the canonical UCM HMI raster (measured per-pixel parity for HMI itself — see §6 Temperature). The remaining divergence is **aggregation, not physics:** InVEST samples T_air per *building* over the 600 m `t_air_average_radius`; the app applies the formula per *pixel*. This is why the dollar figure should be read as order-of-magnitude.
- **Main caveat** — **Building-extent dependent.** MN coverage is the InVEST sample-data buildings shapefile (downtown core); SA coverage is OSM-typed pixels (~29 % of buildings). Areas outside the typed-buildings footprint contribute $0. The per-typed-building caption beneath the headline is the city-agnostic comparable number.

#### Avoided Carbon Cost (MN) / Carbon Storage Value (SA)

- **What it shows** — Dollar value of the carbon metric using EPA's Social Cost of Carbon. **MN:** avoided cost/year (annual flow). **SA:** stock value (one-time).
- **How it is computed** — `carbon_tons_co2 × EPA_SOCIAL_COST_CARBON ($190/ton CO2e, EPA 2023, 2 % discount, 2030 emissions)`. Derived from the Carbon metric — no additional inputs.
- **Units** — **MN:** USD/yr. **SA:** USD (one-time).
- **Validation status** — **MN:** `Prototype` (inherits from MN Carbon). **SA:** `≈ NatCap method` (inherits from SA Carbon four-pool). The methodology matches NatCap's Vibrant Land report convention; the SC-CO2 vintage differs (EPA 2023 vs Vibrant Land's IWG 2021 value) — *methodology alignment, current parameter vintage*.
- **Main caveat** — Inherits all the carbon caveats above. The dollar value is meaningful only insofar as the underlying carbon value is.

### 📊 Cost Effectiveness

Three sub-ratios under Economic — `Cost / Acre-Foot Prevented`, `Cost / °F Cooling`, `Cost / 1,000 People Fed`. App-level synthesis: no InVEST model provides these directly.

| Ratio | Formula | Notes |
|---|---|---|
| Cost / Acre-Foot Prevented | `total_cost_$ ÷ (baseline_runoff − scenario_runoff)` | N/A when scenario increases runoff, **or when runoff reduction is below ~10 ac-ft** (screening floor). |
| Cost / °F Cooling | `total_cost_$ ÷ (−temp_change_f)` | N/A when scenario warms, **or when cooling magnitude is below ~0.05 °F** (screening floor). Inherits the ±2 °F uncertainty of the HM-to-temp calibration. |
| Cost / 1,000 People Fed | `total_cost_$ ÷ (people_fed ÷ 1,000)` | N/A when no food-forest pixels, **or when fewer than 100 people fed** (screening floor). Inherits the food yield benchmark uncertainty. |

All three return **N/A** when the denominator is zero or negative (no improvement vs baseline), when total cost is zero (no conversions), **or when the benefit denominator is below the per-metric screening floor.** The floors guard against spuriously sharp dollar figures from near-zero denominators — region scenarios that produce only a fraction of an acre-foot of runoff reduction or hundredths of a degree of cooling would otherwise read as e.g. "$961k per ac-ft," which looks precise but isn't informative. All three are flagged `Prototype` — app-level synthesis without an InVEST counterpart.

---

## 7. City-specific notes

### Minneapolis

- **Extent and rationale.** The active city is downtown + near-neighborhoods (~123 km², ~154 K residents) — chosen because it has complete metric coverage including Flood Damage Avoided and Cooling Energy Savings, which require per-building-type data only available in the InVEST UFR sample dataset.
- **Buildings.** InVEST UFR sample shapefile, 3,788 polygons with per-type codes (0=other, 1=commercial, 2=residential, 3=industrial). Powers the per-type dollar metrics.
- **Population.** US Census 2020 block-level totals (Hennepin County FIPS 27053).
- **UCM (cooling).** `UHI_MAX_C = 2.05 °C` (from InVEST UCM args JSON for MN). Yields a 3.69 °F/HMI factor. Biophysical table is the InVEST UCM sample data.
- **UNA (nature access).** Per-city 2SFCA parameters: **250 m²/capita** demand, **1000 m** search radius, **exponential** decay — adopted from the NatCap MN-project canonical configuration.
- **Carbon.** Per-cover-class annual rate proxy (Food Forest 3.5, Green Infrastructure 2.0, High Density 0.0 t CO2e/acre/yr). MN does not have a NatCap four-pool bundle available — per-city framing principle applies; the framework is provisional.
- **Minneapolis Full (hidden).** A 374 × 607 EPSG:5070 raster covering the full city boundary (~149 km²) is implemented in the codebase as `'Minneapolis Full, MN'` with `available=False` — hidden from the UI pending per-building-type data for the expanded area. To re-expose: flip `available` back to `True` (pipeline + rasters + verified baselines are still in place).

#### Option A buildings semantics

For Minneapolis Full, OSM-only building polygons (185,490) lack the per-type codes the InVEST sample shapefile carries for downtown. The app detects this at module load via a `BUILDINGS_HAVE_TYPES` flag and degrades gracefully: spatial-placement mask is still built from the OSM polygons (so green-conversions still avoid building footprints); **Cooling Energy Savings** and **Flood Damage Avoided** return "—" with an explanatory tooltip. This is Option A — the alternative (synthesizing type codes from OSM building tags) was deferred.

### San Antonio

- **Extent and rationale.** The SA AOI is a bounding box covering Bexar County (~3,060 km²) rather than the city municipal boundary. Every upstream dataset feeds in at county granularity (Census 2020 FIPS 48029, SSURGO TX029, TIGER tracts) or requires a bbox clip from a larger distribution (Geofabrik Texas OSM, CGIAR Global-AI/ET0). The raster captures 1,906,325 people — between San Antonio proper (~1.4M) and the full county (~2.0M).
- **CRS.** EPSG:5070 (NAD83 / Conus Albers Equal-Area) — NLCD's native equal-area CRS. Differs from Minneapolis downtown (EPSG:26915 / UTM 15N); equal-area is preferred for SA's larger area-based analyses.
- **Buildings.** OSM polygons from the Geofabrik Texas extract (345,900 polygons). String building types are mapped to InVEST type codes 1/2/3 for ~29 % of pixels (Option A semantics — see Minneapolis Full above for the rationale). The Cooling Energy Savings card surfaces a coverage caveat when typed-pixel coverage is below 95 %.
- **LULC framework.** NatCap's **compound NLCD × NLUD × tree-canopy** framework — see §5 (Land-use alignment).
- **UCM (cooling).** `UHI_MAX_C = 11 °C` (estimate for Köppen BSh climate; see the NatCap SA README). Yields a 6.30 °F/HMI factor. SA cooling biophysical table is tuned for hot semi-arid on four high-impact NLCD classes (Shrub/Scrub, Evergreen Forest, Deciduous Forest, Hay/Pasture), anchored on eddy-covariance Kc measurements, FAO-56 Kc tables, and Stewart-Oke (2012) albedo ranges. Row-by-row provenance lives in `data/sa/cooling/biophysical_table_sources.md`. These are medium-confidence interim values pending a SA-calibrated InVEST UCM args run.
- **UNA (nature access).** Per-city 2SFCA parameters: **16.7 m²/capita** demand, **800 m** search radius, **dichotomy** decay — adopted from the NatCap SA-project canonical configuration (NatCap SA README).
- **Carbon.** Four-pool stock framework via NatCap's compound `carbon__nlcd_nlud_tree.csv` (1,984 rows × four pools). Reported as one-time stock change. Aligned with the methodology in NatCap's 2023 Vibrant Land report (Guerry et al.); the social-cost-of-carbon vintage differs (EPA 2023 vs IWG 2021) — methodology alignment, current parameter vintage. Carbon-rate sliders in Advanced Settings have **no effect for SA** — the four-pool table is the data, not a user input.
- **Flood is ~scenario-invariant.** NatCap's published SA finding (Vibrant Land + supporting presentations): the citywide flood metric is nearly insensitive to scenario choice — developed land is a small share of the bbox, so the total-metric movement under any realistic green-conversion scenario is small. The dashboard echoes this — directionality of greening (GI is generally most flood-effective per converted pixel) is real, but the headline number won't move much. Read the Flood Retention card with this scope effect in mind.
- **Block-group framing.** Per-tract aggregations for SA use NatCap-canonical ACS block-groups (1,124 polygons covering the City of San Antonio), matching the Vibrant Land Figure 10 framing. Other models' AOI uses the bbox.

#### Cross-city Heat Mitigation Index comparison

InVEST UCM's per-pixel Cooling Capacity (`CC = 0.6·shade + 0.2·albedo + 0.2·ETI`) feeds HMI. Across cities, baseline HMI values differ — and the variation is **not** primarily driven by climate. The ETI term is normalized inside each AOI as `Kc × ET / max(ET_in_AOI)` — so absolute ET (mm/yr) cancels out via the division. Only the per-class Kc lookup and the spatial gradient of ET *within* each city's bbox affect ETI. The shade term (weight 0.6) and albedo term (weight 0.2) are pure land-cover lookups with no climate dependence at all.

| City | natural % | forest+wetland % | mean shade | mean Kc | `BASELINE_HM` (mean HMI) |
|---|---:|---:|---:|---:|---:|
| Minneapolis (downtown) | 51.6 % | 2.7 % | 0.059 | 0.242 | **0.1859** |
| Minneapolis Full | 8.3 % | 1.8 % | 0.073 | 0.397 | **0.1600** |
| San Antonio | 55.4 % | **14.9 %** | **0.198** | 0.684 | **0.2866** |

San Antonio's baseline HMI is **54 % higher than Minneapolis downtown's** because SA's bbox contains 14.9 % forest + woody-wetland pixels (NLCD 41 + 90, both with shade = 1) versus only 2.7 % in MN downtown's. The 0.6 weight on shade means that single difference accounts for ~80 % of the HMI gap. Higher absolute ET in SA contributes ~zero to HMI because the formula cancels it.

**Takeaway for cross-city interpretation:** higher baseline HMI in a hotter city does **not** mean the model is rewarding hot climates with cooling potential. It means the city's land-cover composition is more vegetated. To compare the *effectiveness of greening interventions* across cities, look at scenario-vs-baseline HMI deltas, not absolute HMI values.

---

## 8. What is InVEST-aligned vs prototype-specific

### Division of labor

The Ecosystem Explorer combines NatCap-curated data, InVEST-aligned biophysical models, and prototype-specific scenario logic.

| Component | Source |
|---|---|
| **LULC raster input** | NatCap-curated where available (San Antonio: compound NLCD × NLUD × tree-canopy framework from NatCap's 2024 NASA Urban project). Otherwise NLCD 2021 (Minneapolis: InVEST UFR/UCM/UNA sample data). |
| **Biophysical model evaluation** | InVEST-aligned numpy reimplementations of Urban Cooling, Urban Flood Risk, Urban Nature Access, Urban Mental Health, and Carbon. Validated against canonical `natcap.invest.*` outputs where applicable — see the per-model sub-anchors below. |
| **Scenario placement logic** | Prototype-specific. The five placement strategies are Ecosystem Explorer heuristics — InVEST models are placement-agnostic (see `docs/research/INVEST_PLACEMENT.md`). |
| **Optimization / search** | Prototype-specific. The surrogate-driven optimizer is reframed as **scenario discovery** — see below. |

### Official InVEST alignment

Per-metric alignment status is maintained in [`docs/internal/NATCAP_ALIGNMENT.md`](docs/internal/NATCAP_ALIGNMENT.md) §3 "Metric methodology fidelity". The per-model sub-anchors below preserve the cross-reference targets used elsewhere in the repo and in the in-app help.

The five urban InVEST models and the Carbon model each have their own alignment narrative — see §6 alongside the corresponding metric card (Temperature Change → Official InVEST alignment — UCM; Nature Access → Official InVEST alignment — UNA; Preventable MH Cases → Official InVEST alignment — UMH). Carbon and UFR alignment are folded into their §6 metric mini-templates (Validation status field).

**Crop Production (no overlap).** InVEST Crop Production models 172 staple crops via climate-binned percentile distributions or fertilizer-response regressions. The model does not support food forests, agroforestry, or polyculture. The app's Food Production metric uses a yield benchmark applied to a food-forest proxy class — the systems are fundamentally different, the parity rating is N/A rather than Proxy.

### The optimizer as scenario discovery

The **Discover scenarios to validate** panel runs in one of two modes depending on whether a region or ownership filter is active. Both modes are scenario-discovery tools — they surface options worth testing more rigorously, not a single best answer. Neither computes Pareto-optimal solutions in NatCap ROOT's rigorous LP sense.

**Mode 1 — citywide surrogate (no filter active).** A Random Forest surrogate trained on the precomputed scenario library predicts metrics for ~10,000 random `(pct, GI%, FF%)` candidates, filters to those meeting the user's minimum-target sliders, and returns up to 5 suggestions ranked by a balanced score. The values shown on each suggestion are **surrogate predictions**, not engine outputs. Apply a suggestion to run the full engine and replace the predictions with engine-verified values; provenance flips to **"Surrogate-suggested."**

**Mode 2 — region-prefilter + engine-verify ("Optimize selected area," region or ownership filter active).** A two-stage pipeline keeps the search fast while making the displayed values real:

1. **Prefilter** — the Phase-0.5-validated Fast surrogate (~90 recipes, 100 trees, ranking ρ 0.83–0.98 vs the engine on region-scoped candidates) scores every candidate and picks ~40 Pareto-efficient ones across the objective metrics.
2. **Engine-verify** — the full per-pixel engine evaluates each shortlisted recipe on the active `region ∩ ownership` mask. K × ~2.1 s ≈ 1–2 minutes; a progress bar shows `i / K`.
3. **Rank + dedup** — weight-sum the engine values (sliders 0–1 per metric, direction-corrected: lower-better for cost and runoff), greedy knob-distance dedup to 5 meaningfully-distinct recipes.

Returned values are **engine-true region-local** — not predictions, no P10/P90 bands. The shortlist may not be exhaustive, so the header reads *"Best tested mixes — selected area"* — framing them as the best among the candidates the engine actually tested rather than implying global optimality. Apply on a returned record sets the sliders, reruns the engine on the active mask, and flips provenance to **"Engine-verified — region-optimized"** (distinct from "Surrogate-suggested"). Changing a weight slider does not auto-rerank — re-click **Optimize selected area** to apply new weights (the v1 implementation re-runs the full pipeline; the spec's "instant rerank without engine re-run" property is a future optimization, not the shipped behavior).

| | Mode 1: citywide surrogate | Mode 2: Optimize selected area |
|---|---|---|
| **When it fires** | No region/ownership filter active | Region and/or ownership filter active |
| **Knobs** | Minimum-target sliders (flood / cooling / food / runoff cap / carbon) | Weight sliders (cooling / flood / carbon / food / cost) |
| **Search size** | ~10,000 random candidates | Surrogate Pareto over the ~90-recipe training grid → cap at K ≈ 40 for engine-verify |
| **Values displayed** | Surrogate predictions with P10/P90 bands | Engine-true region-local, **no bands** |
| **When to trust the number** | After Apply (full-engine rerun) | At display time (already engine-evaluated) |
| **Provenance after Apply** | "Surrogate-suggested" | "Engine-verified — region-optimized" |
| **Caveat surfaced in caption** | Predictions need verification | Results are real; shortlist may not be exhaustive |
| **Region / ownership awareness** | None — surrogate is citywide-trained and region-blind | Engine evaluates on `region ∩ ownership` directly; surrogate's role is shortlist-only |
| **Comparison to NatCap ROOT** | Both modes are RF + heuristic ranking; neither computes ROOT's LP frontiers (max Σ wᵢ Vᵢₛₐ xₛₐ at spatial-decision-unit level with production possibility frontiers and agreement maps). ROOT remains a deferred reference point. | |

**Confidence range (P10 / P90) — scope.** The orange-diamond error bars on the Tradeoff Space chart and the uncertainty columns in the optimizer-result table are **10th / 90th percentile predictions across the Random Forest's trees** (`predict_with_uncertainty` in `surrogate.py`). They reflect the surrogate's per-prediction disagreement, not measurement uncertainty. **Scope is Mode 1 only:** the live metric cards, the validated per-pixel engine, the displayed-validated taxonomy, AND the Mode 2 region-optimizer results do NOT carry P10/P90 — those values are engine-true. The bands belong exclusively to the Mode 1 optimizer overlay and its result table.

**Spatial-geometry blindness — still applies to the surrogate.** Even in Mode 2 the surrogate cannot see where pixels land (it inputs only `pct, GI%, FF%`). But the surrogate's role stops at shortlisting; the engine-verify step uses the real per-pixel raster math and the real region mask, so the displayed values reflect actual placement geometry. The spatial-trend-only caveat applies to Mode 1's predictions, not to Mode 2's displayed numbers.

For the deep mechanics (Phase-0/0.5 grounding measurements, candidate pipeline, normalization, dedup, machine assertions), see `docs/internal/REGION_OPTIMIZER_SPEC.md` and `docs/internal/DESIGN_NOTES.md`. For RF tree counts per model-quality mode and lookup-table internals see `docs/internal/ARCHITECTURE.md` (At a glance + Layer 1/2/3 sections).

### Export for InVEST

The sidebar's **Export for InVEST** section (San Antonio only for v1) packages the currently-displayed scenario as a runnable canonical InVEST 3.19.0 input bundle:

```
ecosystem_explorer_export_<city_slug>_<scenario_id>_<timestamp>.zip
├── README.md                        (how-to-run; bundle-relative paths)
├── metadata.json                    (provenance, generator, per-model validation)
├── inputs/
│   ├── prototype/                   (rasters on the 30 m EPSG:5070 prototype grid)
│   │   ├── scenario_lulc_evaluated_30m_5070.tif  (compound — UCM / UNA / Carbon-alt)
│   │   ├── baseline_lulc_evaluated_30m_5070.tif
│   │   ├── scenario_lulc_nlcdtree_30m_5070.tif   (NLCD×tree — UFR)
│   │   ├── baseline_lulc_nlcdtree_30m_5070.tif
│   │   ├── scenario_ndvi_30m_5070.tif            (UMH `ndvi_alt`)
│   │   └── baseline_ndvi_30m_5070.tif            (UMH `ndvi_base`)
│   ├── shared/                      (population, ET, soil, AOIs, prevalence vectors)
│   └── biophysical/                 (UCM / UNA / Carbon compound tables; NLCD×tree CN table)
└── args/prototype_grid/             (one args.json per model)
    ├── urban_cooling_args.json
    ├── urban_nature_access_args.json
    ├── urban_flood_risk_mitigation_args.json
    ├── carbon_args.json
    ├── urban_mental_health_depression_args.json
    └── urban_mental_health_anxiety_args.json
```

**Running canonical InVEST on the bundle.** From the bundle root (paths in the args files are bundle-root-relative):

```bash
python -c "import json; from natcap.invest import urban_cooling_model as m; m.execute(json.load(open('args/prototype_grid/urban_cooling_args.json')))"
```

Substitute the module and args path for each model (UCM / UNA / UFR / Carbon / UMH). All five execute cleanly on InVEST 3.19.0 — verified on the SA baseline bundle.

**Scenario vs baseline deltas.** Carbon runs baseline-vs-scenario in a single execution. UCM / UNA / UFR / UMH produce one result per LULC — to get the delta, run each twice (scenario LULC, then baseline LULC). The bundle's README documents this.

**Bundle-level caveats:** UCM is biophysical-cooling only (`do_energy_valuation = False`); UFR damage valuation is omitted for SA (Path C, matching NatCap's Vibrant Land report); the prototype-extent AOI is a bbox polygon, not a hydrologic watershed; UMH uses synthetic uniform `risk_rate` and a synthetic NDVI proxy — algorithmic parity validated, input-quality caveats remain. Out of scope for v1: source-grid args files for NatCap's 10 m fixed-scenario rasters, batch export from the saved-scenarios table, round-trip import of canonical InVEST results.

---

## 9. Known limitations

General limitations independent of city — see §7 for per-city limitations.

- **Indices vs impacts.** The dashboard surfaces both indices (Flood Retention, HMI, NDVI) and impact estimates (acre-feet runoff, °F cooling, dollar damage, preventable cases). Indices are tight; impact estimates carry calibration uncertainty (±2 °F for temperature, order-of-magnitude for dollar metrics) and should be read as planning signals, not engineering numbers.
- **Synthetic NDVI.** NDVI is assigned per NLCD class as a proxy. This affects both the NDVI metric card and the InVEST Urban Mental Health Model outputs. Replacing with satellite NDVI (Sentinel-2 / Landsat) would be a meaningful upgrade.
- **Per-city design storm.** Flood metrics use each city's NatCap-canonical depth — Minneapolis 3.94″ / 100 mm (NatCap MN-project `args.json`), San Antonio 6.18″ / 157 mm (NatCap SA README). Results for larger storms will differ.
- **Modelable extent for Nature Access.** The Nature Access headline is the share of the *modelable-extent* population (residents on pixels with valid land cover), not the whole city. The tooltip frames this denominator.
- **Heat exposure proxy.** The "Target High Heat-Exposure Areas" toggle uses NLCD development intensity (23 > 22 > 21) as a stand-in for neighborhood heat vulnerability. This is a land-use proxy, not a measured temperature or socioeconomic index. A formal Heat Vulnerability Index (e.g. CDC/ATSDR HVI by census tract) is intended for a future version.
- **Carbon rates are placeholder for MN.** MN uses a provisional per-cover-class annual rate (USDA NRCS / IPCC midpoints). Treat as directional only until locally calibrated values are available. SA uses NatCap's four-pool framework directly and is on firmer ground methodologically.
- **Food yield is a benchmark.** Yield is a per-acre benchmark for managed food forests, not site-specific. Actual yield depends on species mix, soil, and management intensity. NLCD 41 (deciduous forest) is the proxy class — no NLCD class specifically represents food forests.
- **Cost estimates are placeholders.** Default per-acre costs ($50k GI, $10k FF, $5k HD) are mid-points to show how the math works, not sourced from local studies. Use the sliders.
- **Optimizer is a surrogate.** The Random Forest surrogate is an approximation trained on pre-computed scenarios. Optimizer suggestions are candidates, not validated results — Apply them to the sliders to see the full canonical-engine evaluation (see §8).
- **Optimizer cannot see spatial geometry.** The surrogate inputs only `pct, GI%, FF%` — it cannot see where pixels are placed. Nature Access predictions in particular are spatial-trend-only.

---

## 10. What to validate before decision use

The dashboard is **not intended for:**

- Parcel-level siting decisions
- Precise impact prediction
- Final policy or investment decisions without further analysis

If you have a candidate scenario you want to take seriously, here is what to validate first:

| Concern | What to do |
|---|---|
| **Site-specific feasibility** | A converted pixel here may be a parking lot, a private yard, or a protected riparian strip. The dashboard's placement strategies treat all developed pixels (NLCD 21–24) minus buildings and roads as equivalent. Confirm parcel ownership, zoning, corridor design, and adjacency with site-specific data before committing to a placement. |
| **Precise dollar magnitudes** | The Cooling Energy Savings, Flood Damage Avoided, and Cost Effectiveness ratios are order-of-magnitude. For an investment case, plug actual local cost data into the sidebar sliders, then replace the benchmark inputs (per-type damage rates, per-type AC consumption rates) with locally measured values. |
| **Carbon sequestration rates (MN)** | MN's per-cover-class rates are provisional regional benchmarks, not site-calibrated. For carbon-credit accounting or net-zero reporting, replace with measured rates for the specific species mix and management regime. |
| **Mental health outputs** | The UMH metric uses synthetic NDVI and uniform national CDC prevalence. For site-specific MH analysis, replace the NDVI raster with satellite-derived NDVI and use per-tract prevalence from local public-health data. |
| **Optimizer suggestions** | Apply each candidate to the sliders. The displayed cards then reflect a full-raster evaluation by the canonical-engine-verified models, not a surrogate prediction. Re-verify there. |
| **Full canonical InVEST re-run** | For SA: use the **Export for InVEST** bundle (§8) and re-run each model in canonical `natcap.invest`. Compare the resulting rasters and aggregated outputs against the dashboard's reported numbers. For MN: the same workflow is on the roadmap. |
| **NatCap published reference values** | Where the dashboard surfaces a NatCap published value (green `NatCap published value` badge in the fixed-scenario reference view), it is displaying NatCap's number, not reproducing it. If the published number is decision-relevant, consult the underlying NatCap report (e.g. Vibrant Land for SA) for the methodology and assumptions behind that figure. |

The honest stance: this tool is for **exploration and shortlisting**. The numbers are good enough to identify candidates worth deeper analysis; they are not good enough to substitute for the deeper analysis itself.
