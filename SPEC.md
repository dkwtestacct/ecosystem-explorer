# Ecosystem Explorer — Design Specification

## Purpose

The Ecosystem Explorer is a Streamlit-based decision-support tool for exploring how reallocating developed urban land across three land use types — green infrastructure (constructed wetlands), food forest, and high-density development — affects three ecosystem services simultaneously: stormwater management, urban cooling, and food production.

The tool is built for **comparative scenario exploration**, not precise prediction. It is most useful in early-stage planning conversations where teams need a quick way to understand tradeoffs across land-use strategies before committing to detailed analysis.

---

## Intended Users

- Urban planners and sustainability staff exploring land-use allocation options
- Researchers using the tool as a platform for communicating multi-objective tradeoffs
- Stakeholders and community participants in planning processes (via guided facilitation)
- Students and educators learning about ecosystem services and urban greening

**Not intended for:** parcel-level siting decisions, regulatory filings, final investment decisions, or any context where precise quantitative outputs are required without further analysis.

---

## Key Design Decisions

### 1. City-scale aggregation, not parcel-level resolution

All results are computed as city-wide aggregates (mean CN, mean HM, total food production). This is a deliberate choice to keep the interface simple and the computation fast. The tool answers "what happens if X% of the city converts?" rather than "which specific parcels should convert?"

This makes results robust to input uncertainty and keeps the tool approachable for non-technical users, at the cost of spatial specificity.

### 2. NLCD land cover as both input and proxy for conversions

The tool uses the National Land Cover Database (NLCD 2021) at 30m resolution as its primary input. Conversions are modeled by reassigning NLCD codes to converted pixels:

| Land use | NLCD code | Why this proxy |
|----------|-----------|----------------|
| Green Infrastructure | 90 (woody wetlands) | Closest NLCD class to constructed wetlands; strong CN and HM values |
| Food Forest | 41 (deciduous forest) | Best available proxy for canopy structure; no NLCD class for agroforestry |
| High Density | 24 (developed, high intensity) | Direct match |

Using NLCD codes means biophysical parameters (CN, HM) can be looked up from existing InVEST/NatCap biophysical tables without requiring custom field data.

### 3. SCS Curve Number method for flood

The USDA Soil Conservation Service (SCS) Curve Number method is used to convert land cover and soil group to a runoff depth for a 2-inch design storm. This is the most widely used method for urban stormwater estimation, is well-understood by practitioners, and requires only NLCD and soil survey data (both nationally available).

The 2-inch design storm represents a common minor event rather than an extreme flood scenario. It was chosen for consistent cross-scenario comparison, not for sizing infrastructure.

### 4. InVEST Urban Cooling Model for temperature

The Heat Mitigation (HM) Index from the InVEST Urban Cooling Model is used as a proxy for urban cooling potential. `HM = (shade + kc) / 2` per pixel, where `shade` and `kc` are per-lucode parameters from the InVEST biophysical table.

The 4°F/HM unit conversion factor is the approximate midpoint of the published InVEST UCM range (2–5°C per HM unit) and has not been locally calibrated for Minneapolis. All temperature outputs carry roughly ±2°F uncertainty and should be treated as directional.

### 5. Pre-computed lookup table for instant slider response

To avoid running pixel-by-pixel raster computations on every slider interaction, results are pre-computed for all valid slider combinations (step=5) at startup and cached in a dict keyed by `(pct_converted, gi_pct, ff_pct)`. This gives instant response for the main scenario view.

The lookup table uses default cost parameters and uniform random pixel placement. When the user changes cost sliders or enables heat-priority mode, the lookup table is bypassed and the full computation runs live.

### 6. Random Forest surrogate for the optimizer

The Smart Scenario Search searches ~10,000 candidate scenarios to find Pareto-optimal inputs meeting user-specified minimum targets. Running the full biophysical model 10,000 times would take too long for interactive use.

Instead, a Random Forest (100 trees, multi-output) is trained on a pre-computed grid of 150 scenarios (pct ∈ {0,10,…,50}, gi/ff ∈ {0,25,…,100}). The RF predicts flood, HM, and food for any input in under a second. The 10th/90th percentile spread across trees is used as an uncertainty band.

The surrogate currently covers only the three ecological outputs. Cost and heat-priority mode are not yet included in the surrogate training and are excluded from optimizer constraints.

### 7. CITIES dict as the single source of truth for multi-city extensibility

All city-specific configuration (data paths, baseline CN and HM, reference scenario benchmarks) lives in the `CITIES` dict at the top of `app.py`. Adding a new city requires only adding an entry to this dict and placing data files in the declared directories. No other code changes are needed.

City selection in the sidebar happens before data loading, so `@st.cache_data` functions receive city-specific paths as parameters and cache results per city.

### 8. Cost-effectiveness ratios as comparative indicators

Three cost-effectiveness ratios ($/acre-foot prevented, $/°F cooling, $/1,000 people fed) are computed to let users compare scenarios on a normalized basis. These return N/A rather than divide-by-zero when the denominator is zero or negative (e.g. when a scenario increases runoff rather than reducing it).

Default per-acre costs are illustrative estimates only, not sourced from specific studies. They are intended to demonstrate the concept. Users are expected to adjust sliders to reflect local project costs.

### 9. Pareto frontier on saved scenarios

The tradeoff chart displays a Pareto frontier over all saved scenarios — the subset where no other saved scenario is better on all three metrics (flood, HM, food) simultaneously. This helps users identify which scenarios represent genuinely efficient tradeoffs rather than simply being high on one metric.

### 10. Heat-priority weighted placement

When the "Target High Heat-Exposure Areas" toggle is on, developed pixels are sampled with weights proportional to NLCD development intensity (code 23 → 1.0, code 22 → 0.6, code 21 → 0.3) rather than uniformly at random. This concentrates conversions in higher-intensity developed areas as a proxy for heat-exposed locations.

The lookup table (which uses uniform placement) is bypassed in this mode; all results are computed live.

---

## Intentional Simplifications

These are known gaps that were deliberately left in place, either because they are acceptable for the tool's stated purpose or because addressing them requires data not yet available.

**Spatial placement is not siting-aware.** Pixels are selected randomly or intensity-weighted across the whole city. The tool does not account for parcel ownership, zoning, existing uses, contiguity, or ecological corridors. This is appropriate for city-scale exploration but unsuitable for siting specific projects.

**Food production baseline is zero.** The food production metric counts only pixels *converted by the scenario*, not pre-existing NLCD 41 (deciduous forest) pixels in the baseline. This is a deliberate modeling choice: the tool measures the marginal contribution of scenario decisions, not total city food potential.

**Food forest yield is a flat benchmark.** 11,500 lbs/acre/year is a rough estimate from NatCap food forest studies, assuming a mature, well-managed system. It does not vary by species, soil, climate, or management. Outputs are directional only.

**Temperature calibration is not locally validated.** The 4°F/HM unit factor is a literature midpoint for the InVEST UCM and has not been calibrated against measured Minneapolis temperatures. Cooling outputs carry ±2°F uncertainty. Local air temperature is also influenced by wind, humidity, and urban geometry not captured in the model.

**Heat-exposure proxy uses development intensity, not measured vulnerability.** The heat-priority weighting uses NLCD code intensity (23 > 22 > 21) as a stand-in for neighborhood heat exposure. This is a land-use proxy, not a measured temperature surface or socioeconomic vulnerability index.

**Implementation costs are illustrative.** Default $/acre values ($50k GI, $10k FF, $5k HD) are order-of-magnitude estimates not derived from specific projects or published cost curves. Cost outputs should be treated as rough comparative indicators only.

**The surrogate does not cover cost or heat-priority mode.** The optimizer cannot constrain on cost and does not account for the spatial effects of heat-priority weighting. Surrogate predictions reflect ecological outcomes under uniform random placement at default costs.

**Reference scenario benchmarks are Minneapolis-specific.** The four extreme-scenario benchmarks (all food forest, all green infra, all high density, baseline) in the tradeoff chart are computed from Minneapolis 2021 NLCD data. Each city added to the `CITIES` dict must supply its own values.

---

## Planned Future Features

**Additional cities.** San Antonio, TX is already defined as a placeholder entry in the `CITIES` dict. The data pipeline is designed to accept any city with NLCD, SSURGO, and InVEST biophysical table inputs.

**Heat Vulnerability Index integration.** The current heat-exposure proxy (NLCD development intensity) is intended to be replaced with a formal Heat Vulnerability Index — for example, the CDC/ATSDR HVI by census tract. This would make the heat-priority mode a genuine equity-weighted intervention tool rather than a land-use proxy.

**Surrogate coverage of cost and heat-priority mode.** The optimizer currently excludes cost constraints and does not model the spatial effects of heat-weighted placement. A future surrogate training pass that varies cost parameters and includes heat-priority scenarios would enable full optimizer coverage.

**Parcel-level siting constraints.** A future version could integrate parcel ownership data, zoning layers, or opportunity site maps to filter which pixels are actually eligible for conversion, replacing the current assumption that all developed land is freely convertible.

**Local temperature calibration.** Replacing the generic 4°F/HM unit factor with a Minneapolis-specific calibration (using measured surface temperatures or a validated UCM run) would reduce cooling output uncertainty from ±2°F to a tighter, defensible range.

**Corridor and connectivity analysis.** Green infrastructure and food forest outcomes depend not just on total area but on spatial configuration. A future version could score scenarios on patch connectivity or corridor continuity using graph-based landscape metrics.
