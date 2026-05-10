# Ecosystem Explorer — One-Page Summary

## What it does

The Ecosystem Explorer simulates how reallocating developed urban land across three land-use types (green infrastructure, food forest, high-density development) affects a basket of ecological, social, and economic outcomes simultaneously. Built for early-stage planning conversations and tradeoff exploration — not for precise prediction or final investment decisions.

---

## Three cities supported

| City | Extent | CRS | Buildings | Notes |
|------|--------|-----|-----------|-------|
| **Minneapolis, MN** (downtown) | 360 × 356 px ≈ 122.8 km² | EPSG:26915 (UTM 15N) | InVEST sample, 3,788 polygons with type codes 0–3 | Original view; full per-type dollar metrics. |
| **Minneapolis Full, MN** | 374 × 607 px ≈ 204.3 km² | EPSG:5070 (Conus Albers) | OSM, 185,490 polygons, no type codes | Full city + suburb fringe; **Option A** buildings semantics — see below. |
| **San Antonio, TX** | 1984 × 1713 px ≈ 3,058 km² | EPSG:5070 (Conus Albers) | OSM Geofabrik TX, 345,900 polygons (string types) | Bexar County. Soil 49 % D-class Vertisols (clay) vs Hennepin's 0 %. Annual PET ~1,650 mm/yr (50 % above Minneapolis). **Option A** buildings semantics. |

The city picker lives in the sidebar. Each city has its own cached lookup table; switching between them takes ~30 s the first time (longer for SA's larger raster) and is instant after.

---

## Three land-use types

| Land Use | NLCD Code | Best for |
|----------|-----------|---------|
| Green Infrastructure | 90 (woody wetlands) | Flood reduction, carbon |
| Food Forest | 41 (deciduous forest proxy) | Cooling, food production, carbon, NDVI |
| High Density | 24 (developed high intensity) | — (worst on most ecological metrics) |

---

## 14 metric cards across three categories

The Scenario tab is organized as **Ecological (5) · Human & Social (4) · Economic (5, in two rows)**. Every tooltip starts with a one-line confidence label so users can gauge methodological strength at a glance.

### Ecological (5)

| Metric | Unit | Confidence label |
|--------|------|-----------|
| Flood Risk Reduction | index 0–100 (`100 − mean_CN`) | Raster-based calculation |
| Temperature Change | °F vs unmodified baseline | Raster-based calculation |
| Runoff Volume | acre-feet per 2-inch design storm | Raster-based calculation |
| Carbon Sequestration | t CO2e/yr from converted pixels (k notation above 1,000) | Provisional assumption |
| NDVI | mean vegetation index 0–1 | Synthetic proxy |

### Human & Social (4)

| Metric | Unit | Confidence label |
|--------|------|-----------|
| Nature Access | % of residents whose access score exceeds 0.3 | Proximity estimate |
| Nature Quality Score | population-weighted mean access score 0–1 | Composite proxy |
| Preventable MH Cases | depression + anxiety cases/yr (InVEST UMH) | Model-based estimate |
| Avoided MH Costs | $/yr healthcare burden avoided | Model-based estimate |

### Economic (5, in two rows)

| Metric | Row | Unit | Confidence label |
|--------|-----|------|-----------|
| Food Production | 1 | M lbs/yr from food forest pixels | Provisional assumption |
| Est. Implementation Cost | 1 | $M total ($/acre slider × converted area) | Order-of-magnitude estimate |
| Flood Damage Avoided | 2 | $ from runoff-reduction × per-building damage rate | Order-of-magnitude estimate |
| Cooling Energy Savings | 2 | $/yr from per-building avoided AC consumption | Order-of-magnitude estimate |
| Avoided Carbon Cost | 2 | $/yr at EPA Social Cost of Carbon ($190/ton CO2e, EPA 2023) | Model-based estimate |

A **Cost Effectiveness** sub-section under Economic exposes three ratios: $/ac-ft prevented · $/°F cooling · $/1,000 people fed. Lower is better; N/A when the denominator is zero or negative.

---

## How the metrics are computed

- **Flood Risk Reduction & Runoff Volume** — USDA SCS Curve Number method per pixel (NLCD land cover × SSURGO soil hydrologic group, sourced via the USDA Soil Data Access REST API for Minneapolis Full; InVEST sample for downtown). Aggregated to a city-wide mean CN, converted to runoff depth via $S = 1000/CN - 10$ and $R = (P - 0.2S)^2 / (P + 0.8S)$ for a 2-inch design storm, then scaled by total developed acreage. `BASELINE_CN` is recomputed dynamically at module load to match the live `evaluate_scenario` lookup, so deltas at `pct_converted=0` are exactly zero.
- **Temperature Change & Cooling Capacity (CC)** — Full **InVEST Urban Cooling Model**: per-pixel `CC_raw = 0.6·shade + 0.2·albedo + 0.2·ETI` where `ETI = Kc · ET_annual / ET_max`, then **Gaussian-smoothed at σ = 15 px (450 m)** to spatially propagate cooling onto neighboring pixels (matches InVEST's `green_area_cooling_distance`). The card reports `mean(CC)` labeled "Cooling Capacity" — an approximation of the canonical InVEST HMI. Delta to °F via `UHI_MAX_C × 1.8 = 3.69 °F per CC unit` (Minneapolis `uhi_max=2.05 °C` from the InVEST args JSON). Deltas below 0.1 °F display as "No change".
- **Cooling Energy Savings** — Canonical InVEST UCM energy-valuation formula, per pixel: `ΔT_°C = ΔCC × UHI_MAX_C` clamped non-negative; `kWh = consumption_rate × ΔT_°C × pixel_area_m²`; `$ = kWh × 0.13` (US average residential 2024). The `consumption` column is documented as `kWh/(m²·°C)` so the per-degree response is already encoded — no separate fractional sensitivity factor. Sums over building pixels. **Returns $0 for Minneapolis Full** because OSM polygons lack the per-type codes the formula requires (Option A — see limitations).
- **Carbon Sequestration** — Counts only newly converted pixels × per-cover rates from `CARBON_SEQ_RATES` (default 3.5 / 2.0 / 0.0 t CO2e/acre/yr for FF / GI / HD). Rates are user-overridable in Advanced Settings.
- **NDVI** — Synthetic proxy assigned per-NLCD-code (woody wetlands 0.70, food forest 0.75, high-density 0.10). Not derived from satellite imagery.
- **Nature Access & Nature Quality Score** — InVEST Urban Nature Access biophysical table (per-class `urban_nature` score and `search_radius_m`). Search radii **capped at 1,000 m** (`NATURE_RADIUS_CAP_M`) so water/forest classes don't saturate the AOI. Per pixel, the access score is the *maximum* of `urban_nature × in_range` across all natural classes — a pixel near multiple nature types takes the highest single class (prevents double-counting). **Nature Access**: % of population with access score > 0.3. **Nature Quality Score**: population-weighted mean access score (continuous companion). Population from US Census 2020 block totals.
- **Preventable MH Cases & Avoided MH Costs** — InVEST Urban Mental Health Model (v3.19.0). Per-pixel `NE = gaussian_filter(NDVI_proxy, σ = 300 m / 30 m px = 10 px)`; `ΔNE = NE_scenario − NE_baseline`; `RR = exp(ln(RR₀.₁) × 10 × ΔNE)`; `PC = (1 − RR) × baseline_prevalence × population`. Two outcomes summed (depression + anxiety). Effect sizes from Liu et al. 2023 meta-analysis (RR 0.96 / 0.97 per 0.1 NDVI gain), prevalence from CDC 2023 (21 % / 19 % ever-diagnosed), cost-of-illness $8,467 / $5,765 per case (US nominal; InVEST docs cite ~$11K USD-PPP default). Replaced the earlier weighted-composite Wellbeing Score.
- **Food Production** — Food-forest pixel count × 0.222 acres/pixel × 11,500 lbs/acre/year (NatCap benchmark, mature managed system).
- **Flood Damage Avoided** — Per-building potential damage from `Damage_loss_table_MN.csv` (per-type $/m² × footprint area), scaled by the scenario's runoff reduction vs baseline. **Returns $0 for Minneapolis Full** (Option A).
- **Est. Implementation Cost** — Sum of converted acres × per-acre cost slider for each land-use class.
- **Cost Effectiveness** — Implementation cost ÷ benefit per metric. Returns N/A when the denominator is zero or negative.

---

## Spatial placement constraints

- **Building footprints** — Conversions can't land on top of buildings. Minneapolis downtown uses the InVEST sample shapefile (3,788 polygons); Minneapolis Full uses OSM (185,490 polygons).
- **Road exclusion** — Conversions can't land on roads. OSM road network from the Geofabrik state extract, **filtered with Option B** to drop sub-pixel-width surfaces (footway, cycleway, steps, service, path, pedestrian, unclassified, track*). For downtown: 5,495 segments covering ~29 % of AOI; for full city: 10,984 segments. Buildings + roads are unioned into one `BUILDINGS_RASTER` mask; the leftover developed pixels form the `CONVERTIBLE_PIXELS` pool.
- **Heat-vulnerability priority** — Optional toggle weighting placement toward high-intensity-developed pixels (NLCD 23) as a heat-exposure proxy.

---

## How to use it (the headline features)

- **Scenario tab** — slider-driven scenario builder showing all 14 metric cards plus a collapsible **Baseline vs Scenario Comparison** table that color-codes improvements (green) vs regressions (red, with runoff treated as inverse).
- **Tradeoff Analysis tab** — Plotly chart of the entire scenario space with the active scenario, saved scenarios, optimizer suggestions, and Pareto frontier overlaid. Per-city `REF_SCENARIOS` (Baseline, All Food Forest, All Green Infra, All High Density at 50 % conversion) plot as colored markers. Below: **Best Scenarios by Goal** (five canonical winners drawn from the pre-computed library, each with an Apply button), then a Save-this-scenario flow with named saved scenarios.
- **Map View tab** — spatial map of where conversions occur, plus a heat-vulnerability red-wash overlay slider (default opacity 0.3). Map renders via matplotlib `imshow` (pixel-row/column space), so EPSG:5070 and EPSG:26915 cities both render correctly without per-CRS handling.
- **Smart Scenario Search optimizer** — Random Forest surrogate trained on the live scenario grid; samples 10,000 random (pct, GI%, FF%) combinations against user-set minimums on flood, cooling, food, and carbon, and returns up to 5 Pareto-efficient suggestions.
- **Advanced Settings (sidebar)** — overrides for Food Forest carbon rate, Green Infrastructure carbon rate, and a **Model Quality Mode** radio (Fast prototype / Balanced / High resolution) that swaps the surrogate's training set between 90, ~726, and 2,541 scenarios with corresponding tree-count adjustments.

---

## Baseline constants (live, per city)

| Constant | Mpls downtown | Mpls Full | San Antonio | Computation |
|---|---:|---:|---:|---|
| `BASELINE_CN` | 75.67 | 77.68 | 76.54 | Mean CN over `cn_table[lulc_idx, soil]` for the unmodified LULC × soil grid |
| `BASELINE_HM` (= mean CC) | 0.1859 | 0.1600 | 0.2866 | Mean of the smoothed-CC raster |
| `BASELINE_NDVI` | 0.2326 | 0.2072 | 0.4242 | Mean of synthetic NDVI proxy |
| Population | ~154 K | 463,794 | 1,906,323 | Census 2020 county-level totals in the bbox |

Both `BASELINE_CN` and `BASELINE_HM` are dynamically overridden at module load, so the hardcoded values in `CITIES['<city>']['baseline_cn'/'baseline_hm']` are documentation only — the live values track whatever the current pipeline produces.

---

## Key assumptions and limitations

- **Option A buildings semantics for Minneapolis Full** — OSM polygons have no per-type codes (0=other, 1=commercial, 2=residential, 3=industrial), so the per-type lookups feeding **Cooling Energy Savings** and **Flood Damage Avoided** are unavailable for the expanded city. Both cards display "—" with explanatory tooltip; the spatial-placement mask still works because it doesn't need types. A future Overpass-API-sourced building set with `building=*` subkeys would unlock the dollar metrics city-wide.
- **Spatial placement is stylized** — converted pixels are picked randomly (or heat-weighted), not by parcel feasibility, ownership, corridor design, or zoning.
- **InVEST UCM divergences from canonical** — we apply the Gaussian convolution but skip the per-building `t_air_average_radius` aggregation; mean(CC) approximates but is not identical to canonical HMI. See `data/invest/cooling/UCM_AUDIT.md` for the full divergence log.
- **Nature Access search radii capped at 1,000 m** — the InVEST UNA defaults (5 km for water/forest) saturate the AOI; the cap restores meaningful per-scenario variation but means our values aren't directly comparable to a canonical InVEST UNA run.
- **Carbon and food rates are provisional** — defaults come from broad regional benchmarks (USDA NRCS, NatCap), not site-specific data.
- **Mental health estimates** use CDC lifetime prevalence rates (depression 21 %, anxiety 19 %) as the at-risk pool and nominal US cost-of-illness figures ($8,467 / $5,765 per case) — suitable for directional comparison, not precise economic valuation. NDVI is also a synthetic per-NLCD-class proxy, not satellite-derived. Treat the dollar figures as order-of-magnitude.
- **Implementation costs are illustrative** — per-acre values are order-of-magnitude placeholders.
- **Surrogate covers ecological + carbon + UMH outcomes** — not cost, not heat-priority placement effects. Verify any promising surrogate suggestion through the main sliders.

---

## Computation architecture

| Layer | Purpose | Scale |
|-------|---------|-------|
| Full-resolution raster simulation | Pixel-level biophysical truth | Per-scenario, on demand |
| Pre-computed lookup table | Instant slider response | 2,541 (pct, GI%, FF%) entries at step=5, **per city** |
| Pre-computed dense grid | Surrogate training set | 726 scenarios at step=5/10 (built offline by `precompute_scenarios.py --city ...`) |
| Random Forest surrogate | Rapid scenario search at query time | Trains on 90 / ~726 / 2,541 rows depending on Model Quality Mode |
| Surrogate sampling at optimization time | Explore wide tradeoff space | ~10,000 random candidates per Optimize click |

`SCENARIO_SCHEMA_VERSION = 14` invalidates cached lookup tables across changes to the metric schema. Per-city caches don't collide because the cache key includes the data directory and filename arguments to `load_data`.
