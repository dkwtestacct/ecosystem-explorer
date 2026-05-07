# Ecosystem Explorer — One-Page Summary

## What it does

The Ecosystem Explorer simulates how reallocating developed urban land across three land-use types (green infrastructure, food forest, high-density development) affects a basket of ecological, social, and economic outcomes simultaneously. Built for early-stage planning conversations and tradeoff exploration — not for precise prediction or final investment decisions.

---

## Three land-use types

| Land Use | NLCD Code | Best for |
|----------|-----------|---------|
| Green Infrastructure | 90 (woody wetlands) | Flood reduction, carbon |
| Food Forest | 41 (deciduous forest proxy) | Cooling, food production, carbon, NDVI |
| High Density | 24 (developed high intensity) | — (worst on most ecological metrics) |

---

## 10 metric cards across three categories

The Scenario tab is organized as **Ecological (5) · Human & Social (2) · Economic (3)**. Every tooltip starts with a one-line confidence label so users can gauge methodological strength at a glance.

### Ecological (5)

| Metric | Unit | Confidence label |
|--------|------|-----------|
| Flood Risk Reduction | index 0–100 (`100 − mean_CN`) | Raster-based calculation |
| Temperature Change | °F vs unmodified baseline | Raster-based calculation |
| Runoff Volume | acre-feet per 2-inch design storm | Raster-based calculation |
| Carbon Sequestration | t CO2e/yr from converted pixels (k notation above 1,000) | Provisional assumption |
| NDVI | mean vegetation index 0–1 | Synthetic proxy |

### Human & Social (2)

| Metric | Unit | Confidence label |
|--------|------|-----------|
| Nature Access | % of residents within ~800 m of nature | Proximity estimate |
| Urban Wellbeing Score | weighted composite 0–1 (NDVI + cooling + access) | Composite proxy |

### Economic (3)

| Metric | Unit | Confidence label |
|--------|------|-----------|
| Food Production | M lbs/yr from food forest pixels | Provisional assumption |
| Est. Implementation Cost | $M total ($/acre slider × converted area) | Order-of-magnitude estimate |
| Cost Effectiveness | $/ac-ft prevented · $/°F cooling · $/1,000 people fed (three sub-ratios) | Order-of-magnitude estimate |

---

## How the metrics are computed

- **Flood Risk Reduction & Runoff Volume** — USDA SCS Curve Number method per pixel (NLCD land cover × SSURGO soil hydrologic group). Aggregated to a city-wide mean CN, converted to runoff depth via $S = 1000/CN - 10$ and $R = (P - 0.2S)^2 / (P + 0.8S)$ for a 2-inch design storm, then scaled by total developed acreage.
- **Temperature Change** — InVEST Urban Cooling Model HM index (`HM = (shade + kc) / 2` per pixel), converted to °F via a 4 °F/HM-unit factor (literature midpoint, ±2 °F uncertainty).
- **Carbon Sequestration** — Counts only newly converted pixels × per-cover rates from `CARBON_SEQ_RATES` (default 3.5 / 2.0 / 0.0 t CO2e/acre/yr for FF / GI / HD). Rates are user-overridable in Advanced Settings.
- **NDVI** — Synthetic proxy assigned per-NLCD-code (woody wetlands 0.70, food forest 0.75, high-density 0.10). Not derived from satellite imagery.
- **Nature Access** — Euclidean distance transform of the green-pixel mask × 30 m, thresholded at 800 m. Population from US Census 2020 block totals (Hennepin County, joined to TIGER 2020 blocks, rasterized to the NLCD grid). Reports `% of residents within 800 m of nature`. Proximity proxy, not a walkshed model.
- **Urban Wellbeing Score** — Weighted composite of normalized NDVI, HM, and Nature Access %. Default weights (0.2 / 0.4 / 0.4) sum to 1.0; sliders in Advanced Settings let users retune. Not a validated mental-health model.
- **Food Production** — Food-forest pixel count × 0.222 acres/pixel × 11,500 lbs/acre/year (NatCap benchmark, mature managed system).
- **Est. Implementation Cost** — Sum of converted acres × per-acre cost slider for each land-use class.
- **Cost Effectiveness** — Implementation cost ÷ benefit per metric. Returns N/A when the denominator is zero or negative.

---

## How to use it (the headline features)

- **Scenario tab** — slider-driven scenario builder showing all 10 metric cards plus a collapsible **Baseline vs Scenario Comparison** table that color-codes improvements (green) vs regressions (red, with runoff treated as inverse).
- **Tradeoff Analysis tab** — Plotly chart of the entire scenario space with the active scenario, saved scenarios, optimizer suggestions, and Pareto frontier overlaid. Below it: **Best Scenarios by Goal** (five canonical winners drawn from the 2,541-entry pre-computed library, each with an Apply button), then a Save-this-scenario flow that prompts inline for a custom name and adds the result to a downloadable CSV.
- **Map View tab** — spatial map of where conversions occur, plus a heat-vulnerability red-wash overlay slider (default opacity 0.3).
- **Smart Scenario Search optimizer** — Random Forest surrogate trained on the live scenario grid; samples 10,000 random (pct, GI%, FF%) combinations against user-set minimums on flood, cooling, food, and carbon, and returns up to 5 Pareto-efficient suggestions.
- **Advanced Settings (sidebar)** — overrides for Food Forest carbon rate, Green Infrastructure carbon rate, three Urban Wellbeing Score weight sliders, and a **Model Quality Mode** radio (Fast prototype / Balanced / High resolution) that swaps the surrogate's training set between 90, ~726, and 2,541 scenarios with corresponding tree-count adjustments.
- **Named saved scenarios + CSV export** — every saved scenario gets a user-chosen `display_name` that appears in the saved-scenarios table, the chart hover labels, and the exported CSV (primary download button at the top of the Saved Scenarios expander).

---

## Key assumptions and limitations

- **Extent caveat for Nature Access** — the NLCD raster covers only ~10.8 km × 10.7 km of downtown Minneapolis (~154 k residents in extent), not the full city.
- **Spatial placement is stylized** — converted pixels are picked randomly (or heat-weighted), not by parcel feasibility, ownership, corridor design, or zoning.
- **Carbon and food rates are provisional** — defaults come from broad regional benchmarks (USDA NRCS, NatCap), not site-specific data.
- **Temperature calibration not locally validated** — the 4 °F/HM-unit factor is a literature midpoint; ±2 °F uncertainty.
- **Wellbeing weights are arbitrary** until empirically validated against a local quality-of-life dataset. Treat as directional only — explicitly not a mental-health model.
- **Implementation costs are illustrative** — per-acre values are order-of-magnitude placeholders.
- **Surrogate covers ecological + carbon outcomes only** — not wellbeing, not cost, not heat-priority placement effects. Verify any promising surrogate suggestion through the main sliders.

---

## Computation architecture

| Layer | Purpose | Scale |
|-------|---------|-------|
| Full-resolution raster simulation | Pixel-level biophysical truth | Per-scenario, on demand |
| Pre-computed lookup table | Instant slider response | 2,541 (pct, GI%, FF%) entries at step=5 |
| Pre-computed dense grid (`data/scenarios_dense.csv`) | Optional surrogate training set | 726 scenarios at finer step=5/10 (built offline by `precompute_scenarios.py`) |
| Random Forest surrogate | Rapid scenario search at query time | Trains on 90 / ~726 / 2,541 rows depending on Model Quality Mode |
| Surrogate sampling at optimization time | Explore wide tradeoff space | ~10,000 random candidates per Optimize click |
