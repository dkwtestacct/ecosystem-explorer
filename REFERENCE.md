# Ecosystem Explorer — UI Reference

Every visible element in the app, what it means, and how it is computed.

---

## Metric Cards — Row 1 (three columns)

### Flood Risk Reduction

| Field | Detail |
|-------|--------|
| **Represents** | A unitless index of how much the scenario reduces storm runoff potential relative to fully paved land. Higher is better. |
| **Formula** | `100 − mean_CN`, where `mean_CN` is the area-weighted average SCS Curve Number across all pixels. |
| **Data source** | `UFR_biophysical_table_MN.csv` (CN values by lucode × soil group), `soil_group_MN.tif`, `LULC_NLCD_2021_MN.tif`. |
| **Delta** | Difference vs baseline (`BASELINE_CN = 75.7` → baseline index = 24.3). |
| **Caveats** | Not a direct percentage reduction in runoff volume — it is a CN-derived index. Green infrastructure (NLCD 90) drives the largest improvements; high-density development (NLCD 24) makes it worse. |

---

### Temperature Change

| Field | Detail |
|-------|--------|
| **Represents** | Approximate air-temperature change vs the unmodified baseline; positive = cooler, negative = warmer. |
| **Formula** | `ΔHM × 4.0 °F`, where `ΔHM = mean_HM_scenario − BASELINE_HM (0.2719)` and `HM = (shade + kc) / 2` averaged over all pixels. |
| **Data source** | `biophysical_table_urban_cooling.csv` (shade and kc columns per NLCD lucode). |
| **Delta** | Raw HM values for the scenario and baseline shown side-by-side (e.g. "HM 0.3142 vs 0.2719"). |
| **Caveats** | Calibration factor of 4 °F/HM unit is approximate for Minneapolis; accuracy is roughly ±2 °F. The HM index is a proxy, not a direct temperature measurement. |

---

### Runoff Prevented

| Field | Detail |
|-------|--------|
| **Represents** | Total stormwater runoff volume generated under this scenario's land cover for a 2-inch design storm, across all developed land. Lower = less runoff, i.e. more is being retained. |
| **Formula** | SCS method: `S = (1000/CN) − 10`, `Ia = 0.2S`, `Q_in = (P−Ia)²/(P−Ia+S)`, then `(Q_in/12) × total_developed_acres` in acre-feet. `P = 2.0 inches`. |
| **Data source** | Derived from `mean_CN` (see Flood Risk Reduction) and total developed acreage (`n_developed_pixels × 0.222 acres`). |
| **Caveats** | Shows the scenario runoff volume, not an explicit delta from baseline. The cost-effectiveness ratio "Cost / Acre-Foot Prevented" is where the baseline delta is computed. A 2-inch storm represents a common minor design event; results for larger storms will differ. |

---

## Metric Cards — Row 2 (two columns)

### Food Production

| Field | Detail |
|-------|--------|
| **Represents** | Annual food yield from pixels converted to food forest land cover, in millions of pounds per year. |
| **Formula** | `n_food_forest_pixels × 0.222 acres/pixel × 11,500 lbs/acre/year ÷ 1,000,000`. |
| **Data source** | Pixel count from the scenario LULC raster (NLCD code 41 pixels). Yield benchmark from San Antonio NatCap study. |
| **Delta** | People fed, computed as `total_lbs ÷ 2,000 lbs/person/year`. |
| **Caveats** | 11,500 lbs/acre/year is a benchmark estimate, not site-specific. NLCD code 41 (deciduous forest) is used as a proxy for food-producing tree cover. Actual yield depends on species mix, management, and soil quality. |

---

### Est. Implementation Cost

| Field | Detail |
|-------|--------|
| **Represents** | Rough order-of-magnitude total capital cost to implement all conversions in the scenario. |
| **Formula** | `(n_GI_pixels × 0.222 × cost_GI) + (n_FF_pixels × 0.222 × cost_FF) + (n_HD_pixels × 0.222 × cost_HD)`, divided by 1,000,000 for $M display. |
| **Data source** | Pixel counts from the scenario LULC; per-acre costs from the sidebar sliders. |
| **Caveats** | Does not include maintenance, land acquisition, or displacement costs. Defaults ($50k/acre GI, $10k/acre FF, $5k/acre HD) are rough midpoints; use the sliders to reflect local conditions. |

---

## Metric Cards — Row 3: Cost-Effectiveness Ratios

All three ratios show **N/A** when the denominator is zero or negative (no improvement vs baseline) or when total cost is zero (no conversions).

### Cost / Acre-Foot Prevented

| Field | Detail |
|-------|--------|
| **Represents** | Implementation dollars spent per acre-foot of runoff reduction vs the unmodified baseline. |
| **Formula** | `total_cost_$ ÷ (BASELINE_RUNOFF_ACFT − scenario_runoff_acft)`. |
| **Caveats** | N/A if the scenario increases runoff (e.g. all high-density). Lower is better. |

---

### Cost / °F Cooling

| Field | Detail |
|-------|--------|
| **Represents** | Implementation dollars spent per degree Fahrenheit of cooling vs baseline. |
| **Formula** | `total_cost_$ ÷ cooling_f`, where `cooling_f` is the °F delta (positive = cooler). |
| **Caveats** | N/A if the scenario is warmer than baseline. Inherits the ±2 °F uncertainty of the HM-to-temperature calibration. |

---

### Cost / 1,000 People Fed

| Field | Detail |
|-------|--------|
| **Represents** | Implementation dollars spent per 1,000 people whose annual food needs could be met by the scenario's food forest yield. |
| **Formula** | `total_cost_$ ÷ (people_fed ÷ 1,000)`. |
| **Caveats** | N/A if no food forest pixels are present. Inherits the food yield benchmark uncertainty. |

---

## Sidebar Controls

### City

| Field | Detail |
|-------|--------|
| **Type** | Selectbox |
| **Options** | All keys from the `CITIES` dict; unavailable cities are labelled "(coming soon)". |
| **Effect** | Selecting an unavailable city shows a warning and halts execution via `st.stop()`. Selecting an available city sets data paths and baseline constants for all downstream computation. |

---

### % of Developed Land to Convert

| Field | Detail |
|-------|--------|
| **Type** | Slider, range 0–50, step 1 |
| **Effect** | Determines how many developed pixels (NLCD 21/22/23) are randomly sampled for conversion. At 50%, half of all developed pixels are converted. |
| **Caveats** | Pixels are chosen randomly (or heat-priority weighted); spatial placement is not optimised for corridors or parcels. |

---

### Green Infrastructure % / Food Forest % / High Density %

| Field | Detail |
|-------|--------|
| **Type** | Number inputs, range 0–100, step 5; must sum to exactly 100 |
| **Effect** | Splits the converted pixels into three groups. Green Infrastructure → NLCD 90 (woody wetlands). Food Forest → NLCD 41 (deciduous forest proxy). High Density → NLCD 24 (developed, high intensity). |
| **Caveats** | High Density auto-fills as the remainder but can be manually overridden. The app shows a warning and stops if the three values do not sum to 100. |

---

### Green Infrastructure ($/acre) / Food Forest ($/acre) / High Density Infill ($/acre)

| Field | Detail |
|-------|--------|
| **Type** | Sliders; GI $5k–$150k (default $50k), FF $1k–$50k (default $10k), HD $1k–$50k (default $5k) |
| **Effect** | Scales the implementation cost calculation. Does not affect ecological outcomes (CN, HM, food). |
| **Caveats** | The lookup table is pre-computed at default costs; cost is recomputed from pixel counts and current slider values on every interaction. |

---

### Target Heat-Vulnerable Areas (toggle)

| Field | Detail |
|-------|--------|
| **Type** | Toggle, default off; under subheader "Spatial Priority" |
| **Effect** | When on, converted pixels are sampled with probability weights: NLCD 23 → 1.0, NLCD 22 → 0.6, NLCD 21 → 0.3. Concentrates interventions in higher-intensity developed areas. Negative or NaN weights are clamped to 0 before normalisation. |
| **Caveats** | This is a land-use intensity proxy, not a measured heat or socioeconomic vulnerability index. The lookup table (which uses uniform random placement) is bypassed in this mode; scenarios are computed live. Intended as a placeholder until CDC/ATSDR HVI data is integrated. |

---

### Example Scenario Buttons

| Button | Sets |
|--------|------|
| 🌳 Food Forest (Cooling + Food Focus) | 10% converted, 100% food forest |
| 🌊 Green Infrastructure (Flood Mitigation) | 10% converted, 100% green infrastructure |
| 🏙️ High Density Development | 10% converted, 100% high density |

Buttons work by writing to `st.session_state` and calling `st.rerun()`.

---

### Find Best Scenario (Optimizer)

| Control | Detail |
|---------|--------|
| **Min flood reduction** | Slider 0–90, step 5. Minimum acceptable Flood Risk Reduction index. |
| **Min cooling (HM)** | Slider 0.0–1.0, step 0.05. Minimum acceptable mean HM value (not °F delta). |
| **Min food production (M lbs)** | Slider 0.0–MAX_FOOD, step 0.01. Minimum acceptable food production. |
| **Optimize button** | Samples ~10,000 random (pct, GI%, FF%) combinations, predicts outcomes with the RF surrogate, filters to those meeting all three minimums, computes the Pareto front, de-duplicates near-identical points, and returns up to 5 top suggestions ranked by a balanced score: `flood/100 + HM/1.1 + food/MAX_FOOD`. |
| **Caveats** | Surrogate predictions carry uncertainty (shown as 10th–90th percentile bands). Cost and heat-priority mode are not inputs to the surrogate — apply those after selecting a suggestion. |

---

## Tradeoff Chart (Tradeoff Analysis tab)

Built with Plotly. X and Y axes show the two primary ecological metrics; bubble size encodes food production for saved and optimised scenarios only.

### Axes

| Element | Detail |
|---------|--------|
| **X-axis: Flood Risk Reduction** | `100 − mean_CN`. Range fixed 0–100. Higher = better. |
| **Y-axis: Heat Mitigation Index** | Mean HM across all pixels. Range fixed 0–1.1. Higher = better (more cooling potential). Note: this is the raw HM value, not the °F delta shown in the metric card. |

---

### Reference Benchmarks (coloured markers)

Four fixed points representing extreme all-one-landcover scenarios for Minneapolis. Hardcoded in `REF_SCENARIOS`; will need to be made city-specific when new cities are added.

| Benchmark | Flood | HM | Interpretation |
|-----------|-------|----|----------------|
| Baseline | 24.3 | 0.2719 | Current unmodified land cover |
| All Food Forest (NLCD 41) | 29.9 | 0.8284 | Every developed pixel → deciduous forest |
| All Green Infra (NLCD 90) | 83.0 | 0.8633 | Every developed pixel → woody wetlands |
| All High Density (NLCD 24) | 18.8 | 0.1923 | Every developed pixel → high-intensity development |

---

### Convex Hull (feasible space)

| Element | Detail |
|---------|--------|
| **Represents** | The boundary of all outcomes reachable across the pre-computed scenario grid and lookup table. |
| **Method** | `scipy.spatial.ConvexHull` on (flood_reduction, mean_HM) coordinates of all pre-computed scenarios. |
| **Caveats** | The hull is over the discrete grid, not a continuous surface — scenarios near the hull boundary are achievable but the interior is not uniformly dense. |

---

### Current Scenario (purple star)

The outcome of the current slider settings. Crosshair lines extend from the star to both axes for easier reading.

---

### Saved Scenarios (purple circles)

Scenarios explicitly saved by clicking "Save this scenario." Bubble size scales with food production (`base + 120 × sqrt(food / MAX_FOOD)`). Hovering shows the scenario name and all three metric values.

---

### Pareto Frontier (gold markers + dashed line)

| Element | Detail |
|---------|--------|
| **Represents** | The subset of saved scenarios where no other saved scenario is strictly better on all three metrics simultaneously (flood, HM, food). |
| **Method** | Iterative dominance check: a point is Pareto-efficient if no other point dominates it on all three dimensions. |
| **Caveats** | Computed over saved scenarios only; adding more saved scenarios can change which points are on the frontier. |

---

### Optimised Suggestions (orange diamonds)

Returned by the surrogate optimizer after clicking "Optimize." Error bars show the 10th–90th percentile range across Random Forest trees for flood (horizontal) and HM (vertical).

---

## Bar Charts (Scenario tab)

Three side-by-side bars comparing the current scenario to the unmodified baseline on the three primary metrics.

| Chart | Y-axis | Baseline value | Scenario value |
|-------|--------|---------------|----------------|
| Flood Risk | Mean Curve Number (lower = less runoff) | `BASELINE_CN` (75.7) | `mean_CN` from scenario |
| Urban Cooling | Heat Mitigation Index (higher = more cooling) | `BASELINE_HM` (0.2719) | `mean_HM` from scenario |
| Food Production | Million lbs/year | 0.0 (no food forest in baseline) | `food_mln_lbs` from scenario |

A horizontal dashed line at the baseline value is drawn on the Flood Risk and Cooling charts for visual reference. The Food Production y-axis is scaled to `MAX_FOOD × 1.1` across all pre-computed scenarios.

---

## Map View (Map View tab)

Pixel-level raster showing which developed pixels changed and to what land cover.

| Colour | Meaning |
|--------|---------|
| Gray (`#d3d3d3`) | Developed pixel, unchanged |
| Teal (`#2196a0`) | Converted to Green Infrastructure (NLCD 90) |
| Green (`#4caf50`) | Converted to Food Forest (NLCD 41) |
| Red (`#e53935`) | Converted to High Density (NLCD 24) |
| White | Outside city boundary (`NODATA = -128`) |

When heat-priority mode is on, teal/green/red pixels are concentrated in higher-intensity developed areas rather than uniformly distributed.
