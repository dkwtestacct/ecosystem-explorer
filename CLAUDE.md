# Ecosystem Explorer — CLAUDE.md

## What the app does

Streamlit app that lets users explore tradeoffs between converting developed urban land into
green infrastructure, food forests, or high-density development. For a selected city and
conversion scenario, it computes:

- **Flood risk** via the SCS Curve Number method
- **Urban cooling** via a Heat Mitigation (HM) index
- **Food production** from a food-forest yield benchmark
- **Implementation cost** from per-acre cost sliders
- **Cost-effectiveness ratios** (cost per acre-foot of runoff prevented, per °F cooling, per 1k people fed)

It also runs a pre-computed scenario grid to train a Random Forest surrogate model, which
powers a Pareto-optimal scenario optimizer (set minimum targets → get ranked suggestions).

Run with: `streamlit run app.py`

---

## Data files

All data lives under `data/`. Each city gets its own subdirectory pair.

### Minneapolis (current)

| File | Description |
|------|-------------|
| `data/flood/LULC_NLCD_2021_MN.tif` | Land use / land cover raster (NLCD 2021) used for CN calculation |
| `data/flood/soil_group_MN.tif` | Hydrologic soil group raster (values 1–4 = A/B/C/D) |
| `data/flood/UFR_biophysical_table_MN.csv` | Curve numbers by NLCD lucode × soil group (CN_A, CN_B, CN_C, CN_D). Includes NLCD code 82 (Cultivated Crops). |
| `data/flood/Damage_loss_table_MN.csv` | Not currently used in the app |
| `data/cooling/land_use_2021.tif` | Land use raster used for HM index and spatial scenario mapping |
| `data/cooling/biophysical_table_urban_cooling_MN.csv` | shade, Kc, albedo per lucode. Includes NLCD code 82 (Cultivated Crops). |
| `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif` | Reference ET raster (1 km, MN-only). Bilinear-resampled to the 30 m NLCD grid; used in the InVEST CC formula's ETI term. |
| `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/energy_consumption.csv` | Per-building-type AC consumption rate (kWh/m²/yr) for the cooling-energy-savings dollar metric. |
| `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/buildings.shp` | InVEST-sample building footprints with `type` ∈ {0=other, 1=commercial, 2=residential, 3=industrial}, rasterized at startup as `BUILDINGS_TYPE_RASTER`. |

### San Antonio (in progress — `available: False`)

| File | Description | Status |
|------|-------------|--------|
| `data/sa/flood/lulc_nlcd_2021_sa.tif` | Raw NLCD 2021 clipped to SA bbox via MRLC WCS (EPSG:5070, 30 m, 1984×1713 px) | done |
| `data/sa/flood/land_use_2021_sa.tif` | Canonical SA LULC raster (same CRS/grid) | done |
| `data/sa/flood/UFR_biophysical_table_SA.csv` | CN values by lucode × soil group | placeholder copy of MN |
| `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` | shade / Kc / albedo per lucode | placeholder copy of MN — SA's hotter/drier climate likely needs lower Kc and shade values for some classes (pending NatCap project tuning) |
| `data/sa/flood/soil_group_SA.tif` | SSURGO hydrologic soil group rasterized to LULC grid | pending |
| `data/sa/cooling/reference_evapotranspiration_annual.tif` | Reference ET raster | pending |
| `data/sa/population/sa_pop_2020.tif` | Census 2020 block totals rasterized to LULC grid | pending |
| `data/sa/flood/buildings.shp`, `roads.shp`, `tracts.shp` | Bexar County vector layers | pending |
| `data/sa/flood/Damage_loss_table_SA.csv`, crop-yield table | SA-specific damage rates and crop yields | pending |

LULC fetch + baseline-constant computation is reproducible via
`download_sa_data.py`. Detailed sourcing notes are in `data/sa/README.md`.

**Canonical CRS for San Antonio: EPSG:5070** (NAD83 / Conus Albers, NLCD's
native equal-area CRS). Differs from Minneapolis (EPSG:26915 / UTM 15N) —
equal-area is preferred for SA's larger area-based analyses.

**Biophysical-table naming convention:** every city has its own CN and
cooling tables, suffixed with the city's two-letter code (`_MN`, `_SA`).
Each `CITIES` entry declares its filenames via `cn_table_file` and
`cooling_table_file`; `load_data` joins these against the city's
`data_dir_flood` / `data_dir_cooling`. Even when SA's values are still
copies of MN, the per-city files exist so future climate-specific tuning
doesn't risk affecting Minneapolis.

---

## City configuration

Cities are defined in the `CITIES` dict near the top of `app.py`. Each entry:

```python
CITIES = {
    'City Name, ST': {
        'data_dir_flood':     'data/<city>/flood',   # path to flood data directory
        'data_dir_cooling':   'data/<city>/cooling', # path to cooling data directory
        'cn_table_file':      'UFR_biophysical_table_<XX>.csv',
        'cooling_table_file': 'biophysical_table_urban_cooling_<XX>.csv',
        'baseline_cn':        <float>,               # mean CN of the unmodified LULC
        'baseline_hm':        <float>,               # mean HM of the unmodified LULC
        'crs':                '<EPSG code>',         # canonical CRS for this city
        'available':          True | False,          # False = show "coming soon", block execution
    },
}
```

**To add a new city:**
1. Add an entry to `CITIES` with `available: False` until data is ready.
2. Place the required data files in the declared directories.
3. Compute baseline CN and HM from the unmodified LULC and fill them in.
4. Set `available: True`.

City selection happens in the sidebar **before** data loading. When a city is selected,
`DATA_DIR_FLOOD`, `DATA_DIR_COOLING`, `CN_TABLE_FILE`, `COOLING_TABLE_FILE`,
`BASELINE_CN`, and `BASELINE_HM` are set as module-level names from `city_cfg`.
All downstream functions reference these names at call time (standard Python
global resolution), so they automatically use the right values.

`load_data(data_dir_flood, data_dir_cooling)` is `@st.cache_data` — different cities get
separate cache entries via the path parameters.

---

## Key constants

### Global (not city-specific)

| Constant | Value | Meaning |
|----------|-------|---------|
| `PIXEL_AREA_ACRES` | 0.222 | Acres per raster pixel |
| `FOOD_FOREST_LBS_ACRE` | 11,500 | Food forest yield benchmark (lbs/acre/year) — from San Antonio NatCap study |
| `DESIGN_STORM_INCHES` | 2.0 | Rainfall depth used for the SCS runoff calculation |
| `UHI_MAX_C` | 2.05 | Minneapolis urban-heat-island max anomaly (°C). Source: InVEST `urban_cooling_model_args_MN.json`. Used in CC→ΔT conversion. Should become per-city `city_cfg['uhi_max_c']` once SA's InVEST args are available. |
| `HM_TO_FAHRENHEIT` | 3.69 | Derived as `UHI_MAX_C × 1.8`. 1 CC unit ≈ 3.69 °F cooling vs fully paved (Minneapolis). |
| `GREEN_AREA_COOLING_DISTANCE_M` | 450 | Gaussian convolution kernel radius for CC smoothing, from InVEST args JSON. `_CC_SIGMA_PX = 450/30 = 15` at 30 m NLCD resolution. |
| `COST_PER_KWH_USD` | 0.13 | US average residential electricity price (EIA 2024). Used to convert avoided-AC-kWh into $. |
| `PIXEL_AREA_M2` | 900 | NLCD 30 × 30 m pixel area in m². Used for cooling energy savings (consumption rate is kWh/(m²·°C)/yr from `energy_consumption.csv`). |
| `LBS_PER_PERSON_YEAR` | 2,000 | Average American food consumption used to convert lbs → people fed |
| `DEVELOPED_CODES` | [21, 22, 23] | NLCD lucodes treated as convertible developed land |
| `CODE_GREEN_INFRA` | 90 | NLCD lucode for woody wetlands (green infrastructure proxy) |
| `CODE_FOOD_FOREST` | 41 | NLCD lucode for deciduous forest (food forest proxy) |
| `CODE_HIGH_DENSITY` | 24 | NLCD lucode for high-intensity development |
| `NODATA` | -128 | Sentinel value in rasters marking outside-boundary pixels |

### City-specific (set at runtime from `city_cfg`)

| Name | Minneapolis value | San Antonio (preliminary) | Meaning |
|------|-------------------|---------------------------|---------|
| `BASELINE_CN` | 75.7 | 65.97 | Mean curve number of unmodified developed land |
| `BASELINE_HM` | 0.2719 | 0.2917 | Mean cooling capacity (CC = 0.6·shade + 0.2·albedo + 0.2·ETI per InVEST UCM) of unmodified developed land. Reported on the UI as "HM". |
| `DATA_DIR_FLOOD` | `data/flood` | `data/sa/flood` | Directory containing flood model inputs |
| `DATA_DIR_COOLING` | `data/cooling` | `data/sa/cooling` | Directory containing cooling model inputs |

SA values are computed by `download_sa_data.py`. They are preliminary:
CN uses CN_B as a default soil group until SSURGO is rasterized, and HM
uses a `0.6·shade + 0.2·albedo + 0.2·kc` proxy until SA gets its own
reference-ET raster (Minneapolis already uses the full InVEST CC formula
with ET; SA's value will shift slightly once ET is wired in).
`available` is still `False` in `CITIES['San Antonio, TX']`.
| `BASELINE_RUNOFF_ACRE_FEET` | computed | Runoff from baseline CN over a 2-inch storm; used for cost-effectiveness ratios |

### Cost defaults ($/acre, adjustable via sidebar sliders)

| Constant | Default | Represents |
|----------|---------|------------|
| `DEFAULT_COST_GI` | $50,000 | Green infrastructure / constructed wetlands |
| `DEFAULT_COST_FF` | $10,000 | Food forest establishment |
| `DEFAULT_COST_HD` | $5,000 | Marginal high-density infill |

---

## Architecture notes

- **Lookup table** (`compute_lookup_table`): pre-computes all valid slider positions at step=5
  for instant UI response. The scenario grid (step=10/25) is used only for surrogate training.
- **Surrogate model**: Random Forest trained on the scenario grid; used by the optimizer to
  search ~10k random scenarios for Pareto-optimal suggestions. Uncertainty bands come from
  10th/90th percentile across RF trees.
- **Equity weighting**: `equity_weights` raster weights high-intensity developed pixels (NLCD 23)
  higher for the heat-priority conversion mode. Currently a proxy; TODO is to replace with a
  real CDC/ATSDR Heat Vulnerability Index by census tract.
- **`REF_SCENARIOS`**: hardcoded Minneapolis benchmark points (all-one-landcover extremes) shown
  on the tradeoff plot. Will need to become city-specific when new cities are added.
- **InVEST Urban Cooling Model**: `_compute_cc_raster` computes per-pixel CC = `0.6·shade + 0.2·albedo + 0.2·ETI`,
  then **Gaussian-smooths the result with σ = 15 px (450 m)** to spatially propagate cooling onto
  neighbouring pixels per InVEST's `green_area_cooling_distance` step. The mean of this smoothed
  CC raster is the `mean_hm` reported in scenario results (UI label: "Cooling Capacity" / "CC").
  `compute_cooling_energy_savings(cc_raster)` converts ΔCC → ΔT °C (× `UHI_MAX_C`) →
  kWh saved (× `consumption_rate × pixel_area`) → $/yr (× `COST_PER_KWH_USD`). Per-pixel,
  not per-building polygon — see `UCM_AUDIT.md` for the open-divergences list. Both functions
  are called inside `evaluate_scenario`. Module-level precompute: `ET_RESIZED`, `MAX_ET_REF`,
  `BUILDINGS_TYPE_RASTER`, `CONSUMPTION_RATE_PER_PIXEL`, `_BASELINE_HM_RASTER` (the smoothed
  baseline CC).
- **OSM road exclusion**: Road footprints are unioned into `BUILDINGS_RASTER` so the
  convertible-pixels pool excludes both buildings and impassable surfaces.
  `download_osm_minneapolis.py` fetches the Geofabrik Minnesota state extract, clips to the
  AOI, and applies **Option B class filter** (`ROADS_DROP_CLASSES`) — drops `footway`,
  `cycleway`, `steps`, `service`, `path`, `pedestrian`, `unclassified`, `track*`. These are
  sub-pixel-width surfaces that would over-count the non-convertible mask at 30 m NLCD
  resolution. Retained set: motorway, trunk, primary, secondary, tertiary, residential,
  living-street, and on/off-ramp links — **5,495 segments covering ~29 % of AOI**. After
  unioning with buildings, **~65 % of developed pixels (NLCD 21–24) remain convertible**
  (33,357 of 51,430). Rasterization is unbuffered line-to-pixel via `rasterio.features.rasterize`
  with `dtype="uint8"`; output is binary 0/1.
- **`SCENARIO_SCHEMA_VERSION = 11`** — bump on every change that shifts `evaluate_scenario`
  outputs so cached lookup tables get regenerated. Recent bumps: 7→8 (UCM rework: ET fix,
  Gaussian convolution, canonical energy formula); 8→9 (ET nodata sentinel masked);
  9→10 (full Geofabrik OSM road network, 62 % AOI); 10→11 (Option B road filter, ~29 % AOI).

---

## Blocked / pending work

- **Full Minneapolis extent integration — BLOCKED on SSURGO soil raster.** A 374 × 607 px expanded
  NLCD raster covering the full 148.9 km² city boundary has been downloaded to
  `data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif` (gitignored) but is not wired into the
  app. Current `data/flood/soil_group_MN.tif` covers only ~71 km² of the 204 km² expanded extent.
  Need full Hennepin County SSURGO data via <https://websoilsurvey.nrcs.usda.gov>, rasterized to
  match the new grid. See `check_expanded_coverage.py` for the coverage analysis. Population
  raster and OSM clip would also need regeneration on the new grid.

---

## Coding conventions

- **No bare globals for city data** — always pull city-specific values from `city_cfg` or the
  derived runtime names (`BASELINE_CN`, `BASELINE_HM`, etc.). Don't hardcode Minneapolis values
  outside of the `CITIES` dict.
- **Cached functions use path params as cache keys** — `load_data`, `compute_scenario_grid`,
  `compute_lookup_table`, and `train_surrogate` all accept the data directory paths so Streamlit
  caches city results separately. `compute_scenario_grid` and `compute_lookup_table` also take a
  `schema_version=SCENARIO_SCHEMA_VERSION` arg — bump that constant whenever the surrogate-target
  columns change (e.g., adding a new metric to `evaluate_scenario`'s return dict) and Streamlit
  will automatically invalidate cached grids and lookup tables. Both functions also assert the
  presence of `REQUIRED_TARGET_COLUMNS` so a missing column fails loudly instead of producing a
  `KeyError` deep inside `train_surrogate`. `train_surrogate` additionally takes `mode_key` and
  `n_estimators` args, both of which participate in the cache key so the Model Quality radio
  in Advanced Settings retrains automatically when the user changes mode.
- **Three Model Quality modes (Fast prototype / Balanced / High resolution)** — selected via
  the Advanced Settings radio (`st.session_state['model_quality']`). The mode determines:
  (1) `scenario_df` source — Fast prototype uses `compute_scenario_grid(step_pct=10,
  step_alloc=25)` (~90), Balanced prefers `data/scenarios_dense.csv` else
  `compute_scenario_grid(step_pct=5, step_alloc=10)` (~726), High resolution reuses the
  2,541-entry lookup table as training data (free — those rows are already computed for
  instant slider response); (2) `n_estimators` via `SURROGATE_TREES = {"Fast prototype":
  100, "Balanced": 200, "High resolution": 300}`. The tree count is **intentionally hidden
  from the UI** — only the training-scenario story is shown to users. The Balanced default
  CSV is built offline by `precompute_scenarios.py`, which stubs `streamlit` so it can
  `import app` and reuse `evaluate_scenario`, `_compute_carbon`, `calculate_nature_access`,
  and `pop_count_raster` without duplicating logic. **Conceptual separation:** training
  scenarios (1) and tree count (2) are surrogate-side knobs; the optimizer's ~10,000
  random candidate samples at search time are independent and unchanged across modes.
- **N/A over division errors** — cost-effectiveness ratios return `None` (displayed as "N/A")
  when the denominator is zero or negative. Never let a divide-by-zero surface to the user.
- **Metric formatters are helpers, not inline f-strings** — use `_fmt_runoff()`, `_fmt_food()`,
  `_fmt_people()`, `_fmt_ce()` for display formatting so the logic lives in one place.
- **`st.stop()` for unavailable cities** — always call it before data loading, not after.
  Nothing expensive should run for an unavailable city.
- **Scenario LULC is not stored in the lookup table** — `scenario_lulc` is stripped from cached
  results (`if k != 'scenario_lulc'`) and recomputed on demand for the map view to keep memory
  usage manageable.
- **Metric cards are grouped into four labeled sections** — 🌿 Ecological (5 cards in two
  rows: row 1 has Flood Risk Reduction, Temperature Change, and Runoff Volume in 3 columns;
  row 2 has Carbon Sequestration and NDVI in 2 columns),
  👥 Human & Social (only Nature Access for now — Mental Health Index has been removed
  pending real implementation, and NDVI moved to Ecological since it's a vegetation
  measure, not a social one),
  📈 Economic (2 cards), 📊 Cost Effectiveness (3 cards). Each group is separated by
  `st.divider()`. Keep this grouping when adding new metrics — place new cards in the section
  that matches their category rather than appending to a flat list.
- **NDVI is a synthetic proxy** — values come from a per-NLCD-code lookup
  (`NDVI_PROXY` plus `NDVI_OTHER_DEVELOPED` / `NDVI_OTHER_NATURAL` defaults), not from
  satellite imagery. `BASELINE_NDVI` is computed once at startup from the unmodified
  `cooling_lulc` raster; scenario `mean_ndvi` is computed inside `evaluate_scenario` and
  flows through the lookup table and any cached scenario results.
- **Carbon sequestration counts converted pixels only** — `CARBON_SEQ_RATES` maps the three
  conversion target codes (`CODE_FOOD_FOREST`, `CODE_GREEN_INFRA`, `CODE_HIGH_DENSITY`) to
  provisional regional USDA/IPCC rates in tons CO2e/acre/yr (3.5, 2.0, 0.0). Inside
  `evaluate_scenario`, `carbon_tons_co2_yr` is computed inline from `n_for`, `n_wet`, `n_hd`
  and pixel area — there is no per-cell raster pass and no startup baseline (baseline = 0,
  same convention as `food_mln_lbs`). The value flows through the lookup table and cached
  scenario results. Treat as directional only — not locally calibrated.
- **Carbon rates are user-overridable** — the sidebar `⚙️ Advanced Settings` expander
  exposes `carbon_rate_ff` and `carbon_rate_gi` sliders backed by `st.session_state`. Both
  main-panel `evaluate_scenario` calls (the lookup-refresh `_fresh` and the heat-priority
  branch) pass these values through; `evaluate_scenario` falls back to `CARBON_SEQ_RATES`
  defaults when the kwargs are `None`. The precomputed lookup table is built with defaults,
  but `carbon_tons_co2_yr` is recomputed live in the lookup-refresh path so slider changes
  always take effect.
- **Urban Wellbeing Score is a weighted composite** — `compute_wellbeing_score(ndvi, hm,
  nature_pct, w_ndvi, w_cooling, w_nature)` returns
  `w_ndvi*ndvi + w_cooling*hm + w_nature*nature_pct/100` rounded to 3dp. Defaults
  `DEFAULT_WGT_NDVI=0.2`, `DEFAULT_WGT_COOLING=0.4`, `DEFAULT_WGT_NATURE=0.4` sum to 1.0.
  Three sliders in Advanced Settings (`wgt_ndvi`, `wgt_cooling`, `wgt_nature`) are read
  via `st.session_state.get(key, DEFAULT_*)` at both `evaluate_scenario` call sites — same
  pattern as the carbon rates, but using `.get()` with explicit defaults so first-run
  ordering doesn't matter. The Human & Social card recomputes `_baseline_wellbeing`
  every rerun against `BASELINE_NDVI`, `BASELINE_HM`, `BASELINE_NATURE_ACCESS_PCT`, and
  the **currently selected** weights, so changing weights doesn't make the delta
  misleading. Wellbeing is **not** in the surrogate's training targets — the optimizer
  doesn't search over it; cached scenario_df rows reflect default-weight wellbeing only.
  Help text explicitly disclaims any validated mental-health interpretation.
- **The surrogate predicts six outputs** — `train_surrogate` fits the Random Forest on
  `[flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet, carbon_tons_co2_yr,
  nature_access_pct]`, so `predict_with_uncertainty` returns `(n, 6)` arrays.
  `optimize_scenario` adds a `min_carbon` constraint (`mean_preds[:, 4] >= min_carbon`)
  alongside the existing flood, cooling, food, and runoff filters. The carbon column flows
  into the candidate DataFrame (with `carbon_lower` / `carbon_upper` uncertainty bands) and
  the optimizer results display, which renames it to "Carbon (tons CO2e/yr)". Note: the
  carbon surrogate is trained at the **default** rates baked into `scenario_df`; user
  overrides via Advanced Settings do not retrain the surrogate, so optimizer carbon
  predictions reflect default rates even when sliders are adjusted. Nature access is the
  6th output and carries an explicit caveat: the surrogate cannot see the spatial geometry
  that drives the metric (placement of converted pixels relative to existing parks and
  population centers), so its predictions are an indicative trend, not a precise spatial
  estimate.
- **Nature Access is a Euclidean-distance proximity proxy** —
  `calculate_nature_access(scenario_lulc, pop_count_raster)` runs
  `distance_transform_edt(~nature_mask) * PIXEL_SIZE_M` (30 m/pixel) over
  `NATURE_CODES = [41, 42, 43, 52, 71, 90, 95]`, marks pixels within
  `ACCESS_DISTANCE_M = 800` as "in access," and sums the per-pixel population counts
  inside that mask. Returns `(access_pct, people_with_access)`. Population data comes
  from `data/population/minneapolis_pop_2020.tif`, built by `download_census_pop.py`
  from US Census 2020 block-level totals (P1_001N for Hennepin County, FIPS 27053)
  joined to TIGER 2020 tabulation-block polygons and rasterized to the NLCD grid
  (each block's pop spread uniformly across its pixels). At startup `app.py` calls
  `load_population_data(...)` inside a `try/except (FileNotFoundError,
  RasterioIOError)`; on failure it falls back to a uniform `np.ones(...)` raster with
  `POPULATION_DATA_AVAILABLE = False` so the app still launches. The metric card's
  help text branches on the flag. **Extent caveat:** the NLCD template covers only
  ~10.8 km × 10.7 km (~154k residents) — a downtown-and-near-neighborhoods cutout,
  not all of Minneapolis. This is explicitly a proximity proxy: no street network,
  no barriers, no slope.
- **REFERENCE.md is not rendered in-app** — the sidebar uses `st.sidebar.link_button` to
  open `REFERENCE.md` on GitHub in a new tab rather than embedding the content inline. The
  GitHub URL is hardcoded; update it if the repo moves.
- **Post-optimize banner uses `st.session_state.just_optimized`** — set to `True` on a
  successful optimize, cleared on optimize-with-no-results or by the dismiss-X button.
  When the flag is set, two prompts render: a large success banner under the divider and
  an `st.info` line directly above the tab bar. **Do not auto-clear inside `with tab2:`**
  — Streamlit executes every `with tabX:` block on every rerun regardless of which tab
  is visible, so an auto-clear there fires on the next unrelated rerun instead of when
  the user actually opens the tab. Streamlit has no API for detecting tab switches, so
  the dismiss-X button (or running a new optimization) is the only way to clear the flag.
