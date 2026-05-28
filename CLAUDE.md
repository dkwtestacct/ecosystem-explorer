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

### San Antonio (`available: True` as of 2026-05-10)

| File | Description | Status |
|------|-------------|--------|
| `data/sa/flood/lulc_nlcd_2021_sa.tif` | Raw NLCD 2021 clipped to SA bbox via MRLC WCS (EPSG:5070, 30 m, 1984×1713 px) | done |
| `data/sa/flood/land_use_2021_sa.tif` | Prior canonical SA NLCD-only LULC raster (same CRS/grid). Kept as fallback/reference; the live SA pipeline now reads `land_use_compound_sa.tif` (Brief 27) and reduces to NLCD via the crosswalk. | done; superseded by compound LULC |
| `data/sa/flood/land_use_compound_sa.tif` | NatCap compound NLCD×NLUD×tree-canopy LULC reprojected from `data/sa/natcap_2024/lulc_overlay_3857.tif` to EPSG:5070 + nearest-neighbor resampled at 30 m (1984×1713). 800 unique compound lucodes in raster (of 1,984 possible per crosswalk). 1.06 % nodata at clipped extent edges. Brief 27 foundational adoption. | done (Brief 27) |
| `data/sa/natcap_2024/lulc_crosswalk.csv` | NatCap LULC crosswalk: each `lucode` (0–1983) maps to its constituent NLCD/NLUD/tree-canopy bins plus `is_realistic_to_create` flag. Loaded by `load_lulc_crosswalk()`; used to build the `COMPOUND_TO_NLCD` reduction lookup, the `COMPOUND_TO_NLCD_TREE` NLCD×tree-canopy reduction (SA flood CN path), and the three `COMPOUND_AFTER_*` per-target lookups. | done (Brief 27) |
| `data/sa/flood/biophys_floodmitig_sa.csv` | **Live SA flood CN table.** NatCap's San-Antonio-specific Curve Numbers, keyed by NLCD×tree-canopy 3-tier codes (e.g. 211/212/213 = Developed Open × low/med/high canopy; tier reduces CN as canopy rises). Single 2-digit codes for water/ice (11/12) and forests (41/42/43); plus SA scenario codes 997/998/999 (unreachable by current conversion path). 53 rows. Looked up via `reduce_compound_to_nlcd_tree(COMPOUND_TO_NLCD_TREE)` in `evaluate_scenario`'s SA branch; canopy mapping `tier = max(tree, 1)`. Design-storm-saturation framework — see NATCAP_COLLABORATION.md question 12. | done (2026-05-29) |
| `data/sa/flood/UFR_biophysical_table_SA.csv` | **Superseded** Minneapolis-placeholder CN table (NLCD-keyed, MN values). No longer wired (replaced by `biophys_floodmitig_sa.csv`); kept on disk for reference and still read by the one-time `download_sa_data.py` QA diagnostic. | superseded |
| `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` | Retired (Brief 28b). Kept on disk for reference; the per-class rationale sidecar `data/sa/cooling/biophysical_table_sources.md` documents the historical Köppen-BSh tuning. The live SA UCM path uses the compound table below. | retired |
| `data/sa/natcap_2024/ucm__nlcd_nlud_tree.csv` | NatCap compound NLCD×NLUD×tree-canopy UCM biophysical table (1,984 rows × 27 cols, keyed on compound `lucode` 0–1983). Provides per-pixel `shade`, `kc`, `albedo`, `green_area`, `building_intensity` indexed directly by the compound LULC raster. Referenced via SA's `cooling_table_file` config. | done (Brief 28b) |
| `data/sa/natcap_2024/una__nlcd_nlud_tree.csv` | NatCap compound NLCD×NLUD×tree-canopy UNA biophysical table (1,984 rows × 21 cols, keyed on compound `lucode` 0–1983). Provides per-pixel `urban_nature` ∈ {0.0, 0.5, 1.0} indexed directly by the compound LULC raster via the per-city `urban_nature_arr` numpy lookup. `search_radius_m` column is all zeros — the radius is an args-level scalar from `city_cfg['una_search_radius_m']`, not the per-row table value. Referenced via SA's `una_table_file` config. | done (Brief 29) |
| `data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv` | NatCap compound NLCD×NLUD×tree-canopy Carbon biophysical table (1,984 rows × 27 cols, keyed on compound `lucode` 0–1983). Provides per-pixel four-pool carbon storage — `c_above`, `c_below`, `c_soil`, `c_dead` (tons C/ha) — indexed directly by the compound LULC raster via per-city `c_above_arr`/`c_below_arr`/`c_soil_arr`/`c_dead_arr` numpy arrays. Three additional columns (`c_embedded_storage`, `c_embedded_emissions`, `c_annual_emissions`) describe urban-accounting flows the prototype doesn't currently use. Referenced via SA's `carbon_table_file` config. | done (Brief 30) |
| `data/sa/natcap_2024/acs_block_groups_3857.gpkg` | NatCap ACS block-group polygons covering the City of San Antonio (1,124 polygons, EPSG:3857). SA's `tracts_file` config pointer. Consumed only by `compute_per_tract_summary`'s Neighborhood breakdown table — no biophysical metric depends on the file. Reprojected to EPSG:5070 at load time. Per-city dashboard caption is conditional ("Census tracts" for MN, "Census block groups" for SA). | done (Brief 31) |
| `data/sa/flood/soil_group_sa.tif` | SSURGO hydrologic soil group rasterized to LULC grid | done (TX029, 49 % D-class clay-rich Vertisols) |
| `data/sa/cooling/et_annual_sa.tif` | Reference ET raster (CGIAR Global-AI/ET0 v3.1, 1,580–1,716 mm/yr) | done |
| `data/sa/population/sa_pop_2020.tif` | Census 2020 block totals rasterized to LULC grid | done (1.91 M in raster) |
| `data/sa/roads_sa.geojson` | OSM roads (Geofabrik TX, Option B filter) | done (55,553 segments) |
| `data/sa/buildings_sa.gpkg` | OSM buildings (Geofabrik TX, GeoPackage; raw GeoJSON 185 MB exceeded GitHub limit) | done (345,900 polygons) |
| `data/sa/tracts_bexar.shp` | TIGER 2020 Bexar County tracts | done (375 tracts) |
| `data/sa/flood/Damage_loss_table_SA.csv`, crop-yield table | SA-specific damage rates and crop yields | pending — Option A semantics in the meantime ($0 dollar metrics) |
| `data/sa/precomputed/nature_distance_<lucode>.npy` | Float32 distance-to-class fields for the static nature lucodes (11, 42, 43, 52, 71, 81, 95) at the SA grid (1713 × 1984). 7 × 13 MB ≈ 91 MB. Loaded at module load by the per-city cache layer; recomputed + re-cached on shape/dtype mismatch. | done |
| `data/precomputed/minneapolis_mn/nature_distance_<lucode>.npy` | Same, MN downtown grid (356 × 360). 7 × 501 KB ≈ 3.4 MB. | done |

Pipeline scripts: `download_sa_data.py` (NLCD), `download_ssurgo_sa.py` +
`process_ssurgo_sa.py` (soil), `download_census_pop_sa.py` (population),
`download_et_sa.py` (CGIAR ET0), `download_osm_sa.py` (roads + buildings),
`process_tracts_sa.py` (tracts).
Detailed sourcing notes in `data/sa/README.md`.

OSM buildings carry `type` as OSM strings ('house', 'apartments', 'retail', …)
not the integer 0–3 codes InVEST expects. SA now maps those strings to InVEST
type codes 1/2/3 via `_OSM_BUILDING_TO_INVEST_TYPE` in app.py (≈29 % pixel
coverage — untyped polygons such as `building=yes`, NaN, `roof`, and
`storage_tank` are left at 0 and excluded from per-type lookups). This
**lights up the Cooling Energy Savings card** for SA as a conservative lower
bound. The Cooling Energy Savings tooltip surfaces the coverage caveat
whenever `BUILDINGS_TYPE_COVERAGE < 0.95`. **Flood Damage Avoided for SA
now renders as "Flood Retention" (% volume reduction)** rather than a
dollar figure — `damage_table_file` is `None` because NatCap's Vibrant
Land report (Guerry et al. 2023) used InVEST UFRM for SA but explicitly
did not enable damage valuation. The prototype matches this methodology
(Brief 33, Path C). MN keeps the dollar-based "Flood Damage Avoided"
metric (InVEST UFRM sample data includes the damage table).

**Canonical CRS for San Antonio: EPSG:5070** (NAD83 / Conus Albers, NLCD's
native equal-area CRS). Differs from Minneapolis (EPSG:26915 / UTM 15N) —
equal-area is preferred for SA's larger area-based analyses.

**Biophysical-table naming convention:** every city has its own CN and
cooling tables, suffixed with the city's two-letter code (`_MN`, `_SA`).
Each `CITIES` entry declares its filenames via `cn_table_file` and
`cooling_table_file`; `load_data` joins these against the city's
`data_dir_flood` / `data_dir_cooling`. SA's UCM / UNA / Carbon tables
are now NatCap compound-keyed (Briefs 28b / 29 / 30); MN tables remain
NLCD-keyed.

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
| `UHI_MAX_C` | per-city | Read from `city_cfg['uhi_max_c']` at module load (i.e. on every script rerun) — NOT a fixed global. MN downtown: 2.05 °C (InVEST `urban_cooling_model_args_MN.json`); MN Full: 2.05 °C (same AOI climate); SA: 11 °C (NatCap SA README, Brief 14). Used in `compute_cooling_energy_savings` for CC → ΔT °C conversion. Consumers: app.py:1422 (cooling) and app.py:3307 (assumptions tab display). |
| `DESIGN_STORM_INCHES` | per-city | Read from `city_cfg['design_storm_inches']` at module load — NOT a fixed global (post-Brief 23). MN downtown: 3.94" (100 mm per NatCap MN `invest_urban_flood_risk_args_MN.json`); MN Full: 3.94" (same MN-project framing); SA: 6.18" (157 mm per NatCap SA README). Derived `DESIGN_STORM_MM = DESIGN_STORM_INCHES × 25.4` for tooltip display. The SCS-CN formula uses inches internally. |
| `HM_TO_FAHRENHEIT` | per-city | Derived as `UHI_MAX_C × 1.8`. MN: 3.69 °F/CC; SA: 6.30 °F/CC. Rebound every rerun alongside `UHI_MAX_C`. |
| `GREEN_AREA_COOLING_DISTANCE_M` | 450 | InVEST UCM `green_area_cooling_distance` (d_cool), from InVEST args JSON. Drives the exponential-decay kernel for `CC_park` in the canonical HMI. `_HMI_DECAY_PX = 450/30 = 15` at 30 m NLCD resolution. |
| `COST_PER_KWH_USD` | 0.13 | US average residential electricity price (EIA 2024). Used to convert avoided-AC-kWh into $. |
| `EPA_SOCIAL_COST_CARBON` | 190 | $/ton CO2e — EPA 2023 final rule, 2 % discount rate, 2030 emissions. Multiplied by `carbon_tons_co2` → `carbon_value_usd`. Per-city semantics (annual flow MN vs one-time stock SA) and the methodology-matches-but-SC-CO2-vintage-differs alignment with NatCap's Vibrant Land report live in `DESIGN_NOTES.md` "SA Carbon four-pool framework adoption" and `NATCAP_COLLABORATION.md` Brief 30 decision row. |
| `PIXEL_AREA_M2` | 900 | NLCD 30 × 30 m pixel area in m². Used for cooling energy savings (consumption rate is kWh/(m²·°C)/yr from `energy_consumption.csv`). |
| `NATURE_RADIUS_CAP_M` | 1000 | Upper cap applied to every `search_radius_m` in the InVEST UNA table. Without this cap, water/forest classes (5 km radius) saturate the AOI to 100 % nature access. Caps at ~12-min walking distance, matches the table's own value for "Developed, Open Space" (urban parks). |
| `RR_0_1_NDVI_DEPRESSION` | 0.96 | InVEST UMH relative risk per 0.1 NDVI increase, depression. Source: Liu et al. 2023 meta-analysis. |
| `RR_0_1_NDVI_ANXIETY` | 0.97 | Same, anxiety. |
| `BIR_DEPRESSION` | 0.21 | Baseline depression prevalence (CDC 2023, ever-diagnosed). |
| `BIR_ANXIETY` | 0.19 | Baseline anxiety prevalence. |
| `COST_PER_DEPRESSION_CASE_USD` | 8467 | Annual cost-of-illness per case (US nominal). InVEST docs cite ~$11K USD-PPP — our default is slightly lower. |
| `COST_PER_ANXIETY_CASE_USD` | 5765 | Same, anxiety. |
| `UMH_SEARCH_RADIUS_M` | 300 | InVEST UMH NDVI exposure radius (Li et al. 2025). Pre-computed `_UMH_SIGMA_PX = 10` (= 300 m / 30 m px). NE raster is Gaussian-smoothed with `scipy.ndimage.gaussian_filter`, matching InVEST canonical behavior. |
| `LBS_PER_PERSON_YEAR` | 2,000 | Average American food consumption used to convert lbs → people fed |
| `DEVELOPED_CODES` | [21, 22, 23] | NLCD lucodes treated as convertible developed land |
| `CODE_GREEN_INFRA` | 90 | NLCD lucode for woody wetlands (green infrastructure proxy) |
| `CODE_FOOD_FOREST` | 41 | NLCD lucode for deciduous forest (food forest proxy) |
| `CODE_HIGH_DENSITY` | 24 | NLCD lucode for high-intensity development |
| `NODATA` | -128 | Sentinel value in rasters marking outside-boundary pixels |

### City-specific (set at runtime from `city_cfg`)

| Name | MN downtown | MN Full | San Antonio | Meaning |
|------|------------:|--------:|------------:|---------|
| `BASELINE_CN` | 75.67 | 77.68 | 76.54 | Mean curve number of unmodified land × soil grid |
| `BASELINE_HM` (= mean CC) | 0.1859 | 0.1600 | 0.3937 | Mean Cooling Capacity (`0.6·shade + 0.2·albedo + 0.2·ETI`) over the AOI. Cross-city interpretation caveat below; Brief 28b magnitude history in DESIGN_NOTES.md and HISTORY.md. |
| `BASELINE_NDVI` | 0.2326 | 0.2072 | 0.4242 | Mean synthetic NDVI proxy |
| Population | ~154 K | 463,794 | 1,906,323 | Census 2020 county-level totals in the bbox |

All three numeric baselines are dynamically recomputed at module load (the hardcoded values in `CITIES[city]['baseline_*']` are documentation only — the live overrides keep them in sync with the current pipeline).

> **Cross-city `BASELINE_HM` caveat:** SA's HM is *higher* than both Minneapolis values despite SA being the hotter city — driven by shade-side factors (forest/woody-wetland pixel fraction + tree-canopy variation on developed land) rather than absolute ET. When comparing scenario impact across cities, prefer **CC deltas** over absolute CC values. Full breakdown in REFERENCE.md "Cross-city Cooling Capacity comparison".
| `BASELINE_RUNOFF_ACRE_FEET` | computed | Runoff from baseline CN over the city's design storm (per-city `DESIGN_STORM_INCHES`); used for cost-effectiveness ratios |

### Cost defaults ($/acre, adjustable via sidebar sliders)

| Constant | Default | Represents |
|----------|---------|------------|
| `DEFAULT_COST_GI` | $50,000 | Green infrastructure / constructed wetlands |
| `DEFAULT_COST_FF` | $10,000 | Food forest establishment |
| `DEFAULT_COST_HD` | $5,000 | Marginal high-density infill |

---

## Module layout

The codebase has been incrementally split as it grew. Current state:

- **`app.py`** — Streamlit UI, runtime state, loaders, `evaluate_scenario` and its biophysical helpers, metric cards, map and tradeoff rendering. Still the bulk of the code.
- **`config.py`** — Per-city configuration (`CITIES` dict) and cost defaults. Read-only; mutations belong in `app.py`'s runtime state.
- **`surrogate.py`** — Random Forest surrogate model (training, prediction with uncertainty bands) and the Pareto optimizer. Streamlit-agnostic; the `@st.cache_resource` wrapper lives at the call site in `app.py`.
- **`verify_baselines.py`** — CLI baseline regression check. Snapshots `evaluate_scenario` outputs for each city × scenario × placement strategy to `tests/baselines/<city>__<scenario>__<strategy>.json`. Currently 4 scenarios × 5 strategies × 2 cities = 40 baselines; runtime ~90 seconds. Run before commit when changes could have cross-cutting effects; run with `--update` after intentional changes.

Further extractions (loaders, scenario.py, plots.py) remain deferred — they're more tightly coupled to Streamlit's runtime than the surrogate or config blocks were.

- **Methodology / InVEST parity** — documented in `REFERENCE.md` under "Official InVEST alignment." Includes a parity table for every metric and per-model gap notes (UFR, UCM, UNA, UMH, Carbon, Crop Production).

---

## Architecture notes

- **Lookup table + live refresh** (`compute_lookup_table`): pre-computes all valid slider
  positions at step=5, caching the core raster-derived metrics (CN, flood reduction, runoff,
  cooling HM, MH cases/costs). On each slider interaction the app also runs a full
  `evaluate_scenario()` and overwrites ~12 of 27 result fields with live values — specifically
  `scenario_lulc`, food, NDVI, carbon, avoided carbon cost, nature access/quality, flood
  damage avoided, cooling energy savings, and implementation cost. This hybrid pattern keeps
  the expensive raster aggregates cached while ensuring slider-sensitive parameters (carbon
  rates, cost sliders) and post-schema metrics always reflect the current state. When
  a non-random placement strategy is active, the lookup table is bypassed entirely. The scenario grid
  (step=10/25) is used only for surrogate training.
- **Surrogate model** (`surrogate.py`): Random Forest trained on the scenario grid; used by the
  optimizer to search ~10k random scenarios for Pareto-optimal suggestions. Uncertainty bands
  come from 10th/90th percentile across RF trees. Cache wrapper lives in `app.py`
  (`_cached_train_surrogate`); the underlying functions are Streamlit-agnostic and can be
  imported standalone for testing.
- **Placement strategies** (`evaluate_scenario`'s `placement_strategy` kwarg): five strategies
  for selecting which convertible pixels actually get converted. `random` (default) is uniform
  sampling; `flood-focused`, `cooling-focused`, `equity-focused`, and `balanced` weight sampling
  toward suitability components computed from existing module-level rasters (CN table, baseline
  CC, population, access score). Helper functions `_compute_suitability_weights` and
  `_select_pixels_for_conversion` are in `app.py` near `evaluate_scenario`. UI exposed as a
  sidebar radio picker. The legacy `use_heat_priority=True` kwarg remains in the function
  signature for backward compatibility and internally translates to
  `placement_strategy='cooling-focused'`. See REFERENCE.md "Placement strategies" for the
  suitability formulas and honest caveats, and `INVEST_PLACEMENT.md` for the underlying
  research (InVEST is placement-agnostic — the strategies are an app feature, not a parity
  requirement).
- **Equity weighting**: `equity_weights` raster weights high-intensity developed pixels (NLCD 23)
  higher; used as one component in the cooling-focused placement strategy. Currently a proxy;
  TODO is to replace with a real CDC/ATSDR Heat Vulnerability Index by census tract.
- **`REF_SCENARIOS`**: hardcoded Minneapolis benchmark points (all-one-landcover extremes) shown
  on the tradeoff plot. Will need to become city-specific when new cities are added.
- **InVEST Urban Cooling Model — canonical Heat Mitigation Index (HMI).**
  `_compute_hmi_raster` computes the canonical InVEST UCM HMI: per-pixel
  CC = `0.6·shade + 0.2·albedo + 0.2·ETI` (`_compute_cc_raw_pure`), then
  **HMI = `max(CC_local, CC_park)`** where `CC_park` (`_compute_cc_park_raster`)
  is the exponentially distance-weighted CC sourced from green areas, applied
  only where a pixel has ≥2 ha of green within `d_cool = 450 m`
  (`_compute_green_area_sum` checked against the 2-hectare threshold).
  Convolutions use `scipy.signal.fftconvolve` with an InVEST-canonical edge
  correction (`_convolve_edge_corrected`, reproducing
  `pygeoprocessing.convolve_2d(ignore_nodata_and_edges=True)`). The mean of the
  HMI raster is the `mean_hm` reported in scenario results (UI label: "Cooling
  Capacity" / "CC"). `compute_cooling_energy_savings(hmi_raster)` converts
  ΔHMI → ΔT °C (× `UHI_MAX_C`) → kWh saved (× `consumption_rate × pixel_area`)
  → $/yr (× `COST_PER_KWH_USD`); still per-pixel, not per-building polygon —
  the one remaining UCM divergence. The HMI algorithm is validated against
  `natcap.invest.urban_cooling_model.execute()` at MAE = 0.0000, r = 1.0000
  (`compare_ucm_invest.py`); see `UCM_AUDIT.md` for the implementation-status
  writeup and REFERENCE.md "Official InVEST alignment — UCM" for the per-metric
  parity table. Both functions are called inside `evaluate_scenario`; the loader
  builds the baseline via the pure variant `_compute_hmi_raster_pure`.
  Module-level precompute: `ET_RESIZED`, `MAX_ET_REF`, `BUILDINGS_TYPE_RASTER`,
  `CONSUMPTION_RATE_PER_PIXEL`, `_BASELINE_HM_RASTER` (the baseline HMI raster),
  and the static convolution kernels `_HMI_EXP_KERNEL` / `_HMI_DICH_KERNEL`.
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
- **`SCENARIO_SCHEMA_VERSION = 25`** — bump on every change that shifts `evaluate_scenario`
  outputs so cached lookup tables get regenerated. Full per-bump history in `HISTORY.md`
  "Schema version log"; per-brief reasoning in `DESIGN_NOTES.md`.
- **City runtime state (`CityState` + `_load_city_runtime_state`).** All heavy
  per-city allocations — rasters from `load_data`, the population raster, the
  resized ET raster, building/road/tract rasterisations, the static nature-
  distance fields, the baseline CC / NE / access-score rasters, plus the
  derived `baseline_hm` / `baseline_cn` scalars — live in an immutable
  `CityState` NamedTuple built by `@st.cache_resource def
  _load_city_runtime_state(city_key: str) -> CityState`. Cached on `city_key`
  so the heavy work runs at most once per (city, session); Streamlit reruns
  fetch the same NamedTuple from cache instantly instead of re-allocating.
  This is the single architectural fix that made the app fit Streamlit
  Cloud's 1 GB ceiling. After the loader returns, module-level globals
  (`cooling_lulc`, `ET_RESIZED`, `BUILDINGS_RASTER`, ...) are aliased to the
  matching state members via pointer rebinding — downstream function bodies
  read them as bare names without parameter threading. **The two baseline
  scalars `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` are intentionally
  NOT aliased to module-level** — every downstream call site reads them via
  the state handle. This prevents silent staleness if a future code path
  fails to refresh a global on city switch (arrays would crash on shape
  mismatch; scalars would just produce wrong-but-plausible numbers). The
  cached `compute_scenario_grid` and `compute_lookup_table` take a leading-
  underscore `_state` (skip-hashed) plus an explicit `city_key` (hashed
  cache discriminator).
- **`plot_spatial_map` memory pattern.** Matplotlib `imshow` layers in this
  function used to peak at ~378 MB transient per slider rerun on SA's
  1713×1984 AOI — the single biggest avoidable allocation after the
  cache_resource refactor. Two mitigations:
  (1) **uint8 RGB / RGBA layers.** The base `rgb` array and the heat-overlay
  `overlay_rgba` array are `np.uint8` instead of `float64` (matplotlib
  imshow accepts uint8 directly). The tract-overlay cmap output stays
  `float32` because fractional alpha matters there.
  (2) **Aspect-preserving downsample to `_PLOT_MAX_DIM = 1024`** via
  `scipy.ndimage.zoom`, applied once at the top of the function before any
  layer is built. `order=0` (nearest-neighbor) for the integer LULC raster
  (category integrity must be preserved); `order=1` (bilinear) for
  continuous overlays. `tract_value` is `nan_to_num`'d before bilinear
  zoom with a separate nearest-neighbor pass on the validity mask to gate
  the cmap alpha. Streamlit displays the figure at ~600 px wide regardless,
  so rendering the full AOI is wasted memory.
- **Precomputed static rasters.** Module-level allocations that are static for the
  lifetime of the deploy can be persisted to `<city_cfg['precomputed_dir']>/<artifact>.npy`
  and reloaded inside `_load_city_runtime_state` instead of recomputed on first
  cache miss. Currently only `PRECOMPUTED_NATURE_DISTANCES` uses this pattern:
  one `float32` `.npy` per static nature lucode (11, 42, 43, 52, 71, 81, 95)
  under `nature_distance_<lucode>.npy`. The loader validates
  `arr.shape == cooling_lulc.shape and arr.dtype == np.float32` before
  trusting the cache; on mismatch it falls back to live compute and re-saves.
  Live compute is preserved as a fallback so cities mid-onboarding (no
  checked-in artifacts yet) still work. Per-city cache locations:
  `data/precomputed/minneapolis_mn`, `data/precomputed/minneapolis_full_mn`,
  `data/sa/precomputed`. To regenerate for a city, delete the directory and
  re-run the app (or `precompute_scenarios.py`) for that city.
- **Dynamic baselines.** `baseline_hm` and `baseline_cn` are computed inside
  `_load_city_runtime_state` from the unmodified LULC raster, using the same
  lookups `evaluate_scenario` uses. The
  `CITIES['<city>']['baseline_hm' / 'baseline_cn']` values are
  documentation-only. Live values are read everywhere as
  `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` (see CityState entry
  above for why these two scalars get the explicit-state-handle treatment).

---

## Blocked / pending work

- **SA is the default test bed for memory-sensitive changes.** SA's
  1713 × 1984 grid is the prototype's largest AOI; if SA fits inside
  Streamlit Cloud's 1 GB worker ceiling, the smaller MN / Mpls-Full
  grids fit by definition. The Streamlit Cloud memory-fit workstream
  (resolved 2026-05-11) landed five mitigations whose combination
  brought peak memory under the ceiling — see `HISTORY.md`
  "Streamlit Cloud memory-fit workstream" for the full stack.
- **Stratified Impervious Siting (placement-step control).** Currently the stochastic placement step samples uniformly from the building/road-filtered NLCD 21–24 pool, treating all impervious-intensity classes as equivalent for siting. Proposal: expose impervious-intensity stratification to users via sidebar control, allowing them to direct placement toward NLCD 21 (≥20% impervious, open-space dominant), NLCD 22/23 (low-medium intensity), or NLCD 24 (≥80% impervious, high-intensity mitigation / depaving). Use `_distance_transform_edt` against `BUILDINGS_RASTER` for optional micro-siting refinement (e.g. "open lot" vs "private yard" via 15m/30m distance thresholds). Frame strictly as impervious-intensity stratification, not policy/ownership tiering — NLCD classes correlate with but do not equal ownership. Open questions for scoping session: (a) mutually-exclusive radio buttons vs multi-select vs per-tier weight sliders; (b) whether to dynamically clamp slider max based on selected tier's available acreage; (c) whether stratified placement empirically resolves the Nature Access saturation issue noted in REFERENCE.md (validate before claiming). Source: Gemini-3 proposal, iterated through 3 versions on Claude critique; v3 is the version to scope from.
- **SA flood damage — resolved (Brief 33, Path C).** Dashboard renders
  "Flood Retention" (% volume reduction) for SA instead of monetized
  damage, matching NatCap's Vibrant Land (Guerry et al. 2023) reporting.
  Underlying `avoided_flood_damage_usd` field still returns $0 for SA
  (no schema change). See `DESIGN_NOTES.md` "SA flood damage table —
  resolved (Path C, Brief 33)". Reversible if NatCap surfaces a
  preference for SA-specific damage values (Path A).
- **SA flood biophysical — integrated 2026-05-29.** The SA flood model
  uses NatCap's NLCD × tree-canopy 3-tier CN table
  (`biophys_floodmitig_sa.csv`). The previous MN-placeholder table is no
  longer wired but kept on disk for reference (still read by
  `download_sa_data.py:121` as a one-time QA diagnostic). Methodology
  documented in `NATCAP_COLLABORATION.md` question 12: NatCap's CN
  values reflect a design-storm-saturation framework where soil
  infiltration capacity is exceeded under the 24-hour 100-year storm;
  under this framework, GI scenarios produce minimal flood-volume
  effect for SA. NatCap's own modeling confirms this ("essentially no
  difference between garden, food forest, park, or vacant vegetated
  space" — Ben NDR and Flood Mar_2023.pptx, slide 7).
- **Heat Vulnerability Index — still pending.** The `equity_weights`
  raster is a proxy (NLCD intensity-coded), not a real CDC/ATSDR HVI by
  census tract. Replacing it is the next data-quality upgrade.
- **Minneapolis Full extent (hidden from UI).** Live in `CITIES` but
  `available=False`. See `HISTORY.md` "Full Minneapolis extent" for the
  activation+hiding rationale and pipeline path details.
- **`load_data` is parameterized via `city_cfg`** — `lulc_file`, `soil_file`,
  `cooling_lulc_file`, plus the module-level loaders for ET / energy / UNA /
  buildings / roads / tracts all read from `city_cfg`. Biophysical tables (CN +
  cooling) use a fallback path via `_resolve_table()` so cities with custom
  `data_dir`s (Mpls Full → `data/minneapolis_expanded/`) can still reference
  the project-shared tables. CRS reads via `city_cfg['crs']`. See `HISTORY.md`
  "`load_data` parameterization (2026-05-09)" for the transition record.

---

## Coding conventions

- **Float32 for module-level geospatial arrays.** Any full-AOI raster computed
  or loaded at module load (population, ET, consumption-rate, baseline rasters,
  precomputed distance fields, etc.) must be `np.float32`, not the numpy default
  `float64`. SA's 1713 × 1984 grid is 27 MB per float64 raster vs 13.6 MB per
  float32 — at 8+ such arrays this is the difference between fitting in
  Streamlit Cloud's 1 GB worker and OOM-killing on startup. Float64 is reserved
  for: accumulators inside `evaluate_scenario`, anything summing across millions
  of pixels, or anywhere precision loss could shift a metric output. When in
  doubt, downcast — float32 carries 24-bit mantissa precision (~7 decimal digits)
  which is well beyond the precision of any geospatial input we ingest.
- **No bare globals for city data** — always pull city-specific values from
  `city_cfg`, the derived runtime aliases (`cooling_lulc`, `ET_RESIZED`,
  `BUILDINGS_RASTER`, ...), or the explicit-state handle
  `_CURRENT_CITY_STATE` (for `baseline_hm` / `baseline_cn`, which are NOT
  aliased to module-level by design — see Architecture notes). Don't
  hardcode Minneapolis values outside of the `CITIES` dict.
- **Pure-variant helpers for code the loader calls.** Heavy compute helpers
  that the loader invokes (currently `_compute_hmi_raster`) come in two
  variants: `_fn(scenario_lulc)` reads module aliases populated by the loader, and
  `_fn_pure(scenario_lulc, *deps)` takes its dependencies explicitly. The
  loader uses the pure variant because the module aliases haven't been
  rebound yet at the moment the loader runs; downstream code uses the
  zero-arg wrapper.
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
  step_alloc=25)` (~90), Balanced prefers the active city's `dense_scenarios_file`
  (e.g. `data/scenarios_dense_mpls.csv` / `_sa.csv`) else
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
- **Unavailable cities are filtered out of the sidebar selector** at the
  selectbox-options stage, so the "coming soon" UI branch is no longer
  reachable. If a future city needs an explicit "data being prepared"
  state, add it back as a filter on `available` plus a sidebar caption,
  not as an `st.stop()` after data loading.
- **Scenario LULC is not stored in the lookup table** — `scenario_lulc` is stripped from cached
  results (`if k != 'scenario_lulc'`) to keep memory usage manageable. It is one of the ~12
  fields recomputed by the live `evaluate_scenario()` call on every slider interaction (see
  "Lookup table + live refresh" above).
- **Cooling Energy Savings — dual display (city total + per-typed-building rate).**
  The card shows the city-wide dollar total as the headline metric AND a
  small caption beneath it formatted as `~$N/yr per typed building`. The
  per-pixel rate is `cooling_energy_savings_usd /
  np.sum(BUILDINGS_TYPE_RASTER > 0)` and is the **city-agnostic
  comparable** number — the total is footprint-scope-dependent (MN's
  building dataset is a downtown InVEST sample, SA's is county-wide OSM,
  so the totals are not directly comparable). The caption is suppressed
  when `cooling_energy_savings_usd < $1,000` (e.g. HD-only scenarios)
  where the rate would be uninformative. Formatting tiers: `>$1000`
  rounded to nearest $10, `>$100` to nearest $1, else to two decimals.
  This dual-display pattern lives only on Cooling Energy Savings for now
  — don't extend to other metric cards without an actual user signal that
  the comparability gap matters elsewhere.
- **Metric cards are grouped into four labeled sections** — 🌿 Ecological (5 cards in two
  rows: row 1 has Flood Risk Reduction, Temperature Change, and Runoff Volume in 3 columns;
  row 2 has Carbon Sequestration and NDVI in 2 columns),
  👥 Human & Social (3 cards: Nature Access, Preventable MH Cases, Avoided MH Costs — the
  InVEST Urban Mental Health v3.19.0 outputs replaced the earlier weighted-composite
  Wellbeing Score; Nature Access was reimplemented as canonical InVEST UNA 2SFCA and
  restored 2026-05-22, while the Nature Quality Score card stays removed — see WHATS_NEW),
  💵 Economic (5 cards in two rows: row 1 has Food Production + Est. Implementation Cost
  in 2 columns; row 2 has Flood Damage Avoided + Cooling Energy Savings + Avoided Carbon
  Cost in 3 columns — the EPA Social Cost of Carbon dollar metric is `carbon_tons_co2 ×
  EPA_SOCIAL_COST_CARBON` via the `carbon_value_usd` field, deterministic so not in the
  surrogate; SA card reads "Carbon Storage Value" one-time vs MN's "Avoided Carbon Cost"/yr
  per the per-city framing — see Brief 30),
  📊 Cost Effectiveness (3 sub-ratios under their own header). Each group is separated by
  `st.divider()`. **13 metric cards total**. Keep this grouping when adding new metrics — place
  new cards in the section that matches their category rather than appending to a flat list.
- **NDVI is a synthetic proxy** — values come from a per-NLCD-code lookup
  (`NDVI_PROXY` plus `NDVI_OTHER_DEVELOPED` / `NDVI_OTHER_NATURAL` defaults), not from
  satellite imagery. `BASELINE_NDVI` is computed once at startup from the unmodified
  `cooling_lulc` raster; scenario `mean_ndvi` is computed inside `evaluate_scenario` and
  flows through the lookup table and any cached scenario results.
- **Carbon — per-city methodology.** SA uses `_compute_carbon_four_pool(scenario_lulc_carbon,
  baseline_lulc_carbon)` (one-time t CO2 stock change). MN uses
  `_compute_carbon(n_wet, n_for, n_hd)` (annual t CO2e/yr). The unified return-dict
  key `carbon_tons_co2` carries either framing; the city-conditional
  `_CARBON_IS_STOCK` flag (set once after module alias rebinding) drives
  dashboard card labels, optimizer slider unit suffixes, and comparison-table
  formatting. `EPA_SOCIAL_COST_CARBON × carbon_tons_co2 = carbon_value_usd`.
  Methodology rationale + magnitude evidence in DESIGN_NOTES.md "SA Carbon
  four-pool framework adoption".
- **Carbon sequestration (MN only) counts converted pixels** — `CARBON_SEQ_RATES` maps the three
  conversion target codes (`CODE_FOOD_FOREST`, `CODE_GREEN_INFRA`, `CODE_HIGH_DENSITY`) to
  provisional regional USDA/IPCC rates in tons CO2e/acre/yr (3.5, 2.0, 0.0). Inside
  `evaluate_scenario`, `carbon_tons_co2` is computed inline from `n_for`, `n_wet`, `n_hd`
  and pixel area — there is no per-cell raster pass and no startup baseline (baseline = 0,
  same convention as `food_mln_lbs`). The value flows through the lookup table and cached
  scenario results. Treat as directional only — not locally calibrated.
- **Carbon rates are user-overridable (MN only)** — the sidebar `⚙️ Advanced Settings` expander
  exposes `carbon_rate_ff` and `carbon_rate_gi` sliders backed by `st.session_state`. Both
  main-panel `evaluate_scenario` calls (the lookup-refresh `_fresh` and the non-random-strategy
  branch) pass these values through; `evaluate_scenario` falls back to `CARBON_SEQ_RATES`
  defaults when the kwargs are `None`. The precomputed lookup table is built with defaults,
  but `carbon_tons_co2` is recomputed live in the lookup-refresh path so slider changes
  always take effect. (SA sliders have no effect — the four-pool table is the data, not a
  user input.)
- **InVEST Urban Mental Health Model (v3.19.0)** — `calculate_mental_health_impact(scenario_lulc,
  baseline_ne_raster, pop_count)` returns `(preventable_mh_cases, avoided_mh_cost_usd)`. Per-pixel
  formula: NE = `gaussian_filter(NDVI_proxy, sigma=10 px)` (10 px = 300 m at 30 m NLCD per
  Li et al. 2025); ΔNE = NE_scenario − NE_baseline; RR = `exp(ln(RR₀.₁) × 10 × ΔNE)`;
  PC = `(1 − RR) × baseline_prevalence × population`. Sums depression + anxiety. Constants:
  `RR_0_1_NDVI_DEPRESSION=0.96`, `RR_0_1_NDVI_ANXIETY=0.97` (Liu et al. 2023 meta-analysis);
  `BIR_DEPRESSION=0.21`, `BIR_ANXIETY=0.19` (CDC 2023 ever-diagnosed); per-case cost-of-illness
  $8,467 / $5,765 (US nominal). Returns (0, 0) at the unmodified baseline by construction.
  `_BASELINE_NE_RASTER` is precomputed once at module load. (The previous
  `compute_wellbeing_score` composite metric + weight sliders were removed
  entirely when UMH landed; see `HISTORY.md` "Wellbeing Score" for context.)
  See REFERENCE.md "Official InVEST alignment — UMH" for parity status and
  divergences (uniform BIR vs per-admin, Gaussian kernel vs uniform buffer,
  synthetic vs satellite NDVI).
- **The surrogate predicts six outputs** — `train_surrogate` fits the Random Forest on
  `[flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet, carbon_tons_co2,
  nature_access_pct]`, so `predict_with_uncertainty` returns `(n, 6)` arrays. The two
  InVEST UMH metrics (`preventable_mh_cases`, `avoided_mh_cost_usd`) are **not** RF
  targets — they are computed deterministically inside `evaluate_scenario` from the
  scenario's NDVI exposure, so the surrogate doesn't need to predict them.
  (`evaluate_scenario` returns many more fields than these six — intermediate
  metrics, pixel counts, the scenario name — the six above are just the columns the
  surrogate learns. `REQUIRED_TARGET_COLUMNS` lists eight must-exist grid columns,
  a superset of the six RF targets plus the two deterministic MH metrics.)
  `optimize_scenario` adds a `min_carbon` constraint (`mean_preds[:, 4] >= min_carbon`)
  alongside the existing flood, cooling, food, and runoff filters. The carbon column flows
  into the candidate DataFrame (with `carbon_lower` / `carbon_upper` uncertainty bands) and
  the optimizer results display, which renames it to "Carbon (tons CO2e/yr)" for MN
  or "Carbon (tons CO2e stock)" for SA (city-conditional via `_CARBON_IS_STOCK`). Note: the
  carbon surrogate is trained at the **default** rates baked into `scenario_df`; user
  overrides via Advanced Settings do not retrain the surrogate, so optimizer carbon
  predictions reflect default rates even when sliders are adjusted. Nature access is the
  6th output and carries an explicit caveat: the surrogate cannot see the spatial geometry
  that drives the metric (placement of converted pixels relative to existing parks and
  population centers), so its predictions are an indicative trend, not a precise spatial
  estimate.
- **Nature Access is canonical InVEST UNA (2SFCA), re-implemented in numpy** —
  `calculate_nature_access(scenario_lulc, pop_count_raster)` runs a two-step
  floating catchment area calculation (`_una_supply_percapita` /
  `_invest_una_pct_pop_supply_ge_demand`): per-pixel urban-nature area =
  `URBAN_NATURE_PROPORTION[lucode] × pixel_area`; the population raster and the
  R_j nature/population ratio are each convolved with a dichotomy disk kernel
  (`_UNA_KERNEL`, uniform 800 m radius); the headline `pct_pop_supply_ge_demand`
  is the population-weighted share of pixels where `urban_nature_supply_percapita
  ≥ UNA_DEMAND_M2_PER_CAPITA` (16.7 m²/capita). Returns
  `(access_pct, 0.0, people_with_access)` — the middle slot is a retained legacy
  tuple position (was the removed Nature Quality Score). Validated against
  `natcap.invest.urban_nature_access.execute()` at MAE ≈ 0 (see REFERENCE.md
  "Official InVEST alignment — UNA"). Population data comes from
  `data/population/minneapolis_pop_2020.tif`, built by `download_census_pop.py`
  from US Census 2020 block-level totals (P1_001N for Hennepin County, FIPS 27053)
  joined to TIGER 2020 tabulation-block polygons and rasterized to the NLCD grid
  (each block's pop spread uniformly across its pixels). At startup `app.py` calls
  `load_population_data(...)` inside a `try/except (FileNotFoundError,
  RasterioIOError)`; on failure it falls back to a uniform `np.ones(...)` raster with
  `POPULATION_DATA_AVAILABLE = False` so the app still launches. **Extent caveat:**
  the NLCD template covers only ~10.8 km × 10.7 km (~154k residents) — a
  downtown-and-near-neighborhoods cutout, not all of Minneapolis; the headline
  reports the modelable-extent population (~43%), the rest sitting on cooling-LULC
  nodata pixels the model cannot evaluate. The search kernel is Euclidean: no
  street network, no barriers, no slope.
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
- **Stop-and-report sentinels are signal, not noise.** Briefs often include sentinel checks
  in their investigate-first section — file presence, content-string matches, line counts,
  "should-not-already-exist" assertions. When a sentinel fires, the precondition the brief
  required isn't actually met. The right response is to stop, report what was expected
  vs. what was found, surface options to the user, and wait for direction. Do NOT override
  the trigger by skipping the check or reframing the content as "close enough." Examples
  from real session experience: a v2.5 doc wasn't placed at the expected path (browser
  cache served v1), and a sentinel "should contain 'Per-city parameter framing'" caught it
  — without the sentinel, CC would have committed wrong content. A "DESIGN_NOTES.md
  should not already have a `# Design Notes` heading" sentinel caught an existing heading
  and prevented duplicate-heading commit. Sentinels are the parts of the brief where CC
  judgment is *constrained*, not just guided.
- **Planning artifacts can run in parallel with the investigations that inform them.**
  Drafting the next brief (or planning doc) while CC executes the current one
  compresses session wall-clock without compressing thought. The default to wait
  for "full information" before drafting planning artifacts is usually too
  conservative — most planning content is methodology-driven and data-independent.
  Use `[CC: detail pending from Brief N]` placeholders for the specific findings
  that haven't landed yet; fill them in once the predecessor brief reports.
  Example from real session experience: Brief 26 (SA integration planning doc)
  was held until Brief 24 (structural audit) landed, even though ~70% of its
  content — CRS choice rationale, integration sequence, planning structure —
  was data-independent. Only the specific tie-breaker logic in Decision 2 and
  the extent-loss tradeoff in Decision 1 needed Brief 24's findings. The lesson:
  default to drafting in parallel; mark pending findings with placeholders;
  fill them in when the data arrives.
- **WHATS_NEW + Underway discipline.** WHATS_NEW entries each clear a strict bar:
  the change *happened* (not queued/upcoming), is from the past ~7 days, would be
  noticed by a returning user, and reads as one line without internal vocabulary
  or specific parameter values. Doc-only changes, methodology refactors, and
  internal infrastructure don't qualify. The Underway section is for forward-
  looking work a user will *recognize when they see it* (a new model, a new
  city, a UI feature) — internal methodology refinements like biophysical-table
  swaps don't qualify there either. Empty by default; renders only when there's
  something genuinely user-anticipatable in flight. Same bar for On the radar
  entries — name a specific thing the user will see change, not an abstract
  direction. The trim bar applies on every brief that touches WHATS_NEW — when
  in doubt, cut rather than keep.
- **Interface changes require auditing all consumers.** When a brief changes the
  shape of a shared interface — adding a field to a return dict, changing a
  function signature, adding a config key — the brief's scope boundary must
  enumerate every consumer of that interface. The default consumer list is broader
  than it feels: not just direct callers in the same file, but also scripts that
  import the function (`precompute_scenarios.py`, `verify_baselines.py`, any
  standalone utility), tests that exercise the interface, and serialized formats
  that capture the interface's output (CSVs, JSON baselines). Brief 28b learned
  this the expensive way: adding `scenario_lulc_ucm` to `evaluate_scenario`'s
  return dict required stripping the field in three places — two were caught in
  the brief, one (`precompute_scenarios.py`) was missed and surfaced as a 15-min
  regeneration's worth of garbage CSV data. The lesson: when changing an
  interface, grep for every caller before writing the scope boundary.
  - **Per-city serialized artifacts are independent consumers — list them
    one-by-one.** Brief 30's `carbon_tons_co2_yr` → `carbon_tons_co2` rename
    regenerated `data/scenarios_dense_sa.csv` (commit `3648ae9`) but not
    `data/scenarios_dense_mpls.csv`. The mismatch went undetected for weeks
    because Fast prototype mode (the default) uses live `compute_scenario_grid`
    and never reads the CSV — only Balanced and High Resolution modes do.
    Surfaced today (2026-05-26) by Brief A.4 verify driving "High resolution"
    on MN: Brief C.2's gate downgraded to Balanced, Balanced read the stale
    MN CSV, the surrogate trainer crashed with `KeyError:
    "['carbon_tons_co2'] not in index"`. Fixed by `b45a423`. The rule that
    falls out: when a schema bump affects column names or required columns,
    every `data/scenarios_dense_<city>.csv` must be regenerated separately
    (one per city), and the regen step must be enumerated per-city in the
    brief's scope boundary — not folded into a single "regenerate the
    dense CSV" line that may only happen for whichever city the author is
    actively testing.
- **Verify referenced constants before relying on them for alignment arguments.**
  When a brief references specific numerical constants from external sources
  (NatCap publications, EPA documents, NLCD specs, InVEST sample data) to
  justify methodology choices or set order-of-magnitude reference points,
  verify the prototype's *current* value of that constant before treating it
  as a shared assumption. The same standard can have multiple vintages
  (e.g., IWG 2021 SC-CO2 at $53/t @ 3 % vs. EPA 2023 at $190/t @ 2 %); assuming
  alignment without verification produces silent drift between brief intent
  and code reality. Brief 30 v2 learned this when its citywide-carbon sanity
  check (340,000 t × $53 ≈ $17.6M) didn't apply at the prototype's actual
  `EPA_SOCIAL_COST_CARBON = 190`. CC caught it in investigate-first; the
  resolution kept the more current EPA 2023 value and reframed the alignment
  as methodology-only (see next bullet).
- **"Methodology matches; constant differs" is a legitimate alignment pattern.**
  When aligning with a NatCap publication that uses older parameter values
  from a standard reference document, the prototype can align on *methodology*
  while keeping the *more current vintage* of the underlying parameter. The
  alignment narrative is "we use the same approach NatCap published in
  <report>, with parameter values from the most current authoritative source."
  Brief 30 used this pattern: SA Carbon now uses the Vibrant Land four-pool
  stock methodology with the prototype's EPA 2023 SC-CO2 (rather than Vibrant
  Land's IWG 2021 value). This is not a compromise — it's a documented
  decision. Without this pattern, a future session might either blindly
  downgrade a constant to match an older publication, or treat the constant
  divergence as methodology misalignment when it isn't. Document both the
  methodology alignment and the constant divergence in `DESIGN_NOTES.md` and
  `NATCAP_COLLABORATION.md` when this pattern applies.
