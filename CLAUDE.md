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
| `data/sa/flood/land_use_2021_sa.tif` | Canonical SA LULC raster (same CRS/grid) | done |
| `data/sa/flood/UFR_biophysical_table_SA.csv` | CN values by lucode × soil group | placeholder copy of MN |
| `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` | shade / Kc / albedo per lucode | placeholder copy of MN — SA's hotter/drier climate likely needs lower Kc and shade values for some classes (pending NatCap project tuning) |
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
`process_tracts_sa.py` (tracts), `verify_sa_baselines.py` (baseline check).
Detailed sourcing notes in `data/sa/README.md`.

OSM buildings carry `type` as OSM strings ('house', 'apartments', 'retail', …)
not the integer 0–3 codes InVEST expects. SA now maps those strings to InVEST
type codes 1/2/3 via `_OSM_BUILDING_TO_INVEST_TYPE` in app.py (≈29 % pixel
coverage — untyped polygons such as `building=yes`, NaN, `roof`, and
`storage_tank` are left at 0 and excluded from per-type lookups). This
**lights up the Cooling Energy Savings card** for SA as a conservative lower
bound. The Cooling Energy Savings tooltip surfaces the coverage caveat
whenever `BUILDINGS_TYPE_COVERAGE < 0.95`. **Flood Damage Avoided still
renders $0 for SA** because `damage_table_file` is `None` — reusing MN's
damage table would produce a wrong-but-numeric figure on a different
building stock (see "Pending — SA flood damage table sourcing"). Future
enhancement: refine `building=yes` cases via secondary OSM tags
(`shop=*`, `amenity=*`, `office=*`); see REFERENCE.md.

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
| `EPA_SOCIAL_COST_CARBON` | 190 | $/ton CO2e — EPA 2023 final rule "Methodology for Estimating the Social Cost of Greenhouse Gases", central estimate at 2 % discount rate for 2030 emissions. Multiplied by `carbon_tons_co2_yr` to produce the Avoided Carbon Cost dollar metric. |
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
| `BASELINE_HM` (= mean CC) | 0.1859 | 0.1600 | 0.2866 | Mean Cooling Capacity (`0.6·shade + 0.2·albedo + 0.2·ETI`) over the AOI |
| `BASELINE_NDVI` | 0.2326 | 0.2072 | 0.4242 | Mean synthetic NDVI proxy |
| Population | ~154 K | 463,794 | 1,906,323 | Census 2020 county-level totals in the bbox |

All three numeric baselines are dynamically recomputed at module load (the hardcoded values in `CITIES[city]['baseline_*']` are documentation only — the live overrides keep them in sync with the current pipeline).

> **Cross-city `BASELINE_HM` caveat:** SA's HM is *higher* than both Minneapolis values despite SA being the hotter city. This is **not** a result of higher absolute ET — the InVEST CC formula's ETI term normalises ET within each AOI (`Kc × ET / max(ET)`), so absolute mm/yr cancels out. The real driver is the shade term (weight 0.6, dominant): SA's bbox contains 14.9 % forest+woody-wetland pixels (NLCD 41/42/43/90, all with `shade=1`) versus 2.7 % in MN downtown and 1.8 % in MN Full. SA's mean shade across the AOI (0.198) is 3.4× MN downtown's (0.059). When comparing scenario impact across cities, prefer **CC deltas** over absolute CC values. See REFERENCE.md "Cross-city Cooling Capacity comparison" for the full breakdown.
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
  neighboring pixels per InVEST's `green_area_cooling_distance` step. The mean of this smoothed
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
- **`SCENARIO_SCHEMA_VERSION = 15`** — bump on every change that shifts `evaluate_scenario`
  outputs so cached lookup tables get regenerated. Recent bumps: 7→8 (UCM rework: ET fix,
  Gaussian convolution, canonical energy formula); 8→9 (ET nodata sentinel masked);
  9→10 (full Geofabrik OSM road network, 62 % AOI); 10→11 (Option B road filter, ~29 % AOI);
  11→12 (NATURE_RADIUS_CAP_M = 1000 m fixes nature-access saturation; BASELINE_CN now dynamically
  computed at module load); 12→13 (load_data parameterized via city_cfg path keys; Minneapolis
  Full activated); 13→14 (InVEST Urban Mental Health v3.19.0 added — preventable_mh_cases +
  avoided_mh_cost_usd as new surrogate targets, replaces Urban Wellbeing Score metric card);
  **14→15 (San Antonio activated with full pipeline: SSURGO TX029 + Census Bexar +
  CGIAR ET0 + TIGER 48 + Geofabrik TX OSM; new EPA Social Cost of Carbon dollar
  metric in Economic row; pre-flight data-check function added; PIXEL_AREA_ACRES
  harmonized to 0.2224 globally).**
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

- **San Antonio stable on Streamlit Cloud (2026-05-11).** The 1011 keepalive
  loop OOM on slider interaction was resolved by a stack of changes:
  float32 downcast of module-level geospatial arrays, disk-cached static
  nature-distance fields, `@st.cache_resource`-backed `_load_city_runtime_state`
  (so heavy work runs once per session per city instead of every rerun),
  in-place ops in the `_compute_cc_raster_pure` chain, and uint8 +
  1024px-cap downsample in `plot_spatial_map` (which was allocating
  ~378 MB transient per rerun on the SA AOI before the fix). SA is the
  default test bed for any future memory-sensitive change.
- **Cross-city cooling comparability — pending.** Cooling Energy Savings
  for MN is computed against the **InVEST sample buildings shapefile**
  (3,788 polygons rasterising to ~447 pixels, ~0.4 km² — downtown only),
  while SA is computed against **Geofabrik OSM buildings for all of Bexar
  County** (345,900 polygons; 36,860 typed pixels covering ~33 km²). The
  per-typed-building-pixel rate is in line between the two cities
  (~$1,200/yr/pixel at the 50/0/100/0 scenario), but the absolute dollar
  values are NOT directly comparable across cities — SA's number is a
  county-wide sum, MN's is a downtown-subset sum. Awaiting a decision on
  how to surface the scope difference to users (tooltip caveat vs adding
  OSM buildings to MN vs per-rate display). Don't pull MN/SA cooling
  values into the same chart or comparison sentence until this is
  resolved.
- **SA flood damage table sourcing — pending.** `CITIES['San Antonio, TX']
  ['damage_table_file']` is `None`, so `compute_flood_damage_avoided` returns
  $0 for SA even now that OSM building polygons carry InVEST type codes.
  Reusing `Damage_loss_table_MN.csv` was rejected — the MN per-m² damage
  values are calibrated to MN's InVEST sample buildings (specific size /
  construction profile), and applying them to 345k SA OSM polygons would
  produce a misleading number on a different building stock. Sourcing a
  SA-specific damage table is its own (small) followup commit. Until then,
  the Flood Damage Avoided card honestly renders "—" / $0 for SA.
- **Heat Vulnerability Index — still pending.** The `equity_weights`
  raster is a proxy (NLCD intensity-coded), not a real CDC/ATSDR HVI by
  census tract. Replacing it is the next data-quality upgrade.
- **Full Minneapolis extent — RESOLVED 2026-05-09, HIDDEN FROM UI 2026-05-11.**
  `'Minneapolis Full, MN'` is a live city in CITIES but `available=False` so it does NOT appear in
  the sidebar selector. Reason: per-building-type dollar metrics (Flood Damage Avoided, Cooling
  Energy Savings) require InVEST sample buildings with `type` ∈ {0,1,2,3}, which only cover the
  downtown extent — Mpls Full uses OSM polygons with no type codes (Option A), so those cards
  degrade to "—". Showing only the downtown city in the UI keeps the metric coverage complete.
  All pipeline + rasters + verified baselines remain in the repo; flip back to `True` once a
  typed building dataset exists for the expanded area. Pipeline: SSURGO via SDA REST API →
  process_ssurgo.py → soil_group_mpls_full.tif; Census 2020 → process_pop_expanded.py →
  pop_mpls_full.tif; Geofabrik state OSM → process_osm_expanded.py → roads_mpls_full.geojson +
  buildings_mpls_full.gpkg; TIGER 2020 → tracts_hennepin.shp. Schema bumped 12 → 13.
- **load_data parameterization (2026-05-09).** `load_data()` now takes `lulc_file`, `soil_file`,
  `cooling_lulc_file` from `city_cfg`. Module-level loaders for ET, energy table, UNA table,
  buildings, roads, and tracts also read from city_cfg. Biophysical tables (CN + cooling) use a
  fallback path via `_resolve_table()` so cities with custom data_dirs (Mpls Full pointing at
  `data/minneapolis_expanded/`) can still reference the project-shared tables in `data/flood/`
  and `data/cooling/`. EPSG:26915 hardcodes replaced with `city_cfg['crs']`.

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
  that the loader invokes (currently `_compute_cc_raster`,
  `_compute_hm_raster`, `_compute_access_score_raster`) come in two variants:
  `_fn(scenario_lulc)` reads module aliases populated by the loader, and
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
- **Unavailable cities are filtered out of the sidebar selector** at the
  selectbox-options stage, so the "coming soon" UI branch is no longer
  reachable. If a future city needs an explicit "data being prepared"
  state, add it back as a filter on `available` plus a sidebar caption,
  not as an `st.stop()` after data loading.
- **Scenario LULC is not stored in the lookup table** — `scenario_lulc` is stripped from cached
  results (`if k != 'scenario_lulc'`) and recomputed on demand for the map view to keep memory
  usage manageable.
- **Metric cards are grouped into four labeled sections** — 🌿 Ecological (5 cards in two
  rows: row 1 has Flood Risk Reduction, Temperature Change, and Runoff Volume in 3 columns;
  row 2 has Carbon Sequestration and NDVI in 2 columns),
  👥 Human & Social (4 cards in 4 columns: Nature Access, Nature Quality Score, Preventable
  MH Cases, Avoided MH Costs — the InVEST Urban Mental Health v3.19.0 outputs replaced the
  earlier weighted-composite Wellbeing Score),
  💵 Economic (5 cards in two rows: row 1 has Food Production + Est. Implementation Cost
  in 2 columns; row 2 has Flood Damage Avoided + Cooling Energy Savings + Avoided Carbon
  Cost in 3 columns — the EPA Social Cost of Carbon dollar metric is `carbon_tons_co2_yr ×
  EPA_SOCIAL_COST_CARBON`, deterministic so not in the surrogate),
  📊 Cost Effectiveness (3 sub-ratios under their own header). Each group is separated by
  `st.divider()`. **14 metric cards total**. Keep this grouping when adding new metrics — place
  new cards in the section that matches their category rather than appending to a flat list.
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
- **InVEST Urban Mental Health Model (v3.19.0)** — `calculate_mental_health_impact(scenario_lulc,
  baseline_ne_raster, pop_count)` returns `(preventable_mh_cases, avoided_mh_cost_usd)`. Per-pixel
  formula: NE = `gaussian_filter(NDVI_proxy, sigma=10 px)` (10 px = 300 m at 30 m NLCD per
  Li et al. 2025); ΔNE = NE_scenario − NE_baseline; RR = `exp(ln(RR₀.₁) × 10 × ΔNE)`;
  PC = `(1 − RR) × baseline_prevalence × population`. Sums depression + anxiety. Constants:
  `RR_0_1_NDVI_DEPRESSION=0.96`, `RR_0_1_NDVI_ANXIETY=0.97` (Liu et al. 2023 meta-analysis);
  `BIR_DEPRESSION=0.21`, `BIR_ANXIETY=0.19` (CDC 2023 ever-diagnosed); per-case cost-of-illness
  $8,467 / $5,765 (US nominal). Returns (0, 0) at the unmodified baseline by construction.
  `_BASELINE_NE_RASTER` is precomputed once at module load. The previous `compute_wellbeing_score`
  composite + `wgt_ndvi/wgt_cooling/wgt_nature` sliders + `DEFAULT_WGT_*` constants have been
  removed entirely — UMH outputs are derived from peer-reviewed effect sizes rather than
  user-tunable weights, so there's nothing to expose in the sidebar.
- **The surrogate predicts eight outputs** — `train_surrogate` fits the Random Forest on
  `[flood_reduction, mean_hm, food_mln_lbs, runoff_acre_feet, carbon_tons_co2_yr,
  nature_access_pct, preventable_mh_cases, avoided_mh_cost_usd]`, so `predict_with_uncertainty`
  returns `(n, 8)` arrays.
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
