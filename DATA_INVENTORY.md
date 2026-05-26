# Data Inventory

**Purpose:** A canonical reference for every data source the Urban Ecosystem Tradeoff Explorer prototype consumes. Covers both supported cities (Minneapolis, San Antonio) and the dormant Minneapolis Full configuration, organized by data category rather than by city. The goal is that any future session can read this file and understand what data exists, where it lives, what it's for, and what's pending — without re-discovering the same context from scratch.

**Status:** v1 finalized 2026-05-24. v2 audited 2026-05-26 against the live repo (post-Brief 33) — reconciled with Briefs 14 (SA UHI), 23 (per-city rainfall), 27 (compound LULC), 28b (compound UCM), 29 (compound UNA), 30 (four-pool Carbon), 31 (ACS block groups), 33 (Flood Retention reframe). Maintained alongside `REFERENCE.md`, `DESIGN_NOTES.md`, and `NATCAP_ALIGNMENT.md`.

**Scope:** External data sources the prototype consumes — rasters, vectors, tabular data, biophysical parameter tables. Derived intermediates and runtime-computed rasters are *not* enumerated here (those live in code).

---

## 1. Top-level data tree

```
data/
├── cooling/                     # MN downtown cooling — LULC + biophysical table
│   ├── biophysical_table_urban_cooling_MN.csv
│   └── land_use_2021.tif
├── flood/                       # MN downtown flood — LULC + soil + CN/damage tables
│   ├── Damage_loss_table_MN.csv
│   ├── LULC_NLCD_2021_MN.tif
│   ├── soil_group_MN.tif
│   └── UFR_biophysical_table_MN.csv
├── invest/                      # InVEST sample bundles (MN; pre-built reference inputs)
│   ├── cooling/UCM_AUDIT.md
│   ├── cooling/UrbanCooling_sample_data/UrbanCooling/  (LULC, ET, buildings, args.json, energy_consumption.csv, biophysical_table_urban_cooling.csv)
│   ├── flood/UFR_sample_data_MN/                       (LULC, AOI, buildings, args.json, Damage_loss_table_MN.csv)
│   └── nature_access/UrbanNatureAccess_sample_data_MN/ (LULC, tracts, population, args.json, LULC_attribute_table_UNA.csv)
├── minneapolis_expanded/        # MN Full (dormant city config)
│   ├── lulc_nlcd_2021_mpls_full.tif
│   ├── pop_mpls_full.tif
│   ├── roads_mpls_full.geojson
│   ├── buildings_mpls_full.gpkg
│   ├── soil_group_mpls_full.tif
│   ├── ssurgo_hennepin_hsg.{shp,dbf,prj,...}    (raw SSURGO polygons)
│   ├── ssurgo_hydgrp_hennepin.csv               (raw SSURGO HSG attributes)
│   └── tracts_hennepin.{shp,dbf,prj,...}
├── osm/                         # Raw OSM extracts (Geofabrik) — large, ~7.2 GB total
│   ├── minnesota.shp.zip                        (raw Minnesota state extract)
│   ├── minneapolis_buildings.geojson            (filtered MN city-wide buildings)
│   ├── minneapolis_roads.geojson                (filtered MN city-wide roads, Option B class set)
│   └── texas.shp.zip                            (raw Texas state extract)
├── population/                  # MN population
│   ├── minneapolis_pop_2020.tif
│   └── tiger/                                   (TIGER block-shapefiles cache)
├── sa/                          # San Antonio (current independent setup)
│   ├── README.md
│   ├── buildings_sa.gpkg
│   ├── roads_sa.geojson
│   ├── tracts_bexar.{shp,dbf,prj,shx,cpg}       (current SA tracts at this top level)
│   ├── tracts_bexar/            (raw TIGER 2020 Texas tract shapefile)
│   ├── cooling/                 (SA cooling — biophysical table + ET + raw CGIAR archive)
│   │   ├── biophysical_table_urban_cooling_SA.csv
│   │   ├── biophysical_table_sources.md
│   │   ├── et_annual_sa.tif
│   │   └── cgiar_et0/                           (raw CGIAR Global-AI/ET0 v3.1 download)
│   ├── flood/                   (SA flood — LULC + soil + CN table + raw SSURGO)
│   │   ├── land_use_2021_sa.tif
│   │   ├── lulc_nlcd_2021_sa.tif
│   │   ├── soil_group_sa.tif
│   │   ├── UFR_biophysical_table_SA.csv
│   │   ├── ssurgo_bexar_hsg.{shp,...}           (raw SSURGO polygons, .gitignored)
│   │   └── ssurgo_hydgrp_bexar.csv              (raw SSURGO HSG attributes, .gitignored)
│   ├── natcap_2024/             (NatCap-curated SA dataset received 2026-05-23 — see §2/§9)
│   └── population/sa_pop_2020.tif
├── scenarios_dense_mpls.csv     # Per-city dense lookup tables (Brief 4 era)
├── scenarios_dense_mpls_full.csv
└── scenarios_dense_sa.csv
```

`du -sh data/*/`: cooling 296 K · flood 320 K · invest 3.2 M · minneapolis_expanded 115 M · osm **7.2 G** · population 971 M · sa **1.3 G**. `du -sh data/sa/*/`: cooling 835 M · flood 27 M · natcap_2024 321 M · population 396 K · tracts_bexar 51 M.

---

## 2. Land cover and land use (LULC)

The prototype uses **NLCD 2021** (legacy MRLC product) across all currently active cities. NLCD 2021 is also the LULC vintage shipped with the InVEST UFR, UCM, and UNA sample data.

### Minneapolis (downtown) — two rasters

| Role | Path | CRS | Dimensions | Source | Provenance |
|---|---|---|---|---|---|
| Cooling & scenario LULC | `data/cooling/land_use_2021.tif` | EPSG:26915 (UTM 15N) | 356 × 360, int16 | InVEST UNA sample bundle | **Byte-identical** to the InVEST UNA sample LULC (MD5 `56d1080fa70576cad15896642a107a3d`). See `UNA_LULC_INVESTIGATION.md`. |
| Flood / CN LULC | `data/flood/LULC_NLCD_2021_MN.tif` | EPSG:26915 | 356 × 360, int16 | InVEST UFR sample bundle | Same AOI and grid as the cooling LULC but a distinct file (MD5 `a8687db9f76394aa1333b8a3d35ec57e`). |

Minneapolis downtown is the only city where flood and cooling use *different* LULC rasters — an artifact of inheriting from separate InVEST sample bundles (UFR vs UCM/UNA). The originals are preserved under `data/invest/flood/UFR_sample_data_MN/LULC_NLCD_2021_MN.tif` and `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_NLCD_2021.tif`.

### Minneapolis Full (dormant) — single raster

| Role | Path | CRS | Dimensions | Source |
|---|---|---|---|---|
| All LULC | `data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif` | EPSG:5070 (CONUS Albers) | 607 × 374, uint8 | NLCD 2021 via MRLC WCS, fetched by `download_minneapolis_nlcd.py` |

`available=False` in the city config — hidden from the city selector but retained so scripts/tests can still reference it by key.

### San Antonio — live LULC rasters (Brief 27 dual-raster pipeline)

| Role | Path | CRS | Dimensions | Source |
|---|---|---|---|---|
| Compound LULC (live canonical, compound-keyed) | `data/sa/flood/land_use_compound_sa.tif` | EPSG:5070 | 1713 × 1984 | NatCap `lulc_overlay_3857.tif` reprojected EPSG:3857 → EPSG:5070 with nearest-neighbor at 30 m (Brief 27). 800 unique compound lucodes of 1,984 possible. The live SA pipeline reads this raster; UCM/UNA/Carbon biophysical tables join directly on its compound lucode. |
| NLCD-only LULC (fallback / NLCD-keyed paths) | `data/sa/flood/land_use_2021_sa.tif` | EPSG:5070 | 1713 × 1984, uint8 | NLCD 2021 via MRLC WCS, fetched by `download_sa_data.py`. Retained as fallback; CN biophysical table (`UFR_biophysical_table_SA.csv`) is NLCD-keyed and reduces compound → NLCD via `lulc_crosswalk.csv`. |

Both LULC rasters live in `data/sa/flood/` on the same 1713 × 1984 EPSG:5070 grid. `lulc_file` config entry points to the NLCD-only raster; `compound_lulc_file` points to the compound raster (`config.py:271`). A second copy `data/sa/flood/lulc_nlcd_2021_sa.tif` also exists in the same folder (provenance unclear; sibling of the canonical NLCD file).

### San Antonio — NatCap-curated source rasters (raw delivery, EPSG:3857)

| Role | Path | CRS | Dimensions | Pixel size | Source |
|---|---|---|---|---|---|
| Compound NLCD+NLUD+Tree overlay LULC (raw) | `data/sa/natcap_2024/lulc_overlay_3857.tif` | EPSG:3857 (Web Mercator) | 2106 × 2218 | 34.5 m | NatCap NASA Urban project, Aug 2024 vintage. Reprojected by Brief 27 to the live `land_use_compound_sa.tif` above. |
| Source NLCD layer (retained) | `data/sa/natcap_2024/nlcd_3857.tif` | EPSG:3857 | 2106 × 2218 | 34.5 m | NLCD 2021 reprojected by NatCap |
| Source NLUD layer (retained) | `data/sa/natcap_2024/nlud_3857.tif` | EPSG:3857 | 2106 × 2218 | 34.5 m | National Land Use Database, USGS |
| Source tree canopy layer (retained) | `data/sa/natcap_2024/tree_3857.tif` | EPSG:3857 | 2106 × 2218 | 34.5 m | NLCD 2021 tree canopy product |

All four rasters share the same 2106 × 2218 grid at 34.5 m pixel size, EPSG:3857. Extent: 98.84°W–98.19°W, 29.16°N–29.76°N — roughly the Bexar County urban core. Brief 27 reprojected to EPSG:5070 + nearest-neighbor resampled at 30 m to align with the existing SA stack (`land_use_compound_sa.tif`). All NatCap rasters are gitignored (`data/sa/natcap_2024/*.tif`) except small CSVs/docs.

The compound `lulc_overlay_3857.tif` encodes a **Cartesian product lucode space**: 16 NLCD codes × 31 NLUD classes × 4 tree-canopy levels = 1,984 distinct lucodes. The biophysical tables (UCM/UNA/Carbon) are keyed on this compound lucode — live as SA's biophysical tables across Briefs 28b (UCM), 29 (UNA), 30 (Carbon four-pool stock change).

**Committed files in `data/sa/natcap_2024/`** (provenance + small data, grep-able where applicable):

| File | Content |
|---|---|
| `README.docx` + `README.txt` | NatCap's original SA dataset README (predates `README_San_Antonio_InVEST_model_inputs.docx`). |
| `Notes_on_NASA_Urban_parameterization_QA.docx` + `.txt` | NatCap's NASA Urban project parameterization QA notes. |
| `README_San_Antonio_InVEST_model_inputs.docx` + `.txt` | NatCap's per-InVEST-model SA input recipe — args.json-equivalent values for UCM, Carbon, UNA, UFR, NDR (read 2026-05-24). Source of the Brief 14 UHI calibration and the NDR integration scope. |
| `Ecosystem_Explorer_-_Meeting_Note.docx` + `.txt` | NatCap meeting note with project context: Symposium 2026 dates, Google AI for Science proposal framing, six-model SA scope, "wallpaper approach" definition (read 2026-05-24). |
| `ucm__nlcd_nlud_tree.csv` | NatCap SA Urban Cooling biophysical table — 1,984-row compound NLCD×NLUD×tree-canopy lookup with shade/kc/albedo/green_area/building_intensity per compound lucode. **Live as SA's UCM biophysical table (Brief 28b)** via `cooling_table_file` config pointer; retired the prior Köppen BSh-tuned NLCD table. |
| `una__nlcd_nlud_tree.csv` | NatCap SA Urban Nature Access LULC attribute table — 1,984-row categorical (0/0.5/1.0) `urban_nature` score per compound lucode. **Live as SA's UNA biophysical table (Brief 29)** via `una_table_file` config pointer. |
| `carbon__nlcd_nlud_tree.csv` | NatCap SA Carbon table — 1,984-row, four-pool (above/below/soil/dead carbon, tons C/ha) per compound lucode. **Live as SA's Carbon biophysical table (Brief 30)** — driving the SA-only one-time stock-change methodology (`_CARBON_IS_STOCK = True`); MN retains its single-rate annual flow. |
| `lulc_crosswalk.csv` | Cross-reference table mapping NLCD codes × NLUD codes × tree-canopy bins → compound lucodes used in the three biophysical tables above. Essential for interpreting them. |
| `acs_block_group_equity_data.csv` | Census ACS demographic + equity data joined to SA block groups. Source for any equity-by-group analysis. |
| `acs_block_groups_3857.gpkg` | SA Census block group polygons in EPSG:3857. Adopted Brief 31 as SA's `tracts_file` (replaced the prototype's Bexar County bbox); reprojected to EPSG:5070 at load time. 1,124 polygons covering the City of San Antonio. |
| `classification_structure_qaqc.xlsx` | NatCap's methodology QA/QC documentation for the compound LULC framework. Binary file (Excel); kept as-is for provenance. |

`.docx` files preserved alongside `textutil`-converted `.txt` versions so the contents are grep-able in the repo.

### NLCD vintage — legacy NLCD 2021 confirmed for NatCap data

Resolved 2026-05-24. `nlcd_3857.tif` contains 16 unique non-zero values: `{11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95}` plus 0/nodata. This is **consistent with the legacy NLCD 21-class schema** — no Annual NLCD-specific codes are present. The prototype's continued use of legacy NLCD 2021 aligns with what NatCap shipped here.

USGS replaced legacy NLCD with **Annual NLCD** in 2024 (revised methodology, 21 → 16 classes). MRLC states: *"Legacy NLCD data are not directly comparable to the newer Annual NLCD data."* The prototype stays on legacy NLCD 2021 because the InVEST sample data and biophysical tables are calibrated against the 21-class legacy schema. Migrating to Annual NLCD would require re-validating every lucode mapping and regenerating all baselines.

---

## 3. Soil hydrologic groups (SSURGO)

USDA Soil Survey Geographic Database, rasterized to match the NLCD grid. Soil hydrologic group (A/B/C/D) is a key input to the SCS Curve Number equation for flood modeling: Group A soils (sandy) absorb water; Group D (clay-rich) sheds water.

### Minneapolis (downtown)

| Path | Source | Coverage |
|---|---|---|
| `data/flood/soil_group_MN.tif` | InVEST UFR sample shapefile, pre-rasterized | Downtown core, ~71.1 km² |

### Minneapolis Full (dormant)

| Path | Source | Coverage | Pipeline |
|---|---|---|---|
| `data/minneapolis_expanded/soil_group_mpls_full.tif` | USDA Soil Data Access REST API | Full Hennepin County, ~1,572 km², 32,442 polygons | `download_ssurgo.py` → `process_ssurgo.py` |

Raw inputs preserved as `data/minneapolis_expanded/ssurgo_hennepin_hsg.{shp,dbf,prj,shx,cpg}` + `ssurgo_hydgrp_hennepin.csv`. 9% of polygons (Udorthents / engineered urban fill / open water) have no `hydgrp` value and are reassigned to C-class per NRCS convention.

### San Antonio

| Path | Source | Pipeline |
|---|---|---|
| `data/sa/flood/soil_group_sa.tif` | USDA SSURGO API | `download_ssurgo_sa.py` → `process_ssurgo_sa.py` |

Raw inputs preserved as `data/sa/flood/ssurgo_bexar_hsg.{shp,dbf,prj,shx,cpg}` + `ssurgo_hydgrp_bexar.csv` (both gitignored).

---

## 4. Buildings

Two sources per city, used for different purposes. The InVEST sample buildings have per-building **type codes** (commercial / residential / industrial / other) that drive the *Cooling Energy Savings* and *Flood Damage Avoided* dollar metrics. The OSM buildings are **untyped** and used only for the placement non-convertible mask.

### Minneapolis (downtown) — typed UFR sample

| Path | Source | Count | Type codes |
|---|---|---|---|
| `data/invest/flood/UFR_sample_data_MN/buildings.shp` | InVEST UFR sample shapefile | 3,788 polygons | 0=other, 1=commercial, 2=residential, 3=industrial |

### Minneapolis (city-wide) — untyped OSM

| Path | Source | Count | Pipeline |
|---|---|---|---|
| `data/osm/minneapolis_buildings.geojson` | OpenStreetMap via Geofabrik Minnesota state extract (raw zip kept at `data/osm/minnesota.shp.zip`) | 185,490 polygons | `download_osm_minneapolis.py` |

For Minneapolis downtown's placement-strategy non-convertible mask, the prototype now unions UFR-sample buildings with Geofabrik OSM buildings (~113k city-wide). Conversions can't land on any OSM building anywhere in the AOI. Cooling Energy Savings and Flood Damage Avoided continue to use the typed UFR-sample subset.

### Minneapolis Full

| Path | Source |
|---|---|
| `data/minneapolis_expanded/buildings_mpls_full.gpkg` | Geofabrik OSM, filtered to Hennepin County |

### San Antonio

| Path | Source | Pipeline |
|---|---|---|
| `data/sa/buildings_sa.gpkg` | OpenStreetMap via Geofabrik Texas extract (raw zip `data/osm/texas.shp.zip`) | `download_osm_sa.py` |

SA has **no per-building type codes available** — Cooling Energy Savings and Flood Damage Avoided degrade to `$0` with explanatory tooltips. The NatCap shared-Drive folder contains a separate `building footprints/` directory that may include typed SA buildings (not yet downloaded; see §15 Q6).

### NatCap-provided buildings (potential future source)

The screenshot-visible "building footprints" folder in your Drive's "Shared with me" is not yet downloaded. May contain NatCap-curated typed buildings for SA. *Open question; queued for SA NatCap integration workstream.*

---

## 5. Roads

OpenStreetMap road network, used to mask roads from conversion-eligible pixels. Filtered using **Option B**: drop sub-pixel-width surfaces (footway, cycleway, steps, service, path, pedestrian, unclassified, track*). Retained: motorway, trunk, primary, secondary, tertiary, residential, living-street, and on/off-ramp links.

### Minneapolis (downtown + city-wide)

| Path | Count | Coverage | Pipeline |
|---|---|---|---|
| `data/osm/minneapolis_roads.geojson` | 5,495 segments | ~29% AOI coverage | `download_osm_minneapolis.py` |

### Minneapolis Full

| Path | Count | Pipeline |
|---|---|---|
| `data/minneapolis_expanded/roads_mpls_full.geojson` | 10,984 segments | `process_osm_expanded.py` |

### San Antonio

| Path | Pipeline |
|---|---|
| `data/sa/roads_sa.geojson` | `download_osm_sa.py` |

Rasterized at startup and unioned into `BUILDINGS_RASTER` so green-conversions can't land on streets.

### NatCap-provided roads (potential future source)

A "roads" folder is visible in your Drive's "Shared with me" but not yet downloaded.

---

## 6. Population

Used for the InVEST UNA per-capita supply calculation, neighborhood reporting, and (legacy) the equity-focused placement strategy — renamed to `undersupply-focused` in Brief 9, which removed the population-multiplier framing in favor of canonical per-capita UNA supply deficit.

### Minneapolis (downtown)

| Path | Source | Pipeline | Vintage |
|---|---|---|---|
| `data/population/minneapolis_pop_2020.tif` | US Census 2020 block-level (canonical) | `download_census_pop.py` | 2020 |

**Caveat:** the repo also contains `clip_worldpop.py`, an *alternative* MN downtown population pipeline that produces the same output path from a USA WorldPop raster. Both scripts target `data/population/minneapolis_pop_2020.tif`. The canonical setup per REFERENCE.md / CLAUDE.md is the Census pipeline. The on-disk file's provenance cannot be determined from the file alone; if WorldPop was the source, totals would diverge from the Census reference.

TIGER 2020 tabulation-block polygons (cached under `data/population/tiger/`) join to the Census table via `GEOID20` and rasterize to the active grid.

### Minneapolis Full

| Path | Source | Pipeline |
|---|---|---|
| `data/minneapolis_expanded/pop_mpls_full.tif` | Census 2020 blocks, Hennepin County | `process_pop_expanded.py` |

### San Antonio (current)

| Path | Source | Pipeline |
|---|---|---|
| `data/sa/population/sa_pop_2020.tif` | US Census 2020 blocks, Bexar County FIPS 48029 | `download_census_pop_sa.py` |

### San Antonio (NatCap-curated, pending integration)

| Path | Vintage | Dimensions | Pixel size |
|---|---|---|---|
| `data/sa/natcap_2024/population_per_pixel_2020_3857.tif` | 2020 | 2106 × 2218 | 34.5 m (EPSG:3857) |

19 MB raster, per-pixel population counts at 34.5 m. Source attribution is not embedded in the GeoTIFF metadata; likely derived from WorldPop or a NatCap-internal downscaling. Same grid mismatch issue as the LULC (EPSG:3857 vs current EPSG:5070).

---

## 7. Census tracts and demographics

For per-tract neighborhood-improvement reporting overlay on the Map View tab.

### Minneapolis (downtown)

| Source | Count | Note |
|---|---|---|
| TIGER 2020 (inherited from InVEST UNA sample at `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/AOI_admin_boundaries_census_tracts.shp`) | 27 tracts intersecting the InVEST sample AOI | Subset of Hennepin County tracts |

### Minneapolis Full

| Path | Source | Count | Pipeline |
|---|---|---|---|
| `data/minneapolis_expanded/tracts_hennepin.shp` | TIGER 2020 | All 329 Hennepin County tracts | `process_tracts_expanded.py` |

### San Antonio

| Path | Source | Count | Pipeline |
|---|---|---|---|
| `data/sa/tracts_bexar.shp` (operational) + raw at `data/sa/tracts_bexar/tl_2020_48_tract.shp` | TIGER 2020 | All 375 Bexar County tracts | `process_tracts_sa.py` |

### NatCap-provided ACS block groups (adopted Brief 31)

| Path | Content | Note |
|---|---|---|
| `data/sa/natcap_2024/acs_block_groups_3857.gpkg` | Census block-group polygons for the SA study area (1,124 polygons, EPSG:3857; reprojected to EPSG:5070 at load time) | Live as SA's `tracts_file`; consumed by `compute_per_tract_summary`'s Neighborhood breakdown table. Per-city dashboard caption is conditional ("Census tracts" for MN, "Census block groups" for SA). |
| `data/sa/natcap_2024/acs_block_group_equity_data.csv` | Processed ACS demographics (percent BIPOC, per-capita income) + zonal-stat results from canonical InVEST runs | Joins to gpkg by `GEO_ID`. Not yet wired into the dashboard. |

CSV includes bivariate-color-scheme fields for plotting (percent_bipoc bins × average_temp bins → bivariate_category → hex colors). Intended for equity-overlay maps.

---

## 8. Reference evapotranspiration (ET₀)

Annual ET₀ raster, used in the UCM Kc × ETI calculation.

### Minneapolis (both)

| Path | Source | Resolution | Extent |
|---|---|---|---|
| `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif` | InVEST UCM sample data | 1 km, ~10 × 10 km native extent | Bilinear-resampled to the NLCD grid |

For Minneapolis Full, the raster extrapolates beyond its native extent at the bbox corners. Nodata sentinel (65535) is masked before resize.

### San Antonio (current)

| Path | Source | Resolution | Range |
|---|---|---|---|
| `data/sa/cooling/et_annual_sa.tif` | CGIAR Global-AI/ET0 v3.1 (raw 645 MB zip preserved at `data/sa/cooling/cgiar_et0/Global-AI_ET0__annual_v3_1.zip` → reprojected `et0_v31_yr.tif`), reprojected to EPSG:5070 | 30 arc-seconds (~1 km) | Annual PET 1,580–1,716 mm/yr (mean 1,657) for the SA bbox |

Pipeline: `download_et_sa.py`. SA's PET is ~50% higher than MN's ~1,140 mm/yr, but enters the CC formula via normalized ETI so absolute mm/yr cancels (see REFERENCE.md "Cross-city Heat Mitigation Index comparison").

### San Antonio (NatCap-curated, pending integration)

| Path | Dimensions | Pixel size | Verdict |
|---|---|---|---|
| `data/sa/natcap_2024/et0_annual_cgiar_3857.tif` | **60 × 63** | **1,215 m** | **Probably not safe to adopt as-is.** |

The NatCap-curated CGIAR ET₀ file is much coarser than the operational SA raster (1,215 m vs the operational ~1 km source resampled to 30 m). Same extent as the NatCap LULC (98.84°W–98.19°W, 29.16°N–29.76°N) but ~40× lower resolution. If migrating to the NatCap stack, ET₀ should be re-downloaded from CGIAR at the native source resolution rather than adopted from this file. See §15 Q5.

---

## 9. Biophysical parameter tables

The lookup tables that translate LULC codes into per-pixel model parameters. **This is the area where the NatCap data diverges most from the current setup.**

### 9.1 Curve Number (UFR / flood)

Per-city tables, NOT in `config.py` — declared via `CITIES[city]['cn_table_file']`:

| City | Path | Source |
|---|---|---|
| MN downtown | `data/flood/UFR_biophysical_table_MN.csv` | InVEST UFR sample, MN |
| SA | `data/sa/flood/UFR_biophysical_table_SA.csv` | InVEST UFR sample extended for SA; NLCD code 82 (Cultivated Crops) added because ~6.8% of the SA bbox is cropland |

The NatCap dataset does **not** include a Curve Number table. CN values for the compound lucode space would need to be derived (likely by mapping compound lucode → underlying NLCD → existing CN table).

### 9.2 Urban Cooling biophysical (shade/Kc/albedo)

**Live per-city tables:**

| City | Path | Lookup key | Notes |
|---|---|---|---|
| MN (both) | `data/cooling/biophysical_table_urban_cooling_MN.csv` (declared via `CITIES[city]['cooling_table_file']`) | NLCD lucode | From InVEST UCM args JSON for the MN AOI (humid continental Köppen Dfa) |
| SA | `data/sa/natcap_2024/ucm__nlcd_nlud_tree.csv` | Compound lucode (NLCD × NLUD × tree-canopy) | NatCap compound table, adopted Brief 28b. 1,984 rows × 27 columns; per-pixel `shade`, `kc`, `albedo`, `green_area`, `building_intensity`. Tree canopy is the dominant signal — any pixel with `tree_canopy_cover='high'` gets shade=0.66 (same as forest) regardless of underlying NLCD. NLUD context tweaks Kc and albedo by up to 10 % based on expected irrigation. |

**Retired:** `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` (Köppen BSh-tuned NLCD-keyed table). Kept on disk for reference; per-class rationale sidecar `data/sa/cooling/biophysical_table_sources.md` documents the historical tuning. Brief 28b's compound-table swap shifted SA `baseline_hm` from 0.2866 → 0.3937 (+37 %) by crediting per-pixel tree-canopy variation the per-NLCD framework couldn't represent.

### 9.3 Urban Nature Access biophysical

**Live per-city tables:**

| City | Path | Lookup key | Notes |
|---|---|---|---|
| MN (both) | `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv` | NLCD lucode | Per-class `urban_nature` score (0/0.5/1) + per-class `search_radius_m`. |
| SA | `data/sa/natcap_2024/una__nlcd_nlud_tree.csv` | Compound lucode | NatCap compound table, adopted Brief 29. 1,984 rows × 21 cols. Per-pixel `urban_nature` ∈ {0.0, 0.5, 1.0} indexed via per-city `urban_nature_arr` numpy lookup. Distribution: 976 rows = 1.0, 960 = 0.0, 48 = 0.5; the 0.5 score appears only for Conservation-class NLUD pixels. |

Shared per-city scalars: demand `UNA_DEMAND_M2_PER_CAPITA = 16.7` (constant in app.py — per-city values match in current configs). Per-city `search_radius_m` is an args-level scalar from `city_cfg['una_search_radius_m']` (800 m for SA per Brief 22, confirmed by NatCap's `kernel_800.0.tif`); MN's NLCD-keyed radii are capped at 1000 m (`NATURE_RADIUS_CAP_M`) at runtime to prevent saturation. The SA NatCap table's `search_radius_m` column is all zeros — the radius is the args-level scalar, not the per-row table value.

### 9.4 Carbon biophysical

**Live per-city methodologies — annual flow (MN) vs. one-time stock (SA):**

| City | Source | Lookup key | Methodology |
|---|---|---|---|
| MN (both) | USDA NRCS / IPCC per-cover-class rates (embedded in app.py / config.py) | NLCD lucode → single sequestration rate (tons CO₂e/ha/yr) | Annual flow — multiplied by converted area. User-overridable via Advanced Settings sliders (`carbon_rate_ff`, `carbon_rate_gi`). |
| SA | `data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv` | Compound lucode | One-time stock change (Brief 30 adoption). Four pools `c_above` / `c_below` / `c_soil` / `c_dead` (tons C/ha) indexed via per-city `c_above_arr`/`c_below_arr`/`c_soil_arr`/`c_dead_arr` numpy arrays. Three additional columns (`c_embedded_storage`, `c_embedded_emissions`, `c_annual_emissions`) describe urban-accounting flows the prototype doesn't use. |

The unified return-dict key `carbon_tons_co2` carries either framing; the city-conditional `_CARBON_IS_STOCK` flag drives dashboard card labels and unit suffixes. See `DESIGN_NOTES.md` "SA Carbon four-pool framework adoption" for rationale and the methodology-matches-but-SC-CO2-vintage-differs alignment with NatCap's Vibrant Land report.

### 9.5 Food Forest yield

| Variable | MN value | SA value | Source |
|---|---|---|---|
| `FOOD_FOREST_LBS_ACRE` | 11,500 | 8,500 (placeholder) | MN: NatCap MN benchmark. SA: conservative estimate for hot semi-arid, pending NatCap SA project numbers |

Stored in `config.py` per-city. Single-value parameter, not a table.

---

## 10. Climate / urban-heat-island parameters

| Variable | MN | SA | Source |
|---|---|---|---|
| `UHI_MAX_C` | 2.05 °C | 11 °C | MN: InVEST UCM args JSON at `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/invest_urban_cooling_model_args_MN.json`. SA: NatCap SA README canonical value, migrated in Brief 14 (2026-05-24); replaced the prior 3.5 °C estimate. |
| `HM_TO_FAHRENHEIT` | 3.69 °F/HMI | 19.80 °F/HMI | Computed as `UHI_MAX_C × 1.8` per city |
| `DESIGN_STORM_INCHES` | 3.94 (100 mm) | 6.18 (157 mm) | NatCap per-city canonical, migrated in Brief 23 (2026-05-24): MN from `invest_urban_flood_risk_args_MN.json`, SA from NatCap SA README. Replaced the prior global `2.0` placeholder. |

Other UCM constants (`d_cool = 450 m`, `r = 600 m` blending) are model-architecture defaults, not city-specific.

---

## 11. Pre-computed validation outputs

### From NatCap-curated SA dataset (received 2026-05-23)

`data/sa/natcap_2024/InVEST Results/` — canonical InVEST model outputs run by NatCap on their curated inputs. Locally available, gitignored. Three subfolders:

| Subfolder | Key files | Use |
|---|---|---|
| `ucm/` | `hm.tif` (14 MB), `uhi_results.{shp,dbf,prj,shx}` (per-AOI vector summary), plus full intermediate kernels under `intermediate/`: `cc.tif`, `cc_park.tif`, `T_air.tif`, `T_air_nomix.tif`, `green_area.tif`, `green_area_sum.tif`, `eti.tif`, `kc.tif`, `albedo.tif`, `shade.tif`, `lulc.tif`, `ref_eto.tif`, plus reprojected AOI shapefile and `area_kernel.tif`. No `args.json`. | Ground truth for UCM validation. The prototype's `compare_ucm_invest.py` can diff against `hm.tif` once SA migration lands. |
| `carbon/` | `tot_c_cur.tif`, `report.html`, plus `intermediate_outputs/`. No `args.json`. | Ground truth for Carbon validation. |
| `una/` | `output/`: `urban_nature_supply_percapita.tif`, `urban_nature_balance_percapita.tif`, `urban_nature_balance_totalpop.tif`, `urban_nature_demand.tif`, `accessible_urban_nature.tif`, `admin_boundaries.gpkg`. `intermediate/`: `aligned_lulc.tif`, `aligned_population.tif`, `aligned_valid_pixels_mask.tif`, `distance_weighted_population_within_800.0.tif`, `kernel_800.0.tif`, `masked_lulc.tif`, `masked_population.tif`, `oversupplied_population.tif`, `undersupplied_population.tif`, `urban_nature_area.tif`, `urban_nature_population_ratio.tif`, `reprojected_admin_boundaries.gpkg`. No `args.json`. | Ground truth for UNA validation. The kernel filename (`kernel_800.0.tif`) confirms NatCap's search radius is 800 m — see §15 Q3. |

**Each model's `taskgraph_cache/` contains only `taskgraph_data.db` (a SQLite cache InVEST uses internally for incremental re-runs); does NOT contain a record of the user-facing run args.**

### From prototype itself

| Path | Content | Generator |
|---|---|---|
| `analysis/placement_diagnostic/layer{1,2,3}_*.csv` | Placement-strategy diagnostic measurements (suitability variance, chosen-pixel scores, metric outcomes) under the Brief 9 reformulated strategies | `placement_strategy_diagnostic.py` |
| `tests/baselines/*.json` | 40 baseline scenario snapshots (2 cities × 4 scenarios × 5 strategies) for regression testing. After Brief 9: equity-focused renamed to undersupply-focused; current `SCENARIO_SCHEMA_VERSION = 25` (last bumped Brief 30 for the SA Carbon four-pool framework). | Auto-regenerated by `verify_baselines.py --update`. |
| `comparisons/` | Seven files — `carbon_food_forest_mn.csv`, `ucm_baseline_mn.csv`, `ucm_diff_baseline_mn.tif`, `una_baseline_mn.csv`, `una_diff_baseline_mn.tif`, `una_lulc_comparison_mn.csv`, `surrogate_validation.csv`. Each is a one-shot validation snapshot produced by `compare_*_invest.py` or `validate_*.py` and committed for reference. | `compare_ucm_invest.py`, `compare_una_invest.py`, `compare_carbon_invest.py`, `validate_surrogate_predictions.py` |
| `data/scenarios_dense_{mpls,mpls_full,sa}.csv` | Pre-computed dense scenario lookup tables for instant slider response (per-city files arrived with Brief 4). The legacy bare `data/scenarios_dense.csv` was removed 2026-05-26 — it was dead-on-disk after every configured city got its own `dense_scenarios_file` key. | `precompute_scenarios.py` |

### Auxiliary

| Path | Status |
|---|---|
| `cache/` | Empty directory at repo root. Intended as a runtime scratch location; safely ignored. |
| `__pycache__/` | Python bytecode. Not data. |

The earlier `data/precomputed/` distance-transform cache (consumed by the homegrown reachability proxy that Brief 9 retired) was cleaned up 2026-05-24.

---

## 12. Documentation files (not data, but worth indexing)

For completeness — these are project documentation files in the repo root:

| File | Purpose |
|---|---|
| `REFERENCE.md` | User-facing methodology reference. Metric cards, data sources, computation architecture. |
| `DESIGN_NOTES.md` | Internal methodology decisions. Options considered, rationale, open questions. |
| `NATCAP_ALIGNMENT.md` | Per-surface alignment status with NatCap recommendations. Six tables. |
| `INVEST_PLACEMENT.md` | Per-InVEST-model placement-strategy analysis. |
| `PLACEMENT_STRATEGY_DIAGNOSTIC.md` | Empirical measurements of placement-strategy effects (Brief 6 baseline + Brief 9 reformulation findings). |
| `ALPHAEARTH_FEASIBILITY.md` | Research on AlphaEarth Foundations as future LULC source. |
| `UNA_DIVERGENCE_CASE_STUDIES.md`, `UNA_METHODOLOGY_CROSS_CHECK.md`, `UNA_QUALITY_SCORE_SENSITIVITY.md`, `UNA_LULC_INVESTIGATION.md` | UNA-specific investigations leading to the "temporarily removed" status of the Nature Quality Score. |
| `CLAUDE.md` | Working principles for Claude sessions. |
| `SPEC.md` | One-line: design specification — purpose, intended users, scope boundaries. Pre-dates most of the per-model methodology docs. |
| `SUMMARY.md`, `SUMMARY.docx`, `SUMMARY.pdf` | High-level prototype summary, refreshed in Brief 8 Addendum 2. |
| `REFERENCE.docx` | Word-format copy of REFERENCE.md, may lag the live `.md`. |
| `requirements.txt` | Python dependencies. |
| `runtime.txt` | One-line: `python-3.11`. Streamlit Cloud uses this to pin the Python version. |

---

## 13. Status of data integrations

| Integration | Status | Tracked in |
|---|---|---|
| Minneapolis downtown (cooling, flood, UNA, buildings, soil, population, tracts) | ✅ Active and validated | REFERENCE.md, DESIGN_NOTES.md |
| Minneapolis Full | ⏸️ Dormant — `available=False` in city config | DESIGN_NOTES.md |
| San Antonio current (independent NLCD + four-class hot-semi-arid cooling tuning) | ✅ Active | REFERENCE.md (SA caveats) |
| San Antonio NatCap-curated (NLCD+NLUD+tree compound lucode framework) | ✅ Adopted across LULC raster (Brief 27), UCM (Brief 28b), UNA (Brief 29), Carbon (Brief 30), and ACS block groups (Brief 31) | This doc §2/§6/§8/§9. Reflected in NATCAP_ALIGNMENT.md, CITY_PARITY.md, REFERENCE.md per their respective briefs. |
| Annual NLCD migration | 🔵 Not pursued — would require complete revalidation | DESIGN_NOTES.md "NLCD legacy vs Annual NLCD" |
| AlphaEarth satellite embeddings | 🔵 Feasibility research only | ALPHAEARTH_FEASIBILITY.md |
| SA building damage rates per-city | ✅ Resolved Brief 33 (Path C — match NatCap's Vibrant Land methodology, render "Flood Retention" % volume reduction instead of monetized damage). Reversible if NatCap surfaces SA-specific damage values. | DESIGN_NOTES.md "SA flood damage table — resolved (Path C, Brief 33)" |

---

## 14. Per-script inventory (data-producing pipelines)

For each download/process script in the repo root, what does it produce?

| Script | Produces | City |
|---|---|---|
| `download_minneapolis_nlcd.py` | MN NLCD raster (downtown) | MN downtown |
| `download_census_pop.py` | MN population raster from Census 2020 blocks → `data/population/minneapolis_pop_2020.tif` | MN downtown |
| `clip_worldpop.py` | **Alternative** MN population pipeline from USA WorldPop → same output path as `download_census_pop.py`. Only one source can be active on disk at a time. | MN downtown |
| `download_osm_minneapolis.py` | MN buildings + roads from Geofabrik OSM | MN downtown |
| `download_ssurgo.py` | Soil polygons via USDA SDA REST API | MN (downtown + expanded) |
| `process_ssurgo.py` | Rasterized soil hydrologic groups for MN Full | MN Full |
| `process_pop_expanded.py` | Population raster for MN Full | MN Full |
| `process_tracts_expanded.py` | Tracts for MN Full (all 329 Hennepin) | MN Full |
| `process_osm_expanded.py` | Roads for MN Full | MN Full |
| `download_sa_data.py` | SA NLCD raster | SA |
| `download_et_sa.py` | SA ET₀ raster from CGIAR Global-AI/ET0 v3.1 | SA |
| `download_census_pop_sa.py` | SA population from Census 2020 blocks | SA |
| `download_osm_sa.py` | SA buildings + roads from OSM | SA |
| `download_ssurgo_sa.py` | SA soil polygons | SA |
| `process_ssurgo_sa.py` | Rasterized soil for SA | SA |
| `process_tracts_sa.py` | Tracts for SA (all 375 Bexar) | SA |
| `precompute_scenarios.py` | Per-city dense scenario lookup CSV (instant slider response) | All cities |

**Validation / diagnostic scripts** (consume but do not produce inventory items):

`compare_carbon_invest.py`, `compare_ucm_invest.py`, `compare_una_invest.py`, `compare_una_lulc.py`, `validate_scenarios.py`, `validate_surrogate_predictions.py`, `verify_baselines.py`, `verify_cooling.py`, `check_expanded_coverage.py` (specifically: two coverage checks for the MN Full NLCD raster — does it cover the legal Minneapolis city boundary, and does the existing soil raster cover the new LULC extent), `placement_strategy_diagnostic.py`, `analyze_placement_diagnostic.py`.

---

## 15. Open questions

Captured here so they don't get forgotten across sessions.

1. ~~**NLCD vintage of NatCap SA data.**~~ ✅ Resolved 2026-05-24: `nlcd_3857.tif` uses **legacy NLCD 2021** schema (16 unique values from the legacy 21-class set). Aligns with the prototype's continued use of legacy NLCD.

2. **NatCap UNA demand parameter.** What per-capita demand value did NatCap use for SA when computing the `una/` outputs in InVEST Results? **Not resolvable from `data/sa/natcap_2024/InVEST Results/una/`** — only model outputs are preserved; no `args.json` or run-log file is present. The `urban_nature_demand.tif` output may encode the runtime demand at every pixel; reading its single non-zero value would resolve this. Resolution determines whether the current prototype's `UNA_DEMAND_M2_PER_CAPITA = 16.7` aligns with NatCap SA, or needs a per-city override.

3. **NatCap UNA search radius.** Partial answer 2026-05-24: the intermediate file `kernel_800.0.tif` in `InVEST Results/una/intermediate/` confirms NatCap used **800 m** — matching the prototype's current value. Full args.json not present.

4. ~~**NatCap UHI_MAX_C for SA.**~~ ✅ Resolved 2026-05-24 (Brief 14): SA `UHI_MAX_C` migrated from the 3.5 °C estimate to NatCap's canonical 11 °C (per the NatCap SA README). See §10.

5. ~~**`et0_annual_cgiar_3857.tif` resolution/extent.**~~ ✅ Resolved 2026-05-24: 60 × 63 pixels at 1,215 m. **Probably not safe to adopt as-is** — ~40× coarser than the operational SA ET raster. If migrating to the NatCap stack, ET₀ should be re-downloaded from CGIAR at native resolution.

6. ~~**SA buildings — typed?**~~ ✅ Resolved: SA's OSM building strings (`house`, `apartments`, `retail`, …) are mapped to InVEST type codes 1/2/3 via `_OSM_BUILDING_TO_INVEST_TYPE` in app.py (~29 % pixel coverage; untyped polygons left at 0 and excluded from per-type lookups). Lights up the SA Cooling Energy Savings card as a conservative lower bound. Flood Damage path resolved separately by Brief 33 (Path C — % volume reduction, no monetization).

7. **NLUD provenance.** What is NLUD's source? USGS? Vintage? Coverage? Likely documented in `data/sa/natcap_2024/Notes on NASA Urban parameterization QA.docx` — full text not yet extracted.

8. **MN downtown population source.** Both `download_census_pop.py` (Census) and `clip_worldpop.py` (WorldPop) produce `data/population/minneapolis_pop_2020.tif`. Canonical setup per docs is Census, but the on-disk file's provenance can't be determined from the file alone. Running the Census pipeline to re-verify would settle this.

9. **CRS + grid mismatch between NatCap data and current SA stack.** NatCap rasters are EPSG:3857 at 34.5 m; current SA stack is EPSG:5070 at 30 m. Integration requires either reprojecting NatCap → 5070 (loses native NatCap grid) or migrating the whole SA stack → 3857 (loses NLCD's native equal-area projection). Decision required at start of SA migration workstream.

10. **Other Drive-shared folders.** "Shared with me" in the user's Drive includes `Minneapolis/`, `roads/`, `building footprints/`, `Urban model sample data same AOI Minneapolis/`, `README_San Antonio InVEST model inputs/`, and `Ecosystem Explorer - Meeting Note`. Several may contain useful data not yet inventoried. Triage as part of SA integration workstream.

---

## Maintenance

This doc should be updated whenever:

- A new data source is added (new download script, new raster, new CSV in `data/`)
- A data source migrates (e.g., switching from NLCD-only SA setup to NatCap compound lucode setup)
- A status changes (Active → Dormant, Pending → Active, etc.)
- A new open question is identified or an existing one is resolved

Same discipline as `WHATS_NEW`, `NATCAP_ALIGNMENT.md`, `DESIGN_NOTES.md` updates.
