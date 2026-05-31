# Data Inventory

**Audience:** Internal
**Status:** Current
**Use this for:** Every data file the app consumes — per-city, per-category, with provenance and status
**Do not use this for:** Per-city parameter parity claims (→ `CITY_PARITY.md`), how metrics are computed (→ `../../REFERENCE.md`), model/metric alignment status (→ `NATCAP_ALIGNMENT.md`), per-decision rationale (→ `DESIGN_NOTES.md`), or live blockers (→ `OPEN_QUESTIONS.md`)
**Source of truth for:** What data files exist, where they come from, and what reads them

---

A catalog of every external data file the prototype consumes. Organized by category (LULC / soil / buildings / roads / population / tracts / ET / biophysical / climate / pre-computed). Every row carries a controlled **Status** so the catalog answers "is this file active, missing, derived, or retired" without prose.

**Status vocabulary (5 values):**

- **`Committed`** — file lives in git; loaded at runtime as a primary input.
- **`Local-only`** — file exists on disk but is gitignored (size, license, or external-provenance reasons). Required for SA runtime; recreate via the NatCap drive pull or the named pipeline.
- **`Pending`** — data we expect to integrate but don't yet have; see Notes for the source and the blocking link (typically `OPEN_QUESTIONS.md` or `NATCAP_COLLABORATION.md` §6).
- **`Derived`** — produced by a script from other inputs; rebuild via the named command. The script is the source of truth; the file is a cache.
- **`Retired`** — preserved on disk for audit/reference but no longer wired into the runtime pipeline.

Per-city parameter parity claims (MD5-match assertions, "byte-identical to InVEST sample," etc.) live in `CITY_PARITY.md`. This catalog records the file's path + source + status; CITY_PARITY adds the parity assertion.

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
├── sa/                          # San Antonio (live + NatCap-curated)
│   ├── README.md
│   ├── buildings_sa.gpkg
│   ├── roads_sa.geojson
│   ├── tracts_bexar.{shp,dbf,prj,shx,cpg}       (current SA tracts at this top level)
│   ├── tracts_bexar/            (raw TIGER 2020 Texas tract shapefile)
│   ├── cooling/                 (SA cooling — biophysical table + ET + raw CGIAR archive)
│   ├── flood/                   (SA flood — LULC + soil + CN table + raw SSURGO)
│   ├── natcap_2024/             (NatCap-curated SA dataset received 2026-05-23 — see §2/§9)
│   └── population/sa_pop_2020.tif
├── scenarios_dense_mpls.csv     # Per-city dense lookup tables
├── scenarios_dense_mpls_full.csv
└── scenarios_dense_sa.csv
```

`du -sh data/*/`: cooling 296 K · flood 320 K · invest 3.2 M · minneapolis_expanded 115 M · osm **7.2 G** · population 971 M · sa **1.3 G**. `du -sh data/sa/*/`: cooling 835 M · flood 27 M · natcap_2024 321 M · population 396 K · tracts_bexar 51 M.

---

## 2. Land cover and land use (LULC)

The prototype uses **NLCD 2021** (legacy MRLC product) across all currently active cities. NLCD 2021 is also the LULC vintage shipped with the InVEST UFR, UCM, and UNA sample data. NatCap's August 2024 NLCD layer (`data/sa/natcap_2024/nlcd_3857.tif`) is also legacy NLCD 2021 (16 unique values from the 21-class set) — confirmed 2026-05-24.

### Minneapolis (downtown) — two rasters

| Role | Path | CRS | Dimensions | Status | Source / Notes |
|---|---|---|---|---|---|
| Cooling & scenario LULC | `data/cooling/land_use_2021.tif` | EPSG:26915 (UTM 15N) | 356 × 360, int16 | Committed | InVEST UNA sample bundle. Parity assertion in `CITY_PARITY.md` MN UNA. |
| Flood / CN LULC | `data/flood/LULC_NLCD_2021_MN.tif` | EPSG:26915 | 356 × 360, int16 | Committed | InVEST UFR sample bundle. Same AOI + grid as the cooling LULC but distinct file. |

Minneapolis downtown is the only city where flood and cooling use *different* LULC rasters — an artifact of inheriting from separate InVEST sample bundles. The originals are preserved under `data/invest/flood/UFR_sample_data_MN/` and `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/`.

### Minneapolis Full (dormant) — single raster

| Role | Path | CRS | Dimensions | Status | Source / Pipeline |
|---|---|---|---|---|---|
| All LULC | `data/minneapolis_expanded/lulc_nlcd_2021_mpls_full.tif` | EPSG:5070 (CONUS Albers) | 607 × 374, uint8 | Committed | NLCD 2021 via MRLC WCS; `download_minneapolis_nlcd.py`. `available=False` in city config. |

### San Antonio — live LULC rasters (dual-raster pipeline)

| Role | Path | CRS | Dimensions | Status | Source / Notes |
|---|---|---|---|---|---|
| Compound LULC (live canonical, compound-keyed for UCM/UNA/Carbon) | `data/sa/flood/land_use_compound_sa.tif` | EPSG:5070 | 1713 × 1984 | Derived | NatCap `lulc_overlay_3857.tif` reprojected EPSG:3857 → EPSG:5070 with nearest-neighbor at 30 m. 800 unique compound lucodes of 1,984 possible. |
| NLCD-only LULC (flood path; spatial-map render) | `data/sa/flood/land_use_2021_sa.tif` | EPSG:5070 | 1713 × 1984, uint8 | Committed | NLCD 2021 via MRLC WCS; `download_sa_data.py`. The live SA flood CN table is NLCD × tree-canopy keyed and the compound raster reduces via `reduce_compound_to_nlcd_tree` (`COMPOUND_TO_NLCD_TREE`, built in `load_lulc_crosswalk`). |
| Raw NLCD clip (audit trail) | `data/sa/flood/lulc_nlcd_2021_sa.tif` | EPSG:5070 | 1713 × 1984 | Retired | Raw NLCD 2021 download via MRLC WCS, preserved for source-provenance. Not read at runtime — the live SA pipeline consumes the two rasters above. |

**CRS choice — settled.** The SA stack runs EPSG:5070 (NLCD's native equal-area projection) because the prototype's metrics are area-based. NatCap rasters arrive in EPSG:3857 (Web Mercator) and are reprojected at preparation time, not at runtime. Documented per-decision rationale in `DESIGN_NOTES.md` §3.3.

### San Antonio — NatCap-curated source rasters (raw delivery, EPSG:3857)

All four rasters share the same 2106 × 2218 grid at 34.5 m pixel size, EPSG:3857. Extent: 98.84°W–98.19°W, 29.16°N–29.76°N — roughly the Bexar County urban core. Gitignored (`data/sa/natcap_2024/*.tif`).

| Role | Path | Status | Source / Notes |
|---|---|---|---|
| Compound NLCD+NLUD+tree overlay (raw) | `data/sa/natcap_2024/lulc_overlay_3857.tif` | Local-only | NatCap NASA Urban project, Aug 2024 vintage. Reprojected to the live `land_use_compound_sa.tif` above. |
| Source NLCD layer | `data/sa/natcap_2024/nlcd_3857.tif` | Local-only | NLCD 2021 reprojected by NatCap. Legacy 21-class schema confirmed. |
| Source NLUD layer | `data/sa/natcap_2024/nlud_3857.tif` | Local-only | National Land Use Database, USGS. Full provenance pending — likely documented in `Notes on NASA Urban parameterization QA.docx`. |
| Source tree canopy layer | `data/sa/natcap_2024/tree_3857.tif` | Local-only | NLCD 2021 tree canopy product. |

The compound `lulc_overlay_3857.tif` encodes a **Cartesian product lucode space**: **16** distinct NLCD codes × **31** distinct NLUD simple codes × **4** tree-canopy bins (`tree_canopy_cover` ∈ {`none`, `low`, `medium`, `high`}; `tree` codes 0–3) = **1,984 distinct lucodes**, fully exhaustive (not curated). The biophysical tables (UCM / UNA / Carbon) are keyed on this compound lucode — see §9. Per-decision rationale for the compound-LULC adoption: `DESIGN_NOTES.md` §3.3.

The compound `code` column (4–6-digit IDs) does **not** follow a clean positional encoding — the obvious hypothesis `nlcd*100 + nlud*10 + tree` matches only 8 of 1,984 rows. The serial **`lucode`** column (0–1983) is the actual join key the biophysical tables use; the `code` encoding scheme is undocumented in the data and would need NatCap clarification if it matters for integration.

### Prototype vs NatCap LULC raster — side-by-side

The live `land_use_compound_sa.tif` (Derived row above) is produced by reprojecting NatCap's `lulc_overlay_3857.tif` onto the prototype's existing SA grid. For reference / audit, the side-by-side dimensions of the prototype's NLCD-only SA raster and NatCap's compound source:

| Property | Prototype NLCD-only (`land_use_2021_sa.tif`) | NatCap compound (`lulc_overlay_3857.tif`) |
|---|---|---|
| CRS | EPSG:5070 (NAD83 / Conus Albers) | EPSG:3857 (Web Mercator) |
| Dimensions | 1984 × 1713 | 2106 × 2218 |
| Resolution | 30 m | 34.5 m |
| Lucode dtype | `uint8` (NLCD only) | `int16` (compound codes) |
| Lucode range | 11–95 (15 unique codes) | 0–1913 (820 unique of 1,984 possible; ~41 %) |
| NoData | 0 | −1 |
| Extent (lat/lon) | 98°48′54″W–98°11′17″W, 29°12′–29°38′58″N | 98°50′48″W–98°11′38″W, 29°9′32″–29°45′27″N |
| File size | 4.2 MB | 9.4 MB |

**Extent difference.** NatCap extends ~6 minutes farther north, ~3 farther south, ~2 farther west; same east edge. Both centered on San Antonio. The reprojection that produces `land_use_compound_sa.tif` clips to the prototype's existing 1984 × 1713 EPSG:5070 grid; ~1 % nodata coverage on the reprojected output reflects this clip. Methodology implications of the compound adoption (per-pixel ag/maintenance signals, Köppen-BSh tuning retirement, etc.) live in `DESIGN_NOTES.md` §3.3 + §3.4.

### Committed files in `data/sa/natcap_2024/`

The small CSVs + doc files that live alongside the gitignored rasters.

| File | Status | Content |
|---|---|---|
| `README.docx` + `README.txt` | Committed | NatCap's original SA dataset README. |
| `Notes_on_NASA_Urban_parameterization_QA.docx` + `.txt` | Committed | NASA Urban project parameterization QA notes. Documents the canopy-weighted parameter framework (paras 123–138). |
| `README_San_Antonio_InVEST_model_inputs.docx` + `.txt` | Committed | NatCap's per-InVEST-model SA input recipe — args.json-equivalent values for UCM / Carbon / UNA / UFR / NDR. |
| `Ecosystem_Explorer_-_Meeting_Note.docx` + `.txt` | Committed | NatCap meeting note: Symposium 2026 dates, six-model SA scope. |
| `ucm__nlcd_nlud_tree.csv` | Committed | NatCap SA Urban Cooling biophysical table — 1,984-row compound NLCD × NLUD × tree-canopy lookup. Live as SA's UCM biophysical table. |
| `una__nlcd_nlud_tree.csv` | Committed | NatCap SA Urban Nature Access — 1,984-row categorical `urban_nature` score per compound lucode. Live as SA's UNA biophysical table. |
| `carbon__nlcd_nlud_tree.csv` | Committed | NatCap SA Carbon — 1,984-row four-pool (c_above / c_below / c_soil / c_dead). Live as SA's Carbon biophysical table. |
| `lulc_crosswalk.csv` | Committed | NLCD × NLUD × tree-canopy → compound-lucode lookup; the join key for the three biophysical tables above. Also carries per-row realism flags — **`is_realistic_to_create`** (populated; gates which compound classes a scenario can meaningfully produce — load-bearing for SA conversion + future region-selection placement work) and **`is_realistic_to_paint`** (all NaN in current data; reserved column). Conversion-target lookup logic in `DESIGN_NOTES.md` §4.1. |
| `acs_block_group_equity_data.csv` | Committed | Census ACS demographic + equity data joined to SA block groups. |
| `acs_block_groups_3857.gpkg` | Committed | SA Census block group polygons (EPSG:3857; reprojected to EPSG:5070 at load time). 1,124 polygons covering the City of San Antonio. Live as SA's `tracts_file`. |
| `classification_structure_qaqc.xlsx` | Committed | NatCap's methodology QA/QC for the compound LULC framework. Binary file. |

`.docx` files preserved alongside `textutil`-converted `.txt` versions so the contents are grep-able.

---

## 3. Soil hydrologic groups (SSURGO)

USDA Soil Survey Geographic Database, rasterized to match the NLCD grid. Soil hydrologic group (A/B/C/D) is a key input to the SCS Curve Number equation for flood modeling.

| City | Path | Status | Source / Pipeline |
|---|---|---|---|
| MN downtown | `data/flood/soil_group_MN.tif` | Committed | InVEST UFR sample shapefile, pre-rasterized. ~71.1 km². |
| MN Full | `data/minneapolis_expanded/soil_group_mpls_full.tif` | Committed | USDA Soil Data Access REST API; `download_ssurgo.py` → `process_ssurgo.py`. Full Hennepin County (~1,572 km², 32,442 polygons). 9 % of polygons reassigned to C-class per NRCS convention. |
| MN Full raw inputs | `data/minneapolis_expanded/ssurgo_hennepin_hsg.{shp,...}` + `ssurgo_hydgrp_hennepin.csv` | Committed | Raw SSURGO polygons + attributes; preserved alongside the rasterized output. |
| SA | `data/sa/flood/soil_group_sa.tif` | Committed | USDA SSURGO API; `download_ssurgo_sa.py` → `process_ssurgo_sa.py`. |
| SA raw inputs | `data/sa/flood/ssurgo_bexar_hsg.{shp,...}` + `ssurgo_hydgrp_bexar.csv` | Local-only | Raw SSURGO polygons + attributes; gitignored. |

---

## 4. Buildings

Two sources per city, used for different purposes. The InVEST sample buildings have per-building **type codes** (commercial / residential / industrial / other) that drive the *Cooling Energy Savings* and *Flood Damage Avoided* dollar metrics. The OSM buildings are **untyped** and used only for the placement non-convertible mask.

The split-config rationale (placement-constraint inputs vs model-input data) lives in `CITY_PARITY.md` MN UFR section.

| City / Role | Path | Status | Count / Notes |
|---|---|---|---|
| MN downtown — typed UFR sample | `data/invest/flood/UFR_sample_data_MN/buildings.shp` | Committed | 3,788 polygons. Type codes: 0=other, 1=commercial, 2=residential, 3=industrial. |
| MN city-wide — untyped OSM | `data/osm/minneapolis_buildings.geojson` | Committed | 185,490 polygons; `download_osm_minneapolis.py`. |
| MN city-wide — raw zip | `data/osm/minnesota.shp.zip` | Local-only | Geofabrik Minnesota state extract; gitignored due to size. |
| MN Full | `data/minneapolis_expanded/buildings_mpls_full.gpkg` | Committed | Geofabrik OSM, filtered to Hennepin County. |
| SA | `data/sa/buildings_sa.gpkg` | Committed | OSM via Geofabrik Texas; `download_osm_sa.py`. Per-building strings (`house`, `apartments`, …) mapped to InVEST type codes 1/2/3 via `_OSM_BUILDING_TO_INVEST_TYPE` (~29 % pixel coverage). Lights up SA Cooling Energy Savings as a conservative lower bound. |
| SA raw zip | `data/osm/texas.shp.zip` | Local-only | Geofabrik Texas state extract; gitignored. |
| NatCap-provided buildings | (Drive subfolder, not downloaded) | Pending | NatCap may have curated typed buildings for SA. Triaged 2026-05-24: the visible `Minneapolis/building footprints/` folder turned out to be the same Geofabrik OSM extract, no new data. SA-curated buildings status: unverified. |

---

## 5. Roads

OpenStreetMap road network, used to mask roads from conversion-eligible pixels. Filtered using **Option B**: drop sub-pixel-width surfaces (footway, cycleway, steps, service, path, pedestrian, unclassified, track*). Retained: motorway, trunk, primary, secondary, tertiary, residential, living-street, on/off-ramp links.

| City | Path | Status | Count / Pipeline |
|---|---|---|---|
| MN downtown + city-wide | `data/osm/minneapolis_roads.geojson` | Committed | 5,495 segments; ~29 % AOI coverage. `download_osm_minneapolis.py`. |
| MN Full | `data/minneapolis_expanded/roads_mpls_full.geojson` | Committed | 10,984 segments; `process_osm_expanded.py`. |
| SA | `data/sa/roads_sa.geojson` | Committed | `download_osm_sa.py`. |
| NatCap-provided roads | (Drive subfolder, not downloaded) | Pending | The Drive's `roads/` subfolder contains the same Geofabrik OSM extract — no new data per 2026-05-24 inspection. |

All road rasters are unioned into `BUILDINGS_RASTER` at startup so conversions can't land on streets. See `../../CLAUDE.md` "OSM road exclusion".

---

## 6. Population

Used for the InVEST UNA per-capita supply calculation, neighborhood reporting, and (legacy) the equity-focused placement strategy — renamed to `undersupply-focused`.

| City | Path | Status | Source / Pipeline / Notes |
|---|---|---|---|
| MN downtown | `data/population/minneapolis_pop_2020.tif` | Committed | US Census 2020 block-level (canonical). **Both `download_census_pop.py` (Census) and `clip_worldpop.py` (WorldPop, alternative) target the same output path.** On-disk provenance not encoded in the file; if WorldPop was the source, totals would diverge from the Census reference. **Rebuild command (resolves provenance): `python download_census_pop.py`.** |
| MN downtown TIGER cache | `data/population/tiger/` | Committed | TIGER 2020 tabulation-block polygons; joins to the Census table via `GEOID20`. |
| MN Full | `data/minneapolis_expanded/pop_mpls_full.tif` | Committed | Census 2020 blocks, Hennepin County; `process_pop_expanded.py`. |
| SA | `data/sa/population/sa_pop_2020.tif` | Committed | US Census 2020 blocks, Bexar County FIPS 48029; `download_census_pop_sa.py`. |
| SA NatCap-curated | `data/sa/natcap_2024/population_per_pixel_2020_3857.tif` | Local-only | 19 MB, 2106 × 2218, 34.5 m pixel, EPSG:3857. Source attribution not embedded in the file; likely WorldPop or a NatCap-internal downscaling. Same CRS mismatch as the LULC. |

---

## 7. Census tracts and demographics

For per-tract neighborhood-improvement reporting overlay on the Map View tab.

| City | Path | Status | Count / Notes |
|---|---|---|---|
| MN downtown | `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/AOI_admin_boundaries_census_tracts.shp` | Committed | TIGER 2020; 27 tracts intersecting the InVEST sample AOI. |
| MN Full | `data/minneapolis_expanded/tracts_hennepin.shp` | Committed | TIGER 2020; all 329 Hennepin County tracts. `process_tracts_expanded.py`. |
| SA | `data/sa/tracts_bexar.shp` (+ raw `tracts_bexar/tl_2020_48_tract.shp`) | Committed | TIGER 2020; all 375 Bexar County tracts. `process_tracts_sa.py`. |
| SA NatCap ACS block groups (live as SA's `tracts_file`) | `data/sa/natcap_2024/acs_block_groups_3857.gpkg` | Committed | EPSG:3857; reprojected to EPSG:5070 at load time. 1,124 polygons covering the City of SA. Consumed by `compute_per_tract_summary`. |
| SA NatCap reference outputs | `data/sa/natcap_reference_outputs.csv` | Derived | NatCap's published SA citywide scenario outputs extracted from `nootenboom_results/citywide_results_UPDATED.xlsx`. 49 rows (7 prototype metrics × 7 scenarios), 3 validation states. Built by `extract_natcap_reference_outputs.py` (re-runnable). Read by `natcap_validation.py`. |

ACS equity CSV (`data/sa/natcap_2024/acs_block_group_equity_data.csv`) includes bivariate color-scheme fields for plotting; not yet wired into the dashboard.

---

## 8. Reference evapotranspiration (ET₀)

Annual ET₀ raster, used in the UCM Kc × ETI calculation.

| City | Path | Status | Resolution / Notes |
|---|---|---|---|
| MN (both) | `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif` | Committed | 1 km native, ~10 × 10 km native extent; bilinear-resampled to the NLCD grid. For MN Full, the raster extrapolates beyond its native extent at the bbox corners. Nodata sentinel (65535) masked before resize. |
| SA | `data/sa/cooling/et_annual_sa.tif` | Derived | CGIAR Global-AI/ET0 v3.1, reprojected EPSG:5070; 30 arc-seconds (~1 km), 1,580–1,716 mm/yr (mean 1,657) for the SA bbox. **Rebuild command:** `python download_et_sa.py`. |
| SA — raw CGIAR archive | `data/sa/cooling/cgiar_et0/Global-AI_ET0__annual_v3_1.zip` + `et0_v31_yr.tif` | Local-only | 645 MB zip; gitignored. Audit trail for the `et_annual_sa.tif` derivation. |
| SA NatCap-curated ET | `data/sa/natcap_2024/et0_annual_cgiar_3857.tif` | Retired | 60 × 63 pixels at 1,215 m — ~40× coarser than the operational SA raster. **Not safe to adopt as-is**; if migrating to the NatCap stack, ET₀ should be re-downloaded from CGIAR at native resolution. |

SA's PET is ~50 % higher than MN's, but enters the CC formula via normalized ETI so absolute mm/yr cancels. See `../../REFERENCE.md` "Cross-city Heat Mitigation Index comparison".

---

## 9. Biophysical parameter tables

The lookup tables that translate LULC codes into per-pixel model parameters. Per-city parameter values for each model live in `CITY_PARITY.md` per-city sections; this catalog records the files + status.

### 9.1 Curve Number (UFR / flood)

Per-city tables, declared via `CITIES[city]['cn_table_file']`.

| City | Path | Status | Lookup key / Notes |
|---|---|---|---|
| MN | `data/flood/UFR_biophysical_table_MN.csv` | Committed | NLCD lucode. InVEST UFR sample. |
| SA (live) | `data/sa/flood/biophys_floodmitig_sa.csv` | Committed | NLCD × tree-canopy 3-tier compound key (53 rows). Design-storm-saturation framework — see NATCAP_COLLABORATION Q12. |
| SA (superseded) | `data/sa/flood/UFR_biophysical_table_SA.csv` | Retired | Prior MN-placeholder CN table; kept on disk for reference. Read once by `download_sa_data.py` as a one-time QA diagnostic. |

NatCap's compound biophysical bundle does **not** ship a Curve Number table; SA flood CN values come from NatCap's separate flood-mitig table.

### 9.2 Urban Cooling biophysical (shade / Kc / albedo)

| City | Path | Status | Lookup key / Notes |
|---|---|---|---|
| MN (both) | `data/cooling/biophysical_table_urban_cooling_MN.csv` | Committed | NLCD lucode. From InVEST UCM args JSON. |
| SA (live) | `data/sa/natcap_2024/ucm__nlcd_nlud_tree.csv` | Committed | Compound lucode (NLCD × NLUD × tree-canopy). 1,984 rows × 27 cols. Tree canopy is a dominant signal (high-canopy pixels get shade ≈ 0.66 regardless of NLCD). |
| SA (retired) | `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` | Retired | Köppen BSh-tuned NLCD-keyed table. Kept on disk for reference; per-class rationale in `data/sa/cooling/biophysical_table_sources.md`. |

### 9.3 Urban Nature Access biophysical

| City | Path | Status | Lookup key / Notes |
|---|---|---|---|
| MN (both) | `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv` | Committed | NLCD lucode. Per-class `urban_nature` score + per-class `search_radius_m`. |
| SA | `data/sa/natcap_2024/una__nlcd_nlud_tree.csv` | Committed | Compound lucode. 1,984 rows × 21 cols. Per-pixel `urban_nature` ∈ {0.0, 0.5, 1.0} (distribution: 976 / 48 / 960). The 0.5 score appears only for Conservation-class NLUD pixels. |

Per-city `urban_nature_demand_per_capita`, `search_radius_m`, and `decay_function` values live in `CITY_PARITY.md` under each city's `### UNA` table. The SA NatCap table's `search_radius_m` column is all zeros — the radius is an args-level scalar, not a per-row table value. Partial confirmation 2026-05-24: NatCap's intermediate `kernel_800.0.tif` confirms SA used 800 m. SA UNA demand parameter: NatCap's `urban_nature_demand.tif` output may encode the runtime value; reading its single non-zero value would resolve it (full args.json not present).

### 9.4 Carbon biophysical

| City | Path | Status | Lookup key / Methodology |
|---|---|---|---|
| MN (both) | per-cover-class rates embedded in `app.py` / `config.py` | Committed | NLCD lucode → single sequestration rate (tons CO₂e/ha/yr). Annual flow methodology. User-overridable via Advanced Settings sliders. |
| SA | `data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv` | Committed | Compound lucode. 1,984 rows × 27 cols. Four pools `c_above` (max 105.7), `c_below` (max 8.0), `c_soil` (max 259.0 — dominant pool), `c_dead` (max 14.4) — all in tons C/ha. One-time stock-change methodology (Vibrant Land precedent). Three additional columns (`c_embedded_storage`, `c_embedded_emissions`, `c_annual_emissions`) describe urban-accounting flows the prototype doesn't use. |

The unified return-dict key `carbon_tons_co2` carries either framing; the city-conditional `_CARBON_IS_STOCK` flag drives dashboard card labels and unit suffixes. Per-decision rationale: `DESIGN_NOTES.md` §6.4.

### 9.5 Food Forest yield

`FOOD_FOREST_LBS_ACRE` is a per-city scalar in `config.py` (MN 11,500; SA 8,500 placeholder). Single-value parameter, not a table. SA value pending NatCap per-crop CoSA numbers — see NATCAP_COLLABORATION ask 4a.

---

## 10. Climate / urban-heat-island parameters

Per-city scalars in `config.py`; current per-city values + parity status in `CITY_PARITY.md` per-city UCM / UFR tables. This section records the runtime mechanism, not the values.

| Variable | Per-city in | Notes |
|---|---|---|
| `UHI_MAX_C` | `config.py`'s `CITIES[city]['uhi_max_c']` | Read at module load. `HM_TO_FAHRENHEIT = UHI_MAX_C × 1.8` derives per city. |
| `DESIGN_STORM_INCHES` | `config.py`'s `CITIES[city]['design_storm_inches']` | Read at module load. Derived `DESIGN_STORM_MM = DESIGN_STORM_INCHES × 25.4` used in tooltip display. |

Other UCM constants (`d_cool = 450 m`, `r = 600 m` blending) are model-architecture defaults, not city-specific.

---

## 11. Pre-computed validation outputs

### From the NatCap-curated SA dataset (received 2026-05-23)

`data/sa/natcap_2024/InVEST Results/` — canonical InVEST model outputs run by NatCap on their curated inputs. Locally available, gitignored.

| Subfolder | Status | Key files / Use |
|---|---|---|
| `ucm/` | Local-only | `hm.tif` (14 MB), `uhi_results.{shp,...}`, intermediate kernels (`cc.tif`, `cc_park.tif`, `T_air.tif`, `T_air_nomix.tif`, `green_area.tif`, `green_area_sum.tif`, `eti.tif`, `kc.tif`, `albedo.tif`, `shade.tif`, `lulc.tif`, `ref_eto.tif`). No `args.json`. Ground truth for UCM validation. |
| `carbon/` | Local-only | `tot_c_cur.tif`, `report.html`, intermediate outputs. No `args.json`. Ground truth for Carbon validation. |
| `una/` | Local-only | `output/urban_nature_supply_percapita.tif`, `urban_nature_balance_percapita.tif`, `urban_nature_balance_totalpop.tif`, `urban_nature_demand.tif`, `accessible_urban_nature.tif`. `intermediate/kernel_800.0.tif` (confirms NatCap's 800 m search radius). No `args.json`. Ground truth for UNA validation. |

Each model's `taskgraph_cache/` contains only `taskgraph_data.db` (a SQLite cache InVEST uses internally for incremental re-runs); does NOT contain a record of the user-facing run args.

### From the prototype itself

| Path | Status | Generator |
|---|---|---|
| `analysis/placement_diagnostic/layer{1,2,3}_*.csv` | Derived | `placement_strategy_diagnostic.py`. Placement-strategy diagnostic measurements. |
| `tests/baselines/*.json` | Derived | `verify_baselines.py --update`. 40 baseline scenario snapshots (2 cities × 4 scenarios × 5 strategies). Current `SCENARIO_SCHEMA_VERSION = 27`. |
| `comparisons/` (7 files) | Derived | `compare_*_invest.py`, `validate_surrogate_predictions.py`. One-shot validation snapshots committed for reference. |
| `data/scenarios_dense_{mpls,mpls_full,sa}.csv` | Derived | `precompute_scenarios.py`. Per-city dense scenario lookup tables for instant slider response. |

### Auxiliary

| Path | Status | Notes |
|---|---|---|
| `cache/` | Committed | Empty directory at repo root. Runtime scratch location. |
| `__pycache__/` | Local-only | Python bytecode; not data. |

The earlier `data/precomputed/` distance-transform cache (consumed by the homegrown reachability proxy that was retired) was cleaned up 2026-05-24.

---

## 12. Status of data integrations (snapshot)

The per-row Status column on every catalog table above carries the per-file integration status. This section is the compact snapshot — one line per integration, with a pointer to its detailed home.

| Integration | Status | Detail |
|---|---|---|
| Minneapolis downtown (cooling, flood, UNA, buildings, soil, population, tracts) | ✅ Active | §2–§9 + `CITY_PARITY.md` MN section |
| Minneapolis Full | ⏸️ Dormant (`available=False`) | §2–§9 MN Full rows; `HISTORY.md` "Full Minneapolis extent" |
| San Antonio NatCap-curated stack (compound LULC + UCM/UNA/Carbon biophysical + ACS block groups) | ✅ Adopted across LULC raster, UCM, UNA, Carbon, and ACS block groups | §2 + §7 + §9; `CITY_PARITY.md` SA section; `NATCAP_ALIGNMENT.md` §3 |
| SA NDR (Nutrient Delivery Ratio) | ⏸️ Pending DEM + watersheds | `NATCAP_COLLABORATION.md` ask 5 |
| SA per-building damage values | ✅ Resolved (Path C — "Flood Volume Reduction" reframe) | `DESIGN_NOTES.md` §6.5 |
| MN Carbon four-pool bundle | ⏸️ Pending NatCap data | `NATCAP_COLLABORATION.md` ask 4b |
| Annual NLCD migration | 🔵 Not pursued | `DESIGN_NOTES.md` §3.1 |
| AlphaEarth satellite embeddings | 🔵 Feasibility research only | `docs/research/ALPHAEARTH_FEASIBILITY.md` |

---

## 13. Per-script inventory (data-producing pipelines)

For each download/process script, what does it produce?

| Script | Produces | City |
|---|---|---|
| `scripts/data/download_minneapolis_nlcd.py` | MN NLCD raster (downtown) | MN downtown |
| `scripts/data/download_census_pop.py` | MN population raster from Census 2020 blocks → `data/population/minneapolis_pop_2020.tif` | MN downtown |
| `scripts/data/clip_worldpop.py` | **Alternative** MN population pipeline from USA WorldPop → same output path. Only one source can be active on disk at a time. | MN downtown |
| `scripts/data/download_osm_minneapolis.py` | MN buildings + roads from Geofabrik OSM | MN downtown |
| `scripts/data/download_ssurgo.py` | Soil polygons via USDA SDA REST API | MN (downtown + expanded) |
| `scripts/data/process_ssurgo.py` | Rasterized soil hydrologic groups for MN Full | MN Full |
| `scripts/data/process_pop_expanded.py` | Population raster for MN Full | MN Full |
| `scripts/data/process_tracts_expanded.py` | Tracts for MN Full (all 329 Hennepin) | MN Full |
| `scripts/data/process_osm_expanded.py` | Roads for MN Full | MN Full |
| `scripts/data/download_sa_data.py` | SA NLCD raster | SA |
| `scripts/data/download_et_sa.py` | SA ET₀ raster from CGIAR Global-AI/ET0 v3.1 | SA |
| `scripts/data/download_census_pop_sa.py` | SA population from Census 2020 blocks | SA |
| `scripts/data/download_osm_sa.py` | SA buildings + roads from OSM | SA |
| `scripts/data/download_ssurgo_sa.py` | SA soil polygons | SA |
| `scripts/data/process_ssurgo_sa.py` | Rasterized soil for SA | SA |
| `scripts/data/process_tracts_sa.py` | Tracts for SA (all 375 Bexar) | SA |
| `scripts/data/extract_natcap_reference_outputs.py` | `data/sa/natcap_reference_outputs.csv` (NatCap's published citywide scenario outputs) | SA |
| `precompute_scenarios.py` | Per-city dense scenario lookup CSV | All cities |

**Validation / diagnostic scripts** (consume but don't produce inventory items): `validation/compare_carbon_invest.py`, `validation/compare_ucm_invest.py`, `validation/compare_una_invest.py`, `validation/compare_umh_invest.py`, `validation/verify_cooling.py`; `diagnostics/compare_una_lulc.py`, `analyze_placement_diagnostic.py`, `placement_strategy_diagnostic.py`, `check_expanded_coverage.py`, `validate_surrogate_predictions.py`; `verify_baselines.py`. Environment + two-env validation harness in `docs/dev/CONTRIBUTING.md`.

---

## Maintenance

Update when:

- A new data source is added (new download script, new raster, new CSV in `data/`) — add a row with the appropriate Status.
- A data source migrates (e.g., NLCD-only SA → NatCap compound) — update the existing row's Status and add a `Retired` row for the prior file if it's preserved on disk.
- A Status changes (`Pending` → `Committed`, `Committed` → `Retired`, etc.).

Same discipline as `WHATS_NEW`. Pair with `CITY_PARITY.md` and `NATCAP_ALIGNMENT.md` updates when the same finding affects all three docs.
