# City Parity

**Purpose:** Track how closely the prototype's per-city parameters and methodology match NatCap's published configurations for each city. The unit of comparison is *the city*, not the methodology — answering "how aligned is the prototype with NatCap on Minneapolis?" rather than "is the prototype's UCM canonical?"

**Audience:** Daniel and future Claude sessions. Not shared with NatCap.

**Naming:** Refer to NatCap collaborators as "NatCap" — no individual names.

**Relationship to other docs:**

- `NATCAP_ALIGNMENT.md` — alignment by *methodology* (six tables: per-metric, per-vocabulary). Methodology fidelity.
- `NATCAP_COLLABORATION.md` — running conversation log (asks, decisions, open questions).
- This doc — alignment by *city*. Concrete per-city, per-parameter parity matrix.

**Working principle (post 2026-05-24):** NatCap parameters are project-specific by design. Each city's project is tuned to its own policy framing. The right form of alignment is *per-city* — match MN-side parameters to NatCap's MN project; match SA-side parameters to NatCap's SA project. See `NATCAP_COLLABORATION.md` § "Per-city parameter framing."

**Status legend:**

- ✅ **Aligned** — prototype value matches NatCap project value
- ⚠️ **Diverges** — prototype value differs; documented or expected
- ❌ **Not implemented** — model or metric exists in NatCap setup but not in prototype
- ❓ **Unknown** — NatCap value not yet observed; prototype value is independent or unverified

---

## Minneapolis (downtown)

**Active in production.** Uses InVEST UFR + UCM + UNA sample data heritage. NatCap reference: `data/invest/mn_sample_data_natcap_2026/` (three InVEST sample bundles, args.json files inside, received 2026-05-24).

### UCM (Urban Cooling)

| Parameter | Prototype | NatCap MN project | Status |
|---|---|---|---|
| `uhi_max` | 2.05 °C | 2.05 °C | ✅ |
| `t_ref` | (no analog — pure-delta) | 23.2 °C | ✅ (informational only) |
| `cc_method` | `factors` | `factors` | ✅ |
| `green_area_cooling_distance` | 450 m | 450 m | ✅ |
| `t_air_average_radius` | 600 m | 600 m | ✅ |
| `cc_weight_shade` | 0.6 (InVEST default) | "" (uses InVEST default 0.6) | ✅ |
| `cc_weight_albedo` | 0.2 (InVEST default) | "" (uses InVEST default 0.2) | ✅ |
| `cc_weight_eti` | 0.2 (InVEST default) | "" (uses InVEST default 0.2) | ✅ |
| `do_energy_valuation` | True | True | ✅ |
| `do_productivity_valuation` | False | False | ✅ |
| Biophysical table | `data/cooling/biophysical_table_urban_cooling_MN.csv` | `biophysical_table_urban_cooling.csv` (in MN bundle) | ✅ Verified identical 2026-05-24 on all 14 shared lucodes (shade, kc, albedo, green_area, building_intensity). Prototype has one extra row, lucode 82 (Cultivated Crops, documented SA-support addition). |
| LULC raster | `data/cooling/land_use_2021.tif` (byte-identical to NatCap UNA sample) | `land_use_2021.tif` (in MN bundle) | ✅ (likely; MD5-confirmed against UNA sample bundle in `UNA_LULC_INVESTIGATION.md`) |
| ET raster | `data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif` (~1 km, 10×10 km extent) | `reference_evapotranspiration_annual.tif` (in MN bundle) | ✅ MD5 verified identical 2026-05-24 (`fdf460d9bf5ef5a3641f74af117cbd66`, 4,712 bytes both). Same file from same InVEST UCM sample bundle. |

**UCM summary:** Fully aligned on args, biophysical table, and ET raster (all MD5-verified 2026-05-24).

### UFR (Urban Flood Risk Mitigation)

| Parameter | Prototype | NatCap MN project | Status |
|---|---|---|---|
| `rainfall_depth` | **100 mm (3.94", NatCap MN canonical, Brief 23)** | **100 mm** | ✅ Aligned 2026-05-24 (Brief 23). |
| LULC raster | `data/flood/LULC_NLCD_2021_MN.tif` | `LULC_NLCD_2021_MN.tif` | ✅ (same file, MD5-confirmed) |
| Soil HSG raster | `data/flood/soil_group_MN.tif` | `soil_group_MN.tif` | ✅ |
| Buildings | `data/flood/buildings.shp` (UFR sample) | `buildings.shp` | ✅ |
| CN biophysical table | `data/flood/UFR_biophysical_table_MN.csv` | `UFR_biophysical_table_MN.csv` | ✅ Verified identical 2026-05-24 on all 14 shared lucodes (CN_A/CN_B/CN_C/CN_D + NLCD_Land label). Prototype has one extra row, lucode 82 (Cultivated Crops, SA support). |
| Damage loss table | `data/invest/flood/UFR_sample_data_MN/Damage_loss_table_MN.csv` | `Damage_loss_table_MN.csv` (Roads=40, Commercial=120, Residential=150, Industrial=100) | ✅ Verified bit-identical 2026-05-24 — same file from same InVEST UFR sample bundle. |
| AOI watersheds | `data/invest/flood/UFR_sample_data_MN/admin_boundaries_census_tracts.shp` (declared as `tracts_file` in `config.py`) | `admin_boundaries_census_tracts.shp` | ✅ MD5 verified identical 2026-05-24 (`acbd9a8d28892dd4dfaf003b896235b4` on `.shp`; `.dbf` and `.prj` also identical). Same file from same InVEST UFR sample bundle. |

**UFR summary:** Same data sources as NatCap; rainfall depth aligned to MN-project canonical 2026-05-24 (Brief 23). CN biophysical and damage loss tables verified identical.

### UNA (Urban Nature Access)

| Parameter | Prototype | NatCap MN project | Status |
|---|---|---|---|
| `urban_nature_demand_per_capita` | **250 m²/capita** (NatCap MN-project canonical, Brief 22) | **250 m²/capita** | ✅ Aligned 2026-05-24 (Brief 22). |
| `search_radius` | **1000 m** (NatCap MN-project canonical, Brief 22) | **1000 m** | ✅ Aligned 2026-05-24 (Brief 22). |
| `decay_function` | **exponential** (NatCap MN-project canonical, Brief 22) — canonical InVEST `pygeoprocessing.kernels.exponential_decay_kernel` form, `max_distance = ceil(radius_px) * 2 + 1`, `expected_distance = radius_px` | **exponential** | ✅ Aligned 2026-05-24 (Brief 22). |
| `search_radius_mode` | uniform radius | uniform radius | ✅ |
| `aggregate_by_pop_group` | False | False | ✅ |
| `population_group_radii_table` | not used | empty | ✅ |
| LULC raster | `data/cooling/land_use_2021.tif` (UNA-sample LULC) | `LULC_NLCD_2021.tif` (in MN bundle) | ✅ MD5 verified identical 2026-05-24 (`56d1080fa70576cad15896642a107a3d`, 297,417 bytes both). Confirms the prototype's cooling LULC is byte-identical to NatCap's MN UNA sample LULC — consistent with the earlier `UNA_LULC_INVESTIGATION.md` finding. |
| LULC attribute table | `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv` | `LULC_attribute_table_UNA.csv` | ✅ Verified bit-identical 2026-05-24 (14 rows: lucode, lulc_desc, urban_nature, search_radius_m). Same file from same InVEST UNA sample bundle. |
| Population raster | Census 2020 blocks, rasterized via `download_census_pop.py` (360×356 px at 30 m, ~154k people; **deliberate — newer decennial vintage**) | `total_population_census_2010.tif` (270×266 px at 30 m, ~130k people; **NatCap MN sample shipped with 2010 — older bundle**) | ⚠️ Vintage differs (2020 vs 2010). Prototype's choice of 2020 is intentional. 154k vs 130k is consistent with MN downtown population growth from 2010→2020. Worth confirming in future NatCap conversation whether they consider 2010 still current or have updated to 2020 internally. |

**UNA summary:** All three parameters and the LULC attribute table now match MN-project canonical (Brief 22, 2026-05-24). Population raster vintage is the one remaining ⚠️ (2020 prototype vs 2010 NatCap sample bundle).

### Carbon

| Parameter | Prototype | NatCap MN project | Status |
|---|---|---|---|
| Methodology | Single per-NLCD-class rate (tons CO₂e/ha/yr) | Four-pool InVEST Carbon (above/below/soil/dead) | ⚠️ Methodology simplification |
| Rate source | USDA NRCS / IPCC per-cover-class | `[CC: any MN Carbon bundle in Drive?]` | ❓ — no MN Carbon bundle observed in shared Drive |

**Carbon summary:** Methodology divergence (single-rate vs four-pool). NatCap MN Carbon parameters not observed in the shared data — may be a gap (no MN Carbon bundle in `data/invest/mn_sample_data_natcap_2026/`).

### UMH (Urban Mental Health)

| Parameter | Prototype | NatCap UMH model | Status |
|---|---|---|---|
| RR per 0.1 NDVI | 0.96 (depression) / 0.97 (anxiety), from Liu et al. 2023 | InVEST UMH effect sizes (same source family) | ✅ |
| Baseline prevalence | Uniform national (CDC 2023): 0.21 depression / 0.19 anxiety | Per-administrative-unit BIR (vector input) | ⚠️ Improvised — uniform vs per-admin |
| Cost of illness | National COI estimate | National COI estimate | ✅ |

**UMH summary:** Largely aligned with the InVEST UMH model. The uniform-national baseline prevalence is a known simplification.

### Food Forest yield

| Parameter | Prototype | NatCap MN | Status |
|---|---|---|---|
| `FOOD_FOREST_LBS_ACRE` | 11,500 (NatCap MN benchmark) | Single benchmark; per-crop CoSA model for SA | ✅ (single-value alignment) |

### Minneapolis summary

| Model | Status |
|---|---|
| UCM | ✅ Fully aligned on args and biophysical table |
| UFR | ✅ Rainfall depth aligned with MN-project canonical (Brief 23); CN + damage tables verified identical |
| UNA | ✅ All three parameters + biophysical table aligned with MN-project canonical (Brief 22). Population vintage (2020 vs 2010) is the one remaining minor divergence. |
| Carbon | ⚠️ Methodology simplification (single rate vs four-pool) |
| UMH | ✅ Aligned (uniform-national prevalence improvised) |
| Food Forest | ✅ |

**Overall MN parity:** Mostly aligned. UCM, UFR, UNA, and UMH are all tight after Brief 22 (UNA) and Brief 23 (UFR rainfall). Carbon is methodologically simplified (single rate vs four-pool) — the one remaining methodology gap.

---

## San Antonio

**Active in production.** NatCap reference: `data/sa/natcap_2024/` (compound NLCD+NLUD+tree LULC + UCM/UNA/Carbon biophysical tables + canonical InVEST results, received 2026-05-23). Plus the README at `data/sa/natcap_2024/README_San_Antonio_InVEST_model_inputs.docx`.

### UCM (Urban Cooling)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `uhi_max` | 11 °C (post-Brief 14) | 11 °C (heat-wave-day scenario) | ✅ |
| `t_ref` | (no analog — pure-delta) | 35 °C | ✅ (informational only) |
| `cc_method` | `factors` | `factors` | ✅ |
| `green_area_cooling_distance` | 450 m | 450 m | ✅ |
| `t_air_average_radius` | 600 m | 600 m | ✅ |
| `cc_weight_shade` | 0.6 | 0.6 | ✅ |
| `cc_weight_albedo` | 0.2 | 0.2 | ✅ |
| `cc_weight_eti` | 0.2 | 0.2 | ✅ |
| `do_energy_valuation` | False (SA has no typed buildings) | False | ✅ |
| `do_productivity_valuation` | False | False | ✅ |
| Biophysical table | Köppen-BSh-tuned per-NLCD (14 rows; 4 classes tuned, rest at MN defaults); `data/sa/cooling/biophysical_table_urban_cooling_SA.csv`; columns: `lucode, lulc_desc, shade, kc, albedo, green_area, building_intensity` | Compound NLCD×NLUD×tree-canopy lookup (1,984 rows × 27 cols); `ucm__nlcd_nlud_tree.csv`; columns include `lucode, code, nlcd, lulc_desc, nlud_simple, nlud_simple_class, tree_canopy_cover` (keys) + per-pixel ag/maintenance signals (`fertilizer, pesticide, irrigation, planting_diversity, mowing, public_access, green_space, building_type`) + model params (`shade, kc, albedo, green_area, building_intensity`) + `bioregion`. Value ranges sane (shade 0–1, kc 0–1.1, albedo 0.06–0.80, green_area 0–1, building_intensity 0–1). | ⚠️ Methodology divergence — NatCap uses compound LULC keyed on `lucode` (serial 0–1983); prototype uses per-NLCD. Integration adopts the compound framework. See "SA Compound LULC Framework" subsection below. |
| LULC raster | `data/sa/flood/land_use_compound_sa.tif` (compound NLCD×NLUD×tree-canopy, reprojected to EPSG:5070 + nearest-neighbor resampled at 30 m to the prototype's 1984×1713 grid; reduced via `lulc_crosswalk.csv` to NLCD codes for the existing per-NLCD biophysical table) | `lulc_overlay_3857.tif` (compound, EPSG:3857) | ✅ Adopted 2026-05-24 (Brief 27); per-model compound-keyed table pending Brief 28 |
| ET raster | CGIAR Global-AI/ET0 v3.1 reprojected (~30 arcsec / ~1 km) | `et0_annual_cgiar_3857.tif` (60×63 px at 1,215 m — unusably coarse) | ⚠️ Prototype uses higher-resolution version; NatCap raster not adoptable as-is |

**UCM summary:** Args fully aligned post-Brief-14. LULC raster aligned with NatCap's compound framework 2026-05-24 (Brief 27); biophysical table still per-NLCD with Köppen-BSh tuning, with the compound-keyed `ucm__nlcd_nlud_tree.csv` adoption queued for Brief 28. NatCap's provided ET raster is unusably coarse; prototype's reprojected CGIAR raster is the better source.

### UFR (Urban Flood Risk Mitigation)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `rainfall_depth` | **157 mm (6.18", NatCap SA canonical, Brief 23)** | **157 mm (NatCap SA README)** | ✅ Aligned 2026-05-24 (Brief 23). |
| LULC raster | `data/sa/flood/land_use_compound_sa.tif` (compound NLCD×NLUD×tree-canopy reduced to NLCD codes via crosswalk; same raster used by UCM, shared between flood + cooling per SA's single-LULC convention) | `sa_lc_w_20ac_foodfor_10m.tif` and `sa_lc_w_40ac_foodfor_10m.tif` (pre-computed food-forest scenarios at 10 m) | ⚠️ Methodology divergence: NatCap pre-computes 2 scenarios; prototype runs live (compound LULC adopted at the raster layer 2026-05-24 Brief 27) |
| Soil HSG raster | `data/sa/flood/soil_group_SA.tif` (Bexar County SSURGO, ~30 m) | `sa_env_hsg_int_10m.tif` (10 m) | ⚠️ Different resolution |
| CN biophysical table | Per-NLCD CN values; `data/sa/flood/UFR_biophysical_table_SA.csv` (15 rows; same schema as MN with the +1 NLCD-82 cultivated-crops row for SA cropland) | `biophys_floodmitig_sa.csv` (not yet in `data/sa/natcap_2024/`; path referenced in NatCap README) | ❓ — values not yet diff'd; NatCap table not yet shared in `data/sa/natcap_2024/` |
| Damage loss table | Blank — no per-building damage values | "(leaving blank)" per NatCap README | ✅ Both leave blank (shared data gap) |
| Buildings | Geofabrik OSM (untyped) | Not specified in README | ❓ |

**UFR summary:** Rainfall depth aligned 2026-05-24 (Brief 23). NatCap's methodology pre-computes alternative-LULC scenarios rather than running live per-pct; this is a methodology choice neither right nor wrong but worth documenting. CN biophysical table not yet diff'd (NatCap's `biophys_floodmitig_sa.csv` not in the shared folder).

### UNA (Urban Nature Access)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `urban_nature_demand_per_capita` | 16.7 m²/capita | 16.7 m²/capita | ✅ |
| `search_radius` | 800 m | 800 m | ✅ |
| `decay_function` | dichotomy | dichotomy | ✅ |
| `search_radius_mode` | uniform radius | uniform radius | ✅ |
| `aggregate_by_pop_group` | False | False | ✅ |
| LULC attribute table | Per-NLCD `urban_nature` score | `una__nlcd_nlud_tree.csv` (1,984 rows × 21 cols); columns: keys (`lucode, code, nlcd, lulc_desc, nlud_simple, nlud_simple_class, tree_canopy_cover`) + ag/maintenance signals + `urban_nature` (categorical 0/0.5/1.0: 976 / 960 / 48 rows respectively) + `search_radius_m` (all zeros — the radius is an args-level scalar set at runtime, not in the table) | ⚠️ Methodology divergence — NatCap uses compound LULC keyed on `lucode`; prototype uses per-NLCD. Integration adopts the compound framework. See "SA Compound LULC Framework" below. |
| Population raster | TIGER 2020 block totals, rasterized | `population_per_pixel_2020_3857.tif` (19 MB, higher resolution) | ⚠️ Different sources |
| AOI | Bexar County bbox | `acs_block_group.gpkg` | ⚠️ Different — NatCap uses census block-group polygons |

**UNA summary:** Args fully aligned. Biophysical table and LULC are different methodologies; integration queued. Population and AOI sources differ.

### Carbon

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| Methodology | Single per-NLCD-class rate | Four-pool (above/below/soil/dead) per compound lucode | ⚠️ Methodology divergence |
| Rate source | USDA NRCS / IPCC | `carbon__nlcd_nlud_tree.csv` (1,984 rows × 27 cols): keys (`lucode, code, nlcd, LULC_name, nlud_simple, tree_canopy_cover`) + ag/maintenance signals + four canonical pools (`c_above` max 106, `c_below` max 8, `c_soil` max 259, `c_dead` max 14 — all tons C/ha) + three unused urban-accounting columns (`c_embedded_storage`, `c_embedded_emissions`, `c_annual_emissions`) | ❌ Not adopted; integration queued. See "SA Compound LULC Framework" below. |

**Carbon summary:** Methodology divergence + table format divergence. Adopting NatCap's four-pool framework is a real upgrade queued for integration.

### UMH (Urban Mental Health)

| Parameter | Prototype | NatCap UMH model | Status |
|---|---|---|---|
| Same as MN — UMH model uses uniform national prevalence; no city-specific NatCap config | ✅ | (n/a) | ✅ |

### NDR (Nutrient Delivery Ratio)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| Model implementation | **Not implemented** | Implemented per NatCap README (biophysical table, DEM at 3 m, watersheds, runoff proxy at 32-inch precip) | ❌ Not implemented |

**NDR summary:** Missing model. NatCap considers NDR one of six SA models; the prototype implements 5/6. See NATCAP_COLLABORATION.md Active asks.

### Food Forest yield

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `FOOD_FOREST_LBS_ACRE` | 8,500 (placeholder, hot semi-arid estimate) | Per-crop via `CoSA_Crop_production_ESModeling` (referenced in meeting note) | ⚠️ Methodology simplification (single benchmark vs per-crop) |

### SA Compound LULC Framework (structural inventory)

NatCap's SA data uses a compound LULC framework that overlays three signals: NLCD land cover, NLUD land use, and tree canopy cover. The compound lucode encodes all three; the biophysical tables (UCM/UNA/Carbon) are keyed on it.

**Cross-reference table (`lulc_crosswalk.csv`):**

- **1,984 rows × 15 columns**, fully exhaustive across the combinatorial space.
- Distinct NLCD codes: **16** (`{11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95}` — legacy NLCD 21-class).
- Distinct NLUD simple codes: **31** (1, 2, 3, 4, 11, 12, 13, 14, 15, 20, 30, 41, 42, 51, 52, 53, 60, 70, 80, 90, 100, 110, 120, 131, 132, 133, 134, 140, 151, 152, 153).
- Distinct tree-canopy bins: **4** (`tree_canopy_cover` = `none`, `low`, `medium`, `high`; corresponding `tree` codes 0, 1, 2, 3).
- Total combinations: 16 × 31 × 4 = **1,984** — exactly the row count, fully exhaustive (not curated).
- Compound `code` column is a 4–6-digit ID that does **not** follow a clean positional encoding (the obvious hypothesis `nlcd*100 + nlud*10 + tree` matches only 8 of 1,984 rows). The serial `lucode` column (0..1983) is the actual join key the biophysical tables use; NatCap's encoding for `code` isn't documented in the data and would need NatCap clarification if it matters for integration.
- Frequency columns (`frequency`, `frequency bins`) flag how common each combo is in the AOI. Practicality flags (`is_realistic_to_create`, `is_realistic_to_paint`) indicate which compound classes are physically meaningful to assign in a scenario.

**UCM biophysical (`ucm__nlcd_nlud_tree.csv`):**
- 1,984 rows × 27 columns.
- Keys: `lucode, code, nlcd, lulc_desc, nlud_simple, nlud_simple_class, nlud_simple_subclass, tree, tree_canopy_percentage, tree_canopy_cover, tree_canopy_colors`.
- Context: `bioregion` (`NA28 Southern Mixed Forests & Blackland Prairies` for SA), `notes`.
- Per-pixel maintenance/use signals: `fertilizer, pesticide, irrigation, planting_diversity, mowing, public_access, green_space, building_type`.
- Model parameters: `shade` (0–1), `kc` (0–1.1), `albedo` (0.056–0.80), `green_area` (0–1), `building_intensity` (0–1).

**UNA biophysical (`una__nlcd_nlud_tree.csv`):**
- 1,984 rows × 21 columns.
- Keys + maintenance signals same as UCM (minus `bioregion` / `notes` / `building_intensity`).
- Model parameters: `urban_nature` (categorical 0/0.5/1.0 — 976 / 960 / 48 rows respectively) and `search_radius_m` (all zero — the radius is an args-level scalar, not a per-class table value).

**Carbon biophysical (`carbon__nlcd_nlud_tree.csv`):**
- 1,984 rows × 27 columns.
- Keys + maintenance signals same shape.
- Four-pool model parameters (tons C/ha): `c_above` (max 105.7), `c_below` (max 8.0), `c_soil` (max 259.0 — dominant pool), `c_dead` (max 14.4).
- Three urban-accounting columns not used in this project's parameterization: `c_embedded_storage`, `c_embedded_emissions`, `c_annual_emissions`.

**LULC raster comparison:**

| | Prototype | NatCap |
|---|---|---|
| Path | `data/sa/flood/land_use_2021_sa.tif` | `data/sa/natcap_2024/lulc_overlay_3857.tif` |
| Dimensions | 1984 × 1713 | 2106 × 2218 |
| CRS | EPSG:5070 (NAD83 / Conus Albers) | EPSG:3857 (Web Mercator) |
| Resolution | 30 m | 34.5 m |
| Lucode dtype | `uint8` (NLCD only) | `int16` (compound codes) |
| Lucode range | 11–95 (15 unique codes) | 0–1913 (820 unique compound codes; ~41% of 1,984 theoretical) |
| NoData | 0 | -1 |
| Extent (lat/lon) | 98°48'54"W to 98°11'17"W, 29°12' to 29°38'58"N | 98°50'48"W to 98°11'38"W, 29°9'32" to 29°45'27"N |
| File size | 4.2 MB | 9.4 MB |

**Extent difference flag:** the rasters cover substantially overlapping but not identical geographic areas. NatCap extends ~6 minutes farther north, ~3 farther south, ~2 farther west; same east edge. Both centered on San Antonio. Integration will need to either clip both to a common extent or accept a coverage shift.

**Integration implications.** The compound LULC framework is not a parameter swap — it's a methodology adoption. Adopting it means: (1) the SA LULC raster changes from NLCD-only (15 unique codes, 30 m, EPSG:5070) to compound (820 unique codes used out of 1,984 possible, 34.5 m, EPSG:3857) — requires reprojection or migrating the whole SA stack to 3857; (2) the three biophysical tables (UCM/UNA/Carbon) all change to compound-keyed lookups; (3) the prototype's SA-specific Köppen-BSh tuning becomes obsolete (NatCap's tables capture climate-relevant variation via the tree-canopy and NLUD signals); (4) per-pixel ag/maintenance signals (fertilizer, irrigation, mowing, etc.) become available as new inputs the prototype doesn't currently use. Likely multi-brief workstream.

### San Antonio summary

| Model | Status |
|---|---|
| UCM | ✅ Args fully aligned (post-Brief 14); ⚠️ biophysical table/LULC framework divergence (integration queued) |
| UFR | ⚠️ Methodology divergence (live vs pre-computed scenarios); rainfall depth aligned with SA-project canonical (Brief 23) |
| UNA | ✅ Args fully aligned; ⚠️ biophysical table/LULC framework divergence (integration queued) |
| Carbon | ⚠️ Methodology simplification (single rate vs four-pool); integration queued |
| UMH | ✅ |
| NDR | ❌ Not implemented |
| Food Forest | ⚠️ Single benchmark vs per-crop CoSA |

**Overall SA parity:** All implemented UCM/UNA args are aligned post-Brief-14. The big-picture divergence is the LULC framework — NatCap uses compound NLCD+NLUD+tree-canopy; prototype uses NLCD-only. Integration queued as Briefs 17+. NDR is a real missing model.

---

## Minneapolis Full (dormant)

**`available=False` — hidden from production city selector.** Retained in code for scripts/tests but not user-facing. Per-building-type dollar metrics aren't yet sourced for the expanded extent. Parity-tracking deferred until/unless reactivated.

| Model | Status |
|---|---|
| All | ⏸️ Deferred — city is dormant. When reactivated, populate from MN downtown's row as a starting point; verify against any NatCap MN Full reference if such exists. |

---

## Open questions about parity

These are city-specific parity questions; the methodology-agnostic ones live in NATCAP_COLLABORATION.md.

1. ~~MN UNA parameter divergences and MN UFR rainfall depth~~ — ✅ Both resolved 2026-05-24 (Briefs 22 + 23). Per-city alignment applied: MN UNA now uses NatCap MN-project canonical, MN/SA UFR rainfall now uses each project's canonical depth.

2. **Does NatCap have a MN Carbon bundle?** No `Carbon_sample_data_MN.zip` appeared in the shared Drive. May exist elsewhere; ask.

3. **Where is the "Livable Cities/San Antonio/Results/" folder** referenced in NatCap's README? May contain additional SA data not in `data/sa/natcap_2024/`.

---

## Maintenance

Update when:

- A new alignment fix lands (e.g., Brief 14 updated SA UCM `uhi_max`)
- NatCap shares a new dataset (e.g., MN sample data audit on 2026-05-24)
- A previously-❓ row is verified (biophysical-table diff completed)
- A new methodology divergence is identified
- A city's status changes (e.g., MN Full reactivated)

Pair with `NATCAP_ALIGNMENT.md` and `NATCAP_COLLABORATION.md` updates when the same finding affects all three docs.
