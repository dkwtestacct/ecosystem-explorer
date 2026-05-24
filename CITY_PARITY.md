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
| `rainfall_depth` | 50.8 mm (2 inches, prototype design-storm choice) | 100 mm | ⚠️ Diverges. Prototype's 2-inch design storm is its own choice, not aligned with either NatCap project's value. |
| LULC raster | `data/flood/LULC_NLCD_2021_MN.tif` | `LULC_NLCD_2021_MN.tif` | ✅ (same file, MD5-confirmed) |
| Soil HSG raster | `data/flood/soil_group_MN.tif` | `soil_group_MN.tif` | ✅ |
| Buildings | `data/flood/buildings.shp` (UFR sample) | `buildings.shp` | ✅ |
| CN biophysical table | `data/flood/UFR_biophysical_table_MN.csv` | `UFR_biophysical_table_MN.csv` | ✅ Verified identical 2026-05-24 on all 14 shared lucodes (CN_A/CN_B/CN_C/CN_D + NLCD_Land label). Prototype has one extra row, lucode 82 (Cultivated Crops, SA support). |
| Damage loss table | `data/invest/flood/UFR_sample_data_MN/Damage_loss_table_MN.csv` | `Damage_loss_table_MN.csv` (Roads=40, Commercial=120, Residential=150, Industrial=100) | ✅ Verified bit-identical 2026-05-24 — same file from same InVEST UFR sample bundle. |
| AOI watersheds | `data/invest/flood/UFR_sample_data_MN/admin_boundaries_census_tracts.shp` (declared as `tracts_file` in `config.py`) | `admin_boundaries_census_tracts.shp` | ✅ MD5 verified identical 2026-05-24 (`acbd9a8d28892dd4dfaf003b896235b4` on `.shp`; `.dbf` and `.prj` also identical). Same file from same InVEST UFR sample bundle. |

**UFR summary:** Same data sources as NatCap. Rainfall depth diverges — prototype's choice, not NatCap-aligned. CN biophysical and damage loss tables verified identical 2026-05-24.

### UNA (Urban Nature Access)

| Parameter | Prototype | NatCap MN project | Status |
|---|---|---|---|
| `urban_nature_demand_per_capita` | **16.7 m²/capita** (SA-project value) | **250 m²/capita** | ⚠️ Diverges (15×). See NATCAP_COLLABORATION.md Q1. |
| `search_radius` | **800 m** (SA-project value) | **1000 m** | ⚠️ Diverges. |
| `decay_function` | **dichotomy** (SA-project value) | **exponential** | ⚠️ Diverges (methodology). |
| `search_radius_mode` | uniform radius | uniform radius | ✅ |
| `aggregate_by_pop_group` | False | False | ✅ |
| `population_group_radii_table` | not used | empty | ✅ |
| LULC raster | `data/cooling/land_use_2021.tif` (UNA-sample LULC) | `LULC_NLCD_2021.tif` (in MN bundle) | ✅ MD5 verified identical 2026-05-24 (`56d1080fa70576cad15896642a107a3d`, 297,417 bytes both). Confirms the prototype's cooling LULC is byte-identical to NatCap's MN UNA sample LULC — consistent with the earlier `UNA_LULC_INVESTIGATION.md` finding. |
| LULC attribute table | `data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv` | `LULC_attribute_table_UNA.csv` | ✅ Verified bit-identical 2026-05-24 (14 rows: lucode, lulc_desc, urban_nature, search_radius_m). Same file from same InVEST UNA sample bundle. |
| Population raster | Census 2020 blocks, rasterized via `download_census_pop.py` | `total_population_census_2010.tif` | ⚠️ Vintage differs (2020 vs 2010) |

**UNA summary:** Three substantial parameter divergences — demand, radius, decay. Prototype uses SA-project values; NatCap's MN project uses different framing. LULC attribute table itself is verified identical (the parameter divergences are runtime args, not table values). **High-priority open question:** should the prototype switch to MN-project values for MN? See NATCAP_COLLABORATION.md.

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
| UFR | ⚠️ Rainfall depth diverges; CN + damage tables verified identical |
| UNA | ⚠️ Three parameter divergences (demand, radius, decay) — using SA-project values. Biophysical table itself verified identical. |
| Carbon | ⚠️ Methodology simplification (single rate vs four-pool) |
| UMH | ✅ Aligned (uniform-national prevalence improvised) |
| Food Forest | ✅ |

**Overall MN parity:** Mixed. UCM and UMH are tight. UFR has one parameter divergence (rainfall depth) atop verified-identical tables. UNA has three substantial parameter divergences atop a verified-identical biophysical table. Carbon is methodologically simplified. The MN UNA parameter gap is the single biggest divergence for the city.

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
| Biophysical table | Köppen-BSh-tuned per-NLCD (4 classes tuned, rest at MN defaults); `data/sa/cooling/biophysical_table_*.csv` | Compound NLCD×NLUD×tree-canopy lookup (1,984 rows); `ucm__nlcd_nlud_tree.csv` | ⚠️ Methodology divergence — NatCap uses compound LULC; prototype uses per-NLCD. Integration queued (Briefs 17+). |
| LULC raster | `data/sa/flood/land_use_2021_sa.tif` (NLCD-only, EPSG:5070) | `lulc_overlay_3857.tif` (compound, EPSG:3857) | ⚠️ Different LULC; integration queued |
| ET raster | CGIAR Global-AI/ET0 v3.1 reprojected (~30 arcsec / ~1 km) | `et0_annual_cgiar_3857.tif` (60×63 px at 1,215 m — unusably coarse) | ⚠️ Prototype uses higher-resolution version; NatCap raster not adoptable as-is |

**UCM summary:** Args fully aligned post-Brief-14. Biophysical table and LULC are different methodologies; NatCap's compound-lucode framework is queued for integration. NatCap's provided ET raster is unusably coarse; prototype's reprojected CGIAR raster is the better source.

### UFR (Urban Flood Risk Mitigation)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `rainfall_depth` | 50.8 mm (2 inches) | 157 mm | ⚠️ Diverges. Prototype's 2-inch design storm doesn't match NatCap's value. |
| LULC raster | `data/sa/flood/land_use_2021_sa.tif` (NLCD-only) | `sa_lc_w_20ac_foodfor_10m.tif` and `sa_lc_w_40ac_foodfor_10m.tif` (pre-computed food-forest scenarios at 10 m) | ⚠️ Methodology divergence: NatCap pre-computes 2 scenarios; prototype runs live |
| Soil HSG raster | `data/sa/flood/soil_group_SA.tif` (Bexar County SSURGO, ~30 m) | `sa_env_hsg_int_10m.tif` (10 m) | ⚠️ Different resolution |
| CN biophysical table | Per-NLCD CN values | `biophys_floodmitig_sa.csv` | ❓ — values not yet diff'd |
| Damage loss table | Blank — no per-building damage values | "(leaving blank)" per NatCap README | ✅ Both leave blank (shared data gap) |
| Buildings | Geofabrik OSM (untyped) | Not specified in README | ❓ |

**UFR summary:** Rainfall depth diverges. NatCap's methodology pre-computes alternative-LULC scenarios rather than running live per-pct; this is a methodology choice neither right nor wrong but worth documenting. CN table not yet diff'd.

### UNA (Urban Nature Access)

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| `urban_nature_demand_per_capita` | 16.7 m²/capita | 16.7 m²/capita | ✅ |
| `search_radius` | 800 m | 800 m | ✅ |
| `decay_function` | dichotomy | dichotomy | ✅ |
| `search_radius_mode` | uniform radius | uniform radius | ✅ |
| `aggregate_by_pop_group` | False | False | ✅ |
| LULC attribute table | Per-NLCD `urban_nature` score | `una__nlcd_nlud_tree.csv` (1,984 rows; categorical 0/0.5/1.0 keyed on compound lucode) | ⚠️ Methodology divergence — NatCap uses compound LULC; prototype uses per-NLCD. Integration queued. |
| Population raster | TIGER 2020 block totals, rasterized | `population_per_pixel_2020_3857.tif` (19 MB, higher resolution) | ⚠️ Different sources |
| AOI | Bexar County bbox | `acs_block_group.gpkg` | ⚠️ Different — NatCap uses census block-group polygons |

**UNA summary:** Args fully aligned. Biophysical table and LULC are different methodologies; integration queued. Population and AOI sources differ.

### Carbon

| Parameter | Prototype | NatCap SA project | Status |
|---|---|---|---|
| Methodology | Single per-NLCD-class rate | Four-pool (above/below/soil/dead) per compound lucode | ⚠️ Methodology divergence |
| Rate source | USDA NRCS / IPCC | `carbon__nlcd_nlud_tree.csv` (1,984 rows × 4 pools) | ❌ Not adopted; integration queued |

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

### San Antonio summary

| Model | Status |
|---|---|
| UCM | ✅ Args fully aligned (post-Brief 14); ⚠️ biophysical table/LULC framework divergence (integration queued) |
| UFR | ⚠️ Rainfall depth diverges; methodology divergence (live vs pre-computed scenarios) |
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

1. **MN UNA parameter divergences and MN UFR rainfall depth** — both are city-parity-relevant but the decisions belong to NATCAP_COLLABORATION.md. See its open-questions section for the MN UNA values (demand 16.7→250, radius 800→1000, decay dichotomy→exponential) and the MN UFR rainfall (50.8→100 mm) questions.

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
