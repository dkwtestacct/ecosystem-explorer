"""config.py — per-city configuration and related constants.

Lifted out of app.py to make the city-comparison surface legible in
isolation. Read-only; mutations belong in app.py's runtime state.

Import pattern:
    from config import CITIES, DEFAULT_COST_GI, DEFAULT_COST_FF, DEFAULT_COST_HD

Or for namespaced access:
    import config
    cfg = config.CITIES["Minneapolis, MN"]
"""

# ── City configuration ─────────────────────────────────────────────────────────
CITIES = {
    'Minneapolis, MN': {
        'data_dir_flood':       'data/flood',
        'data_dir_cooling':     'data/cooling',
        'cn_table_file':        'UFR_biophysical_table_MN.csv',
        'cooling_table_file':   'biophysical_table_urban_cooling_MN.csv',
        # Path keys consumed by load_data + module-level loaders. lulc_file
        # and soil_file resolve relative to data_dir_flood; cooling_lulc_file
        # to data_dir_cooling. Everything else is a project-relative path.
        'lulc_file':            'LULC_NLCD_2021_MN.tif',
        'soil_file':            'soil_group_MN.tif',
        'cooling_lulc_file':    'land_use_2021.tif',
        'pop_file':             'data/population/minneapolis_pop_2020.tif',
        'roads_file':           'data/osm/minneapolis_roads.geojson',
        'dense_scenarios_file': 'data/scenarios_dense_mpls.csv',
        'buildings_file':       'data/invest/flood/UFR_sample_data_MN/buildings.shp',
        # Comprehensive Geofabrik OSM building footprints (~113k city-wide).
        # Unioned into the placement non-convertible mask only; the typed
        # buildings_type_raster that drives the $ metrics still comes from
        # buildings_file (the InVEST UFR sample). See app.py Phase 9b.
        'mask_buildings_file':  'data/osm/minneapolis_buildings.geojson',
        'damage_table_file':    'data/invest/flood/UFR_sample_data_MN/Damage_loss_table_MN.csv',
        'energy_table_file':    'data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/energy_consumption.csv',
        'et_file':              'data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif',
        'tracts_file':          'data/invest/flood/UFR_sample_data_MN/admin_boundaries_census_tracts.shp',
        'una_table_file':       'data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv',
        'baseline_cn':          75.7,
        # 0.1859 = mean(smoothed CC) on the MN baseline LULC after the InVEST
        # UCM rework (ET nodata fix, Gaussian convolution, canonical formula).
        # Auto-recomputed at module load from `_BASELINE_HM_RASTER`, so this
        # value is only a documentation placeholder.
        'baseline_hm':          0.1859,
        'pixel_area_acres':     0.2224,
        'food_forest_lbs_acre': 11_500,
        'uhi_max_c':            2.05,   # InVEST UCM args JSON for the MN AOI
        # NatCap MN UNA project canonical (per Brief 22, source:
        # data/invest/mn_sample_data_natcap_2026/UrbanNatureAccess_sample_data_MN/
        # invest_urban_nature_access_args_MN.json — demand=250, radius=1000,
        # decay=exponential). Different framing than SA's WHO-minimum /
        # heat-wave scenario; per-city alignment is the NatCap pattern.
        'una_demand_m2_per_capita': 250,
        'una_search_radius_m':      1000,
        'una_decay_function':       'exponential',
        # NatCap MN UFR project canonical (Brief 23, source:
        # data/invest/mn_sample_data_natcap_2026/UFR_sample_data_MN/
        # invest_urban_flood_risk_args_MN.json — rainfall_depth=100 mm).
        # 100 mm = 3.94 inches; SCS-CN formula uses inches internally.
        'design_storm_inches':      3.94,
        'available':            True,
        'crs':                  'EPSG:26915',
        'precomputed_dir':      'data/precomputed/minneapolis_mn',
        # Reference points plotted on the tradeoff scatter. `cooling` is mean
        # HMI under the canonical InVEST UCM algorithm (HMI = max(CC_local,
        # CC_park)); `flood` is 100 − mean CN. Recomputed 2026-05-21 (seed=42,
        # random placement) after the canonical-HMI landing — see REFERENCE.md
        # "Reference Benchmarks". Each "All X" scenario is pct_converted=50
        # with 100 % allocation to that single land cover; "Baseline" is the
        # unmodified LULC.
        'ref_scenarios': {
            'Baseline':                     {'flood': 24.3,  'cooling': 0.1944, 'color': 'steelblue'},
            'All Food Forest (NLCD 41)':    {'flood': 26.1,  'cooling': 0.4146, 'color': 'green'},
            'All Green Infra (NLCD 90)':    {'flood': 43.3,  'cooling': 0.4235, 'color': 'teal'},
            'All High Density (NLCD 24)':   {'flood': 21.4,  'cooling': 0.1698, 'color': 'red'},
        },
    },
    'Minneapolis Full, MN': {
        'data_dir_flood':       'data/minneapolis_expanded',
        'data_dir_cooling':     'data/minneapolis_expanded',
        # Reuses the MN biophysical tables — same NLCD class space, same
        # USDA-standard CN values; no city-specific tuning yet. The cooling
        # tables sit under data/cooling/ and data/flood/ (NOT under
        # data_dir_cooling/data_dir_flood). load_data tries each in turn.
        'cn_table_file':        'UFR_biophysical_table_MN.csv',
        'cooling_table_file':   'biophysical_table_urban_cooling_MN.csv',
        'lulc_file':            'lulc_nlcd_2021_mpls_full.tif',
        'soil_file':            'soil_group_mpls_full.tif',
        # Same file used for both flood and cooling (the InVEST sample MN
        # split is a downtown-only artifact — full city is one raster).
        'cooling_lulc_file':    'lulc_nlcd_2021_mpls_full.tif',
        'pop_file':             'data/minneapolis_expanded/pop_mpls_full.tif',
        'roads_file':           'data/minneapolis_expanded/roads_mpls_full.geojson',
        'dense_scenarios_file': 'data/scenarios_dense_mpls_full.csv',
        # OSM buildings have no per-type codes, so per-type lookups (energy
        # savings, flood damage avoided) degrade to $0 with explanatory
        # tooltips; the BUILDINGS_RASTER mask still works for spatial
        # placement. See REFERENCE.md "Option A buildings semantics".
        # GeoPackage — the equivalent GeoJSON was 102 MB, over GitHub's 100 MB
        # hard limit; .gpkg compresses the same 185,490 polygons to 49 MB.
        'buildings_file':       'data/minneapolis_expanded/buildings_mpls_full.gpkg',
        'damage_table_file':    'data/invest/flood/UFR_sample_data_MN/Damage_loss_table_MN.csv',
        'energy_table_file':    'data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/energy_consumption.csv',
        # Reuse the InVEST sample ET raster; bilinear-extrapolates beyond its
        # native ~10 × 10 km extent at the AOI corners. Order-of-magnitude OK.
        'et_file':              'data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/reference_evapotranspiration_annual.tif',
        'tracts_file':          'data/minneapolis_expanded/tracts_hennepin.shp',
        'una_table_file':       'data/invest/nature_access/UrbanNatureAccess_sample_data_MN/LULC_attribute_table_UNA.csv',
        'baseline_cn':          None,    # computed dynamically at module load
        'baseline_hm':          None,    # computed dynamically at module load
        'pixel_area_acres':     0.2224,  # NLCD 30 m in EPSG:5070
        'food_forest_lbs_acre': 11_500,
        'uhi_max_c':            2.05,    # same MN AOI climate as downtown
        # Same MN-project UNA framing as downtown (Brief 22). MN Full is
        # `available=False` but shares the MN canonical-params choice.
        'una_demand_m2_per_capita': 250,
        'una_search_radius_m':      1000,
        'una_decay_function':       'exponential',
        # Same MN-project UFR rainfall as downtown (Brief 23).
        'design_storm_inches':      3.94,  # 100 mm per NatCap MN args.json
        # Hidden from the UI pending per-building-type data for the expanded
        # area. OSM-derived buildings carry no `type` codes, so the
        # Flood Damage Avoided and Cooling Energy Savings $-metrics degrade
        # to $0 — incomplete coverage relative to the downtown extent (which
        # uses the InVEST UFR sample shapefile with `type` ∈ {0,1,2,3}).
        # Flip back to True once a typed building dataset for the expanded
        # area exists. The pipeline / data are still tracked; only the UI
        # entry is suppressed.
        'available':            False,
        'crs':                  'EPSG:5070',
        'precomputed_dir':      'data/precomputed/minneapolis_full_mn',
        'notes': (
            'Full city coverage 204 km² vs 122 km² downtown. Same biophysical '
            'tables as Minneapolis, MN. SSURGO soil + Census 2020 population '
            'rasterized to a 374 × 607 EPSG:5070 grid; Geofabrik OSM re-clipped '
            'to the same extent. Cooling-energy-savings and flood-damage-avoided '
            'metrics return $0 because OSM buildings lack per-type codes — '
            'spatial-placement mask still works. See REFERENCE.md.'
        ),
        # Recomputed via verify_cooling.py --city "Minneapolis Full, MN" (seed=42)
        # against the expanded EPSG:5070 grid. Each "All X" scenario is
        # pct_converted=50 with 100 % allocation to that single land cover.
        'ref_scenarios': {
            'Baseline':                     {'flood': 22.3,  'cooling': 0.1600, 'color': 'steelblue'},
            'All Food Forest (NLCD 41)':    {'flood': 23.9,  'cooling': 0.2821, 'color': 'green'},
            'All Green Infra (NLCD 90)':    {'flood': 37.0,  'cooling': 0.2864, 'color': 'teal'},
            'All High Density (NLCD 24)':   {'flood': 19.5,  'cooling': 0.1383, 'color': 'red'},
        },
    },
    'San Antonio, TX': {
        'data_dir_flood':       'data/sa/flood',
        'data_dir_cooling':     'data/sa/cooling',
        # SA flood CN integration deferred 2026-05-28 pending NatCap response.
        # The staged biophys_floodmitig_sa.csv (now renamed
        # biophys_floodmitig_sa_STAGED_pending_natcap.csv) is loadable and the
        # new NLCD×tree-canopy lookup path (reduce_compound_to_nlcd_tree in
        # app.py) is fully implemented but NOT currently called — the SA flood
        # CN lookup was reverted to the 2-digit NLCD path. NatCap's table
        # diverges systematically from NRCS TR-55 (most strikingly wetlands at
        # CN ~88-92 vs NRCS 30, with smaller anomalies for developed-low/med,
        # grassland, and shrub/scrub — see NATCAP_COLLABORATION.md question 12).
        # Adopting it as-is would invert the prototype's "GI mitigates flooding"
        # narrative for SA in ways we can't justify with NatCap's delivered docs
        # (the `Ben NDR and Flood Mar_2023.pptx` referenced in the README isn't
        # in the share). Until NatCap clarifies, this points at the
        # MN-placeholder table (also known-wrong, but in a familiar way). A
        # follow-up brief re-enables the new path (config one-liner + revert
        # the two reverted lookup sites).
        'cn_table_file':        'UFR_biophysical_table_SA.csv',
        # Brief 28b: switched from the Köppen-BSh-tuned per-NLCD prototype
        # table to NatCap's compound NLCD×NLUD×tree-canopy UCM table (1,984
        # rows). Keyed on the compound `lucode` 0–1983 — UCM consumers index
        # the compound raster (`cooling_lulc_compound`) directly. The retired
        # `data/sa/cooling/biophysical_table_urban_cooling_SA.csv` is kept on
        # disk for historical reference and is no longer loaded. Relative
        # `../natcap_2024/` traversal mirrors the existing `cooling_lulc_file`
        # convention so we don't duplicate the file.
        'cooling_table_file':   '../natcap_2024/ucm__nlcd_nlud_tree.csv',
        # Path keys. Inputs sourced 2026-05-09:
        #   - SSURGO via download_ssurgo_sa.py (TX029, 6,090 polygons,
        #     44 % D-class clay-rich vs Hennepin's 0 % pure-D)
        #   - Census 2020 pop via download_census_pop_sa.py (Bexar 48029,
        #     1,906,325 people in raster — between SA proper 1.4 M and
        #     full Bexar 2.0 M)
        #   - CGIAR Global-AI/ET0 v3.1 via download_et_sa.py (1,580–1,716
        #     mm/yr, ~50 % above MN's ~1,150 mm/yr)
        # All blocking inputs sourced 2026-05-09 / 10:
        #   roads + buildings: Geofabrik TX state extract + Option B filter
        #     (55,553 road segments, 345,900 building polygons in SA bbox).
        #     Buildings stored as GeoPackage (binary, 92 MB) because the
        #     equivalent GeoJSON exceeds GitHub's 100 MB hard limit.
        #   tracts: TIGER 2020 (375 Bexar tracts).
        # Still TODO: SA-specific damage table + dense surrogate grid (the
        # latter is built automatically by precompute_scenarios.py once the
        # city is `available=True`).
        # OSM buildings carry `type` as OSM strings ('house', 'apartments',
        # ...) not the integer 0–3 codes InVEST expects, so SA uses Option A
        # buildings semantics: spatial-placement mask works, energy/damage
        # cards display "—" with explanatory tooltip.
        'lulc_file':            'land_use_2021_sa.tif',
        'soil_file':            'soil_group_sa.tif',
        # SA has one canonical LULC raster shared between flood + cooling;
        # MN's separate cooling LULC is an InVEST-sample convention. Use a
        # relative-path traversal from data_dir_cooling so we don't have to
        # duplicate the 4 MB file.
        'cooling_lulc_file':    '../flood/land_use_2021_sa.tif',
        'pop_file':             'data/sa/population/sa_pop_2020.tif',
        'roads_file':           'data/sa/roads_sa.geojson',
        'dense_scenarios_file': 'data/scenarios_dense_sa.csv',  # built by precompute_scenarios.py
        'buildings_file':       'data/sa/buildings_sa.gpkg',
        'damage_table_file':    None,   # SA project deliverables — TODO
        'energy_table_file':    'data/invest/cooling/UrbanCooling_sample_data/UrbanCooling/energy_consumption.csv',
        'et_file':              'data/sa/cooling/et_annual_sa.tif',
        # Brief 31: switched from `data/sa/tracts_bexar.shp` (TIGER 2020 Bexar
        # County tracts, 375 polygons) to NatCap's ACS block-group polygons
        # (1,124 polygons covering the City of San Antonio at finer
        # granularity). The block-group polygons match the framing NatCap uses
        # for equity analysis in Vibrant Land (Guerry et al. 2023, Figures 5
        # + 10). The polygon file feeds only `compute_per_tract_summary`'s
        # "Neighborhood breakdown" table — no biophysical metric depends on
        # it. The retained `tracts_bexar.shp` file remains on disk for
        # reference. EPSG:3857 → reprojected to city CRS at load time.
        'tracts_file':          'data/sa/natcap_2024/acs_block_groups_3857.gpkg',
        # Brief 29: NatCap's SA-curated compound NLCD×NLUD×tree-canopy UNA
        # biophysical table (1,984 rows; urban_nature ∈ {0.0, 0.5, 1.0}).
        # Indexed directly by the compound LULC raster (no NLCD reduction);
        # captures NLUD + tree-canopy variation per pixel that the prior
        # per-NLCD borrowed-from-MN table couldn't represent.
        'una_table_file':       'data/sa/natcap_2024/una__nlcd_nlud_tree.csv',
        # Brief 30: NatCap's SA-curated compound NLCD×NLUD×tree-canopy Carbon
        # biophysical table (1,984 rows × 27 cols). Four pools (c_above,
        # c_below, c_soil, c_dead in tons C/ha) indexed directly by the
        # compound LULC raster. Switches SA from the per-conversion-type
        # single-rate annual proxy to the canonical InVEST four-pool stock
        # framework. SA only — MN keeps the single-rate proxy via
        # CARBON_SEQ_RATES (no four-pool data available for MN).
        # Methodology matches NatCap's Vibrant Land (Guerry et al. 2023)
        # framework; SC-CO2 constant (EPA_SOCIAL_COST_CARBON, $190/t @ 2%,
        # EPA 2023) is the prototype's choice — more current than Vibrant
        # Land's IWG 2021 ($53/t @ 3%) but the same US-government standard
        # lineage. See DESIGN_NOTES.md "SA Carbon four-pool framework adoption".
        'carbon_table_file':    'data/sa/natcap_2024/carbon__nlcd_nlud_tree.csv',
        # Documentation only (sourced from real SSURGO TX029, CGIAR ET0 v3.1,
        # full InVEST UCM CC pipeline) — the live override at module load
        # recomputes these baselines from the rasters and is authoritative.
        'baseline_cn':          76.54,
        'baseline_hm':          0.2866,
        'baseline_ndvi':        0.4242,
        'pixel_area_acres':     0.2224,  # NLCD 30 m in EPSG:5070
        # Conservative SA-specific estimate for a pecan / fig / mulberry / nopal
        # mix (per NatCap SA Urban Agriculture project report). No published
        # per-crop yield numbers were shipped with the SA inputs, so 8,500 is
        # a placeholder set below the 11,500 MN benchmark to reflect lower
        # productivity in hot semi-arid climate. Replace with the project's
        # weighted average once those yield benchmarks are documented.
        'food_forest_lbs_acre': 8_500,
        # NatCap's curated SA InVEST inputs (`data/sa/natcap_2024/
        # README_San_Antonio_InVEST_model_inputs.docx`) specify a heat-wave-day
        # scenario: reference_air_temperature 35 °C + uhi_effect 11 °C → ~46 °C
        # / ~115 °F peak urban. Adopted 2026-05-24 per the project's
        # NatCap-canonical alignment principle. The prototype previously
        # used 3.5 °C, an average-summer-day estimate. ΔT in °F = ΔHMI × 19.8.
        # (The prototype reports pure deltas — `reference_air_temperature`
        # has no analog to mirror.) See DESIGN_NOTES.md "UCM args alignment".
        'uhi_max_c':            11,
        # SA UNA from NatCap's curated SA dataset README (Brief 22) —
        # WHO-minimum demand, dichotomy decay. Different framing than MN's
        # aspirational targets / exponential decay; both are NatCap-canonical
        # for their respective project frames.
        'una_demand_m2_per_capita': 16.7,
        'una_search_radius_m':      800,
        'una_decay_function':       'dichotomy',
        # NatCap SA UFR rainfall per the README (Brief 23): 157 mm = 6.18 inches.
        # SA's heavier sub-tropical convective rain regime vs MN's 100 mm.
        'design_storm_inches':      6.18,
        # NatCap compound LULC framework (Brief 27, foundational adoption).
        # `compound_lulc_file` is reprojected EPSG:3857 → EPSG:5070 with
        # nearest-neighbor at 30 m (1984×1713 grid) by hand-run gdalwarp; see
        # `data/sa/README.md` and `SA_INTEGRATION_PLAN.md`. `crosswalk_file`
        # maps each compound `lucode` (0–1983) to its constituent NLCD/NLUD/
        # tree-canopy bins; the prototype reduces compound → NLCD via the
        # crosswalk's `nlcd` column at load time so existing per-NLCD
        # biophysical tables (UCM/UFR/UNA) keep working unchanged. Briefs
        # 28–30 will swap individual model tables to compound-keyed versions.
        'compound_lulc_file':       'land_use_compound_sa.tif',
        'crosswalk_file':           '../natcap_2024/lulc_crosswalk.csv',
        # Fallback compound lucodes for the three conversion targets, picked
        # by surveying `is_realistic_to_create=yes` rows in the crosswalk and
        # preferring the highest-frequency representative of each target NLCD:
        #   FF (NLCD 41) → 1310: Deciduous Forest, Timber NLUD, medium canopy
        #   GI (NLCD 90) →  122: Woody Wetlands, Wetland NLUD, medium canopy
        #   HD (NLCD 24) →  341: Developed High Intensity, Residential/Urban, low canopy
        # Used by Brief 28+ when an NLUD/tree-canopy combination from a source
        # pixel doesn't appear with the target NLCD in the crosswalk. Logged
        # in DESIGN_NOTES.md for durable rationale.
        'default_ff_lucode':        1310,
        'default_gi_lucode':         122,
        'default_hd_lucode':         341,
        # Re-enabled now that data/scenarios_dense_sa.csv is committed
        # (Balanced mode reads the precomputed grid instead of recomputing
        # the 25–50 min lookup table) and the High-Resolution lookup compute
        # is deferred behind explicit user opt-in. Fast mode still computes
        # a coarse grid live; if that turns out to OOM the 1 GB free-tier
        # worker, flip back to False.
        'available':            True,
        'crs':                  'EPSG:5070',
        'precomputed_dir':      'data/sa/precomputed',
        'notes': (
            'Data source: NatCap SA Urban Agriculture Project 2023. '
            'LULC from NLCD 2021, SSURGO from USDA SDA (TX029, 44 % D-class), '
            'population from Census 2020 Bexar (FIPS 48029, 1.91 M in raster), '
            'reference ET from CGIAR Global-AI/ET0 v3.1 (1,580–1,716 mm/yr). '
            'Baseline constants recomputed live at module load.'
        ),
        # Reference points for the tradeoff scatter. `cooling` is mean HMI
        # under the canonical InVEST UCM algorithm (HMI = max(CC_local,
        # CC_park)); `flood` is 100 − mean CN. Recomputed 2026-05-21 (seed=42,
        # random placement) on the SA EPSG:5070 grid. Each "All X" scenario is
        # pct_converted=50 with 100 % allocation to that single land cover;
        # "Baseline" is the unmodified LULC.
        'ref_scenarios': {
            'Baseline':                     {'flood': 23.5, 'cooling': 0.3071, 'color': 'steelblue'},
            'All Food Forest (NLCD 41)':    {'flood': 24.3, 'cooling': 0.4071, 'color': 'green'},
            'All Green Infra (NLCD 90)':    {'flood': 33.5, 'cooling': 0.4275, 'color': 'teal'},
            'All High Density (NLCD 24)':   {'flood': 22.0, 'cooling': 0.2927, 'color': 'red'},
        },
    },
}

# ── Cost defaults ($/acre) ─────────────────────────────────────────────────────
DEFAULT_COST_GI   = 50_000   # Green infrastructure / woody wetlands
DEFAULT_COST_FF   = 10_000   # Food forest
DEFAULT_COST_HD   =  5_000   # High density development
