# Architecture

**Audience:** Internal
**Status:** Current
**Use this for:** Understanding how the app is organized before reading code
**Do not use this for:** Metric definitions (→ REFERENCE.md), design rationale (→ DESIGN_NOTES.md), or collaboration history
**Source of truth for:** System components and data flow

---

## 1. System overview

The Ecosystem Explorer is a Streamlit app that lets users explore tradeoffs in urban land-use scenarios. Underneath the UI sits a layered system: a per-pixel biophysical engine (canonical-InVEST-aligned numpy reimplementations), a lookup-table cache for instant slider response, a Random Forest surrogate for fast scenario search, plus first-class subsystems for scenario sourcing, validation/provenance, and InVEST-bundle export.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  UI (Streamlit)                                                          │
│  Sidebar sliders · Scenario tab · Tradeoff Analysis · Map View           │
└──────────────┬──────────────────────────────────┬───────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐         ┌────────────────────────────────────┐
│ Scenario sources (§2)    │         │ Validation + provenance (§6)       │
│ 4 PROVENANCE_* constants │         │ per-card badge + scenario header   │
│ (Baseline / NatCap-fixed │         │ (renderers wired into every card   │
│  / Explorer / Optimizer) │         │  + the scenario tab top)           │
└─────────────┬────────────┘         └────────────┬───────────────────────┘
              │                                   │
              ▼                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Runtime data model (§3)                                                │
│ CityState NamedTuple — rasters, lookup arrays, baseline rasters,      │
│ population, buildings/roads, masks, distance fields                    │
│ + CRS invariants                                                       │
└────────┬──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Three computational layers (§5)                                        │
│ Layer 1 — Raster simulations (full canonical InVEST math, per pixel)  │
│ Layer 2 — Lookup table (precomputed aggregates + live overwrite)      │
│ Layer 3 — Surrogate model (RF over slider space, ~10k-candidate opt)  │
└────────┬──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Outputs                                                                │
│ Metric cards · tradeoff chart · map view · saved scenarios CSV ·       │
│ InVEST-bundle export (§7)                                              │
└──────────────────────────────────────────────────────────────────────┘
```

Each computational layer exists for a different reason: realism (Layer 1), speed (Layer 2), wider search space (Layer 3). Scenario sources, validation/provenance, and export are first-class subsystems alongside them — they have code homes (`natcap_scenarios.py` / `natcap_validation.py` / `export_invest_bundle.py`) and aren't documentation-only ideas.

For deeper detail: REFERENCE.md covers methodology (what each metric means); DESIGN_NOTES.md covers internal design decisions (why each choice was made); NATCAP_ALIGNMENT.md is the authoritative source for the validation badge taxonomy + per-metric alignment status; DATA_INVENTORY.md is the authoritative source for the per-city data files.

---

## 2. Scenario sources

Every scenario carries one of four **provenance** values from `natcap_scenarios.PROVENANCE_*` (re-exported via `export_invest_bundle` as `eib.PROVENANCE_*`):

| Constant | Source label (rendered) | Where it fires |
|---|---|---|
| `PROVENANCE_BASELINE` | `Baseline` | `results['pct_converted'] == 0` |
| `PROVENANCE_NATCAP_FIXED` | `NatCap published reference` | the fixed-scenario reference view (SA only) |
| `PROVENANCE_EXPLORER` | `Explorer-generated` | slider-driven scenarios |
| `PROVENANCE_OPTIMIZER` | `Surrogate-suggested` | `st.session_state['applied_from_optimizer']` is set (the user clicked Apply on an optimizer suggestion) |

**SA-only scope for `PROVENANCE_NATCAP_FIXED`.** The fixed-scenario taxonomy `SA_NATCAP_FIXED_SCENARIOS` lives only for SA — MN has no NatCap fixed-scenario corpus, and the reference view is gated by `selected_city.startswith("San Antonio")`. For SA fixed alternatives (FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX), the rasters NatCap provided are **flood-only**: compound LULC inputs for those alternatives weren't shipped (see OPEN_QUESTIONS.md). The export bundle for fixed alternatives is gated accordingly — flood-only, with compound-model args explicitly marked unavailable in `metadata.json` (see §7).

**Saved and exported are NOT separate provenance values.** A "saved scenario" carries whatever provenance the active scenario had at save time; an "exported scenario" propagates the active provenance into the bundle's `metadata.json`.

---

## 3. Runtime data model

### CityState

The active city's state is a frozen `CityState` NamedTuple produced by `_load_city_runtime_state(city_key)`. Fields:

- **Rasters** — `cooling_lulc`, `cooling_lulc_compound` (SA only — `None` for MN), `soil_group`, `ndvi_baseline`, `pop_count_raster`, `et_raster`, `buildings_mask`, `buildings_type`, `roads_mask`, `tracts_index`
- **Lookup arrays** (compound-keyed for SA, NLCD-keyed for MN) — `cn_a_arr`, `cn_b_arr`, `cn_c_arr`, `cn_d_arr` (CN per lucode × soil group); `shade_arr`, `kc_arr`, `albedo_arr` (UCM biophysical); `urban_nature_arr` (UNA); `c_above_arr`, `c_below_arr`, `c_soil_arr`, `c_dead_arr` (Carbon four-pool); `compound_to_nlcd`, `compound_to_nlcd_tree`, `compound_after_ff`, `compound_after_gi`, `compound_after_hd` (SA conversion lookups)
- **Baseline rasters** (precomputed once) — `baseline_hmi_raster`, `baseline_ne_raster`, `baseline_access_score`, `baseline_runoff_pixel`, `convertible_mask`
- **Distance fields** — `nature_distance: dict[int, np.ndarray]` (per-NLCD-lucode distance rasters, persisted under `data/precomputed/<city>/nature_distance_<lucode>.npy`)
- **Derived scalars** — `baseline_hm`, `baseline_cn`, `baseline_ndvi`
- **CRS + transform metadata** — `ref_transform`, `crs_wkt`

After the loader returns, ~25 module-level names are rebound to fields of the NamedTuple (`cooling_lulc = _CURRENT_CITY_STATE.cooling_lulc`, `ET_RESIZED = …`, `BUILDINGS_RASTER = …`, `_BASELINE_HM_RASTER = …`, etc.) so downstream function bodies can read them as bare globals. This pattern keeps the code surface stable while moving heavy allocations behind `@st.cache_resource`.

**Exception — `baseline_hm` and `baseline_cn` are NOT aliased to module level.** Every consumer reads them via `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn`. The reason is silent-staleness defense: if a future code path forgets to refresh a module-level scalar on city switch, the wrong-but-plausible number would silently flow through every downstream computation. Going through the state handle forces correctness — arrays would crash on shape mismatch, but scalars look fine. The two scalars get explicit-handle treatment to close that gap.

### CRS handling

Every raster the prototype reads at runtime is in its city's canonical equal-area CRS — EPSG:26915 (NAD83 / UTM 15N) for Minneapolis, EPSG:5070 (NAD83 / Conus Albers Equal-Area) for San Antonio. Both are equal-area or near-equal-area at the relevant latitudes (UTM ground-area distortion at MN is ~0.05 %, well within rounding), so `PIXEL_AREA_ACRES = 0.2224` is correct for the 30 × 30 m runtime pixels.

**Source data in other CRSs is reprojected at preparation time, not at runtime.** NatCap's San Antonio compound LULC delivery is in EPSG:3857 (Web Mercator), which heavily distorts area at non-equatorial latitudes and is unsuitable for area-based math. The source compound LULC (`data/sa/natcap_2024/lulc_overlay_3857.tif`) was reprojected to the live EPSG:5070 raster (`data/sa/flood/land_use_compound_sa.tif`) using nearest-neighbor resampling at 30 m before it ever enters the runtime pipeline. The 3857 source files are preserved on disk for provenance but are not read by `app.py`.

The Streamlit map rendering uses EPSG:3857 internally (because tile servers and Folium expect it), but this is a one-way display conversion applied after all area math has happened in equal-area space. No area-dependent metric is computed in 3857.

**Runtime assertion.** Every `rasterio.open(...)` site in `app.py` calls `_assert_raster_crs(src, expected_crs, file_path)` after opening; the helper raises `ValueError` with a clear file-naming message if the raster's CRS doesn't match the city's canonical CRS. Defense-in-depth against future data-integration mistakes — a 3857 raster (or any non-equal-area CRS) accidentally introduced would crash loudly with the offending path named, rather than silently producing wrong area math.

### Per-city configuration

City configs live in `config.CITIES` as a dict-of-dicts. Each entry declares input file paths, the canonical CRS, biophysical-table filenames, per-city scalars (`uhi_max_c`, `design_storm_inches`, `una_demand_m2_per_capita`, `una_search_radius_m`, `una_decay_function`), and `available: bool`. The sidebar selector populates from `available=True` entries. `_load_city_runtime_state` reads the active city's config dict and produces the matching `CityState`. Adding a new city is a config entry + the on-disk input files; no code changes needed for the loader path.

Active cities in the UI today: **Minneapolis, MN** (downtown extent) and **San Antonio, TX** (Bexar bbox). **Minneapolis Full, MN** is implemented in the codebase with `available=False` (hidden — see §10).

---

## 4. Scenario evaluation pipeline

`evaluate_scenario(pct_converted, gi_pct, ff_pct, hd_pct, placement_strategy, seed, ...)` is the canonical entry point. It returns a result dict consumed by every metric card, the tradeoff chart, the map view, the export bundle, and the surrogate's training data. The pipeline:

1. **Build the converted-pixels set.** Sample from `CONVERTIBLE_PIXELS` (developed LULC minus buildings minus roads) using the chosen `placement_strategy`'s suitability surface (the focused strategies use canonical InVEST quantities — `Q_{p,i}` for flood-focused, HMI for cooling-focused, `urban_nature_supply_percapita` for undersupply-focused; see REFERENCE.md "Placement strategies"). The "Target High Heat-Exposure Areas" sidebar toggle, when on, additionally weights NLCD 23 > 22 > 21.
2. **Produce the scenario LULC raster.** For MN: direct NLCD swap. For SA: compound-context-preserving conversion via `lulc_crosswalk.csv` with documented default fallback lucodes (`DEFAULT_FF_LUCODE`, `DEFAULT_GI_LUCODE`, `DEFAULT_HD_LUCODE`) when a (NLUD, tree-canopy) tuple has no matching compound row.
3. **Run the per-pixel biophysical models.** UFR (Curve Number runoff over the per-city design storm); UCM (canonical Heat Mitigation Index `max(CC_local, CC_park)`); UNA (canonical 2SFCA per-capita supply vs demand); UMH (canonical edge-corrected buffer-mean NDVI exposure + `PC = (1 − RR) × BIR × POP`); Carbon (SA: four-pool stock delta; MN: per-cover annual rate proxy). All implementations are numpy ports of `natcap.invest.*` — see §6 and REFERENCE.md "Official InVEST alignment" for the per-model parity status.
4. **Aggregate to scenario-level outputs.** Means, sums, and derived per-target counts (including the SA fallback-pixel counts) flow into the return dict.
5. **Return a result dict.** ~27 keys covering both the per-pixel-derived metrics and the slider-sensitive metrics.

**Lookup-overlay safety contract.** When Layer 2's lookup table is in use (High resolution mode), the lookup hit short-circuits ~17 of the 27 fields; the remaining **10 fields** are recomputed live on every slider interaction (the scenario LULC rasters, NDVI, food, carbon, dollar metrics, cost). This is a runtime invariant — the *why* (the schema-vs-slider-sensitivity gap that drove the overlay) lives in **DESIGN_NOTES §4**; §5 below references the field list without restating the rationale.

---

## 5. Three computational layers

The three layers exist for different reasons: realism (Layer 1), speed (Layer 2), wider search space (Layer 3).

### Layer 1 — Raster simulations

**What it does.** Per-pixel biophysical computation for the chosen scenario. The entry point is `evaluate_scenario()` (see §4); the per-model implementations are numpy ports of `natcap.invest.*` (UCM, UFR, UNA, UMH, Carbon). All layers above depend on Layer 1; without per-pixel calculation, the prototype has no ground truth.

**Speed cost.** A single SA `evaluate_scenario()` call takes ~0.9 seconds (on the 1713 × 1984 grid). MN downtown takes ~0.03 seconds. Layer 1 alone is too slow for interactive slider response on SA — Layer 2 closes that gap.

**Validation.** Validated against canonical `natcap.invest.*.execute()` where comparable inputs exist. See NATCAP_ALIGNMENT.md for the per-metric parity numbers.

### Layer 2 — Lookup table + live refresh

**What it does.** Pre-computes Layer 1 across the full slider space and stores the results in the active city's `data/scenarios_dense_<city>.csv` (per `dense_scenarios_file` in the CITIES config) and in `compute_lookup_table`'s in-memory cache. At runtime the UI looks up the user's current slider position and returns the cached raster aggregates instantly.

**Three model-quality modes** are selectable from the sidebar (`Model quality`). They differ in what gets precomputed at startup; **slider response always runs `evaluate_scenario()` live in every mode** — the modes vary in whether a precomputed lookup is also available to short-circuit the expensive raster aggregates.

| Mode | Startup precomputation | Slider response |
|---|---|---|
| **Fast prototype** (default) | ~90-scenario training grid via `compute_scenario_grid(step_pct=10, step_alloc=25)` | Live `evaluate_scenario()` only — no lookup table built |
| **Balanced** | Denser ~726-scenario CSV per city (`data/scenarios_dense_<city>.csv`, precomputed offline via `precompute_scenarios.py --city '<city>' --step-pct 5 --step-alloc 10`) | Live `evaluate_scenario()` only — no lookup table built |
| **High resolution** | Full 2,541-entry lookup table via `compute_lookup_table` — 25–50 min on SA's 3.4 M-pixel grid; rebuilt only when `SCENARIO_SCHEMA_VERSION` invalidates the cache | Lookup hit for cached raster aggregates **plus** live `evaluate_scenario()` to refresh the ~12 live-overwrite fields |

In all three modes the precomputed scenario set (90, 726, or 2,541) feeds Layer 3's surrogate training and the tradeoff chart.

**Live-overwrite fields (the safety-contract list).** On each slider interaction in High resolution mode, the lookup hit provides cached values for most metrics, but the following **10 of 27 fields** are recomputed live by `evaluate_scenario()` and overwrite the cached values:

- `scenario_lulc` (for the Map View tab)
- `scenario_lulc_ucm` (the UCM-view raster — compound for SA, NLCD for MN — so downstream `compute_per_tract_summary` can re-run HMI helpers on the right lucode-space view)
- `food_mln_lbs`, `people_fed`
- `mean_ndvi`
- `carbon_tons_co2`, `carbon_value_usd` (per-city semantics — annual flow MN, one-time stock SA)
- `flood_damage_avoided_usd`, `cooling_energy_savings_usd`
- `total_cost_mln` (recomputed via `compute_cost` from current cost-slider values)

The overwrite covers metrics that depend on per-rerun state (fresh rasters, cost sliders, carbon-rate sliders); fields loaded from the lookup (`nature_access_pct`, `mean_hm`, `flood_reduction`, `runoff_acre_feet`, the MH metrics, the `n_wet`/`n_for`/`n_hd` counts, etc.) are schema-current because `SCENARIO_SCHEMA_VERSION` is part of the cache key — no defensive overwrite needed for those. When any non-random placement strategy is active, the lookup table is bypassed entirely and all metrics run live. The decision rationale lives in DESIGN_NOTES §4.

**Cache invalidation via `SCENARIO_SCHEMA_VERSION`.** Bumped whenever the metric schema or one of the upstream pipeline pieces changes. Current value: **27**. The constant is hashed into the Streamlit `@st.cache_data` keys for `compute_scenario_grid`, `compute_lookup_table`, and the surrogate training cache, so a bump forces recomputation. Per-city caches don't collide because the cache key also includes the city's data directories and filename arguments. Per-bump rationale: `docs/archive/HISTORY.md` "Schema version log".

### Layer 3 — Surrogate model

**What it does.** A random-forest regressor trained on the precomputed scenario set selected by the active model-quality mode. Predicts metric outcomes for arbitrary continuous scenario inputs (any `pct, GI%, FF%`). Lives in `surrogate.py`; the trained estimator is cached by `_cached_train_surrogate` (see §9).

**Used by the "Find Best Scenario" panel.** The user specifies minimum flood / cooling / food / carbon constraints; the optimizer samples ~10,000 random `(pct, GI%, FF%)` candidates, predicts outcomes with the RF surrogate, filters to those meeting all minimums, computes the Pareto front, de-duplicates near-identical points, and returns up to 5 top suggestions ranked by a balanced score (`flood/100 + HMI/1.1 + food/MAX_FOOD`).

**The three knobs** (surrogate-side; the per-mode optimizer candidate count of 10,000 is independent):

| Knob | Setting | Notes |
|---|---|---|
| Training scenarios | Fast prototype 90 / Balanced ~726 / High resolution 2,541 | Selected by `model_quality` mode (see Layer 2) |
| Random Forest tree count | Fast prototype 100 / Balanced 200 / High resolution 300 | `SURROGATE_TREES` constant — intentionally hidden from the UI |
| Optimizer candidate count | 10,000 | Independent of mode; sampled at optimization time |

**Outputs.** The surrogate predicts six metrics: `flood_reduction`, `mean_hm`, `food_mln_lbs`, `runoff_acre_feet`, `carbon_tons_co2`, `nature_access_pct`. UMH metrics (`preventable_mh_cases`, `avoided_mh_cost_usd`) are computed deterministically from NDVI exposure and are not RF targets.

**Spatial-geometry blindness.** The surrogate inputs only `(pct, GI%, FF%)` — it cannot see the geometry of where pixels are placed. Nature Access predictions in particular capture only the headline trend, not the per-pixel placement effect. The optimizer's diamonds + uncertainty bars on the tradeoff chart are predictions; treat them as candidates worth verifying. When the user clicks **Apply** on a suggestion, the displayed metric cards reflect a full-raster evaluation by Layer 1 (not a surrogate prediction).

**Uncertainty bands** come from 10th / 90th percentile across RF trees. When all trees agree the bars are tight; when they disagree the bars grow — signal that the model is extrapolating into territory it has seen little of during training.

**Relationship to NatCap ROOT.** The surrogate is a fast app-specific approximation, distinct from ROOT's true linear-programming multi-objective optimization at the spatial-decision-unit level (max Σ wᵢ Vᵢₛₐ xₛₐ with random or user-specified weights, producing production possibility frontiers and agreement maps). See §10.

---

## 6. Validation and provenance layer

Two rendered surfaces:

- **Per-card validation badge** — `_render_validation_caption(col, metric_name, scenario_context, explicit_status=None)` (in `app.py`); the badge state is computed by `natcap_validation.render_validation_badge(metric, scenario_context, explicit_status=None)`. Renders as an inline `st.caption` under each metric card.
- **Per-scenario provenance header** — `_render_scenario_provenance_header(provenance, scenario_label, ...)`; the (source label, validation line, color) triple is sourced from `_PROVENANCE_HEADER_INFO` keyed by the four `PROVENANCE_*` constants from §2. Renders as a prominent header above the metric grid.

**Authoritative taxonomy lives in NATCAP_ALIGNMENT.md §2.** The four-state per-card badge vocabulary (`NatCap published value` / `≈ NatCap method` / `≈ Aligned method` / `Prototype`) plus the per-metric-evidence nuance (temperature can cite measured per-pixel parity; carbon is methodology adoption without per-pixel parity) and the per-metric × per-context switch (NatCap published value fires only in the fixed-scenario reference view) are documented there. ARCHITECTURE §6 covers the *components* — what code renders the badges and where they're wired — and does NOT restate the taxonomy. See NATCAP_ALIGNMENT §2 for the badge meanings, REFERENCE §4 for the user-facing explanation, and DESIGN_NOTES §8 for the design rationale.

**Per-card badge wiring.** Every metric-card column gets a `_render_validation_caption(col, ...)` call after its `st.metric(...)` call. The call site supplies either a CSV-derived `metric_name` (for metrics tracked in `data/sa/natcap_reference_outputs.csv`) or an `explicit_status` string for non-CSV cards (runoff, NDVI, cost-effectiveness ratios, carbon-$ on SA, etc.).

**Validation in the export bundle.** The bundle's `metadata.json → validation` block records each model's state using a **two-state taxonomy distinct from the per-card badge's four states**: `validated` (per-pixel parity measured against canonical `natcap.invest.*.execute()` — emitted for UCM, UNA, UMH) or `methodology_aligned` (canonical method, no per-pixel parity check — emitted for UFR, Carbon). Each entry includes the reference InVEST version + a per-model notes string sourced from NATCAP_ALIGNMENT.md. A downstream user opening an exported bundle reads the validation context without re-deriving it from these docs. See §7.

---

## 7. Export for InVEST

The sidebar's **Export for InVEST** section (SA only for v1) packages the currently-displayed scenario as a runnable canonical InVEST 3.19.0 input bundle. The builder is `export_invest_bundle.build_invest_bundle(spec: BundleSpec)`; it returns zip bytes and is called from a Streamlit download button.

**Bundle structure** (verified — all five InVEST 3.19.0 urban models execute cleanly on the SA baseline bundle):

```
ecosystem_explorer_export_<city_slug>_<scenario_id>_<timestamp>.zip
├── README.md                                            (how-to-run; bundle-relative paths)
├── metadata.json                                        (provenance, generator, per-model validation)
├── inputs/
│   ├── prototype/                                       (rasters on the 30 m EPSG:5070 prototype grid)
│   │   ├── scenario_lulc_evaluated_30m_5070.tif         (compound — UCM / UNA / Carbon-alt)
│   │   ├── baseline_lulc_evaluated_30m_5070.tif
│   │   ├── scenario_lulc_nlcdtree_30m_5070.tif          (NLCD×tree — UFR)
│   │   ├── baseline_lulc_nlcdtree_30m_5070.tif
│   │   ├── scenario_ndvi_30m_5070.tif                   (UMH ndvi_alt)
│   │   └── baseline_ndvi_30m_5070.tif                   (UMH ndvi_base)
│   ├── shared/                                          (population, ET, soil HSG, AOIs, prevalence vectors)
│   └── biophysical/                                     (UCM / UNA / Carbon compound tables; SA NLCD×tree CN table)
└── args/prototype_grid/                                 (one args.json per InVEST urban model)
    ├── urban_cooling_args.json
    ├── urban_nature_access_args.json
    ├── urban_flood_risk_mitigation_args.json
    ├── carbon_args.json
    ├── urban_mental_health_depression_args.json         (effect_size = RR per 0.1 NDVI, depression)
    └── urban_mental_health_anxiety_args.json
```

**`metadata.json` fields** (from the `BundleSpec` dataclass + serializer):

- `format_version` — bundle schema version
- `prototype_git_commit` — short SHA of the prototype's HEAD at export time
- `scenario_schema_version` — current `SCENARIO_SCHEMA_VERSION`
- `export_timestamp_utc`
- `city`
- `scenario` — block with `provenance` (one of the four `PROVENANCE_*` constants from §2), `scenario_name`, `pct_converted`, `gi_pct`, `ff_pct`, `hd_pct`, `placement_strategy`
- `generator` — block with `type` (matching the provenance) and generator-specific parameters
- `raster_lineage` — per-raster source path inside the bundle + the on-disk source it was derived from
- `validation` — per-model state (one entry per UCM / UNA / UFR / Carbon / UMH), each emitting one of two values: `validated` (UCM, UNA, UMH — per-pixel parity measured) or `methodology_aligned` (UFR, Carbon — canonical method, no per-pixel parity check). This is the **export-bundle's two-state taxonomy** — distinct from the per-card badge's four states (see §6)
- `model_availability` — per-model `available: bool` with `reason` string for the cases where it isn't (e.g. for fixed alternatives: *"NatCap did not ship a compound LULC for this fixed scenario; only flood is exported."*)

**Running canonical InVEST on the bundle.** From the bundle root, e.g.:

```bash
python -c "import json; from natcap.invest import urban_cooling_model as m; m.execute(json.load(open('args/prototype_grid/urban_cooling_args.json')))"
```

Substitute the module and args path for each of the five models. The bundle's README documents the scenario-vs-baseline delta pattern (Carbon runs both LULCs in one execution; the others run each LULC separately).

**Export ≠ already-validated.** `metadata.json`'s `validation` block records the prototype's own measured parity against canonical InVEST per model. Running canonical `execute()` on the bundle produces fresh canonical outputs which the user can then compare against the prototype's reported card values. Validation travels with the bundle; the bundle isn't itself a validation result.

---

## 8. Major modules and responsibilities

**Root (8):**

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, sidebar state, metric cards, `evaluate_scenario` and biophysical helpers, `CityState` loader, scenario-source plumbing, tradeoff/map render |
| `config.py` | `CITIES` dict (per-city paths + parameters) and global cost defaults; read-only |
| `surrogate.py` | Random Forest surrogate + optimizer (Streamlit-agnostic) |
| `natcap_scenarios.py` | NatCap fixed-scenario loader, `flood_reduction_from_nlcd_tree` helper, `PROVENANCE_*` taxonomy (Streamlit-agnostic) |
| `natcap_validation.py` | `render_validation_badge`, `published_delta`, reference-CSV reads (Streamlit-agnostic) |
| `export_invest_bundle.py` | InVEST 3.19.0 export-bundle assembly (rasters + biophysical tables + per-model args.json + metadata.json + README, zipped) |
| `verify_baselines.py` | Baseline regression test — snapshots `evaluate_scenario` for 40 (city × scenario × strategy) combinations against committed JSON snapshots; the pre-commit gate |
| `precompute_scenarios.py` | Offline dense-CSV builder for Balanced-mode training set; stubs Streamlit and reuses `evaluate_scenario` |

**`validation/` (5):** `compare_carbon_invest.py`, `compare_ucm_invest.py`, `compare_umh_invest.py`, `compare_una_invest.py`, `verify_cooling.py` — canonical-InVEST parity comparators.

**`diagnostics/` (5):** `compare_una_lulc.py`, `analyze_placement_diagnostic.py`, `placement_strategy_diagnostic.py`, `check_expanded_coverage.py`, `validate_surrogate_predictions.py`. (`validate_scenarios.py` retired — see HISTORY.)

**`scripts/data/` (17):** `download_*`, `process_*`, `clip_worldpop.py`, `extract_natcap_reference_outputs.py` — data-pipeline scripts. See DATA_INVENTORY.md.

**Why numpy, not canonical `natcap.invest` at runtime.** The prototype implements UCM/UFR/UNA/UMH/Carbon as numpy ports instead of calling `natcap.invest.*.execute()` directly. One-sentence gist: it's the only way to hit interactive slider response, since `natcap.invest` only ships `execute()` (full disk I/O round-trip per call) — no `execute_from_arrays` exists. Full rationale parked for DESIGN_NOTES §6 (model-evaluation design).

**Testing.** `verify_baselines.py` is the regression-test gate: it runs all 40 (city × scenario × strategy) baselines and diffs every output field against committed JSON snapshots in `tests/baselines/`. Run before commits that could shift outputs. `--update` re-snapshots after intentional methodology changes. Validation comparators in `validation/` are separate end-to-end checks against canonical `natcap.invest.*.execute()`.

---

## 9. Caching and performance

Six functions in `app.py` carry Streamlit cache decorators: **2 `@st.cache_resource` + 4 `@st.cache_data`**. They split across three caching tiers (in-memory Streamlit cache, on-disk persistent artifacts, lazy live compute) with distinct invalidation rules.

### `@st.cache_resource` (2 functions)

For shared non-serializable objects (NamedTuples, model instances, large arrays).

| Function | Decorator site | Cache behavior |
|---|---|---|
| `_load_city_runtime_state(city_key)` | `app.py:2254` — `@st.cache_resource(max_entries=1, show_spinner="Loading city data — first interaction may take a minute…")` | Returns the `CityState` NamedTuple. **`max_entries=1` is load-bearing — see "city-switch eviction" below.** |
| `_cached_train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling, scenario_signature, ...)` | `app.py:2947` | RF surrogate. Cached on the scenario-df identity + paths + mode signature; the underscored `_scenario_df` is skip-hashed. Switching `Model quality` mode within a city rebuilds the surrogate. |

### `@st.cache_data` (4 functions)

For serializable, deduplicated computation results (DataFrames, dicts).

| Function | Decorator site | Cache key includes |
|---|---|---|
| `load_lulc_crosswalk(crosswalk_path, default_ff, default_gi, default_hd)` | `app.py:579` | path + default lucodes — SA compound LULC crosswalk loader |
| `load_data(data_dir_flood, data_dir_cooling, cn_table_file, cooling_table_file, lulc_file, soil_file, cooling_lulc_file)` | `app.py:747` | the path arguments — initial raster + biophysical-table load |
| `compute_scenario_grid(_state, city_key, data_dir_flood, data_dir_cooling, step_pct=10, step_alloc=25, ...)` | `app.py:2029` | `city_key` + paths + step args — Fast-prototype training grid |
| `compute_lookup_table(_state, city_key, data_dir_flood, data_dir_cooling, schema_version=SCENARIO_SCHEMA_VERSION)` | `app.py:2078` | `city_key` + paths + `SCENARIO_SCHEMA_VERSION` — High-res lookup |

The underscored `_state` arguments are skip-hashed (Streamlit convention for passing non-trivially-hashable objects).

### City-switch behavior — `max_entries=1` eviction

When the user picks a different city in the sidebar:

- **`_load_city_runtime_state(city_key)`** — `max_entries=1` means only the active city's `CityState` lives in cache. **Every city switch evicts the prior city's state and rebuilds (~10–30 s on SA — the spinner string fires).** A round-trip MN → SA → MN is *two* rebuilds of MN, not "instant on the second time." This is the explicit memory-fit tradeoff: keep one heavy NamedTuple resident, accept rebuild cost on every switch, fit inside Streamlit Cloud's 1 GB worker ceiling. Persisted on-disk artifacts under `data/precomputed/<city>/` (`nature_distance_<lucode>.npy`) bypass live re-compute on the rebuild path.
- **`load_data(...)`** — cached per (paths). Cities have disjoint paths → disjoint cache entries. **Not** evicted by `cache_resource` policy (different decorator class). Survives city switches.
- **`load_lulc_crosswalk(...)`** — survives city switches (SA-only; called once per session typically).
- **`compute_scenario_grid` + `compute_lookup_table`** — cached per `(city_key + paths + schema_version)`. Survive city switches per the `cache_data` policy.
- **`_cached_train_surrogate`** — cached per `(scenario_df identity + paths + signature)`. Switching model-quality modes within a city rebuilds the surrogate.

### Cache invalidation triggers

Four distinct triggers:
1. **City switch** — evicts `CityState` (the only `max_entries=1` cache); other caches survive.
2. **`Model quality` mode switch** — rebuilds `_cached_train_surrogate`; `compute_scenario_grid` / `compute_lookup_table` already-cached entries survive (the new mode just selects which entry feeds the surrogate).
3. **`SCENARIO_SCHEMA_VERSION` bump** — invalidates `compute_scenario_grid` + `compute_lookup_table` across all cities (schema_version is a cache key).
4. **Slider edit** — no cache invalidation; the live-overwrite path (§5 Layer 2) recomputes 10 fields.

### Three caching tiers

| Tier | Mechanism | Invalidation |
|---|---|---|
| Streamlit in-memory cache | the six functions above | the four triggers |
| Persistent on-disk artifacts | `data/precomputed/<city>/nature_distance_<lucode>.npy` | delete the directory; not Streamlit-managed |
| Lazy live compute | `evaluate_scenario` slider re-runs | every interaction |

---

## 10. Known architectural boundaries

### What the layers do not do

- **Layer 1 — Raster simulations.** Per-pixel canonical math, but no per-building T_air aggregation for UCM (the prototype computes the dollar metric per pixel rather than InVEST's per-building 600 m blending radius). Also no native-encoded UFR for SA's fixed alternatives — see §2 SA-only scope.
- **Layer 2 — Lookup table coverage gap.** The lookup is *grid-shaped* — it covers fixed `(scenario, pct, strategy)` tuples. The Find Best Scenario tab needs a much higher-dimensional continuous space than the grid can represent. That's Layer 3.
- **Layer 3 — Surrogate blindness.** The surrogate inputs only `(pct, GI%, FF%)` — it cannot see the spatial geometry of where pixels get placed. Nature Access predictions in particular are spatial-trend-only. Optimizer suggestions are candidates, not validated outputs — Apply them to see the full canonical-engine evaluation.

### Why not ROOT

NatCap's ROOT performs true linear-programming multi-objective optimization at the spatial-decision-unit (SDU) level: `max Σ wᵢ Vᵢₛₐ xₛₐ` with random or user-specified weights, producing production possibility frontiers and agreement maps. The Ecosystem Explorer's surrogate is a much simpler RF approximation over three sliders — it accelerates exploration but doesn't compute Pareto frontiers in the ROOT sense or agreement maps. ROOT remains a deferred reference point; see DESIGN_NOTES §11 for the deferred-approach rationale.

### Hidden cities

`'Minneapolis Full, MN'` is live in `CITIES` with `available=False` so it does NOT appear in the sidebar selector. The full-city extent uses OSM-only buildings without per-type codes, which prevents Cooling Energy Savings and Flood Damage Avoided cards from computing their dollar metrics. Re-expose by flipping `available=True` once typed-building data exists for the expanded area — all upstream pipeline, rasters, and verified baselines are still in place.

### MN flood damage table vs SA "Flood Volume Reduction"

For MN downtown the dollar Flood Damage Avoided card computes against `Damage_loss_table_MN.csv`. For SA the same card relabels to "Flood Volume Reduction" (no dollar figure) — NatCap's Vibrant Land report used InVEST UFRM for SA but explicitly did not enable damage valuation, and the prototype matches. See REFERENCE.md §6 Flood Damage Avoided / Flood Volume Reduction.

---

## 11. Future architecture hooks

The current `evaluate_scenario` signature is extensible. A future addition: a `selected_region_mask: np.ndarray | None` parameter that intersects with `CONVERTIBLE_PIXELS` to constrain candidates to a user-drawn polygon or a per-tract opt-in mask. The conceptual seam already exists — `candidate_pixels = CONVERTIBLE_PIXELS ∩ selected_region_mask` — but no UI surface yet drives it.

This is the one architectural hook worth noting today; other future directions (additional InVEST models, a stratified-impervious-intensity placement option, real CDC/ATSDR Heat Vulnerability Index integration, AlphaEarth NDVI replacement) live in `DESIGN_NOTES` and `OPEN_QUESTIONS` because they're rationale + status rather than architectural seams.
