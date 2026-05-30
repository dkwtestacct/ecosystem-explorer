# ARCHITECTURE.md refresh — content map

**Audience:** Internal
**Status:** In review — `[VERIFY]` 1–8 resolved; awaiting approval before rewrite
**Use this for:** Driving the ARCHITECTURE.md §1–11 refresh, anchor-stability discipline, and the link-fix pass
**Do not use this for:** Current architecture description — this doc is scaffolding for the refresh, not ARCHITECTURE itself
**Source of truth for:** The current → target mapping and locked resolutions for the ARCHITECTURE refresh

---

**Transient scaffolding.** Delete after the refresh + link-fix passes land and the inbound-ref inventory is exhausted. Same discipline as `REFERENCE_CONTENT_MAP.md`.

**Purpose:** the current → target mapping for the ARCHITECTURE refresh, so nothing is dropped and the new structure absorbs the right content from the in-flight REFERENCE work. Sequence: CC finalizes the `[VERIFY]` items against the code → we approve → CC rewrites §1–11 + link-fix + verify, stop-and-report before commit.

The current doc is already structural and current-state — it just predates the scenario-sourcing, validation/provenance, and export subsystems. This refresh modernizes it to match the system that now exists, and it is the doc that **owns absorbing REFERENCE's Layer 2/3 mechanical depth** (REFERENCE's Commit A was dropped; this refresh does that work).

---

## Target structure

1. System overview
2. Scenario sources
3. Runtime data model
4. Scenario evaluation pipeline
5. Three computational layers
6. Validation and provenance layer
7. Export for InVEST
8. Major modules and responsibilities
9. Caching and performance
10. Known architectural boundaries
11. Future architecture hooks

Header keeps the 5-field block: **Use this for:** "Understanding how the app is organized before reading code" / **Do not use this for:** "Metric definitions (→ REFERENCE.md), design rationale (→ DESIGN_NOTES.md), or collaboration history" / **Source of truth for:** "System components and data flow."

---

## Global editorial rules

- **Current system map, not narrative history.** Strip brief numbers (e.g. the "Brief 27" in CRS handling → "the SA compound LULC source is reprojected from EPSG:3857 to EPSG:5070 at prep time"). This is the doc someone reads *before opening app.py*.
- **Stay in lane.** Rationale → DESIGN_NOTES; metric meanings → REFERENCE; per-model MAE/alignment status → NATCAP_ALIGNMENT; data-file inventory → DATA_INVENTORY; chronology → HISTORY. Keep only enough example to make the structure understandable.
- **Validation/provenance vocabulary (§6) is shared.** Use the locked two-surface vocabulary verbatim — per-card badge (`NatCap published value` / `≈ NatCap method` / `≈ Aligned method` / `Prototype`) and the per-scenario provenance header (the four source/validation pairs). Must match REFERENCE §4 and DESIGN_NOTES §8 word-for-word; the three docs do different jobs with the same words (ARCH = system component; REFERENCE = user meaning; DESIGN_NOTES = design rationale).

---

## Current → target mapping

| Current section | → Target | Transformation / notes |
|---|---|---|
| Header status block | Header | Update Use/Don't-use/Source-of-truth as above. |
| "For deeper detail" note | Header / §1 | Fold into the header cross-refs. |
| At a glance (3-layer diagram + prose) | §1 | Keep the orienting prose. Replace the old 3-layer ASCII with the new full-system diagram (UI → scenario-source layer → raster/provenance layer → three compute layers → outputs). Layer detail moves to §5. |
| CRS handling | §3 | The equal-area invariant + prep-time reprojection + the `_assert_raster_crs` runtime guard. Load-bearing — give it explicit space, don't let it dissolve. Strip the brief number. |
| Layer 1 — Raster simulations | §4 + §5 + §6 | `evaluate_scenario()` flow → §4 pipeline. Per-pixel InVEST models → §5 Layer 1. "Validation status" (MAE) → §6 gist + NATCAP_ALIGNMENT for the numbers. "Speed cost" → §9. |
| Layer 2 — Lookup table | §5 (Layer 2) | **Absorbs REFERENCE depth** (see below): model-quality modes, the live-overwrite fields, schema mechanics. "Coverage gap" → §10. "Why / speed" → §9. |
| Layer 3 — Surrogate model | §5 (Layer 3) | **Absorbs REFERENCE depth**: smart-search controls, deep surrogate/optimizer mechanics. "Limitations" → §10. "Training" (cached) → §9. "Why not ROOT" → §10 (brief). **Resolve the training-count inconsistency** — see `[VERIFY] 4` (resolved below). |
| Why numpy, not canonical natcap.invest | **DESIGN_NOTES §6** + §6 gist | Rationale (latency, no `execute_from_arrays`) → DESIGN_NOTES §6 (Model evaluation design); park for the DESIGN_NOTES refresh, ARCH keeps a one-sentence gist + forward cross-ref. "Validation not replacement" + the `compare_*_invest.py` pointer → §6. MAE numbers → NATCAP_ALIGNMENT. |
| Data flow (diagram) | §1 / §4 | Superseded by the new §1 system diagram + the §4 pipeline description. Data-on-disk pointer stays → DATA_INVENTORY. |
| Per-city configuration | §3 + §8 | `CITIES` dict structure → §3/§8. Active cities + MN Full dormant (`available=False`) → §3 (or note in §10). |
| Testing | §8 (or small standalone) | `verify_baselines.py`, 40 baselines, regression gate. Keep — it's real architecture (the gate). |
| Where to read next (table) | §8 + header | Upgrade into the §8 module-responsibility map; keep the cross-ref table. |

**New sections (mostly new content — verify against code):** §2 Scenario sources, §6 Validation and provenance (partly from scattered current content), §7 Export for InVEST, §9 Caching and performance (consolidates scattered speed/cache mentions), §10 Known architectural boundaries (consolidates the Layer limitations + coverage gap + why-not-ROOT), §11 Future hooks (region-selection mask — keep minimal/speculative: `candidate_pixels = convertible_pixels ∩ selected_region_mask`).

---

## Depth absorbed IN — from REFERENCE (into §5)

REFERENCE's rewrite strips this depth to a gist; this refresh pulls the **full** content into §5 from the pre-REFERENCE-rewrite version in git (REFERENCE lines **335–371** and **683–749**, also itemized in `REFERENCE_CONTENT_MAP.md`'s ARCHITECTURE-split section):

- Model-quality modes table (Fast prototype / Balanced / High resolution) — startup precompute + slider response per mode → §5 Layer 2 / §9.
- The live-overwrite field list + `SCENARIO_SCHEMA_VERSION` mechanics → §5 Layer 2 (+ §9 cache invalidation).
- Smart Scenario Search controls + the surrogate output list + the "relates to high-resolution lookup" paragraph → §5 Layer 3.
- "Why a surrogate / how it thinks" + optimizer mechanics → §5 Layer 3.
- The training-scenarios / trees / optimizer-candidates "three knobs" framing → §5 Layer 3.

---

## Content moving OUT of ARCHITECTURE

- **"Why numpy" rationale → DESIGN_NOTES §6.** Parked for the DESIGN_NOTES refresh; ARCH keeps a gist + forward cross-ref (same rightward flow REFERENCE→ARCHITECTURE used). The full content lives in git / this map until the DESIGN_NOTES pass places it.
- **Per-model MAE / alignment numbers → NATCAP_ALIGNMENT.** ARCH §6 says "validated against canonical InVEST; see NATCAP_ALIGNMENT for per-model diffs," not the figures.
- **Data-file paths → DATA_INVENTORY** (already cross-reffed; keep ARCH light on paths).

---

## Resolved findings (`[VERIFY]` 1–8 — see below for each)

The eight `[VERIFY]` items are resolved against the live code in the section that follows. Findings flagged ready for review.

### `[VERIFY] 1` — Scenario-source taxonomy + `PROVENANCE_*` constants

**Source: `natcap_scenarios.py:18–28` (PROVENANCE_* constants); `app.py:3324–3354` (`_PROVENANCE_HEADER_INFO`); `app.py:4322–4328`, `4065`, `4076`, `4093` (provenance assignment).**

There are **four** PROVENANCE constants in code, not five — they live in `natcap_scenarios.py` and are re-exported as `eib.PROVENANCE_*` (via `export_invest_bundle.py` re-import):

```
PROVENANCE_BASELINE       = "baseline"
PROVENANCE_NATCAP_FIXED   = "natcap_fixed_scenario"
PROVENANCE_EXPLORER       = "explorer_generated"
PROVENANCE_OPTIMIZER      = "optimizer_suggested"
```

Each maps to a (Source-label, Validation-line, color) triple in `_PROVENANCE_HEADER_INFO`:

| Constant | Source label | Validation line | Color |
|---|---|---|---|
| `PROVENANCE_BASELINE` | `Baseline` | engine verified vs canonical InVEST; absolute NatCap citywide figures not reproduced | blue |
| `PROVENANCE_NATCAP_FIXED` | `NatCap published reference` | displayed from NatCap output; exact scenario raster / aggregation not available | green |
| `PROVENANCE_EXPLORER` | `Explorer-generated` | canonical engine verified; scenario not NatCap-published | blue |
| `PROVENANCE_OPTIMIZER` | `Surrogate-suggested` | engine-validated; full-raster evaluated — exploratory candidate for further validation | blue |

**Runtime assignment (`app.py:4322–4328`, `4065`, `4076`, `4093`):**
- `_scen_provenance = PROVENANCE_BASELINE` when `results['pct_converted'] == 0`
- `PROVENANCE_OPTIMIZER` when `st.session_state.get("applied_from_optimizer")` is True
- `PROVENANCE_EXPLORER` otherwise

**Saved vs exported are NOT separate provenance values.** A "saved scenario" carries whatever provenance its source had at save time (recorded as `saved["provenance"]` per Brief #5; `app.py:5417` backfills older saves to BASELINE/EXPLORER from `pct_converted`). An "exported scenario" carries the active scenario's provenance into the export bundle's `metadata.json`.

**SA-only sources.**
- `PROVENANCE_NATCAP_FIXED` is **SA-only** in practice. The fixed-scenario taxonomy lives in `SA_NATCAP_FIXED_SCENARIOS` (`natcap_scenarios.py:30+`). MN has no NatCap fixed-scenario corpus; `_render_fixed_scenario_reference_view` is gated by `selected_city.startswith("San Antonio")` at `app.py:5354`.
- For SA fixed scenarios, the rasters NatCap provided are **flood-only** for the six fixed alternatives (FF_20ac, FF_40ac, FF_MAX, UA_20ac, UA_40ac, UA_MAX) — the compound LULC for those alternatives wasn't shipped; see `OPEN_QUESTIONS.md`. The export bundle for those alternatives is gated accordingly (`export_invest_bundle.py:_build_fixed_alternative_bundle` exports flood-only with the compound-model args marked unavailable in `metadata.json`).

**§2 diagram should match these four constants exactly.** Note SA-only PROVENANCE_NATCAP_FIXED and flood-only export gating for fixed alternatives.

### `[VERIFY] 2` — Runtime data model / `CityState`

**Source: `app.py:828–880` (`CityState` NamedTuple definition); `app.py:884` (`_load_city_runtime_state`); `app.py:996+` (module-level alias rebinding after loader call).**

The container is a **`CityState` NamedTuple** populated by **`_load_city_runtime_state(city_key: str) → CityState`**, decorated `@st.cache_resource`.

**Fields** (full list from the NamedTuple definition):

```python
class CityState(NamedTuple):
    # Rasters
    cooling_lulc:           np.ndarray        # main LULC raster (compound for SA)
    cooling_lulc_compound:  np.ndarray | None # the compound LULC (SA only — None for MN)
    soil_group:             np.ndarray
    ndvi_baseline:          np.ndarray        # synthetic NDVI proxy
    pop_count_raster:       np.ndarray        # Census 2020 population, float32
    et_raster:              np.ndarray        # reference ET, resized to LULC grid
    buildings_mask:         np.ndarray        # binary; rasterized building footprints
    buildings_type:         np.ndarray        # InVEST type codes 0/1/2/3 per pixel
    roads_mask:             np.ndarray        # binary; rasterized road network
    tracts_index:           np.ndarray        # per-pixel tract or block-group id
    # Lookup arrays (compound-keyed for SA; NLCD-keyed for MN)
    cn_a_arr, cn_b_arr, cn_c_arr, cn_d_arr:  np.ndarray   # CN per lucode × soil group
    shade_arr, kc_arr, albedo_arr:            np.ndarray  # UCM biophysical lookups
    urban_nature_arr:                         np.ndarray  # UNA biophysical
    c_above_arr, c_below_arr, c_soil_arr, c_dead_arr: np.ndarray  # Carbon four-pool
    compound_to_nlcd:        np.ndarray | None  # compound → NLCD reduction (SA only)
    compound_to_nlcd_tree:   np.ndarray | None  # NLCD × tree for SA flood
    compound_after_ff, compound_after_gi, compound_after_hd: np.ndarray | None  # SA conversion lookups
    # Baseline rasters (precomputed once)
    baseline_hmi_raster:     np.ndarray       # canonical HMI baseline
    baseline_ne_raster:      np.ndarray       # UMH NE baseline
    baseline_access_score:   np.ndarray       # UNA per-capita supply baseline
    baseline_runoff_pixel:   np.ndarray       # per-pixel runoff under baseline
    convertible_mask:        np.ndarray       # developed pixels minus buildings/roads
    # Distance fields (precomputed at startup; persisted under data/precomputed/<city>/)
    nature_distance:         dict[int, np.ndarray]  # NLCD lucode → distance raster
    # Derived scalars (computed once at load)
    baseline_hm:             float            # mean of baseline_hmi_raster
    baseline_cn:             float            # area-weighted CN under baseline
    baseline_ndvi:           float
    # CRS + transform + reference metadata
    ref_transform:           rasterio.Affine
    crs_wkt:                 str
```

**Build / cache:** `@st.cache_resource def _load_city_runtime_state(city_key: str) → CityState` — cached on `city_key` so heavy work runs at most once per (city, session); Streamlit reruns fetch the same NamedTuple from cache. City-switching reuses the second city's cached state instantly if it has been loaded before.

**Backward-compat module-level aliases.** Immediately after the loader returns, ~25 module-level names are rebound to fields of the NamedTuple (`cooling_lulc = _CURRENT_CITY_STATE.cooling_lulc`, `ET_RESIZED = …`, `BUILDINGS_RASTER = …`, `_BASELINE_HM_RASTER = …`, etc.). Downstream function bodies read them as bare names. **`baseline_hm` and `baseline_cn` are NOT aliased** — every call site reads them via `_CURRENT_CITY_STATE.baseline_hm` / `.baseline_cn` (the explicit-handle treatment prevents silent staleness if a future code path fails to refresh a global on city switch; scalars would silently produce wrong-but-plausible numbers, arrays would crash).

**§3 must call out the explicit-handle exception for `baseline_hm` / `baseline_cn`.**

### `[VERIFY] 3` — Caching: `@st.cache_resource` vs `@st.cache_data` + city-switch behavior

**Source: `grep -nE '@st\.cache_(resource|data)' app.py` — full list below.**

`@st.cache_resource` is for shared, non-serializable objects (NamedTuples, model instances, large arrays). `@st.cache_data` is for serializable, deduplicated computation results (DataFrames, dicts of scalars). The split is canonical Streamlit.

**Total: 6 cached functions — 2 `@st.cache_resource` + 4 `@st.cache_data`.**

**`@st.cache_resource`** (2 functions):
- `_load_city_runtime_state(city_key)` — decorator at `app.py:2254` (`@st.cache_resource(max_entries=1, show_spinner="Loading city data — first interaction may take a minute…")`); returns the `CityState` NamedTuple. **`max_entries=1` is load-bearing — see city-switch note below.**
- `_cached_train_surrogate(_scenario_df, data_dir_flood, data_dir_cooling, scenario_signature, ...)` — at `app.py:2947`. RF surrogate. Cached on the scenario-df identity + paths + mode signature; underscored `_scenario_df` is skip-hashed.

**`@st.cache_data`** (4 functions):
- `load_lulc_crosswalk(crosswalk_path, default_ff, default_gi, default_hd)` — at `app.py:579`. SA compound LULC crosswalk loader (CSV → DataFrame).
- `load_data(data_dir_flood, data_dir_cooling, cn_table_file, cooling_table_file, lulc_file, soil_file, cooling_lulc_file)` — at `app.py:747`. Initial raster + biophysical-table load. Cached on the path arguments (so different cities get different cache entries).
- `compute_scenario_grid(_state, city_key, data_dir_flood, data_dir_cooling, step_pct=10, step_alloc=25, ...)` — at `app.py:2029`. The per-city scenario training grid (Fast prototype mode source). Underscored `_state` is skip-hashed.
- `compute_lookup_table(_state, city_key, data_dir_flood, data_dir_cooling, schema_version=SCENARIO_SCHEMA_VERSION)` — at `app.py:2078`. The dense lookup table (High resolution mode source). Underscored `_state` is skip-hashed.

**City-switch behavior — `max_entries=1` on the big cache.** When the user picks a different city:
- `_load_city_runtime_state(city_key)` — **`max_entries=1` means only the active city's `CityState` lives in cache. Every city switch EVICTS the prior city's state and rebuilds (~10–30 s on SA — the spinner string fires).** MN → SA → MN is *two* rebuilds of MN, not "instant on the second time." This is the explicit memory-fit tradeoff: keep one heavy NamedTuple resident, accept rebuild cost on every switch, fit inside Streamlit Cloud's 1 GB worker ceiling. Persisted artifacts under `data/precomputed/<city>/` (`nature_distance_<lucode>.npy`) bypass live re-compute on rebuild.
- `load_data(...)` — cached per (paths). Cities have disjoint paths → disjoint cache entries. **NOT evicted by the `cache_resource` policy** (different decorator class). Survives city switches.
- `load_lulc_crosswalk(...)` — survives city switches (SA-only; called once per session typically).
- `compute_scenario_grid` + `compute_lookup_table` — cached per `(city_key, schema_version + paths)`. Survive city switches per the `cache_data` policy.
- `_cached_train_surrogate` — cached per `(scenario_df identity, paths, signature)`. Switching model-quality modes within a single city rebuilds the surrogate.

**Schema-bump cache invalidation.** `SCENARIO_SCHEMA_VERSION` participates in the cache key for `compute_scenario_grid` and `compute_lookup_table` (as the literal `schema_version` argument). Bumping the constant invalidates both caches across all cities; the `CityState` cache survives unless `max_entries=1` evicts it for the city-switch reason above.

**§9 must:**
- Get the count right: **2 `cache_resource` + 4 `cache_data` = 6 cached functions**.
- Distinguish the SIX cache keys + the four invalidation triggers (city switch, mode switch, schema bump, slider edit).
- Call out **`max_entries=1` explicitly** — it changes the city-switch story from "instant on re-pick" to "rebuild on every switch."
- Distinguish three caching tiers: (a) Streamlit in-memory cache (six functions above), (b) persistent on-disk artifacts under `data/precomputed/<city>/`, (c) lazy live compute.

### `[VERIFY] 4` — Layer 2 → Layer 3 training relationship (RESOLVED — the inconsistency reconciled)

**Source: `app.py:5072–5083` (training-set selection by mode); `surrogate.py:101–179` (`train_surrogate`); `compute_scenario_grid` args at `app.py:2735` (defaults `step_pct=10, step_alloc=25`); `precompute_scenarios.py` (dense CSV generator); `compute_lookup_table` at `app.py:2785`.**

**The inconsistency in the current doc** — "~600 rows lookup" vs "trained on ~90 sims" — was a confusion across THREE different precomputed sets, all of which feed the surrogate depending on `model_quality` mode:

| Mode | Surrogate training set | Source | Row count |
|---|---|---|---|
| Fast prototype (default) | `compute_scenario_grid(step_pct=10, step_alloc=25)` | live compute at startup | **~90** |
| Balanced | `data/scenarios_dense_<city>.csv` (offline-built by `precompute_scenarios.py --city '<city>' --step-pct 5 --step-alloc 10`) | on-disk dense CSV | **~726** |
| High resolution | the full 2,541-entry lookup table from `compute_lookup_table` | live compute at startup (~25–50 min on SA's grid) | **2,541** |

**The surrogate trains on whichever set is selected by the mode**, not on a single fixed source. So:
- "~90 sims" — true for the **default** (Fast prototype) mode.
- "~600 rows" — that was the historical name for the Balanced dense CSV (actual current count is ~726 at step_pct=5, step_alloc=10).
- "2,541-entry lookup" — High resolution mode also doubles as the source for instant slider response (the lookup short-circuits the expensive raster aggregates) AND as the surrogate's training data.

**Tree count** (`SURROGATE_TREES` in `app.py`):
- Fast prototype: `n_estimators=100`
- Balanced: `n_estimators=200`
- High resolution: `n_estimators=300`

The "three knobs" framing (training scenarios / trees / optimizer candidates) maps: training scenarios + trees are mode-coupled, optimizer candidate count (10,000) is independent.

**§5 Layer 3 must enumerate all three modes and the per-mode row count + tree count.** The "~90 sims" form is fine only if it explicitly names "Fast prototype mode." Don't carry the contradiction.

### `[VERIFY] 5` — Module map (post-migration)

**Source: `app.py:20–30` (imports), `ls validation/ diagnostics/ scripts/data/`, and the recent migration commits (`0235c22` docs migration, `4453c20` code reorg).**

**At repo root (8 modules):**

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, sidebar state, metric cards, `evaluate_scenario` and biophysical helpers, `CityState` + loader, scenario-source plumbing, tradeoff/map render |
| `config.py` | `CITIES` dict (per-city paths + parameters) and global cost defaults; read-only |
| `surrogate.py` | Random-forest surrogate + Pareto-style optimizer; Streamlit-agnostic |
| `natcap_scenarios.py` | NatCap fixed-scenario loader, `flood_reduction_from_nlcd_tree` helper, `PROVENANCE_*` taxonomy; Streamlit-agnostic |
| `natcap_validation.py` | `render_validation_badge`, `published_delta`, reference-CSV reads; Streamlit-agnostic |
| `export_invest_bundle.py` | InVEST 3.19.0 export-bundle assembly (rasters + biophys tables + per-model args.json + metadata.json + README), zipped |
| `verify_baselines.py` | Baseline regression check — snapshots `evaluate_scenario` for 40 (city × scenario × strategy) combos against committed JSON. CI gate before commits. |
| `precompute_scenarios.py` | Offline dense-CSV builder for Balanced-mode training set; stubs streamlit + reuses `evaluate_scenario` |

**`validation/` (5):** `compare_carbon_invest.py`, `compare_ucm_invest.py`, `compare_umh_invest.py`, `compare_una_invest.py`, `verify_cooling.py` — canonical-InVEST parity comparators.

**`diagnostics/` (5 — one retired):** `compare_una_lulc.py`, `analyze_placement_diagnostic.py`, `placement_strategy_diagnostic.py`, `check_expanded_coverage.py`, `validate_surrogate_predictions.py`. (`validate_scenarios.py` retired in `943119d`.)

**`scripts/data/` (17):** all `download_*`, `process_*`, `clip_worldpop.py`, `extract_natcap_reference_outputs.py`. Data-pipeline scripts.

**§8 must include this exact module list. The `validation/` and `diagnostics/` dirs are post-migration — the old "compare_*_invest.py at root" addresses in older docs are no longer current.**

### `[VERIFY] 6` — Export bundle: output structure + metadata fields

**Source: `export_invest_bundle.py` (the `build_invest_bundle` builder); the bundle README template at `_readme()`.**

**Output structure** (verified on the SA baseline bundle — all 5 InVEST 3.19.0 urban models execute cleanly):

```
ecosystem_explorer_export_<city_slug>_<scenario_id>_<timestamp>.zip
├── README.md
├── metadata.json
├── inputs/
│   ├── prototype/
│   │   ├── scenario_lulc_evaluated_30m_5070.tif   (compound — UCM / UNA / Carbon-alt)
│   │   ├── baseline_lulc_evaluated_30m_5070.tif
│   │   ├── scenario_lulc_nlcdtree_30m_5070.tif    (NLCD×tree — UFR)
│   │   ├── baseline_lulc_nlcdtree_30m_5070.tif
│   │   ├── scenario_ndvi_30m_5070.tif             (UMH ndvi_alt)
│   │   └── baseline_ndvi_30m_5070.tif             (UMH ndvi_base)
│   ├── shared/                                     (population, ET, soil HSG, AOIs, prevalence vectors)
│   └── biophysical/                                (UCM / UNA / Carbon compound tables; SA NLCD×tree CN table)
└── args/prototype_grid/
    ├── urban_cooling_args.json
    ├── urban_nature_access_args.json
    ├── urban_flood_risk_mitigation_args.json
    ├── carbon_args.json
    ├── urban_mental_health_depression_args.json    (effect_size = RR per 0.1 NDVI, depression)
    └── urban_mental_health_anxiety_args.json
```

**`metadata.json` fields** (per the `BundleSpec` dataclass + serializer):
- `format_version` — bundle schema (current: 1)
- `prototype_git_commit` — short SHA of `app.py`'s HEAD at export time
- `scenario_schema_version` — current `SCENARIO_SCHEMA_VERSION`
- `export_timestamp_utc`
- `city` — full city name from `CITIES`
- `scenario` — block with `provenance` (one of the four PROVENANCE_* constants), `scenario_name`, `pct_converted`, `gi_pct`, `ff_pct`, `hd_pct`, `placement_strategy`
- `generator` — block with `type` (`baseline` / `natcap_fixed_scenario` / `explorer_generated` / `optimizer_suggested`) and generator-specific parameters
- `raster_lineage` — block describing each input raster's source path inside the bundle + the on-disk source it was derived from
- `validation` — per-model state pulled from `docs/internal/NATCAP_ALIGNMENT.md` (one entry per UCM / UNA / UFR / Carbon / UMH; values are `validated_pixel_parity`, `validated_aggregate_only`, `method_aligned_unvalidated`, `prototype`)
- `model_availability` — per-model `available: true|false` flag with `reason` string for fixed alternatives where compound inputs weren't shipped (`reason: "NatCap did not ship a compound LULC for this fixed scenario; only flood is exported."`)

**The "export ≠ already validated" framing is mandatory** — `metadata.json`'s `validation` block records the prototype's own measured parity against canonical InVEST per model; running canonical `execute()` on the bundle produces fresh canonical outputs that the user is then free to compare against the prototype's reported card values.

**§7 must enumerate the file structure + all metadata.json keys (above) + the export-≠-validated framing.**

### `[VERIFY] 7` — Current `SCENARIO_SCHEMA_VERSION`

**Source: `grep -n "^SCENARIO_SCHEMA_VERSION = " app.py`.**

```
SCENARIO_SCHEMA_VERSION = 27
```

Per `app.py:1950`: bumped when "UMH neighborhood-exposure kernel changed from Gaussian to canonical buffer-mean (InVEST 3.19.0 per-pixel parity) — `preventable_mh_cases` / `avoided_mh_cost_usd` shift for every conversion scenario. Full per-bump rationale in `docs/archive/HISTORY.md` 'Schema version log'."

§5 and §9 should say **27** as the current value and cross-ref `docs/archive/HISTORY.md` for the bump history.

### `[VERIFY] 8` — Inbound-reference inventory for ARCHITECTURE

**Source: grep across `.md` and `.py` files.**

#### External `.md` refs

| File:line | Cited anchor / target | Action |
|---|---|---|
| `README.md:42` | `docs/internal/ARCHITECTURE.md` (Start here bullet) | leave; full-path doc ref |
| `README.md:53` | `docs/internal/ARCHITECTURE.md` (Start here last bullet) | leave |
| `README.md:60` | `docs/internal/ARCHITECTURE.md` (Documentation map row) | leave |
| `docs/internal/STRATEGY.md:265` | sibling `ARCHITECTURE.md` (Related documents) | leave (sibling, bare) |
| `docs/internal/DESIGN_NOTES.md:9` | sibling `ARCHITECTURE.md` "three-layer system overview" (no anchor — section name only) | leave |
| `docs/internal/CITY_PARITY.md` | (none observed) | n/a |
| `docs/dev/CONTRIBUTING.md:6` | `../internal/ARCHITECTURE.md` | leave |
| `docs/archive/SPEC_original.md:6` | `../internal/ARCHITECTURE.md` (Do not use this for) | leave |

#### External `.py` refs (textual section anchors)

| File:line | Cited anchor | Action |
|---|---|---|
| `app.py:728` | `docs/internal/ARCHITECTURE.md "CRS handling"` | retarget after refresh — "CRS handling" content moves to §3 |
| `app.py:743` | `docs/internal/ARCHITECTURE.md 'CRS handling'` | same retarget |

**No other `.py` files cite ARCHITECTURE section anchors.**

#### Internal-self-refs in ARCHITECTURE.md

Current ARCHITECTURE doesn't reference its own sections by anchor name — the doc is short (176 lines) and self-contained. The refresh's link-fix is therefore minimal on the inbound side: **2 external `.py` refs to "CRS handling"** + a handful of `.md` refs that point at ARCHITECTURE.md generically (no section anchor) and don't need updating.

**Anchor-stability requirement.** Only **"CRS handling"** has measurable inbound traffic (`app.py:728`, `:743`). The refresh's §3 must keep this anchor name verbatim OR update the two `app.py` refs in the same commit.

---

## Resolved decisions (bake in)

- ARCHITECTURE owns the REFERENCE Layer 2/3 depth-absorption (REFERENCE Commit A dropped). REFERENCE does not edit ARCHITECTURE.
- "Why numpy" rationale flows to DESIGN_NOTES §6 (parked until that refresh); ARCH keeps gist + cross-ref.
- §6 validation/provenance uses the locked two-surface vocabulary, consistent with REFERENCE §4 and DESIGN_NOTES §8.
- §2 (scenario sources), §6 (validation/provenance), §7 (export) are first-class architectural sections — these are real subsystems now (`natcap_scenarios.py` / `natcap_validation.py` / `export_invest_bundle.py` are imported by `app.py`), not documentation-only ideas.
- **Anchor "CRS handling" retained verbatim in §3** (preserves 2 inbound `app.py` refs without a code-change requirement).

---

## Mapping rows worth flagging

1. **Current "Per-city configuration" → §3 + §8.** The split is mechanical: `CITIES` dict structure (the *what*) → §3 Runtime data model; per-city specifics (the *which*) → §8 Module responsibilities under `config.py`. Worth making this explicit in the rewrite — readers will look in §3 for "how does the app know which city to load" and in §8 for "what's in config.py."
2. **Layer 1 split → §4 + §5 + §6 is a three-way fan-out.** The current Layer 1 section conflates flow (`evaluate_scenario` call chain), per-model implementation, and per-model validation status. The refresh's three-way split is correct but needs care — the `evaluate_scenario` call chain belongs to §4 (pipeline); the *biophysical engine* belongs to §5 Layer 1; validation status belongs to §6. Mapping table is right; the rewrite needs discipline to keep them separate.
3. **`@st.cache_data` vs `@st.cache_resource` distinction is load-bearing.** §9 must explain WHY the four cached functions split that way (NamedTuples and large arrays in cache_resource because cache_data would re-pickle them every miss; per-mode lookup tables in cache_data because they're DataFrames). Otherwise §9 reads as a list of decorators without the why.
4. **The `data/precomputed/<city>/` on-disk artifacts are a third caching tier.** §9 should distinguish (a) Streamlit in-memory cache, (b) on-disk persistent artifacts under `data/precomputed/<city>/`, and (c) lazy live compute. The on-disk artifacts have a separate invalidation rule (delete the dir) — not Streamlit-managed.

---

## Sequence after this map is approved

1. **Approved + map committed.** ← awaiting approval.
2. **CC rewrites ARCHITECTURE §1–11.** Stop-and-report draft before commit. Pulls REFERENCE.md:335–371 + 683–749 from git into §5; absorbs the locked validation/provenance vocabulary in §6; uses the locked `CityState` field list + cache-tier breakdown; preserves "CRS handling" anchor verbatim in §3 (or co-updates `app.py:728`, `:743` in the same commit).
3. **Commit** — one commit for the ARCHITECTURE refresh.
4. **Cross-ref sweep.** REFERENCE.md's gist cross-refs (which currently point at ARCHITECTURE's three-layer section) retarget to the new §5. Held until after step 3 commits.
5. **Delete this map file** after step 4 lands.
