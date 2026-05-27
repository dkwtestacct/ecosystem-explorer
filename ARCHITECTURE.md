# Architecture

**Purpose:** A high-level map of how the Urban Ecosystem Tradeoff Explorer is built. For someone who needs to understand the system before reading code, or who wants to know where to look for a specific concern.

**Audience:** Anyone — Daniel, future Claude sessions, NatCap collaborators, anyone picking up the project after a break.

**For deeper detail:** `REFERENCE.md` covers methodology (what each metric means, which model produced it). `DESIGN_NOTES.md` covers internal design decisions (options considered, chosen, why). This doc is the orienting overview that sits above both.

---

## At a glance

The prototype is a Streamlit app that lets users explore tradeoffs in urban land-use scenarios — how different allocations across green infrastructure, food forests, and high-density development affect flood risk, cooling, food production, mental health, carbon, and cost.

Three layers sit underneath the UI:

```
┌─────────────────────────────────────────────────────────────────┐
│  UI (Streamlit)                                                  │
│  Sidebar sliders, metric cards, Tradeoff Analysis,              │
│  Find Best Scenario, Map View                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────────────────────┐
        ▼                ▼                                ▼
┌──────────────┐  ┌─────────────────┐  ┌────────────────────────────┐
│ Layer 1      │  │ Layer 2         │  │ Layer 3                    │
│ Raster sims  │  │ Lookup table    │  │ Surrogate model            │
│              │  │                 │  │                            │
│ Per-pixel    │  │ Pre-computed    │  │ Random-forest predictions  │
│ InVEST       │  │ scenarios.csv   │  │ trained on ~90 sims        │
│ calculations │  │                 │  │                            │
│              │  │                 │  │                            │
│ Realism      │  │ Speed           │  │ Wide-scenario search       │
└──────────────┘  └─────────────────┘  └────────────────────────────┘
```

Each layer exists for a different reason. Together they let users explore conversions at interactive speeds (Layer 2), see InVEST-canonical biophysical detail (Layer 1), and search a much larger space than the lookup table can hold (Layer 3).

---

## CRS handling

Every raster the prototype reads at runtime is in its city's canonical equal-area CRS — EPSG:26915 (NAD83 / UTM 15N) for Minneapolis, EPSG:5070 (NAD83 / Conus Albers Equal-Area) for San Antonio. Both are equal-area or near-equal-area at the relevant latitudes (UTM ground-area distortion at MN is ~0.05 %, well within rounding), so `PIXEL_AREA_ACRES = 0.2224` is correct for the 30 × 30 m runtime pixels.

**Source data in other CRSs is reprojected at preparation time, not at runtime.** NatCap's San Antonio compound LULC delivery is in EPSG:3857 (Web Mercator), which heavily distorts area at non-equatorial latitudes and is unsuitable for area-based math. Brief 27 reprojected the source compound LULC (`data/sa/natcap_2024/lulc_overlay_3857.tif`) to the live EPSG:5070 raster (`data/sa/flood/land_use_compound_sa.tif`) using nearest-neighbor resampling at 30 m before it ever enters the runtime pipeline. The 3857 source files are preserved on disk for provenance (see `DATA_INVENTORY.md` Section 2) but are not read by `app.py`.

The Streamlit map rendering uses EPSG:3857 internally (because tile servers and Folium expect it), but this is a one-way display conversion applied after all area math has happened in equal-area space. No area-dependent metric is computed in 3857.

**Runtime assertion.** Every `rasterio.open(...)` site in `app.py` calls `_assert_raster_crs(src, expected_crs, file_path)` after opening; the helper raises `ValueError` with a clear file-naming message if the raster's CRS doesn't match the city's canonical CRS. Defense-in-depth against future data-integration mistakes — a 3857 raster (or any non-equal-area CRS) accidentally introduced would crash loudly with the offending path named, rather than silently producing wrong area math.

---

## Layer 1 — Raster simulations

**What it does.** For a given scenario (specified by city, percent converted, mix of cover types, and placement strategy), compute the actual InVEST biophysical metrics per pixel. The function entry point is `evaluate_scenario()` in `app.py`.

**Models implemented (per city):**

- **InVEST Urban Cooling Model (UCM)** — Heat Mitigation Index, temperature deltas, cooling energy savings
- **InVEST Urban Flood Risk Mitigation (UFR)** — SCS-CN runoff, flood damage avoided
- **InVEST Urban Nature Access (UNA)** — per-capita nature supply, % of population meeting demand
- **InVEST Urban Mental Health (UMH)** — preventable cases, avoided costs
- **Carbon Storage** — sequestration tons CO₂e/yr (single-rate proxy, not full InVEST 4-pool)

**Validation status.** UCM matches `natcap.invest.urban_cooling_model.execute()` at MAE=0. UFR uses canonical SCS-CN runoff. UNA uses canonical 2SFCA implementation. UMH uses InVEST RR formula. Carbon is methodologically simplified (see CITY_PARITY.md).

**Why it exists.** Realism. Layer 2 and Layer 3 derive from Layer 1. Without per-pixel biophysical calculation, the prototype has no ground truth.

**Speed cost.** A single SA `evaluate_scenario()` call takes ~0.9 seconds. MN downtown takes ~0.03 seconds. Too slow for interactive slider response on SA, which is why Layer 2 exists.

**For deeper reading:** `REFERENCE.md` (per-metric methodology), `INVEST_PLACEMENT.md` (placement-strategy formulas), `PLACEMENT_STRATEGY_DIAGNOSTIC.md` (empirical findings).

---

## Layer 2 — Lookup table

**What it does.** Pre-computes Layer 1 across the full slider space (city × scenario × pct × placement strategy) and stores the results in the active city's `data/scenarios_dense_<city>.csv` (per `dense_scenarios_file` in the CITIES config). At runtime the UI looks up the user's current slider position and returns the answer instantly.

**Generation.** `precompute_scenarios.py` enumerates the grid (typically 4 scenarios × ~10 pct values × 5 placement strategies × 3 cities ≈ 600 rows) and runs `evaluate_scenario()` for each. Takes ~15 minutes for a full regenerate.

**Bumps.** Whenever a substantive change affects scenario outputs, `SCENARIO_SCHEMA_VERSION` increments and the lookup table is regenerated. The baseline test suite (`verify_baselines.py`) sits on top of the same data.

**Why it exists.** Speed. Interactive slider response requires sub-second updates. Layer 2 makes the common case (user dragging sliders) instant by hiding Layer 1's cost behind a precomputation step.

**Coverage gap.** The lookup table is *grid-shaped* — it covers fixed (scenario, pct, strategy) tuples. The Find Best Scenario tab needs to search a much higher-dimensional continuous space than the grid can represent. That's Layer 3.

**For deeper reading:** `precompute_scenarios.py` for generation logic. `DESIGN_NOTES.md` for the SCENARIO_SCHEMA_VERSION discipline.

---

## Layer 3 — Surrogate model

**What it does.** A random-forest regressor trained on ~90 pre-computed Layer 1 simulations. Predicts metric outcomes for arbitrary continuous scenario inputs (any pct, any allocation mix). Lives in `surrogate.py`.

**Used by.** The Find Best Scenario tab. The user specifies what they're optimizing for (e.g., "minimize flood risk subject to cooling ≥ X"), and the surrogate evaluates thousands of candidate scenarios in seconds, finding Pareto-efficient frontiers. Layer 1 is too slow for this; Layer 2 doesn't cover the continuous input space.

**Training.** Surrogate is trained at app startup via `train_surrogate()`, cached with `@st.cache_resource` so it persists across reruns. Training data comes from the active city's `data/scenarios_dense_<city>.csv` (the same Layer 2 lookup table); training itself takes a few seconds.

**Optimization.** Once trained, the optimizer samples candidate scenarios (typically thousands), uses the surrogate to predict each one's metrics, filters by user-specified constraints, and returns the best Pareto-efficient candidates.

**Limitations.** Placement strategy is not yet a surrogate input — the optimizer effectively assumes the user's currently-selected placement strategy. Cost-effectiveness isn't a surrogate target; it's computed downstream from surrogate predictions.

**Why it exists.** Exploration. Users want to ask "what's the best mix?" rather than "what happens at this specific mix?" Layer 3 enables that high-dimensional search.

**Why not ROOT?** NatCap's [ROOT](https://natcap.github.io/ROOT/index.html) does linear-programming-based optimization for ecosystem services and is a real alternative. The prototype's surrogate is a different (simpler, ML-based) approach. See NATCAP_COLLABORATION.md for context — ROOT is acknowledged but not pursued.

**For deeper reading:** `surrogate.py` for the random-forest training and Pareto-filtering logic.

---

## Why numpy reimplementation, not canonical `natcap.invest`

The prototype implements InVEST urban-model logic in numpy rather than calling `natcap.invest.urban_cooling_model.execute(args)` and similar. Two reasons:

- **Latency.** Canonical InVEST is built on `taskgraph`, a desktop pipeline framework that reads inputs from disk, executes in a worker process, and writes outputs back to disk. For Bexar County extent at 30 m (~3.4 M pixels), a single `execute()` call takes minutes. Streamlit's rerun-on-interaction model would make every slider move re-trigger the pipeline — incompatible with the prototype's three-layer caching architecture, which serves slider responses in milliseconds.
- **No `execute_from_arrays()` API.** The canonical API takes file paths in its args dict. There is no in-memory variant. Working around this would require writing temporary `.tif` files on every slider move.

**Validation, not replacement.** The prototype's numpy implementations are validated against canonical `natcap.invest` runs in `compare_*_invest.py` scripts (`compare_ucm_invest.py`, `compare_una_invest.py`, `compare_carbon_invest.py`). `NATCAP_ALIGNMENT.md` tracks the per-model validation diffs — e.g., UNA matches canonical at MAE 0.0234 m²/person, Pearson r = 1.000000. The prototype's runtime is fast; its correctness is anchored to canonical InVEST through these offline validation runs.

If you wanted publishable canonical InVEST results for a specific scenario, you would run `natcap.invest` offline against that scenario's LULC raster and use those outputs directly. That is the offline-validation path, not the interactive-prototype path.

---

## Data flow

```
Source data            Layer 1            Layer 2            Layer 3
──────────────────    ──────────────    ────────────────    ──────────────
data/ (rasters,        evaluate_         scenarios_dense    surrogate.py
biophysical tables,    scenario()        .csv                random forest
config.py)             per-pixel         pre-computed        trained on ~90
                       InVEST            grid lookup         sims; predicts
                       calculations                          arbitrary mixes
       │                    │                  │                  │
       └────────────────────┴──────────────────┴──────────────────┘
                                     │
                                     ▼
                              Streamlit UI
                              (sliders, cards, tabs)
```

**For deeper reading on data:** `DATA_INVENTORY.md` (every external data source the prototype consumes, per-city). `CITY_PARITY.md` (per-city alignment with NatCap's published configurations).

---

## Per-city configuration

Each city is a row in `config.py`'s `CITIES` dict, with paths to its LULC/soil/buildings/ET rasters, biophysical tables, and city-specific scalars (UHI magnitude, food yield benchmark). Adding a new city is in principle a matter of adding a new row and providing its data — the layered architecture is city-agnostic.

Currently active cities: Minneapolis (downtown), San Antonio. Minneapolis Full is dormant (`available=False`) but retained in the config so scripts/tests can still reference it.

---

## Testing

`verify_baselines.py` runs the full pipeline (Layer 1) for every (city, scenario, strategy) tuple and compares against snapshotted outputs in `tests/baselines/`. Currently 40 baselines. Bumps to `SCENARIO_SCHEMA_VERSION` invalidate baselines; running `verify_baselines.py --update` regenerates them.

This is the regression gate. Every brief that touches biophysical math or scenario outputs must end with 40/40 passing.

---

## Where to read next

| If you want to understand... | Read |
|---|---|
| What each user-facing metric means | `REFERENCE.md` |
| Why a particular design choice was made | `DESIGN_NOTES.md` |
| What data the prototype consumes | `DATA_INVENTORY.md` |
| How aligned the prototype is with NatCap canonical, per city | `CITY_PARITY.md` |
| Per-methodology NatCap alignment status | `NATCAP_ALIGNMENT.md` |
| The collaboration log with NatCap | `NATCAP_COLLABORATION.md` |
| Empirical placement-strategy effect sizes | `PLACEMENT_STRATEGY_DIAGNOSTIC.md` |
| Per-InVEST-model placement analysis | `INVEST_PLACEMENT.md` |
