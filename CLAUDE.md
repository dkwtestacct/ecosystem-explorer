# Ecosystem Explorer — CLAUDE.md

**Audience:** Claude sessions working in this repo
**Status:** Current
**Use this for:** How an AI assistant should work here — the behavioral contract, invariants, key commands
**Do not use this for:** Per-decision rationale (→ `docs/internal/DESIGN_NOTES.md`), data inventory (→ `docs/internal/DATA_INVENTORY.md`), system architecture (→ `docs/internal/ARCHITECTURE.md`), or methodology (→ `REFERENCE.md`)
**Source of truth for:** The agent's operating manual

---

*Stays at repo root for Claude Code auto-discovery. Do not move.*

## What this app does (one paragraph)

Streamlit app that lets users explore tradeoffs in urban land-use conversions — green infrastructure, food forests, high-density development — and the resulting flood / cooling / nature-access / mental-health / food / carbon / cost effects. Per-pixel biophysical engine (numpy ports of `natcap.invest.*`, validated against canonical InVEST at MAE ≈ 0 for UCM / UNA / UMH). Surfaces NatCap's published San Antonio project values as labeled reference points. Run with `streamlit run app.py`. Active cities: Minneapolis (downtown) + San Antonio; Minneapolis Full is dormant.

## Agent read-order

For any task: read this file → read `docs/internal/ARCHITECTURE.md` (system structure) → consult the task's owning doc per `README.md`'s **Documentation map** (one question, one home). Do not duplicate the doc-map here — README owns it.

---

## Behavioral contract — the rules every session honors

These are the rails that govern every commit. Each is a hard rule, not a guideline.

1. **Phase 0 — investigate first.** Before any non-trivial change, read the code and confirm the brief's premises against live state. Sentinel checks ("X should already exist", "Y should not yet appear") catch wrong premises before they ship. Stop-and-report if a sentinel fires.
2. **Stop-and-report on gate-worthy commits; batch the rest.** **Gate** (stop-and-report before the commit): cross-doc content moves or deletes; validation / honesty-surface changes; cite-breaking restructures; baseline-affecting code changes; first-of-a-kind work. **Batch** (commit locally, report at the end of the batch): mechanical edits, already-decided changes, repetitive applies of a settled pattern. **The push is the real gate** — local commits accumulate; the user reviews and approves the batch before push. Never assume "small enough" without checking the gate criteria.
3. **Single commit per concern.** One coherent change per commit. Don't bundle unrelated fixes; don't split a single change into setup/finalize. The commit message names the concern.
4. **No `Co-Authored-By` trailers** on commits in this project. Plain commit messages only.
5. **Bump `SCENARIO_SCHEMA_VERSION` on math/output changes; `verify_baselines.py` 40/40 is the regression gate.** Whenever `evaluate_scenario`'s return-dict shape or values can shift, bump the version (it's hashed into `@st.cache_data` keys, so cached lookup tables auto-invalidate). Run `verify_baselines.py` before any commit that could shift outputs; expect 40/40 pass. `--update` re-snapshots after intentional changes.
6. **Absorb-before-delete.** When content moves from doc A to doc B, the destination doc absorbs first (own commit); only then does the source doc strip. Same rule for code refactors — the new home lands before the old is removed. Inbound cites get co-updated in the closing sweep or alongside the rename.
7. **No `git push` until the user explicitly says so.** Commits land locally; batches accumulate. The user reviews everything before push. Treat push as a user-only action.

## What not to break (invariants — stated as rules)

These survive every rewrite:

- **CRS assertion at every `rasterio.open`.** `_assert_raster_crs(src, expected_crs, file_path)` runs immediately after open; mismatched CRS crashes loudly with the offending path named. Defense against silent area-math errors from accidentally-introduced 3857 rasters. Mechanism in `ARCHITECTURE.md` §3 "CRS handling".
- **Per-city scalars are config-driven, not hardcoded.** Every per-city value (`uhi_max_c`, `design_storm_inches`, UNA params, biophysical-table filenames) lives in `config.py`'s `CITIES` dict. User-visible strings interpolate from `_CURRENT_CITY_STATE.*` or module-level `city_cfg`-derived constants. `grep -n "Minneapolis\|\bMN\b" app.py` should turn up no hardcoded city names in user-facing strings.
- **Dynamic baselines, not enshrined values.** `BASELINE_HM` / `BASELINE_CN` / `BASELINE_NDVI` are live-recomputed inside `_load_city_runtime_state` at module load. The `CITIES[city]['baseline_*']` values are documentation-only — the live overrides are authoritative. Don't hardcode-and-trust.
- **Single-home doc contract — one question, one home.** Each question has exactly one source-of-truth doc per `README.md`'s Documentation map. Everyone else cross-refs that home, doesn't duplicate. Cross-doc duplication is the bug to root out, not the convenience to add.

## Key commands

```
streamlit run app.py                # run the Streamlit app
verify_baselines.py                 # regression gate; 40 city × scenario × strategy snapshots
verify_baselines.py --update        # re-snapshot after an intentional methodology change
precompute_scenarios.py             # offline dense-CSV builder for Balanced-mode training data
```

Streamlit Cloud's worker ceiling is 1 GB; SA's 1713 × 1984 grid is the largest AOI and the default test bed for memory-sensitive changes. If SA fits, MN and Mpls-Full fit by definition.

---

## Coding rules (the four inbound-cited rules)

The four rules in this section are explicitly cited from `app.py` and `docs/internal/DESIGN_NOTES.md`. Anchor names are stable.

### Pure-variant helpers

Heavy compute helpers that `_load_city_runtime_state` invokes come in two variants:

- `_fn(scenario_lulc)` — reads module aliases populated by the loader.
- `_fn_pure(scenario_lulc, *deps)` — takes its dependencies explicitly.

The loader uses the **pure variant** because the module aliases haven't been rebound yet at the moment the loader runs; downstream code uses the zero-arg wrapper.

Currently applies to `_compute_hmi_raster` / `_compute_hmi_raster_pure` (UCM), `_una_supply_percapita` / `_una_supply_percapita_pure` (UNA), `_compute_carbon_four_pool` / `_compute_carbon_four_pool_pure` (Carbon). Same pattern for any new module-alias-reading helper.

### Interface changes require auditing all consumers

When a change modifies the shape of a shared interface — adding a field to `evaluate_scenario`'s return dict, changing a function signature, adding a config key — the change's scope must enumerate **every consumer** of that interface, not just direct callers in the same file:

- Scripts that import the function (`precompute_scenarios.py`, `verify_baselines.py`, any standalone utility)
- Tests that exercise the interface (validation/diagnostics scripts that build their own stubs)
- Serialized formats that capture the interface's output (CSVs, JSON baselines)

**Per-city serialized artifacts are independent consumers — list them one-by-one.** Each `data/scenarios_dense_<city>.csv` must be regenerated separately per city when a schema bump changes column names or required columns. Folding "regenerate the dense CSV" into one line that only happens for the city the author is actively testing causes silent CSV staleness that crashes weeks later in a different model-quality mode.

### Buildings — typing

OSM buildings carry `type` as strings (`'house'`, `'apartments'`, `'retail'`, …), not the integer 0–3 codes InVEST UCM expects. SA maps OSM strings → InVEST type codes 1/2/3 via `_OSM_BUILDING_TO_INVEST_TYPE` in `app.py`. About 29 % of pixels carry a typed code; untyped polygons (`building=yes`, NaN, `roof`, `storage_tank`, …) are left at 0 and excluded from per-type lookups. This **lights up the Cooling Energy Savings card** for SA as a conservative lower bound; the tooltip surfaces the coverage caveat whenever `BUILDINGS_TYPE_COVERAGE < 0.95`.

MN downtown uses the InVEST UFR sample shapefile (already typed) for the dollar-metric raster; OSM footprints feed the **placement mask** only — the split-config rationale lives in `docs/internal/CITY_PARITY.md` MN UFR section.

### OSM road exclusion

Road footprints are unioned into `BUILDINGS_RASTER` so the convertible-pixels pool excludes both buildings and impassable surfaces. The road-class filter is **Option B** (`ROADS_DROP_CLASSES`): drops `footway`, `cycleway`, `steps`, `service`, `path`, `pedestrian`, `unclassified`, `track*`. These are sub-pixel-width surfaces that would over-count the non-convertible mask at 30 m NLCD resolution. Retained classes: motorway, trunk, primary, secondary, tertiary, residential, living-street, and on/off-ramp links. After union with buildings, **~65 % of developed pixels (NLCD 21–24) remain convertible** for MN; SA's broader extent shrinks the convertible pool similarly. Rasterization is unbuffered line-to-pixel via `rasterio.features.rasterize` with `dtype="uint8"`.

---

## Per-rewrite session disciplines

These emerge from doc-suite rewrite work and travel with any future rewrite of similar shape.

- **Stop-and-report sentinels are signal, not noise.** Briefs include sentinel checks ("X should not already exist", "Y should match expected content") for a reason. When a sentinel fires, the brief's precondition isn't met — stop, surface what was found, wait for direction. Don't override the trigger by skipping the check or reframing content as "close enough."
- **Verify referenced constants before relying on them.** When a brief or content map cites specific numerical constants (NatCap publications, EPA documents, NLCD specs) to justify methodology, verify the prototype's *current* value of that constant before treating it as a shared assumption. Same standard can have multiple vintages (e.g. IWG 2021 SC-CO2 at $53/t vs EPA 2023 at $190/t).
- **"Methodology matches; constant differs" is a legitimate alignment pattern.** When aligning with a NatCap publication that uses older parameter values, the prototype can align on *methodology* while keeping the *more current vintage* of the underlying parameter (with the divergence documented). See `DESIGN_NOTES.md` §6.4 for the canonical example (SA Carbon Vibrant Land alignment + EPA 2023 SC-CO2).
- **Planning artifacts can run in parallel with the investigations that inform them.** Drafting the next brief / content map while the current one executes compresses session wall-clock without compressing thought. Use `[CC: detail pending from Brief N]` placeholders for data-dependent specifics; fill them in once the predecessor reports.
- **WHATS_NEW + Underway discipline.** `WHATS_NEW_ENTRIES` are **blockbuster user-facing capabilities only — brief, one line each, no minor or internal changes**. The bar: the change has *already shipped*, would be noticed by a returning user within ~7 days, and is the kind of thing the user wants to know they can now do (a new feature, a new city, a major UI surface). Excluded: doc-only changes, internal refactors, methodology validations whose only user-visible effect is a confidence-badge upgrade, framing/labeling tweaks. When in doubt, cut. `UNDERWAY_ENTRIES` is forward-looking work the user will recognize when they see it (a new model, a new city, a UI feature). Both default to empty; populate only when a change clears the bar.

---

## Quick pointers

| For… | Read |
|---|---|
| What each metric means + per-card validation | `REFERENCE.md` |
| System structure (the three layers, CRS, caching, validation surfaces) | `docs/internal/ARCHITECTURE.md` |
| Why a design choice was made | `docs/internal/DESIGN_NOTES.md` (Decision / Why / Alternatives / Consequences / Revisit / Code touchpoints) |
| Per-city parameter values + per-city data parity | `docs/internal/CITY_PARITY.md` |
| Per-metric alignment status + validation badge taxonomy | `docs/internal/NATCAP_ALIGNMENT.md` |
| Collaboration history, asks, decisions, gaps | `docs/internal/NATCAP_COLLABORATION.md` |
| Data files / paths / source provenance | `docs/internal/DATA_INVENTORY.md` |
| Live blockers (parked / pending) | `docs/internal/OPEN_QUESTIONS.md` |
| Schema-version log + retired-infrastructure + completed-workstream specifics | `docs/archive/HISTORY.md` |
| Environment setup + canonical-InVEST validation harnesses | `docs/dev/CONTRIBUTING.md` |

**Global constants** that aren't per-city live in `app.py` (`PIXEL_AREA_ACRES`, `EPA_SOCIAL_COST_CARBON`, `FOOD_FOREST_LBS_ACRE`, `LBS_PER_PERSON_YEAR`, `DEVELOPED_CODES`, `NODATA`, etc.). Cost defaults live in `config.py`. Per-city parameter values live in `config.py`'s `CITIES` dict and are documented in `CITY_PARITY.md`. Don't enshrine values here.
