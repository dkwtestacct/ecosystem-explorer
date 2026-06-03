# Scenario CSV Export — Build Spec

**Audience:** Internal
**Status:** Building — serialization only; zero gates.
**Depends on:** Scenario Record Pass (live — record carries the full placement provenance), Comparison Table (live — Source/Validation/Area/Ownership columns), Eligibility Funnel (live), Scenario Audit Expander (live — same record-reads, fuller view).
**Builds:** A "Download scenario summary (CSV)" button under the Compare scenarios table. One row per scenario (current + saved-for-city). Every column reads the record — no recomputation.
**Source of truth for:** the column order, the per-scenario row-builder, and the verify contract (CSV round-trips record fields).

---

## Why

The audit expander gives one scenario's record at a glance; the comparison table gives several scenarios side-by-side with a curated column set. The CSV is the **data-complete, downloadable form of the comparison table** — the same rows, every record field, every computed metric. A reviewer (or a future you) can open it in their tool of choice, sort, filter, plot, share, archive. It's the "ground truth" snapshot of what the dashboard knows about a scenario set, ready to leave the app cleanly.

It's pure serialization: every cell reads a field that already exists on the scenario record or on `results` / `_saved`. No new computations, no methodology drift risk.

## Location

Under the Compare scenarios table on the Tradeoff Analysis tab (tab2), right after the `st.caption(...)` that notes the column basis. A single `st.download_button` rendering the CSV in memory (`io.StringIO` + `pandas.DataFrame.to_csv`).

## Rows

One row per scenario:
- The current scenario (provenance computed locally just like the comparison-table current row).
- Every saved scenario for the active city (`_saved_for_city`).
- **NatCap fixed anchors are intentionally excluded** — they don't carry a full record (no slider mix, no placement, no eligible/converted acres). Including them would force "—" cells across most columns and dilute the round-trip guarantee. The comparison table already surfaces them; the CSV stays focused on Explorer / Baseline / Optimizer scenarios that carry a complete record.

## Columns (in order)

Field groups; "—" for missing values; no recomputation:

**Identity + provenance**
- `scenario_label` — the display name the user sees in the comparison table.
- `city` — active city display name.
- `provenance` — `eib.PROVENANCE_*` value (BASELINE / EXPLORER / OPTIMIZER).
- `source_label` — full augmented Source-line string (with selected-region / ownership suffixes when active).
- `validation` — locked badge vocab from `_PROVENANCE_HEADER_INFO`.

**Region + ownership (the full placement provenance)**
- `region_layer` — `region_selection.layer` or empty.
- `region_selected_ids` — `region_selection.selected_ids` joined by `|` (so it round-trips as one CSV cell).
- `region_selected_area_acres` — `region_selection.selected_area_acres` or empty for citywide.
- `region_eligible_acres` — `eligible_pixels_in_region × PIXEL_AREA_ACRES`.
- `region_converted_acres` — `region_selection.converted_acres`.
- `ownership_mode` — bare mode string (`public` / `vacant` / `vacant_public`) or empty.
- `ownership_label` — `OWNERSHIP_MODES[mode]['label']` when active; **empty cell** when no ownership filter is active. (Pandas read_csv's default `na_values` list includes the literal string `"None"`, so writing it as the no-filter sentinel would silently coerce to NaN on parse. The comparison-table column and audit expander still render `"None"` via `_cs_ownership_for_row` at display time — only this serialization path stays empty for no-filter rows.)
- `ownership_source` — from `CITIES[city]['ownership_layer'].source` when active, else empty.
- `ownership_data_date` — same path, `.data_date`.

**Conversion mix + placement**
- `pct_converted`
- `green_infrastructure_pct`
- `food_forest_pct`
- `pct_highdensity` — = 100 − GI − FF.
- `placement_strategy`
- `random_seed` — `42` for saves; current row reads the locked seed.
- `scenario_schema_version` — `SCENARIO_SCHEMA_VERSION`.

**Citywide metrics (a curated representative set; matches the comparison table's columns)**
- `flood_reduction` (the prototype CN-inversion index, 0–100)
- `temp_change_f`
- `mean_hm`
- `mean_ndvi`
- `food_mln_lbs`
- `carbon_tons_co2`
- `carbon_value_usd`
- `cooling_energy_savings_usd`
- `nature_access_pct`
- `people_with_nature_access`
- `preventable_mh_cases`
- `avoided_mh_cost_usd`
- `total_cost_mln`
- `runoff_acre_feet`

**Region-local mirrors (when present)**
- One `region_local__{metric}` column per decomposable metric in `_REGION_LOCAL_METRICS`. Empty for citywide scenarios.

The audit expander surfaces a subset; the CSV is the union. Order is stable so consumers can rely on the column layout across runs.

## No recomputation

Every cell is either a direct read or a single-operation derivation (acres = pixels × `PIXEL_AREA_ACRES`; `pct_highdensity = 100 − GI − FF`). No mask intersection, no metric re-aggregation, no engine call. The CSV's correctness reduces to "the record was correct" — the verify gate doesn't have new math to assert.

## Filename

`scenario_summary_{city_slug}_{YYYY-MM-DD}.csv` — UTC date so the same dashboard state from two browsers produces the same filename.

## Verification

- `verify_baselines.py` — 40/40 byte-identical (display-only / serialization; no math touched).
- **NEW** — programmatic CSV round-trip smoke: build the row payload from a synthetic region+ownership saved scenario, serialize, parse back via `pandas.read_csv(io.StringIO(...))`, assert every record field round-trips equal.
- Eyeball — open the CSV from a region+ownership SA scenario and a citywide MN scenario. Values sensible, no `nan` / `Object Object` cells, schema_version reads `32`.

## Out of scope

- NatCap anchor rows (no full record; intentionally excluded).
- JSON-line / Parquet export.
- "Selected scenarios only" filter — every row goes in; the user can filter downstream.
- Replay-import (CSV → saved scenarios).

## Not touched

- `evaluate_scenario`, schema version, save handler, comparison table column layout.
- The CSV is independent of the export bundle (which is a per-scenario zip with rasters); the bundle remains the per-scenario hand-off, this is the comparison-table-wide snapshot.
