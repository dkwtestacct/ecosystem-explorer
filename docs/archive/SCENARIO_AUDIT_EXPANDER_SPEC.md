# Scenario Audit Expander — Build Spec

**Audience:** Internal
**Status:** Building — display only; zero gates.
**Depends on:** Scenario Record Pass (live — record fields), Subset Invariants Pass (live), Eligibility Funnel (live — same record-reads).
**Builds:** A small "Scenario audit" expander rendered under the scenario title on the Scenario tab. Every field reads the record; no recomputation.
**Source of truth for:** the field list, the uniform-record rendering rule, and where the expander sits in the page.

---

## Why

The scenario record now carries the full placement provenance — Source, Area, Ownership, Placement strategy, Seed, Eligible / Converted acres, Validation, Schema version. The comparison table surfaces a subset (Source / Validation / Area / Ownership); the funnel surfaces the acres chain. The audit expander is the **complete view of a single scenario's record in one place** — the same fields a planner or methodology reviewer would want to inspect when asking "what exactly did this scenario do, and how is it grounded?"

It's a view, not a new computation. Every cell reads a record field directly; "inapplicable" never means "blank" — it means the uniform-record value ("Citywide", "None") so the field list is consistent across all scenario types.

## Rendering

**Location:** Under the scenario title on the Scenario tab — after the provenance header (`_render_scenario_provenance_header` at `app.py:5022-5023`) and before the metric cards. A collapsed `st.expander("Scenario audit", expanded=False)` so it's available but not noisy.

**Fields (in order):**

| Field | Source (record path) | Citywide / inapplicable value |
|---|---|---|
| Source | `results['source_label']` or composed from provenance + suffixes | the provenance label alone |
| Area | `results['region_selection']` → composed via `_cs_area_for_row` | `"Citywide"` |
| Ownership | `results['ownership_filter']` → composed via `_cs_ownership_for_row` | `"None"` |
| Placement | the active `placement_strategy` (sidebar value, mirrors saved record) | `"random"` (the default) |
| Seed | `42` (the locked seed; surfaced because the record carries it) | `42` |
| Eligible acres | `results['region_selection']['eligible_pixels_in_region']` × `PIXEL_AREA_ACRES` | citywide convertible count × `PIXEL_AREA_ACRES` |
| Converted acres | `results['region_selection']['converted_acres']` | same field — always populated |
| Validation | provenance-derived label from `_PROVENANCE_HEADER_INFO` (locked badge vocab) | the same locked vocab — "Baseline" / "NatCap reference" / "Explorer-generated" / "Surrogate-suggested" |
| Export schema | `SCENARIO_SCHEMA_VERSION` (integer; today's value is 32) | same |

**Locked badge vocabulary** — Validation cell uses the exact phrasing from `_PROVENANCE_HEADER_INFO` (no new status terms introduced).

## Dynamic-string discipline

- `$`-escape any value that could contain a literal `$` (none of these fields realistically do — city names, region labels, ownership labels, numeric counts — but the discipline holds).
- Render via `st.dataframe` (prose-safe by construction) or `st.markdown` with explicit escapes. Going with `st.dataframe` for consistency with the funnel's renderer.

## Data sources (one rule)

Every cell reads the record. No recomputation, no parallel truth.

- `results['region_selection']` is already populated by `evaluate_scenario` + caller stamping.
- `results['ownership_filter']` is already the bare mode string in `results`.
- The composition helpers `_cs_area_for_row(row)` and `_cs_ownership_for_row(row)` already exist in `app.py` (from the Scenario Record Pass); reuse them so the audit expander and the comparison table render the same values identically.
- `SCENARIO_SCHEMA_VERSION` is a module-level constant in `app.py`.

## Verification

- `verify_baselines.py` — 40/40 byte-identical (display-only).
- Eyeball:
  - A region + ownership SA scenario (e.g. District 5 + vacant_public): Area reads `Council district 5`; Ownership reads `Vacant publicly-owned land`; Placement / Seed / Eligible / Converted populate from the record.
  - A citywide MN scenario: Area reads `Citywide`; Ownership reads `None`; Eligible reads the citywide convertible acres; everything else populates normally.

## Out of scope

- Edit / re-export the record from the expander.
- "Why this validation status" pop-out (already in the per-metric badge tooltips).
- Audit history (older saved-scenario records over time).

## Not touched

- `evaluate_scenario` math, schema version, save handler, comparison table.
- Provenance header, metric cards, region-local view.
