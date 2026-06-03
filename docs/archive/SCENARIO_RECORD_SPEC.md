# Scenario Record + Region-Aware Saved Scenarios — Build Spec

**Audience:** Internal
**Status:** Building — queued after the optimizer-sync fix + UI-text pass, both landed.
**Depends on:** Region Selection (live), Ownership Filter (SA live), Region-Local Metrics (live), Honesty-Surface Pass (live — already serializes `region_selection` + bare `ownership_filter` in export bundles).
**Builds:** A unified "scenario record" view of any scenario the user works with — citywide, region-constrained, or region + ownership-constrained — that captures the recipe (sliders + placement + seed + masks) AND the metrics it produced, with a save model that stores recipe + metrics (raster regenerated on demand). Comparison table extended to surface Area + Ownership columns so placement constraints are visible alongside Source / Validation.
**Source of truth for:** the record schema, the save model, the reproducibility contract, the comparison-table column extensions, and the explicit scope boundary against the deferred views (eligibility funnel / scenario audit / optimizer reversal / downloadable CSV).

---

## Why

Region Selection and Ownership Filter both shape a scenario's *placement* — but a saved scenario's row in the comparison table doesn't make that visible. Today, a "District 5 vacant-public" scenario reads the same way in the table as a citywide scenario, except for a small ` · selected-region placement · vacant publicly-owned land` suffix on the Source cell. A planner comparing four scenarios — citywide-balanced / District 5 / public-land-only / NatCap-reference — can't line them up by placement at a glance. Two new columns fix that.

At the same time, the brief unifies the *concept*: the saved-scenario dict, the export bundle's metadata.json, and the in-app comparison rows are all viewing the same underlying object — a scenario *record*. Specifying that record explicitly opens up the deferred views (eligibility funnel, scenario audit expander, optimizer reversal, downloadable CSV) without re-specifying the shape every time.

## The record

A scenario record is a dict with these fields. **Bold = newly required for this pass; the rest already persist in either `results` or the saved-scenario dict.**

```
{
  "city":               str,     # active city display name, e.g. "San Antonio, TX"
  "provenance":         str,     # eib.PROVENANCE_* (BASELINE / NATCAP_FIXED / EXPLORER / OPTIMIZER)
  "source_label":       str,     # the augmented Source line (e.g. "Explorer-generated · selected-region placement")
  "validation_status":  str,     # locked badge vocab — derived from provenance via _PROVENANCE_HEADER_INFO
  "sliders": {
    "pct_converted":            int,
    "green_infrastructure_pct": int,
    "food_forest_pct":          int,
  },
  "placement_strategy": str,     # 'random' | 'flood-focused' | 'cooling-focused' | 'undersupply-focused' | 'balanced'
  "random_seed":        int,     # ★ NEW — always captured (every strategy uses rng; ranking strategies sample stochastically with weights)
  "cost": {"gi": int, "ff": int, "hd": int},
  "region_selection": {          # already persisted via results['region_selection']
    "mode":                       "selected_regions" | "entire_aoi",
    "layer":                      str | None,   # e.g. "council_districts"
    "selected_ids":               list[str] | None,
    "selected_area_acres":        float | None,
    "eligible_pixels_in_region":  int,
    "converted_acres":            float,        # ★ NEW — sum of n_wet+n_for+n_hd × PIXEL_AREA_ACRES
  },
  "ownership_filter":   str | None,   # bare mode key ('public' | 'vacant' | 'vacant_public' | None) — kept thin
  "metrics":           dict,          # the full citywide metric set (engine-computed)
  "region_local":      dict | None,   # region-clipped metric set when a region mask is active
}
```

### Design decisions

- **Keep `ownership_filter` as a bare mode string in the underlying record.** The rich `{enabled, city, allowed_classes, source, data_date}` view is *composed at render time* from `OWNERSHIP_MODES` + `CITIES[city]['ownership_layer']` config. This avoids a `SCENARIO_SCHEMA_VERSION` bump and keeps the record minimal. The export bundle and the comparison table both get the same composed view from the same composition rule.
- **Keep `region_selection.eligible_pixels_in_region` instead of duplicating as `eligible_acres`.** Acres = pixels × `PIXEL_AREA_ACRES`; same information. The display layer formats acres; the underlying record carries pixels.
- **`random_seed` always captured.** Even though today's seed is hardcoded to 42, capturing it forward-compatibly is required so future seed-variation work (or per-scenario seeds) reproduces. All five strategies use rng (`rng.choice(replace=False, p=weights)` for the ranking strategies); the brief's "deterministic by construction" framing isn't accurate — they're stochastic-but-seeded. With seed captured, all five are recipe-reproducible.
- **Save model — recipe + stored metrics, raster regenerated on demand.** Comparison table reads stored metrics → exact and stable, zero recompute risk. Raster is regenerated on demand (re-view, re-export) by calling `evaluate_scenario(**recipe)`. Reproducibility guaranteed by the seeding contract.

## Reproducibility contract

Regenerating a saved scenario's raster from its recipe must reproduce its stored metrics.

**This is already guaranteed by the existing 40/40 baseline gate.** `verify_baselines.py:402-411` calls `app.evaluate_scenario(**params, seed=42, placement_strategy=strategy)` for every (city × scenario × strategy) combination and asserts results against committed snapshots. The 40 snapshots ARE the (recipe → metrics) regeneration test, run per strategy.

**New assertion this pass adds is incremental** — saved-scenario round-trip:
1. Build a representative scenario *record* (sliders + strategy + seed + costs).
2. Call `evaluate_scenario(**recipe)`.
3. Assert regenerated metric dict matches the record's stored metrics.

Formalizes the saved-scenario reproducibility contract at the record-API surface rather than at the call signature.

## Comparison-table columns

Add two new columns between `Validation` and the existing metric columns:

| Column | Citywide | Region-active | Region + Ownership-active | NatCap reference rows |
|---|---|---|---|---|
| **Area** | `"Citywide"` | `"{layer-display} {label}"` (single) / `"{N} selected {layer-display-plural}"` (multi) | same as region-active (ownership shows in next column) | `"Citywide"` |
| **Ownership** | `"None"` | `"None"` | `OWNERSHIP_MODES[mode]['label']` | `"None"` |

Both columns derive from the same record fields (`region_selection`, `ownership_filter`) used everywhere else. Dynamic strings → `\$`-escape and render-as-prose verification (region labels and ownership labels don't contain `$` today, but be defensive).

The existing Source suffix (`" · selected-region placement"` / `" · vacant publicly-owned land"`) stays — Source describes "where this row came from" while Area/Ownership describe "what placement constraint shaped it." Both signals are informative; users scanning the table for placement parity will use the new columns.

The "different sources aren't directly comparable as precision numbers" caption framing stays — Area/Ownership columns make placement constraints visible alongside source provenance; the comparison caveat doesn't shift.

## Build

### 1. Extend `evaluate_scenario`'s `region_selection` payload (`app.py:2205-2217`)

Add `converted_acres` field (= `(n_wet + n_for + n_hd) * PIXEL_AREA_ACRES`). Backward-compatible — additive.

### 2. Extend save handler (`app.py:6491-6521`)

After the existing capture, append `random_seed = 42` to the saved dict. (Today's seed is hardcoded; this captures it so future code can vary it without breaking older saves.)

### 3. Comparison-table — Area + Ownership columns (`app.py:6280-6353`)

Introduce two row-builder helpers:
- `_cs_area_for_row(row, city_state)` — reads `row['region_selection']` (if any) and the layer's display name; returns the formatted label or `"Citywide"`.
- `_cs_ownership_for_row(row)` — reads `row['ownership_filter']` (bare mode string); returns `OWNERSHIP_MODES[mode]['label']` or `"None"`.

Apply to every row builder: NatCap anchors (citywide / no ownership), current row, saved rows. Column order: Scenario · Source · Validation · **Area · Ownership** · metric columns.

Add `Area` and `Ownership` to the `column_config` block so the column widths and (optional) tooltips render consistently with Source / Validation.

### 4. Saved-scenario round-trip assertion (`verify_baselines.py`)

Add a small block after the existing assertions. For each city:
1. Pick the canonical "balanced" scenario from `_SCENARIOS_PER_CITY`.
2. For each placement strategy:
   - Build the recipe (sliders + strategy + seed).
   - Call `evaluate_scenario(**recipe)`.
   - Compare against the same baseline snapshot the 40/40 already asserts.

This is mostly redundant with the 40/40 — the value is in the *framing*: the assertion stands up the "saved-record → regen → reproduces metrics" contract explicitly. Diffs count separately so a saved-scenario API regression doesn't read as a generic 40/40 failure.

## Verification gates

- `verify_baselines.py` — 40/40 + region/ownership/reconciliation + smoke + honesty-surface, all hold trivially (record capture shouldn't move any metric).
- **NEW** — saved-scenario round-trip assertion (per placement strategy, both cities).
- In-app smoke: save a region + ownership scenario, confirm the record persists with `random_seed`, `converted_acres`, and the existing region/ownership fields; confirm the comparison-table row shows correct Area + Ownership.

## Not in scope this pass

Deferred views of the same record — do not pull in:

- **Eligibility funnel (P3)** — convertible / eligible-after-region / eligible-after-ownership / converted, surfaced as a panel.
- **Scenario audit expander (P5)** — inline expander on each saved row showing the full record.
- **Optimizer reversal (P4)** — using a saved record's recipe to seed an optimizer search starting from a region-constrained scenario.
- **Downloadable CSV** of saved scenarios.

All four use the record specified above; each is its own batch when prioritized.

## Not touched

- `evaluate_scenario`'s metric outputs (only `region_selection.converted_acres` is new — additive).
- `SCENARIO_SCHEMA_VERSION` stays at 31.
- `OWNERSHIP_MODES` / `CITIES` config — composition layer reads from existing structure.
- Existing Source-line suffixes on the main panel header / export bundle / comparison Source column.

## Decision (resolved at build time) — ownership shape, in-memory vs export

**Question:** the brief's `ownership_filter {enabled, city, allowed_classes, source, data_date}` is a rich shape. Should that be the underlying storage, or a render-time view?

**Decision:** asymmetric — bare in memory, rich at export.
- **In-memory `results['ownership_filter']`** stays a bare mode string (`'public' | 'vacant' | 'vacant_public' | None`). No consumer audit; no schema change forced through every reader.
- **Comparison table** composes the rich Area + Ownership view at render time from `OWNERSHIP_MODES + CITIES`.
- **Export `metadata.json`** composes the rich `{mode, label, allowed_classes, source, data_date}` dict at the bundle-build site so the exported bundle is self-describing. `BundleSpec.ownership_filter` type widened from `Optional[str]` to `Optional[dict]`.

**`SCENARIO_SCHEMA_VERSION` bumped 31 → 32** for the metadata.json shape change (ownership_filter str → dict; region_selection adds converted_acres). Results dict picks up `region_selection.converted_acres` additively; `verify_baselines._SNAPSHOT_SKIP_KEYS` already skips `region_selection`, so 40/40 baselines stay byte-identical (no re-baseline).
