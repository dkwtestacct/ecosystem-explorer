# Subset Invariants + State-Transition Guards — Build Spec

**Audience:** Internal
**Status:** Building — zero human gates; the deliverable is the verification.
**Depends on:** Region Selection (live), Ownership Filter (SA live), Interactive Region Map (live), Scenario Record Pass (live).
**Builds:** Three subset-invariant assertions that run as a standing gate in `verify_baselines.py`, plus state-transition guards in `app.py` that prevent stale region / ownership state from carrying across a city switch.
**Source of truth for:** the three invariants, the test-matrix, the city-switch reset semantics, and the funnel-plumbing requirement that the deferred eligibility-funnel work will reuse.

---

## Why

Region Selection and Ownership Filter constrain *where* conversions can land. Today nothing in the standing gate confirms that the engine actually honors those constraints — only that the engine produces consistent per-strategy metrics (the 40/40) and that the eligibility scalar `eligible_pixels_in_region` matches an independent recompute. **Neither of those would catch a placement-stage bug** that wrote conversions to pixels outside the selected region, outside the ownership mask, or onto buildings/roads.

The three invariants spec what "honors the constraint" means in the only form that's automatically testable: a subset relation on the post-evaluation rasters. Once they're standing assertions in `verify_baselines.py`, any future change that breaks the placement contract surfaces immediately as a numbered cell failure rather than as a subtle metric drift weeks later.

City-switch is a related defense: the masks themselves rebuild every rerun, but the **widget-key session_state entries** (`region_apply_within`, `region_labels_<layer>`, `ownership_filter_choice`) persist. Without an explicit reset, switching SA→MN→SA preserves whatever region/ownership the user had picked in the original SA session — across an interruption that should logically clear placement intent.

## The three invariants (every scenario, always)

After `evaluate_scenario`:
- Let `converted_mask = (BASELINE_LULC != results['scenario_lulc'])`. Boolean raster the same shape as `BASELINE_LULC`.
- Let `eligible_mask` be the convertible pool projected onto the raster — `True` for pixels in `CONVERTIBLE_PIXELS`, `False` elsewhere.
- Let `region_mask` be `selected_region_mask` if region selection is active, else `None`.
- Let `ownership_mask` be the boolean codes-match raster if ownership filter is active, else `None`.

Assert all three separately (defense in depth — catches an eligible-mask miscomposition that happens to still subset the region/ownership masks):

1. **`(converted_mask & ~eligible_mask).sum() == 0`** — converted ⊆ eligible (always).
2. **If region active:** `(converted_mask & ~region_mask).sum() == 0` — converted ⊆ region.
3. **If ownership active:** `(converted_mask & ~ownership_mask).sum() == 0` — converted ⊆ ownership.

`(converted & ~mask).sum()` (not equality) so the failure message can report the count of out-of-mask pixels and the (row, col) of the first offender.

## Test matrix

Six cells in concept; per-city availability dictates which run:

| Cell | Region | Ownership | SA | MN |
|---|---|---|---|---|
| 1. region only | active | inactive | ✓ | ✓ |
| 2. region + ownership | active | active | ✓ | — (no ownership data) |
| 3. ownership only | inactive | active | ✓ | — |
| 4. citywide | inactive | inactive | ✓ | ✓ |
| 5. tiny region (~10 eligible px) | synthetic mask | inactive | ✓ | ✓ |
| 6. multi-region (2–3 selected) | active | inactive | ✓ | ✓ |

Total: 6 cells × SA + 4 cells × MN = **10 matrix cells**. For each, all three invariants apply (the gated ones skip when their mask is None).

The tiny-region cell is constructed as a synthetic 5×5 patch intersected with the convertible pool, so the eligible count is in the tens. Tests two failure modes: (a) over-conversion when `n_convert > eligible`, (b) crash on `pct_converted * eligible / 100 < 1`. Neither should occur.

**Rule:** if any cell fails on first run, that's a real spatial bug. Surface the cell, the count of out-of-mask pixels, an example offender pixel, and the fix. **Do not soften the assertion.** The assertion *is* the spec.

## State-transition guards

City-switch (`app.py:222+` already handles slider + optimizer-state reset; extends to placement state):

- **`region_apply_within`** — set back to `"Entire analysis area"` (the radio's default).
- **`region_labels_*`** — every layer-keyed multiselect state cleared.
- **`region_layer`** — selected layer cleared.
- **`region_map_picker_event`**, **`region_map_picker_layer`** — clear any in-flight click event from the previous city's interactive map.
- **`ownership_filter_choice`** — set back to `"No filter"`. Belt-and-suspenders for non-SA cities where the widget doesn't render, but ensures a clean return to SA from MN.

These changes write to session_state at the top of the script, before the sidebar renders. The mask-rebuild path in the sidebar then sees clean values and re-derives masks correctly for the new city.

## Funnel plumbing (for the deferred eligibility funnel)

While computing the subset checks, expose these intermediate cardinalities so the deferred eligibility-funnel work (P3) reuses them:

- `total_px` — `ref_shape[0] * ref_shape[1]`
- `developed_px` — `len(state.developed_pixels)`
- `convertible_px` — `len(state.convertible_pixels)` (= developed minus buildings/roads)
- `region_px` — `region_mask.sum()` when region active, else None
- `region_eligible_px` — `(convertible_in_raster & region_mask).sum()` when region active, else equal to `convertible_px`
- `ownership_px` — `ownership_mask.sum()` when ownership active, else None
- `final_eligible_px` — `(convertible_in_raster & region_mask & ownership_mask).sum()` with None masks treated as all-True
- `converted_px` — `converted_mask.sum()`

The funnel reads: `developed → convertible → region_eligible → final_eligible → converted`. Each hop drops a count; the deferred panel renders these directly. "Existing nature" doesn't get its own hop because natural pixels are excluded *upstream* of the funnel by the DEVELOPED_CODES filter — the "developed" hop already encodes the natural exclusion.

## Verification gate

- `verify_baselines.py` — existing 40/40 byte-identical (additions, no math movement).
- **NEW** — 10-cell subset-invariant matrix (SA 6 + MN 4), three checks per cell where applicable.
- **NEW** — guard transition test: programmatically set SA region + ownership in session_state, run the city-switch reset block, assert region_apply_within / region_labels_* / region_layer / ownership_filter_choice are all cleared.

## Not in scope

- The eligibility-funnel UI panel (P3) — the cardinalities are surfaced; the panel that displays them is its own batch.
- The scenario-audit expander (P5).
- The optimizer-reversal flow (P4).
- A region+ownership baseline snapshot — covered by the subset invariants without the cost of a baseline file per cell.

## Not touched

- `evaluate_scenario` math.
- `SCENARIO_SCHEMA_VERSION` stays at 32 — no metadata or results shape change.
- Any per-city scalar.
