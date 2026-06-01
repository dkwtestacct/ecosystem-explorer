# Eligibility Funnel — Build Spec

**Audience:** Internal
**Status:** Building — display + one reconciliation assertion; zero human gates.
**Depends on:** Region Selection (live), Ownership Filter (SA live), Interactive Region Map (live), Subset Invariants Pass (live — surfaces the cardinalities the funnel renders), Scenario Record Pass (live — `region_selection.converted_acres` already in record).
**Builds:** Expand the tab-3 area panel from `Selected / Eligible / Converted` (three cards) into a vertical funnel that shows where pixels drop out at each step (selected → developed → after roads/buildings/nature → after ownership → converted).
**Source of truth for:** the funnel rows + per-constraint adaptation + the single reconciliation assertion.

---

## Why

The current area panel reports the *start* and *end* of placement (`Selected area` → `Eligible for placement` → `Converted`) but skips the middle. A planner picking District 5 + vacant-public sees the number drop from 12,346 acres → 337 acres → 34 acres without an account of which masks did the dropping. The funnel makes the chain legible:

```
Selected area:                       12,346 acres
Developed land:                       2,876 acres
After roads/buildings/existing nature: 337 acres
After ownership filter:                34 acres
Converted:                             34 acres
```

The subset-invariants pass already surfaces every one of these cardinalities (in `verify_baselines.py`'s subset matrix output); this batch reuses the same arithmetic at render time and adds the reconciliation assertion that ties the funnel to the record.

## Rendering

**Location:** `app.py:6906+` (inside `with tab3:`, in the region-map summary panel, gated on `_t3_layer_cfg is not None`). The existing 3-column metric layout is replaced by a vertical readout — a 2-column `st.dataframe` (Step / Acres) reads cleanly and respects existing styling.

**Rows, adapted to active constraints:**

| Row | Region only | Region + Ownership | Citywide / ownership-only |
|---|---|---|---|
| Selected area | ✓ (region area) | ✓ | — |
| Developed land | ✓ (developed ∩ region) | ✓ | — |
| After roads/buildings/existing nature | ✓ (= eligible_pixels_in_region) | ✓ (convertible ∩ region) | — |
| After ownership filter | — | ✓ (= eligible_pixels_in_region) | — |
| Converted | ✓ | ✓ | — |

Citywide-no-filter and ownership-only-without-region: **funnel hidden** (already hidden by the existing `_t3_layer_cfg is not None` gate — no region layer means no interactive map panel). Per-spec call: "minimal or hidden, your call" → hidden. Ownership-only context already has a small eligibility caption in the sidebar.

## Data sources (no parallel truth)

Every funnel number sources from either `results['region_selection']` (already populated by `evaluate_scenario` + caller-stamping) or a one-line `numpy` intersection against `_CURRENT_CITY_STATE.developed_pixels` / `convertible_pixels`:

- **Selected area:** `results['region_selection']['selected_area_acres']`
- **Developed land:** `region_mask[developed_pixels[:, 0], developed_pixels[:, 1]].sum() × PIXEL_AREA_ACRES` *(the one new computation — single-line intersection of masks already in hand)*
- **After roads/buildings/existing nature:**
  - Region only: equals `results['region_selection']['eligible_pixels_in_region'] × PIXEL_AREA_ACRES`
  - Region + ownership: `region_mask[convertible_pixels[:, 0], convertible_pixels[:, 1]].sum() × PIXEL_AREA_ACRES` *(convertible ∩ region, pre-ownership)*
- **After ownership filter (only when ownership active):** equals `results['region_selection']['eligible_pixels_in_region'] × PIXEL_AREA_ACRES` *(the combined-mask `eligible_pixels_in_region` IS convertible ∩ region ∩ ownership when ownership is active — that's by construction of the live app's combined-mask flow at `app.py:4636-4641`)*
- **Converted:** `results['region_selection']['converted_acres']`

The "existing nature" exclusion isn't a separate hop — natural pixels (NLCD codes outside 21–24) are excluded upstream by the `developed_pixels` filter, so the "Developed land" hop encodes it.

## Acres conversion

Single source: `PIXEL_AREA_ACRES = 0.2224` in `app.py:32`. Every funnel cell × `PIXEL_AREA_ACRES`. Same factor the record's `selected_area_acres` / `converted_acres` already use.

## Reconciliation assertion (the only automated check)

In `verify_baselines.py`, after each subset-invariants matrix cell whose case is region-active:

1. Recompute the funnel chain from raw masks: `selected_area_px`, `developed_in_region_px`, `convertible_in_region_px`, `final_eligible_px`, `converted_px`.
2. Assert the chain's **final-eligible** matches `results['region_selection']['eligible_pixels_in_region']`.
3. Assert the chain's **converted_acres** (= converted_px × PIXEL_AREA_ACRES) matches `results['region_selection']['converted_acres']`.

This ties the funnel to the record — no parallel truth. The chain can't drift from the record without surfacing a failure.

**Monotonicity is free.** The subset invariants (`converted ⊆ eligible`, `converted ⊆ region`, `converted ⊆ ownership`) already guarantee each step ⊆ the prior. So a non-monotonic chain can't render; no separate monotonicity assertion needed.

## Verification gates

- `verify_baselines.py` — 40/40 byte-identical (display-only addition, no math movement).
- **NEW** — funnel reconciliation assertion across the matrix cells where region is active (4 SA cells × MN 3 cells = 7 cells; both checks per cell = 14 reconciliation diffs to count).

Eyeball: chain reads right, acres sensible, tiny-region case shows a believable drop (~5 acres → ~5 acres → ~5 acres at pct=100 — the funnel is short but coherent).

## Out of scope

- **Ownership-only-without-region funnel.** The existing sidebar caption already covers the case minimally; a separate ownership-only funnel is its own batch if desired.
- **Per-strategy funnel.** Placement strategy doesn't change the eligible mask, only which pixels within it are picked — funnel is strategy-agnostic.
- **Animated transitions / sparkline / sankey.** Plain table now; richer renderings are a future polish pass.
- **Funnel CSV export.** Saved-scenario record export covers this (the values are already in the record per the Scenario Record Pass).

## Not touched

- `evaluate_scenario` math.
- `SCENARIO_SCHEMA_VERSION` stays at 32.
- Existing area-panel caption ("Council district 5" / "Metric cards show citywide impact…") — those stay below the funnel.
