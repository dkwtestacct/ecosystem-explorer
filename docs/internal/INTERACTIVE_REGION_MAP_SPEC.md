# Interactive Region Map — Build Spec

**Audience:** Internal
**Status:** Queued — next workstream after the honesty-pass + UI-polish push. Opens with a recon step (current Map View tech) before the interaction is finalized.
**Depends on:** Region Selection (the `selected_region` state + the boundary overlay, both live). Region-local + ownership are live and unaffected.
**Builds:** click-to-select district interaction + converted-pixels map overlay + a region area/eligibility summary — the selection UX and visual payoff for a region feature that's otherwise live but dropdown-only.
**Source of truth for:** the recon, the interaction decision, the build, and scope.

---

## Problem

The region feature is live — selection, region-local columns, ownership — but the *map* isn't interactive. You pick a district from a dropdown, and the map draws the boundary but not the conversions. The demo moment that makes it land — "I clicked District 5, and the changes happen only there" — doesn't exist yet. This adds it.

## Phase 0 — map recon (report, not a gate)

Before the interaction is specced precisely, confirm what the Map View is built on: Folium / pydeck / `st.map` / a static image? Is it already interactive (are click events available)? How is the existing boundary overlay layered on? This determines whether click-to-select is a small wiring job or needs a rendering-stack change. CC reports; the decision below is made against the answer.

## Decision required (yours, after recon)

- **Selection model:** single-click to select, shift/ctrl-click to add districts, click-again or a Clear button to deselect.
- **Coexistence with the dropdown:** the map and the existing district dropdown should be two inputs to *one* `selected_region` state — clicking the map updates the dropdown and vice versa, not a second parallel selector. Confirm that's the intent; the alternative (map replaces the dropdown) is more work and loses keyboard/accessibility.

## Build (batch)

- **Click-to-select** — wire map click / shift-click to the existing `selected_region` state (the same state the dropdown sets, so they stay in sync). No new masking logic — it's a new *input* to a mask that already exists.
- **Converted-pixels overlay** — the boundary outline is already drawn; add the actual converted pixels rendered inside it (from the scenario result, never a mock), so "changes only inside the region" is literally visible. Needs the converted-pixel set surfaced to the map layer.
- **Region area/eligibility summary (#3)** — a small panel: selected region, eligible land (acres), converted land (acres), "citywide impact shown above." Pulls from the `region_selection` block already in results.

## Scope

**In:** click/shift-click selection synced to the existing state; converted-pixels overlay; the area summary.
**Out:** freehand / draw-your-own regions (a larger Phase-2 item); region-constrained optimization (still future); any change to mask logic (this is input + display only); MN-specific map work beyond what's already city-agnostic.

## Minimal gating (zero gates)

The recon is informational (reported, not gated). The interaction is your decision, like the region-local boundary call. The build is batch — it's selection-input + display, so 40/40 confirms it doesn't touch the model, and your eyeball confirms the click works and the overlay shows conversions *only* inside the region. No new assertion needed beyond 40/40; the mask it feeds is already guarded by the region-targeted baseline checks.

## Honesty

The overlay must show the *actual* converted pixels, never a representative mock — "changes only here" is a claim about real placement, so it has to be the real placement. The area summary's "citywide impact shown above" keeps the honest pairing: the map shows *where*, the metric cards are still citywide unless the region-local columns are read.
