# UI-Text Pass — Build Spec

**Audience:** Internal
**Status:** Built — landed after the map-click optimizer-sync fix; text/display only.
**Depends on:** Region Selection (live), Region-Local Metrics (live), Ownership Filter (SA live), Interactive Region Map (live). No model, mask, or evaluation logic touched.
**Builds:** Text/label/copy polish across the Scenario tab, the interactive region map panel, the sidebar, and What's New — to make the "click a region → see citywide + region-local results" story read clearly to a planner who hasn't sat through a demo.
**Source of truth for:** the wording decisions in this pass + the entire-area / multiple-regions / SA + MN fallback rules.

---

## Why

The interactive region map landed; the optimizer-sync fix made the click coherent. But the surrounding copy still reads like an engineer's first pass — "Citywide impact shown above" (tab-relative), "the sidebar multiselect stays in sync" (mechanism, not capability), region-named labels hardcoded to "tract" / "district". The honesty-surface pass already locked the badge/provenance vocabulary; this is the matching prose pass on the rest of the surface.

This is text/display only. No string change should affect any metric, mask, or validation surface.

## Scope (each item below has the same shape — what / where / fallbacks)

### 1. Citywide caption above metric cards

**What:** When a region is selected, render above the metric cards:
> "Citywide impact from changes placed in {region label} (in {city})."

**Where:** Scenario tab, above the metric-card row.
**Fallbacks:**
- Entire-area (no region): omit the caption.
- Multiple regions: "Citywide impact from changes placed in {N} selected {layer-noun-plural} (in {city})." List the labels inline if short (≤ 3); otherwise N.
- City (SA / MN): `{city}` is the active-city display name (`"San Antonio, TX"` / `"Minneapolis, MN"`).
**Dynamic-string discipline:** `$` is escaped (`\$`) in any rendered prose that could include cost values from the same row; verify renders as prose, not LaTeX.

### 2. Region-local table heading rename

**What:** `"Region-local view"` → `"Selected-region impact"`.
**Where:** Scenario-tab region-local table heading.
**Fallback:** Heading is unchanged when no region selected (the block is hidden in that state).

### 3. Sidebar caption rewrite

**What:** Replace `"Metrics show citywide impact."` with
> "Metric cards show citywide impact. The Region-local view compares outcomes inside the selected area with the citywide result."

**Where:** Region Selection sidebar block, post-eligibility caption.

### 4. Food / cost / carbon equality tooltip

**What:** Short tooltip (ⓘ on the region-local table header):
> "For direct conversion metrics (food production, cost, carbon), region totals equal citywide totals when all converted pixels are inside the selected region — this matches the locked clip-clean treatment."

**Where:** Region-local table header info icon.
**Fuller form:** REFERENCE.md region-local section (one-line addition pointing at the same explanation).
**Framing rule:** This is correct behavior, not a caveat. The note explains *why* the columns can read equal; it does not apologize for the equality.

### 5. Adaptive comparison-table title

**What:**
- Only the current row present → `"Current scenario summary"`.
- Any saved / reference / optimized rows present → `"Compare scenarios"`.

**Where:** Scenario-tab comparison table heading.

### 6. Map-panel label generalization

**What:** Rename the first map-panel metric label from `"Selected {tract|district}"` to `"Selected area"`. The specific region id moves to the caption beneath the panel: e.g. `"Downtown census tract 27053012101"` (MN) / `"Council district 5"` (SA).
**Derivation:** The region-id caption is generated from the active `region_layer` (`region_layer_display_names[layer_key]` + the selected label[s]).
**Fallbacks:**
- Single region: `"{layer-display-name} {label}"`.
- Multiple regions: `"{N} selected {layer-display-name-plural}"`.

### 7. Thicker outline on selected polygon

**What:** Bump the selected-polygon line width on the interactive region map. Fill alone is subtle on small districts; a heavier stroke makes the selection unambiguous.
**Where:** Tab 3 plotly figure, line-width branch for `_is_sel`.
**Display only — no event-handling change.**

### 8. De-tech the selector instruction text

**What:** Replace the current caption beneath the interactive map with:
> "Click a {region-noun} to select it. Shift-click or Ctrl-click to select multiple. Land-use changes will be placed only inside the selected area; the Scenario tab shows both citywide and region-local results."

Drop the `"the sidebar multiselect stays in sync"` mechanism phrasing.
**Region-noun:** derived from the active `region_layer_display_names`.

### 9. Tab-agnostic "above" fix

**What:** Replace `"Citywide impact shown on the metric cards above."` with
> "Metric cards show citywide impact; the Scenario tab also includes region-local readings for the selected area."

**Where:** Interactive region map summary panel (tab 3). "Above" is wrong when the user is on a different tab.

### 10. What's New restructure

**What:** Reorganize the in-app "What's New" panel into two sections:

- **Interactive scenario placement** — Region Selection, Region-local view, Ownership Filter (SA-only call-out).
- **Validation and handoff** — provenance badges, comparison table, scenario export.

Shorten Export and Comparison bullets to one line each. **Cut the "Data/model updates" section entirely** — that's implementation history; it belongs in the changelog / "On the radar" (or just out). Keep provenance only if it reads as a user-visible capability, not as a methodology footnote.
**On the radar:** stays a separate adjacent section.
**Locked vocabulary:** All copy aligns with the honesty-surface badge vocabulary (NatCap reference / Baseline / Engine-validated Explorer / Surrogate-suggested optimizer suggestion). No new status terms introduced.
**Base text:** Deborah's drafted What's New, minus the Data/model-updates section.

### 11. Ownership eligibility caveat — tooltip, not What's New

**What:** Add a tooltip on the ownership filter selectbox:
> "Ownership classes are used as a planning screen only; parcel availability and legal feasibility are not verified."

**Where:** SA-only ownership-filter `st.sidebar.selectbox`. Not a What's New bullet.

## Out of scope (explicit)

- The optimizer caveat. Moot — the optimizer guard already disables Optimize when a region or ownership mask is active.
- The "Where Changes Happen" subtitle. Skip.

## Dynamic-string discipline (applies to every interpolated string above)

1. Escape `$` (`\$`) in any prose that could include a cost-formatted value from the same row, so Streamlit's markdown renderer doesn't read it as LaTeX.
2. Manually verify each dynamic string renders as prose with a representative payload (single region, multiple regions, no region) on SA and MN.
3. Region-noun and city-noun derivations come from `_CURRENT_CITY_STATE.region_layer_display_names` and the active city display name in `CITIES` — never hardcoded.

## Verification gate

- `verify_baselines.py` — 40/40 + region + ownership + reconciliation + honesty-surface completeness assertions. All should pass trivially (text-only changes).
- Eyeball pass on SA + MN, with: entire-area, single region, multiple regions, ownership filter active (SA only).

## Not touched

No `evaluate_scenario` interface change, no schema bump, no precompute change, no per-city scalar change, no validation-status change.
