# Region Selection — Phase 1 build spec

**Audience:** Internal
**Status:** Locked — Phase 0 investigation pending; build queued after Phase 0 stop-and-report
**Use this for:** The canonical Phase 1 scope, gate tiers, and design decisions for the Region Selection feature
**Do not use this for:** Per-decision rationale for unrelated features (→ `DESIGN_NOTES.md`), live blocker dashboard (→ `OPEN_QUESTIONS.md` §3.1 — which mirrors a summary of this spec)
**Source of truth for:** The Region Selection Phase 1 build — what's in scope, what's out, the six commits and their gate tiers

---

**Goal:** let the user choose *where* land-use changes are placed (from existing polygons), instead of relying solely on algorithmic placement strategies. Moves the app from "scenario sketchpad" toward "spatial planning workbench" — and reduces overclaiming, because the *where* becomes planner-driven and transparent rather than the optimizer implying it knows optimal placement.

**Discipline:** Phase-0 investigate-first, then reviewable commits. Gate the seams (mask infrastructure, lookup bypass, provenance) where review catches things; batch the lower-risk commits (UI, overlay, testing). `verify_baselines.py` 40/40 on every code commit. Same loosened tiering as the doc work.

---

## Scope — Phase 1 ONLY

**In:**
- Region *selection* from **existing** polygon layers (no drawing).
- Dropdown / multiselect UI in the sidebar (no map-click yet).
- Region-constrained **placement**: `candidate_pixels = CONVERTIBLE_PIXELS ∩ selected_region_mask`.
- Region-selected scenarios **bypass the lookup table** and run full-raster live.
- Display-only **map overlay** highlighting the selected region (read-only — not click-to-select).
- Structured `region_selection` provenance block (mode / layer / selected_ids / selected_area_acres / eligible_pixels_in_region) in the result + export metadata + saved scenarios.
- Optimizer **disabled/qualified** when a region is selected.
- Exploratory provenance + a "selected-region placement" label.

**Explicitly OUT (later phases):**
- Click-to-select polygons on the map (Phase 2).
- Freehand / drawn polygons, parcel editing, per-pixel painting, undo/redo, saved geometries (Phase 3+).
- Region-clipped per-area metric scorecards (fast-follow — see Decision).
- Region-specific lookup/surrogate or region-level optimization (genuinely future — surrogate retraining).

---

## DECISION — metric aggregation (settled: citywide)

**Region-constrained placement, citywide-impact metrics.** Report the whole-AOI delta of the region-placed conversions, labeled *"Conversions placed within [region]; metrics show citywide impact."* Honest for the non-local models (UNA 800 m, UCM ~600 m, UMH 300 m), no halo-width to invent, minimal UI. Deborah's "metrics stay the same" settles this — the reported numbers stay citywide; the region selector constrains *where conversions go*, not where the metric is clipped. The caption clause is the honesty guard so a region selector doesn't read as region-clipped numbers.

**Fast-follow (not Phase 1): region-clipped local metrics.** Runoff / carbon / food are per-pixel and clip cleanly to the region; the non-local three stay citywide with a per-metric note. Closer to the "outcomes for that selected area" framing, but adds per-metric aggregation UI. Contained change to metrics-reporting + labeling only — placement, architecture, and UI are identical, so it can land later without rework.

---

## Phase 0 — investigate first (no code)

Confirm against the live app before touching anything:
1. **Where `CONVERTIBLE_PIXELS` is built and consumed** (CityState load; the candidate-selection step inside `evaluate_scenario`). Confirm the `∩ selected_region_mask` slots in at candidate-selection without disturbing the placement-strategy suitability ranking (the strategy should rank *within* the masked set).
2. **The existing vector→raster-on-grid path** used for tracts / buildings / roads (the `acs_block_groups_3857.gpkg` → EPSG:5070 reproject-at-load path). The region mask reuses it: selected polygons → boolean mask on the active grid.
3. **The `evaluate_scenario` signature + where the lookup-bypass decision is currently made** (High-res mode lookup hit vs live). Confirm where to branch "region selected → force live."
4. **Provenance plumbing** — how `PROVENANCE_EXPLORER` is currently set, so a region scenario reads as exploratory placement.
5. **Council-district polygons for SA** — confirm the source (City of San Antonio open GIS), and that it reprojects + rasterizes cleanly through the existing vector path. This is the one new data-sourcing step; surface any snags in the Phase-0 report.

Stop-and-report Phase 0 findings before the build.

---

## Build — reviewable commits

**Commit 1 — region-mask infrastructure.** *[gate — touches `evaluate_scenario`, baseline-sensitive]* `evaluate_scenario(..., selected_region_mask: np.ndarray | None = None)`. When provided, `candidate_pixels = CONVERTIBLE_PIXELS ∩ selected_region_mask`. Rasterize selected polygons via the existing path; cache the rasterized masks (a region layer's polygons are stable per city). No UI yet — testable via direct call. `selected_region_mask=None` is exactly current behavior (regression-safe).

**Commit 2 — lookup bypass.** *[gate — changes the eval path]* Region-selected → skip the lookup, run live. Off-grid scenarios can't use the citywide precomputed lookup; this is expected and acceptable (advanced feature, slower is fine).

**Commit 3 — sidebar UI.** *[batch]* "Apply changes within" → *Entire analysis area* (default, current behavior) / *Selected regions* → region-layer selector → multiselect of regions within it. Caption, plain language: *"Conversions will be placed only inside the selected region, after excluding roads, buildings, and existing natural land. Metrics show citywide impact."*

**Commit 4 — map overlay.** *[batch]* Display-only highlight of the selected region(s) on the existing map — read-only, no click handling. Purely for legibility/trust; distinct from Phase 2 click-to-select.

**Commit 5 — provenance + metadata + optimizer guard.** *[gate — honesty surface]* Region scenario → `PROVENANCE_EXPLORER` with a "selected-region placement" label; the provenance header makes the *placement is planner-chosen (exploratory); the per-pixel engine is the same validated math* distinction explicit. Write the structured `region_selection` block (mode / layer / selected_ids / selected_area_acres / eligible_pixels_in_region) into the result dict, the export bundle's metadata.json, and saved scenarios. Optimizer disabled or qualified when a region is active ("Region-specific optimization is not yet implemented. Apply region selection using the sliders.").

**Commit 6 — testing + edges.** *[batch]* Add at least one region-selected baseline per city to `verify_baselines.py`; bump `SCENARIO_SCHEMA_VERSION` (the result dict gains the `region_selection` block). Edge cases: empty region (no convertible pixels → graceful message via `eligible_pixels_in_region == 0`, no crash); region smaller than one pixel; "Entire analysis area" = mask `None` = unchanged path (still uses lookup). Key acceptance test: select one district, 10 % conversion, verify changed pixels appear **only** inside that region.

---

## Polygon layers — MVP = most interpretable, not just most available

The selection layer is chosen for *UX and planner-interpretability*, not for whichever is already rasterized. Granularity matters: a multiselect over ~10 council districts is usable; over 1,124 block groups is not.

- **SA primary: council districts** (~10) — best for planning/demo interpretation, right granularity for a multiselect. Not in the data inventory yet, so this needs a small sourcing step (public City of San Antonio GIS → reproject EPSG:5070 → rasterize via the existing vector path; a `download_*`-style script). Modest lift, worth it.
- **SA fallback: Bexar census tracts** (375) — already available; less intuitive than districts but workable.
- **SA: ACS block groups** (1,124, already the `tracts_file`) — **not** a selection layer. Too granular for a multiselect, even though the rasterization infra is already wired. (My earlier draft had this backwards — it's an infra win but a UX loss.)
- **MN primary: downtown census tracts** (27) — already available.
- **MN: neighborhoods** — later, only if a neighborhood layer is added.
- **Default everywhere:** "Entire analysis area" (no mask).

---

## Honesty / provenance framing

A region scenario is **exploratory placement**: the *where* is planner-chosen (exploratory), the per-pixel engine running inside the region is the same validated InVEST-aligned math. That distinction is the feature's honesty win — say it in the label. It aligns with NatCap's framing (InVEST scores a LULC raster; it doesn't decide where change happens) and is more defensible than algorithmic placement implying optimality.

---

## Doc consistency (folds into the back-half, not new work)

The feature moving from idea → build flips a decision the in-flight docs made:
- **OPEN_QUESTIONS:** region-selection moves from *deferred/speculative* → *planned / in development* (the metric-aggregation choice above is its live design decision).
- **ARCHITECTURE §11:** the seam's "no UI surface yet drives it" gets a near-term horizon (full §11 update lands *after* the build, when the surface exists — not now).
- **WHATS_NEW:** the locked "in the works" one-liner (already queued for the Commit-B lane).
Keep all three as *planned/in-development*, not *shipped*, until the build lands.

---

## Sequencing

Spec ready now; metric-aggregation settled (citywide). CC finishes the doc work first (the parallel-track plan: independent batch + DATA_INVENTORY→OQ + sweep + push). Then Phase 0 → the six build commits, gating only Commits 1, 2, and 5 per the tiers above, with `verify_baselines.py` 40/40 on each code commit.
