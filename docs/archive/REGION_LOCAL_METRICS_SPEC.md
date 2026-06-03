# Region-Local Metrics — Build Spec

**Audience:** Internal
**Status:** Ready to build after Ownership Integration closes. Boundary treatment LOCKED: **(b)** clip + spillover caveat. Zero gates (see build sequence).
**Depends on:** Region Selection Phase 1 (the region mask + `results['region_selection']`). Independent of Ownership.
**Builds:** region-clipped metric reporting shown alongside the existing citywide metrics, for region scenarios.
**Source of truth for:** what region-local means per model, the build sequence, and the honesty contract.

---

## Problem

Phase 1 constrains *placement* to a region but reports *citywide* aggregates (the honest "metrics show citywide impact" caption). Planners also want the change *within* the region — "if I green District 5, what happens in District 5." This adds that as a second, clearly-labeled reading, without removing the citywide one.

## Decision required (yours)

Spatial-reach models bleed across the region boundary: UNA ~800 m, UCM ~600 m, UMH ~300 m; carbon and flood are local and clip cleanly. For the reach models, a region-clipped metric necessarily excludes effects that a region's conversions push *just outside* its edge. Three options:

- **(a) Clip to region pixels.** Simplest; silently excludes spillover-out.
- **(b) Clip + spillover caveat** ✅ **LOCKED**. Clip the delta to the region, show it next to citywide, and caption that reach models also produce effects beyond the boundary not counted in the local figure.
- **(c) Buffer the region** by each model's reach before aggregating. More "complete" per model, but the measured area no longer matches the region the user drew — confusing.

**(b) is locked** — the honest middle, and the only option where the measured area equals the drawn region. Build note: the spillover caveat must name *every* reach model actually displayed (UNA ~800 m, UCM ~600 m, UMH ~300 m), not just the longest two.

## Design / the seam

The models already produce per-pixel scenario and baseline rasters citywide. Region-local = aggregate the **per-pixel delta (scenario − baseline) over the existing region mask** (the same `selected_region_mask` Phase 1 builds), using each model's native aggregation. No change to model computation — this is a second aggregation of results already computed.

Key subtlety: metrics clip differently. Most are per-pixel quantities clipped to region pixels; the population-based ones (nature access, mental health) clip to the *population* inside the region, not pixels — which naturally follows region residents even when the greenspace they use sits outside the boundary. The locked per-model treatment:

| Metric | Region-local treatment |
| --- | --- |
| Carbon | Clip to region pixels. Clean — no caveat. |
| Food production | Clip to converted pixels in region. Clean. |
| Cost | Clip to converted pixels in region. Clean. |
| Flood / runoff | Region-mean-CN-derived volume scaled to region developed area (a closed-form SCS-CN application, NOT a per-pixel sum) + **routing caveat**: not routed hydrology, so it's not flood protection *delivered to* the region. The region value legitimately differs from citywide because both the regional mean CN and the regional developed area differ from their citywide counterparts — that's the rate/ratio shape, not an aggregation bug. |
| Cooling (UCM, ~600 m) | Clip to region pixels + **spillover caveat**. |
| Nature access (UNA, ~800 m) | Clip to **population** inside region + cross-boundary caveat (supply/access can cross the edge). |
| Mental health (UMH, ~300 m) | Clip to **population** inside region + exposure-kernel caveat. |

Every treatment reconciles at full-AOI (clip-to-everything == citywide), so the decomposability is machine-guarded by the assertion below regardless of pixel- vs population-clipping.

## Build sequence + gate tiers (zero gates)

The one judgment that used to want a gate — *which models are region-decomposable* — converts to an **automated assertion** instead, so no human stop is needed. The invariant: for any model marked decomposable, region-local computed over the **entire AOI** (region = everything) must equal that model's citywide value, because clipping to "all pixels" is the citywide sum. A model wrongly marked decomposable fails this reconciliation and trips the assertion like a 40/40 failure. The only error the assertion *doesn't* catch — a model wrongly marked *non*-decomposable — is harmless (it just shows "citywide only" conservatively). So the dangerous direction is machine-guarded and the safe direction costs nothing.

**Commit 1 — aggregation (batch).** Implement the locked per-model treatment table above (pixel-clip vs population-clip per metric); compute region-clipped values into `results['region_local']`. Add the **full-AOI reconciliation assertion** to verify_baselines (every treated model: region-local-over-everything == citywide). 40/40 unaffected (non-region scenarios never populate this).

**Commit 2 — display (batch).** For region scenarios, show region-local next to citywide per model, with the locked caveats: the flood routing caveat, and one reach caveat naming all three reach models — "Cooling, nature access, and mental-health exposure effects can extend beyond the selected boundary. The region-local column summarizes people/pixels inside the selected area; spillover effects outside the boundary are reflected in the citywide column." Keep the existing "metrics show citywide impact" caption on the citywide column.

**Commit 3 — provenance + metadata (batch).** region-local values into `metadata.json`, labeled, with the boundary-treatment note (option (b) + the per-model decomposability flags). Reuses the existing provenance pattern.

**Commit 4 — testing + edges (batch).** zero-region, single-pixel region, all-citywide (no region) inert; schema bump if the block grows; assertion that a known region's decomposable delta clips correctly.

## Scope

**In:** region-clipped reporting for region scenarios, for decomposable models; honest handling of non-decomposable ones.
**Out:** attribution metrics (the effect of region conversions *wherever they land* — a different, harder question); buffering (unless you pick (c)); region-local for the optimizer (region scenarios already disable it); MN (works wherever a region mask exists, but validate on SA first).

## Honesty contract

Region-local is **exploratory** placement reporting. It says "the change experienced within the drawn region," not "the total impact of acting here." The spillover caveat is the honesty guard for reach models; the "citywide only" label is the guard for non-decomposable ones. **Never show a region-local number without its citywide companion** — the pairing is what keeps it honest.
