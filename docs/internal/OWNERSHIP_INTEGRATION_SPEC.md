# Ownership Integration — Build Spec

**Audience:** Internal
**Status:** Ready to build (after Region Selection Phase 1 closes + pushes — done at `7e8b5da`)
**Depends on:** Region Selection Phase 1 (the mask seam); Ownership Phase 0 (`docs/research/ownership/`)
**Builds:** ownership-constrained placement as a second masked layer
**Source of truth for:** What's in scope, what's out, the build sequence + gate tiers, and the honesty contract

---

## What this is

Let users restrict conversions to publicly-owned and/or vacant parcels, composing with region selection. Architecturally identical to region selection — `candidate_pixels ∩ ownership_mask` — so it reuses the Phase 1 seam rather than adding new machinery to `evaluate_scenario`. This is a notably smaller build than region selection was.

## Honest framing (the spine)

Ownership type is **derived, not authoritative.** BCAD parcels carry an `Owner` name string, not an ownership-type field, so public/private and the entity breakdown are inferred from owner-name patterns + the `Exempts` flag (~88% public-acreage coverage in Phase 0). And at 30m resolution ownership is approximate: 69% of parcels are sub-pixel, so the mask is reliable for large parcels (parks, big public/vacant tracts — the actionable ones) and pixelated for residential subdivisions.

Therefore ownership-constrained placement ships as an **exploratory** layer — validated engine, exploratory placement — with documented caveats and an honesty caption when active. It must never imply parcel-level certainty.

**"Public" means government-owned** (city / county / state / federal / ISD / river authority), NOT all tax-exempt. Churches, charities, and universities are tax-exempt but not land a city can act on.

## Scope

**In:** full-county BCAD parcel pull + classification + rasterization to the SA grid; two signals — `is_public` (government-owned) and `is_vacant` (`State_cd` C* OR `ImprVal == 0`); a UI toggle to restrict placement, composing with region selection; provenance + metadata for ownership-filtered scenarios; the coarseness caption when active.

**Out:** hand-tuning the ~12% public-acreage residual (follow-up; 88% is the prototype bar); the finer entity breakdown beyond public/private + vacant (start coarse); parcel-level editing; easements / ROW ownership (not in the data); MN ownership (BCAD is Bexar-specific — SA only).

## Architecture — the key simplification

The ownership masks are boolean rasters on the active grid, built exactly like the region rasters and stored on CityState alongside `region_rasters`.

`evaluate_scenario` already takes a single `selected_region_mask`. **Ownership composes in the caller:** the UI builds `region_mask` (from districts) and `ownership_mask` (from the toggle), ANDs them, and passes the combined mask as `selected_region_mask`. So:

```
combined_mask = region_mask ∩ ownership_mask   # built by the caller
candidate_pixels = CONVERTIBLE_PIXELS ∩ combined_mask
```

`evaluate_scenario` is **unchanged** — it already takes one mask. All the work is in mask construction (loader + caller), the metadata, the provenance label, and the caption. `eligible_pixels_in_region` naturally reflects the full constraint (region ∩ ownership ∩ convertible).

## Build commits (gate tiers)

- **0.5 Data acquisition** (GATE — first-of-kind data + the classifier design): `scripts/data/download_bexar_parcels.py` (paginated BCAD REST, `maps.bexar.org/.../Parcels/MapServer/0`, geoJSON, ~600 pages at 1,000/page or bulk; fields Owner / State_cd / ImprVal / Exempts / Acres / geometry; reproject 2278 → 5070). Classify (owner-name → entity, corroborated by `Exempts`; vacancy = `State_cd` C* OR `ImprVal == 0`). Rasterize public + vacant to the SA grid. Commit `data/sa/sa_ownership.gpkg` (+ cached rasters) + a DATA_INVENTORY entry, license recorded as "not explicitly stated; Bexar County GIS / BCAD; attribution cited." Report classifier coverage + public/vacant pixel counts.
- **1 Ownership-mask infrastructure** (GATE — touches the seam): build the masks on CityState; the caller composes `region ∩ ownership`. `verify_baselines` 40/40 (no-ownership-filter default byte-identical).
- **2 UI toggle + honesty caption** (batch): "Restrict to publicly-owned and/or vacant parcels"; the coarseness caption when active; the combined eligible denominator updates live (same pattern as region).
- **3 Provenance + metadata** (GATE — honesty surface): label augmentation ("· public/vacant land"); record the ownership selection in the metadata block; **validation states unchanged** (exploratory placement, validated engine — same as any Explorer scenario).
- **4 Testing + edges** (batch): zero-eligible-after-ownership-filter caption; schema bump if the metadata block grows; the targeted assertion.

`verify_baselines` 40/40 on every code commit; the no-ownership-filter default is byte-identical. What's New gets one blockbuster line only when the feature ships (a new selection dimension).

## UI

A toggle under the region selector: "Restrict to publicly-owned and/or vacant parcels." Headline = **vacant publicly-owned** (public AND vacant — the actionable set); consider separate public / vacant toggles for flexibility. Coarseness caption when active: "Ownership is derived from parcel records and approximate at this resolution — reliable for large parcels, less so for small lots."

## Honesty discipline

Exploratory layer; use the **locked badge vocabulary** — do not invent a new label set. The derived-ownership + coarseness caveats surface in the UI when active and in the export metadata as a known divergence. Never imply parcel-level certainty.
