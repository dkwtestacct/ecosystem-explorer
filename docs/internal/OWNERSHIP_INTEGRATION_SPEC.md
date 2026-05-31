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

**In:** full-county BCAD parcel pull + classification + rasterization to the SA grid; two signals — `is_public` (government-owned: city / county / state / federal / ISD / river_auth) and `is_vacant` (**exempt-keyed**: `State_cd` C* on any parcel, plus `ImprVal == 0` only on parcels NOT carrying an `EX-X*` total-exemption flag — see `docs/research/ownership/PHASE_0_INVESTIGATION.md` for the empirical evidence behind the rule); a UI toggle to restrict placement, composing with region selection; provenance + metadata for ownership-filtered scenarios; the coarseness caption when active.

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

Net: **one GATE at 0.5** (the first-of-a-kind data foundation). The rest are batch; the checks across them are `verify_baselines` 40/40 + eyeball + the user's pre-push review + a consolidated honesty pass at Commit 4.

- **0.5 Data acquisition** (GATE — the one first-of-a-kind foundation: canonical data + classifier design): `scripts/data/download_bexar_parcels.py` (paginated BCAD REST, `maps.bexar.org/.../Parcels/MapServer/0`, geoJSON, 711 pages at 1,000/page; fields Owner / State_cd / ImprVal / Exempts / Acres / geometry; reproject 2278 → 5070). Classify (owner-name regex → entity, corroborated by `Exempts`; vacancy = exempt-keyed rule per Phase 0). Rasterize public + vacant to the SA grid. Commit the **runtime raster** `data/sa/sa_public_vacant_30m.tif` (~396 KB) + DATA_INVENTORY entry; the 281 MB polygon GeoPackage is **gitignored** and archived outside the repo (Streamlit Cloud deploy-weight budget). License recorded as "not explicitly stated; Bexar County GIS / BCAD; attribution cited."
- **1 Ownership-mask infrastructure** (batch): build the masks on CityState; the caller composes `region ∩ ownership` through the existing `selected_region_mask` kwarg — `evaluate_scenario` is unchanged. `verify_baselines` 40/40 (no-ownership-filter default byte-identical) is the baseline-safety check; no seam change to gate on.
- **2 UI toggle + honesty caption** (batch): "Restrict to publicly-owned and/or vacant parcels"; the coarseness caption when active; the combined eligible denominator updates live (same pattern as region).
- **3 Provenance + metadata** (batch — reuses the Region Selection provenance pattern verbatim): label augmentation ("· public/vacant land"); record the ownership selection in the metadata block; **validation states unchanged** (exploratory placement, validated engine — same as any Explorer scenario). Not a fresh honesty surface — it's the same pattern already shipped and reviewed for region selection.
- **4 Testing + edges** (batch): zero-eligible-after-ownership-filter caption; schema bump if the metadata block grows; the targeted assertion; **consolidated honesty pass** over the badge / coarseness caveat / provenance label / metadata block — the closing sanity that the surfaces read honestly together. This is the honesty gate for the build.

`verify_baselines` 40/40 on every code commit; the no-ownership-filter default is byte-identical. What's New gets one blockbuster line only when the feature ships (a new selection dimension).

## UI

A toggle under the region selector: "Restrict to publicly-owned and/or vacant parcels." Headline = **vacant publicly-owned** (public AND vacant — the actionable set); consider separate public / vacant toggles for flexibility. Coarseness caption when active: "Ownership is derived from parcel records and approximate at this resolution — reliable for large parcels, less so for small lots."

## Honesty discipline

Exploratory layer; use the **locked badge vocabulary** — do not invent a new label set. The derived-ownership + coarseness caveats surface in the UI when active and in the export metadata as a known divergence. Never imply parcel-level certainty.
