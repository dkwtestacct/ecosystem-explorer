# SA UNA / biophysical extent investigation (Brief A2)

**Audience:** Internal — research note
**Status:** Live reference (2026-05-29 finding, decision held)
**Use this for:** The measured-extent comparison between the prototype's SA UNA computation footprint and NatCap's ACS block-group polygons, and the don't-mask decision that follows
**Do not use this for:** Current-state SA UNA parity claim (→ `../../internal/CITY_PARITY.md` SA UNA section) or the design rationale (→ `../../internal/DESIGN_NOTES.md` §2.5)
**Source of truth for:** The Brief A2 measurement detail and decision rationale

---

## Question

Yingjie's roadmap (`docs/internal/NATCAP_COLLABORATION.md`) says NatCap's SA UNA uses `acs_block_group.gpkg` as the AOI; the prototype was thought to use a City-of-SA clipped extent. If they differ materially, per-scenario UNA comparison against NatCap's published outputs would be over different areas — a hidden methodology gap that would invalidate any matching.

## Measurement (LULC-valid mask vs the ACS block-group polygons rasterised onto the 30 m EPSG:5070 grid)

| Footprint | Pixels | Area (km²) | Population |
|---|---:|---:|---:|
| Prototype UNA extent (Bexar County bbox, LULC-valid) | 3,398,592 | **3,059** | **1,906,325** |
| NatCap ACS block groups (1,124 polygons, City of SA) | 2,799,438 | **2,519** | **1,878,866** |
| Block-groups ∩ prototype extent | 2,799,438 | 2,519 | 1,878,866 |

The block groups are a **strict subset** of the prototype extent (the intersection equals the block-group coverage — no block-group pixel falls outside the prototype's bbox).

| Overlap metric | Value |
|---|---|
| **Area IoU** | **0.824** |
| **Population overlap** | **98.6 %** |
| Exurban population (in bbox, outside block groups) | **27,457 people (1.4 %)** — sparse rural Bexar County land |

## Architectural insight — why this isn't a config swap

The prototype's UNA path is **raster-only**: `calculate_nature_access(scenario_lulc, pop_count_raster)` takes no AOI vector. The modelable extent is wherever the LULC and population rasters have valid data. The `acs_block_groups_3857.gpkg` polygon file in the SA data folder feeds **only** `compute_per_tract_summary` (the neighborhood-breakdown table in tab 2), not any biophysical model. So "swapping the AOI" would mean *adding* a block-group mask to the UNA computation — coupling the polygons into the biophysical path where they currently aren't — not repointing a config path.

## Decision: document, don't mask (Option b)

The area IoU (0.824) tripped the brief's 0.95 gate, but UNA's headline metric is **population-weighted** and the extents are 98.6 % population-aligned. The discrepancy is 1.4 % of population on sparse exurban land.

Per-pixel `urban_nature_supply_percapita` is computed identically regardless of which polygons aggregate it — so the real validation need (matching NatCap's per-block-group `ntr_bal_avg` in `nootenboom_results`) is met by **aggregating the prototype's supply raster per block group**, which is a Track C concern, not an A2 AOI change. Masking the UNA path would cost a code change + full SA baseline/dense-CSV regen + schema bump for a sub-1 % effect — cost exceeds value.

**No code, no baseline, no schema change in Brief A2.**

## Forward note

Track A3 (CSV population) and Track C (parity validation vs `nootenboom_results`) should both use **per-block-group aggregation** of the prototype's supply raster, not the citywide headline, for SA UNA comparison.

## Where the conclusion now lives

- **Current-state parity claim** — `../../internal/CITY_PARITY.md` SA section, "SA biophysical extent vs ACS block-group polygons" callout (carries the IoU + pop-overlap numbers + don't-mask decision in summary form)
- **Per-decision rationale** — `../../internal/DESIGN_NOTES.md` §2.5 SA AOI ACS block groups
- **Computed-vs-displayed implication** — `../../internal/NATCAP_ALIGNMENT.md` §4 (one-line reference pointing here)
- **Chronological brief narrative** — `../../archive/HISTORY.md` "Brief A2 (2026-05-29) — SA UNA AOI investigation"

This document is the durable Brief-A2 single home for the investigation detail.
