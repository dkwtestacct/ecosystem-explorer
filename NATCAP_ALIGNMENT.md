# NatCap Alignment

This document tracks how the Ecosystem Explorer prototype aligns with
NatCap (Stanford / Natural Capital Project) recommendations,
methodologies, and research directions across 5 dimensions. Useful for
collaboration conversations and for honest framing of the prototype's
positioning.

When a commit changes anything tracked here, update the relevant table.
Same discipline as `WHATS_NEW` and `verify_baselines.py`.

## 1. Metric Methodology Fidelity

The parity table — how close each app metric is to its canonical InVEST
implementation. Moved here from `REFERENCE.md`; the per-model gap notes
(UFR, UCM, UNA, UMH, Carbon, Crop) remain in `REFERENCE.md`'s "Official
InVEST alignment" section.

| App metric | Current implementation | InVEST analogue | Parity | Confidence |
|---|---|---|---|---|
| Flood Risk Reduction | CN-based area-weighted index (`100 − mean_CN`) | Urban Flood Risk Mitigation (retention index) | Implemented | High |
| Runoff Volume | SCS CN per-pixel runoff × developed acreage, per-city design storm (MN: 100 mm / 3.94"; SA: 157 mm / 6.18" per NatCap projects, Brief 23) | Urban Flood Risk Mitigation (Q_mm, flood_vol) | Implemented | High |
| Flood Damage Avoided | `total_potential_damage × runoff_reduction_fraction` | Urban Flood Risk Mitigation (serv_blt indicator) | Approximate | Medium |
| Temperature Change | Canonical HMI = `max(CC_local, CC_park)`, `ΔHMI × UHI_MAX_C × 1.8` | Urban Cooling (HMI → T_air → anomaly) | Implemented | High |
| Cooling Energy Savings | `consumption_rate × ΔHMI × UHI_MAX_C × pixel_area × $/kWh`, applied per pixel | Urban Cooling (energy module: `consumption × ΔT_air × $/kWh` per building) | Approximate | Medium |
| Nature Access | Canonical InVEST UNA via 2SFCA — `urban_nature_supply_percapita ≥ urban_nature_demand`, share of modelable-extent population. Numpy implementation, validated against `natcap.invest.urban_nature_access.execute()` at MAE ≈ 0. **Per-city parameters** (Brief 22): MN uses 250 m²/capita demand, 1000 m radius, exponential decay (MN-project canonical). SA uses 16.7, 800, dichotomy (SA-project canonical). See `DESIGN_NOTES.md`. | Urban Nature Access (2SFCA supply/demand/balance) | Implemented | Medium |
| Nature Quality Score | **Removed from the dashboard 2026-05-21.** Earlier population-weighted mean of the 0-1 proxy access score; the proxy itself was retired when canonical 2SFCA was implemented. Quality Score had no canonical InVEST analog. | Urban Nature Access (SUP_DEMadm_cap) | N/A | — |
| Preventable MH Cases | `(1 − RR) × BIR × pop`; NE via Gaussian-smoothed synthetic NDVI | Urban Mental Health v3.19.0 (same formula) | Implemented | Medium |
| Avoided MH Costs | Preventable cases × per-case cost-of-illness | Urban Mental Health (cost module) | Implemented | Medium |
| Carbon Sequestration | Single aggregate rate per cover class × converted area | Carbon Storage and Sequestration (4-pool storage snapshot) | Proxy | Prototype |
| Avoided Carbon Cost | Sequestration × EPA Social Cost of Carbon ($190/ton) | Carbon (has its own NPV valuation with discount rate) | Inspired-by | Medium |
| Food Production | Food-forest yield benchmark × area (11,500 lbs/acre MN, 8,500 SA per NatCap SA Urban Agriculture report) | Crop Production (climate-binned staple-crop yields) | N/A | Prototype |
| NDVI | Synthetic per-NLCD-class proxy lookup | (not a standalone InVEST model) | N/A | Prototype |

**Parity taxonomy:**
- **Implemented** — App calculation follows InVEST methodology directly.
- **Approximate** — Math is in the spirit of InVEST but takes documented shortcuts (named in the per-model gap notes in `REFERENCE.md`).
- **Proxy** — Output is in the same family as InVEST but the method is fundamentally different.
- **Inspired-by** — Uses InVEST framing but the underlying calculation isn't from any InVEST model.
- **N/A** — No meaningful InVEST counterpart.

Parity status is about *methodological fidelity*. Confidence tier (per the in-app badges) is about *output quality*. A metric can be Approximate-parity and High-confidence (math is solid even if it shortcuts InVEST), or Proxy-parity and Medium-confidence (output is trustworthy for planning even though the method diverges).

## 2. Data Source Alignment

Tracks alignment between the prototype's data inputs and NatCap-curated /
recommended equivalents.

| Data type | Current source | NatCap recommendation | Status |
|---|---|---|---|
| LULC (Minneapolis) | `data/cooling/land_use_2021.tif` — byte-identical to InVEST UNA sample LULC (`LULC_NLCD_2021.tif`) | InVEST UNA sample data | ✅ Aligned |
| LULC (San Antonio) | NLCD 2021, custom-clipped to Bexar County | Not provided by NatCap (SA is not in InVEST sample data) | ⚠️ Improvised |
| Population (Minneapolis) | US Census 2020 block-level (P1_001N, Hennepin County) | Standard NLCD-grid rasterization | ✅ Aligned |
| Population (San Antonio) | US Census 2020 block-level (P1_001N, Bexar County) | Standard NLCD-grid rasterization | ✅ Aligned |
| UNA biophysical table | InVEST sample: `LULC_attribute_table_UNA.csv` | NatCap-published canonical table | ✅ Aligned |
| UCM biophysical table | MN: InVEST UCM sample values; SA: tuned for Köppen BSh climate (classes 41/42/52/81) | NatCap-published canonical table | ✅ Aligned (MN) / ⚠️ Tuned (SA) |
| Building footprints (Minneapolis) | Split-config — Geofabrik OSM footprints (~113k, city-wide) feed the placement mask (`mask_buildings_file`); the InVEST UFR sample shapefile (`buildings_file`) drives the typed $-metric raster | Comprehensive OSM building footprints | ✅ Aligned |
| Road network (Minneapolis) | Geofabrik OSM (Minnesota extract, Option B class filter), rasterized into the non-convertible mask | Geofabrik OSM extracts, recommended | ✅ Aligned |
| Buildings + roads (San Antonio) | Geofabrik OSM (Texas extract) — `buildings_sa.gpkg` (345,900 polygons), `roads_sa.geojson` (55,553 segments) | Geofabrik OSM extracts, recommended | ✅ Aligned |
| NatCap MN production config (UNA) | Adopted 2026-05-24 (Brief 22): demand=250 m²/capita, radius=1000 m, decay=exponential per `data/invest/mn_sample_data_natcap_2026/UrbanNatureAccess_sample_data_MN/invest_urban_nature_access_args_MN.json` | NatCap's MN-specific UNA configuration | ✅ Aligned |
| NatCap SA production config (UNA, UCM, UFR) | Pending access to NatCap SA folder | NatCap's SA Urban Agriculture project data | ⏸️ Pending data access |

**Split-config buildings (Minneapolis).** Placement-constraint inputs and
model inputs serve different purposes, so they are sourced separately. The
non-convertible *placement mask* unions comprehensive Geofabrik OSM building
footprints (`mask_buildings_file`, ~113k city-wide); the *typed $-metric
raster* — Cooling Energy Savings, Flood Damage Avoided — stays on the InVEST
UFR sample shapefile (`buildings_file`), which carries the per-building InVEST
type codes those metrics require. NatCap's framing explicitly separates
placement-constraint data (where OSM is recommended) from model-input data
(where typed sample data is canonical), so the split is an alignment, not a
compromise.

## 3. Parameter Alignment

Tracks alignment between the prototype's parameter choices and
NatCap-validated values. See `DESIGN_NOTES.md` for the rationale per
parameter.

| Parameter | Current value | NatCap-validated value | Status |
|---|---|---|---|
| UNA `urban_nature_demand` | MN: 250 m²/capita (NatCap MN args.json); SA: 16.7 m²/capita (NatCap SA README) | Per-city NatCap project values | ✅ Aligned (per-city, Brief 22) |
| UNA `search_radius_mode` | `'uniform radius'` (both cities) | `'uniform radius'` (both NatCap projects) | ✅ Aligned |
| UNA `search_radius` | MN: 1000 m (NatCap MN); SA: 800 m (NatCap SA) | Per-city NatCap project values | ✅ Aligned (per-city, Brief 22) |
| UNA `decay_function` | MN: `'exponential'` (NatCap MN); SA: `'dichotomy'` (NatCap SA) | Per-city NatCap project values | ✅ Aligned (per-city, Brief 22) |
| UCM Heat Mitigation Index | Canonical `max(CC_local, CC_park)` | Same — InVEST canonical | ✅ Aligned (validated MAE ≈ 0) |
| UCM `UHI_MAX_C` | MN: 2.05 °C (InVEST UCM args JSON); SA: 11 °C (NatCap canonical, heat-wave-day scenario per `data/sa/natcap_2024/README_San_Antonio_InVEST_model_inputs.docx`) | MN: InVEST args JSON value; SA: NatCap-published README | ✅ Aligned (MN + SA, 2026-05-24 Brief 14) |
| UMH RR per 0.1 NDVI | 0.96 (depression) / 0.97 (anxiety), from Liu et al. 2023 | InVEST UMH effect sizes (same source family) | ✅ Aligned |
| UMH baseline prevalence | 0.21 / 0.19, uniform national (CDC 2023) | InVEST UMH takes per-administrative-unit BIR from a vector input | ⚠️ Improvised — uniform national rates, not per-admin |
| Carbon sequestration rate | Single per-class rate | InVEST Carbon 4-pool model (more detailed) | ⚠️ Simplified |
| NDVI source | Synthetic proxy from NLCD lookup | Satellite-derived (e.g., AlphaEarth) recommended | ⚠️ Improvised |

## 4. Spatial Fidelity

Tracks the prototype's spatial representation vs NatCap recommendations.

| Aspect | Current state | NatCap recommendation | Status |
|---|---|---|---|
| AOI extent (MN) | Downtown + near-neighborhoods, ~123 km², ~154k residents (InVEST UFR sample AOI) | NatCap's MN study extent | ✅ Aligned (uses NatCap-provided AOI) |
| AOI extent (SA) | Bexar County bbox, ~1,907k residents | NatCap's SA Urban Agriculture study extent | ⏸️ Pending verification |
| Placement constraints | Three-layer non-convertible mask: buildings + roads excluded via the rasterized mask; existing nature never a candidate (pool is developed NLCD 21–24 only). Random or strategy-weighted selection within the remaining pool. | 3-layer mask: buildings + roads + existing nature | ✅ Aligned |
| Building footprint coverage | Placement mask uses comprehensive OSM footprints city-wide for every city; the typed $-metrics use the InVEST UFR sample for MN (downtown core, where its per-building type codes are valid) | Comprehensive OSM building footprints | ✅ Aligned |
| Road network coverage | OSM road network rasterized into the non-convertible mask, all cities | OSM road network | ✅ Aligned |
| LULC resolution | 30 m / pixel (NLCD standard) | 30 m / pixel | ✅ Aligned |

## 5. Research Direction Status

Tracks the directions NatCap has identified for future work and the
prototype's status on each.

| Direction | NatCap mention | Current status | Notes |
|---|---|---|---|
| OSM buildings + roads as placement constraints | "Simpler approaches" in NatCap document | ✅ Implemented | Roads + comprehensive OSM building footprints unioned into the non-convertible mask for every city (MN via the split-config `mask_buildings_file`). See `DESIGN_NOTES.md` placement strategy section. |
| Wallpaper approach | "Simpler approaches" in NatCap document | ⏸️ Interpretation unclear; to discuss with NatCap | See `DESIGN_NOTES.md` |
| PLUS land-use simulation | "Existing models" in NatCap document | 🔵 Considered, deferred | C++/Qt app, integration heavy; see `DESIGN_NOTES.md` |
| CLUE land-use simulation | "Existing models" in NatCap document | 🔵 Considered, deferred | Java-based; same constraints as PLUS |
| LCM (Land Change Modeler) | "Existing models" in NatCap document | 🔵 Not pursued — proprietary | Can't ship in an open-source prototype |
| AlphaEarth satellite embeddings (NDVI replacement) | Identified as future direction | 🔵 Feasibility researched, not implemented | See `ALPHAEARTH_FEASIBILITY.md` |
| Land ownership layer | "Future consideration" in NatCap document | 🔵 Not pursued | Out of scope for current prototype |
| San Antonio as full pilot | Active NatCap research direction | 🔄 In progress | Pending SA data folder access |
| Carbon Storage and Sequestration model (deeper) | Listed as additional model for consideration | 🔵 Not pursued | Current single-rate approach simpler |
| Urban Mental Health model (already integrated) | Listed as additional model for consideration | ✅ Implemented | InVEST UMH v3.19.0 |
| ROOT (Restoration Opportunities Optimization Tool) | Mentioned in NatCap document | 🔵 Not pursued | Different optimization framework |

## 6. Vocabulary and Reporting Alignment

Tracks how the prototype's user-facing vocabulary (metric card names, tooltips, axis labels, prose) aligns with InVEST canonical terminology. Surfaced by the 2026-05-23 vocabulary audit (`NATCAP_VOCABULARY_AUDIT.md`).

| Surface | Current wording | InVEST canonical | Status |
|---|---|---|---|
| Temperature Change card underlying quantity | Heat Mitigation Index (HMI) | `hm.tif`, `mean(HMI)` | ✅ Aligned (renamed from "Cooling Capacity / CC" 2026-05-23) |
| Tradeoff plot Y axis | Heat Mitigation Index | `hm` | ✅ Aligned (renamed 2026-05-23) |
| Temperature assumption tab kernel description | exponential decay at d_cool, eq. 118 | exponential decay | ✅ Aligned (corrected from "Gaussian" 2026-05-23) |
| Flood Risk Reduction card | App's `100 − mean_CN` index, with explicit pointer to InVEST UFR `rnf_rt_idx = mean(1 − Q/P)` | `rnf_rt_idx` | ⚠️ Documented divergence — the app's index is monotone but not identical to UFR's canonical retention index. |
| Flood Damage Avoided card | App's dollar-scaled formula, with explicit pointer to InVEST UFR `serv_blt` indicator caveats | `serv_blt` (indicator only, currency·m³ units) | ⚠️ Documented divergence — the app produces dollars; InVEST itself treats `serv_blt` as an indicator only. |
| Nature Access card | `pct_pop_supply_ge_demand`, canonical UNA 2SFCA | `Pund_adm` / `Povr_adm` framing | ✅ Aligned (uses canonical UNA quantity) |
| Preventable MH Cases card | InVEST UMH formula | `preventable_cases.tif` | ✅ Aligned |
| Avoided MH Costs card | InVEST UMH formula with cross-reference to canonical "preventable_cost" naming | `preventable_cost.tif` | ✅ Aligned (cross-reference added 2026-05-23) |
| Cost-Effectiveness section | App-level synthesis, no InVEST analog, with pointer to ROOT | (no InVEST analog) | ⚠️ App-specific — explicit. |
| Balanced placement strategy | App-specific heuristic, with pointer to ROOT | (no InVEST analog) | ⚠️ App-specific — explicit. |
| Smart Scenario Search / surrogate optimizer | App-specific, with pointer to ROOT for true LP optimization | ROOT (LP, Pareto, agreement maps) | ⚠️ App-specific — explicit. |
| `undersupply-focused` placement strategy (renamed from `equity-focused`) | `max(0, urban_nature_demand − urban_nature_supply_percapita)` per pixel — canonical UNA per-capita supply deficit, no population multiplier | InVEST UNA's `urban_nature_balance_percapita` framing | ✅ Aligned 2026-05-23 (Brief 9). Saved scenarios with legacy `equity-focused` key are routed via shim. |
| `flood-focused` placement strategy | Per-pixel runoff `Q_{p,i}` from SCS-CN equation at the per-city design storm — matches InVEST UFR `Q_mm.tif`. Brief 23 made the storm depth per-city. | InVEST UFR's per-pixel `Q_{p,i}` runoff (eq. 127) | ✅ Aligned 2026-05-23 (Brief 9, refreshed Brief 23). |
| `cooling-focused` placement strategy | `(1 − baseline_HMI) × (1 / (1 + distance_to_buildings_px))` — canonical HMI + distance transform on `BUILDINGS_RASTER` | Canonical HMI + true building-proximity raster | ✅ Aligned 2026-05-23 (Brief 9). Previously used bare CC sub-component + NLCD-intensity three-value proxy. |

## Status legend

- ✅ **Aligned** — current state matches NatCap recommendation
- ⚠️ **Improvised** — current state differs from or substitutes for NatCap recommendation
- ⏸️ **Pending** — alignment pending data access, decisions, or external input
- 🔄 **In progress** — actively working toward alignment
- 🔵 **Considered, deferred** — option evaluated, not currently pursued (with rationale)

## How to update this document

When a commit changes any of:
- A metric's implementation status (Table 1)
- A data source (Table 2)
- A parameter value (Table 3)
- An AOI, placement constraint, or coverage area (Table 4)
- A research direction's status (Table 5)
- A user-facing vocabulary surface (Table 6)

...update the relevant table row(s) as part of the commit. Same
discipline as `WHATS_NEW` updates.
