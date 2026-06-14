# NatCap Alignment

**Audience:** Internal
**Status:** Current
**Use this for:** How aligned the engine is with canonical InVEST (validated) and how NatCap's published values are surfaced (displayed, not reproduced) — per metric, with validation status and caveats
**Do not use this for:** Per-city parameter values or per-city data parity (→ CITY_PARITY.md), per-city collaboration narrative (→ NATCAP_COLLABORATION.md), or per-decision rationale (→ DESIGN_NOTES.md)
**Source of truth for:** Model/metric alignment and validation status (organized by model/metric)

---

## 1. Alignment summary

This document covers the prototype's alignment with canonical InVEST organized **by model/metric**. The validation story sits at two levels, and they are not interchangeable:

- **Per-pixel parity against canonical InVEST** — the prototype reimplements UCM / UFR / UNA / UMH / Carbon in numpy and validates each against `natcap.invest.*.execute()` on matched inputs (`compare_*_invest.py` harnesses). HMI MAE = 0.0000 / r = 1.0000 (UCM, Brief 28b); **UNA 2SFCA supply_percapita: r = 1.000000 / ~5.5e-7 relative MAE over 70,868 MN px vs `natcap.invest` 3.19.0, matched-but-independent + guard** (`compare_una_supply_invest.py`); UMH per-pixel parity — **MN** MAE ≤ 1.1e-9 / r = 1.000000, **SA** MAE ≤ 2.3e-6 / r ≥ 0.99875 / |Δtotal| ≤ 0.15% (Brief B; see §6); **SA Carbon four-pool: per-pixel MAE ≈ 0 (3.3e-7 Mg C) / r = 1.0 vs `natcap.invest` 3.19.0 in matched units** (`compare_carbon_sa_fourpool_invest.py`, Relay 69); **UFR runoff-retention index: per-pixel MAE ≈ 0 (5e-8) / r = 1.0 vs `natcap.invest` UFRM 3.19.0** (`compare_ufr_invest.py`, Relay 71). Results in `comparisons/*_parity.csv`; see §6. This is the prototype's anchored reproduction claim. (The UFR *retention index* is the per-pixel-validated reading; the lumped Flood Index `100 − mean_CN` and Runoff Volume stay aligned-method.)
- **Citywide-absolute reproduction of NatCap's published values** — **NOT established.** NatCap's SA UCM `args.json` isn't in the drive pull, and `tot_c_cur.tif` doesn't aggregate to NatCap's published 107.32 M t CO2e by any standard interpretation. The published numbers came from runs whose inputs (UCM args + carbon aggregation script) weren't shared. See OPEN_QUESTIONS.md "Per-scenario compound LULC inputs."

The result: the prototype reproduces **canonical InVEST per-pixel**, uses canonical InVEST methodology, and **surfaces NatCap's own published scenario outcomes** as reference. It does *not* claim independent reproduction of NatCap's citywide aggregates.

**This file is organized by model/metric.** For per-city parameter values and per-city data parity, see CITY_PARITY.md. For the collaboration record (asks, decisions, gaps), see NATCAP_COLLABORATION.md. For per-decision rationale, see DESIGN_NOTES.md.

---

## 2. Validation badge taxonomy

The dashboard surfaces alignment status on each metric card via a four-state badge vocabulary. This section is the **authoritative spec** — app code is ground truth above it; REFERENCE §4 / ARCHITECTURE §6 / DESIGN_NOTES §8 cross-reference here and don't restate.

| Badge | Meaning | What it claims | What it does not claim |
|---|---|---|---|
| **Green "NatCap published value"** | Displayed directly from NatCap reference output | This is NatCap's published number | Not independently reproduced |
| **Blue "≈ NatCap method"** | Computed with NatCap-aligned project method/data | Methodologically aligned | Not necessarily matched to a published scenario |
| **Blue "≈ Aligned method"** | InVEST-style/canonical method, no project-specific anchor | Comparable methodology | Not NatCap project-specific |
| **Gray "Prototype"** | Exploratory proxy or assumption | Useful for exploration | Not a final quantitative result |

1. **Per-metric evidence varies within "≈ NatCap method."** Temperature cites measured per-pixel parity (HMI MAE ≈ 0, Brief 28b); SA carbon is four-pool methodology adoption (Brief 30) that **now also carries measured per-pixel parity** (MAE ≈ 0 / r = 1.0 vs `natcap.invest` 3.19.0, Relay 69 — see §6), so it can cite measured parity just as temperature does. The per-card tooltip surfaces this nuance.
2. **Badges are per-metric × per-context.** A `natcap_published` metric shows **"NatCap published value"** *only* in the fixed-scenario reference view; in baseline / Explorer / optimizer contexts the same metric shows **"≈ NatCap method"** (the prototype computed it). This is what prevents an Explorer-scenario number from reading as a NatCap-published one.

### Validated reference outputs (SA)

`data/sa/natcap_reference_outputs.csv` is the source of truth for *what NatCap publishes* for the SA project scenarios (baseline + FF/UA × 20ac/40ac/MAX), extracted from `nootenboom_results/citywide_results_UPDATED.xlsx` by `extract_natcap_reference_outputs.py` (re-runnable — the script is the provenance). Long format, one row per prototype-metric × scenario (49 rows), absolute values + explicit baseline rows. Three `validation_status` states:

- **`natcap_published`** (directly comparable, with tolerance): **temp_change_f** (NatCap `avg_temp_f`, °F; tol 5 % / 0.1 °F) and **carbon_tons_co2** (NatCap `c_sequestration` tons C × 44/12; tol 1 %). Compared as **deltas** (scenario − baseline), since the prototype reports these as deltas.
- **`aligned_method`** (canonical method, no clean citywide comparison, no tolerance check): **nature_access_pct** (NatCap `ntr_bal_avg` is a per-block-group balance aggregate ≈ 107 — a *different statistic*; see Track C per-block-group aggregation), **children_nature_access_pct** (child-reweighted UNA access share; inherits the UNA 2SFCA method, no separate parity claim), **cooling_energy_savings_usd** (NatCap citywide all-buildings spend vs prototype typed-OSM ~29 % coverage), **flood_reduction** (the lumped Flood Index `100 − mean_CN` — a scalar proxy, no NatCap published value, no per-pixel parity claim), **runoff_retention_idx** (canonical UFR per-pixel `1 − Q/P`, **validated at MAE ≈ 0 / r = 1.0 vs UFRM 3.19.0** per Relay 71 — aligned-method only because there's no published NatCap SA flood value), **preventable_mh_cases** (canonical UMH at MAE ≈ 0 per Brief B, but UMH wasn't in NatCap's SA project).
- **`prototype`** (no canonical analog): **food_mln_lbs**.

Read via `natcap_validation.py` (`load_reference_outputs` / `lookup_reference` / `compare_to_reference` — delta-aware for the published metrics). Surfaced in the dashboard via per-metric validation badges per the §2 taxonomy above.

---

## 3. Metric methodology fidelity

How close each app metric is to its canonical InVEST implementation. Per-model gap notes (UFR, UCM, UNA, UMH, Carbon, Crop) live in REFERENCE.md "Official InVEST alignment." Per-city parameter *values* live in CITY_PARITY.md.

| App metric | Current implementation | InVEST analogue | Parity | Confidence |
|---|---|---|---|---|
| Flood Index | CN-based area-weighted **lumped** index (`100 − mean_CN`) — a scalar proxy, not a per-pixel UFRM output | Urban Flood Risk Mitigation (retention index) | Implemented (lumped proxy) | High |
| Runoff Volume | SCS CN **lumped mean-CN** runoff × developed acreage at the per-city design storm — not a per-pixel UFRM output | Urban Flood Risk Mitigation (`Q_mm.tif`, `flood_vol`) | Implemented (lumped proxy) | High |
| Runoff Retention | Canonical per-pixel `rnf_rt_idx = mean(1 − Q/P)` (`cn_array_to_retention_index`). **Validated to per-pixel parity** against `natcap.invest` UFRM 3.19.0 `runoff_retention_index` at MAE ≈ 0 / r = 1.0 on value-identical CN (`compare_ufr_invest.py`, Relay 71). | Urban Flood Risk Mitigation (`runoff_retention_index`) | Implemented (per-pixel validated) | High |
| Flood Damage Avoided | `total_potential_damage × runoff_reduction_fraction` | Urban Flood Risk Mitigation (`serv_blt` indicator) | Approximate | Medium |
| Temperature Change | Canonical HMI = `max(CC_local, CC_park)`, `ΔHMI × UHI_MAX_C × 1.8` | Urban Cooling (HMI → T_air → anomaly) | Implemented | High |
| Cooling Energy Savings | `consumption_rate × ΔHMI × UHI_MAX_C × pixel_area × $/kWh`, applied per pixel | Urban Cooling (energy module: `consumption × ΔT_air × $/kWh` per building) | Approximate | Medium |
| Nature Access | Canonical InVEST UNA via 2SFCA — `urban_nature_supply_percapita ≥ urban_nature_demand`, share of modelable-extent population. Numpy implementation (`calculate_nature_access`), **validated to per-pixel parity** against `natcap.invest.urban_nature_access` 3.19.0 `urban_nature_supply_percapita` — r = 1.000000 / AOI-sum 0.00001% over 70,868 MN px, matched-but-independent + guard (`compare_una_supply_invest.py` → `comparisons/una_supply_parity_mn.csv`); the 0.054 m²/person per-pixel MAE is float-precision noise on the ~99,000 m²/person field (fit slope = 1.0000, intercept ≈ 0; ~98% interior), not a scale/offset. Per-city parameters per CITY_PARITY UNA tables; rationale in DESIGN_NOTES §2.2. | Urban Nature Access (2SFCA supply/demand/balance) | Implemented (per-pixel validated) | High |
| Children's Nature Access | UNA 2SFCA access share reweighted by Census 2020 under-18 population; supply/demand stays on total pop, inherits UNA's canonical-method validation. | Urban Nature Access (child-disaggregated share) | Implemented | Medium |
| Preventable MH Cases | `(1 − RR) × BIR × pop`; NE = edge-corrected **buffer-mean** of synthetic NDVI over a flat binary disk via `_convolve_edge_corrected` (the canonical InVEST UMH kernel). **Validated against `natcap.invest.urban_mental_health.execute()` v3.19.0** — **MN** MAE ≤ 1.1e-9 / r = 1.000000; **SA** MAE ≤ 2.3e-6 / r ≥ 0.99875 / |Δtotal| ≤ 0.15% (SA residual: canonical's radius padding + edge-crop alignment + pygeoprocessing FFT noise on the 1713 × 1984 grid). Validation is on the synthetic NDVI proxy, not satellite NDVI. | Urban Mental Health v3.19.0 (same PC formula + NE kernel) | Implemented | High |
| Avoided MH Costs | Preventable cases × per-case cost-of-illness (inherits the Preventable MH Cases validation above — linear in cases) | Urban Mental Health (cost module) | Implemented | High |
| Carbon Sequestration / Storage Change | Per-city framework. **SA**: InVEST four-pool stock framework (above-ground + below-ground + soil + dead) × compound LULC delta × 44/12, per NatCap's Vibrant Land (Guerry et al. 2023) methodology — one-time stock change in t CO2. **MN**: single aggregate annual rate per cover class × converted area (proxy). Per-city methodology in CITY_PARITY Carbon rows; rationale in DESIGN_NOTES §6.4. | Carbon Storage and Sequestration (4-pool storage snapshot) | SA: Implemented; MN: Proxy | SA: Medium / MN: Prototype |
| Carbon Storage Value (SA) / Avoided Carbon Cost (MN) | Carbon × `EPA_SOCIAL_COST_CARBON = $190/t` (EPA 2023 final rule, 2 % discount). Methodology matches Vibrant Land's stock × SC-CO2 framing; SC-CO2 vintage differs (Vibrant Land cited IWG 2021 $53/t @ 3 %). | Carbon (has its own NPV valuation with discount rate) | Inspired-by | Medium |
| Food Production | Food-forest yield benchmark × area (per-city benchmarks in CITY_PARITY Food Forest rows) | Crop Production (climate-binned staple-crop yields) | N/A | Prototype |
| NDVI | Synthetic per-NLCD-class proxy lookup | (not a standalone InVEST model) | N/A | Prototype |

**Parity taxonomy:**

- **Implemented** — App calculation follows InVEST methodology directly.
- **Approximate** — Math is in the spirit of InVEST but takes documented shortcuts (named in the per-model gap notes in REFERENCE.md).
- **Proxy** — Output is in the same family as InVEST but the method is fundamentally different.
- **Inspired-by** — Uses InVEST framing but the underlying calculation isn't from any InVEST model.
- **N/A** — No meaningful InVEST counterpart.

Parity is *methodological fidelity*. Confidence tier (per the in-app badges) is *output quality*. A metric can be Approximate-parity and High-confidence (math is solid even if it shortcuts InVEST), or Proxy-parity and Medium-confidence (output is trustworthy for planning even though the method diverges).

---

## 4. Computed vs displayed

The prototype distinguishes between two kinds of values that appear on screen:

- **Computed** — the prototype's own numpy pipeline produces the number for the active scenario. This covers everything in §3 above; the §2 badge taxonomy classifies the resulting confidence.
- **Displayed** — the value is surfaced directly from a NatCap reference output (no prototype computation). Only `natcap_published`-class metrics shown in the fixed-scenario reference view fall here; everywhere else (baseline / Explorer / optimizer) the same metric is *computed*.

This is what the per-context switch in §2 enforces: a `natcap_published` metric shows **"NatCap published value"** (Green) in the fixed-scenario view (displayed) but **"≈ NatCap method"** (Blue) anywhere else (computed).

**A3 status — comparison-READY, never executed.** The CSV stores NatCap's published values and `natcap_validation.compare_to_reference` implements the delta tolerance check, but **no end-to-end `evaluate_scenario → compare_to_reference` pipeline has ever run** — the only `natcap_published` metrics (`temp_change_f`, `carbon_tons_co2`) are exactly those gated by the unavailable compound scenario inputs (OPEN_QUESTIONS). The only callers of `compare_to_reference` are the four-line `__main__` smoke test with hardcoded values.

**SA UNA / biophysical extent.** The prototype's per-pixel `urban_nature_supply_percapita` is computed identically regardless of which polygons aggregate it. NatCap's ACS block-group polygons are a strict subset of the prototype's biophysical extent (Bexar County bbox); area IoU = 0.824, population overlap = 98.6 % (per-pixel detail + numbers in CITY_PARITY SA UNA section; investigation note at `../research/una/`). For NatCap project-scenario validation (Track C), aggregate the prototype's supply raster per block group rather than comparing the citywide headline.

---

## 5. Known methodological divergences

Where the prototype's canonical-method implementation takes documented shortcuts. None of these is hidden — every one surfaces on the user-facing card via tooltip, caption, or the §2 badge.

| Divergence | Why | Documented in |
|---|---|---|
| **UMH baseline prevalence** is uniform national CDC rates (0.21 / 0.19), not per-administrative-unit `risk_rate` vectors. | Per-tract MH-prevalence data not available for MN / SA. | MH card tooltip; DESIGN_NOTES §6.3. |
| **NDVI source** is a synthetic per-NLCD-class proxy lookup, not satellite-derived (e.g. AlphaEarth). | Satellite NDVI integration is a future workstream; feasibility researched in `../research/ALPHAEARTH_FEASIBILITY.md`. | NDVI card tooltip; flagged on MH card help. |
| **Cooling Energy Savings** is per-pixel aggregation, not per-building T_air sampling over the 600 m blending radius. | Methodology shortcut; affects only the dollar metric, not Temperature Change. | REFERENCE.md UCM section; CITY_PARITY UCM summary. |
| **Flood Damage Avoided** produces dollar values; InVEST UFR's `serv_blt` is officially an indicator only (currency · m³ units). | Documented divergence — the prototype scales to dollars for usability. | REFERENCE.md Flood Damage Avoided card; tooltip. |
| **MN Carbon** uses a single per-NLCD-class annual rate (proxy), not the four-pool InVEST framework. | No NatCap MN four-pool data exists in the shared Drive; per-city framing keeps each city on its own published methodology. | DESIGN_NOTES §6.4; CITY_PARITY MN Carbon row; pending NatCap data per NATCAP_COLLABORATION. |
| **Carbon Storage Value SC-CO2 vintage** uses EPA 2023 ($190/t, 2 % discount) — more current than Vibrant Land's IWG 2021 ($53/t, 3 %). Methodology matches; vintage intentionally differs. | EPA 2023 final rule is the more current US-government standard. | DESIGN_NOTES §6.4; NATCAP_COLLABORATION decisions log. |
| **Food Production** uses a single per-city yield benchmark; NatCap's framework is InVEST Crop Production with per-crop parameterization (`CoSA_Crop_production_ESModeling`). | Per-crop data not yet obtained from NatCap. | NATCAP_COLLABORATION open ask 4a. |

For deferred alternative *approaches* (PLUS / CLUE / LCM land-use simulators, ROOT optimization), see DESIGN_NOTES §11.

---

## 6. Export-to-InVEST validation boundary

The InVEST export bundle (DESIGN_NOTES §9; ARCHITECTURE §7) is the bridge for users who want canonical InVEST results on the prototype's inputs. The bundle's `metadata.json → validation` block records each model's state using a **two-state taxonomy distinct from the §2 per-card badge's four states**:

| Bundle state | Meaning | Emitted for |
|---|---|---|
| **`validated`** | Per-pixel parity measured against canonical `natcap.invest.*.execute()` | UCM, UNA, UMH, Carbon, UFR† |
| **`methodology_aligned`** | Canonical method, no per-pixel parity check | (none currently) |

**† UFR's `validated` is scoped to its per-pixel output.** The UFRM-equivalent per-pixel quantity — the **runoff-retention index** (`1 − Q/P`) — is validated to per-pixel parity (Relay 71, below). The dashboard's headline **Flood Index** (`100 − mean_CN`) and **Runoff Volume** are lumped mean-CN proxies, **not** per-pixel UFRM outputs, and remain aligned-method without a parity claim.

Each entry includes the reference InVEST version (3.19.0) and a per-model notes string sourced from this file. The two-state taxonomy answers a yes/no question per model (does the prototype's output match canonical?), while the §2 four-state badge captures per-metric methodology nuance (e.g. temperature, carbon, and the UFR retention index all cite measured per-pixel parity, while the lumped Flood Index / Runoff Volume readings do not). They are distinct surfaces for distinct audiences.

**Carbon — validated to per-pixel parity (Relay 69).** SA's four-pool carbon storage is **validated** against canonical `natcap.invest.carbon` 3.19.0 in matched units by a committed harness, `validation/compare_carbon_sa_fourpool_invest.py` (+ result artifact `comparisons/carbon_sa_fourpool_parity.csv`): **per-pixel MAE = 3.3e-7 Mg C, Pearson r = 1.0, AOI-sum diff −1e-6 %** over 3.36 M valid pixels, with a non-vacuous perturbation guard (perturbing one pool density moves the InVEST result). The harness feeds canonical `carbon.execute()` the **identical** SA compound LULC + **identical** four-pool C-density table the evaluator reads, so the residual is arithmetic, not input drift. Units: InVEST `c_storage_bas` is metric tons/hectare (per-pixel **density**); the evaluator reports per-pixel **total** Mg C (× pixel-area-ha 0.09) — both normalized before comparison (the raw, un-normalized density-vs-total gap is ~121 Mg C, purely the units artifact). This **supersedes the withdrawn 3.16.2 record** (commit 463912b, corrected Relay 68): that figure is now reproduced and grounded by a committed harness on the version-consistent 3.19.0 — no recorded carbon number without a committed reproducer. Distinct from `validation/compare_carbon_invest.py`, the MN single-pool annual-sequestration-vs-stock AOI-sum check (not a parity test).

**UFR Runoff Retention — validated to per-pixel parity (Relay 71).** The prototype's runoff-retention index (`rnf_rt_idx = mean(1 − Q/P)`, `app.cn_array_to_retention_index`) is **validated** against canonical `natcap.invest` UFRM 3.19.0's `runoff_retention_index` by a committed harness, `validation/compare_ufr_invest.py` (+ result artifact `comparisons/ufr_sa_retention_parity.csv`): **per-pixel MAE ≈ 5e-8, Pearson r = 1.0, AOI-sum diff −2e-6 %** over 3.36 M pixels, with a non-vacuous perturbation guard. Drift-free by construction — UFRM is fed a synthetic LULC + CN table that reproduce the evaluator's **actual** per-pixel CN (CN identity verified to the bit), so the only thing under test is UFRM's S_max→Q→retention chain vs the evaluator's formula. The crux (Relay 70) is settled: UFRM 3.19.0 uses **λ = 0.2**, `S_max = 25400/CN − 254` mm, `Q = (P − λ·S_max)²/(P + (1−λ)·S_max)` — algebraically identical to the evaluator's `Ia = 0.2·S`, `S = 1000/CN − 10` in (the mm-vs-inch scale cancels in the dimensionless `Q/P`), so there is **no** Ia-convention divergence. Scope: this validates the **per-pixel retention index only**; the lumped Flood Index (`100 − mean_CN`) and Runoff Volume are derived proxies, not per-pixel UFRM outputs, and stay aligned-method.

**Export ≠ already-validated.** The bundle records the prototype's own measured parity against canonical InVEST per model. Running canonical `execute()` on the bundle produces fresh canonical outputs which the user can then compare against the prototype's reported card values. Validation travels with the bundle; the bundle isn't itself a validation result.

**Phase 3 verification (Brief D1, 2026-05-29):** all five InVEST 3.19.0 urban models execute cleanly on the SA baseline bundle (UCM ✓, UNA ✓, UFR ✓, Carbon ✓, UMH-depression ✓, UMH-anxiety ✓).

---

## 7. Link to city parity

This file covers alignment by model/metric. The per-city parameter values, per-city data inputs (paths, MD5, source provenance), and the per-city × per-model at-a-glance status matrix live in **CITY_PARITY.md**. NatCap-collaboration narrative (asks, decisions, shared data) lives in **NATCAP_COLLABORATION.md**. Per-decision rationale lives in **DESIGN_NOTES.md**.

When a commit changes anything tracked here, update the §3 Metric methodology fidelity row or the §5 Known methodological divergences row as part of the commit.
